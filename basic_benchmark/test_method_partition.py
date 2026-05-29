import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from psycopg2 import sql

from basic_benchmark import efconfig
from basic_benchmark.common_function import prepare_query_dataset, run_test, get_index_type
from controller.method import (
    build_and_materialize_workload_aware_plan,
    create_indexes_for_materialized_partitions,
    drop_indexes_for_materialized_partitions,
    load_current_access_overlays,
    load_current_tenant_overlays,
    load_current_partitions,
)
from controller.method.common import DEFAULT_QUERY_DATASET_PATH
from services.config import get_db_connection


def _collect_index_state(partitions):
    state = {"missing": [], "by_type": {}, "missing_auxiliary": [], "legacy_layout": []}
    conn = get_db_connection()
    auxiliary_index_map = {}
    column_map = {}
    table_names = {partition.table_name for partition in partitions}
    for partition in partitions:
        for accelerator_pattern in (partition.metadata.get("accelerator_patterns", []) or []):
            table_name = accelerator_pattern.get("table_name")
            if table_name:
                table_names.add(str(table_name))
    overlays = load_current_tenant_overlays()
    access_overlays = load_current_access_overlays()
    for overlay in overlays:
        table_name = overlay.get("table_name")
        if table_name:
            table_names.add(str(table_name))
    for overlay in access_overlays:
        table_name = overlay.get("table_name")
        if table_name:
            table_names.add(str(table_name))
    try:
        with conn.cursor() as cur:
            for table_name in sorted(table_names):
                cur.execute(
                    """
                    SELECT indexname
                    FROM pg_indexes
                    WHERE schemaname = current_schema()
                      AND tablename = %s;
                    """,
                    [table_name],
                )
                auxiliary_index_map[table_name] = {str(row[0]) for row in cur.fetchall()}
                cur.execute(
                    """
                    SELECT column_name
                    FROM information_schema.columns
                    WHERE table_name = %s;
                    """,
                    [table_name],
                )
                column_map[table_name] = {str(row[0]) for row in cur.fetchall()}
    finally:
        conn.close()
    for partition in partitions:
        index_type = get_index_type(partition.table_name)
        if index_type is None:
            state["missing"].append(partition.table_name)
        else:
            state["by_type"].setdefault(index_type, []).append(partition.table_name)
        columns = column_map.get(partition.table_name, set())
        if "pattern_id" not in columns:
            state["legacy_layout"].append(partition.table_name)
            continue
        expected_indexes = {
            f"{partition.table_name}_pattern_idx",
            f"{partition.table_name}_pattern_document_idx",
        }
        actual_indexes = auxiliary_index_map.get(partition.table_name, set())
        if not expected_indexes.issubset(actual_indexes):
            state["missing_auxiliary"].append(partition.table_name)
        for accelerator_pattern in (partition.metadata.get("accelerator_patterns", []) or []):
            accelerator_table_name = str(accelerator_pattern.get("table_name", ""))
            if not accelerator_table_name:
                continue
            accelerator_index_type = get_index_type(accelerator_table_name)
            if accelerator_index_type is None:
                state["missing"].append(accelerator_table_name)
            elif accelerator_index_type != index_type:
                state["by_type"].setdefault(accelerator_index_type, []).append(accelerator_table_name)
            accelerator_columns = column_map.get(accelerator_table_name, set())
            if "pattern_id" not in accelerator_columns:
                state["legacy_layout"].append(accelerator_table_name)
                continue
            accelerator_expected_indexes = {
                f"{accelerator_table_name}_pattern_idx",
                f"{accelerator_table_name}_pattern_document_idx",
            }
            accelerator_actual_indexes = auxiliary_index_map.get(accelerator_table_name, set())
            if not accelerator_expected_indexes.issubset(accelerator_actual_indexes):
                state["missing_auxiliary"].append(accelerator_table_name)
    for overlay in overlays:
        overlay_table_name = str(overlay.get("table_name", ""))
        if not overlay_table_name:
            continue
        overlay_index_type = get_index_type(overlay_table_name)
        if overlay_index_type is None:
            state["missing"].append(overlay_table_name)
        else:
            state["by_type"].setdefault(overlay_index_type, []).append(overlay_table_name)
        overlay_columns = column_map.get(overlay_table_name, set())
        if "pattern_id" not in overlay_columns:
            state["legacy_layout"].append(overlay_table_name)
            continue
    for overlay in access_overlays:
        overlay_table_name = str(overlay.get("table_name", ""))
        if not overlay_table_name:
            continue
        overlay_index_type = get_index_type(overlay_table_name)
        if overlay_index_type is None:
            state["missing"].append(overlay_table_name)
        else:
            state["by_type"].setdefault(overlay_index_type, []).append(overlay_table_name)
        overlay_columns = column_map.get(overlay_table_name, set())
        if "pattern_id" not in overlay_columns:
            state["legacy_layout"].append(overlay_table_name)
            continue
    return state


def _ensure_index_state(partitions, index_type: str) -> None:
    state = _collect_index_state(partitions)
    wrong_type_tables = []
    for actual_type, table_names in state["by_type"].items():
        if actual_type != index_type:
            wrong_type_tables.extend(table_names)
    if state["missing"] or wrong_type_tables or state["missing_auxiliary"] or state["legacy_layout"]:
        drop_indexes_for_materialized_partitions()
        create_indexes_for_materialized_partitions(index_type=index_type)


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _ratio_tag(value) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _compose_generator_label(generator_type: str, result_tag, protection_overlay_space_ratio) -> str:
    label = str(generator_type or "")
    tag = result_tag
    if tag is None and protection_overlay_space_ratio is not None:
        tag = f"protect{_ratio_tag(protection_overlay_space_ratio)}"
    if not tag:
        return label
    safe_tag = "".join(
        char if char.isalnum() or char in {"-", "_"} else "_"
        for char in str(tag)
    )
    return f"{label}_{safe_tag}" if label else safe_tag


def test_method_partition_search(
    iterations=1,
    enable_index=True,
    index_type="hnsw",
    statistics_type="sql",
    generator_type="tree-based",
    record_recall=True,
    warm_up=True,
    query_num=1000,
    prepare_before_test=False,
    min_pattern_support=16,
    min_pattern_query_mass=0.0,
    safe_density_threshold=0.35,
    supplemental_edge_penalty=0.25,
    supplemental_edge_gain_threshold=0.0,
    target_partition_count=None,
    overlay_space_ratio=0.25,
    protection_overlay_space_ratio=None,
    route_limit=64,
    partition_fetch_multiplier=4,
    workload_limit=None,
    document_limit=None,
    result_tag=None,
    use_ground_truth_cache=False,
    query_dataset_path=DEFAULT_QUERY_DATASET_PATH,
):
    effective_query_num = max(1, int(query_num))
    effective_query_dataset_path = query_dataset_path or DEFAULT_QUERY_DATASET_PATH
    queries = prepare_query_dataset(
        regenerate=False,
        num_queries=effective_query_num,
        query_dataset_path=effective_query_dataset_path,
    )

    effective_workload_limit = effective_query_num
    if workload_limit is not None and int(workload_limit) != effective_query_num:
        print(
            f"[method_partition] workload_limit={int(workload_limit)} differs from query_num={effective_query_num}; "
            "using query_num so planner and benchmark read the same query subset.",
            flush=True,
        )
    effective_protection_overlay_space_ratio = (
        float(overlay_space_ratio)
        if protection_overlay_space_ratio is None
        else float(protection_overlay_space_ratio)
    )

    if prepare_before_test:
        build_and_materialize_workload_aware_plan(
            min_pattern_support=min_pattern_support,
            min_pattern_query_mass=min_pattern_query_mass,
            safe_density_threshold=safe_density_threshold,
            supplemental_edge_penalty=supplemental_edge_penalty,
            supplemental_edge_gain_threshold=supplemental_edge_gain_threshold,
            target_partition_count=target_partition_count,
            overlay_space_ratio=overlay_space_ratio,
            protection_overlay_space_ratio=protection_overlay_space_ratio,
            query_dataset_path=effective_query_dataset_path,
            workload_limit=effective_workload_limit,
            document_limit=document_limit,
            create_indexes=False,
            index_type=index_type,
        )

    partitions = load_current_partitions()
    if not partitions:
        raise RuntimeError(
            "No method partitions are materialized. Run with --prepare true or build the plan first."
        )

    if enable_index:
        _ensure_index_state(partitions, index_type)
    else:
        drop_indexes_for_materialized_partitions()

    efconfig.dynamic_partition_route_limit = int(route_limit)
    efconfig.dynamic_partition_partition_fetch_multiplier = int(partition_fetch_multiplier)
    effective_generator_type = _compose_generator_label(
        generator_type,
        result_tag,
        effective_protection_overlay_space_ratio,
    )
    run_test(
        queries,
        "method_partition",
        iterations=iterations,
        enable_index=enable_index,
        statistics_type=statistics_type,
        generator_type=effective_generator_type,
        index_type=index_type,
        record_recall=record_recall,
        warm_up=warm_up,
        use_ground_truth_cache=use_ground_truth_cache,
        queries_num=effective_query_num,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run method_partition benchmark with the ACL Prefix-DAG planner.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--enable-index", type=_str_to_bool, default=True)
    parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    parser.add_argument("--statistics-type", choices=["sql", "system"], default="sql")
    parser.add_argument("--generator-type", default="tree-based")
    parser.add_argument("--record-recall", type=_str_to_bool, default=True)
    parser.add_argument("--warm-up", type=_str_to_bool, default=True)
    parser.add_argument("--query-num", type=int, default=1000)
    parser.add_argument("--prepare", type=_str_to_bool, default=False)
    parser.add_argument("--min-pattern-support", type=int, default=16)
    parser.add_argument("--min-pattern-query-mass", type=float, default=0.0)
    parser.add_argument("--safe-density-threshold", type=float, default=0.35)
    parser.add_argument("--supplemental-edge-penalty", type=float, default=0.25)
    parser.add_argument("--supplemental-edge-gain-threshold", type=float, default=0.0)
    parser.add_argument("--target-partition-count", type=int, default=None)
    parser.add_argument("--overlay-space-ratio", type=float, default=0.25)
    parser.add_argument("--protection-overlay-space-ratio", type=float, default=None)
    parser.add_argument("--route-limit", type=int, default=16)
    parser.add_argument("--partition-fetch-multiplier", type=int, default=1)
    parser.add_argument("--workload-limit", type=int, default=None)
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--result-tag", default=None)
    parser.add_argument("--use-ground-truth-cache", type=_str_to_bool, default=False)
    parser.add_argument("--query-dataset-path", default=DEFAULT_QUERY_DATASET_PATH)
    args = parser.parse_args()

    test_method_partition_search(
        iterations=args.iterations,
        enable_index=args.enable_index,
        index_type=args.index_type,
        statistics_type=args.statistics_type,
        generator_type=args.generator_type,
        record_recall=args.record_recall,
        warm_up=args.warm_up,
        query_num=args.query_num,
        prepare_before_test=args.prepare,
        min_pattern_support=args.min_pattern_support,
        min_pattern_query_mass=args.min_pattern_query_mass,
        safe_density_threshold=args.safe_density_threshold,
        supplemental_edge_penalty=args.supplemental_edge_penalty,
        supplemental_edge_gain_threshold=args.supplemental_edge_gain_threshold,
        target_partition_count=args.target_partition_count,
        overlay_space_ratio=args.overlay_space_ratio,
        protection_overlay_space_ratio=args.protection_overlay_space_ratio,
        route_limit=args.route_limit,
        partition_fetch_multiplier=args.partition_fetch_multiplier,
        workload_limit=args.workload_limit,
        document_limit=args.document_limit,
        result_tag=args.result_tag,
        use_ground_truth_cache=args.use_ground_truth_cache,
        query_dataset_path=args.query_dataset_path,
    )
