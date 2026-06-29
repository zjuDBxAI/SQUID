import argparse
import importlib
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from basic_benchmark import efconfig
from basic_benchmark.common_function import get_index_type, prepare_query_dataset, run_test
from controller.baseline.acl_partition import (
    build_and_materialize_acl_partition_plan,
    create_indexes_for_materialized_partitions,
    drop_indexes_for_materialized_partitions,
    list_current_plan_partition_tables,
)


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _sync_efconfig_value(name: str, value) -> None:
    setattr(efconfig, name, value)
    try:
        top_level_efconfig = importlib.import_module("efconfig")
    except Exception:
        return
    setattr(top_level_efconfig, name, value)


def _ensure_index_state(index_type: str) -> None:
    table_names = list_current_plan_partition_tables()
    missing = []
    wrong_type = []
    for table_name in table_names:
        actual_type = get_index_type(table_name)
        if actual_type is None:
            missing.append(table_name)
        elif actual_type != index_type:
            wrong_type.append(table_name)
    if missing or wrong_type:
        drop_indexes_for_materialized_partitions()
        create_indexes_for_materialized_partitions(index_type=index_type)


def test_acl_partition_search(
    iterations=1,
    enable_index=True,
    index_type="hnsw",
    statistics_type="sql",
    generator_type="tree-based",
    record_recall=True,
    warm_up=True,
    query_num=1000,
    prepare_before_test=False,
    ef_search=100,
    document_limit=None,
    result_tag=None,
    use_ground_truth_cache=False,
    show_progress=True,
):
    queries = prepare_query_dataset(regenerate=False, num_queries=max(1, int(query_num)))

    if prepare_before_test:
        build_and_materialize_acl_partition_plan(
            document_limit=document_limit,
            create_indexes=False,
            index_type=index_type,
            show_progress=bool(show_progress),
        )

    partitions = list_current_plan_partition_tables()
    if not partitions:
        raise RuntimeError("No ACL partitions are materialized. Run with --prepare true first.")

    if enable_index:
        _ensure_index_state(index_type)
    else:
        drop_indexes_for_materialized_partitions()

    _sync_efconfig_value("ef_search", int(ef_search))
    _sync_efconfig_value("acl_partition_ef_search", int(ef_search))

    effective_generator_type = str(generator_type or "")
    if result_tag:
        effective_generator_type = f"{effective_generator_type}_{result_tag}" if effective_generator_type else str(result_tag)

    run_test(
        queries,
        "acl_partition",
        iterations=iterations,
        enable_index=enable_index,
        statistics_type=statistics_type,
        generator_type=effective_generator_type,
        index_type=index_type,
        record_recall=record_recall,
        warm_up=warm_up,
        use_ground_truth_cache=use_ground_truth_cache,
        queries_num=max(1, int(query_num)),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run ACL-per-partition baseline benchmark.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--enable-index", type=_str_to_bool, default=True)
    parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    parser.add_argument("--statistics-type", choices=["sql", "system"], default="sql")
    parser.add_argument("--generator-type", default="tree-based")
    parser.add_argument("--record-recall", type=_str_to_bool, default=True)
    parser.add_argument("--warm-up", type=_str_to_bool, default=True)
    parser.add_argument("--query-num", type=int, default=1000)
    parser.add_argument("--prepare", type=_str_to_bool, default=False)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--result-tag", default=None)
    parser.add_argument("--use-ground-truth-cache", type=_str_to_bool, default=False)
    parser.add_argument("--show-progress", type=_str_to_bool, default=True)
    args = parser.parse_args()

    test_acl_partition_search(
        iterations=args.iterations,
        enable_index=args.enable_index,
        index_type=args.index_type,
        statistics_type=args.statistics_type,
        generator_type=args.generator_type,
        record_recall=args.record_recall,
        warm_up=args.warm_up,
        query_num=args.query_num,
        prepare_before_test=args.prepare,
        ef_search=args.ef_search,
        document_limit=args.document_limit,
        result_tag=args.result_tag,
        use_ground_truth_cache=args.use_ground_truth_cache,
        show_progress=args.show_progress,
    )
