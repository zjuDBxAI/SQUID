import argparse
import importlib
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from basic_benchmark import efconfig
from basic_benchmark.common_function import get_index_type, prepare_query_dataset, run_test
from controller.kmeans import (
    build_and_materialize_kmeans_plan,
    create_indexes_for_materialized_partitions,
    drop_indexes_for_materialized_partitions,
    load_current_partitions,
    list_materialized_partition_tables,
)
from controller.kmeans.common import DEFAULT_QUERY_DATASET_PATH


def _sync_efconfig_value(name: str, value) -> None:
    setattr(efconfig, name, value)
    try:
        top_level_efconfig = importlib.import_module("efconfig")
    except Exception:
        return
    setattr(top_level_efconfig, name, value)


def _delete_efconfig_value(name: str) -> None:
    if hasattr(efconfig, name):
        delattr(efconfig, name)
    try:
        top_level_efconfig = importlib.import_module("efconfig")
    except Exception:
        return
    if hasattr(top_level_efconfig, name):
        delattr(top_level_efconfig, name)


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


def _compose_generator_label(generator_type: str, result_tag) -> str:
    label = str(generator_type or "")
    tag = result_tag or "cost"
    safe_tag = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(tag))
    return f"{label}_{safe_tag}" if label else safe_tag


def _ensure_index_state(partitions, index_type: str) -> None:
    missing = []
    wrong_type = []
    for partition in partitions:
        actual_type = get_index_type(partition.table_name)
        if actual_type is None:
            missing.append(partition.table_name)
        elif actual_type != index_type:
            wrong_type.append(partition.table_name)
    if missing or wrong_type:
        drop_indexes_for_materialized_partitions()
        create_indexes_for_materialized_partitions(index_type=index_type)


def _skip_index_maintenance() -> bool:
    return str(os.environ.get("SKIP_INDEX_MAINTENANCE", "")).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    }


def test_kmeans_partition_search(
    iterations=1,
    enable_index=True,
    index_type="squidhnsw",
    statistics_type="sql",
    generator_type="tree-based",
    record_recall=True,
    warm_up=True,
    query_num=1000,
    prepare_before_test=False,
    private_replication_budget_ratio=0.0,
    embedding_dim=None,
    document_limit=None,
    ef_search=120,
    show_progress=True,
    result_tag=None,
    use_ground_truth_cache=False,
    query_dataset_path=DEFAULT_QUERY_DATASET_PATH,
    enable_split=True,
    private_edge_top_d=32,
    table_prefix=None,
    versioned_plan=False,
):
    effective_query_num = max(1, int(query_num))
    queries = prepare_query_dataset(
        regenerate=False,
        num_queries=effective_query_num,
        query_dataset_path=query_dataset_path or DEFAULT_QUERY_DATASET_PATH,
    )

    if prepare_before_test:
        build_and_materialize_kmeans_plan(
            private_replication_budget_ratio=float(private_replication_budget_ratio),
            ef_search=int(ef_search),
            embedding_dim=embedding_dim,
            document_limit=document_limit,
            query_dataset_path=query_dataset_path or DEFAULT_QUERY_DATASET_PATH,
            create_indexes=bool(enable_index) if bool(versioned_plan) else False,
            index_type=index_type,
            show_progress=bool(show_progress),
            enable_split=bool(enable_split),
            private_edge_top_d=int(private_edge_top_d),
            table_prefix=table_prefix,
            replace_current=not bool(versioned_plan),
            drop_stale=not bool(versioned_plan),
        )
        if versioned_plan:
            return

    partitions = load_current_partitions(refresh=True)
    if not partitions:
        raise RuntimeError("No kmeans partitions are materialized. Run with --prepare true first.")

    materialized_tables = set(list_materialized_partition_tables())
    missing_tables = [partition.table_name for partition in partitions if partition.table_name not in materialized_tables]
    if missing_tables:
        preview = ", ".join(missing_tables[:5])
        suffix = "" if len(missing_tables) <= 5 else f" ... (+{len(missing_tables) - 5} more)"
        raise RuntimeError(
            "Current kmeans plan metadata references missing partition tables: "
            f"{preview}{suffix}. Run with --prepare true to rebuild the plan and materialized tables."
        )

    if _skip_index_maintenance():
        pass
    elif enable_index:
        _ensure_index_state(partitions, index_type)
    else:
        drop_indexes_for_materialized_partitions()

    _sync_efconfig_value("kmeans_index_type", str(index_type))
    if ef_search is not None:
        _sync_efconfig_value("ef_search", int(ef_search))
        _sync_efconfig_value("kmeans_ef_search", int(ef_search))
    else:
        _delete_efconfig_value("kmeans_ef_search")
    default_tag = f"cost_split_b{_ratio_tag(private_replication_budget_ratio)}_ef{int(ef_search)}"
    effective_generator_type = _compose_generator_label(generator_type, result_tag or default_tag)
    run_test(
        queries,
        "kmeans_partition",
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
    parser = argparse.ArgumentParser(description="Run kmeans tenant-cluster partition benchmark.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--enable-index", type=_str_to_bool, default=True)
    parser.add_argument("--index-type", choices=["squidhnsw", "hnsw", "ivfflat"], default="squidhnsw")
    parser.add_argument("--statistics-type", choices=["sql", "system"], default="sql")
    parser.add_argument("--generator-type", default="tree-based")
    parser.add_argument("--record-recall", type=_str_to_bool, default=True)
    parser.add_argument("--warm-up", type=_str_to_bool, default=True)
    parser.add_argument("--query-num", type=int, default=1000)
    parser.add_argument("--prepare", type=_str_to_bool, default=False)
    parser.add_argument("--private-replication-budget-ratio", type=float, default=0.0)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--ef-search", type=int, default=120)
    parser.add_argument("--show-progress", type=_str_to_bool, default=True)
    parser.add_argument("--result-tag", default=None)
    parser.add_argument("--use-ground-truth-cache", type=_str_to_bool, default=False)
    parser.add_argument("--query-dataset-path", default=DEFAULT_QUERY_DATASET_PATH)
    parser.add_argument("--enable-split", type=_str_to_bool, default=True)
    parser.add_argument("--private-edge-top-d", type=int, default=32)
    parser.add_argument("--table-prefix", default=None)
    parser.add_argument("--versioned-plan", type=_str_to_bool, default=False)
    args = parser.parse_args()

    test_kmeans_partition_search(
        iterations=args.iterations,
        enable_index=args.enable_index,
        index_type=args.index_type,
        statistics_type=args.statistics_type,
        generator_type=args.generator_type,
        record_recall=args.record_recall,
        warm_up=args.warm_up,
        query_num=args.query_num,
        prepare_before_test=args.prepare,
        private_replication_budget_ratio=args.private_replication_budget_ratio,
        embedding_dim=args.embedding_dim,
        document_limit=args.document_limit,
        ef_search=args.ef_search,
        show_progress=args.show_progress,
        result_tag=args.result_tag,
        use_ground_truth_cache=args.use_ground_truth_cache,
        query_dataset_path=args.query_dataset_path,
        enable_split=args.enable_split,
        private_edge_top_d=args.private_edge_top_d,
        table_prefix=args.table_prefix,
        versioned_plan=args.versioned_plan,
    )
