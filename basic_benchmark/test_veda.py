"""Benchmark harness for the PostgreSQL Veda/EffVeda baseline."""

import argparse
import importlib
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from basic_benchmark import efconfig
from basic_benchmark.common_function import get_index_type, prepare_query_dataset, run_test
from controller.baseline.veda import (
    build_and_materialize_veda_plan,
    create_indexes_for_materialized_partitions,
    drop_indexes_for_materialized_partitions,
    load_current_partitions,
)
from controller.baseline.veda.common import DEFAULT_QUERY_DATASET_PATH


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


def _ratio_tag(value) -> str:
    text = f"{float(value):.6g}"
    return text.replace("-", "m").replace(".", "p")


def _normalize_search_mode(value: str) -> str:
    normalized = str(value or "coordinated").strip().lower().replace("-", "_")
    if normalized in {"coordinated", "coord", "effveda", "default"}:
        return "coordinated"
    if normalized in {"naive", "baseline", "simple"}:
        return "naive"
    if normalized in {"ours", "kmeans", "adaptive", "single_pass"}:
        return "ours"
    raise argparse.ArgumentTypeError(f"Invalid search mode: {value}")


def _normalize_sql_timing_mode(value: str) -> str:
    normalized = str(value or "fair").strip().lower().replace("-", "_")
    if normalized in {"fair", "aligned", "all_sql", "full_sql"}:
        return "fair"
    if normalized in {"legacy", "paper", "original"}:
        return "legacy"
    raise argparse.ArgumentTypeError(f"Invalid SQL timing mode: {value}")


def _compose_generator_label(
    generator_type: str,
    algorithm: str,
    result_tag,
    storage_amplification: float,
    threshold: int,
    search_mode: str,
    sql_timing_mode: str,
) -> str:
    label = str(generator_type or "")
    tag = result_tag or f"{algorithm}_sa{_ratio_tag(storage_amplification)}_l{int(threshold)}"
    if search_mode:
        tag = f"{tag}_search_{search_mode}"
    if sql_timing_mode:
        tag = f"{tag}_timing_{sql_timing_mode}"
    safe_tag = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in str(tag))
    return f"{label}_{safe_tag}" if label else safe_tag


def _ensure_index_state(partitions, index_type: str) -> None:
    missing = []
    wrong_type = []
    for partition in partitions:
        if str(partition.node_kind) != "index":
            continue
        actual_type = get_index_type(partition.table_name)
        if actual_type is None:
            missing.append(partition.table_name)
        elif actual_type != index_type:
            wrong_type.append(partition.table_name)
    if missing or wrong_type:
        drop_indexes_for_materialized_partitions()
        create_indexes_for_materialized_partitions(index_type=index_type)


def test_veda_search(
    iterations=1,
    enable_index=True,
    index_type="hnsw",
    statistics_type="sql",
    generator_type="tree-based",
    record_recall=True,
    warm_up=True,
    query_num=1000,
    prepare_before_test=False,
    algorithm="effveda",
    indexing_threshold=1000,
    storage_amplification=1.2,
    ef_search=100,
    document_limit=None,
    hnsw_iterative_scan="off",
    hnsw_max_scan_tuples=None,
    search_mode="coordinated",
    sql_timing_mode="fair",
    result_tag=None,
    use_ground_truth_cache=False,
    query_dataset_path=DEFAULT_QUERY_DATASET_PATH,
    show_progress=True,
):
    effective_query_num = max(1, int(query_num))
    queries = prepare_query_dataset(
        regenerate=False,
        num_queries=effective_query_num,
        query_dataset_path=query_dataset_path or DEFAULT_QUERY_DATASET_PATH,
    )

    if prepare_before_test:
        build_and_materialize_veda_plan(
            algorithm=algorithm,
            indexing_threshold=int(indexing_threshold),
            storage_amplification=float(storage_amplification),
            ef_search=int(ef_search),
            document_limit=document_limit,
            create_indexes=False,
            index_type=index_type,
            show_progress=bool(show_progress),
        )

    partitions = load_current_partitions(refresh=True)
    if not partitions:
        raise RuntimeError("No Veda nodes are materialized. Run with --prepare true first.")

    if enable_index:
        _ensure_index_state(partitions, index_type)
    else:
        drop_indexes_for_materialized_partitions()

    _sync_efconfig_value("ef_search", int(ef_search))
    _sync_efconfig_value("veda_ef_search", int(ef_search))
    _sync_efconfig_value("veda_hnsw_iterative_scan", str(hnsw_iterative_scan))
    if hnsw_max_scan_tuples is None:
        _sync_efconfig_value("veda_hnsw_max_scan_tuples", None)
    else:
        _sync_efconfig_value("veda_hnsw_max_scan_tuples", int(hnsw_max_scan_tuples))
    normalized_search_mode = _normalize_search_mode(search_mode)
    normalized_sql_timing_mode = _normalize_sql_timing_mode(sql_timing_mode)
    _sync_efconfig_value("veda_search_mode", normalized_search_mode)
    _sync_efconfig_value("veda_sql_timing_mode", normalized_sql_timing_mode)

    effective_generator_type = _compose_generator_label(
        generator_type,
        str(algorithm),
        result_tag,
        float(storage_amplification),
        int(indexing_threshold),
        normalized_search_mode,
        normalized_sql_timing_mode,
    )
    run_test(
        queries,
        "veda",
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
    parser = argparse.ArgumentParser(description="Run Veda/EffVeda benchmark.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--enable-index", type=_str_to_bool, default=True)
    parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    parser.add_argument("--statistics-type", choices=["sql", "system"], default="sql")
    parser.add_argument("--generator-type", default="tree-based")
    parser.add_argument("--record-recall", type=_str_to_bool, default=True)
    parser.add_argument("--warm-up", type=_str_to_bool, default=True)
    parser.add_argument("--query-num", type=int, default=1000)
    parser.add_argument("--prepare", type=_str_to_bool, default=False)
    parser.add_argument("--algorithm", choices=["veda", "effveda"], default="effveda")
    parser.add_argument("--indexing-threshold", type=int, default=1000)
    parser.add_argument("--storage-amplification", type=float, default=1.2)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--hnsw-iterative-scan", default="off")
    parser.add_argument("--hnsw-max-scan-tuples", type=int, default=None)
    parser.add_argument("--search-mode", type=_normalize_search_mode, default="coordinated")
    parser.add_argument("--sql-timing-mode", type=_normalize_sql_timing_mode, default="fair")
    parser.add_argument("--result-tag", default=None)
    parser.add_argument("--use-ground-truth-cache", type=_str_to_bool, default=False)
    parser.add_argument("--query-dataset-path", default=DEFAULT_QUERY_DATASET_PATH)
    parser.add_argument("--show-progress", type=_str_to_bool, default=True)
    args = parser.parse_args()

    test_veda_search(
        iterations=args.iterations,
        enable_index=args.enable_index,
        index_type=args.index_type,
        statistics_type=args.statistics_type,
        generator_type=args.generator_type,
        record_recall=args.record_recall,
        warm_up=args.warm_up,
        query_num=args.query_num,
        prepare_before_test=args.prepare,
        algorithm=args.algorithm,
        indexing_threshold=args.indexing_threshold,
        storage_amplification=args.storage_amplification,
        ef_search=args.ef_search,
        document_limit=args.document_limit,
        hnsw_iterative_scan=args.hnsw_iterative_scan,
        hnsw_max_scan_tuples=args.hnsw_max_scan_tuples,
        search_mode=args.search_mode,
        sql_timing_mode=args.sql_timing_mode,
        result_tag=args.result_tag,
        use_ground_truth_cache=args.use_ground_truth_cache,
        query_dataset_path=args.query_dataset_path,
        show_progress=args.show_progress,
    )
