"""Benchmark harness for the PostgreSQL SIEVE baseline."""

import argparse
import importlib
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from basic_benchmark import efconfig
from basic_benchmark.common_function import prepare_query_dataset, run_test, get_index_type
from controller.baseline.SIEVE import (
    build_and_materialize_sieve_plan,
    create_indexes_for_materialized_partitions,
    drop_indexes_for_materialized_partitions,
    load_current_partitions,
)

DEFAULT_QUERY_DATASET_PATH = os.path.join(project_root, "basic_benchmark", "query_dataset.json")


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
    partitions = load_current_partitions(refresh=True)
    if not partitions:
        raise RuntimeError("No SIEVE partitions are materialized. Run with --prepare true first.")
    missing = []
    wrong_type = []
    for partition in partitions:
        actual_type = get_index_type(partition.table_name)
        if actual_type is None:
            missing.append(partition.table_name)
        elif actual_type != index_type:
            wrong_type.append(partition.table_name)
    actual_root_type = get_index_type("sieve_documentblocks_root")
    if actual_root_type is None:
        missing.append("sieve_documentblocks_root")
    elif actual_root_type != index_type:
        wrong_type.append("sieve_documentblocks_root")
    if missing or wrong_type:
        drop_indexes_for_materialized_partitions()
        create_indexes_for_materialized_partitions(index_type=index_type)


def test_sieve_partition_search(
    iterations=1,
    enable_index=True,
    index_type="hnsw",
    statistics_type="sql",
    generator_type="tree-based",
    record_recall=True,
    warm_up=True,
    query_num=1000,
    prepare_before_test=False,
    query_dataset_path=DEFAULT_QUERY_DATASET_PATH,
    historical_filters_percentage=0.25,
    workload_window_size=1000000,
    index_budget=2.0,
    bitvector_cutoff=1000,
    m=16,
    ef_construction=40,
    ef_search=10,
    heterogeneous_indexing=True,
    heterogeneous_search=True,
    enable_multipartition_search=False,
    document_limit=None,
    show_progress=True,
):
    effective_query_num = max(1, int(query_num))
    queries = prepare_query_dataset(
        regenerate=False,
        num_queries=effective_query_num,
        query_dataset_path=query_dataset_path or DEFAULT_QUERY_DATASET_PATH,
    )

    _sync_efconfig_value("ef_search", int(ef_search))
    _sync_efconfig_value("sieve_ef_search", int(ef_search))

    if prepare_before_test:
        build_and_materialize_sieve_plan(
            query_dataset_path=query_dataset_path or DEFAULT_QUERY_DATASET_PATH,
            historical_filters_percentage=float(historical_filters_percentage),
            workload_window_size=int(workload_window_size),
            index_budget=float(index_budget),
            bitvector_cutoff=int(bitvector_cutoff),
            m=int(m),
            ef_construction=int(ef_construction),
            ef_search=int(ef_search),
            heterogeneous_indexing=bool(heterogeneous_indexing),
            heterogeneous_search=bool(heterogeneous_search),
            enable_multipartition_search=bool(enable_multipartition_search),
            document_limit=document_limit,
            create_indexes=False,
            index_type=index_type,
            show_progress=bool(show_progress),
        )

    partitions = load_current_partitions(refresh=True)
    if not partitions:
        raise RuntimeError("No SIEVE partitions are materialized. Run with --prepare true first.")

    if enable_index:
        _ensure_index_state(index_type)
    else:
        drop_indexes_for_materialized_partitions()

    run_test(
        queries,
        "sieve",
        iterations=iterations,
        enable_index=enable_index,
        statistics_type=statistics_type,
        generator_type=generator_type,
        index_type=index_type,
        record_recall=record_recall,
        warm_up=warm_up,
        use_ground_truth_cache=True,
        queries_num=effective_query_num,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SIEVE baseline benchmark.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--enable-index", type=_str_to_bool, default=True)
    parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    parser.add_argument("--statistics-type", choices=["sql", "system"], default="sql")
    parser.add_argument("--generator-type", default="tree-based")
    parser.add_argument("--record-recall", type=_str_to_bool, default=True)
    parser.add_argument("--warm-up", type=_str_to_bool, default=True)
    parser.add_argument("--query-num", type=int, default=1000)
    parser.add_argument("--prepare", type=_str_to_bool, default=False)
    parser.add_argument("--query-dataset-path", default=DEFAULT_QUERY_DATASET_PATH)
    parser.add_argument("--historical-filters-percentage", type=float, default=0.25)
    parser.add_argument("--workload-window-size", type=int, default=1000000)
    parser.add_argument("--index-budget", type=float, default=2.0)
    parser.add_argument("--bitvector-cutoff", type=int, default=1000)
    parser.add_argument("--m", type=int, default=16)
    parser.add_argument("--ef-construction", type=int, default=40)
    parser.add_argument("--ef-search", type=int, default=10)
    parser.add_argument("--heterogeneous-indexing", type=_str_to_bool, default=True)
    parser.add_argument("--heterogeneous-search", type=_str_to_bool, default=True)
    parser.add_argument("--enable-multipartition-search", type=_str_to_bool, default=False)
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--show-progress", type=_str_to_bool, default=True)
    args = parser.parse_args()

    test_sieve_partition_search(
        iterations=args.iterations,
        enable_index=args.enable_index,
        index_type=args.index_type,
        statistics_type=args.statistics_type,
        generator_type=args.generator_type,
        record_recall=args.record_recall,
        warm_up=args.warm_up,
        query_num=args.query_num,
        prepare_before_test=args.prepare,
        query_dataset_path=args.query_dataset_path,
        historical_filters_percentage=args.historical_filters_percentage,
        workload_window_size=args.workload_window_size,
        index_budget=args.index_budget,
        bitvector_cutoff=args.bitvector_cutoff,
        m=args.m,
        ef_construction=args.ef_construction,
        ef_search=args.ef_search,
        heterogeneous_indexing=args.heterogeneous_indexing,
        heterogeneous_search=args.heterogeneous_search,
        enable_multipartition_search=args.enable_multipartition_search,
        document_limit=args.document_limit,
        show_progress=args.show_progress,
    )
