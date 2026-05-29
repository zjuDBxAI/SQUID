from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import statistics
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BASIC_BENCHMARK_ROOT = PROJECT_ROOT / "basic_benchmark"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(BASIC_BENCHMARK_ROOT) not in sys.path:
    sys.path.append(str(BASIC_BENCHMARK_ROOT))

from controller.latent_access.load_result_to_database import (
    build_and_materialize_latent_access_plan,
    clear_current_plan,
    create_indexes_for_materialized_partitions,
    drop_materialized_partitions,
    get_current_plan_summary,
    initialize_plan_schema,
    load_current_partitions,
)


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _add_training_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--atom-count", type=int, default=16)
    parser.add_argument("--semantic-cell-count", type=int, default=24)
    parser.add_argument("--residual-quantile", type=float, default=0.95)
    parser.add_argument("--access-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=0.3)
    parser.add_argument("--semantic-knn", type=int, default=8)
    parser.add_argument("--semantic-knn-weight", type=float, default=0.15)
    parser.add_argument("--max-atoms-per-semantic-cell", type=int, default=3)
    parser.add_argument("--min-partition-documents", type=int, default=8)
    parser.add_argument("--sparsity", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=10)
    parser.add_argument("--z-inner-iterations", type=int, default=3)
    parser.add_argument("--momentum-weight", type=float, default=0.05)
    parser.add_argument("--min-atom-support", type=float, default=4.0)
    parser.add_argument("--revive-every", type=int, default=3)
    parser.add_argument("--revive-residual-quantile", type=float, default=0.92)
    parser.add_argument("--training-limit", type=int, default=None)
    parser.add_argument("--route-limit", type=int, default=16)
    parser.add_argument("--partition-fetch-multiplier", type=int, default=6)


def _training_kwargs_from_args(args: argparse.Namespace) -> dict:
    return {
        "atom_count": args.atom_count,
        "semantic_cell_count": args.semantic_cell_count,
        "residual_quantile": args.residual_quantile,
        "access_weight": args.access_weight,
        "semantic_weight": args.semantic_weight,
        "semantic_knn": args.semantic_knn,
        "semantic_knn_weight": args.semantic_knn_weight,
        "max_atoms_per_semantic_cell": args.max_atoms_per_semantic_cell,
        "min_partition_documents": args.min_partition_documents,
        "sparsity": args.sparsity,
        "max_iterations": args.max_iterations,
        "z_inner_iterations": args.z_inner_iterations,
        "momentum_weight": args.momentum_weight,
        "min_atom_support": args.min_atom_support,
        "revive_every": args.revive_every,
        "revive_residual_quantile": args.revive_residual_quantile,
        "training_limit": args.training_limit,
    }


def _benchmark_search_kwargs_from_args(args: argparse.Namespace) -> dict:
    return {
        "route_limit": args.route_limit,
        "partition_fetch_multiplier": args.partition_fetch_multiplier,
        "use_ground_truth_cache": args.use_ground_truth_cache,
    }


def _partition_stats(partitions) -> dict:
    if not partitions:
        return {
            "partition_count": 0,
            "main_partition_count": 0,
            "residual_partition_count": 0,
            "singleton_partition_count": 0,
            "avg_docs_per_partition": 0.0,
            "median_docs_per_partition": 0.0,
            "max_docs_per_partition": 0,
        }
    doc_counts = [partition.document_count for partition in partitions]
    residual_count = sum(1 for partition in partitions if partition.residual_flag)
    singleton_count = sum(1 for partition in partitions if partition.document_count == 1)
    return {
        "partition_count": len(partitions),
        "main_partition_count": len(partitions) - residual_count,
        "residual_partition_count": residual_count,
        "singleton_partition_count": singleton_count,
        "avg_docs_per_partition": float(statistics.mean(doc_counts)),
        "median_docs_per_partition": float(statistics.median(doc_counts)),
        "max_docs_per_partition": int(max(doc_counts)),
    }


def _json_default(value):
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def _print_json(payload: dict) -> None:
    print(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True, default=_json_default))


def command_init(_: argparse.Namespace) -> None:
    initialize_plan_schema()
    print("Initialized latent_access control-plane schema.")


def command_build(args: argparse.Namespace) -> None:
    plan = build_and_materialize_latent_access_plan(
        **_training_kwargs_from_args(args),
        create_indexes=args.create_indexes,
        index_type=args.index_type,
    )
    payload = {
        "plan_metadata": plan.metadata,
        "training_metadata": plan.model.training_metadata,
        "partition_stats": _partition_stats(plan.partitions),
        "sample_partition_ids": [partition.partition_id for partition in plan.partitions[:10]],
    }
    _print_json(payload)


def command_summary(_: argparse.Namespace) -> None:
    summary = get_current_plan_summary(refresh=True)
    partitions = load_current_partitions(refresh=True)
    if summary is None:
        print("No current latent_access plan found.")
        return
    payload = {
        "plan_summary": summary,
        "partition_stats": _partition_stats(partitions),
        "sample_partition_ids": [partition.partition_id for partition in partitions[:10]],
    }
    _print_json(payload)


def command_index(args: argparse.Namespace) -> None:
    create_indexes_for_materialized_partitions(
        index_type=args.index_type,
        parallel=not args.no_parallel,
        max_workers=args.max_workers,
        hnsw_m=args.hnsw_m,
        hnsw_ef_construction=args.hnsw_ef_construction,
        hnsw_threads=args.hnsw_threads,
        disable_sync_commit=not args.keep_sync_commit,
    )
    print(f"Built indexes for latent_access partitions with index_type={args.index_type}.")


def command_benchmark(args: argparse.Namespace) -> None:
    from basic_benchmark.test_latent_access import test_latent_access_search

    test_latent_access_search(
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
        **_training_kwargs_from_args(args),
        **_benchmark_search_kwargs_from_args(args),
    )
    print("Completed latent_access benchmark run.")


def command_clear(_: argparse.Namespace) -> None:
    drop_materialized_partitions()
    clear_current_plan()
    print("Cleared latent_access materialized partitions and current plan metadata.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified CLI for latent_access experiments.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="Initialize latent_access control-plane tables.")
    init_parser.set_defaults(func=command_init)

    build_parser = subparsers.add_parser("build", help="Train, plan, and materialize a latent_access layout.")
    _add_training_arguments(build_parser)
    build_parser.add_argument("--create-indexes", action="store_true")
    build_parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    build_parser.set_defaults(func=command_build)

    summary_parser = subparsers.add_parser("summary", help="Show the current latent_access plan summary.")
    summary_parser.set_defaults(func=command_summary)

    index_parser = subparsers.add_parser("index", help="Build ANN indexes for current latent_access partitions.")
    index_parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    index_parser.add_argument("--no-parallel", action="store_true")
    index_parser.add_argument("--max-workers", type=int, default=None)
    index_parser.add_argument("--hnsw-m", type=int, default=16)
    index_parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    index_parser.add_argument("--hnsw-threads", type=int, default=None)
    index_parser.add_argument("--keep-sync-commit", action="store_true")
    index_parser.set_defaults(func=command_index)

    benchmark_parser = subparsers.add_parser("benchmark", help="Run the benchmark-facing latent_access search pipeline.")
    _add_training_arguments(benchmark_parser)
    benchmark_parser.add_argument("--iterations", type=int, default=1)
    benchmark_parser.add_argument("--enable-index", type=_str_to_bool, default=True)
    benchmark_parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    benchmark_parser.add_argument("--statistics-type", choices=["sql", "system"], default="sql")
    benchmark_parser.add_argument("--generator-type", default="")
    benchmark_parser.add_argument("--record-recall", type=_str_to_bool, default=True)
    benchmark_parser.add_argument("--warm-up", type=_str_to_bool, default=True)
    benchmark_parser.add_argument("--query-num", type=int, default=1000)
    benchmark_parser.add_argument("--prepare", type=_str_to_bool, default=False)
    benchmark_parser.add_argument("--ef-search", default="adaptive")
    benchmark_parser.add_argument("--use-ground-truth-cache", type=_str_to_bool, default=False)
    benchmark_parser.set_defaults(func=command_benchmark)

    clear_parser = subparsers.add_parser("clear", help="Drop materialized partitions and clear current plan metadata.")
    clear_parser.set_defaults(func=command_clear)
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
