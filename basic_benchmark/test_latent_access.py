import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.logger import get_logger

logger = get_logger(__name__)

from basic_benchmark.common_function import get_index_type, prepare_query_dataset, run_test
from controller.latent_access.load_result_to_database import (
    build_and_materialize_latent_access_plan,
    get_partition_table_name,
    load_current_partitions,
    drop_indexes_for_materialized_partitions,
)
from basic_benchmark import efconfig


def _collect_index_state(partitions):
    state = {"missing": [], "by_type": {}}
    for partition in partitions:
        table_name = partition.metadata.get("table_name", get_partition_table_name(partition.partition_id))
        index_type = get_index_type(table_name)
        if index_type is None:
            state["missing"].append(table_name)
            continue
        state["by_type"].setdefault(index_type, []).append(table_name)
    return state


def _validate_index_state(partitions, index_type: str) -> None:
    state = _collect_index_state(partitions)
    wrong_type_tables = []
    for actual_type, table_names in state["by_type"].items():
        if actual_type != index_type:
            wrong_type_tables.extend(table_names)
    if not state["missing"] and not wrong_type_tables:
        return
    raise RuntimeError(
        "LatentAccess indexes are not ready. Run basic_benchmark/build_latent_access_indexes.py first."
    )


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def test_latent_access_search(
    iterations=1,
    enable_index=True,
    index_type="hnsw",
    statistics_type="sql",
    generator_type="",
    record_recall=True,
    warm_up=True,
    atom_count=32,
    semantic_cell_count=64,
    residual_quantile=0.9,
    access_weight=1.0,
    semantic_weight=0.35,
    query_num=1000,
    prepare_before_test=False,
    ef_search="adaptive",
    semantic_knn=8,
    semantic_knn_weight=0.2,
    max_atoms_per_semantic_cell=4,
    min_partition_documents=4,
    sparsity=2,
    max_iterations=25,
    z_inner_iterations=4,
    momentum_weight=0.1,
    min_atom_support=1.0,
    revive_every=3,
    revive_residual_quantile=0.85,
    training_limit=None,
    route_limit=16,
    partition_fetch_multiplier=6,
    use_ground_truth_cache=False,
):
    if prepare_before_test:
        build_and_materialize_latent_access_plan(
            atom_count=atom_count,
            semantic_cell_count=semantic_cell_count,
            residual_quantile=residual_quantile,
            access_weight=access_weight,
            semantic_weight=semantic_weight,
            semantic_knn=semantic_knn,
            semantic_knn_weight=semantic_knn_weight,
            max_atoms_per_semantic_cell=max_atoms_per_semantic_cell,
            min_partition_documents=min_partition_documents,
            sparsity=sparsity,
            max_iterations=max_iterations,
            z_inner_iterations=z_inner_iterations,
            momentum_weight=momentum_weight,
            min_atom_support=min_atom_support,
            revive_every=revive_every,
            revive_residual_quantile=revive_residual_quantile,
            training_limit=training_limit,
            create_indexes=False,
            index_type=index_type,
        )

    partitions = load_current_partitions()
    if not partitions:
        raise RuntimeError(
            "No LatentAccess partitions are materialized. "
            "Run basic_benchmark/build_latent_access_partitions.py first."
        )

    if enable_index:
        _validate_index_state(partitions, index_type)
    else:
        drop_indexes_for_materialized_partitions()

    efconfig.ef_search = ef_search
    efconfig.latent_route_limit = int(route_limit)
    efconfig.latent_partition_fetch_multiplier = int(partition_fetch_multiplier)
    queries = prepare_query_dataset(regenerate=False, num_queries=query_num)
    run_test(
        queries,
        "latent_access",
        iterations=iterations,
        enable_index=enable_index,
        statistics_type=statistics_type,
        generator_type=generator_type,
        index_type=index_type,
        record_recall=record_recall,
        warm_up=warm_up,
        use_ground_truth_cache=use_ground_truth_cache,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LatentAccess benchmark pipeline.")
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--enable-index", type=_str_to_bool, default=True)
    parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    parser.add_argument("--statistics-type", choices=["sql", "system"], default="sql")
    parser.add_argument("--generator-type", default="")
    parser.add_argument("--record-recall", type=_str_to_bool, default=True)
    parser.add_argument("--warm-up", type=_str_to_bool, default=True)
    parser.add_argument("--atom-count", type=int, default=32)
    parser.add_argument("--semantic-cell-count", type=int, default=64)
    parser.add_argument("--residual-quantile", type=float, default=0.9)
    parser.add_argument("--access-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=0.35)
    parser.add_argument("--query-num", type=int, default=1000)
    parser.add_argument("--prepare", type=_str_to_bool, default=False)
    parser.add_argument("--ef-search", default="adaptive")
    parser.add_argument("--semantic-knn", type=int, default=8)
    parser.add_argument("--semantic-knn-weight", type=float, default=0.2)
    parser.add_argument("--max-atoms-per-semantic-cell", type=int, default=4)
    parser.add_argument("--min-partition-documents", type=int, default=4)
    parser.add_argument("--sparsity", type=int, default=2)
    parser.add_argument("--max-iterations", type=int, default=25)
    parser.add_argument("--z-inner-iterations", type=int, default=4)
    parser.add_argument("--momentum-weight", type=float, default=0.1)
    parser.add_argument("--min-atom-support", type=float, default=1.0)
    parser.add_argument("--revive-every", type=int, default=3)
    parser.add_argument("--revive-residual-quantile", type=float, default=0.85)
    parser.add_argument("--training-limit", type=int, default=None)
    parser.add_argument("--route-limit", type=int, default=16)
    parser.add_argument("--partition-fetch-multiplier", type=int, default=6)
    parser.add_argument("--use-ground-truth-cache", type=_str_to_bool, default=False)
    args = parser.parse_args()

    test_latent_access_search(
        iterations=args.iterations,
        enable_index=args.enable_index,
        index_type=args.index_type,
        statistics_type=args.statistics_type,
        generator_type=args.generator_type,
        record_recall=args.record_recall,
        warm_up=args.warm_up,
        atom_count=args.atom_count,
        semantic_cell_count=args.semantic_cell_count,
        residual_quantile=args.residual_quantile,
        access_weight=args.access_weight,
        semantic_weight=args.semantic_weight,
        query_num=args.query_num,
        prepare_before_test=args.prepare,
        ef_search=args.ef_search,
        semantic_knn=args.semantic_knn,
        semantic_knn_weight=args.semantic_knn_weight,
        max_atoms_per_semantic_cell=args.max_atoms_per_semantic_cell,
        min_partition_documents=args.min_partition_documents,
        sparsity=args.sparsity,
        max_iterations=args.max_iterations,
        z_inner_iterations=args.z_inner_iterations,
        momentum_weight=args.momentum_weight,
        min_atom_support=args.min_atom_support,
        revive_every=args.revive_every,
        revive_residual_quantile=args.revive_residual_quantile,
        training_limit=args.training_limit,
        route_limit=args.route_limit,
        partition_fetch_multiplier=args.partition_fetch_multiplier,
        use_ground_truth_cache=args.use_ground_truth_cache,
    )
