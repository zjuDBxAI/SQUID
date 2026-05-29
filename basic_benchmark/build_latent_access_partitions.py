import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from controller.latent_access.load_result_to_database import build_and_materialize_latent_access_plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build and materialize LatentAccess partitions.")
    parser.add_argument("--atom-count", type=int, default=32)
    parser.add_argument("--semantic-cell-count", type=int, default=64)
    parser.add_argument("--residual-quantile", type=float, default=0.9)
    parser.add_argument("--access-weight", type=float, default=1.0)
    parser.add_argument("--semantic-weight", type=float, default=0.35)
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
    parser.add_argument("--create-indexes", action="store_true")
    parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    args = parser.parse_args()

    build_and_materialize_latent_access_plan(
        atom_count=args.atom_count,
        semantic_cell_count=args.semantic_cell_count,
        residual_quantile=args.residual_quantile,
        access_weight=args.access_weight,
        semantic_weight=args.semantic_weight,
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
        create_indexes=args.create_indexes,
        index_type=args.index_type,
    )
