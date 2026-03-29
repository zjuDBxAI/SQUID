from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.append(repo_root_str)

from controller.baseline.HQI.qd_tree import (  # pylint: disable=wrong-import-position
    DEFAULT_QD_TREE_PATH,
    DEFAULT_QD_TREE_PARTITION_PREFIX,
    load_qd_tree,
    persist_partitions_to_postgres,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step 2: Persist QD-tree leaf partitions into PostgreSQL tables."
    )
    parser.add_argument(
        "--tree-path",
        type=str,
        default=str(DEFAULT_QD_TREE_PATH),
        help="Path to the serialized QD-tree produced in step 1.",
    )
    parser.add_argument(
        "--partition-prefix",
        type=str,
        default=DEFAULT_QD_TREE_PARTITION_PREFIX,
        help="Prefix for generated partition tables.",
    )
    parser.add_argument(
        "--no-drop",
        action="store_true",
        help="Do not drop existing QD-tree partition tables before persisting.",
    )
    parser.add_argument(
        "--index-type",
        choices=["hnsw", "ivfflat"],
        default="hnsw",
        help="Index type to build on the vector column for each partition.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 2) // 2),
        help="Number of parallel workers to use when materializing partitions.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    logging.info(
        "Persisting QD-tree partitions (tree_path=%s, prefix=%s, index=%s)",
        args.tree_path,
        args.partition_prefix,
        args.index_type,
    )
    tree_root = load_qd_tree(args.tree_path)
    persist_partitions_to_postgres(
        tree_root,
        partition_prefix=args.partition_prefix,
        drop_existing=not args.no_drop,
        index_type=args.index_type,
        workers=args.workers,
    )
    logging.info("Finished persisting QD-tree partitions.")


if __name__ == "__main__":
    main()
