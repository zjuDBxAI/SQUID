from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[3]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.append(repo_root_str)

from controller.baseline.HQI.qd_tree import (  # pylint: disable=wrong-import-position
    DEFAULT_QD_TREE_PATH,
    build_rbac_qd_tree,
    load_documents_from_db,
    load_query_workload,
    save_qd_tree,
)

DEFAULT_NUM_CENTROIDS = 16
DEFAULT_BATCH_SIZE = 5000
DEFAULT_QUERY_WORKLOAD = repo_root / "basic_benchmark" / "query_dataset.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build an HQI RBAC-aware QD-tree. Only the leaf min size is configurable."
    )
    parser.add_argument(
        "--min-size",
        type=int,
        default=512,
        help="Minimum number of blocks per leaf partition.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    args = parse_args()

    logging.info(
        "Starting QD-tree build with min_partition_size=%s", args.min_size
    )
    logging.info(
        "Using fixed parameters: num_centroids=%s, max_depth=%s, batch_size=%s",
        DEFAULT_NUM_CENTROIDS,
        "unbounded",
        DEFAULT_BATCH_SIZE,
    )

    documents = load_documents_from_db(batch_size=DEFAULT_BATCH_SIZE, limit=None)

    queries = []
    if DEFAULT_QUERY_WORKLOAD.exists():
        queries = load_query_workload(str(DEFAULT_QUERY_WORKLOAD), limit=None)
        logging.info(
            "Loaded %s queries for workload-aware splitting", len(queries)
        )
    else:
        logging.info(
            "Workload file %s not found; proceeding without queries.",
            DEFAULT_QUERY_WORKLOAD,
        )

    tree_root = build_rbac_qd_tree(
        documents,
        queries,
        num_centroids=DEFAULT_NUM_CENTROIDS,
        min_partition_size=args.min_size,
        max_depth=None,
        include_centroid_predicates=True,
    )

    output_path = DEFAULT_QD_TREE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_qd_tree(tree_root, str(output_path))
    logging.info("QD-tree saved to %s", output_path)


if __name__ == "__main__":
    main()
