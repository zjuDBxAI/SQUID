"""Export the QD-tree structure to a Graphviz DOT file."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[4]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.append(repo_root_str)

from controller.baseline.HQI.qd_tree import (  # pylint: disable=wrong-import-position
    DEFAULT_QD_TREE_PARTITION_PREFIX,
    DEFAULT_QD_TREE_PATH,
    export_qd_tree_to_dot,
    get_qd_tree_root,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export QD-tree to Graphviz DOT.")
    parser.add_argument(
        "--tree-path",
        type=str,
        default=str(DEFAULT_QD_TREE_PATH),
        help="Path to the serialized QD-tree pickle.",
    )
    parser.add_argument(
        "--partition-prefix",
        type=str,
        default=DEFAULT_QD_TREE_PARTITION_PREFIX,
        help="Partition prefix used when assigning IDs.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="qd_tree.dot",
        help="Destination DOT filename.",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=None,
        help="Optional depth limit for exported nodes.",
    )
    parser.add_argument(
        "--include-document-roles",
        action="store_true",
        help="Include the union of document roles in leaf labels.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    tree_path = Path(args.tree_path).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    logging.info("Loading QD-tree from %s", tree_path)
    root = get_qd_tree_root(tree_path=str(tree_path), partition_prefix=args.partition_prefix)

    logging.info(
        "Writing DOT file to %s (max_depth=%s include_document_roles=%s)",
        output_path,
        args.max_depth,
        args.include_document_roles,
    )
    export_qd_tree_to_dot(
        root=root,
        output_path=output_path,
        max_depth=args.max_depth,
        include_document_roles=args.include_document_roles,
    )
    logging.info("Done.")


if __name__ == "__main__":
    main()
