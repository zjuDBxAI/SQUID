"""Print the structure of a persisted QD-tree."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List

repo_root = Path(__file__).resolve().parents[4]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.append(repo_root_str)

from controller.baseline.HQI.qd_tree import (  # pylint: disable=wrong-import-position
    DEFAULT_QD_TREE_PARTITION_PREFIX,
    DEFAULT_QD_TREE_PATH,
    extract_leaf_partitions,
    get_qd_tree_root,
    QDTreeNode,
)


def collect_nodes(root: QDTreeNode) -> List[QDTreeNode]:
    stack = [root]
    nodes: List[QDTreeNode] = []
    while stack:
        node = stack.pop()
        nodes.append(node)
        if node.left_child is not None:
            stack.append(node.left_child)
        if node.right_child is not None:
            stack.append(node.right_child)
    return nodes


def dump_tree(root: QDTreeNode) -> None:
    nodes = collect_nodes(root)
    max_depth = max(node.depth for node in nodes)
    by_depth = {depth: [] for depth in range(max_depth + 1)}
    for node in nodes:
        if node.is_leaf():
            label = (
                f"Leaf pid={node.partition_id} size={len(node.documents)}"
                f" roles={sorted(node.required_roles) if hasattr(node, 'required_roles') else []}"
            )
        else:
            predicate = node.split_predicate
            label = f"Split {predicate}"
        by_depth[node.depth].append(label)

    for depth in range(max_depth + 1):
        print(f"Depth {depth}:")
        for label in by_depth[depth]:
            print(f"  - {label}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dump QD-tree structure")
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
    root = get_qd_tree_root(tree_path=str(tree_path), partition_prefix=args.partition_prefix)
    dump_tree(root)


if __name__ == "__main__":
    main()

