"""List partitions associated with the first N roles in the QD-tree."""

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
    get_qd_tree_root,
    get_role_partition_index,
)


def _role_sort_key(role: str) -> tuple:
    try:
        return (0, int(role))
    except ValueError:
        return (1, role)


def parse_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="List partitions per role.")
    parser.add_argument(
        "--tree-path",
        type=str,
        default=str(DEFAULT_QD_TREE_PATH),
        help="Path to the serialized QD-tree.",
    )
    parser.add_argument(
        "--partition-prefix",
        type=str,
        default=DEFAULT_QD_TREE_PARTITION_PREFIX,
        help="Partition prefix used when materializing tables.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Number of roles to display (sorted by role id/name).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level.",
    )
    return parser


def main() -> None:
    parser = parse_args()
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    tree_path = Path(args.tree_path).expanduser().resolve()
    get_qd_tree_root(tree_path=str(tree_path), partition_prefix=args.partition_prefix)
    role_index = get_role_partition_index()
    if not role_index:
        print("Role index is empty. Ensure the QD-tree has been loaded and contains role metadata.")
        return

    roles: List[str] = sorted(role_index.keys(), key=_role_sort_key)
    limit = args.limit if args.limit is None or args.limit > 0 else len(roles)
    print(f"Listing partitions for {min(limit, len(roles))} role(s):\n")

    for role in roles[:limit]:
        partitions = sorted(role_index[role])
        print(f"role={role}  partitions={len(partitions)}")
        for table in partitions:
            print(f"  - {table}")
        print()


if __name__ == "__main__":
    main()
