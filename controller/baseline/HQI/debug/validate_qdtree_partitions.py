"""Utility to verify QD-tree partitions respect role permissions."""

from __future__ import annotations

import argparse
import logging
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

repo_root = Path(__file__).resolve().parents[4]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.append(repo_root_str)

from controller.baseline.HQI.qd_tree import (  # pylint: disable=wrong-import-position
    DEFAULT_QD_TREE_PARTITION_PREFIX,
    DEFAULT_QD_TREE_PATH,
    extract_leaf_partitions,
    get_qd_tree_root,
)
from services.config import get_db_connection  # pylint: disable=wrong-import-position


def fetch_document_roles(doc_ids: Iterable[int]) -> Dict[int, Set[str]]:
    """Return a mapping of document_id -> set(role_id) from the database."""
    doc_ids = sorted({int(doc_id) for doc_id in doc_ids})
    if not doc_ids:
        return {}

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT document_id, array_agg(DISTINCT role_id)
            FROM PermissionAssignment
            WHERE document_id = ANY(%s)
            GROUP BY document_id;
            """,
            (doc_ids,),
        )
        mapping: Dict[int, Set[str]] = defaultdict(set)
        for document_id, role_array in cur.fetchall():
            if role_array is None:
                continue
            mapping[int(document_id)] = {str(role_id) for role_id in role_array}
        return mapping
    finally:
        cur.close()
        conn.close()


def validate_partitions(tree_path: Path, partition_prefix: str, strict_db_check: bool) -> int:
    root = get_qd_tree_root(tree_path=str(tree_path), partition_prefix=partition_prefix)
    leaves = extract_leaf_partitions(root)
    doc_ids = {doc.doc_id for leaf in leaves for doc in leaf.documents}
    db_roles: Dict[int, Set[str]] = {}
    if strict_db_check:
        db_roles = fetch_document_roles(doc_ids)

    errors = 0
    for leaf in leaves:
        required_roles = leaf.required_roles if hasattr(leaf, "required_roles") else set()
        partition_name = leaf.table_name or f"{partition_prefix}_{leaf.partition_id}"
        for doc in leaf.documents:
            doc_roles = doc.accessible_roles
            if not required_roles.issubset(doc_roles):
                logging.error(
                    "Partition %s requires %s but document %s has roles %s",
                    partition_name,
                    sorted(required_roles),
                    doc.doc_id,
                    sorted(doc_roles),
                )
                errors += 1
                continue
            if strict_db_check:
                db_doc_roles = db_roles.get(doc.doc_id, set())
                if doc_roles != db_doc_roles:
                    logging.error(
                        "Document %s roles mismatch: tree=%s db=%s",
                        doc.doc_id,
                        sorted(doc_roles),
                        sorted(db_doc_roles),
                    )
                    errors += 1

    logging.info(
        "Validated %s partitions (%s documents). Errors: %s", len(leaves), len(doc_ids), errors
    )
    return errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate that QD-tree partitions respect role permissions."
    )
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
        help="Prefix used when materializing partition tables.",
    )
    parser.add_argument(
        "--strict-db-check",
        action="store_true",
        help="Cross-check document roles against live database values.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    tree_path = Path(args.tree_path).expanduser().resolve()
    errors = validate_partitions(
        tree_path=tree_path,
        partition_prefix=args.partition_prefix,
        strict_db_check=args.strict_db_check,
    )
    if errors:
        logging.error("Validation finished with %s issue(s).", errors)
        sys.exit(1)
    logging.info("Validation finished successfully.")


if __name__ == "__main__":
    main()

