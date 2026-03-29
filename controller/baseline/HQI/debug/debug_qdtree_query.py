"""
Interactive helper for inspecting a single QD-tree query.

The script shows:
  * user roles
  * candidate partitions selected by the tree (after role filtering)
  * per-partition metadata (required roles, document count)
  * optional SQL results with distances and document roles
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, List, Tuple

import psycopg2
from psycopg2 import sql
import numpy as np

repo_root = Path(__file__).resolve().parents[4]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.append(repo_root_str)

from controller.baseline.HQI.qd_tree import (  # pylint: disable=wrong-import-position
    DEFAULT_QD_TREE_PATH,
    DEFAULT_QD_TREE_PARTITION_PREFIX,
    QDTreeNode,
    get_qd_tree_root,
    load_query_workload,
    _collect_relevant_partitions,
    _collect_partition_document_ids_for_user,
    gather_role_accessible_partitions,
    partition_has_accessible_documents,
    _merge_qd_tree_results,
    _prepare_query_vector,
    _fetch_user_roles,
)
from controller.baseline.pg_row_security.row_level_security import (  # pylint: disable=wrong-import-position
    get_db_connection_for_many_users,
)


def _partition_label(partition: QDTreeNode, prefix: str) -> str:
    pid = partition.partition_id
    table = partition.table_name or f"{prefix}_{pid}"
    roles = sorted(getattr(partition, "required_roles", []))
    size = len(partition.documents)
    return f"{table} (pid={pid}, docs={size}, required_roles={roles})"


def debug_query(
    tree_path: Path,
    partition_prefix: str,
    query_json: Path,
    query_index: int,
    run_sql: bool,
) -> None:
    queries = load_query_workload(str(query_json))
    if query_index < 0 or query_index >= len(queries):
        raise IndexError(f"query_index {query_index} out of range (0..{len(queries)-1})")
    query = queries[query_index]

    root = get_qd_tree_root(tree_path=str(tree_path), partition_prefix=partition_prefix)
    user_id = str(query.user_id)
    user_roles = _fetch_user_roles(user_id)
    print(f"User {user_id} roles: {sorted(user_roles)}")
    print(f"Query vector length: {len(query.vector)}  top_k={query.top_k}")

    query_vec, query_param = _prepare_query_vector(query.vector)
    centroid_partitions = _collect_relevant_partitions(root, user_roles, query_vec)
    centroid_with_access = [
        partition
        for partition in centroid_partitions
        if partition_has_accessible_documents(partition, user_roles)
    ]
    role_partitions = gather_role_accessible_partitions(root, user_roles)
    if centroid_with_access:
        print(
            "Centroid partitions contain accessible docs:",
            [p.table_name or p.partition_id for p in centroid_with_access],
        )
    else:
        print("Centroid partitions have no accessible docs.")
    print(
        "Role-accessible partitions:",
        [p.table_name or p.partition_id for p in role_partitions],
    )

    partitions_dict = {}
    for partition in centroid_with_access:
        key = partition.table_name or f"{partition_prefix}_{partition.partition_id}"
        partitions_dict[key] = partition
    for partition in role_partitions:
        key = partition.table_name or f"{partition_prefix}_{partition.partition_id}"
        partitions_dict.setdefault(key, partition)
    filtered_partitions = list(partitions_dict.values())
    if not filtered_partitions:
        print("No partitions matched this query (after role filtering).")
        return

    print("\nCandidate partitions:")
    for partition in filtered_partitions:
        print("  -", _partition_label(partition, partition_prefix))

    if not run_sql:
        return

    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()
    try:
        aggregated_results: List[Tuple[int, int, Any, float]] = []
        for partition in filtered_partitions:
            table_name = partition.table_name or f"{partition_prefix}_{partition.partition_id}"
            print(f"\nExecuting SQL on {table_name}")
            # Determine the document IDs this user may see inside the partition.
            doc_id_filter = sorted(
                _collect_partition_document_ids_for_user(partition, user_roles)
            )
            if not doc_id_filter:
                print("  (skip) No documents remain after applying role filter.")
                continue
            cur.execute(
                sql.SQL(
                    """
                    SELECT block_id, document_id, block_content,
                           vector <-> %s::vector AS distance
                    FROM {}
                    WHERE document_id = ANY(%s)
                    ORDER BY distance
                    LIMIT %s;
                    """
                ).format(sql.Identifier(table_name)),
                (query_param, doc_id_filter, query.top_k),
            )
            rows = cur.fetchall()
            for block_id, document_id, block_content, distance in rows:
                blk_id = int(block_id)
                doc_id = int(document_id)
                aggregated_results.append((blk_id, doc_id, block_content, float(distance)))
                doc = partition.document_map.get((doc_id, blk_id))
                doc_roles = sorted(doc.accessible_roles) if doc else []
                print(
                    f"    block_id={block_id} document_id={document_id} "
                    f"distance={distance:.6f} roles={doc_roles}"
                )

        if aggregated_results:
            merged = _merge_qd_tree_results(aggregated_results, query.top_k)
            print("\nMerged top-k results (deduplicated):")
            for block_id, document_id, _content, distance in merged:
                print(f"  block_id={block_id} document_id={document_id} distance={distance:.6f}")
        else:
            print("\nNo rows returned from partition SQL queries.")
    finally:
        cur.close()
        conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Debug a single QD-tree query.")
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
        help="Partition prefix used when materializing tables.",
    )
    parser.add_argument(
        "--query-json",
        type=str,
        default=str(repo_root / "basic_benchmark" / "query_dataset.json"),
        help="Path to the query workload JSON file.",
    )
    parser.add_argument(
        "--query-index",
        type=int,
        default=0,
        help="Index of the query (in the JSON file) to inspect.",
    )
    parser.add_argument(
        "--run-sql",
        action="store_true",
        help="Execute SQL searches on the matched partitions.",
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
    query_json = Path(args.query_json).expanduser().resolve()

    debug_query(
        tree_path=tree_path,
        partition_prefix=args.partition_prefix,
        query_json=query_json,
        query_index=args.query_index,
        run_sql=args.run_sql,
    )


if __name__ == "__main__":
    main()
