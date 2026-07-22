#!/usr/bin/env python3
"""Collect partition-layout statistics for access-controlled ANN case studies."""

from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from psycopg2 import sql


PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT))

from services.config import get_db_connection  # noqa: E402
from basic_benchmark.direct_pg_qps import (  # noqa: E402
    PlanSelection,
    Query,
    load_honeybee_routes,
    load_ours_routes,
    load_role_routes,
    load_veda_routes,
    resolve_versioned_plan,
)


METHOD_LABEL = {
    "rls": "RLS",
    "role": "ROLE",
    "hqi": "HQI",
    "honeybee": "HONEYBEE",
    "veda": "VEDA",
    "effveda": "EFFVEDA",
    "ours": "SQUID",
}


@dataclass(frozen=True)
class UserStats:
    method: str
    user_id: int
    route_count: int
    route_vector_count: int
    accessible_vector_count: int
    selectivity: float
    impurity_factor: float
    pure_route_count: int
    mixed_route_count: int


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * q
    lower = math.floor(index)
    upper = math.ceil(index)
    if lower == upper:
        return float(ordered[int(index)])
    frac = index - lower
    return float(ordered[lower] * (1.0 - frac) + ordered[upper] * frac)


def table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", [f"public.{table_name}"])
    return bool(cur.fetchone()[0])


def fetch_scalar(cur, query: str, params: Iterable[object] = ()) -> int:
    cur.execute(query, list(params))
    row = cur.fetchone()
    return int(row[0] or 0) if row else 0


def fetch_user_visible_counts() -> dict[int, int]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if table_exists(cur, "user_accessible_documents"):
                cur.execute(
                    """
                    SELECT user_id, COUNT(*)::BIGINT * 100 AS visible_vectors
                    FROM user_accessible_documents
                    GROUP BY user_id
                    """
                )
            else:
                cur.execute(
                    """
                    SELECT ur.user_id, COUNT(DISTINCT pa.document_id)::BIGINT * 100 AS visible_vectors
                    FROM userroles ur
                    JOIN permissionassignment pa ON pa.role_id = ur.role_id
                    GROUP BY ur.user_id
                    """
                )
            return {int(user_id): int(count) for user_id, count in cur.fetchall()}
    finally:
        conn.close()


def fetch_permission_user_distribution(total_vectors: int) -> list[dict[str, object]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH user_roles_count AS (
                    SELECT user_id, COUNT(*)::BIGINT AS role_count
                    FROM userroles
                    GROUP BY user_id
                ),
                visible AS (
                    SELECT user_id, COUNT(*)::BIGINT AS visible_documents
                    FROM user_accessible_documents
                    GROUP BY user_id
                )
                SELECT u.user_id,
                       COALESCE(ur.role_count, 0)::BIGINT,
                       COALESCE(v.visible_documents, 0)::BIGINT
                FROM users u
                LEFT JOIN user_roles_count ur ON ur.user_id = u.user_id
                LEFT JOIN visible v ON v.user_id = u.user_id
                ORDER BY u.user_id
                """
            )
            rows = []
            for user_id, role_count, visible_documents in cur.fetchall():
                visible_vectors = int(visible_documents or 0) * 100
                rows.append(
                    {
                        "user_id": int(user_id),
                        "role_count": int(role_count or 0),
                        "visible_documents": int(visible_documents or 0),
                        "visible_vectors": visible_vectors,
                        "visible_selectivity": visible_vectors / total_vectors if total_vectors else 0.0,
                    }
                )
            return rows
    finally:
        conn.close()


def fetch_permission_role_distribution() -> list[dict[str, object]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                WITH role_docs AS (
                    SELECT role_id, COUNT(DISTINCT document_id)::BIGINT AS permitted_documents
                    FROM permissionassignment
                    GROUP BY role_id
                ),
                role_users AS (
                    SELECT role_id, COUNT(DISTINCT user_id)::BIGINT AS assigned_users
                    FROM userroles
                    GROUP BY role_id
                )
                SELECT r.role_id,
                       COALESCE(ru.assigned_users, 0)::BIGINT,
                       COALESCE(rd.permitted_documents, 0)::BIGINT
                FROM roles r
                LEFT JOIN role_users ru ON ru.role_id = r.role_id
                LEFT JOIN role_docs rd ON rd.role_id = r.role_id
                ORDER BY r.role_id
                """
            )
            return [
                {
                    "role_id": int(role_id),
                    "assigned_users": int(assigned_users or 0),
                    "permitted_documents": int(permitted_documents or 0),
                    "permitted_vectors": int(permitted_documents or 0) * 100,
                }
                for role_id, assigned_users, permitted_documents in cur.fetchall()
            ]
    finally:
        conn.close()


def fetch_users() -> list[int]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM users ORDER BY user_id")
            return [int(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def sample_queries(limit: int, topk: int) -> list[Query]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id
                FROM users
                ORDER BY user_id
                LIMIT %s
                """,
                [int(limit)],
            )
            users = [int(row[0]) for row in cur.fetchall()]
            cur.execute(
                """
                SELECT vector::text
                FROM documentblocks
                ORDER BY block_id
                LIMIT %s
                """,
                [len(users)],
            )
            vectors = [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()
    return [
        Query(user_id=user_id, vector=vectors[index], topk=int(topk), ground_truth=frozenset())
        for index, user_id in enumerate(users)
        if index < len(vectors)
    ]


def versioned_selection(method: str, memory_ratio: float) -> PlanSelection:
    return resolve_versioned_plan(method, memory_ratio)


def vector_counts_by_table(table_names: Iterable[str]) -> dict[str, int]:
    table_names = sorted({name for name in table_names if name})
    if not table_names:
        return {}
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.relname, COALESCE(s.n_live_tup, 0)::BIGINT
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
                WHERE n.nspname = 'public'
                  AND c.relname = ANY(%s)
                """,
                [table_names],
            )
            return {str(name): int(count) for name, count in cur.fetchall()}
    finally:
        conn.close()


def relation_summary_for_like(name: str, like_pattern: str, include_index: bool = True) -> dict[str, object]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*)::BIGINT,
                    COALESCE(SUM(COALESCE(s.n_live_tup, 0)), 0)::BIGINT,
                    COALESCE(SUM(pg_relation_size(c.oid)), 0)::BIGINT,
                    COALESCE(SUM(pg_indexes_size(c.oid)), 0)::BIGINT,
                    COALESCE(SUM(pg_total_relation_size(c.oid)), 0)::BIGINT
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
                WHERE n.nspname = 'public'
                  AND c.relkind = 'r'
                  AND c.relname LIKE %s
                """,
                [like_pattern],
            )
            row = cur.fetchone()
    finally:
        conn.close()
    table_count, rows, heap_bytes, index_bytes, total_bytes = [int(value or 0) for value in row]
    if not include_index:
        total_bytes = heap_bytes
    return {
        "method": name,
        "partition_tables": table_count,
        "estimated_materialized_vectors": rows,
        "heap_mb": heap_bytes / (1024 * 1024),
        "index_mb": index_bytes / (1024 * 1024),
        "total_mb": total_bytes / (1024 * 1024),
    }


def relation_summary_for_registry(method: str, selection: PlanSelection) -> dict[str, object]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(*) FILTER (WHERE br.relation_kind = 'partition')::BIGINT,
                    COALESCE(SUM(COALESCE(s.n_live_tup, 0)) FILTER (WHERE br.relation_kind = 'partition'), 0)::BIGINT,
                    COALESCE(SUM(pg_relation_size(c.oid)) FILTER (WHERE br.relation_kind = 'partition'), 0)::BIGINT,
                    COALESCE(SUM(pg_indexes_size(c.oid)) FILTER (WHERE br.relation_kind = 'partition'), 0)::BIGINT,
                    COALESCE(SUM(pg_total_relation_size(c.oid)) FILTER (WHERE br.relation_kind = 'partition'), 0)::BIGINT,
                    COUNT(*)::BIGINT,
                    COALESCE(SUM(pg_total_relation_size(c.oid)), 0)::BIGINT
                FROM benchmark_plan_relations br
                JOIN pg_class c ON c.relname = br.relation_name
                JOIN pg_namespace n ON n.oid = c.relnamespace
                LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
                WHERE br.registry_id = %s
                  AND n.nspname = 'public'
                """,
                [int(selection.registry_id or 0)],
            )
            row = cur.fetchone()
    finally:
        conn.close()
    part_count, rows, heap_bytes, index_bytes, part_total, rel_count, all_total = [int(value or 0) for value in row]
    return {
        "method": method,
        "partition_tables": part_count,
        "registered_relations": rel_count,
        "estimated_materialized_vectors": rows,
        "heap_mb": heap_bytes / (1024 * 1024),
        "index_mb": index_bytes / (1024 * 1024),
        "total_mb": part_total / (1024 * 1024),
        "registered_total_mb": all_total / (1024 * 1024),
        "table_prefix": selection.table_prefix or "",
    }


def route_stats_from_routes(
    method: str,
    routes_by_user: dict[int, tuple],
    visible_by_user: dict[int, int],
    all_users: list[int],
) -> list[UserStats]:
    table_counts = vector_counts_by_table(
        route.table_name
        for routes in routes_by_user.values()
        for route in routes
    )
    rows: list[UserStats] = []
    for user_id in all_users:
        routes = tuple(routes_by_user.get(int(user_id), ()))
        route_count = len(routes)
        route_vectors = 0
        accessible_vectors = int(visible_by_user.get(int(user_id), 0))
        pure_routes = 0
        mixed_routes = 0
        impurity_values = []
        for route in routes:
            partition_vectors = int(getattr(route, "partition_vectors", 0) or 0)
            if partition_vectors <= 0:
                partition_vectors = int(table_counts.get(str(route.table_name), 0))
            route_vectors += partition_vectors
            route_accessible = int(getattr(route, "accessible_vectors", 0) or 0)
            if route_accessible > 0 and partition_vectors > 0:
                impurity_values.append(partition_vectors / max(route_accessible, 1))
            elif float(getattr(route, "impurity_factor", 1.0) or 1.0) > 0:
                impurity_values.append(float(getattr(route, "impurity_factor", 1.0) or 1.0))
            if bool(getattr(route, "pure", False)):
                pure_routes += 1
            else:
                mixed_routes += 1
        selectivity = accessible_vectors / route_vectors if route_vectors > 0 else 0.0
        rows.append(
            UserStats(
                method=method,
                user_id=int(user_id),
                route_count=route_count,
                route_vector_count=route_vectors,
                accessible_vector_count=accessible_vectors,
                selectivity=selectivity,
                impurity_factor=statistics.mean(impurity_values) if impurity_values else 0.0,
                pure_route_count=pure_routes,
                mixed_route_count=mixed_routes,
            )
        )
    return rows


def route_stats_for_rls(visible_by_user: dict[int, int], all_users: list[int], total_vectors: int) -> list[UserStats]:
    rows = []
    for user_id in all_users:
        visible = int(visible_by_user.get(int(user_id), 0))
        rows.append(
            UserStats(
                method="RLS",
                user_id=int(user_id),
                route_count=1,
                route_vector_count=total_vectors,
                accessible_vector_count=visible,
                selectivity=visible / total_vectors if total_vectors else 0.0,
                impurity_factor=total_vectors / visible if visible else 0.0,
                pure_route_count=0,
                mixed_route_count=1,
            )
        )
    return rows


def hqi_user_stats(visible_by_user: dict[int, int], all_users: list[int]) -> list[UserStats]:
    """Compute HQI route count from persisted partition tables.

    The local QD-tree pickle can be stale or built for another dataset. The persisted
    partition tables are the authoritative source for this database, so we count a
    user route when a QD-tree partition contains at least one document visible to
    that user.
    """
    conn = get_db_connection()
    route_count_by_user: Counter[int] = Counter()
    route_vectors_by_user: Counter[int] = Counter()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.relname, COALESCE(s.n_live_tup, 0)::BIGINT
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
                WHERE n.nspname = 'public'
                  AND c.relkind = 'r'
                  AND c.relname LIKE 'documentblocks_qdtree_partition_%'
                ORDER BY c.relname
                """
            )
            partitions = [(str(name), int(count or 0)) for name, count in cur.fetchall()]
            for table_name, table_vectors in partitions:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT DISTINCT u.user_id
                        FROM (SELECT DISTINCT document_id FROM {}) d
                        JOIN user_accessible_documents u
                          ON u.document_id = d.document_id
                        """
                    ).format(sql.Identifier(table_name))
                )
                for (user_id,) in cur.fetchall():
                    user_id = int(user_id)
                    route_count_by_user[user_id] += 1
                    route_vectors_by_user[user_id] += int(table_vectors)
    finally:
        conn.close()

    rows = []
    for user_id in all_users:
        route_count = int(route_count_by_user.get(int(user_id), 0))
        route_vectors = int(route_vectors_by_user.get(int(user_id), 0))
        visible = int(visible_by_user.get(int(user_id), 0))
        rows.append(
            UserStats(
                method="HQI",
                user_id=int(user_id),
                route_count=route_count,
                route_vector_count=route_vectors,
                accessible_vector_count=visible,
                selectivity=visible / route_vectors if route_vectors else 0.0,
                impurity_factor=route_vectors / visible if visible else 0.0,
                pure_route_count=0,
                mixed_route_count=route_count,
            )
        )
    return rows


def summarize_user_stats(rows: list[UserStats]) -> list[dict[str, object]]:
    output = []
    by_method: dict[str, list[UserStats]] = defaultdict(list)
    for row in rows:
        by_method[row.method].append(row)
    for method in sorted(by_method, key=lambda name: ["RLS", "ROLE", "HQI", "HONEYBEE", "VEDA", "EFFVEDA", "SQUID"].index(name)):
        values = by_method[method]
        route_counts = [float(item.route_count) for item in values]
        route_vectors = [float(item.route_vector_count) for item in values]
        visible_vectors = [float(item.accessible_vector_count) for item in values]
        selectivities = [float(item.selectivity) for item in values]
        impurity = [float(item.impurity_factor) for item in values if item.impurity_factor > 0]
        output.append(
            {
                "method": method,
                "users_or_queries": len(values),
                "avg_route_partitions": statistics.mean(route_counts) if route_counts else 0.0,
                "p50_route_partitions": percentile(route_counts, 0.50),
                "p90_route_partitions": percentile(route_counts, 0.90),
                "max_route_partitions": max(route_counts) if route_counts else 0.0,
                "avg_route_vectors": statistics.mean(route_vectors) if route_vectors else 0.0,
                "p50_route_vectors": percentile(route_vectors, 0.50),
                "avg_accessible_vectors": statistics.mean(visible_vectors) if visible_vectors else 0.0,
                "p50_accessible_vectors": percentile(visible_vectors, 0.50),
                "avg_route_selectivity": statistics.mean(selectivities) if selectivities else 0.0,
                "p50_route_selectivity": percentile(selectivities, 0.50),
                "avg_impurity_factor": statistics.mean(impurity) if impurity else 0.0,
                "p50_impurity_factor": percentile(impurity, 0.50),
                "avg_pure_routes": statistics.mean([float(item.pure_route_count) for item in values]) if values else 0.0,
                "avg_mixed_routes": statistics.mean([float(item.mixed_route_count) for item in values]) if values else 0.0,
            }
        )
    return output


def partition_distribution_for_registry(method: str, selection: PlanSelection) -> list[dict[str, object]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT br.relation_name,
                       COALESCE(s.n_live_tup, 0)::BIGINT,
                       pg_total_relation_size(c.oid)::BIGINT
                FROM benchmark_plan_relations br
                JOIN pg_class c ON c.relname = br.relation_name
                JOIN pg_namespace n ON n.oid = c.relnamespace
                LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
                WHERE br.registry_id = %s
                  AND n.nspname = 'public'
                  AND br.relation_kind = 'partition'
                ORDER BY br.relation_name
                """,
                [int(selection.registry_id or 0)],
            )
            return [
                {
                    "method": method,
                    "table_name": str(table),
                    "estimated_vectors": int(rows or 0),
                    "total_mb": int(bytes_ or 0) / (1024 * 1024),
                }
                for table, rows, bytes_ in cur.fetchall()
            ]
    finally:
        conn.close()


def partition_distribution_for_like(method: str, like_pattern: str) -> list[dict[str, object]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.relname,
                       COALESCE(s.n_live_tup, 0)::BIGINT,
                       pg_total_relation_size(c.oid)::BIGINT
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
                WHERE n.nspname = 'public'
                  AND c.relkind = 'r'
                  AND c.relname LIKE %s
                ORDER BY c.relname
                """,
                [like_pattern],
            )
            return [
                {
                    "method": method,
                    "table_name": str(table),
                    "estimated_vectors": int(rows or 0),
                    "total_mb": int(bytes_ or 0) / (1024 * 1024),
                }
                for table, rows, bytes_ in cur.fetchall()
            ]
    finally:
        conn.close()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect partition case-study stats.")
    parser.add_argument("--memory-ratio", type=float, default=1.5)
    parser.add_argument("--query-sample", type=int, default=1000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "basic_benchmark/result/direct_pg_qps")
    parser.add_argument("--prefix", default="case_study_treebase_1p5")
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    visible_by_user = fetch_user_visible_counts()
    all_users = fetch_users()
    total_vectors = sum(1 for _ in range(1))
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            total_vectors = fetch_scalar(cur, "SELECT COUNT(*) FROM documentblocks")
    finally:
        conn.close()

    selections = {
        "SQUID": versioned_selection("ours", args.memory_ratio),
        "VEDA": versioned_selection("veda", args.memory_ratio),
        "EFFVEDA": versioned_selection("effveda", args.memory_ratio),
        "HONEYBEE": versioned_selection("honeybee", args.memory_ratio),
    }

    route_rows: list[UserStats] = []
    route_rows.extend(route_stats_for_rls(visible_by_user, all_users, total_vectors))
    route_rows.extend(route_stats_from_routes("ROLE", load_role_routes(), visible_by_user, all_users))
    route_rows.extend(route_stats_from_routes("HONEYBEE", load_honeybee_routes(selections["HONEYBEE"]), visible_by_user, all_users))
    route_rows.extend(route_stats_from_routes("SQUID", load_ours_routes(selections["SQUID"]), visible_by_user, all_users))
    route_rows.extend(route_stats_from_routes("VEDA", load_veda_routes("veda", selections["VEDA"]), visible_by_user, all_users))
    route_rows.extend(route_stats_from_routes("EFFVEDA", load_veda_routes("effveda", selections["EFFVEDA"]), visible_by_user, all_users))
    route_rows.extend(hqi_user_stats(visible_by_user, all_users))

    relation_rows: list[dict[str, object]] = []
    relation_rows.append(relation_summary_for_like("RLS", "documentblocks"))
    relation_rows.append(relation_summary_for_like("ROLE", "documentblocks_role_%"))
    relation_rows.append(relation_summary_for_like("HQI", "documentblocks_qdtree_partition_%"))
    for method, selection in selections.items():
        relation_rows.append(relation_summary_for_registry(method, selection))
    for row in relation_rows:
        row["memory_ratio"] = args.memory_ratio if row["method"] not in {"RLS", "ROLE", "HQI"} else ""
        row["replication_ratio_est"] = (
            float(row.get("estimated_materialized_vectors") or 0) / total_vectors
            if total_vectors
            else 0.0
        )

    partition_rows: list[dict[str, object]] = []
    partition_rows.extend(partition_distribution_for_like("RLS", "documentblocks"))
    partition_rows.extend(partition_distribution_for_like("ROLE", "documentblocks_role_%"))
    partition_rows.extend(partition_distribution_for_like("HQI", "documentblocks_qdtree_partition_%"))
    for method, selection in selections.items():
        partition_rows.extend(partition_distribution_for_registry(method, selection))

    partition_sizes_by_method: dict[str, list[float]] = defaultdict(list)
    for row in partition_rows:
        partition_sizes_by_method[str(row["method"])].append(float(row.get("estimated_vectors") or 0))

    user_rows = [
        {
            "method": row.method,
            "user_id": row.user_id,
            "route_count": row.route_count,
            "route_vector_count": row.route_vector_count,
            "accessible_vector_count": row.accessible_vector_count,
            "route_selectivity": row.selectivity,
            "impurity_factor": row.impurity_factor,
            "pure_route_count": row.pure_route_count,
            "mixed_route_count": row.mixed_route_count,
        }
        for row in route_rows
    ]
    summary_rows = summarize_user_stats(route_rows)
    relation_by_method = {row["method"]: row for row in relation_rows}
    for row in summary_rows:
        relation = relation_by_method.get(str(row["method"]), {})
        partition_sizes = partition_sizes_by_method.get(str(row["method"]), [])
        row.update(
            {
                "partition_tables": relation.get("partition_tables", ""),
                "estimated_materialized_vectors": relation.get("estimated_materialized_vectors", ""),
                "replication_ratio_est": relation.get("replication_ratio_est", ""),
                "total_mb": relation.get("total_mb", ""),
                "registered_total_mb": relation.get("registered_total_mb", relation.get("total_mb", "")),
                "avg_partition_vectors": statistics.mean(partition_sizes) if partition_sizes else 0.0,
                "p50_partition_vectors": percentile(partition_sizes, 0.50),
                "p90_partition_vectors": percentile(partition_sizes, 0.90),
                "max_partition_vectors": max(partition_sizes) if partition_sizes else 0.0,
            }
        )

    visible_counts = list(visible_by_user.values())
    permission_user_rows = fetch_permission_user_distribution(total_vectors)
    permission_role_rows = fetch_permission_role_distribution()
    permission_rows = [
        {
            "metric": "users",
            "value": len(all_users),
        },
        {
            "metric": "avg_visible_vectors",
            "value": statistics.mean(visible_counts) if visible_counts else 0.0,
        },
        {
            "metric": "p50_visible_vectors",
            "value": percentile([float(v) for v in visible_counts], 0.50),
        },
        {
            "metric": "p90_visible_vectors",
            "value": percentile([float(v) for v in visible_counts], 0.90),
        },
        {
            "metric": "max_visible_vectors",
            "value": max(visible_counts) if visible_counts else 0,
        },
        {
            "metric": "avg_visible_selectivity",
            "value": statistics.mean([v / total_vectors for v in visible_counts]) if visible_counts and total_vectors else 0.0,
        },
        {
            "metric": "avg_roles_per_user",
            "value": statistics.mean([float(row["role_count"]) for row in permission_user_rows]) if permission_user_rows else 0.0,
        },
        {
            "metric": "p90_roles_per_user",
            "value": percentile([float(row["role_count"]) for row in permission_user_rows], 0.90),
        },
        {
            "metric": "avg_documents_per_role",
            "value": statistics.mean([float(row["permitted_documents"]) for row in permission_role_rows]) if permission_role_rows else 0.0,
        },
        {
            "metric": "p90_documents_per_role",
            "value": percentile([float(row["permitted_documents"]) for row in permission_role_rows], 0.90),
        },
        {
            "metric": "avg_users_per_role",
            "value": statistics.mean([float(row["assigned_users"]) for row in permission_role_rows]) if permission_role_rows else 0.0,
        },
    ]
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for table in ("documents", "documentblocks", "users", "roles", "userroles", "permissionassignment", "user_accessible_documents"):
                if table_exists(cur, table):
                    permission_rows.append({"metric": f"{table}_rows", "value": fetch_scalar(cur, f"SELECT COUNT(*) FROM {table}")})
    finally:
        conn.close()

    base = output_dir / args.prefix
    write_csv(base.with_name(base.name + "_summary.csv"), summary_rows)
    write_csv(base.with_name(base.name + "_relations.csv"), relation_rows)
    write_csv(base.with_name(base.name + "_user_routes.csv"), user_rows)
    write_csv(base.with_name(base.name + "_partitions.csv"), partition_rows)
    write_csv(base.with_name(base.name + "_permissions.csv"), permission_rows)
    write_csv(base.with_name(base.name + "_permission_users.csv"), permission_user_rows)
    write_csv(base.with_name(base.name + "_permission_roles.csv"), permission_role_rows)

    print(f"summary={base.with_name(base.name + '_summary.csv')}")
    print(f"relations={base.with_name(base.name + '_relations.csv')}")
    print(f"user_routes={base.with_name(base.name + '_user_routes.csv')}")
    print(f"partitions={base.with_name(base.name + '_partitions.csv')}")
    print(f"permissions={base.with_name(base.name + '_permissions.csv')}")
    print(f"permission_users={base.with_name(base.name + '_permission_users.csv')}")
    print(f"permission_roles={base.with_name(base.name + '_permission_roles.csv')}")
    for row in summary_rows:
        print(
            f"{row['method']}: partitions={row.get('partition_tables')} "
            f"avg_routes={float(row['avg_route_partitions']):.3f} "
            f"avg_selectivity={float(row['avg_route_selectivity']):.4f} "
            f"rep={float(row.get('replication_ratio_est') or 0):.3f}"
        )


if __name__ == "__main__":
    main()
