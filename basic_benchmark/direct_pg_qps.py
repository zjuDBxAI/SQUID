#!/usr/bin/env python3
"""Direct PostgreSQL QPS benchmark for materialized vector-search methods.

This harness intentionally bypasses all method-level ``search.py`` functions.
It preloads route metadata once, then times only real ANN SELECT statements,
result fetching, and the common top-k merge. Recall uses the cache afterwards
and never runs in the timed path.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import statistics
import sys
import threading
import time
from contextlib import nullcontext
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Callable

from psycopg2 import sql


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.config import get_db_connection  # noqa: E402
from controller.baseline.HQI.qd_tree import (  # noqa: E402
    DEFAULT_QD_TREE_PARTITION_PREFIX as HQI_DEFAULT_PARTITION_PREFIX,
    _collect_partition_document_ids_for_user as hqi_collect_partition_document_ids_for_user,
    _collect_relevant_partitions as hqi_collect_relevant_partitions,
    _prepare_query_vector as hqi_prepare_query_vector,
    gather_role_accessible_partitions as hqi_gather_role_accessible_partitions,
    get_qd_tree_root,
    partition_has_accessible_documents as hqi_partition_has_accessible_documents,
)

HNSW_ITERATIVE_SCAN_VALUES = ("off", "relaxed_order", "strict_order")


@dataclass(frozen=True)
class Route:
    table_name: str
    pattern_ids: tuple[int, ...] = ()
    pure: bool = False
    route_kind: str = "index"
    impurity_factor: float = 1.0
    partition_vectors: int = 0
    accessible_vectors: int = 0
    cluster_id: int = 0
    partition_id: str = ""
    doc_ids: tuple[int, ...] = ()


@dataclass(frozen=True)
class Query:
    user_id: int
    vector: str
    topk: int
    ground_truth: frozenset[tuple[int, int]]


@dataclass(frozen=True)
class PlanSelection:
    method: str
    memory_ratio: float | None = None
    registry_id: int | None = None
    plan_id: int | None = None
    table_prefix: str | None = None
    metadata: dict | None = None


@dataclass(frozen=True)
class WorkerResult:
    values: list[tuple[float, list[tuple], int, Query]]
    timed_elapsed: float
    query_count: int


class PreparedRouteCache:
    def __init__(self) -> None:
        self._prepared: set[tuple[str, bool, tuple[tuple[str, str], ...]]] = set()

    @staticmethod
    def _statement_name(route: Route, use_sql_filter: bool, settings: tuple[tuple[str, str], ...]) -> str:
        suffix = abs(hash((route.table_name, use_sql_filter, settings)))
        return f"direct_pg_qps_{suffix}"

    def execute(
        self,
        cur,
        route: Route,
        *,
        query_vector: str,
        topk: int,
        use_sql_filter: bool,
        settings: tuple[tuple[str, str], ...] = (),
    ) -> list[tuple]:
        key = (route.table_name, bool(use_sql_filter), tuple(settings))
        statement_name = self._statement_name(route, bool(use_sql_filter), tuple(settings))
        if key not in self._prepared:
            prefix = _settings_prefix(settings) if settings else sql.SQL("")
            if use_sql_filter:
                statement = prefix + sql.SQL(
                    "PREPARE {}(vector, integer[], integer) AS "
                    "SELECT block_id, document_id, vector <-> $1 AS distance "
                    "FROM {} WHERE pattern_id = ANY($2) "
                    "ORDER BY vector <-> $1 LIMIT $3"
                ).format(sql.Identifier(statement_name), sql.Identifier(route.table_name))
            else:
                statement = prefix + sql.SQL(
                    "PREPARE {}(vector, integer) AS "
                    "SELECT block_id, document_id, vector <-> $1 AS distance "
                    "FROM {} ORDER BY vector <-> $1 LIMIT $2"
                ).format(sql.Identifier(statement_name), sql.Identifier(route.table_name))
            cur.execute(statement)
            self._prepared.add(key)
        prefix = _settings_prefix(settings) if settings else sql.SQL("")
        if use_sql_filter:
            cur.execute(
                prefix + sql.SQL("EXECUTE {}(%s::vector, %s::integer[], %s)").format(sql.Identifier(statement_name)),
                [query_vector, list(route.pattern_ids), int(topk)],
            )
        else:
            cur.execute(
                prefix + sql.SQL("EXECUTE {}(%s::vector, %s)").format(sql.Identifier(statement_name)),
                [query_vector, int(topk)],
            )
        return _fetch_ann_rows(cur)


def merge_topk(rows: list[tuple], topk: int) -> list[tuple]:
    seen: set[tuple[int, int]] = set()
    merged: list[tuple] = []
    for row in sorted(rows, key=lambda value: float(value[3])):
        key = (int(row[0]), int(row[1]))
        if key in seen:
            continue
        seen.add(key)
        merged.append(row)
        if len(merged) == topk:
            break
    return merged


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[int(round((len(ordered) - 1) * fraction))]


def round_robin_batches(values: list[Query], count: int) -> list[list[Query]]:
    return [values[index::count] for index in range(min(count, len(values)))]


def table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s) IS NOT NULL", [f"public.{table_name}"])
    return bool(cur.fetchone()[0])


def method_sql_batching(method: str, args: argparse.Namespace) -> bool:
    if args.sql_batching is not None:
        return bool(args.sql_batching)
    return _normalize_method_name(method) != "hqi"


def _normalize_method_name(method: str) -> str:
    normalized = str(method).strip().lower()
    if normalized in {"squid", "ours"}:
        return "ours"
    if normalized in {"honeybee", "anonysys", "dynamic_partition"}:
        return "honeybee"
    if normalized in {"hqi", "qdtree", "qd_tree", "qdtree_partition"}:
        return "hqi"
    return normalized


def resolve_versioned_plan(method: str, memory_ratio: float | None) -> PlanSelection:
    normalized = _normalize_method_name(method)
    if memory_ratio is None or normalized in {"rls", "role", "hqi"}:
        return PlanSelection(method=normalized)
    registry_method = "honeybee" if normalized == "honeybee" else normalized
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if not table_exists(cur, "benchmark_plan_registry"):
                raise RuntimeError(
                    "Versioned plan registry is not initialized in this database. "
                    "Build the requested plan first with "
                    "basic_benchmark/script/squid/build_versioned_plan.py, or run "
                    "basic_benchmark/script/squid/versioned_plan_registry.py init."
                )
            cur.execute(
                """
                SELECT registry_id, plan_id, table_prefix, metadata
                FROM benchmark_plan_registry
                WHERE method = %s
                  AND memory_ratio = %s
                  AND state = 'ready'
                ORDER BY registry_id DESC
                LIMIT 1
                """,
                [registry_method, float(memory_ratio)],
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"No ready versioned plan for method={registry_method} memory_ratio={memory_ratio}")
            return PlanSelection(
                method=normalized,
                memory_ratio=float(memory_ratio),
                registry_id=int(row[0]),
                plan_id=int(row[1]),
                table_prefix=str(row[2]),
                metadata=dict(row[3] or {}),
            )
    finally:
        conn.close()


def _versioned_relation(selection: PlanSelection, relation_kind: str) -> str | None:
    if selection.registry_id is None:
        return None
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT relation_name
                FROM benchmark_plan_relations
                WHERE registry_id = %s
                  AND relation_kind = %s
                ORDER BY relation_name
                LIMIT 1
                """,
                [int(selection.registry_id), relation_kind],
            )
            row = cur.fetchone()
            return str(row[0]) if row else None
    finally:
        conn.close()


def _versioned_relations(selection: PlanSelection, relation_kind: str) -> list[str]:
    if selection.registry_id is None:
        return []
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT relation_name
                FROM benchmark_plan_relations
                WHERE registry_id = %s
                  AND relation_kind = %s
                ORDER BY relation_name
                """,
                [int(selection.registry_id), relation_kind],
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def _partition_relation_by_suffix(selection: PlanSelection) -> dict[int, str]:
    prefix = str(selection.table_prefix or "")
    mapping: dict[int, str] = {}
    if not prefix:
        return mapping
    marker = f"{prefix}_partition_"
    for relation in _versioned_relations(selection, "partition"):
        if relation.startswith(marker):
            suffix = relation[len(marker):]
            if suffix.isdigit():
                mapping[int(suffix)] = relation
    return mapping


def _route_has_visible_vectors(route: Route) -> bool:
    if route.partition_vectors <= 0:
        return False
    if route.pure:
        return route.partition_vectors > 0
    return route.accessible_vectors > 0 and bool(route.pattern_ids)


def _filter_valid_routes(routes: dict[int, list[Route]]) -> dict[int, tuple[Route, ...]]:
    return {
        user_id: tuple(route for route in user_routes if _route_has_visible_vectors(route))
        for user_id, user_routes in routes.items()
    }


def load_queries(query_path: Path, cache_path: Path, limit: int) -> list[Query]:
    query_rows = json.loads(query_path.read_text())
    cache_rows = json.loads(cache_path.read_text())
    if len(cache_rows) < len(query_rows):
        raise ValueError(f"Ground-truth cache has {len(cache_rows)} entries for {len(query_rows)} queries")

    queries: list[Query] = []
    for index, raw in enumerate(query_rows[:limit]):
        cached = cache_rows[index]
        if isinstance(cached, dict):
            expected = cached.get("query", {})
            if expected and (int(expected.get("user_id", -1)) != int(raw["user_id"]) or expected.get("query_vector") != raw["query_vector"]):
                raise ValueError(f"Ground-truth cache mismatch at query {index}")
            cached = cached.get("ground_truth", [])
        ground_truth = frozenset(
            (int(row[0]), int(row[1]))
            for row in cached
            if isinstance(row, (list, tuple)) and len(row) >= 2
        )
        queries.append(Query(int(raw["user_id"]), str(raw["query_vector"]), int(raw.get("topk", 10)), ground_truth))
    return queries


def load_db_sampled_queries(limit: int, topk: int) -> list[Query]:
    """Sample valid query vectors directly from the connected database.

    This mode is intended for QPS-only runs on databases whose vector dimension
    differs from the repository's default query JSON. Recall is intentionally
    disabled by leaving ground_truth empty.
    """
    query_count = max(1, int(limit))
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id
                FROM UserRoles
                GROUP BY user_id
                ORDER BY random()
                LIMIT %s
                """,
                [query_count],
            )
            user_ids = [int(row[0]) for row in cur.fetchall()]
            cur.execute(
                """
                SELECT vector::text
                FROM documentblocks
                ORDER BY random()
                LIMIT %s
                """,
                [query_count],
            )
            vectors = [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()

    if not user_ids:
        raise RuntimeError("No users found in UserRoles for database-sampled QPS queries")
    if not vectors:
        raise RuntimeError("No vectors found in documentblocks for database-sampled QPS queries")

    queries: list[Query] = []
    for index in range(min(query_count, len(user_ids), len(vectors))):
        queries.append(Query(user_ids[index], vectors[index], int(topk), frozenset()))
    return queries


def load_ours_routes(selection: PlanSelection | None = None) -> dict[int, tuple[Route, ...]]:
    selection = selection or PlanSelection(method="ours")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            plan_table = _versioned_relation(selection, "plan_metadata") or "kmeans_current_plan"
            partition_table = _versioned_relation(selection, "partition_metadata") or "kmeans_current_partitions"
            pattern_table = _versioned_relation(selection, "pattern_metadata") or "kmeans_current_patterns"
            route_table = _versioned_relation(selection, "route_metadata") or "kmeans_current_routes"
            if not table_exists(cur, plan_table) or not table_exists(cur, route_table):
                raise RuntimeError("OURS metadata is not materialized")
            if selection.plan_id is None:
                cur.execute(sql.SQL("SELECT plan_id FROM {} ORDER BY plan_id DESC LIMIT 1").format(sql.Identifier(plan_table)))
                plan = cur.fetchone()
                plan_id = int(plan[0]) if plan else None
            else:
                plan_id = int(selection.plan_id)
            if plan_id is None:
                raise RuntimeError("OURS has no active plan")
            cur.execute(
                sql.SQL(
                    """
                SELECT
                    r.tenant_id,
                    r.table_name,
                    r.pattern_ids,
                    p.vector_count,
                    COALESCE(SUM(ap.vector_count), 0)::BIGINT AS visible_vectors,
                    r.cluster_id,
                    r.partition_id
                FROM {} r
                JOIN {} p
                  ON p.plan_id = r.plan_id
                 AND p.partition_id = r.partition_id
                LEFT JOIN {} ap
                  ON ap.plan_id = r.plan_id
                 AND ap.pattern_id = ANY(r.pattern_ids)
                WHERE r.plan_id = %s
                GROUP BY r.tenant_id, r.table_name, r.pattern_ids, p.vector_count,
                         r.route_kind, r.cluster_id, r.partition_id
                ORDER BY r.tenant_id, r.route_kind, r.cluster_id, r.partition_id
                """,
                ).format(
                    sql.Identifier(route_table),
                    sql.Identifier(partition_table),
                    sql.Identifier(pattern_table),
                ),
                [int(plan_id)],
            )
            loaded: dict[int, list[Route]] = {}
            for tenant_id, table_name, pattern_ids, partition_vectors, visible_vectors, cluster_id, partition_id in cur.fetchall():
                partition_count = int(partition_vectors or 0)
                visible_count = int(visible_vectors or 0)
                loaded.setdefault(int(tenant_id), []).append(
                    Route(
                        str(table_name),
                        tuple(int(value) for value in (pattern_ids or ())),
                        partition_count > 0 and visible_count >= partition_count,
                        partition_vectors=partition_count,
                        accessible_vectors=visible_count,
                        cluster_id=int(cluster_id or 0),
                        partition_id=str(partition_id),
                    )
                )
            return _filter_valid_routes(loaded)
    finally:
        conn.close()


def load_veda_routes(algorithm: str, selection: PlanSelection | None = None) -> dict[int, tuple[Route, ...]]:
    selection = selection or PlanSelection(method=algorithm)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            plan_table = _versioned_relation(selection, "plan_metadata") or "veda_current_plan"
            route_table = _versioned_relation(selection, "route_metadata") or "veda_current_user_routes"
            if not table_exists(cur, plan_table) or not table_exists(cur, route_table):
                raise RuntimeError(f"{algorithm} metadata is not materialized")
            if selection.plan_id is None:
                cur.execute(
                    sql.SQL("SELECT plan_id FROM {} WHERE algorithm = %s ORDER BY plan_id DESC LIMIT 1").format(
                        sql.Identifier(plan_table)
                    ),
                    [algorithm],
                )
                plan = cur.fetchone()
                plan_id = int(plan[0]) if plan else None
            else:
                plan_id = int(selection.plan_id)
            if plan_id is None:
                raise RuntimeError(f"No active {algorithm} plan")
            cur.execute(
                sql.SQL(
                    """
                SELECT user_id, table_name, route_kind, pattern_ids, impurity_factor,
                       node_vector_count, accessible_vector_count, node_id
                FROM {}
                WHERE plan_id = %s
                ORDER BY user_id, route_kind, node_id
                """,
                ).format(sql.Identifier(route_table)),
                [int(plan_id)],
            )
            loaded: dict[int, list[Route]] = {}
            for user_id, table_name, route_kind, pattern_ids, impurity_factor, node_vectors, accessible_vectors, node_id in cur.fetchall():
                route_kind = str(route_kind)
                loaded.setdefault(int(user_id), []).append(
                    Route(
                        str(table_name),
                        tuple(int(value) for value in (pattern_ids or ())),
                        route_kind == "index",
                        route_kind,
                        float(impurity_factor or 1.0),
                        partition_vectors=int(node_vectors or 0),
                        accessible_vectors=int(accessible_vectors or 0),
                        partition_id=str(node_id),
                    )
                )
            return {user_id: tuple(routes) for user_id, routes in loaded.items()}
    finally:
        conn.close()


def load_role_routes() -> dict[int, tuple[Route, ...]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if not table_exists(cur, "userroles"):
                raise RuntimeError("UserRoles is unavailable")
            cur.execute("SELECT user_id, role_id FROM userroles ORDER BY user_id, role_id")
            loaded: dict[int, list[Route]] = {}
            for user_id, role_id in cur.fetchall():
                loaded.setdefault(int(user_id), []).append(Route(f"documentblocks_role_{int(role_id)}", (), True))
            return {user_id: tuple(routes) for user_id, routes in loaded.items()}
    finally:
        conn.close()


def load_honeybee_routes(selection: PlanSelection | None = None) -> dict[int, tuple[Route, ...]]:
    selection = selection or PlanSelection(method="honeybee")
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            mapping_table = (
                str((selection.metadata or {}).get("mapping_table") or "")
                or _versioned_relation(selection, "mapping_metadata")
                or "combrolepartitions"
            )
            table_prefix = str(selection.table_prefix or "")
            partition_relation_map = _partition_relation_by_suffix(selection)
            if not table_exists(cur, mapping_table):
                raise RuntimeError(f"Honeybee mapping table {mapping_table} is unavailable")
            cur.execute("SELECT user_id, array_agg(DISTINCT role_id ORDER BY role_id) FROM userroles GROUP BY user_id")
            role_sets = cur.fetchall()
            loaded: dict[int, tuple[Route, ...]] = {}
            for user_id, roles in role_sets:
                cur.execute(
                    sql.SQL("SELECT partition_id FROM {} WHERE comb_role = %s::integer[]").format(
                        sql.Identifier(mapping_table)
                    ),
                    [list(roles)],
                )
                loaded[int(user_id)] = tuple(
                    Route(
                        partition_relation_map.get(int(row[0]))
                        or (f"{table_prefix}_partition_{int(row[0])}" if table_prefix else f"documentblocks_partition_{int(row[0])}"),
                        (),
                        True,
                    )
                    for row in cur.fetchall()
                )
            return loaded
    finally:
        conn.close()



def _load_user_roles_for_queries(queries: list[Query]) -> dict[int, set[str]]:
    user_ids = sorted({int(query.user_id) for query in queries})
    if not user_ids:
        return {}
    roles_by_user: dict[int, set[str]] = {user_id: set() for user_id in user_ids}
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, role_id FROM UserRoles WHERE user_id = ANY(%s) ORDER BY user_id, role_id",
                [user_ids],
            )
            for user_id, role_id in cur.fetchall():
                roles_by_user.setdefault(int(user_id), set()).add(str(role_id))
    finally:
        conn.close()
    return roles_by_user


def load_hqi_routes(
    selection: PlanSelection | None = None,
    queries: list[Query] | None = None,
) -> dict[Query, tuple[Route, ...]]:
    if queries is None:
        raise RuntimeError("HQI direct QPS requires the query workload to precompute per-query routes")
    selection = selection or PlanSelection(method="hqi")
    metadata = dict(selection.metadata or {})
    tree_path = metadata.get("tree_path")
    partition_prefix = str(metadata.get("partition_prefix") or HQI_DEFAULT_PARTITION_PREFIX)
    root = get_qd_tree_root(tree_path=str(tree_path) if tree_path else None, partition_prefix=partition_prefix)
    roles_by_user = _load_user_roles_for_queries(queries)
    loaded: dict[Query, tuple[Route, ...]] = {}
    for query in queries:
        user_roles = roles_by_user.get(int(query.user_id), set())
        if not user_roles:
            loaded[query] = ()
            continue
        query_vector, _query_param = hqi_prepare_query_vector(query.vector)
        centroid_partitions = hqi_collect_relevant_partitions(root, user_roles, query_vector)
        selected_by_table = {}
        for partition in centroid_partitions:
            if hqi_partition_has_accessible_documents(partition, user_roles):
                table_name = partition.table_name or f"{partition_prefix}_{partition.partition_id}"
                selected_by_table[str(table_name)] = partition
        for partition in hqi_gather_role_accessible_partitions(root, user_roles):
            table_name = partition.table_name or f"{partition_prefix}_{partition.partition_id}"
            selected_by_table.setdefault(str(table_name), partition)
        routes: list[Route] = []
        for table_name, partition in selected_by_table.items():
            doc_ids = tuple(sorted(hqi_collect_partition_document_ids_for_user(partition, user_roles)))
            if doc_ids:
                routes.append(Route(table_name=str(table_name), pure=False, doc_ids=doc_ids))
        loaded[query] = tuple(routes)
    return loaded

def configure_session(
    cur,
    ef_search: int,
    jit: str,
    parallel_workers: int,
    hnsw_iterative_scan: str,
    *,
    force_hnsw_planner: bool,
    pg_parallel_route_scan: bool,
) -> None:
    if hnsw_iterative_scan not in HNSW_ITERATIVE_SCAN_VALUES:
        raise ValueError(f"invalid hnsw.iterative_scan value: {hnsw_iterative_scan}")
    cur.execute("LOAD 'vector'")
    cur.execute(f"SET jit = {jit}")
    cur.execute(f"SET max_parallel_workers_per_gather = {int(parallel_workers)}")
    cur.execute(f"SET hnsw.iterative_scan = {hnsw_iterative_scan}")
    if force_hnsw_planner:
        cur.execute("SET enable_seqscan = off")
    else:
        cur.execute("RESET enable_seqscan")
        cur.execute("RESET enable_bitmapscan")
    if pg_parallel_route_scan:
        cur.execute("SET enable_parallel_append = on")
        cur.execute("SET parallel_setup_cost = 0")
        cur.execute("SET parallel_tuple_cost = 0")
        cur.execute("SET min_parallel_table_scan_size = 0")
        cur.execute("SET min_parallel_index_scan_size = 0")
    cur.execute(f"SET hnsw.ef_search = {int(ef_search)}")


def _settings_prefix(settings: tuple[tuple[str, str], ...]) -> sql.SQL:
    statements = [
        sql.SQL("SET {} = {}; ").format(sql.Identifier(name), sql.Literal(value))
        for name, value in settings
    ]
    return sql.SQL("").join(statements)


def _execute_ann_route(
    cur,
    route: Route,
    *,
    query_vector: str,
    topk: int,
    use_sql_filter: bool,
    settings: tuple[tuple[str, str], ...] = (),
) -> list[tuple]:
    """Execute one route; optional GUCs are evaluated before its index scan."""
    prefix = _settings_prefix(settings) if settings else sql.SQL("")
    if use_sql_filter:
        statement = prefix + sql.SQL(
            "SELECT block_id, document_id, vector <-> %s::vector AS distance "
            "FROM {} WHERE pattern_id = ANY(%s) "
            "ORDER BY vector <-> %s::vector LIMIT %s"
        ).format(sql.Identifier(route.table_name))
        params = [query_vector, list(route.pattern_ids), query_vector, topk]
    else:
        statement = prefix + sql.SQL(
            "SELECT block_id, document_id, vector <-> %s::vector AS distance "
            "FROM {} ORDER BY vector <-> %s::vector LIMIT %s"
        ).format(sql.Identifier(route.table_name))
        params = [query_vector, query_vector, topk]
    cur.execute(statement, params)
    return [
        (int(block_id), int(document_id), None, float(distance))
        for block_id, document_id, distance in cur.fetchall()
    ]


def _fetch_ann_rows(cur) -> list[tuple]:
    return [
        (int(block_id), int(document_id), None, float(distance))
        for block_id, document_id, distance in cur.fetchall()
    ]


def _execute_ann_route_client_filter(
    cur,
    route: Route,
    *,
    query_vector: str,
    topk: int,
    candidate_limit: int,
    settings: tuple[tuple[str, str], ...] = (),
) -> list[tuple]:
    prefix = _settings_prefix(settings) if settings else sql.SQL("")
    statement = prefix + sql.SQL(
        "SELECT block_id, document_id, pattern_id, vector <-> %s::vector AS distance "
        "FROM {} ORDER BY vector <-> %s::vector LIMIT %s"
    ).format(sql.Identifier(route.table_name))
    cur.execute(statement, [query_vector, query_vector, max(1, int(candidate_limit))])
    allowed_patterns = set(int(pattern_id) for pattern_id in route.pattern_ids)
    filtered: list[tuple] = []
    for block_id, document_id, pattern_id, distance in cur.fetchall():
        if int(pattern_id) not in allowed_patterns:
            continue
        filtered.append((int(block_id), int(document_id), None, float(distance)))
        if len(filtered) >= int(topk):
            break
    return filtered


def _execute_ann_route_docid_client_filter(
    cur,
    route: Route,
    *,
    query_vector: str,
    topk: int,
    candidate_limit: int,
    settings: tuple[tuple[str, str], ...] = (),
) -> list[tuple]:
    prefix = _settings_prefix(settings) if settings else sql.SQL("")
    statement = prefix + sql.SQL(
        "SELECT block_id, document_id, vector <-> %s::vector AS distance "
        "FROM {} ORDER BY vector <-> %s::vector LIMIT %s"
    ).format(sql.Identifier(route.table_name))
    cur.execute(statement, [query_vector, query_vector, max(1, int(candidate_limit))])
    allowed_docs = set(int(document_id) for document_id in route.doc_ids)
    filtered: list[tuple] = []
    for block_id, document_id, distance in cur.fetchall():
        if int(document_id) not in allowed_docs:
            continue
        filtered.append((int(block_id), int(document_id), None, float(distance)))
        if len(filtered) >= int(topk):
            break
    return filtered


def _route_candidate_select(route: Route, *, use_sql_filter: bool) -> sql.Composed:
    if use_sql_filter:
        return sql.SQL(
            "SELECT block_id, document_id, vector <-> %s::vector AS distance "
            "FROM {} WHERE pattern_id = ANY(%s) "
            "ORDER BY vector <-> %s::vector LIMIT %s"
        ).format(sql.Identifier(route.table_name))
    return sql.SQL(
        "SELECT block_id, document_id, vector <-> %s::vector AS distance "
        "FROM {} ORDER BY vector <-> %s::vector LIMIT %s"
    ).format(sql.Identifier(route.table_name))


def _route_candidate_params(route: Route, query: Query, *, use_sql_filter: bool) -> list[object]:
    if use_sql_filter:
        return [query.vector, list(route.pattern_ids), query.vector, int(query.topk)]
    return [query.vector, query.vector, int(query.topk)]


def _route_candidate_limit(
    route: Route,
    query: Query,
    *,
    use_sql_filter: bool,
    overfetch: int = 1,
    ef_search: int | None = None,
) -> int:
    base_limit = max(1, int(query.topk))
    factor = max(1, int(overfetch))
    limit = base_limit
    if use_sql_filter:
        selectivity = _ours_route_selectivity(route)
        limit = max(limit, int(math.ceil(base_limit * factor / max(selectivity, 0.000001))))
    if ef_search is not None:
        limit = max(limit, max(1, int(ef_search)) * factor)
    if route.partition_vectors > 0:
        limit = min(limit, int(route.partition_vectors))
    return max(1, limit)


def _route_candidate_params_with_limit(
    route: Route,
    query: Query,
    *,
    use_sql_filter: bool,
    overfetch: int,
) -> list[object]:
    limit = _route_candidate_limit(route, query, use_sql_filter=use_sql_filter, overfetch=overfetch)
    if use_sql_filter:
        return [query.vector, list(route.pattern_ids), query.vector, limit]
    return [query.vector, query.vector, limit]


def _route_candidate_select_with_access_cte(route: Route, *, use_sql_filter: bool) -> sql.Composed:
    predicate = sql.SQL(
        "EXISTS ("
        "SELECT 1 FROM direct_pg_qps_accessible_docs AS access_docs "
        "WHERE access_docs.document_id = partition_table.document_id"
        ")"
    )
    if use_sql_filter:
        predicate = sql.SQL("partition_table.pattern_id = ANY(%s) AND ") + predicate
    return sql.SQL(
        "SELECT partition_table.block_id, partition_table.document_id, "
        "partition_table.vector <-> %s::vector AS distance "
        "FROM {} AS partition_table "
        "WHERE "
    ).format(sql.Identifier(route.table_name)) + predicate + sql.SQL(
        " ORDER BY partition_table.vector <-> %s::vector LIMIT %s"
    )


def _route_candidate_params_with_access_cte(route: Route, query: Query, *, use_sql_filter: bool) -> list[object]:
    if use_sql_filter:
        return [query.vector, list(route.pattern_ids), query.vector, int(query.topk)]
    return [query.vector, query.vector, int(query.topk)]


def _route_candidate_select_with_user_access(route: Route, *, use_sql_filter: bool) -> sql.Composed:
    predicate = sql.SQL(
        "EXISTS ("
        "SELECT 1 FROM PermissionAssignment pa "
        "JOIN UserRoles ur ON pa.role_id = ur.role_id "
        "WHERE ur.user_id = %s AND pa.document_id = partition_table.document_id"
        ")"
    )
    if use_sql_filter:
        predicate = sql.SQL("partition_table.pattern_id = ANY(%s) AND ") + predicate
    return sql.SQL(
        "SELECT partition_table.block_id, partition_table.document_id, "
        "partition_table.vector <-> %s::vector AS distance "
        "FROM {} AS partition_table "
        "WHERE "
    ).format(sql.Identifier(route.table_name)) + predicate + sql.SQL(
        " ORDER BY partition_table.vector <-> %s::vector LIMIT %s"
    )


def _route_candidate_params_with_user_access(route: Route, query: Query, *, use_sql_filter: bool) -> list[object]:
    if use_sql_filter:
        return [query.vector, list(route.pattern_ids), int(query.user_id), query.vector, int(query.topk)]
    return [query.vector, int(query.user_id), query.vector, int(query.topk)]


def _honeybee_route_candidate_select(route: Route) -> sql.Composed:
    return sql.SQL(
        "SELECT block_id, document_id, vector <-> %s::vector AS distance "
        "FROM {} AS partition_table "
        "WHERE EXISTS ("
        "SELECT 1 FROM PermissionAssignment pa "
        "JOIN UserRoles ur ON pa.role_id = ur.role_id "
        "WHERE ur.user_id = %s AND pa.document_id = partition_table.document_id"
        ") ORDER BY vector <-> %s::vector LIMIT %s"
    ).format(sql.Identifier(route.table_name))


def _honeybee_route_candidate_params(query: Query) -> list[object]:
    return [query.vector, int(query.user_id), query.vector, int(query.topk)]


def _execute_honeybee_route(
    cur,
    route: Route,
    *,
    query: Query,
    settings: tuple[tuple[str, str], ...] = (),
) -> list[tuple]:
    prefix = _settings_prefix(settings) if settings else sql.SQL("")
    cur.execute(prefix + _honeybee_route_candidate_select(route), _honeybee_route_candidate_params(query))
    return _fetch_ann_rows(cur)


def _candidate_insert(route: Route, *, use_sql_filter: bool) -> sql.Composed:
    return sql.SQL("INSERT INTO pg_temp.direct_pg_qps_candidates ") + _route_candidate_select(
        route, use_sql_filter=use_sql_filter
    ) + sql.SQL("; ")


def _kernel_global_bound_setting(setting_name: str, topk: int) -> tuple[sql.Composed, list[int]]:
    statement = sql.SQL(
        "SELECT set_config({}, COALESCE(("
        "SELECT distance::text FROM pg_temp.direct_pg_qps_candidates "
        "ORDER BY distance OFFSET %s LIMIT 1"
        "), '-1'), false); "
    ).format(sql.Literal(setting_name))
    return statement, [max(0, int(topk) - 1)]


def _sql_topk_from_candidates(topk: int) -> tuple[sql.SQL, list[int]]:
    statement = sql.SQL(
        "SELECT block_id, document_id, distance FROM ("
        "SELECT DISTINCT ON (block_id, document_id) block_id, document_id, distance "
        "FROM pg_temp.direct_pg_qps_candidates "
        "ORDER BY block_id, document_id, distance"
        ") AS deduplicated ORDER BY distance, block_id, document_id LIMIT %s"
    )
    return statement, [int(topk)]


def install_ours_db_function(cur) -> None:
    cur.execute(
        """
        CREATE OR REPLACE FUNCTION pg_temp.direct_pg_qps_ours_hnsw_search(
            p_routes jsonb,
            p_query vector,
            p_topk integer,
            p_ef_search integer
        )
        RETURNS TABLE(block_id bigint, document_id bigint, distance double precision)
        LANGUAGE plpgsql
        AS $$
        DECLARE
            route_item jsonb;
            route_table text;
            route_patterns integer[];
            route_use_filter boolean;
            candidate_table text := 'direct_pg_qps_candidates_' || current_user;
        BEGIN
            PERFORM set_config('hnsw.ef_search', GREATEST(1, p_ef_search)::text, true);

            EXECUTE format(
                'CREATE TEMP TABLE IF NOT EXISTS pg_temp.%I ('
                'block_id BIGINT NOT NULL, '
                'document_id BIGINT NOT NULL, '
                'distance DOUBLE PRECISION NOT NULL'
                ') ON COMMIT PRESERVE ROWS',
                candidate_table
            );
            EXECUTE format('TRUNCATE pg_temp.%I', candidate_table);

            FOR route_item IN SELECT value FROM jsonb_array_elements(p_routes)
            LOOP
                route_table := route_item->>'table_name';
                route_use_filter := COALESCE((route_item->>'use_sql_filter')::boolean, false);
                SELECT COALESCE(array_agg(value::integer), ARRAY[]::integer[])
                INTO route_patterns
                FROM jsonb_array_elements_text(COALESCE(route_item->'pattern_ids', '[]'::jsonb)) AS pattern_value(value);

                IF route_use_filter THEN
                    EXECUTE format(
                        'INSERT INTO pg_temp.%I '
                        'SELECT block_id, document_id, vector <-> $1 AS distance '
                        'FROM %I WHERE pattern_id = ANY($2) '
                        'ORDER BY vector <-> $1 LIMIT $3',
                        candidate_table,
                        route_table
                    )
                    USING p_query, route_patterns, p_topk;
                ELSE
                    EXECUTE format(
                        'INSERT INTO pg_temp.%I '
                        'SELECT block_id, document_id, vector <-> $1 AS distance '
                        'FROM %I ORDER BY vector <-> $1 LIMIT $2',
                        candidate_table,
                        route_table
                    )
                    USING p_query, p_topk;
                END IF;
            END LOOP;

            RETURN QUERY EXECUTE format(
                'SELECT candidate.block_id, candidate.document_id, candidate.distance '
                'FROM ('
                'SELECT DISTINCT ON (candidate_inner.block_id, candidate_inner.document_id) '
                'candidate_inner.block_id, candidate_inner.document_id, candidate_inner.distance '
                'FROM pg_temp.%I AS candidate_inner '
                'ORDER BY candidate_inner.block_id, candidate_inner.document_id, candidate_inner.distance'
                ') AS candidate '
                'ORDER BY candidate.distance, candidate.block_id, candidate.document_id '
                'LIMIT $1',
                candidate_table
            )
            USING p_topk;
        END
        $$;
        """
    )


def _ours_routes_json(routes: list[Route], *, use_rls: bool) -> str:
    return json.dumps(
        [
            {
                "table_name": route.table_name,
                "pattern_ids": list(route.pattern_ids),
                "use_sql_filter": (not route.pure) and not use_rls,
            }
            for route in routes
        ]
    )


def _hnsw_settings(ef_search: int) -> tuple[tuple[str, str], ...]:
    return (("hnsw.ef_search", str(max(1, int(ef_search)))),)


def _route_hnsw_settings(route: Route, ef_search: int, indexed_tables: set[str] | None = None) -> tuple[tuple[str, str], ...]:
    settings = list(_hnsw_settings(ef_search))
    if indexed_tables is not None:
        settings.append(("enable_seqscan", "off" if route.table_name in indexed_tables else "on"))
    return tuple(settings)


class OursRouteParallelExecutor:
    """Shared route-level executor for SQUID-on-HNSW experiments."""

    def __init__(
        self,
        *,
        max_workers: int,
        ef_search: int,
        jit: str,
        parallel_workers: int,
        hnsw_iterative_scan: str,
        pg_parallel_route_scan: bool,
        prepare_routes: bool,
        use_user_access_join: bool,
        native_filter_location: str,
        route_overfetch: int,
    ) -> None:
        self.max_workers = max(1, int(max_workers))
        self.ef_search = int(ef_search)
        self.jit = str(jit)
        self.parallel_workers = int(parallel_workers)
        self.hnsw_iterative_scan = str(hnsw_iterative_scan)
        self.pg_parallel_route_scan = bool(pg_parallel_route_scan)
        self.prepare_routes = bool(prepare_routes)
        self.use_user_access_join = bool(use_user_access_join)
        self.native_filter_location = str(native_filter_location)
        self.route_overfetch = max(1, int(route_overfetch))
        self._executor = ThreadPoolExecutor(max_workers=self.max_workers, thread_name_prefix="ours-route")
        self._local = threading.local()
        self._lock = threading.Lock()
        self._connections: list[tuple[object, object]] = []

    def close(self) -> None:
        self._executor.shutdown(wait=True)
        with self._lock:
            connections = list(self._connections)
            self._connections.clear()
        for conn, cur in connections:
            try:
                cur.close()
            except BaseException:
                pass
            try:
                conn.close()
            except BaseException:
                pass

    def _state(self) -> tuple[object, object, PreparedRouteCache | None]:
        state = getattr(self._local, "state", None)
        if state is not None:
            return state
        conn = get_db_connection()
        conn.autocommit = True
        cur = conn.cursor()
        configure_session(
            cur,
            self.ef_search,
            self.jit,
            self.parallel_workers,
            self.hnsw_iterative_scan,
            force_hnsw_planner=True,
            pg_parallel_route_scan=self.pg_parallel_route_scan,
        )
        cur.execute("SET enable_bitmapscan = off")
        prepared_cache = PreparedRouteCache() if self.prepare_routes else None
        state = (conn, cur, prepared_cache)
        self._local.state = state
        with self._lock:
            self._connections.append((conn, cur))
        return state

    def warmup(self) -> None:
        ready = threading.Barrier(self.max_workers + 1)

        def initialize_worker() -> tuple[object, object, PreparedRouteCache | None]:
            state = self._state()
            ready.wait()
            return state

        futures = [self._executor.submit(initialize_worker) for _ in range(self.max_workers)]
        ready.wait()
        for future in as_completed(futures):
            future.result()

    def _execute_one_route(self, query: Query, route: Route, auth_filter: str) -> list[tuple]:
        conn, cur, prepared_cache = self._state()
        use_rls = auth_filter == "rls"

        def run() -> list[tuple]:
            if use_rls and self.use_user_access_join:
                cur.execute(
                    _route_candidate_select_with_user_access(route, use_sql_filter=(not route.pure)),
                    _route_candidate_params_with_user_access(route, query, use_sql_filter=(not route.pure)),
                )
                return _fetch_ann_rows(cur)
            if auth_filter == "native" and self.native_filter_location == "client" and not route.pure:
                return _execute_ann_route_client_filter(
                    cur,
                    route,
                    query_vector=query.vector,
                    topk=query.topk,
                    candidate_limit=_route_candidate_limit(
                        route,
                        query,
                        use_sql_filter=True,
                        overfetch=self.route_overfetch,
                        ef_search=self.ef_search,
                    ),
                )
            execute_route = prepared_cache.execute if prepared_cache is not None else _execute_ann_route
            return execute_route(
                cur,
                route,
                query_vector=query.vector,
                topk=_route_candidate_limit(
                    route,
                    query,
                    use_sql_filter=(not route.pure) and not use_rls,
                    overfetch=self.route_overfetch,
                ),
                use_sql_filter=(not route.pure) and not use_rls,
            )

        try:
            if use_rls and not self.use_user_access_join:
                return _execute_as_user(cur, query.user_id, run)
            return run()
        except BaseException:
            try:
                conn.rollback()
            except BaseException:
                pass
            raise

    def execute(
        self,
        query: Query,
        used_routes: list[Route],
        *,
        auth_filter: str,
        per_query_parallelism: int,
    ) -> tuple[list[tuple], int]:
        if not used_routes:
            return [], 0
        chunk_size = max(1, min(int(per_query_parallelism), len(used_routes)))
        candidates: list[tuple] = []
        for offset in range(0, len(used_routes), chunk_size):
            futures = [
                self._executor.submit(self._execute_one_route, query, route, auth_filter)
                for route in used_routes[offset : offset + chunk_size]
            ]
            for future in as_completed(futures):
                candidates.extend(future.result())
        return merge_topk(candidates, query.topk), len(used_routes)


def execute_partition_query(
    cur,
    query: Query,
    routes: dict[int, tuple[Route, ...]],
    *,
    ef_search: int,
    sql_batching: bool,
    postgresql_merge: bool,
    auth_filter: str,
) -> tuple[list[tuple], int]:
    """Run each routed HNSW partition and merge its candidates globally."""
    used_routes = routes.get(query.user_id, ())
    use_rls = auth_filter == "rls"

    def run() -> tuple[list[tuple], int]:
        if postgresql_merge:
            if not used_routes:
                return [], 0
            route_queries = [
                sql.SQL("(")
                + _route_candidate_select(route, use_sql_filter=(not route.pure) and not use_rls)
                + sql.SQL(")")
                for route in used_routes
            ]
            params: list[object] = []
            for route in used_routes:
                params.extend(_route_candidate_params(route, query, use_sql_filter=(not route.pure) and not use_rls))
            statement = (
                _settings_prefix(_hnsw_settings(ef_search))
                + sql.SQL("WITH route_candidates AS MATERIALIZED (")
                + sql.SQL(" UNION ALL " ).join(route_queries)
                + sql.SQL(
                    "), deduplicated AS ("
                    "SELECT DISTINCT ON (block_id, document_id) block_id, document_id, distance "
                    "FROM route_candidates ORDER BY block_id, document_id, distance"
                    ") SELECT block_id, document_id, distance FROM deduplicated "
                    "ORDER BY distance, block_id, document_id LIMIT %s"
                )
            )
            params.append(int(query.topk))
            cur.execute(statement, params)
            return _fetch_ann_rows(cur), len(used_routes)

        candidates: list[tuple] = []
        for route in used_routes:
            candidates.extend(
                _execute_ann_route(
                    cur,
                    route,
                    query_vector=query.vector,
                    topk=query.topk,
                    use_sql_filter=(not route.pure) and not use_rls,
                )
            )
        return merge_topk(candidates, query.topk), len(used_routes)

    if use_rls:
        return _execute_as_user(cur, query.user_id, run)
    return run()

def execute_ours_hnsw_query(
    cur,
    query: Query,
    routes: dict[int, tuple[Route, ...]],
    *,
    ef_search: int,
    sql_batching: bool,
    auth_filter: str,
    route_limit: int,
    ours_db_function: bool,
    prepared_cache: PreparedRouteCache | None = None,
    route_executor: OursRouteParallelExecutor | None = None,
    route_parallelism: int = 1,
    precompute_access: bool = False,
    route_overfetch: int = 1,
    native_filter_location: str = "client",
) -> tuple[list[tuple], int]:
    """Run SQUID routes with ordinary HNSW and a selectable auth filter."""
    used_routes = _ordered_ours_routes(routes.get(query.user_id, ()))
    if route_limit > 0:
        used_routes = used_routes[:route_limit]
    use_rls = auth_filter == "rls"

    if route_executor is not None and used_routes:
        return route_executor.execute(
            query,
            used_routes,
            auth_filter=auth_filter,
            per_query_parallelism=int(route_parallelism),
        )

    def run() -> tuple[list[tuple], int]:
        if precompute_access and use_rls:
            cur.execute(
                """
                CREATE TEMP TABLE IF NOT EXISTS direct_pg_qps_accessible_docs (
                    document_id INTEGER PRIMARY KEY
                ) ON COMMIT PRESERVE ROWS
                """
            )
            cur.execute("TRUNCATE direct_pg_qps_accessible_docs")
            cur.execute(
                """
                INSERT INTO direct_pg_qps_accessible_docs(document_id)
                SELECT DISTINCT pa.document_id
                FROM PermissionAssignment pa
                JOIN UserRoles ur ON pa.role_id = ur.role_id
                WHERE ur.user_id = %s
                """,
                [int(query.user_id)],
            )
            if len(used_routes) > 1 and sql_batching:
                route_queries = [
                    sql.SQL("(")
                    + _route_candidate_select_with_access_cte(route, use_sql_filter=(not route.pure))
                    + sql.SQL(")")
                    for route in used_routes
                ]
                params: list[object] = []
                for route in used_routes:
                    params.extend(_route_candidate_params_with_access_cte(route, query, use_sql_filter=(not route.pure)))
                statement = (
                    sql.SQL("WITH route_candidates AS MATERIALIZED (")
                    + sql.SQL(" UNION ALL ").join(route_queries)
                    + sql.SQL(
                        "), deduplicated AS ("
                        "SELECT DISTINCT ON (block_id, document_id) block_id, document_id, distance "
                        "FROM route_candidates ORDER BY block_id, document_id, distance"
                        ") SELECT block_id, document_id, distance FROM deduplicated "
                        "ORDER BY distance, block_id, document_id LIMIT %s"
                    )
                )
                params.append(int(query.topk))
                cur.execute(statement, params)
                return _fetch_ann_rows(cur), len(used_routes)

            candidates: list[tuple] = []
            for route in used_routes:
                cur.execute(
                    _route_candidate_select_with_access_cte(route, use_sql_filter=(not route.pure)),
                    _route_candidate_params_with_access_cte(route, query, use_sql_filter=(not route.pure)),
                )
                candidates.extend(_fetch_ann_rows(cur))
            return merge_topk(candidates, query.topk), len(used_routes)

        if ours_db_function:
            cur.execute(
                """
                SELECT block_id, document_id, distance
                FROM pg_temp.direct_pg_qps_ours_hnsw_search(%s::jsonb, %s::vector, %s, %s)
                """,
                [_ours_routes_json(used_routes, use_rls=use_rls), query.vector, int(query.topk), int(ef_search)],
            )
            return _fetch_ann_rows(cur), len(used_routes)

        use_client_filter = auth_filter == "native" and native_filter_location == "client"
        if use_client_filter:
            candidates: list[tuple] = []
            for route in used_routes:
                if route.pure:
                    execute_route = prepared_cache.execute if prepared_cache is not None else _execute_ann_route
                    candidates.extend(
                        execute_route(
                            cur,
                            route,
                            query_vector=query.vector,
                            topk=query.topk,
                            use_sql_filter=False,
                        )
                    )
                    continue
                candidates.extend(
                    _execute_ann_route_client_filter(
                        cur,
                        route,
                        query_vector=query.vector,
                        topk=query.topk,
                        candidate_limit=_route_candidate_limit(
                            route,
                            query,
                            use_sql_filter=True,
                            overfetch=route_overfetch,
                            ef_search=ef_search,
                        ),
                    )
                )
            return merge_topk(candidates, query.topk), len(used_routes)

        if len(used_routes) > 1 and sql_batching:
            route_queries = [
                sql.SQL("(")
                + _route_candidate_select(route, use_sql_filter=(not route.pure) and not use_rls)
                + sql.SQL(")")
                for route in used_routes
            ]
            params: list[object] = []
            for route in used_routes:
                params.extend(
                    _route_candidate_params_with_limit(
                        route,
                        query,
                        use_sql_filter=(not route.pure) and not use_rls,
                        overfetch=route_overfetch,
                    )
                )
            statement = (
                sql.SQL("WITH route_candidates AS MATERIALIZED (")
                + sql.SQL(" UNION ALL ").join(route_queries)
                + sql.SQL(
                    "), deduplicated AS ("
                    "SELECT DISTINCT ON (block_id, document_id) block_id, document_id, distance "
                    "FROM route_candidates ORDER BY block_id, document_id, distance"
                    ") SELECT block_id, document_id, distance FROM deduplicated "
                    "ORDER BY distance, block_id, document_id LIMIT %s"
                )
            )
            params.append(int(query.topk))
            cur.execute(statement, params)
            return _fetch_ann_rows(cur), len(used_routes)

        candidates: list[tuple] = []
        for route in used_routes:
            execute_route = prepared_cache.execute if prepared_cache is not None else _execute_ann_route
            candidates.extend(
                execute_route(
                    cur,
                    route,
                    query_vector=query.vector,
                    topk=_route_candidate_limit(
                        route,
                        query,
                        use_sql_filter=(not route.pure) and not use_rls,
                        overfetch=route_overfetch,
                    ),
                    use_sql_filter=(not route.pure) and not use_rls,
                )
            )
        return merge_topk(candidates, query.topk), len(used_routes)

    if use_rls and not precompute_access:
        return _execute_as_user(cur, query.user_id, run)
    return run()

def execute_veda_hnsw_query(
    cur,
    query: Query,
    routes: dict[int, tuple[Route, ...]],
    *,
    ef_search: int,
    sql_batching: bool,
    auth_filter: str,
    prepared_cache: PreparedRouteCache | None = None,
    indexed_tables: set[str] | None = None,
    native_filter_location: str = "client",
    route_overfetch: int = 1,
) -> tuple[list[tuple], int]:
    """Run VEDA routes with ordinary HNSW and a selectable auth filter."""
    used_routes = routes.get(query.user_id, ())
    use_rls = auth_filter == "rls"

    def run() -> tuple[list[tuple], int]:
        candidates: list[tuple] = []
        for route in used_routes:
            settings = _route_hnsw_settings(route, ef_search, indexed_tables) if sql_batching else ()
            if auth_filter == "native" and native_filter_location == "client" and not route.pure:
                candidates.extend(
                    _execute_ann_route_client_filter(
                        cur,
                        route,
                        query_vector=query.vector,
                        topk=query.topk,
                        candidate_limit=_route_candidate_limit(
                            route,
                            query,
                            use_sql_filter=True,
                            overfetch=route_overfetch,
                            ef_search=ef_search,
                        ),
                        settings=settings,
                    )
                )
                continue
            execute_route = prepared_cache.execute if prepared_cache is not None else _execute_ann_route
            candidates.extend(
                execute_route(
                    cur,
                    route,
                    query_vector=query.vector,
                    topk=_route_candidate_limit(
                        route,
                        query,
                        use_sql_filter=(not route.pure) and not use_rls,
                        overfetch=route_overfetch,
                    ),
                    use_sql_filter=(not route.pure) and not use_rls,
                    settings=settings,
                )
            )
        return merge_topk(candidates, query.topk), len(used_routes)

    if use_rls:
        return _execute_as_user(cur, query.user_id, run)
    return run()

def _ours_route_selectivity(route: Route) -> float:
    if route.partition_vectors <= 0 or route.accessible_vectors <= 0:
        return 1.0
    return min(1.0, max(0.000001, float(route.accessible_vectors) / float(route.partition_vectors)))


def _ours_route_impurity(route: Route) -> float:
    if route.accessible_vectors <= 0:
        return float("inf")
    if route.partition_vectors <= 0:
        return 1.0
    return float(route.partition_vectors) / float(route.accessible_vectors)


def _ordered_ours_routes(routes: tuple[Route, ...]) -> list[Route]:
    return sorted(
        routes,
        key=lambda route: (
            not route.pure,
            _ours_route_impurity(route),
            route.partition_vectors,
            route.route_kind,
            route.cluster_id,
            route.partition_id,
        ),
    )


def _squidhnsw_settings(
    route: Route, *, base_ef: int, max_ef: int, topk: int, global_bound: float
) -> tuple[tuple[str, str], ...]:
    route_cap = max(1, int(route.partition_vectors)) if route.partition_vectors > 0 else max(1, int(max_ef))
    effective_max_ef = min(max(1, int(max_ef)), route_cap)
    bound = float(global_bound) if math.isfinite(float(global_bound)) else -1.0
    allowed = "" if route.pure else ",".join(str(pattern_id) for pattern_id in route.pattern_ids)
    return (
        ("enable_seqscan", "off"),
        ("enable_bitmapscan", "off"),
        ("squidhnsw.base_ef", str(min(max(1, int(base_ef)), effective_max_ef))),
        ("squidhnsw.max_ef", str(effective_max_ef)),
        ("squidhnsw.topk", str(max(0, int(topk)))),
        ("squidhnsw.global_bound", f"{bound:.17g}"),
        ("squidhnsw.route_selectivity", f"{_ours_route_selectivity(route):.12f}"),
        ("squidhnsw.allowed_patterns", allowed),
    )


def _configure_squidhnsw_route(cur, route: Route, *, base_ef: int, max_ef: int, topk: int, global_bound: float) -> None:
    for name, value in _squidhnsw_settings(
        route, base_ef=base_ef, max_ef=max_ef, topk=topk, global_bound=global_bound
    ):
        cur.execute(sql.SQL("SET {} = {}").format(sql.Identifier(name), sql.Literal(value)))


def execute_ours_kernel_query(
    cur,
    query: Query,
    routes: dict[int, tuple[Route, ...]],
    base_ef: int,
    max_ef: int,
    *,
    sql_batching: bool,
    postgresql_merge: bool,
    route_limit: int,
) -> tuple[list[tuple], int]:
    """Use SQUIDHNSW filtering and merge all route candidates globally."""
    used_routes = _ordered_ours_routes(routes.get(query.user_id, ()))
    if route_limit > 0:
        used_routes = used_routes[:route_limit]
    if postgresql_merge:
        statement = sql.SQL("TRUNCATE pg_temp.direct_pg_qps_candidates; ")
        params: list[object] = []
        for route in used_routes:
            settings = tuple(
                (name, value)
                for name, value in _squidhnsw_settings(
                    route,
                    base_ef=base_ef,
                    max_ef=max_ef,
                    topk=query.topk,
                    global_bound=-1.0,
                )
                if name != "squidhnsw.global_bound"
            )
            bound_statement, bound_params = _kernel_global_bound_setting(
                "squidhnsw.global_bound", query.topk
            )
            statement += _settings_prefix(settings) + bound_statement
            params.extend(bound_params)
            statement += _candidate_insert(route, use_sql_filter=False)
            params.extend(_route_candidate_params(route, query, use_sql_filter=False))
        final_statement, final_params = _sql_topk_from_candidates(query.topk)
        statement += final_statement
        params.extend(final_params)
        cur.execute(statement, params)
        return _fetch_ann_rows(cur), len(used_routes)

    candidates: list[tuple] = []
    for route in used_routes:
        global_bound = _paper_global_bound(candidates, query.topk)
        if sql_batching:
            settings = _squidhnsw_settings(
                route,
                base_ef=base_ef,
                max_ef=max_ef,
                topk=query.topk,
                global_bound=global_bound,
            )
            candidates.extend(
                _execute_ann_route(
                    cur,
                    route,
                    query_vector=query.vector,
                    topk=query.topk,
                    use_sql_filter=False,
                    settings=settings,
                )
            )
        else:
            _configure_squidhnsw_route(
                cur,
                route,
                base_ef=base_ef,
                max_ef=max_ef,
                topk=query.topk,
                global_bound=global_bound,
            )
            candidates.extend(
                _execute_ann_route(
                    cur,
                    route,
                    query_vector=query.vector,
                    topk=query.topk,
                    use_sql_filter=False,
                )
            )
    return merge_topk(candidates, query.topk), len(used_routes)


def _veda_route_selectivity(route: Route) -> float:
    impurity = float(route.impurity_factor or 1.0)
    if impurity <= 0.0:
        return 1.0
    return min(1.0, max(0.000001, 1.0 / impurity))


def _vedahnsw_settings(
    route: Route, *, base_ef: int, max_ef: int, topk: int, global_bound: float
) -> tuple[tuple[str, str], ...]:
    effective_base_ef = min(5000, max(1, int(base_ef)))
    effective_max_ef = min(5000, max(effective_base_ef, int(max_ef)))
    bound = float(global_bound) if math.isfinite(float(global_bound)) else -1.0
    allowed = ",".join(str(pattern_id) for pattern_id in route.pattern_ids)
    return (
        ("enable_seqscan", "off"),
        ("enable_bitmapscan", "off"),
        ("vedahnsw.base_ef", str(effective_base_ef)),
        ("vedahnsw.max_ef", str(effective_max_ef)),
        ("vedahnsw.topk", str(max(0, int(topk)))),
        ("vedahnsw.global_bound", f"{bound:.17g}"),
        ("vedahnsw.route_selectivity", f"{_veda_route_selectivity(route):.12f}"),
        ("vedahnsw.allowed_patterns", allowed),
    )


def _configure_vedahnsw_route(cur, route: Route, *, base_ef: int, max_ef: int, topk: int, global_bound: float) -> None:
    for name, value in _vedahnsw_settings(
        route, base_ef=base_ef, max_ef=max_ef, topk=topk, global_bound=global_bound
    ):
        cur.execute(sql.SQL("SET {} = {}").format(sql.Identifier(name), sql.Literal(value)))


def execute_veda_kernel_query(
    cur,
    query: Query,
    routes: dict[int, tuple[Route, ...]],
    base_ef: int,
    max_ef_cap: int,
    *,
    sql_batching: bool,
    postgresql_merge: bool,
    veda_native_all_routes: bool,
) -> tuple[list[tuple], int]:
    """Run VEDA/EffVeda routes and merge all candidates globally."""
    used_routes = routes.get(query.user_id, ())
    if veda_native_all_routes:
        pure_indices = sorted(
            (route for route in used_routes if route.route_kind != "impure_index"),
            key=lambda route: (route.route_kind != "leftover", route.partition_vectors, route.partition_id),
        )
        impure_indices = sorted(
            (route for route in used_routes if route.route_kind == "impure_index"),
            key=lambda route: (route.impurity_factor, route.partition_vectors, route.partition_id),
        )
        leftovers: list[Route] = []
    else:
        pure_indices = sorted(
            (route for route in used_routes if route.route_kind == "index"),
            key=lambda route: (route.partition_vectors, route.partition_id),
        )
        impure_indices = sorted(
            (route for route in used_routes if route.route_kind == "impure_index"),
            key=lambda route: (route.impurity_factor, route.partition_vectors, route.partition_id),
        )
        leftovers = sorted(
            (route for route in used_routes if route.route_kind == "leftover"),
            key=lambda route: (route.partition_vectors, route.partition_id),
        )

    if postgresql_merge:
        statement = sql.SQL("TRUNCATE pg_temp.direct_pg_qps_candidates; ")
        params: list[object] = []
        kernel_routes = [(route, int(base_ef)) for route in pure_indices]
        kernel_routes.extend(
            (
                route,
                max(int(base_ef), int(math.ceil(float(route.impurity_factor) * float(base_ef)))),
            )
            for route in impure_indices
        )
        for route in leftovers:
            statement += _settings_prefix(
                _hnsw_settings(base_ef) + (("enable_seqscan", "on"), ("enable_bitmapscan", "on"))
            )
            statement += _candidate_insert(route, use_sql_filter=True)
            params.extend(_route_candidate_params(route, query, use_sql_filter=True))

        for route, route_max_ef in kernel_routes:
            settings = tuple(
                (name, value)
                for name, value in _vedahnsw_settings(
                    route,
                    base_ef=base_ef,
                    max_ef=min(
                        max(1, int(max_ef_cap)),
                        max(int(base_ef), int(route_max_ef)),
                    ),
                    topk=query.topk,
                    global_bound=-1.0,
                )
                if name != "vedahnsw.global_bound"
            )
            bound_statement, bound_params = _kernel_global_bound_setting(
                "vedahnsw.global_bound", query.topk
            )
            statement += _settings_prefix(settings) + bound_statement
            params.extend(bound_params)
            statement += _candidate_insert(route, use_sql_filter=False)
            params.extend(_route_candidate_params(route, query, use_sql_filter=False))

        final_statement, final_params = _sql_topk_from_candidates(query.topk)
        statement += final_statement
        params.extend(final_params)
        cur.execute(statement, params)
        return _fetch_ann_rows(cur), len(used_routes)

    candidates: list[tuple] = []

    def search_kernel_route(route: Route, route_max_ef: int) -> None:
        route_max_ef = min(
            max(1, int(max_ef_cap)),
            max(int(base_ef), int(route_max_ef)),
        )
        global_bound = _paper_global_bound(candidates, query.topk)
        if sql_batching:
            settings = _vedahnsw_settings(
                route,
                base_ef=base_ef,
                max_ef=route_max_ef,
                topk=query.topk,
                global_bound=global_bound,
            )
            candidates.extend(
                _execute_ann_route(
                    cur,
                    route,
                    query_vector=query.vector,
                    topk=query.topk,
                    use_sql_filter=False,
                    settings=settings,
                )
            )
        else:
            _configure_vedahnsw_route(
                cur,
                route,
                base_ef=base_ef,
                max_ef=route_max_ef,
                topk=query.topk,
                global_bound=global_bound,
            )
            candidates.extend(
                _execute_ann_route(
                    cur,
                    route,
                    query_vector=query.vector,
                    topk=query.topk,
                    use_sql_filter=False,
                )
            )

    hnsw_settings = (
        _hnsw_settings(base_ef) + (("enable_seqscan", "on"), ("enable_bitmapscan", "on"))
        if sql_batching
        else ()
    )
    for route in leftovers:
        candidates.extend(
            _execute_ann_route(
                cur,
                route,
                query_vector=query.vector,
                topk=query.topk,
                use_sql_filter=True,
                settings=hnsw_settings,
            )
        )

    for route in pure_indices:
        search_kernel_route(route, base_ef)
    for route in impure_indices:
        expanded_ef = max(
            int(base_ef),
            int(math.ceil(float(route.impurity_factor) * float(base_ef))),
        )
        search_kernel_route(route, expanded_ef)

    return merge_topk(candidates, query.topk), len(used_routes)


def _hqi_route_candidate_select(route: Route) -> sql.Composed:
    return sql.SQL(
        "SELECT block_id, document_id, vector <-> %s::vector AS distance "
        "FROM {} WHERE document_id = ANY(%s) "
        "ORDER BY vector <-> %s::vector LIMIT %s"
    ).format(sql.Identifier(route.table_name))


def _hqi_route_candidate_params(route: Route, query: Query) -> list[object]:
    return [query.vector, list(route.doc_ids), query.vector, int(query.topk)]


def _execute_hqi_route(
    cur,
    route: Route,
    *,
    query: Query,
    settings: tuple[tuple[str, str], ...] = (),
) -> list[tuple]:
    prefix = _settings_prefix(settings) if settings else sql.SQL("")
    cur.execute(prefix + _hqi_route_candidate_select(route), _hqi_route_candidate_params(route, query))
    return _fetch_ann_rows(cur)


def execute_hqi_query(
    cur,
    query: Query,
    routes: dict[Query, tuple[Route, ...]],
    *,
    ef_search: int,
    sql_batching: bool,
    auth_filter: str,
    native_filter_location: str = "client",
    route_overfetch: int = 1,
) -> tuple[list[tuple], int]:
    used_routes = routes.get(query, ())
    settings = _hnsw_settings(ef_search) if sql_batching else ()
    use_rls = auth_filter == "rls"

    def hqi_select(route: Route) -> sql.Composed:
        return _route_candidate_select(route, use_sql_filter=False) if use_rls else _hqi_route_candidate_select(route)

    def hqi_params(route: Route) -> list[object]:
        return _route_candidate_params(route, query, use_sql_filter=False) if use_rls else _hqi_route_candidate_params(route, query)

    def run() -> tuple[list[tuple], int]:
        if not used_routes:
            return [], 0
        if auth_filter == "native" and native_filter_location == "client":
            candidates: list[tuple] = []
            for route in used_routes:
                candidates.extend(
                    _execute_ann_route_docid_client_filter(
                        cur,
                        route,
                        query_vector=query.vector,
                        topk=query.topk,
                        candidate_limit=_route_candidate_limit(
                            route,
                            query,
                            use_sql_filter=True,
                            overfetch=route_overfetch,
                            ef_search=ef_search,
                        ),
                        settings=settings,
                    )
                )
            return merge_topk(candidates, query.topk), len(used_routes)
        if sql_batching:
            statement = _settings_prefix(settings) + sql.SQL("WITH route_candidates AS MATERIALIZED (")
            statement += sql.SQL(" UNION ALL " ).join(
                sql.SQL("(") + hqi_select(route) + sql.SQL(")")
                for route in used_routes
            )
            statement += sql.SQL(
                "), deduplicated AS ("
                "SELECT DISTINCT ON (block_id, document_id) block_id, document_id, distance "
                "FROM route_candidates ORDER BY block_id, document_id, distance"
                ") SELECT block_id, document_id, distance FROM deduplicated "
                "ORDER BY distance, block_id, document_id LIMIT %s"
            )
            params: list[object] = []
            for route in used_routes:
                params.extend(hqi_params(route))
            params.append(int(query.topk))
            cur.execute(statement, params)
            return _fetch_ann_rows(cur), len(used_routes)

        candidates: list[tuple] = []
        for route in used_routes:
            if use_rls:
                candidates.extend(
                    _execute_ann_route(
                        cur,
                        route,
                        query_vector=query.vector,
                        topk=query.topk,
                        use_sql_filter=False,
                        settings=settings,
                    )
                )
            else:
                candidates.extend(_execute_hqi_route(cur, route, query=query, settings=settings))
        return merge_topk(candidates, query.topk), len(used_routes)

    return _execute_as_user(cur, query.user_id, run) if use_rls else run()

def _paper_global_bound(rows: list[tuple], topk: int) -> float:
    if len(rows) < topk:
        return float("inf")
    return float(sorted(rows, key=lambda row: float(row[3]))[topk - 1][3])


def _missing_tables(cur, table_names: set[str]) -> list[str]:
    missing: list[str] = []
    for table_name in sorted(table_names):
        cur.execute("SELECT to_regclass(%s) IS NULL", [f"public.{table_name}"])
        if bool(cur.fetchone()[0]):
            missing.append(table_name)
    return missing


def _tables_without_index_am(cur, table_names: set[str], access_method: str) -> list[str]:
    missing: list[str] = []
    for table_name in sorted(table_names):
        cur.execute(
            """
            SELECT NOT EXISTS (
                SELECT 1
                FROM pg_index ix
                JOIN pg_class index_class ON index_class.oid = ix.indexrelid
                JOIN pg_am am ON am.oid = index_class.relam
                WHERE ix.indrelid = to_regclass(%s)
                  AND ix.indisvalid
                  AND am.amname = %s
            )
            """,
            [table_name, access_method],
        )
        if bool(cur.fetchone()[0]):
            missing.append(table_name)
    return missing


def _tables_with_index_am(cur, table_names: set[str], access_method: str) -> set[str]:
    indexed: set[str] = set()
    for table_name in sorted(table_names):
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_index ix
                JOIN pg_class index_class ON index_class.oid = ix.indexrelid
                JOIN pg_am am ON am.oid = index_class.relam
                WHERE ix.indrelid = to_regclass(%s)
                  AND ix.indisvalid
                  AND am.amname = %s
            )
            """,
            [table_name, access_method],
        )
        if bool(cur.fetchone()[0]):
            indexed.add(table_name)
    return indexed


def _preview(values: list[str], limit: int = 5) -> str:
    preview = ", ".join(values[:limit])
    return preview if len(values) <= limit else f"{preview}, ... (+{len(values) - limit})"


def _ensure_query_roles_available(cur, user_ids: set[int], label: str) -> None:
    if not user_ids:
        return
    user_names = sorted(str(user_id) for user_id in user_ids)
    cur.execute(
        """
        SELECT requested.user_id
        FROM unnest(%s::text[]) AS requested(user_id)
        LEFT JOIN pg_roles role_entry ON role_entry.rolname = requested.user_id
        WHERE role_entry.oid IS NULL
           OR NOT pg_has_role(current_user, requested.user_id, 'MEMBER')
        ORDER BY requested.user_id
        LIMIT 5
        """,
        [user_names],
    )
    unavailable_users = [str(row[0]) for row in cur.fetchall()]
    if unavailable_users:
        raise RuntimeError(
            f"{label} direct QPS cannot switch to query users: "
            f"{_preview(unavailable_users)}. Initialize user database roles and grants first."
        )


def _install_partition_rls_policies(cur, table_names: set[str]) -> None:
    if not table_names:
        return
    cur.execute("GRANT SELECT ON PermissionAssignment TO PUBLIC")
    cur.execute("GRANT SELECT ON UserRoles TO PUBLIC")
    for table_name in sorted(table_names):
        table_ident = sql.Identifier(table_name)
        cur.execute(sql.SQL("GRANT SELECT ON {} TO PUBLIC").format(table_ident))
        cur.execute(sql.SQL("ALTER TABLE {} ENABLE ROW LEVEL SECURITY").format(table_ident))
        cur.execute(sql.SQL("ALTER TABLE {} FORCE ROW LEVEL SECURITY").format(table_ident))
        cur.execute(sql.SQL("DROP POLICY IF EXISTS direct_pg_qps_rls_policy ON {}").format(table_ident))
        cur.execute(
            sql.SQL(
                """
                CREATE POLICY direct_pg_qps_rls_policy ON {} FOR SELECT
                USING (
                    EXISTS (
                        SELECT 1
                        FROM PermissionAssignment pa
                        JOIN UserRoles ur ON pa.role_id = ur.role_id
                        WHERE pa.document_id = {}.document_id
                          AND ur.user_id = current_user::int
                    )
                )
                """
            ).format(table_ident, table_ident)
        )

def validate_method_prerequisites(
    name: str,
    queries: list[Query],
    routes: dict[int, tuple[Route, ...]] | None,
    *,
    index_mode: str,
    auth_filter: str,
    veda_native_all_routes: bool = False,
) -> None:
    """Fail before worker startup when a baseline has not been materialized."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if name == "rls":
                cur.execute("SELECT relrowsecurity FROM pg_class WHERE oid = %s::regclass", ["documentblocks"])
                row = cur.fetchone()
                rls_enabled = bool(row and row[0])
                cur.execute("SELECT EXISTS (SELECT 1 FROM pg_policy WHERE polrelid = %s::regclass)", ["documentblocks"])
                has_policy = bool(cur.fetchone()[0])
                sample_user = str(queries[0].user_id) if queries else ""
                cur.execute("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = %s)", [sample_user])
                has_user_role = bool(cur.fetchone()[0])
                cur.execute("SELECT has_table_privilege(%s, %s, %s)", [sample_user, "documentblocks", "SELECT"])
                has_select = bool(cur.fetchone()[0])
                cur.execute("SELECT pg_has_role(current_user, %s, %s)", [sample_user, "MEMBER"])
                can_set_role = bool(cur.fetchone()[0])
                if not (rls_enabled and has_policy and has_user_role and has_select and can_set_role):
                    raise RuntimeError(
                        "RLS is not ready for this direct harness "
                        f"(enabled={rls_enabled}, policy={has_policy}, sample_role={has_user_role}, "
                        f"select={has_select}, benchmark_user_can_set_role={can_set_role}). "
                        "Initialize RLS first; this connection-reuse harness also requires the benchmark login "
                        "to be a member of each user role, or it must use per-user database connections like the original baseline."
                    )
                return

            requested_users = {int(query.user_id) for query in queries}
            if name == "hqi":
                missing_queries = [str(index) for index, query in enumerate(queries) if query not in (routes or {})]
                empty_queries = [
                    str(index) for index, query in enumerate(queries)
                    if not tuple((routes or {}).get(query, ()))
                ]
                if missing_queries or empty_queries:
                    affected = sorted(set(missing_queries + empty_queries), key=int)
                    raise RuntimeError(
                        f"HQI has no materialized route for query indexes: {_preview(affected)}. "
                        "Rebuild or persist the QD-tree baseline before direct QPS."
                    )
                required_routes = [route for query in queries for route in (routes or {}).get(query, ())]
            else:
                route_users = set(routes or {})
                missing_users = sorted(str(user_id) for user_id in requested_users - route_users)
                empty_users = sorted(
                    str(user_id) for user_id in requested_users
                    if not tuple((routes or {}).get(user_id, ()))
                )
                if missing_users or empty_users:
                    affected = sorted(set(missing_users + empty_users), key=int)
                    raise RuntimeError(
                        f"{name} has no materialized route for query users: {_preview(affected)}. "
                        "Rebuild this baseline before direct QPS."
                    )
                required_routes = [route for user_id in requested_users for route in (routes or {}).get(user_id, ())]

            table_names = {route.table_name for route in required_routes}
            missing_tables = _missing_tables(cur, table_names)
            if missing_tables:
                raise RuntimeError(
                    f"{name} metadata references missing partition tables: {_preview(missing_tables)}. "
                    "Rebuild this baseline before direct QPS."
                )

            if name == "role":
                missing_indexes = _tables_without_index_am(cur, table_names, "hnsw")
                if missing_indexes:
                    raise RuntimeError(
                        "ROLE partitions are missing HNSW indexes: "
                        f"{_preview(missing_indexes)}. Rebuild ROLE indexes before direct QPS."
                    )
            elif name == "honeybee":
                _ensure_query_roles_available(cur, requested_users, "HONEYBEE")
                missing_indexes = _tables_without_index_am(cur, table_names, "hnsw")
                if missing_indexes:
                    raise RuntimeError(
                        "HONEYBEE partitions are missing HNSW indexes: "
                        f"{_preview(missing_indexes)}. Rebuild HONEYBEE before direct QPS."
                    )
            elif name == "hqi":
                _ensure_query_roles_available(cur, requested_users, "HQI")
                missing_indexes = _tables_without_index_am(cur, table_names, "hnsw")
                if missing_indexes:
                    raise RuntimeError(
                        "HQI partitions are missing HNSW indexes: "
                        f"{_preview(missing_indexes)}. Run controller/baseline/HQI/persist_tree.py --index-type hnsw."
                    )
            elif name == "ours":
                access_method = "hnsw" if index_mode == "hnsw" else "squidhnsw"
                missing_indexes = _tables_without_index_am(cur, table_names, access_method)
                if missing_indexes:
                    raise RuntimeError(
                        f"SQUID partitions are missing {access_method} indexes: "
                        f"{_preview(missing_indexes)}. Rebuild SQUID with --index-type {access_method}."
                    )
            elif name in {"veda", "effveda"}:
                if index_mode != "hnsw":
                    checked_tables = {
                        route.table_name for route in required_routes
                        if route.route_kind in {"index", "impure_index"}
                    }
                    access_method = "vedahnsw"
                    if veda_native_all_routes:
                        checked_tables = {route.table_name for route in required_routes}
                    missing_indexes = _tables_without_index_am(cur, checked_tables, access_method)
                    if missing_indexes:
                        raise RuntimeError(
                            f"{name} routes are missing {access_method} indexes: {_preview(missing_indexes)}. "
                            f"Rebuild the VEDA baseline with --index-type {access_method}."
                        )

            if auth_filter == "rls":
                _ensure_query_roles_available(cur, requested_users, name.upper())
                _install_partition_rls_policies(cur, table_names)
                conn.commit()
    finally:
        conn.close()

def _execute_as_user(cur, user_id: int, callback):
    cur.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(str(int(user_id)))))
    try:
        result = callback()
    except BaseException:
        cur.connection.rollback()
        try:
            cur.execute("RESET ROLE")
        except BaseException:
            cur.connection.rollback()
        raise
    else:
        cur.execute("RESET ROLE")
        return result


def execute_rls_query(cur, query: Query, *, ef_search: int, sql_batching: bool) -> tuple[list[tuple], int]:
    route = Route("documentblocks", pure=True)
    return _execute_as_user(
        cur,
        query.user_id,
        lambda: (
            _execute_ann_route(
                cur,
                route,
                query_vector=query.vector,
                topk=query.topk,
                use_sql_filter=False,
                settings=_hnsw_settings(ef_search) if sql_batching else (),
            ),
            1,
        ),
    )


def execute_honeybee_partition_query(
    cur,
    query: Query,
    routes: dict[int, tuple[Route, ...]],
    *,
    ef_search: int,
    sql_batching: bool,
    postgresql_merge: bool,
    auth_filter: str,
) -> tuple[list[tuple], int]:
    used_routes = routes.get(query.user_id, ())
    settings = _hnsw_settings(ef_search) if sql_batching else ()
    use_rls = auth_filter == "rls"
    if postgresql_merge:
        if not used_routes:
            return [], 0
        statement = _settings_prefix(settings) + sql.SQL("WITH route_candidates AS MATERIALIZED (")
        statement += sql.SQL(" UNION ALL " ).join(
            sql.SQL("(")
            + (_route_candidate_select(route, use_sql_filter=False) if use_rls else _honeybee_route_candidate_select(route))
            + sql.SQL(")")
            for route in used_routes
        )
        statement += sql.SQL(
            "), deduplicated AS ("
            "SELECT DISTINCT ON (block_id, document_id) block_id, document_id, distance "
            "FROM route_candidates ORDER BY block_id, document_id, distance"
            ") SELECT block_id, document_id, distance FROM deduplicated "
            "ORDER BY distance, block_id, document_id LIMIT %s"
        )
        params: list[object] = []
        for route in used_routes:
            params.extend(_route_candidate_params(route, query, use_sql_filter=False) if use_rls else _honeybee_route_candidate_params(query))
        params.append(int(query.topk))
        cur.execute(statement, params)
        return _fetch_ann_rows(cur), len(used_routes)

    candidates: list[tuple] = []
    for route in used_routes:
        if use_rls:
            candidates.extend(
                _execute_ann_route(
                    cur,
                    route,
                    query_vector=query.vector,
                    topk=query.topk,
                    use_sql_filter=False,
                    settings=settings,
                )
            )
        else:
            candidates.extend(_execute_honeybee_route(cur, route, query=query, settings=settings))
    return merge_topk(candidates, query.topk), len(used_routes)

def execute_honeybee_query(
    cur,
    query: Query,
    routes: dict[int, tuple[Route, ...]],
    *,
    ef_search: int,
    sql_batching: bool,
    postgresql_merge: bool,
    auth_filter: str,
) -> tuple[list[tuple], int]:
    def run() -> tuple[list[tuple], int]:
        return execute_honeybee_partition_query(
            cur,
            query,
            routes,
            ef_search=ef_search,
            sql_batching=sql_batching,
            postgresql_merge=postgresql_merge,
            auth_filter=auth_filter,
        )

    if auth_filter == "rls":
        return _execute_as_user(cur, query.user_id, run)
    return run()

def run_method(
    name: str,
    queries: list[Query],
    routes: dict[int, tuple[Route, ...]] | None,
    args: argparse.Namespace,
    selection: PlanSelection | None = None,
) -> dict[str, float]:
    if name != "rls" and routes is None:
        raise RuntimeError(f"{name} has no routes")
    validate_method_prerequisites(
        name,
        queries,
        routes,
        index_mode=args.index_mode,
        auth_filter=args.auth_filter,
        veda_native_all_routes=bool(args.veda_native_all_routes),
    )
    indexed_tables: set[str] | None = None
    if name in {"veda", "effveda"} and args.index_mode == "hnsw":
        route_tables = {route.table_name for user_routes in (routes or {}).values() for route in user_routes}
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                indexed_tables = _tables_with_index_am(cur, route_tables, "hnsw")
        finally:
            conn.close()
    duration_seconds = max(0.0, float(getattr(args, "duration_seconds", 0.0)))
    measured_queries = queries * max(1, int(args.query_repetitions))
    worker_count = min(max(1, int(args.concurrency)), max(1, len(measured_queries)))
    warmup_batches = round_robin_batches(measured_queries, worker_count)
    scheduler = getattr(args, "scheduler", "dynamic")
    work_queue: queue.Queue[Query] | None = None
    if scheduler == "dynamic":
        work_queue = queue.Queue()
        for query in measured_queries:
            work_queue.put(query)
    ready = threading.Barrier(len(warmup_batches) + 1)
    start = threading.Event()
    route_parallelism = max(1, int(getattr(args, "route_parallelism", 1)))
    route_scheduler = str(getattr(args, "route_scheduler", "inline"))
    use_route_pool = route_scheduler == "pool" or route_parallelism > 1
    route_worker_count = int(getattr(args, "route_worker_count", 0))
    if route_worker_count <= 0:
        route_worker_count = max(route_parallelism, worker_count * route_parallelism)
    route_executor: OursRouteParallelExecutor | None = None
    if name == "ours" and args.index_mode == "hnsw" and use_route_pool:
        route_executor = OursRouteParallelExecutor(
            max_workers=route_worker_count,
            ef_search=args.ef_search,
            jit=args.jit,
            parallel_workers=args.parallel_workers,
            hnsw_iterative_scan=args.hnsw_iterative_scan,
            pg_parallel_route_scan=bool(args.pg_parallel_route_scan),
            prepare_routes=bool(args.prepare_routes),
            use_user_access_join=bool(args.ours_precompute_access and args.auth_filter == "rls"),
            native_filter_location=str(args.native_filter_location),
            route_overfetch=int(args.route_overfetch),
        )
        route_executor.warmup()

    def worker(warmup_batch: list[Query]) -> WorkerResult:
        conn = None
        try:
            pool_only_worker = name == "ours" and args.index_mode == "hnsw" and route_executor is not None
            conn = None if pool_only_worker else get_db_connection()
            # Direct QPS should not accumulate AccessShareLocks for every routed
            # partition in one long transaction, especially for VEDA/EffVeda
            # plans with hundreds of route tables.
            if conn is not None:
                conn.autocommit = True
            cursor_context = conn.cursor() if conn is not None else nullcontext(None)
            with cursor_context as cur:
                if cur is not None:
                    configure_session(
                        cur,
                        args.ef_search,
                        args.jit,
                        args.parallel_workers,
                        args.hnsw_iterative_scan,
                        force_hnsw_planner=args.index_mode == "hnsw",
                        pg_parallel_route_scan=bool(args.pg_parallel_route_scan),
                    )
                    prepared_cache = PreparedRouteCache() if args.prepare_routes else None
                    if name == "ours":
                        # Match the SQUID search path, which forces its custom index only.
                        cur.execute("SET enable_bitmapscan = off")
                        if args.ours_db_function:
                            install_ours_db_function(cur)
                    if args.postgresql_merge and name in {"ours", "veda", "effveda"}:
                        cur.execute(
                            "CREATE TEMP TABLE IF NOT EXISTS direct_pg_qps_candidates ("
                            "block_id BIGINT NOT NULL, document_id BIGINT NOT NULL, "
                            "distance DOUBLE PRECISION NOT NULL"
                            ") ON COMMIT PRESERVE ROWS"
                        )
                else:
                    prepared_cache = None

                def execute_query(query: Query) -> tuple[list[tuple], int]:
                    if name == "rls":
                        return execute_rls_query(
                            cur, query, ef_search=args.ef_search, sql_batching=method_sql_batching(name, args)
                        )
                    if name == "honeybee":
                        return execute_honeybee_query(
                            cur,
                            query,
                            routes,
                            ef_search=args.ef_search,
                            sql_batching=method_sql_batching(name, args),
                            postgresql_merge=args.postgresql_merge,
                            auth_filter=args.auth_filter,
                        )
                    if name == "hqi":
                        return execute_hqi_query(
                            cur,
                            query,
                            routes,
                            ef_search=args.ef_search,
                            sql_batching=method_sql_batching(name, args),
                            auth_filter=args.auth_filter,
                            native_filter_location=str(args.native_filter_location),
                            route_overfetch=int(args.route_overfetch),
                        )
                    if name == "ours":
                        if args.index_mode == "hnsw":
                            return execute_ours_hnsw_query(
                                cur,
                                query,
                                routes,
                                ef_search=args.ef_search,
                                sql_batching=method_sql_batching(name, args),
                                auth_filter=args.auth_filter,
                                route_limit=args.route_limit,
                                ours_db_function=bool(args.ours_db_function),
                                prepared_cache=prepared_cache,
                                route_executor=route_executor,
                                route_parallelism=route_parallelism,
                                precompute_access=bool(args.ours_precompute_access),
                                route_overfetch=int(args.route_overfetch),
                                native_filter_location=str(args.native_filter_location),
                            )
                        return execute_ours_kernel_query(
                            cur,
                            query,
                            routes,
                            args.ef_search,
                            args.squidhnsw_max_ef,
                            sql_batching=method_sql_batching(name, args),
                            postgresql_merge=args.postgresql_merge,
                            route_limit=args.route_limit,
                        )
                    if name in {"veda", "effveda"}:
                        if args.index_mode == "hnsw":
                            return execute_veda_hnsw_query(
                                cur,
                                query,
                                routes,
                                ef_search=args.ef_search,
                                sql_batching=method_sql_batching(name, args),
                                auth_filter=args.auth_filter,
                                prepared_cache=prepared_cache,
                                indexed_tables=indexed_tables,
                                native_filter_location=str(args.native_filter_location),
                                route_overfetch=int(args.route_overfetch),
                            )
                        return execute_veda_kernel_query(
                            cur,
                            query,
                            routes,
                            args.ef_search,
                            args.vedahnsw_max_ef,
                            sql_batching=method_sql_batching(name, args),
                            postgresql_merge=args.postgresql_merge,
                            veda_native_all_routes=bool(args.veda_native_all_routes),
                        )
                    return execute_partition_query(
                        cur,
                        query,
                        routes,
                        ef_search=args.ef_search,
                        sql_batching=method_sql_batching(name, args),
                        postgresql_merge=args.postgresql_merge,
                        auth_filter=args.auth_filter,
                    )

                for _ in range(args.warmup_rounds):
                    for query in warmup_batch:
                        execute_query(query)

                ready.wait()
                start.wait()
                worker_started = time.perf_counter()
                values: list[tuple[float, list[tuple], int, Query]] = []

                def run_timed_query(query: Query) -> None:
                    started = time.perf_counter()
                    result, route_count = execute_query(query)
                    query_elapsed = time.perf_counter() - started
                    values.append((query_elapsed, result, route_count, query))

                if duration_seconds > 0.0:
                    deadline = worker_started + duration_seconds
                    index = 0
                    if not warmup_batch:
                        return WorkerResult(values, time.perf_counter() - worker_started, 0)
                    while time.perf_counter() < deadline:
                        run_timed_query(warmup_batch[index % len(warmup_batch)])
                        index += 1
                elif scheduler == "dynamic":
                    assert work_queue is not None
                    while True:
                        try:
                            query = work_queue.get_nowait()
                        except queue.Empty:
                            break
                        try:
                            run_timed_query(query)
                        finally:
                            work_queue.task_done()
                else:
                    for query in warmup_batch:
                        run_timed_query(query)

                timed_elapsed = time.perf_counter() - worker_started
                return WorkerResult(values, timed_elapsed, len(values))

        except BaseException:
            # A worker that fails before the common start must not strand peers at the barrier.
            ready.abort()
            start.set()
            raise
        finally:
            if conn is not None:
                conn.close()

    values: list[tuple[float, list[tuple], int, Query]] = []
    worker_results: list[WorkerResult] = []
    try:
        with ThreadPoolExecutor(max_workers=len(warmup_batches)) as executor:
            futures = [executor.submit(worker, batch) for batch in warmup_batches]
            try:
                ready.wait()
            except threading.BrokenBarrierError as exc:
                for future in futures:
                    try:
                        future.result()
                    except BaseException as worker_exc:
                        raise RuntimeError(f"{name} worker failed before timed QPS execution: {worker_exc}") from worker_exc
                raise RuntimeError(f"{name} worker startup barrier broke") from exc
            started = time.perf_counter()
            start.set()
            for future in as_completed(futures):
                worker_result = future.result()
                worker_results.append(worker_result)
                values.extend(worker_result.values)
    finally:
        if route_executor is not None:
            route_executor.close()
    total_elapsed = time.perf_counter() - started
    worker_elapsed = [result.timed_elapsed for result in worker_results if result.query_count > 0]
    elapsed = max(worker_elapsed) if worker_elapsed else total_elapsed
    latencies = [item[0] for item in values]
    route_counts = [item[2] for item in values]

    recalls: list[float] = []
    result_counts: list[int] = []
    complete_results: list[float] = []
    for _, result, _, query in values:
        predicted = {(int(row[0]), int(row[1])) for row in result}
        if query.ground_truth:
            recalls.append(len(predicted & query.ground_truth) / len(query.ground_truth))
        result_counts.append(len(result))
        complete_results.append(1.0 if len(result) >= query.topk else 0.0)

    selection = selection or PlanSelection(method=name)
    return {
        "method": name,
        "memory_ratio": selection.memory_ratio,
        "registry_id": selection.registry_id,
        "plan_id": selection.plan_id,
        "table_prefix": selection.table_prefix,
        "queries": len(values),
        "unique_queries": len(queries),
        "query_repetitions": max(1, int(args.query_repetitions)),
        "duration_seconds": duration_seconds if duration_seconds > 0.0 else None,
        "scheduler": scheduler,
        "qps": len(values) / elapsed,
        "avg_latency_ms": statistics.mean(latencies) * 1000,
        "p50_latency_ms": percentile(latencies, 0.50) * 1000,
        "p95_latency_ms": percentile(latencies, 0.95) * 1000,
        "p99_latency_ms": percentile(latencies, 0.99) * 1000,
        "max_latency_ms": max(latencies) * 1000,
        "recall_at_k": statistics.mean(recalls) if recalls else None,
        "recall_evaluated": bool(recalls),
        "avg_routes": statistics.mean(route_counts),
        "p50_routes": percentile(route_counts, 0.50),
        "p95_routes": percentile(route_counts, 0.95),
        "max_routes": max(route_counts),
        "route_scans": sum(route_counts),
        "route_scans_per_second": sum(route_counts) / elapsed if elapsed > 0 else 0.0,
        "avg_results": statistics.mean(result_counts),
        "complete_result_rate": statistics.mean(complete_results),
        "wall_time_seconds": elapsed,
        "total_wall_time_seconds": total_elapsed,
        "sum_latency_seconds": sum(latencies),
        "effective_concurrency": sum(latencies) / elapsed if elapsed > 0 else 0.0,
        "worker_elapsed_min_seconds": min(worker_elapsed) if worker_elapsed else 0.0,
        "worker_elapsed_p50_seconds": percentile(worker_elapsed, 0.50) if worker_elapsed else 0.0,
        "worker_elapsed_p95_seconds": percentile(worker_elapsed, 0.95) if worker_elapsed else 0.0,
        "worker_elapsed_max_seconds": max(worker_elapsed) if worker_elapsed else 0.0,
        "worker_query_count_min": min(result.query_count for result in worker_results) if worker_results else 0,
        "worker_query_count_max": max(result.query_count for result in worker_results) if worker_results else 0,
        "ef_search": int(args.ef_search),
        "index_mode": args.index_mode,
        "auth_filter": args.auth_filter,
        "hnsw_iterative_scan": args.hnsw_iterative_scan,
        "pg_parallel_route_scan": bool(args.pg_parallel_route_scan),
        "veda_native_all_routes": bool(args.veda_native_all_routes) if name in {"veda", "effveda"} else None,
        "sql_batching": method_sql_batching(name, args),
        "prepare_routes": bool(args.prepare_routes),
        "merge_location": (
            "postgresql_function" if name == "ours" and bool(args.ours_db_function)
            else ("postgresql" if args.postgresql_merge else "python")
        ),
        "hnsw_indexed_route_tables": len(indexed_tables) if indexed_tables is not None else None,
        "squidhnsw_max_ef": int(args.squidhnsw_max_ef) if name == "ours" else None,
        "route_limit": int(args.route_limit) if name == "ours" and int(args.route_limit) > 0 else None,
        "route_parallelism": route_parallelism if name == "ours" else None,
        "route_scheduler": route_scheduler if name == "ours" else None,
        "route_overfetch": int(args.route_overfetch) if name in {"ours", "hqi", "veda", "effveda"} else None,
        "native_filter_location": str(args.native_filter_location) if name in {"ours", "hqi", "veda", "effveda"} else None,
        "route_worker_count": route_worker_count if route_executor is not None else None,
        "route_parallel_access_mode": (
            "user_access_join"
            if name == "ours" and route_executor is not None and bool(args.ours_precompute_access and args.auth_filter == "rls")
            else ("set_role_rls" if name == "ours" and route_executor is not None and args.auth_filter == "rls" else None)
        ),
        "ours_precompute_access": bool(args.ours_precompute_access) if name == "ours" else None,
        "ours_db_function": bool(args.ours_db_function) if name == "ours" else None,
        "vedahnsw_max_ef": int(args.vedahnsw_max_ef) if name in {"veda", "effveda"} else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct PostgreSQL vector-search QPS benchmark")
    parser.add_argument("--methods", nargs="+", default=["rls", "role", "honeybee", "hqi", "ours", "veda", "effveda"])
    parser.add_argument("--query-source", choices=["file", "db"], default="file",
                        help="file reads --query-file/--ground-truth-file; db samples vectors and users from the connected database for QPS-only runs")
    parser.add_argument("--query-file", type=Path, default=PROJECT_ROOT / "basic_benchmark" / "query_dataset.json")
    parser.add_argument("--ground-truth-file", type=Path, default=PROJECT_ROOT / "basic_benchmark" / "ground_truth_cache.json")
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--topk", type=int, default=10,
                        help="Top-k used with --query-source db")
    parser.add_argument("--query-repetitions", type=int, default=5,
                        help="Repeat the fixed query workload during measurement to reduce sub-second QPS noise")
    parser.add_argument("--duration-seconds", type=float, default=0.0,
                        help="If positive, run each worker for this many timed seconds instead of a fixed query count")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--scheduler", choices=["dynamic", "static"], default="dynamic",
                        help="Query scheduling during timed QPS. dynamic uses a shared worker queue; static preserves the old round-robin batches.")
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--ef-search", type=int, default=100,
                        help="HNSW ef_search, or SQUIDHNSW base_ef for --methods ours")
    parser.add_argument(
        "--index-mode",
        choices=["native", "hnsw"],
        default="hnsw",
        help="native uses SQUIDHNSW/VEDAHNSW; hnsw compares all partition methods with ordinary HNSW",
    )
    parser.add_argument("--auth-filter", choices=["rls", "native"], default="rls",
                        help="Authorization filter for partition methods. rls SET ROLEs to the query user and relies on table RLS policies; native uses each method original SQL/pattern filter.")
    parser.add_argument("--squidhnsw-max-ef", type=int, default=1000,
                        help="Adaptive SQUIDHNSW expansion cap for --methods ours")
    parser.add_argument("--route-limit", type=int, default=0,
                        help="For --methods ours only, search only the first N ordered routes; 0 means unlimited")
    parser.add_argument("--route-parallelism", type=int, default=1,
                        help="For --methods ours --index-mode hnsw, run up to N routes for one query concurrently")
    parser.add_argument("--route-worker-count", type=int, default=0,
                        help="Shared route-worker connection count for --route-parallelism; 0 derives concurrency * route_parallelism")
    parser.add_argument("--route-scheduler", choices=["inline", "pool"], default="inline",
                        help="For OURS HNSW, inline scans routes in the query worker; pool sends route scans to a fixed worker pool")
    parser.add_argument("--route-overfetch", type=int, default=1,
                        help="For OURS HNSW native SQL filtering, increase impure-route LIMIT by topk/selectivity * this factor")
    parser.add_argument("--native-filter-location", choices=["sql", "client"], default="client",
                        help="For OURS HNSW native auth, apply impure-route pattern filtering in SQL or after pure HNSW candidate fetch")
    parser.add_argument("--ours-db-function", action="store_true",
                        help="For --methods ours --index-mode hnsw, fill candidates and merge top-k inside a pg_temp function")
    parser.add_argument("--ours-precompute-access", action=argparse.BooleanOptionalAction, default=True,
                        help="For --methods ours --index-mode hnsw --auth-filter rls, precompute accessible document ids once per query and reuse them across route scans")
    parser.add_argument("--vedahnsw-max-ef", type=int, default=5000,
                        help="VEDAHNSW expansion cap for --methods veda effveda")
    parser.add_argument("--sql-batching", action=argparse.BooleanOptionalAction, default=None,
                        help="Batch route SQL where enabled. Default: enabled except HQI, which uses per-route mode.")
    parser.add_argument("--prepare-routes", action=argparse.BooleanOptionalAction, default=True,
                        help="PREPARE repeated per-route HNSW SELECT statements inside each worker connection")
    parser.add_argument("--postgresql-merge", action="store_true",
                        help="Experimental: use temporary-table SQL global top-k merge instead of the normal Python merge")
    parser.add_argument("--mode", choices=["direct", "paper-plan"], default="direct",
                        help="Kept for CLI compatibility; VEDA methods always use VEDAHNSW when selected")
    parser.add_argument("--jit", choices=["on", "off"], default="off")
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--pg-parallel-route-scan", action="store_true",
                        help="Lower PostgreSQL parallel costs and enable parallel append inside each worker session")
    parser.add_argument(
        "--hnsw-iterative-scan",
        choices=HNSW_ITERATIVE_SCAN_VALUES,
        default="off",
        help="pgvector HNSW iterative scan mode: off, relaxed_order, or strict_order",
    )
    parser.add_argument("--veda-native-all-routes", action="store_true",
                        help="For VEDA/EffVeda native mode, require and use VEDAHNSW for every route, including leftover nodes.")
    parser.add_argument("--memory-ratio", type=float, default=None,
                        help="Resolve a versioned materialization by method and memory ratio before timed QPS")
    parser.add_argument("--current-label", default=os.environ.get("CURRENT_RESULT_LABEL", "current"),
                        help="Output label used when --memory-ratio is omitted; keeps current-mode datasets separate")
    parser.add_argument("--output-root", type=Path, default=PROJECT_ROOT / "basic_benchmark" / "result" / "direct_pg_qps",
                        help="Root for structured QPS outputs when --output is not set")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.index_mode == "hnsw" and args.postgresql_merge:
        parser.error("--postgresql-merge is only available with --index-mode native")
    if args.route_parallelism < 1:
        parser.error("--route-parallelism must be >= 1")
    if args.route_worker_count < 0:
        parser.error("--route-worker-count must be >= 0")
    if args.ours_db_function and args.route_parallelism > 1:
        parser.error("--ours-db-function and --route-parallelism > 1 are separate execution modes; use one at a time")
    if args.auth_filter == "rls" and args.index_mode == "native":
        native_methods = {_normalize_method_name(method) for method in args.methods} & {"ours", "veda", "effveda"}
        if native_methods:
            parser.error("--auth-filter rls requires --index-mode hnsw for ours/veda/effveda; use --auth-filter native with --index-mode native for the custom index path")

    if args.query_source == "db":
        queries = load_db_sampled_queries(args.query_count, args.topk)
    else:
        queries = load_queries(args.query_file, args.ground_truth_file, args.query_count)
    loaders: dict[str, Callable[[PlanSelection], dict[int, tuple[Route, ...]]]] = {
        "role": lambda selection: load_role_routes(),
        "honeybee": load_honeybee_routes,
        "hqi": lambda selection: load_hqi_routes(selection, queries),
        "ours": load_ours_routes,
        "veda": lambda selection: load_veda_routes("veda", selection),
        "effveda": lambda selection: load_veda_routes("effveda", selection),
    }
    summaries = []
    for method in args.methods:
        method = _normalize_method_name(method)
        try:
            selection = resolve_versioned_plan(method, args.memory_ratio)
            routes = None if method == "rls" else loaders[method](selection)
            summary = run_method(method, queries, routes, args, selection)
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True))
        except (KeyError, RuntimeError, ValueError) as exc:
            print(
                json.dumps(
                    {
                        "method": method,
                        "skipped": str(exc),
                        "error_type": type(exc).__name__,
                    },
                    sort_keys=True,
                )
            )

    output_path = args.output
    if output_path is None:
        method_label = "_".join(_normalize_method_name(method) for method in args.methods)
        memory_label = str(args.current_label) if args.memory_ratio is None else ("memory_" + str(float(args.memory_ratio)).replace(".", "p"))
        ef_label = f"ef_{int(args.ef_search)}"
        timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
        output_path = args.output_root / method_label / memory_label / ef_label / f"{timestamp}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summaries, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
