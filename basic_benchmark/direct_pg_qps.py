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
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from psycopg2 import sql


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from services.config import get_db_connection  # noqa: E402


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


@dataclass(frozen=True)
class Query:
    user_id: int
    vector: str
    topk: int
    ground_truth: frozenset[tuple[int, int]]


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


def load_ours_routes() -> dict[int, tuple[Route, ...]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if not table_exists(cur, "kmeans_current_plan") or not table_exists(cur, "kmeans_current_routes"):
                raise RuntimeError("OURS metadata is not materialized")
            cur.execute("SELECT plan_id FROM kmeans_current_plan ORDER BY plan_id DESC LIMIT 1")
            plan = cur.fetchone()
            if plan is None:
                raise RuntimeError("OURS has no active plan")
            cur.execute(
                """
                SELECT
                    r.tenant_id,
                    r.table_name,
                    r.pattern_ids,
                    p.vector_count,
                    COALESCE(SUM(ap.vector_count), 0)::BIGINT AS visible_vectors,
                    r.cluster_id,
                    r.partition_id
                FROM kmeans_current_routes r
                JOIN kmeans_current_partitions p
                  ON p.plan_id = r.plan_id
                 AND p.partition_id = r.partition_id
                LEFT JOIN kmeans_current_patterns ap
                  ON ap.plan_id = r.plan_id
                 AND ap.pattern_id = ANY(r.pattern_ids)
                WHERE r.plan_id = %s
                GROUP BY r.tenant_id, r.table_name, r.pattern_ids, p.vector_count,
                         r.route_kind, r.cluster_id, r.partition_id
                ORDER BY r.tenant_id, r.route_kind, r.cluster_id, r.partition_id
                """,
                [int(plan[0])],
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
            return {user_id: tuple(routes) for user_id, routes in loaded.items()}
    finally:
        conn.close()


def load_veda_routes(algorithm: str) -> dict[int, tuple[Route, ...]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if not table_exists(cur, "veda_current_plan") or not table_exists(cur, "veda_current_user_routes"):
                raise RuntimeError(f"{algorithm} metadata is not materialized")
            cur.execute("SELECT plan_id FROM veda_current_plan WHERE algorithm = %s ORDER BY plan_id DESC LIMIT 1", [algorithm])
            plan = cur.fetchone()
            if plan is None:
                raise RuntimeError(f"No active {algorithm} plan")
            cur.execute(
                """
                SELECT user_id, table_name, route_kind, pattern_ids, impurity_factor,
                       node_vector_count, accessible_vector_count, node_id
                FROM veda_current_user_routes
                WHERE plan_id = %s
                ORDER BY user_id, route_kind, node_id
                """,
                [int(plan[0])],
            )
            loaded: dict[int, list[Route]] = {}
            for user_id, table_name, route_kind, pattern_ids, impurity_factor, node_vectors, accessible_vectors, node_id in cur.fetchall():
                loaded.setdefault(int(user_id), []).append(
                    Route(
                        str(table_name),
                        tuple(int(value) for value in (pattern_ids or ())),
                        False,
                        str(route_kind),
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


def load_honeybee_routes() -> dict[int, tuple[Route, ...]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if not table_exists(cur, "combrolepartitions"):
                raise RuntimeError("Honeybee CombRolePartitions is unavailable")
            cur.execute("SELECT user_id, array_agg(role_id ORDER BY role_id) FROM userroles GROUP BY user_id")
            role_sets = cur.fetchall()
            loaded: dict[int, tuple[Route, ...]] = {}
            for user_id, roles in role_sets:
                cur.execute("SELECT partition_id FROM combrolepartitions WHERE comb_role = %s::integer[]", [list(roles)])
                loaded[int(user_id)] = tuple(Route(f"documentblocks_partition_{int(row[0])}", (), True) for row in cur.fetchall())
            return loaded
    finally:
        conn.close()


def configure_session(cur, ef_search: int, jit: str, parallel_workers: int) -> None:
    cur.execute("LOAD 'vector'")
    cur.execute(f"SET jit = {jit}")
    cur.execute(f"SET max_parallel_workers_per_gather = {int(parallel_workers)}")
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


def _hnsw_settings(ef_search: int) -> tuple[tuple[str, str], ...]:
    return (("hnsw.ef_search", str(max(1, int(ef_search)))),)


def execute_partition_query(
    cur,
    query: Query,
    routes: dict[int, tuple[Route, ...]],
    *,
    ef_search: int,
    sql_batching: bool,
) -> tuple[list[tuple], int]:
    """Run each routed HNSW partition and merge its candidates globally."""
    used_routes = routes.get(query.user_id, ())
    if sql_batching:
        if not used_routes:
            return [], 0
        route_queries = [
            sql.SQL("(")
            + _route_candidate_select(route, use_sql_filter=not route.pure)
            + sql.SQL(")")
            for route in used_routes
        ]
        params: list[object] = []
        for route in used_routes:
            params.extend(_route_candidate_params(route, query, use_sql_filter=not route.pure))
        statement = (
            _settings_prefix(_hnsw_settings(ef_search))
            + sql.SQL("WITH route_candidates AS MATERIALIZED (")
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
        candidates.extend(
            _execute_ann_route(
                cur,
                route,
                query_vector=query.vector,
                topk=query.topk,
                use_sql_filter=not route.pure,
            )
        )
    return merge_topk(candidates, query.topk), len(used_routes)


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
) -> tuple[list[tuple], int]:
    """Use SQUIDHNSW filtering and merge all route candidates in PostgreSQL."""
    used_routes = _ordered_ours_routes(routes.get(query.user_id, ()))
    if sql_batching:
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
) -> tuple[list[tuple], int]:
    """Run VEDA/EffVeda routes and merge all candidates in PostgreSQL."""
    used_routes = routes.get(query.user_id, ())
    leftovers = [route for route in used_routes if route.route_kind == "leftover"]
    pure_indices = [route for route in used_routes if route.route_kind == "index"]
    impure_indices = sorted(
        (route for route in used_routes if route.route_kind == "impure_index"),
        key=lambda route: route.impurity_factor,
    )

    if sql_batching:
        statement = sql.SQL("TRUNCATE pg_temp.direct_pg_qps_candidates; ")
        params: list[object] = []
        for route in leftovers:
            statement += _settings_prefix(_hnsw_settings(base_ef))
            statement += _candidate_insert(route, use_sql_filter=True)
            params.extend(_route_candidate_params(route, query, use_sql_filter=True))

        kernel_routes = [(route, int(base_ef)) for route in pure_indices]
        kernel_routes.extend(
            (
                route,
                max(int(base_ef), int(math.ceil(float(route.impurity_factor) * float(base_ef)))),
            )
            for route in impure_indices
        )
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
        _configure_vedahnsw_route(
            cur,
            route,
            base_ef=base_ef,
            max_ef=min(max(1, int(max_ef_cap)), max(int(base_ef), int(route_max_ef))),
            topk=query.topk,
            global_bound=_paper_global_bound(candidates, query.topk),
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

    for route in leftovers:
        candidates.extend(
            _execute_ann_route(
                cur,
                route,
                query_vector=query.vector,
                topk=query.topk,
                use_sql_filter=True,
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


def _preview(values: list[str], limit: int = 5) -> str:
    preview = ", ".join(values[:limit])
    return preview if len(values) <= limit else f"{preview}, ... (+{len(values) - limit})"


def validate_method_prerequisites(name: str, queries: list[Query], routes: dict[int, tuple[Route, ...]] | None) -> None:
    """Fail before worker startup when a baseline has not been materialized."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            if name == "rls":
                cur.execute("SELECT relrowsecurity FROM pg_class WHERE oid = 'documentblocks'::regclass")
                row = cur.fetchone()
                rls_enabled = bool(row and row[0])
                cur.execute("SELECT EXISTS (SELECT 1 FROM pg_policy WHERE polrelid = 'documentblocks'::regclass)")
                has_policy = bool(cur.fetchone()[0])
                sample_user = str(queries[0].user_id) if queries else ""
                cur.execute("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = %s)", [sample_user])
                has_user_role = bool(cur.fetchone()[0])
                cur.execute("SELECT has_table_privilege(%s, 'documentblocks', 'SELECT')", [sample_user])
                has_select = bool(cur.fetchone()[0])
                cur.execute("SELECT pg_has_role(current_user, %s, 'MEMBER')", [sample_user])
                can_set_role = bool(cur.fetchone()[0])
                if not (rls_enabled and has_policy and has_user_role and has_select and can_set_role):
                    raise RuntimeError(
                        "RLS is not ready for this direct harness "
                        f"(enabled={rls_enabled}, policy={has_policy}, sample_role={has_user_role}, "
                        f"select={has_select}, benchmark_user_can_set_role={can_set_role}). "
                        "Initialize RLS first; this connection-reuse harness also requires the benchmark login "
                        "to be a member of each user role, or it must use per-user database connections like the original baseline."
                    )
            else:
                requested_users = {int(query.user_id) for query in queries}
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
                    user_ids = sorted(str(user_id) for user_id in requested_users)
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
                        [user_ids],
                    )
                    unavailable_users = [str(row[0]) for row in cur.fetchall()]
                    if unavailable_users:
                        raise RuntimeError(
                            "HONEYBEE direct QPS cannot switch to query users: "
                            f"{_preview(unavailable_users)}. Initialize user database roles and grants first."
                        )
                    missing_indexes = _tables_without_index_am(cur, table_names, "hnsw")
                    if missing_indexes:
                        raise RuntimeError(
                            "HONEYBEE partitions are missing HNSW indexes: "
                            f"{_preview(missing_indexes)}. Rebuild HONEYBEE before direct QPS."
                        )
                elif name == "ours":
                    missing_indexes = _tables_without_index_am(cur, table_names, "squidhnsw")
                    if missing_indexes:
                        raise RuntimeError(
                            "SQUID partitions are missing SQUIDHNSW indexes: "
                            f"{_preview(missing_indexes)}. Rebuild SQUID before direct QPS."
                        )
                elif name in {"veda", "effveda"}:
                    kernel_tables = {
                        route.table_name for route in required_routes
                        if route.route_kind in {"index", "impure_index"}
                    }
                    missing_indexes = _tables_without_index_am(cur, kernel_tables, "vedahnsw")
                    if missing_indexes:
                        raise RuntimeError(
                            f"{name} index routes are missing VEDAHNSW indexes: {_preview(missing_indexes)}. "
                            "Rebuild the VEDA baseline before direct QPS."
                        )
    finally:
        conn.close()


def _execute_as_user(cur, user_id: int, callback):
    cur.execute(sql.SQL("SET ROLE {}").format(sql.Identifier(str(int(user_id)))))
    try:
        return callback()
    finally:
        cur.execute("RESET ROLE")


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


def execute_honeybee_query(
    cur,
    query: Query,
    routes: dict[int, tuple[Route, ...]],
    *,
    ef_search: int,
    sql_batching: bool,
) -> tuple[list[tuple], int]:
    # Dynamic-partition tables may carry RLS policies, so run as the query user.
    return _execute_as_user(
        cur,
        query.user_id,
        lambda: execute_partition_query(
            cur, query, routes, ef_search=ef_search, sql_batching=sql_batching
        ),
    )


def run_method(name: str, queries: list[Query], routes: dict[int, tuple[Route, ...]] | None, args: argparse.Namespace) -> dict[str, float]:
    if name != "rls" and routes is None:
        raise RuntimeError(f"{name} has no routes")
    validate_method_prerequisites(name, queries, routes)
    measured_queries = queries * max(1, int(args.query_repetitions))
    batches = round_robin_batches(measured_queries, args.concurrency)
    ready = threading.Barrier(len(batches) + 1)
    start = threading.Event()

    def worker(batch: list[Query]) -> list[tuple[float, list[tuple], int, Query]]:
        conn = None
        try:
            conn = get_db_connection()
            with conn.cursor() as cur:
                configure_session(cur, args.ef_search, args.jit, args.parallel_workers)
                if name == "ours":
                    # Match the SQUID search path, which forces its custom index only.
                    cur.execute("SET enable_seqscan = off")
                    cur.execute("SET enable_bitmapscan = off")
                if args.sql_batching and name in {"ours", "veda", "effveda"}:
                    cur.execute(
                        "CREATE TEMP TABLE IF NOT EXISTS direct_pg_qps_candidates ("
                        "block_id BIGINT NOT NULL, document_id BIGINT NOT NULL, "
                        "distance DOUBLE PRECISION NOT NULL"
                        ") ON COMMIT PRESERVE ROWS"
                    )
                for _ in range(args.warmup_rounds):
                    for query in batch:
                        if name == "rls":
                            execute_rls_query(cur, query, ef_search=args.ef_search, sql_batching=args.sql_batching)
                        elif name == "honeybee":
                            execute_honeybee_query(cur, query, routes, ef_search=args.ef_search, sql_batching=args.sql_batching)
                        elif name == "ours":
                            execute_ours_kernel_query(cur, query, routes, args.ef_search, args.squidhnsw_max_ef, sql_batching=args.sql_batching)
                        elif name in {"veda", "effveda"}:
                            execute_veda_kernel_query(cur, query, routes, args.ef_search, args.vedahnsw_max_ef, sql_batching=args.sql_batching)
                        else:
                            execute_partition_query(cur, query, routes, ef_search=args.ef_search, sql_batching=args.sql_batching)

                ready.wait()
                start.wait()
                values: list[tuple[float, list[tuple], int, Query]] = []
                for query in batch:
                    started = time.perf_counter()
                    if name == "rls":
                        result, route_count = execute_rls_query(cur, query, ef_search=args.ef_search, sql_batching=args.sql_batching)
                    elif name == "honeybee":
                        result, route_count = execute_honeybee_query(cur, query, routes, ef_search=args.ef_search, sql_batching=args.sql_batching)
                    elif name == "ours":
                        result, route_count = execute_ours_kernel_query(cur, query, routes, args.ef_search, args.squidhnsw_max_ef, sql_batching=args.sql_batching)
                    elif name in {"veda", "effveda"}:
                        result, route_count = execute_veda_kernel_query(cur, query, routes, args.ef_search, args.vedahnsw_max_ef, sql_batching=args.sql_batching)
                    else:
                        result, route_count = execute_partition_query(cur, query, routes, ef_search=args.ef_search, sql_batching=args.sql_batching)
                    query_elapsed = time.perf_counter() - started
                    values.append((query_elapsed, result, route_count, query))
                return values
        except BaseException:
            # A worker that fails before the common start must not strand peers at the barrier.
            ready.abort()
            start.set()
            raise
        finally:
            if conn is not None:
                conn.close()

    values: list[tuple[float, list[tuple], int, Query]] = []
    with ThreadPoolExecutor(max_workers=len(batches)) as executor:
        futures = [executor.submit(worker, batch) for batch in batches]
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
            values.extend(future.result())
    elapsed = time.perf_counter() - started
    latencies = [item[0] for item in values]

    recalls: list[float] = []
    result_counts: list[int] = []
    complete_results: list[float] = []
    for _, result, _, query in values:
        predicted = {(int(row[0]), int(row[1])) for row in result}
        recalls.append(
            len(predicted & query.ground_truth) / len(query.ground_truth)
            if query.ground_truth
            else 0.0
        )
        result_counts.append(len(result))
        complete_results.append(1.0 if len(result) >= query.topk else 0.0)

    return {
        "method": name,
        "queries": len(values),
        "unique_queries": len(queries),
        "query_repetitions": max(1, int(args.query_repetitions)),
        "qps": len(values) / elapsed,
        "avg_latency_ms": statistics.mean(latencies) * 1000,
        "p50_latency_ms": percentile(latencies, 0.50) * 1000,
        "p95_latency_ms": percentile(latencies, 0.95) * 1000,
        "p99_latency_ms": percentile(latencies, 0.99) * 1000,
        "recall_at_k": statistics.mean(recalls),
        "avg_routes": statistics.mean(item[2] for item in values),
        "avg_results": statistics.mean(result_counts),
        "complete_result_rate": statistics.mean(complete_results),
        "wall_time_seconds": elapsed,
        "ef_search": int(args.ef_search),
        "sql_batching": bool(args.sql_batching),
        "merge_location": "postgresql" if args.sql_batching else "python",
        "squidhnsw_max_ef": int(args.squidhnsw_max_ef) if name == "ours" else None,
        "vedahnsw_max_ef": int(args.vedahnsw_max_ef) if name in {"veda", "effveda"} else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Direct PostgreSQL vector-search QPS benchmark")
    parser.add_argument("--methods", nargs="+", default=["rls", "role", "honeybee", "ours", "veda", "effveda"])
    parser.add_argument("--query-file", type=Path, default=PROJECT_ROOT / "basic_benchmark" / "query_dataset.json")
    parser.add_argument("--ground-truth-file", type=Path, default=PROJECT_ROOT / "basic_benchmark" / "ground_truth_cache.json")
    parser.add_argument("--query-count", type=int, default=200)
    parser.add_argument("--query-repetitions", type=int, default=5,
                        help="Repeat the fixed query workload during measurement to reduce sub-second QPS noise")
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--warmup-rounds", type=int, default=1)
    parser.add_argument("--ef-search", type=int, default=100,
                        help="HNSW ef_search, or SQUIDHNSW base_ef for --methods ours")
    parser.add_argument("--squidhnsw-max-ef", type=int, default=1000,
                        help="Adaptive SQUIDHNSW expansion cap for --methods ours")
    parser.add_argument("--vedahnsw-max-ef", type=int, default=5000,
                        help="VEDAHNSW expansion cap for --methods veda effveda")
    parser.add_argument("--sql-batching", action=argparse.BooleanOptionalAction, default=True,
                        help="Batch route searches and merge global top-k in PostgreSQL (default: enabled)")
    parser.add_argument("--mode", choices=["direct", "paper-plan"], default="direct",
                        help="Kept for CLI compatibility; VEDA methods always use VEDAHNSW when selected")
    parser.add_argument("--jit", choices=["on", "off"], default="off")
    parser.add_argument("--parallel-workers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    queries = load_queries(args.query_file, args.ground_truth_file, args.query_count)
    loaders: dict[str, Callable[[], dict[int, tuple[Route, ...]]]] = {
        "role": load_role_routes,
        "honeybee": load_honeybee_routes,
        "ours": load_ours_routes,
        "veda": lambda: load_veda_routes("veda"),
        "effveda": lambda: load_veda_routes("effveda"),
    }
    summaries = []
    for method in args.methods:
        method = method.lower()
        try:
            routes = None if method == "rls" else loaders[method]()
            summary = run_method(method, queries, routes, args)
            summaries.append(summary)
            print(json.dumps(summary, sort_keys=True))
        except (KeyError, RuntimeError) as exc:
            print(json.dumps({"method": method, "skipped": str(exc)}, sort_keys=True))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(summaries, indent=2) + "\n")


if __name__ == "__main__":
    main()
