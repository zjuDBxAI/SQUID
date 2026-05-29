from __future__ import annotations

import importlib
import math
import sys
import time
from typing import Optional

from psycopg2 import sql

from services.config import get_db_connection

from .common import TenantRoute
from .storage import load_tenant_routes


def _resolve_efconfig_module():
    for module_name in ("basic_benchmark.efconfig", "efconfig"):
        module = sys.modules.get(module_name)
        if module is not None:
            return module
    for module_name in ("basic_benchmark.efconfig", "efconfig"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    return None


def _configured_ef_min() -> int:
    efconfig = _resolve_efconfig_module()
    if efconfig is None:
        return 1
    configured = getattr(efconfig, "kmeans_ef_search", getattr(efconfig, "ef_search", None))
    if configured is None:
        return 1
    if isinstance(configured, str):
        normalized = configured.strip().lower()
        if normalized in {"", "adaptive", "auto", "none"}:
            return 1
        return max(1, int(float(normalized)))
    return max(1, int(configured))


def _adaptive_route_ef(route: TenantRoute, *, topk: int, fetch_multiplier: int, ef_min: int) -> int:
    accessible_vectors = int(getattr(route, "accessible_vector_count", 0) or 0)
    partition_vectors = int(getattr(route, "partition_vector_count", 0) or 0)
    if accessible_vectors <= 0 or partition_vectors <= 0:
        return max(1, int(ef_min))
    numerator = int(topk) * max(1, int(fetch_multiplier)) * int(partition_vectors)
    adaptive_ef = int(math.ceil(float(numerator) / float(accessible_vectors)))
    return max(max(1, int(ef_min)), int(adaptive_ef))


def _configured_int(primary_name: str, default: int, *, minimum: int = 1) -> int:
    efconfig = _resolve_efconfig_module()
    if efconfig is None or not hasattr(efconfig, primary_name):
        return max(minimum, int(default))
    return max(minimum, int(getattr(efconfig, primary_name)))


def _configured_value(primary_name: str, default):
    efconfig = _resolve_efconfig_module()
    if efconfig is None or not hasattr(efconfig, primary_name):
        return default
    return getattr(efconfig, primary_name)


def _try_set(cur, statement: str) -> None:
    try:
        cur.execute("SAVEPOINT kmeans_search_config;")
        cur.execute(statement)
        cur.execute("RELEASE SAVEPOINT kmeans_search_config;")
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT kmeans_search_config;")
        cur.execute("RELEASE SAVEPOINT kmeans_search_config;")


def _configure_search_session(cur) -> None:
    cur.execute("SET max_parallel_workers_per_gather = 0;")
    cur.execute("SET jit = off;")
    iterative_scan = str(_configured_value("kmeans_hnsw_iterative_scan", "relaxed_order")).strip().lower()
    if iterative_scan and iterative_scan not in {"0", "false", "off", "none"}:
        _try_set(cur, f"SET hnsw.iterative_scan = {iterative_scan};")
    max_scan_tuples = _configured_value("kmeans_hnsw_max_scan_tuples", 20000)
    if max_scan_tuples is not None:
        _try_set(cur, f"SET hnsw.max_scan_tuples = {max(1, int(max_scan_tuples))};")


def _merge_results(all_results, topk: int):
    seen = set()
    unique_results = []
    all_results.sort(key=lambda row: row[3])
    for row in all_results:
        key = (row[1], row[0])
        if key in seen:
            continue
        seen.add(key)
        unique_results.append(row)
        if len(unique_results) == int(topk):
            break
    return unique_results


def _build_partition_query(
    route: TenantRoute,
    *,
    query_vector,
    topk: int,
    fetch_multiplier: int,
):
    fetch_limit = max(int(topk), int(topk) * max(1, int(fetch_multiplier)))
    return (
        sql.SQL(
            """
            SELECT block_id, document_id, block_content, distance
            FROM (
                SELECT p.block_id, p.document_id, p.block_content,
                       p.vector <-> %s::vector AS distance
                FROM {} p
                WHERE p.pattern_id = ANY(%s)
                ORDER BY distance
                LIMIT %s
            ) routed
            ORDER BY distance
            LIMIT %s;
            """
        ).format(sql.Identifier(route.table_name)),
        [query_vector, list(int(pattern_id) for pattern_id in route.pattern_ids), int(fetch_limit), int(topk)],
    )


def kmeans_partition_search(user_id: int, query_vector, topk: int = 5, statistics_type: str = "sql"):
    if statistics_type == "sql":
        return kmeans_partition_search_statistics_sql(user_id, query_vector, topk)
    if statistics_type == "system":
        return kmeans_partition_search_statistics_system(user_id, query_vector, topk)
    raise ValueError(f"Unknown statistics type: {statistics_type}")


def kmeans_partition_search_statistics_sql(user_id: int, query_vector, topk: int = 5):
    routes = [route for route in load_tenant_routes(int(user_id)) if route.pattern_ids]
    if not routes:
        return [], 0.0
    fetch_multiplier = _configured_int("kmeans_partition_fetch_multiplier", 1)
    conn = get_db_connection()
    total_query_time = 0.0
    all_results = []
    try:
        with conn.cursor() as cur:
            _configure_search_session(cur)
            ef_min = _configured_ef_min()
            for route in routes:
                route_ef = _adaptive_route_ef(
                    route,
                    topk=int(topk),
                    fetch_multiplier=int(fetch_multiplier),
                    ef_min=int(ef_min),
                )
                _try_set(cur, f"SET hnsw.ef_search = {int(route_ef)};")
                query, params = _build_partition_query(
                    route,
                    query_vector=query_vector,
                    topk=int(topk),
                    fetch_multiplier=int(fetch_multiplier),
                )
                cur.execute(sql.SQL("EXPLAIN ANALYZE {}").format(query), params)
                for (line,) in cur.fetchall():
                    if "Execution Time" in line:
                        total_query_time += float(line.split()[-2]) / 1000.0
                cur.execute(query, params)
                all_results.extend(cur.fetchall())
    finally:
        conn.close()
    return _merge_results(all_results, int(topk)), total_query_time


def kmeans_partition_search_statistics_system(user_id: int, query_vector, topk: int = 5):
    started_at = time.time()
    routes = [route for route in load_tenant_routes(int(user_id)) if route.pattern_ids]
    if not routes:
        return [], 0.0
    fetch_multiplier = _configured_int("kmeans_partition_fetch_multiplier", 1)
    conn = get_db_connection()
    all_results = []
    try:
        with conn.cursor() as cur:
            _configure_search_session(cur)
            ef_min = _configured_ef_min()
            for route in routes:
                route_ef = _adaptive_route_ef(
                    route,
                    topk=int(topk),
                    fetch_multiplier=int(fetch_multiplier),
                    ef_min=int(ef_min),
                )
                _try_set(cur, f"SET hnsw.ef_search = {int(route_ef)};")
                query, params = _build_partition_query(
                    route,
                    query_vector=query_vector,
                    topk=int(topk),
                    fetch_multiplier=int(fetch_multiplier),
                )
                cur.execute(query, params)
                all_results.extend(cur.fetchall())
    finally:
        conn.close()
    return _merge_results(all_results, int(topk)), time.time() - started_at
