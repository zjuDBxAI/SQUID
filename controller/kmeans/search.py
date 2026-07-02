from __future__ import annotations

import importlib
import math
import sys
import time

from psycopg2 import sql

from services.config import get_db_connection

from .common import TenantRoute
from .storage import load_tenant_routes


_HNSW_EF_SEARCH_MAX = 5000


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


def _configured_index_type() -> str:
    efconfig = _resolve_efconfig_module()
    if efconfig is None:
        return "squidhnsw"
    return str(getattr(efconfig, "kmeans_index_type", "squidhnsw") or "squidhnsw").strip().lower()


def _clamp_ef(value: int, route: TenantRoute | None = None) -> int:
    ef = max(1, int(value))
    if route is not None:
        partition_vectors = int(getattr(route, "partition_vector_count", 0) or 0)
        if partition_vectors > 0:
            ef = min(ef, partition_vectors)
    return min(ef, _HNSW_EF_SEARCH_MAX)


def _try_set(cur, statement: str) -> None:
    try:
        cur.execute("SAVEPOINT kmeans_search_config;")
        cur.execute(statement)
        cur.execute("RELEASE SAVEPOINT kmeans_search_config;")
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT kmeans_search_config;")
        cur.execute("RELEASE SAVEPOINT kmeans_search_config;")


def _set_if_supported(cur, statement: str) -> bool:
    try:
        cur.execute("SAVEPOINT kmeans_search_config;")
        cur.execute(statement)
        cur.execute("RELEASE SAVEPOINT kmeans_search_config;")
        return True
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT kmeans_search_config;")
        cur.execute("RELEASE SAVEPOINT kmeans_search_config;")
        return False


def _configure_search_session(cur) -> None:
    cur.execute("SET max_parallel_workers_per_gather = 0;")
    cur.execute("SET jit = off;")


def _result_key(row) -> tuple[int, int]:
    return (int(row[1]), int(row[0]))


def _global_bound(results, topk: int) -> float:
    if len(results) < int(topk):
        return float("inf")
    seen = set()
    unique = []
    for row in sorted(results, key=lambda item: item[3]):
        key = _result_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique.append(row)
        if len(unique) == int(topk):
            return float(unique[-1][3])
    return float("inf")


def _merge_results(all_results, topk: int):
    seen = set()
    unique_results = []
    all_results.sort(key=lambda row: row[3])
    for row in all_results:
        key = _result_key(row)
        if key in seen:
            continue
        seen.add(key)
        unique_results.append(row)
        if len(unique_results) == int(topk):
            break
    return unique_results


def _base_ef(*, topk: int, ef_min: int) -> int:
    return _clamp_ef(max(1, int(ef_min), int(topk)))


def _allowed_pattern_ids(route: TenantRoute) -> set[int]:
    return {int(pattern_id) for pattern_id in route.pattern_ids}


def _metadata_selectivity_ef(route: TenantRoute, *, base_ef: int) -> int:
    accessible_vectors = int(getattr(route, "accessible_vector_count", 0) or 0)
    partition_vectors = int(getattr(route, "partition_vector_count", 0) or 0)
    if accessible_vectors <= 0 or partition_vectors <= 0:
        return _clamp_ef(base_ef, route)
    expanded = int(math.ceil(float(base_ef) * float(partition_vectors) / float(accessible_vectors)))
    return _clamp_ef(max(int(base_ef), expanded), route)


def _probe_adaptive_ef(route: TenantRoute, *, probe_rows, base_ef: int) -> int:
    if not probe_rows:
        return _metadata_selectivity_ef(route, base_ef=int(base_ef))

    allowed_patterns = _allowed_pattern_ids(route)
    hit_count = 0
    for row in probe_rows:
        if len(row) >= 5 and int(row[3]) in allowed_patterns:
            hit_count += 1

    if hit_count <= 0:
        return _metadata_selectivity_ef(route, base_ef=int(base_ef))

    candidate_count = max(1, len(probe_rows))
    expanded = int(math.ceil(float(base_ef) * float(candidate_count) / float(hit_count)))
    return _clamp_ef(max(int(base_ef), expanded), route)


def _route_selectivity(route: TenantRoute) -> float:
    accessible_vectors = int(getattr(route, "accessible_vector_count", 0) or 0)
    partition_vectors = int(getattr(route, "partition_vector_count", 0) or 0)
    if accessible_vectors <= 0 or partition_vectors <= 0:
        return 1.0
    return min(1.0, max(0.000001, float(accessible_vectors) / float(partition_vectors)))


def _allowed_patterns_csv(route: TenantRoute) -> str:
    return ",".join(str(int(pattern_id)) for pattern_id in route.pattern_ids)


def _build_authorized_query(
    route: TenantRoute,
    *,
    query_vector,
    limit: int,
):
    return (
        sql.SQL(
            """
            SELECT block_id, document_id, block_content, distance
            FROM (
                SELECT p.block_id, p.document_id, p.block_content,
                       p.vector <-> %s::vector AS distance
                FROM {} p
                WHERE p.pattern_id = ANY(%s)
                ORDER BY p.vector <-> %s::vector
                LIMIT %s
            ) candidates
            ORDER BY distance;
            """
        ).format(sql.Identifier(route.table_name)),
        [query_vector, list(int(pattern_id) for pattern_id in route.pattern_ids), query_vector, int(limit)],
    )


def _configure_squidhnsw_route(cur, route: TenantRoute, *, base_ef: int, max_ef: int, global_bound: float) -> bool:
    supported = _set_if_supported(cur, f"SET squidhnsw.base_ef = {int(base_ef)};")
    supported = _set_if_supported(cur, f"SET squidhnsw.max_ef = {int(max_ef)};") and supported
    supported = _set_if_supported(cur, f"SET squidhnsw.route_selectivity = {_route_selectivity(route):.12f};") and supported
    bound = float(global_bound)
    bound_sql = "1e308" if math.isinf(bound) else repr(max(0.0, bound))
    supported = _set_if_supported(cur, f"SET squidhnsw.global_bound = {bound_sql};") and supported
    supported = _set_if_supported(cur, sql.SQL("SET squidhnsw.allowed_patterns = {};").format(sql.Literal(_allowed_patterns_csv(route))).as_string(cur)) and supported
    return supported


def _execute_authorized_search(cur, route: TenantRoute, *, query_vector, topk: int, base_ef: int, max_ef: int, global_bound: float):
    _configure_squidhnsw_route(cur, route, base_ef=int(base_ef), max_ef=int(max_ef), global_bound=float(global_bound))
    query, params = _build_authorized_query(route, query_vector=query_vector, limit=int(topk))
    cur.execute(query, params)
    return cur.fetchall()


def _explain_authorized_search_time(cur, route: TenantRoute, *, query_vector, topk: int, base_ef: int, max_ef: int, global_bound: float) -> float:
    _configure_squidhnsw_route(cur, route, base_ef=int(base_ef), max_ef=int(max_ef), global_bound=float(global_bound))
    query, params = _build_authorized_query(route, query_vector=query_vector, limit=int(topk))
    cur.execute(sql.SQL("EXPLAIN ANALYZE {}").format(query), params)
    return _extract_execution_time_seconds(cur.fetchall())


def _search_squidhnsw_route(
    cur,
    route: TenantRoute,
    *,
    query_vector,
    topk: int,
    ef_min: int,
    collect_sql_time: bool,
    global_bound: float,
):
    base_ef = _base_ef(topk=int(topk), ef_min=int(ef_min))
    max_ef = _clamp_ef(_HNSW_EF_SEARCH_MAX, route)
    if collect_sql_time:
        elapsed = _explain_authorized_search_time(
            cur,
            route,
            query_vector=query_vector,
            topk=int(topk),
            base_ef=int(base_ef),
            max_ef=int(max_ef),
            global_bound=float(global_bound),
        )
        rows = _execute_authorized_search(
            cur,
            route,
            query_vector=query_vector,
            topk=int(topk),
            base_ef=int(base_ef),
            max_ef=int(max_ef),
            global_bound=float(global_bound),
        )
    else:
        started = time.perf_counter()
        rows = _execute_authorized_search(
            cur,
            route,
            query_vector=query_vector,
            topk=int(topk),
            base_ef=int(base_ef),
            max_ef=int(max_ef),
            global_bound=float(global_bound),
        )
        elapsed = time.perf_counter() - started
    return rows, float(elapsed)


def _build_candidate_query(
    route: TenantRoute,
    *,
    query_vector,
    candidate_limit: int,
):
    return (
        sql.SQL(
            """
            SELECT block_id, document_id, block_content, pattern_id, distance
            FROM (
                SELECT p.block_id, p.document_id, p.block_content, p.pattern_id,
                       p.vector <-> %s::vector AS distance
                FROM {} p
                ORDER BY p.vector <-> %s::vector
                LIMIT %s
            ) candidates
            ORDER BY distance;
            """
        ).format(sql.Identifier(route.table_name)),
        [query_vector, query_vector, int(candidate_limit)],
    )


def _authorized_candidate_results(candidate_rows, route: TenantRoute):
    allowed_patterns = _allowed_pattern_ids(route)
    authorized = []
    for row in candidate_rows:
        if len(row) < 5:
            continue
        block_id, document_id, block_content, pattern_id, distance = row
        if int(pattern_id) in allowed_patterns:
            authorized.append((block_id, document_id, block_content, distance))
    return authorized


def _extract_execution_time_seconds(explain_rows) -> float:
    total = 0.0
    for (line,) in explain_rows:
        if "Execution Time" in line:
            total += float(line.split()[-2]) / 1000.0
    return total


def _execute_candidate_search(cur, route: TenantRoute, *, query_vector, ef_search: int):
    ef_search = _clamp_ef(int(ef_search), route)
    _try_set(cur, f"SET hnsw.ef_search = {int(ef_search)};")
    query, params = _build_candidate_query(route, query_vector=query_vector, candidate_limit=int(ef_search))
    cur.execute(query, params)
    return cur.fetchall()


def _explain_candidate_search_time(cur, route: TenantRoute, *, query_vector, ef_search: int) -> float:
    ef_search = _clamp_ef(int(ef_search), route)
    _try_set(cur, f"SET hnsw.ef_search = {int(ef_search)};")
    query, params = _build_candidate_query(route, query_vector=query_vector, candidate_limit=int(ef_search))
    cur.execute(sql.SQL("EXPLAIN ANALYZE {}").format(query), params)
    return _extract_execution_time_seconds(cur.fetchall())


def _probe_then_search_route(
    cur,
    route: TenantRoute,
    *,
    query_vector,
    topk: int,
    ef_min: int,
    collect_sql_time: bool,
):
    base_ef = _base_ef(topk=int(topk), ef_min=int(ef_min))
    probe_rows = _execute_candidate_search(
        cur,
        route,
        query_vector=query_vector,
        ef_search=int(base_ef),
    )
    final_ef = _probe_adaptive_ef(route, probe_rows=probe_rows, base_ef=int(base_ef))

    if collect_sql_time:
        elapsed = _explain_candidate_search_time(
            cur,
            route,
            query_vector=query_vector,
            ef_search=int(final_ef),
        )
        final_rows = _execute_candidate_search(
            cur,
            route,
            query_vector=query_vector,
            ef_search=int(final_ef),
        )
    else:
        started = time.perf_counter()
        final_rows = _execute_candidate_search(
            cur,
            route,
            query_vector=query_vector,
            ef_search=int(final_ef),
        )
        elapsed = time.perf_counter() - started

    return _authorized_candidate_results(final_rows, route), float(elapsed)


def kmeans_partition_search(user_id: int, query_vector, topk: int = 5, statistics_type: str = "sql"):
    if statistics_type == "sql":
        return kmeans_partition_search_statistics_sql(user_id, query_vector, topk)
    if statistics_type == "system":
        return kmeans_partition_search_statistics_system(user_id, query_vector, topk)
    raise ValueError(f"Unknown statistics type: {statistics_type}")


def _kmeans_partition_search_impl(user_id: int, query_vector, topk: int, *, collect_sql_time: bool):
    routes = [route for route in load_tenant_routes(int(user_id)) if route.pattern_ids]
    if not routes:
        return [], 0.0

    conn = get_db_connection()
    total_query_time = 0.0
    all_results = []
    try:
        with conn.cursor() as cur:
            _configure_search_session(cur)
            ef_min = _configured_ef_min()
            index_type = _configured_index_type()
            for route in routes:
                if index_type == "squidhnsw":
                    route_results, route_time = _search_squidhnsw_route(
                        cur,
                        route,
                        query_vector=query_vector,
                        topk=int(topk),
                        ef_min=int(ef_min),
                        collect_sql_time=bool(collect_sql_time),
                        global_bound=_global_bound(all_results, int(topk)),
                    )
                else:
                    route_results, route_time = _probe_then_search_route(
                        cur,
                        route,
                        query_vector=query_vector,
                        topk=int(topk),
                        ef_min=int(ef_min),
                        collect_sql_time=bool(collect_sql_time),
                    )
                total_query_time += float(route_time)
                all_results.extend(route_results)
    finally:
        conn.close()
    return _merge_results(all_results, int(topk)), float(total_query_time)


def kmeans_partition_search_statistics_sql(user_id: int, query_vector, topk: int = 5):
    return _kmeans_partition_search_impl(int(user_id), query_vector, int(topk), collect_sql_time=True)


def kmeans_partition_search_statistics_system(user_id: int, query_vector, topk: int = 5):
    return _kmeans_partition_search_impl(int(user_id), query_vector, int(topk), collect_sql_time=False)
