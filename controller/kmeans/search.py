from __future__ import annotations

import importlib
import math
import sys
import time

from psycopg2 import sql

from services.config import get_db_connection

from .common import PATTERN_TABLE, TenantRoute, UPDATE_TOMBSTONE_TABLE
from .cost_model import DEFAULT_COST_MODEL, required_base_ef_star
from .storage import get_current_plan_summary, load_tenant_routes
from .acorn import search_kmeans_acorn


_HNSW_EF_SEARCH_FALLBACK_MAX = int(DEFAULT_COST_MODEL.hnsw_max_ef_search)
_FULL_TABLE_RECALL_EF_CAP: int | None = None
_AVERAGE_GLOBAL_SELECTIVITY: float | None = None
_SQUIDHNSW_LINEAR_SCAN_MAX_VECTORS = 0
_SQUIDHNSW_INDEX_CACHE: dict[str, bool] = {}
_SQUIDHNSW_GUC_SUPPORTED: bool | None = None


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
        return "hnsw"
    return str(getattr(efconfig, "kmeans_index_type", "hnsw") or "hnsw").strip().lower()


def _full_table_vector_count() -> int:
    try:
        plan_summary = get_current_plan_summary(refresh=False)
        if plan_summary is not None:
            metadata = dict(plan_summary.get("metadata", {}) or {})
            return int(metadata.get("original_vector_count") or plan_summary.get("vector_count") or 0)
    except Exception:
        return 0
    return 0


def _full_table_recall_base_ef() -> int:
    global _FULL_TABLE_RECALL_EF_CAP
    if _FULL_TABLE_RECALL_EF_CAP is not None:
        return int(_FULL_TABLE_RECALL_EF_CAP)
    full_table_vectors = int(_full_table_vector_count())
    if full_table_vectors > 0:
        cap = int(math.ceil(required_base_ef_star(
            partition_vectors=int(full_table_vectors),
            topk=int(DEFAULT_COST_MODEL.topk),
            model=DEFAULT_COST_MODEL,
        )))
    else:
        cap = int(_HNSW_EF_SEARCH_FALLBACK_MAX)
    _FULL_TABLE_RECALL_EF_CAP = max(1, int(cap))
    return int(_FULL_TABLE_RECALL_EF_CAP)


def _average_global_authorized_selectivity() -> float:
    global _AVERAGE_GLOBAL_SELECTIVITY
    if _AVERAGE_GLOBAL_SELECTIVITY is not None:
        return float(_AVERAGE_GLOBAL_SELECTIVITY)
    plan_summary = get_current_plan_summary(refresh=False)
    full_table_vectors = int(_full_table_vector_count())
    if plan_summary is None or full_table_vectors <= 0:
        _AVERAGE_GLOBAL_SELECTIVITY = 1.0
        return float(_AVERAGE_GLOBAL_SELECTIVITY)
    tenant_count = max(1, int(plan_summary.get("tenant_count", 0) or 0))
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT COALESCE(SUM(pattern_access.accessible_vectors), 0)::FLOAT8
                    FROM (
                        SELECT tenant_id, SUM(vector_count)::BIGINT AS accessible_vectors
                        FROM (
                            SELECT unnest(tenant_ids)::BIGINT AS tenant_id, vector_count
                            FROM {}
                            WHERE plan_id = %s
                        ) expanded
                        GROUP BY tenant_id
                    ) pattern_access;
                    """
                ).format(sql.Identifier(PATTERN_TABLE)),
                [int(plan_summary["plan_id"])],
            )
            total_accessible_vectors = float(cur.fetchone()[0] or 0.0)
    finally:
        conn.close()
    average_accessible_vectors = float(total_accessible_vectors) / float(max(1, int(tenant_count)))
    rho = float(average_accessible_vectors) / float(max(1, int(full_table_vectors)))
    _AVERAGE_GLOBAL_SELECTIVITY = min(1.0, max(0.000001, float(rho)))
    return float(_AVERAGE_GLOBAL_SELECTIVITY)


def _full_table_recall_ef_cap() -> int:
    rho = float(_average_global_authorized_selectivity())
    return int(math.ceil(float(_full_table_recall_base_ef()) / float(rho)))


def _clamp_ef(value: int, route: TenantRoute | None = None) -> int:
    ef = max(1, int(value))
    if route is not None:
        partition_vectors = int(getattr(route, "partition_vector_count", 0) or 0)
        if partition_vectors > 0:
            ef = min(ef, partition_vectors)
    return int(ef)


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


def _force_index_planner(cur) -> None:
    cur.execute("SET enable_seqscan = off;")
    cur.execute("SET enable_bitmapscan = off;")


def _reset_index_planner(cur) -> None:
    cur.execute("RESET enable_seqscan;")
    cur.execute("RESET enable_bitmapscan;")


def _force_linear_planner(cur) -> None:
    cur.execute("RESET enable_seqscan;")
    cur.execute("SET enable_indexscan = off;")
    cur.execute("SET enable_bitmapscan = off;")
    cur.execute("SET enable_indexonlyscan = off;")


def _reset_linear_planner(cur) -> None:
    cur.execute("RESET enable_indexscan;")
    cur.execute("RESET enable_bitmapscan;")
    cur.execute("RESET enable_indexonlyscan;")


def _result_key(row) -> tuple[int, int]:
    return (int(row[1]), int(row[0]))


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



def _global_bound(results, topk: int) -> float:
    if len(results) < int(topk):
        return float("inf")
    ordered = sorted(results, key=lambda row: row[3])
    return float(ordered[int(topk) - 1][3])


def _base_ef(*, topk: int, ef_min: int) -> int:
    return _clamp_ef(max(1, int(ef_min)))


def _allowed_pattern_ids(route: TenantRoute) -> set[int]:
    return {int(pattern_id) for pattern_id in route.pattern_ids}



def _route_selectivity(route: TenantRoute) -> float:
    accessible_vectors = int(getattr(route, "accessible_vector_count", 0) or 0)
    partition_vectors = int(getattr(route, "partition_vector_count", 0) or 0)
    if accessible_vectors <= 0 or partition_vectors <= 0:
        return 1.0
    return min(1.0, max(0.000001, float(accessible_vectors) / float(partition_vectors)))


def _is_pure_route(route: TenantRoute) -> bool:
    accessible_vectors = int(getattr(route, "accessible_vector_count", 0) or 0)
    partition_vectors = int(getattr(route, "partition_vector_count", 0) or 0)
    return partition_vectors > 0 and accessible_vectors >= partition_vectors


def _route_vector_count(route: TenantRoute) -> int:
    return int(getattr(route, "partition_vector_count", 0) or 0)


def _route_impurity(route: TenantRoute) -> float:
    accessible_vectors = int(getattr(route, "accessible_vector_count", 0) or 0)
    partition_vectors = int(getattr(route, "partition_vector_count", 0) or 0)
    if accessible_vectors <= 0:
        return float("inf")
    if partition_vectors <= 0:
        return 1.0
    return float(partition_vectors) / float(accessible_vectors)


def _ordered_search_routes(routes):
    return sorted(
        routes,
        key=lambda route: (
            not _is_pure_route(route),
            _route_impurity(route),
            _route_vector_count(route),
            str(getattr(route, "route_kind", "")),
            int(getattr(route, "cluster_id", 0) or 0),
            str(getattr(route, "partition_id", "")),
        ),
    )


def _should_force_linear_route(route: TenantRoute) -> bool:
    partition_vectors = _route_vector_count(route)
    return partition_vectors > 0 and partition_vectors <= _SQUIDHNSW_LINEAR_SCAN_MAX_VECTORS


def _allowed_patterns_csv(route: TenantRoute) -> str:
    if _is_pure_route(route):
        return ""
    return ",".join(str(int(pattern_id)) for pattern_id in route.pattern_ids)


def _build_authorized_query(
    route: TenantRoute,
    *,
    query_vector,
    limit: int,
    use_sql_filter: bool = True,
):
    if use_sql_filter:
        return (
            sql.SQL(
                """
                SELECT p.block_id, p.document_id, p.block_content,
                       p.vector <-> %s::vector AS distance
                FROM {} p
                WHERE p.pattern_id = ANY(%s)
                  AND NOT EXISTS (
                      SELECT 1
                      FROM {} ts
                      WHERE ts.partition_id = %s
                        AND ts.block_id = p.block_id
                        AND ts.document_id = p.document_id
                        AND ts.version = p.version
                  )
                ORDER BY p.vector <-> %s::vector
                LIMIT %s;
                """
            ).format(sql.Identifier(route.table_name), sql.Identifier(UPDATE_TOMBSTONE_TABLE)),
            [query_vector, list(int(pattern_id) for pattern_id in route.pattern_ids), str(route.partition_id), query_vector, int(limit)],
        )

    return (
        sql.SQL(
            """
            SELECT p.block_id, p.document_id, p.block_content,
                   p.vector <-> %s::vector AS distance
            FROM {} p
            WHERE NOT EXISTS (
                SELECT 1
                FROM {} ts
                WHERE ts.partition_id = %s
                  AND ts.block_id = p.block_id
                  AND ts.document_id = p.document_id
                  AND ts.version = p.version
            )
            ORDER BY p.vector <-> %s::vector
            LIMIT %s;
            """
        ).format(sql.Identifier(route.table_name), sql.Identifier(UPDATE_TOMBSTONE_TABLE)),
        [query_vector, str(route.partition_id), query_vector, int(limit)],
    )


def _squidhnsw_guc_supported(cur) -> bool:
    global _SQUIDHNSW_GUC_SUPPORTED
    if _SQUIDHNSW_GUC_SUPPORTED is not None:
        return bool(_SQUIDHNSW_GUC_SUPPORTED)
    _SQUIDHNSW_GUC_SUPPORTED = _set_if_supported(cur, "SET squidhnsw.base_ef = 1;")
    return bool(_SQUIDHNSW_GUC_SUPPORTED)



def _configure_squidhnsw_route(cur, route: TenantRoute, *, base_ef: int, max_ef: int, topk: int = 0, global_bound: float = float("inf")) -> bool:
    if not _squidhnsw_guc_supported(cur):
        return False
    cur.execute(f"SET squidhnsw.base_ef = {int(base_ef)};")
    cur.execute(f"SET squidhnsw.max_ef = {int(max_ef)};")
    cur.execute(f"SET squidhnsw.topk = {max(0, int(topk))};")
    bound_value = float(global_bound) if math.isfinite(float(global_bound)) else -1.0
    cur.execute(f"SET squidhnsw.global_bound = {bound_value:.17g};")
    cur.execute(f"SET squidhnsw.route_selectivity = {_route_selectivity(route):.12f};")
    cur.execute(sql.SQL("SET squidhnsw.allowed_patterns = {};").format(sql.Literal(_allowed_patterns_csv(route))))
    return True


def _has_squidhnsw_index(cur, table_name: str) -> bool:
    table_key = str(table_name)
    cached = _SQUIDHNSW_INDEX_CACHE.get(table_key)
    if cached is not None:
        return bool(cached)

    try:
        cur.execute("SAVEPOINT kmeans_squidhnsw_index_check;")
        cur.execute(
            """
            SELECT EXISTS (
                SELECT 1
                FROM pg_index i
                JOIN pg_class idx ON idx.oid = i.indexrelid
                JOIN pg_am am ON am.oid = idx.relam
                WHERE i.indrelid = to_regclass(%s)
                  AND i.indisvalid
                  AND am.amname = 'squidhnsw'
            );
            """,
            [table_key],
        )
        exists = bool(cur.fetchone()[0])
        cur.execute("RELEASE SAVEPOINT kmeans_squidhnsw_index_check;")
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT kmeans_squidhnsw_index_check;")
        cur.execute("RELEASE SAVEPOINT kmeans_squidhnsw_index_check;")
        exists = False

    _SQUIDHNSW_INDEX_CACHE[table_key] = exists
    return bool(exists)


def _can_use_squidhnsw_filter(cur, route: TenantRoute, *, base_ef: int, max_ef: int, topk: int = 0, global_bound: float = float("inf")) -> bool:
    if not _configure_squidhnsw_route(cur, route, base_ef=int(base_ef), max_ef=int(max_ef), topk=int(topk), global_bound=float(global_bound)):
        return False
    return _has_squidhnsw_index(cur, route.table_name)


def _execute_authorized_search(
    cur,
    route: TenantRoute,
    *,
    query_vector,
    topk: int,
    base_ef: int,
    max_ef: int,
    use_kernel_filter: bool | None = None,
    global_bound: float = float("inf"),
    ):
    if use_kernel_filter is None:
        use_kernel_filter = _can_use_squidhnsw_filter(cur, route, base_ef=int(base_ef), max_ef=int(max_ef), topk=int(topk), global_bound=float(global_bound))
    query, params = _build_authorized_query(
        route,
        query_vector=query_vector,
        limit=int(topk),
        use_sql_filter=not use_kernel_filter,
    )
    if use_kernel_filter:
        _force_index_planner(cur)
    try:
        cur.execute(query, params)
        return cur.fetchall()
    finally:
        if use_kernel_filter:
            _reset_index_planner(cur)


def _execute_linear_authorized_search(cur, route: TenantRoute, *, query_vector, topk: int):
    query, params = _build_authorized_query(
        route,
        query_vector=query_vector,
        limit=int(topk),
        use_sql_filter=not _is_pure_route(route),
    )
    _force_linear_planner(cur)
    try:
        cur.execute(query, params)
        return cur.fetchall()
    finally:
        _reset_linear_planner(cur)


def _extract_execution_time_seconds(explain_rows) -> float:
    total = 0.0
    for (line,) in explain_rows:
        if "Execution Time" in line:
            total += float(line.split()[-2]) / 1000.0
    return total



def _explain_authorized_search_time(
    cur,
    route: TenantRoute,
    *,
    query_vector,
    topk: int,
    base_ef: int,
    max_ef: int,
    use_kernel_filter: bool | None = None,
    global_bound: float = float("inf"),
    ) -> float:
    if use_kernel_filter is None:
        use_kernel_filter = _can_use_squidhnsw_filter(cur, route, base_ef=int(base_ef), max_ef=int(max_ef), topk=int(topk), global_bound=float(global_bound))
    query, params = _build_authorized_query(
        route,
        query_vector=query_vector,
        limit=int(topk),
        use_sql_filter=not use_kernel_filter,
    )
    if use_kernel_filter:
        _force_index_planner(cur)
    try:
        cur.execute(sql.SQL("EXPLAIN ANALYZE {}").format(query), params)
        return _extract_execution_time_seconds(cur.fetchall())
    finally:
        if use_kernel_filter:
            _reset_index_planner(cur)


def _explain_linear_authorized_search_time(cur, route: TenantRoute, *, query_vector, topk: int) -> float:
    query, params = _build_authorized_query(
        route,
        query_vector=query_vector,
        limit=int(topk),
        use_sql_filter=not _is_pure_route(route),
    )
    _force_linear_planner(cur)
    try:
        cur.execute(sql.SQL("EXPLAIN ANALYZE {}").format(query), params)
        return _extract_execution_time_seconds(cur.fetchall())
    finally:
        _reset_linear_planner(cur)


def _search_linear_route(cur, route: TenantRoute, *, query_vector, topk: int, collect_sql_time: bool):
    if collect_sql_time:
        elapsed = _explain_linear_authorized_search_time(
            cur,
            route,
            query_vector=query_vector,
            topk=int(topk),
        )
        rows = _execute_linear_authorized_search(
            cur,
            route,
            query_vector=query_vector,
            topk=int(topk),
        )
    else:
        started = time.perf_counter()
        rows = _execute_linear_authorized_search(
            cur,
            route,
            query_vector=query_vector,
            topk=int(topk),
        )
        elapsed = time.perf_counter() - started
    return rows, float(elapsed)


def _execute_hnsw_filtered_search(cur, route: TenantRoute, *, query_vector, topk: int, ef_search: int):
    ef_search = _clamp_ef(int(ef_search), route)
    _try_set(cur, f"SET hnsw.ef_search = {int(ef_search)};")
    query, params = _build_authorized_query(
        route,
        query_vector=query_vector,
        limit=int(topk),
        use_sql_filter=not _is_pure_route(route),
    )
    _force_index_planner(cur)
    try:
        cur.execute(query, params)
        return cur.fetchall()
    finally:
        _reset_index_planner(cur)


def _explain_hnsw_filtered_search_time(cur, route: TenantRoute, *, query_vector, topk: int, ef_search: int) -> float:
    ef_search = _clamp_ef(int(ef_search), route)
    _try_set(cur, f"SET hnsw.ef_search = {int(ef_search)};")
    query, params = _build_authorized_query(
        route,
        query_vector=query_vector,
        limit=int(topk),
        use_sql_filter=not _is_pure_route(route),
    )
    _force_index_planner(cur)
    try:
        cur.execute(sql.SQL("EXPLAIN ANALYZE {}").format(query), params)
        return _extract_execution_time_seconds(cur.fetchall())
    finally:
        _reset_index_planner(cur)


def _search_hnsw_filtered_route(cur, route: TenantRoute, *, query_vector, topk: int, ef_min: int, collect_sql_time: bool):
    ef_search = _base_ef(topk=int(topk), ef_min=int(ef_min))
    if collect_sql_time:
        elapsed = _explain_hnsw_filtered_search_time(
            cur,
            route,
            query_vector=query_vector,
            topk=int(topk),
            ef_search=int(ef_search),
        )
        rows = _execute_hnsw_filtered_search(
            cur,
            route,
            query_vector=query_vector,
            topk=int(topk),
            ef_search=int(ef_search),
        )
    else:
        started = time.perf_counter()
        rows = _execute_hnsw_filtered_search(
            cur,
            route,
            query_vector=query_vector,
            topk=int(topk),
            ef_search=int(ef_search),
        )
        elapsed = time.perf_counter() - started
    return rows, float(elapsed)


def _search_squidhnsw_route(
    cur,
    route: TenantRoute,
    *,
    query_vector,
    topk: int,
    ef_min: int,
    collect_sql_time: bool,
    global_bound: float = float("inf"),
):
    if _should_force_linear_route(route):
        return _search_linear_route(
            cur,
            route,
            query_vector=query_vector,
            topk=int(topk),
            collect_sql_time=bool(collect_sql_time),
        )

    base_ef = _base_ef(topk=int(topk), ef_min=int(ef_min))
    max_ef = _clamp_ef(_full_table_recall_ef_cap(), route)
    use_kernel_filter = _can_use_squidhnsw_filter(cur, route, base_ef=int(base_ef), max_ef=int(max_ef), topk=int(topk), global_bound=float(global_bound))
    if not use_kernel_filter:
        raise RuntimeError(
            f"SQUIDHNSW search requires a valid squidhnsw index and GUC support for {route.table_name}; "
            "linear fallback is disabled for KMeans SQUIDHNSW."
        )

    if collect_sql_time:
        elapsed = _explain_authorized_search_time(
            cur,
            route,
            query_vector=query_vector,
            topk=int(topk),
            base_ef=int(base_ef),
            max_ef=int(max_ef),
            use_kernel_filter=bool(use_kernel_filter),
            global_bound=float(global_bound),
        )
        rows = _execute_authorized_search(
            cur,
            route,
            query_vector=query_vector,
            topk=int(topk),
            base_ef=int(base_ef),
            max_ef=int(max_ef),
            use_kernel_filter=bool(use_kernel_filter),
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
            use_kernel_filter=bool(use_kernel_filter),
            global_bound=float(global_bound),
        )
        elapsed = time.perf_counter() - started
    return rows, float(elapsed)



def kmeans_partition_search(user_id: int, query_vector, topk: int = 5, statistics_type: str = "sql"):
    if statistics_type == "sql":
        return kmeans_partition_search_statistics_sql(user_id, query_vector, topk)
    if statistics_type == "system":
        return kmeans_partition_search_statistics_system(user_id, query_vector, topk)
    raise ValueError(f"Unknown statistics type: {statistics_type}")


def _kmeans_partition_search_impl(user_id: int, query_vector, topk: int, *, collect_sql_time: bool):
    system_started = time.perf_counter() if not collect_sql_time else None
    system_excluded_time = 0.0
    # Route plans are switched atomically after their physical tables are ready.
    # Refresh on every query so a worker cannot keep serving a pre-switch route
    # from its process-local cache.
    routes = [route for route in load_tenant_routes(int(user_id), refresh=True) if route.pattern_ids]
    if not routes:
        if system_started is None:
            return [], 0.0
        return [], float(time.perf_counter() - system_started)

    connect_started = time.perf_counter() if system_started is not None else None
    ef_min = _configured_ef_min()
    index_type = _configured_index_type()
    if index_type == "acorn":
        return search_kmeans_acorn(int(user_id), query_vector, int(topk), int(_base_ef(topk=int(topk), ef_min=int(ef_min))))

    conn = get_db_connection()
    if connect_started is not None:
        system_excluded_time += time.perf_counter() - connect_started
    total_query_time = 0.0
    all_results = []
    final_results = []
    try:
        with conn.cursor() as cur:
            _configure_search_session(cur)
            for route in _ordered_search_routes(routes):
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
                    route_results, route_time = _search_hnsw_filtered_route(
                        cur,
                        route,
                        query_vector=query_vector,
                        topk=int(topk),
                        ef_min=int(ef_min),
                        collect_sql_time=bool(collect_sql_time),
                    )
                total_query_time += float(route_time)
                all_results.extend(route_results)

            final_results = _merge_results(all_results, int(topk))
    finally:
        close_started = time.perf_counter() if system_started is not None else None
        conn.close()
        if close_started is not None:
            system_excluded_time += time.perf_counter() - close_started
    if system_started is not None:
        total_query_time = time.perf_counter() - system_started - system_excluded_time
    return final_results, float(total_query_time)


def kmeans_partition_search_statistics_sql(user_id: int, query_vector, topk: int = 5):
    return _kmeans_partition_search_impl(int(user_id), query_vector, int(topk), collect_sql_time=True)


def kmeans_partition_search_statistics_system(user_id: int, query_vector, topk: int = 5):
    return _kmeans_partition_search_impl(int(user_id), query_vector, int(topk), collect_sql_time=False)
