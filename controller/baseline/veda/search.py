from __future__ import annotations

import importlib
import math
import sys
import time

from psycopg2 import sql

from services.config import get_db_connection

from .common import VedaRoute
from .storage import load_user_routes


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


def _configured_int(name: str, default: int, *, minimum: int = 1) -> int:
    efconfig = _resolve_efconfig_module()
    if efconfig is None or not hasattr(efconfig, name):
        return max(minimum, int(default))
    value = getattr(efconfig, name)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"", "adaptive", "auto", "none"}:
            return max(minimum, int(default))
        return max(minimum, int(float(normalized)))
    return max(minimum, int(value))


def _configured_value(name: str, default):
    efconfig = _resolve_efconfig_module()
    if efconfig is None or not hasattr(efconfig, name):
        return default
    return getattr(efconfig, name)


def _try_set(cur, statement: str) -> None:
    try:
        cur.execute("SAVEPOINT veda_search_config;")
        cur.execute(statement)
        cur.execute("RELEASE SAVEPOINT veda_search_config;")
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT veda_search_config;")
        cur.execute("RELEASE SAVEPOINT veda_search_config;")


def _configure_search_session(cur) -> None:
    cur.execute("SET max_parallel_workers_per_gather = 0;")
    cur.execute("SET jit = off;")
    iterative_scan = str(_configured_value("veda_hnsw_iterative_scan", "off")).strip().lower()
    if iterative_scan and iterative_scan not in {"0", "false", "off", "none"}:
        _try_set(cur, f"SET hnsw.iterative_scan = {iterative_scan};")
    max_scan_tuples = _configured_value("veda_hnsw_max_scan_tuples", None)
    if max_scan_tuples is not None:
        normalized_max_scan_tuples = str(max_scan_tuples).strip().lower()
        if normalized_max_scan_tuples not in {"", "0", "false", "off", "none"}:
            _try_set(cur, f"SET hnsw.max_scan_tuples = {max(1, int(max_scan_tuples))};")


def _base_ef_search() -> int:
    return _configured_int("veda_ef_search", _configured_int("ef_search", 100), minimum=1)


def _search_mode() -> str:
    normalized = str(_configured_value("veda_search_mode", "coordinated")).strip().lower().replace("-", "_")
    if normalized in {"coordinated", "coord", "effveda", "default"}:
        return "coordinated"
    if normalized in {"naive", "baseline", "simple"}:
        return "naive"
    if normalized in {"ours", "kmeans", "adaptive", "single_pass"}:
        return "ours"
    raise ValueError(f"Unknown Veda search mode: {normalized}")


def _ours_route_ef(route: VedaRoute, *, topk: int, base_ef: int) -> int:
    accessible_vectors = int(getattr(route, "accessible_vector_count", 0) or 0)
    node_vectors = int(getattr(route, "node_vector_count", 0) or 0)
    if accessible_vectors <= 0 or node_vectors <= 0:
        return max(1, int(base_ef))
    adaptive_ef = int(math.ceil(float(int(topk) * int(node_vectors)) / float(accessible_vectors)))
    return max(max(1, int(base_ef)), int(adaptive_ef))


def _sql_timing_mode() -> str:
    normalized = str(_configured_value("veda_sql_timing_mode", "fair")).strip().lower().replace("-", "_")
    if normalized in {"fair", "aligned", "all_sql", "full_sql"}:
        return "fair"
    if normalized in {"legacy", "paper", "original"}:
        return "legacy"
    raise ValueError(f"Unknown Veda SQL timing mode: {normalized}")


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


def _global_bound(results, topk: int) -> float:
    if len(results) < int(topk):
        return float("inf")
    ordered = sorted(results, key=lambda row: row[3])
    return float(ordered[int(topk) - 1][3])


def _explain_analyze_time(cur, query, params) -> float:
    cur.execute(sql.SQL("EXPLAIN ANALYZE {}").format(query), params)
    total_query_time = 0.0
    for (line,) in cur.fetchall():
        if "Execution Time" in line:
            total_query_time += float(line.split()[-2]) / 1000.0
    return total_query_time


def _release_statement_locks(conn) -> None:
    # Long SQL-statistics runs touch many partition tables. Ending the
    # transaction after each route releases relation locks accumulated by
    # EXPLAIN ANALYZE and avoids exhausting PostgreSQL shared lock memory.
    conn.commit()


def _route_query(route: VedaRoute, *, query_vector, topk: int, limit: int):
    return (
        sql.SQL(
            "\n"
            "            SELECT block_id, document_id, block_content, distance\n"
            "            FROM (\n"
            "                SELECT p.block_id, p.document_id, p.block_content,\n"
            "                       p.vector <-> %s::vector AS distance\n"
            "                FROM {} p\n"
            "                WHERE p.pattern_id = ANY(%s)\n"
            "                ORDER BY distance\n"
            "                LIMIT %s\n"
            "            ) routed\n"
            "            ORDER BY distance\n"
            "            LIMIT %s;\n"
            "            "
        ).format(sql.Identifier(route.table_name)),
        [query_vector, list(int(pattern_id) for pattern_id in route.pattern_ids), int(limit), int(max(topk, limit))],
    )


def _local_unfiltered_probe(cur, route: VedaRoute, *, query_vector, topk: int, collect_sql_time: bool):
    query = sql.SQL(
        """
        SELECT block_id, document_id, block_content, pattern_id, vector <-> %s::vector AS distance
        FROM {}
        ORDER BY distance
        LIMIT %s;
        """
    ).format(sql.Identifier(route.table_name))
    params = [query_vector, int(topk)]
    total_query_time = _explain_analyze_time(cur, query, params) if collect_sql_time else 0.0
    cur.execute(query, params)
    return cur.fetchall(), total_query_time



def _authorized_probe_results(local_unfiltered, route: VedaRoute):
    accessible_patterns = set(int(pattern_id) for pattern_id in route.pattern_ids)
    authorized = []
    for row in local_unfiltered:
        if len(row) < 5:
            continue
        block_id, document_id, block_content, pattern_id, distance = row
        if int(pattern_id) in accessible_patterns:
            authorized.append((block_id, document_id, block_content, distance))
    return authorized


def veda_search(user_id: int, query_vector, topk: int = 5, statistics_type: str = "sql"):
    if statistics_type == "sql":
        return veda_search_statistics_sql(user_id, query_vector, topk)
    if statistics_type == "system":
        return veda_search_statistics_system(user_id, query_vector, topk)
    raise ValueError(f"Unknown statistics type: {statistics_type}")


def veda_search_statistics_sql(user_id: int, query_vector, topk: int = 5):
    return _veda_search_impl(int(user_id), query_vector, int(topk), collect_sql_time=True)


def veda_search_statistics_system(user_id: int, query_vector, topk: int = 5):
    return _veda_search_impl(int(user_id), query_vector, int(topk), collect_sql_time=False)


def _veda_search_impl(user_id: int, query_vector, topk: int, *, collect_sql_time: bool):
    started_at = time.time()
    routes = [route for route in load_user_routes(int(user_id)) if route.pattern_ids]
    if not routes:
        return [], 0.0

    leftovers = [route for route in routes if str(route.route_kind) == "leftover"]
    pure_indices = [route for route in routes if str(route.route_kind) == "index"]
    impure_indices = [route for route in routes if str(route.route_kind) == "impure_index"]
    impure_indices.sort(key=lambda route: float(route.impurity_factor))
    search_mode = _search_mode()
    count_probe_sql_time = collect_sql_time and _sql_timing_mode() == "fair"

    base_ef = _base_ef_search()
    total_query_time = 0.0
    all_results = []
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            _configure_search_session(cur)

            def _search_route(route: VedaRoute, *, ef_search: int, limit: int) -> None:
                nonlocal total_query_time
                _try_set(cur, f"SET hnsw.ef_search = {int(ef_search)};")
                query, params = _route_query(route, query_vector=query_vector, topk=topk, limit=limit)
                if collect_sql_time:
                    total_query_time += _explain_analyze_time(cur, query, params)
                cur.execute(query, params)
                all_results.extend(cur.fetchall())
                _release_statement_locks(conn)

            if search_mode == "ours":
                for route in routes:
                    route_ef = _ours_route_ef(route, topk=topk, base_ef=base_ef)
                    _search_route(route, ef_search=route_ef, limit=topk)
            else:
                for route in leftovers:
                    _search_route(route, ef_search=base_ef, limit=topk)

                for route in pure_indices:
                    _search_route(route, ef_search=base_ef, limit=topk)

                if search_mode == "naive":
                    for route in impure_indices:
                        inflated_ef = max(int(base_ef), int(math.ceil(float(route.impurity_factor) * float(base_ef))))
                        inflated_limit = max(int(topk), int(math.ceil(float(route.impurity_factor) * float(topk))))
                        _search_route(route, ef_search=inflated_ef, limit=inflated_limit)
                else:
                    global_bound = _global_bound(all_results, int(topk))
                    for route in impure_indices:
                        _try_set(cur, f"SET hnsw.ef_search = {int(base_ef)};")
                        local_unfiltered, probe_time = _local_unfiltered_probe(
                            cur,
                            route,
                            query_vector=query_vector,
                            topk=topk,
                            collect_sql_time=count_probe_sql_time,
                        )
                        total_query_time += probe_time
                        _release_statement_locks(conn)
                        all_results.extend(_authorized_probe_results(local_unfiltered, route))
                        local_bound = float(local_unfiltered[int(topk) - 1][4]) if len(local_unfiltered) >= int(topk) else float("inf")
                        global_bound = _global_bound(all_results, int(topk))

                        if math.isfinite(global_bound) and local_bound >= global_bound:
                            continue

                        inflated_ef = max(int(base_ef), int(math.ceil(float(route.impurity_factor) * float(base_ef))))
                        inflated_limit = max(int(topk), int(math.ceil(float(route.impurity_factor) * float(topk))))
                        _search_route(route, ef_search=inflated_ef, limit=inflated_limit)
                        global_bound = _global_bound(all_results, int(topk))
    finally:
        conn.close()

    elapsed = total_query_time if collect_sql_time else time.time() - started_at
    return _merge_results(all_results, int(topk)), float(elapsed)
