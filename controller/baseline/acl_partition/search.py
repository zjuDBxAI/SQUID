from __future__ import annotations

import importlib
import sys
import time

from psycopg2 import sql

from services.config import get_db_connection

from .storage import ACL_PLAN_TABLE, ACL_ROUTE_TABLE


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


def _try_set(cur, statement: str) -> None:
    try:
        cur.execute("SAVEPOINT acl_partition_search_config;")
        cur.execute(statement)
        cur.execute("RELEASE SAVEPOINT acl_partition_search_config;")
    except Exception:
        cur.execute("ROLLBACK TO SAVEPOINT acl_partition_search_config;")
        cur.execute("RELEASE SAVEPOINT acl_partition_search_config;")


def _configure_search_session(cur) -> None:
    cur.execute("SET max_parallel_workers_per_gather = 0;")
    cur.execute("SET jit = off;")
    ef_search = _configured_int("acl_partition_ef_search", _configured_int("ef_search", 100), minimum=1)
    _try_set(cur, f"SET hnsw.ef_search = {int(ef_search)};")
    _try_set(cur, f"SET ivfflat.probes = {int(_configured_int('nprobe', 1, minimum=1))};")


def _explain_analyze_time(cur, query, params) -> float:
    cur.execute(sql.SQL("EXPLAIN ANALYZE {}").format(query), params)
    total_query_time = 0.0
    for (line,) in cur.fetchall():
        if "Execution Time" in line:
            total_query_time += float(line.split()[-2]) / 1000.0
    return total_query_time


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


def _load_user_routes(cur, user_id: int) -> list[str]:
    cur.execute(
        sql.SQL(
            """
            SELECT route.table_name
            FROM {} route
            WHERE route.plan_id = (SELECT MAX(plan_id) FROM {})
              AND route.user_id = %s
            ORDER BY route.pattern_id;
            """
        ).format(sql.Identifier(ACL_ROUTE_TABLE), sql.Identifier(ACL_PLAN_TABLE)),
        [int(user_id)],
    )
    return [str(row[0]) for row in cur.fetchall()]


def _partition_query(table_name: str, *, query_vector, topk: int):
    return (
        sql.SQL(
            """
            SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
            FROM {}
            ORDER BY distance
            LIMIT %s;
            """
        ).format(sql.Identifier(table_name)),
        [query_vector, int(topk)],
    )


def acl_partition_search(user_id: int, query_vector, topk: int = 5, statistics_type: str = "sql"):
    if statistics_type == "sql":
        return acl_partition_search_statistics_sql(user_id, query_vector, topk)
    if statistics_type == "system":
        return acl_partition_search_statistics_system(user_id, query_vector, topk)
    raise ValueError(f"Unknown statistics type: {statistics_type}")


def acl_partition_search_statistics_sql(user_id: int, query_vector, topk: int = 5):
    return _acl_partition_search_impl(int(user_id), query_vector, int(topk), collect_sql_time=True)


def acl_partition_search_statistics_system(user_id: int, query_vector, topk: int = 5):
    return _acl_partition_search_impl(int(user_id), query_vector, int(topk), collect_sql_time=False)


def _acl_partition_search_impl(user_id: int, query_vector, topk: int, *, collect_sql_time: bool):
    started_at = time.time()
    total_query_time = 0.0
    all_results = []

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            _configure_search_session(cur)
            table_names = _load_user_routes(cur, int(user_id))
            if not table_names:
                return [], 0.0
            for table_name in table_names:
                query, params = _partition_query(table_name, query_vector=query_vector, topk=topk)
                if collect_sql_time:
                    total_query_time += _explain_analyze_time(cur, query, params)
                cur.execute(query, params)
                all_results.extend(cur.fetchall())
    finally:
        conn.close()

    elapsed = total_query_time if collect_sql_time else time.time() - started_at
    return _merge_results(all_results, int(topk)), float(elapsed)
