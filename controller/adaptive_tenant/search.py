"""Physical search helpers for adaptive tenant partitions."""

from __future__ import annotations

import atexit
import os
import time
from dataclasses import dataclass, field
from typing import Optional

from psycopg2 import sql

from services.config import get_db_connection

try:
    import efconfig
except Exception:  # pragma: no cover - benchmark-only optional module
    efconfig = None
from .efs_adaptive import get_adaptive_ef_search
from .load_result_to_database import get_current_plan_summary, get_partition_table_name, load_current_partitions


@dataclass(slots=True)
class TenantPartitionRoute:
    tenant_id: int
    partition_ids: tuple[str, ...]
    partition_count: int
    estimated_memory: float
    total_vector_count: int
    total_document_count: int
    metadata: dict = field(default_factory=dict)


@dataclass(slots=True)
class AdaptiveSearchPlan:
    tenant_id: int
    route: TenantPartitionRoute
    plan_summary: Optional[dict]


_CACHED_ROUTE_VERSION: Optional[int] = None
_CACHED_ROUTE_MAP: Optional[dict[int, list]] = None
_CACHED_PLAN_SUMMARY: Optional[dict] = None
_SEARCH_CONN = None
_SEARCH_CONN_PID: Optional[int] = None
_SEARCH_SESSION_READY = False


def _load_cached_plan_state(*, refresh: bool = False):
    global _CACHED_ROUTE_VERSION, _CACHED_ROUTE_MAP, _CACHED_PLAN_SUMMARY

    plan_summary = get_current_plan_summary(refresh=refresh)
    plan_id = int(plan_summary['plan_id']) if plan_summary is not None else 0
    if refresh or _CACHED_ROUTE_MAP is None or _CACHED_ROUTE_VERSION != plan_id:
        partitions = load_current_partitions(refresh=refresh)
        route_map: dict[int, list] = {}
        for partition in partitions:
            for tenant_id in partition.tenant_ids:
                route_map.setdefault(int(tenant_id), []).append(partition)
        _CACHED_ROUTE_VERSION = plan_id
        _CACHED_ROUTE_MAP = route_map
        _CACHED_PLAN_SUMMARY = plan_summary
    return _CACHED_ROUTE_MAP or {}, _CACHED_PLAN_SUMMARY


def _close_search_connection() -> None:
    global _SEARCH_CONN, _SEARCH_CONN_PID, _SEARCH_SESSION_READY
    conn = _SEARCH_CONN
    _SEARCH_CONN = None
    _SEARCH_CONN_PID = None
    _SEARCH_SESSION_READY = False
    if conn is None:
        return
    try:
        conn.close()
    except Exception:
        pass


def _get_search_connection():
    global _SEARCH_CONN, _SEARCH_CONN_PID, _SEARCH_SESSION_READY

    current_pid = os.getpid()
    if _SEARCH_CONN is not None and (_SEARCH_CONN_PID != current_pid or getattr(_SEARCH_CONN, 'closed', 1)):
        _close_search_connection()

    if _SEARCH_CONN is None:
        _SEARCH_CONN = get_db_connection()
        _SEARCH_CONN_PID = current_pid
        _SEARCH_SESSION_READY = False

    if not _SEARCH_SESSION_READY:
        with _SEARCH_CONN.cursor() as cur:
            cur.execute('SET max_parallel_workers_per_gather = 0;')
            cur.execute('SET jit = off;')
        _SEARCH_SESSION_READY = True

    return _SEARCH_CONN


def get_tenant_partition_route(tenant_id: int) -> TenantPartitionRoute:
    route_map, _ = _load_cached_plan_state()
    matching = route_map.get(int(tenant_id), [])
    if not matching:
        return TenantPartitionRoute(
            tenant_id=int(tenant_id),
            partition_ids=(),
            partition_count=0,
            estimated_memory=0.0,
            total_vector_count=0,
            total_document_count=0,
            metadata={'found': False},
        )

    return TenantPartitionRoute(
        tenant_id=int(tenant_id),
        partition_ids=tuple(partition.partition_id for partition in matching),
        partition_count=len(matching),
        estimated_memory=sum(float(partition.estimated_memory) for partition in matching),
        total_vector_count=sum(int(partition.vector_count) for partition in matching),
        total_document_count=sum(int(partition.document_count) for partition in matching),
        metadata={
            'found': True,
            'table_names': [partition.metadata.get('table_name', get_partition_table_name(partition.partition_id)) for partition in matching],
            'tenant_ids': [list(partition.tenant_ids) for partition in matching],
        },
    )


def plan_adaptive_search(tenant_id: int) -> AdaptiveSearchPlan:
    _, plan_summary = _load_cached_plan_state()
    return AdaptiveSearchPlan(
        tenant_id=int(tenant_id),
        route=get_tenant_partition_route(tenant_id),
        plan_summary=plan_summary,
    )


def adaptive_tenant_search(
    user_id: int,
    query_vector,
    topk: int = 5,
    statistics_type: str = 'sql',
):
    if statistics_type == 'sql':
        return adaptive_tenant_search_statistics_sql(user_id, query_vector, topk)
    if statistics_type == 'system':
        return adaptive_tenant_search_statistics_system(user_id, query_vector, topk)
    raise ValueError(f'Unknown statistics type: {statistics_type}')


def _partitions_for_tenant(user_id: int):
    route_map, _ = _load_cached_plan_state()
    return route_map.get(int(user_id), [])


def _configured_fixed_ef_search() -> Optional[int]:
    configured = getattr(efconfig, 'ef_search', 'adaptive') if efconfig is not None else 'adaptive'
    if configured is None:
        return None
    if isinstance(configured, str):
        normalized = configured.strip().lower()
        if normalized in {'', 'adaptive', 'auto', 'none'}:
            return None
        try:
            return max(1, int(float(normalized)))
        except ValueError as exc:
            raise ValueError(f'Invalid ef_search setting: {configured!r}') from exc
    return max(1, int(configured))


def _resolve_partition_ef_search(partition, user_id: int, topk: int) -> int:
    fixed_ef = _configured_fixed_ef_search()
    if fixed_ef is not None:
        return max(topk, fixed_ef)

    tenant_document_count = int(partition.tenant_document_counts.get(user_id, 0))
    sensitivity = max(tenant_document_count / max(partition.document_count, 1), 1e-6)
    pollution = float(partition.metadata.get('pollution', 0.0) or 0.0)
    return max(
        topk,
        int(round(get_adaptive_ef_search(
            sel_whole=sensitivity,
            topk=topk,
            recall_target=partition.recall_target,
            pollution=pollution,
        ))),
    )


def adaptive_tenant_search_statistics_sql(user_id: int, query_vector, topk: int = 5):
    partitions = _partitions_for_tenant(user_id)
    if not partitions:
        return [], 0.0

    conn = _get_search_connection()
    cur = conn.cursor()
    total_query_time = 0.0
    all_results = []
    try:
        for partition in partitions:
            table_name = partition.metadata.get('table_name', get_partition_table_name(partition.partition_id))
            ef_search = _resolve_partition_ef_search(partition, user_id, topk)
            cur.execute(f'SET hnsw.ef_search = {ef_search};')
            explain_query = sql.SQL(
                """
                EXPLAIN ANALYZE
                SELECT p.block_id, p.document_id, p.block_content,
                       p.vector <-> %s::vector AS distance
                FROM {} AS p
                WHERE EXISTS (
                    SELECT 1
                    FROM PermissionAssignment pa
                    JOIN UserRoles ur ON pa.role_id = ur.role_id
                    WHERE ur.user_id = %s
                      AND pa.document_id = p.document_id
                )
                ORDER BY distance
                LIMIT %s;
                """
            ).format(sql.Identifier(table_name))
            cur.execute(explain_query, [query_vector, user_id, topk])
            explain_plan = cur.fetchall()
            for row in explain_plan:
                line = row[0].strip()
                if 'Execution Time' in line:
                    total_query_time += float(line.split()[-2]) / 1000.0

            query = sql.SQL(
                """
                SELECT p.block_id, p.document_id, p.block_content,
                       p.vector <-> %s::vector AS distance
                FROM {} AS p
                WHERE EXISTS (
                    SELECT 1
                    FROM PermissionAssignment pa
                    JOIN UserRoles ur ON pa.role_id = ur.role_id
                    WHERE ur.user_id = %s
                      AND pa.document_id = p.document_id
                )
                ORDER BY distance
                LIMIT %s;
                """
            ).format(sql.Identifier(table_name))
            cur.execute(query, [query_vector, user_id, topk])
            all_results.extend(cur.fetchall())
        conn.rollback()
    finally:
        cur.close()
    return merge_results(all_results, topk), total_query_time


def adaptive_tenant_search_statistics_system(user_id: int, query_vector, topk: int = 5):
    start = time.time()
    partitions = _partitions_for_tenant(user_id)
    if not partitions:
        return [], 0.0

    conn = _get_search_connection()
    cur = conn.cursor()
    all_results = []
    try:
        for partition in partitions:
            table_name = partition.metadata.get('table_name', get_partition_table_name(partition.partition_id))
            ef_search = _resolve_partition_ef_search(partition, user_id, topk)
            cur.execute(f'SET hnsw.ef_search = {ef_search};')
            query = sql.SQL(
                """
                SELECT p.block_id, p.document_id, p.block_content,
                       p.vector <-> %s::vector AS distance
                FROM {} AS p
                WHERE EXISTS (
                    SELECT 1
                    FROM PermissionAssignment pa
                    JOIN UserRoles ur ON pa.role_id = ur.role_id
                    WHERE ur.user_id = %s
                      AND pa.document_id = p.document_id
                )
                ORDER BY distance
                LIMIT %s;
                """
            ).format(sql.Identifier(table_name))
            cur.execute(query, [query_vector, user_id, topk])
            all_results.extend(cur.fetchall())
        conn.rollback()
    finally:
        cur.close()
    return merge_results(all_results, topk), time.time() - start


def merge_results(all_results, topk: int):
    seen = set()
    unique_results = []
    all_results.sort(key=lambda row: row[3])
    for row in all_results:
        key = (row[1], row[0])
        if key in seen:
            continue
        seen.add(key)
        unique_results.append(row)
        if len(unique_results) >= topk:
            break
    return unique_results


atexit.register(_close_search_connection)
