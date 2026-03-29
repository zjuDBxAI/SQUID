"""Persistence and physical materialization helpers for adaptive tenant plans.

This module handles two layers:

1. Control-plane persistence of the latest adaptive tenant partition plan.
2. Physical materialization of that plan into concrete PostgreSQL tables that
   mirror Honeybee's partitioned documentblocks layout.
"""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import re
import time
from typing import Iterable, Optional, Sequence

from psycopg2 import sql

from .planner import AdaptiveTenantPlanner, PlannerResult, PlannedPartition
from .tenant_state import TenantStateRepository
from controller.dynamic_partition.load_result_to_database import _configure_index_session
from services.config import get_db_connection, get_document_vector_dimension, get_maintenance_settings

ADAPTIVE_PARTITION_TABLE_PREFIX = 'adaptive_documentblocks_partition_'
ADAPTIVE_POLICY_NAME = 'adaptive_tenant_access_policy'

_CACHED_PARTITIONS: Optional[list[PlannedPartition]] = None
_CACHED_PLAN_SUMMARY: Optional[dict] = None
_MAX_PARALLEL_WORKERS_CAP = 8


def _default_parallel_worker_count(max_workers: Optional[int] = None) -> int:
    if max_workers is not None:
        return max(1, int(max_workers))
    cpu_count = os.cpu_count() or 1
    return max(1, min(_MAX_PARALLEL_WORKERS_CAP, cpu_count // 2 or 1))


def invalidate_cached_plan_metadata() -> None:
    global _CACHED_PARTITIONS, _CACHED_PLAN_SUMMARY
    _CACHED_PARTITIONS = None
    _CACHED_PLAN_SUMMARY = None


def _default_db_connection_factory():
    return get_db_connection()


def _sanitize_partition_id(partition_id: str) -> str:
    sanitized = re.sub(r'[^a-zA-Z0-9_]+', '_', str(partition_id))
    return sanitized.strip('_') or 'default'


def get_partition_table_name(partition_id: str) -> str:
    return f'{ADAPTIVE_PARTITION_TABLE_PREFIX}{_sanitize_partition_id(partition_id)}'


def initialize_partition_plan_schema(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS adaptive_tenant_current_plan (
                    plan_id BIGSERIAL PRIMARY KEY,
                    action TEXT NOT NULL,
                    alpha DOUBLE PRECISION NOT NULL,
                    baseline_memory DOUBLE PRECISION NOT NULL,
                    memory_limit DOUBLE PRECISION NOT NULL,
                    current_memory DOUBLE PRECISION NOT NULL,
                    total_query_cost DOUBLE PRECISION NOT NULL,
                    current_epoch BIGINT NOT NULL DEFAULT 0,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS adaptive_tenant_current_partitions (
                    partition_id TEXT PRIMARY KEY,
                    plan_id BIGINT NOT NULL REFERENCES adaptive_tenant_current_plan(plan_id) ON DELETE CASCADE,
                    table_name TEXT NOT NULL,
                    document_count BIGINT NOT NULL,
                    vector_count BIGINT NOT NULL,
                    estimated_memory DOUBLE PRECISION NOT NULL,
                    query_rate DOUBLE PRECISION NOT NULL,
                    write_rate DOUBLE PRECISION NOT NULL,
                    recall_target DOUBLE PRECISION NOT NULL,
                    tenant_ids BIGINT[] NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS adaptive_tenant_current_partition_members (
                    partition_id TEXT NOT NULL REFERENCES adaptive_tenant_current_partitions(partition_id) ON DELETE CASCADE,
                    tenant_id BIGINT NOT NULL,
                    tenant_name TEXT NOT NULL,
                    document_count BIGINT NOT NULL,
                    vector_count BIGINT NOT NULL,
                    query_rate DOUBLE PRECISION NOT NULL,
                    write_rate DOUBLE PRECISION NOT NULL,
                    recall_target DOUBLE PRECISION NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    PRIMARY KEY (partition_id, tenant_id)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS adaptive_tenant_current_operations (
                    operation_id BIGSERIAL PRIMARY KEY,
                    plan_id BIGINT NOT NULL REFERENCES adaptive_tenant_current_plan(plan_id) ON DELETE CASCADE,
                    operation_type TEXT NOT NULL,
                    source_partition_id TEXT,
                    target_partition_id TEXT,
                    tenant_id BIGINT,
                    score DOUBLE PRECISION NOT NULL DEFAULT 0,
                    delta_query_cost DOUBLE PRECISION NOT NULL DEFAULT 0,
                    delta_memory DOUBLE PRECISION NOT NULL DEFAULT 0,
                    reason TEXT NOT NULL DEFAULT '',
                    window_marker BIGINT NOT NULL DEFAULT 0,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_adaptive_tenant_members_tenant
                ON adaptive_tenant_current_partition_members (tenant_id);
                """
            )
            cur.execute(
                "ALTER TABLE adaptive_tenant_current_partitions ADD COLUMN IF NOT EXISTS table_name TEXT;"
            )
            cur.execute(
                """
                UPDATE adaptive_tenant_current_partitions
                SET table_name = COALESCE(table_name, %s || regexp_replace(partition_id, '[^a-zA-Z0-9_]+', '_', 'g'))
                WHERE table_name IS NULL;
                """,
                [ADAPTIVE_PARTITION_TABLE_PREFIX],
            )
        conn.commit()
    finally:
        conn.close()




def initialize_materialization_schema(*, db_connection_factory=_default_db_connection_factory) -> None:
    initialize_partition_plan_schema(db_connection_factory=db_connection_factory)

def clear_current_plan(*, db_connection_factory=_default_db_connection_factory) -> None:
    invalidate_cached_plan_metadata()
    initialize_partition_plan_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM adaptive_tenant_current_plan;')
        conn.commit()
    finally:
        conn.close()


def save_planner_result(
    result: PlannerResult,
    *,
    db_connection_factory=_default_db_connection_factory,
    update_tenant_markers: bool = True,
    tenant_state_repository: Optional[TenantStateRepository] = None,
) -> int:
    invalidate_cached_plan_metadata()
    initialize_partition_plan_schema(db_connection_factory=db_connection_factory)
    tenant_repo = tenant_state_repository or TenantStateRepository(db_connection_factory=db_connection_factory)
    tenant_repo.initialize_schema()
    tenant_ids = sorted({tenant_id for partition in result.partitions for tenant_id in partition.tenant_ids})
    tenant_states = {state.tenant_id: state for state in tenant_repo.get_all_tenant_states(tenant_ids=tenant_ids, window_limit=1)}

    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute('DELETE FROM adaptive_tenant_current_plan;')
            cur.execute(
                """
                INSERT INTO adaptive_tenant_current_plan (
                    action,
                    alpha,
                    baseline_memory,
                    memory_limit,
                    current_memory,
                    total_query_cost,
                    current_epoch,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                RETURNING plan_id;
                """,
                [
                    result.action,
                    result.budget.alpha,
                    result.budget.baseline_memory,
                    result.budget.memory_limit,
                    result.budget.current_memory,
                    result.total_query_cost,
                    result.current_epoch,
                    json.dumps({
                        'new_tenant_id': result.new_tenant_id,
                        'new_partition_id': result.new_partition_id,
                    }),
                ],
            )
            plan_id = int(cur.fetchone()[0])

            for partition in result.partitions:
                table_name = get_partition_table_name(partition.partition_id)
                cur.execute(
                    """
                    INSERT INTO adaptive_tenant_current_partitions (
                        partition_id,
                        plan_id,
                        table_name,
                        document_count,
                        vector_count,
                        estimated_memory,
                        query_rate,
                        write_rate,
                        recall_target,
                        tenant_ids,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
                    """,
                    [
                        partition.partition_id,
                        plan_id,
                        table_name,
                        partition.document_count,
                        partition.vector_count,
                        partition.estimated_memory,
                        partition.query_rate,
                        partition.write_rate,
                        partition.recall_target,
                        list(partition.tenant_ids),
                        json.dumps(partition.metadata),
                    ],
                )
                for tenant_id, tenant_name in zip(partition.tenant_ids, partition.tenant_names):
                    tenant_state = tenant_states.get(int(tenant_id))
                    cur.execute(
                        """
                        INSERT INTO adaptive_tenant_current_partition_members (
                            partition_id,
                            tenant_id,
                            tenant_name,
                            document_count,
                            vector_count,
                            query_rate,
                            write_rate,
                            recall_target,
                            metadata
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
                        """,
                        [
                            partition.partition_id,
                            tenant_id,
                            tenant_name,
                            int(partition.tenant_document_counts.get(tenant_id, 0)),
                            int(partition.tenant_vector_counts.get(tenant_id, 0)),
                            float(tenant_state.query_rate_ema if tenant_state else 0.0),
                            float(tenant_state.write_rate_ema if tenant_state else 0.0),
                            float(tenant_state.recall_target if tenant_state else partition.recall_target),
                            json.dumps({}),
                        ],
                    )

            for step in result.merge_steps:
                cur.execute(
                    """
                    INSERT INTO adaptive_tenant_current_operations (
                        plan_id,
                        operation_type,
                        source_partition_id,
                        target_partition_id,
                        tenant_id,
                        score,
                        delta_query_cost,
                        delta_memory,
                        reason,
                        window_marker,
                        metadata
                    )
                    VALUES (%s, 'merge', %s, %s, NULL, %s, %s, %s, %s, %s, %s::jsonb);
                    """,
                    [
                        plan_id,
                        step.left_partition_id,
                        step.merged_partition_id,
                        step.score,
                        step.delta_query_cost,
                        step.delta_memory,
                        step.reason,
                        step.window_marker,
                        json.dumps({
                            'right_partition_id': step.right_partition_id,
                            'merged_tenant_ids': list(step.merged_tenant_ids),
                            'overlap_ratio': step.overlap_ratio,
                        }),
                    ],
                )

            for step in result.split_steps:
                cur.execute(
                    """
                    INSERT INTO adaptive_tenant_current_operations (
                        plan_id,
                        operation_type,
                        source_partition_id,
                        target_partition_id,
                        tenant_id,
                        score,
                        delta_query_cost,
                        delta_memory,
                        reason,
                        window_marker,
                        metadata
                    )
                    VALUES (%s, 'split', %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
                    """,
                    [
                        plan_id,
                        step.source_partition_id,
                        step.target_partition_id,
                        step.tenant_id,
                        step.score,
                        step.delta_query_cost,
                        step.delta_memory,
                        step.reason,
                        step.window_marker,
                        json.dumps({
                            'mode': step.mode,
                            'gain': step.gain,
                            'source_tenant_ids_after': list(step.source_tenant_ids_after),
                            'target_tenant_ids_after': list(step.target_tenant_ids_after),
                        }),
                    ],
                )

            if update_tenant_markers:
                _update_tenant_operation_markers(cur, result)
        conn.commit()
        return plan_id
    finally:
        conn.close()


def load_current_partitions(
    *,
    db_connection_factory=_default_db_connection_factory,
    refresh: bool = False,
) -> list[PlannedPartition]:
    global _CACHED_PARTITIONS
    if not refresh and _CACHED_PARTITIONS is not None:
        return _CACHED_PARTITIONS

    initialize_partition_plan_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    p.partition_id,
                    p.table_name,
                    p.document_count,
                    p.vector_count,
                    p.estimated_memory,
                    p.query_rate,
                    p.write_rate,
                    p.recall_target,
                    p.metadata,
                    m.tenant_id,
                    m.tenant_name,
                    m.document_count,
                    m.vector_count,
                    m.query_rate,
                    m.write_rate,
                    m.recall_target
                FROM adaptive_tenant_current_partitions p
                LEFT JOIN adaptive_tenant_current_partition_members m
                  ON m.partition_id = p.partition_id
                ORDER BY p.estimated_memory, p.partition_id, m.tenant_id;
                """
            )
            rows = cur.fetchall()

        partitions_by_id: dict[str, PlannedPartition] = {}
        ordered_ids: list[str] = []
        for row in rows:
            partition_id = row[0]
            partition = partitions_by_id.get(partition_id)
            if partition is None:
                metadata = row[8] or {}
                metadata.setdefault('table_name', row[1])
                partition = PlannedPartition(
                    partition_id=partition_id,
                    tenant_ids=(),
                    tenant_names=(),
                    document_count=int(row[2]),
                    vector_count=int(row[3]),
                    estimated_memory=float(row[4]),
                    query_rate=float(row[5]),
                    write_rate=float(row[6]),
                    recall_target=float(row[7]),
                    tenant_document_counts={},
                    tenant_vector_counts={},
                    metadata=metadata,
                )
                partitions_by_id[partition_id] = partition
                ordered_ids.append(partition_id)

            tenant_id = row[9]
            if tenant_id is None:
                continue
            tenant_id = int(tenant_id)
            partition.tenant_ids = tuple(partition.tenant_ids) + (tenant_id,)
            partition.tenant_names = tuple(partition.tenant_names) + (row[10],)
            partition.tenant_document_counts[tenant_id] = int(row[11] or 0)
            partition.tenant_vector_counts[tenant_id] = int(row[12] or 0)

        _CACHED_PARTITIONS = [partitions_by_id[partition_id] for partition_id in ordered_ids]
        return _CACHED_PARTITIONS
    finally:
        conn.close()


def get_current_plan_summary(
    *,
    db_connection_factory=_default_db_connection_factory,
    refresh: bool = False,
) -> Optional[dict]:
    global _CACHED_PLAN_SUMMARY
    if not refresh and _CACHED_PLAN_SUMMARY is not None:
        return _CACHED_PLAN_SUMMARY

    initialize_partition_plan_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT plan_id, action, alpha, baseline_memory, memory_limit,
                       current_memory, total_query_cost, current_epoch, metadata, created_at
                FROM adaptive_tenant_current_plan
                ORDER BY plan_id DESC
                LIMIT 1;
                """
            )
            row = cur.fetchone()
        if row is None:
            _CACHED_PLAN_SUMMARY = None
            return None
        _CACHED_PLAN_SUMMARY = {
            'plan_id': int(row[0]),
            'action': row[1],
            'alpha': float(row[2]),
            'baseline_memory': float(row[3]),
            'memory_limit': float(row[4]),
            'current_memory': float(row[5]),
            'total_query_cost': float(row[6]),
            'current_epoch': int(row[7]),
            'metadata': row[8] or {},
            'created_at': row[9],
        }
        return _CACHED_PLAN_SUMMARY
    finally:
        conn.close()


def list_materialized_partition_tables(*, db_connection_factory=_default_db_connection_factory) -> list[str]:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name LIKE %s
                ORDER BY table_name;
                """,
                [f'{ADAPTIVE_PARTITION_TABLE_PREFIX}%'],
            )
            return [row[0] for row in cur.fetchall()]
    finally:
        conn.close()


def drop_materialized_partitions(
    *,
    valid_partition_ids: Optional[Iterable[str]] = None,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    valid_table_names = None
    if valid_partition_ids is not None:
        valid_table_names = {get_partition_table_name(partition_id) for partition_id in valid_partition_ids}
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT table_name
                FROM information_schema.tables
                WHERE table_name LIKE %s;
                """,
                [f'{ADAPTIVE_PARTITION_TABLE_PREFIX}%'],
            )
            existing = [row[0] for row in cur.fetchall()]
            for table_name in existing:
                if valid_table_names is not None and table_name in valid_table_names:
                    continue
                cur.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE;').format(sql.Identifier(table_name)))
        conn.commit()
    finally:
        conn.close()


def create_index_for_partition(
    table_name: str,
    index_type: str = 'hnsw',
    *,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 64,
    hnsw_threads: Optional[int] = None,
    disable_sync_commit: bool = True,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            _configure_index_session(
                cur,
                disable_sync_commit=disable_sync_commit,
                hnsw_threads=hnsw_threads,
            )
            if index_type.lower() == 'hnsw':
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {} 
                        ON {} USING hnsw (vector vector_l2_ops)
                        WITH (m = {m}, ef_construction = {ef});
                        """
                    ).format(
                        sql.Identifier(f'{table_name}_vector_idx'),
                        sql.Identifier(table_name),
                        m=sql.Literal(int(hnsw_m)),
                        ef=sql.Literal(int(hnsw_ef_construction)),
                    )
                )
            elif index_type.lower() == 'ivfflat':
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {}
                        ON {} USING ivfflat (vector vector_l2_ops);
                        """
                    ).format(
                        sql.Identifier(f'{table_name}_vector_idx'),
                        sql.Identifier(table_name),
                    )
                )
            else:
                raise ValueError(f'Unsupported index_type: {index_type}')
        conn.commit()
    finally:
        conn.close()


def drop_indexes_for_materialized_partitions(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            for table_name in list_materialized_partition_tables(db_connection_factory=db_connection_factory):
                cur.execute(
                    sql.SQL('DROP INDEX IF EXISTS {} CASCADE;').format(
                        sql.Identifier(f'{table_name}_vector_idx')
                    )
                )
        conn.commit()
    finally:
        conn.close()


def _create_index_for_partition_timed(
    table_name: str,
    index_type: str,
    *,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_threads: Optional[int],
    disable_sync_commit: bool,
) -> tuple[str, float]:
    started_at = time.time()
    create_index_for_partition(
        table_name,
        index_type=index_type,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
        hnsw_threads=hnsw_threads,
        disable_sync_commit=disable_sync_commit,
    )
    return table_name, time.time() - started_at


def create_indexes_for_materialized_partitions(
    index_type: str = 'hnsw',
    *,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 64,
    hnsw_threads: Optional[int] = None,
    disable_sync_commit: bool = True,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    maintenance_settings = get_maintenance_settings()
    table_names = list_materialized_partition_tables(db_connection_factory=db_connection_factory)
    print(
        'PostgreSQL parameters set: maintenance_work_mem = '
        f"{maintenance_settings['maintenance_work_mem_gb']}GB, "
        f"max_parallel_maintenance_workers = {maintenance_settings['max_parallel_maintenance_workers']}",
        flush=True,
    )
    if not table_names:
        print('AdaptiveTenant index build: no materialized partitions found. Skipping.', flush=True)
        return

    print(
        f'AdaptiveTenant index build: creating {index_type} indexes for {len(table_names)} materialized partitions...',
        flush=True,
    )
    started_at = time.time()
    worker_count = _default_parallel_worker_count(max_workers)
    can_parallel = parallel and worker_count > 1 and db_connection_factory is _default_db_connection_factory

    if can_parallel:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_table = {
                executor.submit(
                    _create_index_for_partition_timed,
                    table_name,
                    index_type,
                    hnsw_m=hnsw_m,
                    hnsw_ef_construction=hnsw_ef_construction,
                    hnsw_threads=hnsw_threads,
                    disable_sync_commit=disable_sync_commit,
                ): table_name
                for table_name in table_names
            }
            completed = 0
            for future in as_completed(future_to_table):
                completed += 1
                table_name, elapsed = future.result()
                print(
                    f'AdaptiveTenant index build: [{completed}/{len(table_names)}] finished {table_name} in {elapsed:.2f}s',
                    flush=True,
                )
    else:
        for idx, table_name in enumerate(table_names, start=1):
            table_start = time.time()
            print(
                f'AdaptiveTenant index build: [{idx}/{len(table_names)}] {table_name}',
                flush=True,
            )
            create_index_for_partition(
                table_name,
                index_type=index_type,
                hnsw_m=hnsw_m,
                hnsw_ef_construction=hnsw_ef_construction,
                hnsw_threads=hnsw_threads,
                disable_sync_commit=disable_sync_commit,
                db_connection_factory=db_connection_factory,
            )
            print(
                f'AdaptiveTenant index build: finished {table_name} in {time.time() - table_start:.2f}s',
                flush=True,
            )

    print(
        f'AdaptiveTenant index build: all indexes completed in {time.time() - started_at:.2f}s',
        flush=True,
    )


def ensure_user_accessible_documents_view(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute('GRANT SELECT ON PermissionAssignment TO PUBLIC;')
            cur.execute('GRANT SELECT ON UserRoles TO PUBLIC;')
            cur.execute('GRANT SELECT ON DocumentBlocks TO PUBLIC;')
            cur.execute('DROP MATERIALIZED VIEW IF EXISTS user_accessible_documents;')
            cur.execute(
                """
                CREATE MATERIALIZED VIEW user_accessible_documents AS
                SELECT ur.user_id, pa.document_id
                FROM UserRoles ur
                JOIN PermissionAssignment pa ON ur.role_id = pa.role_id
                GROUP BY ur.user_id, pa.document_id;
                """
            )
            cur.execute(
                'CREATE INDEX IF NOT EXISTS user_accessible_documents_idx ON user_accessible_documents (user_id, document_id);'
            )
            cur.execute('GRANT SELECT ON user_accessible_documents TO PUBLIC;')
        conn.commit()
    finally:
        conn.close()


def initialize_rls_for_materialized_partitions(*, db_connection_factory=_default_db_connection_factory) -> None:
    ensure_user_accessible_documents_view(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            for table_name in list_materialized_partition_tables(db_connection_factory=db_connection_factory):
                cur.execute(sql.SQL('GRANT SELECT ON {} TO PUBLIC;').format(sql.Identifier(table_name)))
                cur.execute(sql.SQL('ALTER TABLE {} ENABLE ROW LEVEL SECURITY;').format(sql.Identifier(table_name)))
                cur.execute(sql.SQL('ALTER TABLE {} FORCE ROW LEVEL SECURITY;').format(sql.Identifier(table_name)))
                cur.execute(
                    sql.SQL('DROP POLICY IF EXISTS {} ON {};').format(
                        sql.Identifier(ADAPTIVE_POLICY_NAME),
                        sql.Identifier(table_name),
                    )
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE POLICY {} ON {}
                        FOR SELECT
                        USING (
                            EXISTS (
                                SELECT 1
                                FROM user_accessible_documents uad
                                WHERE uad.document_id = {}.document_id
                                  AND uad.user_id = current_user::int
                            )
                        );
                        """
                    ).format(
                        sql.Identifier(ADAPTIVE_POLICY_NAME),
                        sql.Identifier(table_name),
                        sql.Identifier(table_name),
                    )
                )
        conn.commit()
    finally:
        conn.close()


def disable_rls_for_materialized_partitions(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            for table_name in list_materialized_partition_tables(db_connection_factory=db_connection_factory):
                cur.execute(sql.SQL('ALTER TABLE {} DISABLE ROW LEVEL SECURITY;').format(sql.Identifier(table_name)))
                cur.execute(
                    sql.SQL('DROP POLICY IF EXISTS {} ON {};').format(
                        sql.Identifier(ADAPTIVE_POLICY_NAME),
                        sql.Identifier(table_name),
                    )
                )
        conn.commit()
    finally:
        conn.close()


def _resolve_partition_document_ids(
    partition: PlannedPartition,
    *,
    tenant_state_repository: Optional[TenantStateRepository] = None,
    document_cache: Optional[dict[int, set[int]]] = None,
    db_connection_factory=_default_db_connection_factory,
) -> list[int]:
    tenant_repo = tenant_state_repository or TenantStateRepository(db_connection_factory=db_connection_factory)
    if partition.document_ids:
        return sorted(partition.document_ids)
    if document_cache is not None:
        return sorted(set().union(*(document_cache.get(tenant_id, set()) for tenant_id in partition.tenant_ids)))
    cached_docs = tenant_repo.get_many_accessible_document_ids(partition.tenant_ids)
    return sorted(set().union(*(cached_docs.get(tenant_id, set()) for tenant_id in partition.tenant_ids)))


def _materialize_partition_table(
    table_name: str,
    document_ids: Sequence[int],
    *,
    db_connection_factory=_default_db_connection_factory,
) -> str:
    vector_dimension = get_document_vector_dimension()
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE;').format(sql.Identifier(table_name)))
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE {} (
                        block_id BIGINT NOT NULL,
                        document_id INT NOT NULL REFERENCES Documents(document_id),
                        block_content BYTEA NOT NULL,
                        vector VECTOR({dimension}),
                        PRIMARY KEY (block_id, document_id)
                    );
                    """
                ).format(
                    sql.Identifier(table_name),
                    dimension=sql.SQL(str(vector_dimension)),
                )
            )
            if document_ids:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (block_id, document_id, block_content, vector)
                        SELECT block_id, document_id, block_content, vector
                        FROM documentblocks
                        WHERE document_id = ANY(%s);
                        """
                    ).format(sql.Identifier(table_name)),
                    [list(document_ids)],
                )
            cur.execute(sql.SQL('ANALYZE {};').format(sql.Identifier(table_name)))
        conn.commit()
    finally:
        conn.close()
    return table_name


def _materialize_partition_worker(partition_id: str, document_ids: Sequence[int]) -> tuple[str, int, float]:
    started_at = time.time()
    table_name = get_partition_table_name(partition_id)
    _materialize_partition_table(table_name, document_ids)
    return table_name, len(document_ids), time.time() - started_at


def materialize_planner_result(
    result: PlannerResult,
    *,
    index_type: str = 'hnsw',
    create_indexes: bool = True,
    enable_rls: bool = False,
    db_connection_factory=_default_db_connection_factory,
    tenant_state_repository: Optional[TenantStateRepository] = None,
) -> int:
    plan_id = save_planner_result(
        result,
        db_connection_factory=db_connection_factory,
        tenant_state_repository=tenant_state_repository,
    )
    materialize_partitions(
        result.partitions,
        index_type=index_type,
        create_indexes=create_indexes,
        enable_rls=enable_rls,
        db_connection_factory=db_connection_factory,
        tenant_state_repository=tenant_state_repository,
    )
    return plan_id


def materialize_current_plan(
    *,
    index_type: str = 'hnsw',
    create_indexes: bool = True,
    enable_rls: bool = False,
    db_connection_factory=_default_db_connection_factory,
    tenant_state_repository: Optional[TenantStateRepository] = None,
) -> list[str]:
    partitions = load_current_partitions(db_connection_factory=db_connection_factory)
    return materialize_partitions(
        partitions,
        index_type=index_type,
        create_indexes=create_indexes,
        enable_rls=enable_rls,
        db_connection_factory=db_connection_factory,
        tenant_state_repository=tenant_state_repository,
    )


def materialize_partitions(
    partitions: Sequence[PlannedPartition],
    *,
    index_type: str = 'hnsw',
    create_indexes: bool = True,
    enable_rls: bool = False,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    db_connection_factory=_default_db_connection_factory,
    tenant_state_repository: Optional[TenantStateRepository] = None,
) -> list[str]:
    tenant_repo = tenant_state_repository or TenantStateRepository(db_connection_factory=db_connection_factory)
    print(f'AdaptiveTenant materialization: loaded {len(partitions)} planned partitions.')
    valid_partition_ids = [partition.partition_id for partition in partitions]
    drop_materialized_partitions(
        valid_partition_ids=valid_partition_ids,
        db_connection_factory=db_connection_factory,
    )
    tenant_ids = sorted({tenant_id for partition in partitions for tenant_id in partition.tenant_ids})
    document_cache = tenant_repo.get_many_accessible_document_ids(tenant_ids)
    partition_inputs = [
        (partition, _resolve_partition_document_ids(
            partition,
            tenant_state_repository=tenant_repo,
            document_cache=document_cache,
            db_connection_factory=db_connection_factory,
        ))
        for partition in partitions
    ]
    table_names: list[str] = []
    worker_count = _default_parallel_worker_count(max_workers)
    can_parallel = parallel and worker_count > 1 and db_connection_factory is _default_db_connection_factory

    if can_parallel and partition_inputs:
        print(
            f'AdaptiveTenant materialization: building {len(partition_inputs)} partitions in parallel with {worker_count} workers...',
            flush=True,
        )
        partition_to_index = {partition.partition_id: idx for idx, (partition, _) in enumerate(partition_inputs, start=1)}
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_partition = {
                executor.submit(_materialize_partition_worker, partition.partition_id, document_ids): (partition, len(document_ids))
                for partition, document_ids in partition_inputs
            }
            for future in as_completed(future_to_partition):
                partition, document_count = future_to_partition[future]
                table_name, resolved_count, elapsed = future.result()
                print(
                    f'AdaptiveTenant materialization: [{partition_to_index[partition.partition_id]}/{len(partition_inputs)}] finished {partition.partition_id} with {resolved_count} documents in {elapsed:.2f}s',
                    flush=True,
                )
                table_names.append(table_name)
    else:
        for idx, (partition, document_ids) in enumerate(partition_inputs, start=1):
            print(f'AdaptiveTenant materialization: building partition {idx}/{len(partition_inputs)} -> {partition.partition_id}')
            table_name = _materialize_partition_table(
                get_partition_table_name(partition.partition_id),
                document_ids,
                db_connection_factory=db_connection_factory,
            )
            table_names.append(table_name)

    if create_indexes:
        print('AdaptiveTenant materialization: creating indexes for all materialized partitions...')
        create_indexes_for_materialized_partitions(
            index_type=index_type,
            max_workers=max_workers,
            db_connection_factory=db_connection_factory,
        )
    if enable_rls:
        print('AdaptiveTenant materialization: enabling RLS for all materialized partitions...')
        initialize_rls_for_materialized_partitions(db_connection_factory=db_connection_factory)
    return table_names


def materialize_partition(
    partition: PlannedPartition,
    *,
    tenant_state_repository: Optional[TenantStateRepository] = None,
    document_cache: Optional[dict[int, set[int]]] = None,
    db_connection_factory=_default_db_connection_factory,
) -> str:
    document_ids = _resolve_partition_document_ids(
        partition,
        tenant_state_repository=tenant_state_repository,
        document_cache=document_cache,
        db_connection_factory=db_connection_factory,
    )
    print(f'AdaptiveTenant partition {partition.partition_id}: {len(partition.tenant_ids)} tenants, {len(document_ids)} unique documents')
    return _materialize_partition_table(
        get_partition_table_name(partition.partition_id),
        document_ids,
        db_connection_factory=db_connection_factory,
    )


def _update_tenant_operation_markers(cur, result: PlannerResult) -> None:
    touched_merges = {}
    for step in result.merge_steps:
        for tenant_id in step.merged_tenant_ids:
            touched_merges[int(tenant_id)] = int(step.window_marker)

    touched_splits = {}
    for step in result.split_steps:
        touched_splits[int(step.tenant_id)] = int(step.window_marker)

    for tenant_id, window_marker in touched_merges.items():
        cur.execute(
            """
            INSERT INTO adaptive_tenant_profiles (tenant_id, tenant_name, recall_target, metadata)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (tenant_id) DO UPDATE
            SET metadata = adaptive_tenant_profiles.metadata || %s::jsonb,
                updated_at = NOW();
            """,
            [
                tenant_id,
                f'tenant_{tenant_id}',
                0.95,
                json.dumps({'last_merge_window': window_marker}),
                json.dumps({'last_merge_window': window_marker}),
            ],
        )

    for tenant_id, window_marker in touched_splits.items():
        cur.execute(
            """
            INSERT INTO adaptive_tenant_profiles (tenant_id, tenant_name, recall_target, metadata)
            VALUES (%s, %s, %s, %s::jsonb)
            ON CONFLICT (tenant_id) DO UPDATE
            SET metadata = adaptive_tenant_profiles.metadata || %s::jsonb,
                updated_at = NOW();
            """,
            [
                tenant_id,
                f'tenant_{tenant_id}',
                0.95,
                json.dumps({'last_split_window': window_marker}),
                json.dumps({'last_split_window': window_marker}),
            ],
        )



def build_and_materialize_adaptive_plan(
    *,
    alpha: float = 0.5,
    topk: int = 10,
    window_limit: int = 10,
    tenant_ids: Optional[Iterable[int]] = None,
    index_type: str = 'hnsw',
    create_indexes: bool = True,
    enable_rls: bool = False,
    max_split_actions: int = 0,
    prefer_dedicated_on_budget: bool = True,
    split_threshold: float = 0.0,
    merge_threshold: float = 0.0,
    split_cooldown_windows: int = 1,
    merge_cooldown_windows: int = 1,
    db_connection_factory=_default_db_connection_factory,
    tenant_state_repository: Optional[TenantStateRepository] = None,
) -> PlannerResult:
    tenant_repo = tenant_state_repository or TenantStateRepository(db_connection_factory=db_connection_factory)
    tenant_repo.initialize_schema()
    print('AdaptiveTenant build: loading tenant state and planning partitions...')
    planner = AdaptiveTenantPlanner(
        alpha=alpha,
        tenant_state_repository=tenant_repo,
        topk=topk,
        prefer_dedicated_on_budget=prefer_dedicated_on_budget,
        split_threshold=split_threshold,
        merge_threshold=merge_threshold,
        split_cooldown_windows=split_cooldown_windows,
        merge_cooldown_windows=merge_cooldown_windows,
    )
    result = planner.initialize_plan(tenant_ids=tenant_ids, window_limit=window_limit)
    print(f'AdaptiveTenant build: initial plan has {len(result.partitions)} partitions.')
    if max_split_actions > 0:
        tenant_states = tenant_repo.get_all_tenant_states(tenant_ids=tenant_ids, window_limit=window_limit)
        print(f'AdaptiveTenant build: applying up to {max_split_actions} split actions...')
        result = planner.rebalance_partitions(
            result.partitions,
            tenant_states=tenant_states,
            max_split_actions=max_split_actions,
        )
    print('AdaptiveTenant build: materializing current plan...')
    materialize_planner_result(
        result,
        index_type=index_type,
        create_indexes=create_indexes,
        enable_rls=enable_rls,
        db_connection_factory=db_connection_factory,
        tenant_state_repository=tenant_repo,
    )
    return result
