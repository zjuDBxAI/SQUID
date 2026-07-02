from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import time
from typing import Optional

from psycopg2 import sql
from psycopg2.extras import execute_values
from tqdm import tqdm

from controller.dynamic_partition.load_result_to_database import _configure_index_session
from services.config import get_db_connection, get_document_vector_dimension, get_maintenance_settings

from .common import (
    KMEANS_PARTITION_TABLE_PREFIX,
    KMeansPartition,
    KMeansPlan,
    PARTITION_TABLE,
    PATTERN_TABLE,
    PLAN_TABLE,
    ROUTE_TABLE,
    TenantRoute,
)
from .hybrid_planner import HybridACLKMeansPlanner
from .repository import KMeansRepository

_CACHED_PLAN_SUMMARY: Optional[dict[str, object]] = None
_CACHED_PARTITIONS: Optional[list[KMeansPartition]] = None
_CACHED_TENANT_ROUTES: dict[int, list[TenantRoute]] = {}
_POSTGRES_IDENTIFIER_LIMIT = 63
_PARTITION_BATCH_SIZE = 8192
_DEFAULT_INDEX_MAX_WORKERS = 6
_MATERIALIZE_ADVISORY_LOCK_KEY = 2026051201


def _default_db_connection_factory():
    return get_db_connection()


def _safe_index_name(table_name: str, suffix: str) -> str:
    candidate = f"{table_name}_{suffix}"
    if len(candidate) <= _POSTGRES_IDENTIFIER_LIMIT:
        return candidate
    digest = hashlib.blake2b(candidate.encode("utf-8"), digest_size=8).hexdigest()
    compact_suffix = suffix.replace("_", "")
    return f"idx_{digest}_{compact_suffix}"[:_POSTGRES_IDENTIFIER_LIMIT]


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(child) for child in value]
    return value


def _json_dumps(value) -> str:
    return json.dumps(_json_safe(value), allow_nan=False)


def _drop_non_constraint_indexes_for_table(cur, table_name: str) -> None:
    cur.execute(
        """
        SELECT index_class.relname
        FROM pg_index idx
        JOIN pg_class table_class ON table_class.oid = idx.indrelid
        JOIN pg_namespace ns ON ns.oid = table_class.relnamespace
        JOIN pg_class index_class ON index_class.oid = idx.indexrelid
        WHERE ns.nspname = current_schema()
          AND table_class.relname = %s
          AND NOT idx.indisprimary
          AND NOT idx.indisunique;
        """,
        [str(table_name)],
    )
    for (index_name,) in cur.fetchall():
        cur.execute(sql.SQL("DROP INDEX IF EXISTS {} CASCADE;").format(sql.Identifier(str(index_name))))


def _recommended_worker_count(task_count: int, *, max_workers: Optional[int] = None) -> int:
    if task_count <= 1:
        return 1
    if max_workers is not None:
        return max(1, min(int(max_workers), int(task_count)))
    cpu_count = max(1, int(os.cpu_count() or 1))
    return max(1, min(int(task_count), _DEFAULT_INDEX_MAX_WORKERS, max(1, cpu_count // 4)))


def invalidate_cache() -> None:
    global _CACHED_PLAN_SUMMARY, _CACHED_PARTITIONS, _CACHED_TENANT_ROUTES
    _CACHED_PLAN_SUMMARY = None
    _CACHED_PARTITIONS = None
    _CACHED_TENANT_ROUTES = {}


def _acquire_materialize_lock(*, db_connection_factory=_default_db_connection_factory):
    conn = db_connection_factory()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_lock(%s);", [_MATERIALIZE_ADVISORY_LOCK_KEY])
    except Exception:
        conn.close()
        raise
    return conn


def _release_materialize_lock(conn) -> None:
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT pg_advisory_unlock(%s);", [_MATERIALIZE_ADVISORY_LOCK_KEY])
    finally:
        conn.close()


def initialize_schema(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {PLAN_TABLE} (
                    plan_id BIGSERIAL PRIMARY KEY,
                    cluster_count INTEGER NOT NULL,
                    tenant_count INTEGER NOT NULL,
                    partition_count INTEGER NOT NULL,
                    document_count BIGINT NOT NULL,
                    vector_count BIGINT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {PARTITION_TABLE} (
                    partition_id TEXT PRIMARY KEY,
                    plan_id BIGINT NOT NULL REFERENCES {PLAN_TABLE}(plan_id) ON DELETE CASCADE,
                    cluster_id INTEGER NOT NULL,
                    partition_kind TEXT NOT NULL DEFAULT 'private',
                    table_name TEXT NOT NULL,
                    document_count BIGINT NOT NULL,
                    vector_count BIGINT NOT NULL,
                    tenant_ids BIGINT[] NOT NULL,
                    pattern_ids BIGINT[] NOT NULL,
                    document_ids BIGINT[] NOT NULL,
                    document_pattern_pairs JSONB NOT NULL DEFAULT '[]'::jsonb,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                );
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {PATTERN_TABLE} (
                    pattern_id BIGINT NOT NULL,
                    plan_id BIGINT NOT NULL REFERENCES {PLAN_TABLE}(plan_id) ON DELETE CASCADE,
                    tenant_ids BIGINT[] NOT NULL,
                    document_ids BIGINT[] NOT NULL,
                    document_count BIGINT NOT NULL,
                    vector_count BIGINT NOT NULL,
                    weight DOUBLE PRECISION NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    PRIMARY KEY (plan_id, pattern_id)
                );
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {ROUTE_TABLE} (
                    tenant_id BIGINT NOT NULL,
                    plan_id BIGINT NOT NULL REFERENCES {PLAN_TABLE}(plan_id) ON DELETE CASCADE,
                    cluster_id INTEGER NOT NULL,
                    route_kind TEXT NOT NULL,
                    partition_id TEXT NOT NULL REFERENCES {PARTITION_TABLE}(partition_id) ON DELETE CASCADE,
                    table_name TEXT NOT NULL,
                    pattern_ids BIGINT[] NOT NULL,
                    PRIMARY KEY (plan_id, tenant_id, partition_id)
                );
                """
            )
            cur.execute(
                sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS partition_kind TEXT NOT NULL DEFAULT 'private';").format(
                    sql.Identifier(PARTITION_TABLE)
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS route_kind TEXT NOT NULL DEFAULT 'private';").format(
                    sql.Identifier(ROUTE_TABLE)
                )
            )
            cur.execute(
                sql.SQL("ALTER TABLE {} ADD COLUMN IF NOT EXISTS pattern_ids BIGINT[] NOT NULL DEFAULT '{{}}';").format(
                    sql.Identifier(ROUTE_TABLE)
                )
            )
            cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (cluster_id);").format(
                sql.Identifier(f"idx_{ROUTE_TABLE}_cluster"),
                sql.Identifier(ROUTE_TABLE),
            ))
            cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (tenant_id);").format(
                sql.Identifier(f"idx_{ROUTE_TABLE}_tenant"),
                sql.Identifier(ROUTE_TABLE),
            ))
            cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (tenant_ids);").format(
                sql.Identifier(f"idx_{PATTERN_TABLE}_tenant_ids"),
                sql.Identifier(PATTERN_TABLE),
            ))
        conn.commit()
    finally:
        conn.close()


def clear_current_plan(*, db_connection_factory=_default_db_connection_factory) -> None:
    invalidate_cache()
    initialize_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {};").format(sql.Identifier(PLAN_TABLE)))
        conn.commit()
    finally:
        conn.close()


def save_plan(plan: KMeansPlan, *, db_connection_factory=_default_db_connection_factory) -> int:
    invalidate_cache()
    initialize_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {};").format(sql.Identifier(PLAN_TABLE)))
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (
                        cluster_count, tenant_count, partition_count,
                        document_count, vector_count, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING plan_id;
                    """
                ).format(sql.Identifier(PLAN_TABLE)),
                [
                    int(plan.metadata.get("cluster_count", len(plan.partitions))),
                    int(plan.metadata.get("tenant_count", len(plan.tenant_to_cluster))),
                    int(len(plan.partitions)),
                    int(plan.metadata.get("document_count", 0)),
                    int(plan.metadata.get("partition_vector_count", 0)),
                    _json_dumps(plan.metadata),
                ],
            )
            plan_id = int(cur.fetchone()[0])
            pattern_rows = [
                (
                    int(pattern.pattern_id),
                    plan_id,
                    list(pattern.tenant_ids),
                    list(pattern.document_ids),
                    int(pattern.document_count),
                    int(pattern.vector_count),
                    float(pattern.weight),
                    _json_dumps({"score": float(pattern.score), "zone": str(pattern.zone)}),
                )
                for pattern in plan.patterns
            ]
            if pattern_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {PATTERN_TABLE} (
                        pattern_id, plan_id, tenant_ids, document_ids,
                        document_count, vector_count, weight, metadata
                    )
                    VALUES %s;
                    """,
                    pattern_rows,
                    template="(%s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                )
            partition_rows = [
                (
                    partition.partition_id,
                    plan_id,
                    int(partition.cluster_id),
                    str(partition.partition_kind),
                    partition.table_name,
                    int(partition.document_count),
                    int(partition.vector_count),
                    list(partition.tenant_ids),
                    list(partition.pattern_ids),
                    list(partition.document_ids),
                    _json_dumps([[int(d), int(p)] for d, p in partition.document_pattern_pairs]),
                    _json_dumps(partition.metadata),
                )
                for partition in plan.partitions
            ]
            if partition_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {PARTITION_TABLE} (
                        partition_id, plan_id, cluster_id, partition_kind, table_name,
                        document_count, vector_count, tenant_ids, pattern_ids,
                        document_ids, document_pattern_pairs, metadata
                    )
                    VALUES %s;
                    """,
                    partition_rows,
                    template="(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)",
                )
            route_rows = [
                (
                    int(route.tenant_id),
                    plan_id,
                    int(route.cluster_id),
                    str(route.route_kind),
                    str(route.partition_id),
                    str(route.table_name),
                    list(route.pattern_ids),
                )
                for route in plan.tenant_routes
            ]
            if route_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {ROUTE_TABLE} (
                        tenant_id, plan_id, cluster_id, route_kind,
                        partition_id, table_name, pattern_ids
                    )
                    VALUES %s;
                    """,
                    route_rows,
                )
        conn.commit()
        return plan_id
    finally:
        conn.close()

def get_current_plan_summary(*, refresh: bool = False, db_connection_factory=_default_db_connection_factory) -> Optional[dict[str, object]]:
    global _CACHED_PLAN_SUMMARY
    if not refresh and _CACHED_PLAN_SUMMARY is not None:
        return _CACHED_PLAN_SUMMARY
    initialize_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT plan_id, cluster_count, tenant_count, partition_count,
                           document_count, vector_count, metadata
                    FROM {}
                    ORDER BY plan_id DESC
                    LIMIT 1;
                    """
                ).format(sql.Identifier(PLAN_TABLE))
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        _CACHED_PLAN_SUMMARY = None
        return None
    _CACHED_PLAN_SUMMARY = {
        "plan_id": int(row[0]),
        "cluster_count": int(row[1]),
        "tenant_count": int(row[2]),
        "partition_count": int(row[3]),
        "document_count": int(row[4]),
        "vector_count": int(row[5]),
        "metadata": dict(row[6] or {}),
    }
    return _CACHED_PLAN_SUMMARY


def load_current_partitions(*, refresh: bool = False, db_connection_factory=_default_db_connection_factory) -> list[KMeansPartition]:
    global _CACHED_PARTITIONS
    if not refresh and _CACHED_PARTITIONS is not None:
        return _CACHED_PARTITIONS
    plan_summary = get_current_plan_summary(refresh=refresh, db_connection_factory=db_connection_factory)
    if plan_summary is None:
        _CACHED_PARTITIONS = []
        return []
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT partition_id, cluster_id, partition_kind, table_name, tenant_ids,
                           pattern_ids, document_ids, document_pattern_pairs,
                           vector_count, metadata
                    FROM {}
                    WHERE plan_id = %s
                    ORDER BY cluster_id;
                    """
                ).format(sql.Identifier(PARTITION_TABLE)),
                [int(plan_summary["plan_id"])],
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    partitions: list[KMeansPartition] = []
    for partition_id, cluster_id, partition_kind, table_name, tenant_ids, pattern_ids, document_ids, pairs, vector_count, metadata in rows:
        partitions.append(
            KMeansPartition(
                partition_id=str(partition_id),
                cluster_id=int(cluster_id),
                partition_kind=str(partition_kind),
                table_name=str(table_name),
                tenant_ids=tuple(int(value) for value in (tenant_ids or ())),
                pattern_ids=tuple(int(value) for value in (pattern_ids or ())),
                document_ids=tuple(int(value) for value in (document_ids or ())),
                document_pattern_pairs=tuple((int(left), int(right)) for left, right in (pairs or [])),
                vector_count=int(vector_count),
                metadata=dict(metadata or {}),
            )
        )
    _CACHED_PARTITIONS = partitions
    return partitions


def load_tenant_routes(tenant_id: int, *, refresh: bool = False, db_connection_factory=_default_db_connection_factory) -> list[TenantRoute]:
    global _CACHED_TENANT_ROUTES
    tenant_id = int(tenant_id)
    if not refresh and tenant_id in _CACHED_TENANT_ROUTES:
        return _CACHED_TENANT_ROUTES[tenant_id]
    plan_summary = get_current_plan_summary(refresh=refresh, db_connection_factory=db_connection_factory)
    if plan_summary is None:
        _CACHED_TENANT_ROUTES[tenant_id] = []
        return []
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        r.tenant_id,
                        r.partition_id,
                        r.table_name,
                        r.route_kind,
                        r.cluster_id,
                        r.pattern_ids,
                        p.vector_count AS partition_vector_count,
                        COALESCE(SUM(ap.vector_count), 0)::BIGINT AS accessible_vector_count
                    FROM {} r
                    JOIN {} p
                      ON p.plan_id = r.plan_id
                     AND p.partition_id = r.partition_id
                    LEFT JOIN {} ap
                      ON ap.plan_id = r.plan_id
                     AND ap.pattern_id = ANY(r.pattern_ids)
                    WHERE r.plan_id = %s
                      AND r.tenant_id = %s
                    GROUP BY
                        r.tenant_id, r.partition_id, r.table_name, r.route_kind,
                        r.cluster_id, r.pattern_ids, p.vector_count
                    ORDER BY r.route_kind, r.cluster_id, r.partition_id;
                    """
                ).format(
                    sql.Identifier(ROUTE_TABLE),
                    sql.Identifier(PARTITION_TABLE),
                    sql.Identifier(PATTERN_TABLE),
                ),
                [int(plan_summary["plan_id"]), int(tenant_id)],
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    routes = [
        TenantRoute(
            tenant_id=int(row[0]),
            partition_id=str(row[1]),
            table_name=str(row[2]),
            route_kind=str(row[3]),
            cluster_id=int(row[4]),
            pattern_ids=tuple(int(value) for value in (row[5] or ())),
            partition_vector_count=int(row[6] or 0),
            accessible_vector_count=int(row[7] or 0),
        )
        for row in rows
    ]
    _CACHED_TENANT_ROUTES[tenant_id] = routes
    return routes


def load_tenant_pattern_ids(tenant_id: int, *, db_connection_factory=_default_db_connection_factory) -> tuple[int, ...]:
    plan_summary = get_current_plan_summary(db_connection_factory=db_connection_factory)
    if plan_summary is None:
        return ()
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT pattern_id
                    FROM {}
                    WHERE plan_id = %s
                      AND tenant_ids @> ARRAY[%s]::BIGINT[]
                    ORDER BY pattern_id;
                    """
                ).format(sql.Identifier(PATTERN_TABLE)),
                [int(plan_summary["plan_id"]), int(tenant_id)],
            )
            return tuple(int(row[0]) for row in cur.fetchall())
    finally:
        conn.close()


def list_materialized_partition_tables(*, db_connection_factory=_default_db_connection_factory) -> list[str]:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = current_schema()
                  AND tablename LIKE %s
                ORDER BY tablename;
                """,
                [f"{KMEANS_PARTITION_TABLE_PREFIX}%"],
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def list_current_plan_partition_tables(*, db_connection_factory=_default_db_connection_factory) -> list[str]:
    plan_summary = get_current_plan_summary(db_connection_factory=db_connection_factory)
    if plan_summary is None:
        return []
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT table_name
                    FROM {}
                    WHERE plan_id = %s
                    ORDER BY cluster_id, partition_id;
                    """
                ).format(sql.Identifier(PARTITION_TABLE)),
                [int(plan_summary["plan_id"])],
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def materialize_partition(partition: KMeansPartition, *, db_connection_factory=_default_db_connection_factory) -> str:
    conn = db_connection_factory()
    vector_dimension = get_document_vector_dimension()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(partition.table_name)))
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE {} (
                        block_id BIGINT NOT NULL,
                        document_id INT NOT NULL REFERENCES Documents(document_id),
                        pattern_id INT NOT NULL,
                        block_content BYTEA NOT NULL,
                        vector VECTOR({dimension}),
                        PRIMARY KEY (block_id, document_id)
                    );
                    """
                ).format(sql.Identifier(partition.table_name), dimension=sql.SQL(str(vector_dimension)))
            )
            cur.execute(
                """
                CREATE TEMP TABLE IF NOT EXISTS temp_kmeans_partition_document_patterns (
                    document_id BIGINT NOT NULL,
                    pattern_id BIGINT NOT NULL
                ) ON COMMIT DROP;
                """
            )
            cur.execute("TRUNCATE TABLE temp_kmeans_partition_document_patterns;")
            execute_values(
                cur,
                """
                INSERT INTO temp_kmeans_partition_document_patterns (document_id, pattern_id)
                VALUES %s;
                """,
                [(int(document_id), int(pattern_id)) for document_id, pattern_id in partition.document_pattern_pairs],
                page_size=_PARTITION_BATCH_SIZE,
            )
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (block_id, document_id, pattern_id, block_content, vector)
                    SELECT db.block_id, db.document_id, tkp.pattern_id, db.block_content, db.vector
                    FROM documentblocks db
                    JOIN temp_kmeans_partition_document_patterns tkp
                      ON tkp.document_id = db.document_id;
                    """
                ).format(sql.Identifier(partition.table_name))
            )
        conn.commit()
        return partition.table_name
    finally:
        conn.close()


def drop_stale_materialized_partitions(valid_table_names: set[str], *, db_connection_factory=_default_db_connection_factory) -> None:
    existing = set(list_materialized_partition_tables(db_connection_factory=db_connection_factory))
    for table_name in sorted(existing - set(valid_table_names)):
        conn = db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name)))
            conn.commit()
        finally:
            conn.close()


def materialize_plan(
    plan: KMeansPlan,
    *,
    create_indexes: bool = False,
    index_type: str = "squidhnsw",
    show_progress: bool = True,
    db_connection_factory=_default_db_connection_factory,
) -> KMeansPlan:
    print("[kmeans][materialize] waiting for global materialization lock...", flush=True)
    lock_conn = _acquire_materialize_lock(db_connection_factory=db_connection_factory)
    print("[kmeans][materialize] acquired global materialization lock", flush=True)
    try:
        print(f"[kmeans][materialize] saving metadata for {len(plan.partitions)} partitions...", flush=True)
        save_plan(plan, db_connection_factory=db_connection_factory)
        valid_table_names = {str(partition.table_name) for partition in plan.partitions}
        drop_stale_materialized_partitions(valid_table_names, db_connection_factory=db_connection_factory)
        iterator = tqdm(
            list(enumerate(plan.partitions, start=1)),
            desc="KMeans materialize partitions",
            unit="partition",
            disable=not show_progress,
        )
        for index, partition in iterator:
            started_at = time.time()
            table_name = materialize_partition(partition, db_connection_factory=db_connection_factory)
            elapsed = time.time() - started_at
            print(
                f"[kmeans][materialize] [{index}/{len(plan.partitions)}] materialized {table_name} in {elapsed:.2f}s",
                flush=True,
            )
        if create_indexes:
            create_indexes_for_materialized_partitions(index_type=index_type, db_connection_factory=db_connection_factory)
        invalidate_cache()
        return plan
    finally:
        _release_materialize_lock(lock_conn)


def create_index_for_partition(
    table_name: str,
    index_type: str = "squidhnsw",
    *,
    hnsw_m: int = 16,
    hnsw_ef_construction: int = 64,
    disable_sync_commit: bool = True,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            _configure_index_session(cur, disable_sync_commit=disable_sync_commit, hnsw_threads=None)
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (pattern_id);").format(
                    sql.Identifier(_safe_index_name(table_name, "pattern_idx")),
                    sql.Identifier(table_name),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (pattern_id, document_id);").format(
                    sql.Identifier(_safe_index_name(table_name, "pattern_document_idx")),
                    sql.Identifier(table_name),
                )
            )
            normalized_index_type = index_type.lower()
            if normalized_index_type in {"hnsw", "squidhnsw"}:
                include_clause = sql.SQL(" INCLUDE (pattern_id)") if normalized_index_type == "squidhnsw" else sql.SQL("")
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {}
                        ON {} USING {} (vector vector_l2_ops){}
                        WITH (m = {m}, ef_construction = {ef});
                        """
                    ).format(
                        sql.Identifier(_safe_index_name(table_name, "vector_idx")),
                        sql.Identifier(table_name),
                        sql.SQL(normalized_index_type),
                        include_clause,
                        m=sql.Literal(int(hnsw_m)),
                        ef=sql.Literal(int(hnsw_ef_construction)),
                    )
                )
            elif normalized_index_type == "ivfflat":
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING ivfflat (vector vector_l2_ops);").format(
                        sql.Identifier(_safe_index_name(table_name, "vector_idx")),
                        sql.Identifier(table_name),
                    )
                )
            else:
                raise ValueError(f"Unsupported index_type: {index_type}")
        conn.commit()
    finally:
        conn.close()


def _create_index_for_partition_timed(table_name: str, index_type: str) -> tuple[str, float]:
    started_at = time.time()
    create_index_for_partition(table_name, index_type=index_type)
    return table_name, time.time() - started_at


def create_indexes_for_materialized_partitions(
    index_type: str = "squidhnsw",
    *,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    maintenance_settings = get_maintenance_settings()
    table_names = list_current_plan_partition_tables(db_connection_factory=db_connection_factory)
    if not table_names:
        table_names = list_materialized_partition_tables(db_connection_factory=db_connection_factory)
    print(
        "KMeans partition index build: PostgreSQL parameters set: maintenance_work_mem = "
        f"{maintenance_settings['maintenance_work_mem_gb']}GB, "
        f"max_parallel_maintenance_workers = {maintenance_settings['max_parallel_maintenance_workers']}",
        flush=True,
    )
    if not table_names:
        print("KMeans partition index build: no materialized partitions found. Skipping.", flush=True)
        return
    worker_count = _recommended_worker_count(len(table_names), max_workers=max_workers)
    ordered_table_names = sorted(table_names)
    if parallel and worker_count > 1 and db_connection_factory is _default_db_connection_factory:
        print(f"KMeans partition index build: creating {index_type} indexes with {worker_count} workers.", flush=True)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_create_index_for_partition_timed, table_name, index_type): table_name
                for table_name in ordered_table_names
            }
            for index, future in enumerate(as_completed(futures), start=1):
                table_name, elapsed = future.result()
                print(f"KMeans partition index build: [{index}/{len(ordered_table_names)}] finished {table_name} in {elapsed:.2f}s", flush=True)
        return
    for index, table_name in enumerate(ordered_table_names, start=1):
        started_at = time.time()
        create_index_for_partition(table_name, index_type=index_type, db_connection_factory=db_connection_factory)
        elapsed = time.time() - started_at
        print(f"KMeans partition index build: [{index}/{len(ordered_table_names)}] finished {table_name} in {elapsed:.2f}s", flush=True)


def drop_indexes_for_materialized_partitions(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            table_names = list_current_plan_partition_tables(db_connection_factory=db_connection_factory)
            if not table_names:
                table_names = list_materialized_partition_tables(db_connection_factory=db_connection_factory)
            for table_name in table_names:
                cur.execute(
                    """
                    SELECT index_class.relname
                    FROM pg_index idx
                    JOIN pg_class table_class ON table_class.oid = idx.indrelid
                    JOIN pg_namespace ns ON ns.oid = table_class.relnamespace
                    JOIN pg_class index_class ON index_class.oid = idx.indexrelid
                    WHERE ns.nspname = current_schema()
                      AND table_class.relname = %s
                      AND NOT idx.indisprimary
                      AND NOT idx.indisunique;
                    """,
                    [str(table_name)],
                )
                for (index_name,) in cur.fetchall():
                    cur.execute(sql.SQL("DROP INDEX IF EXISTS {} CASCADE;").format(sql.Identifier(str(index_name))))
        conn.commit()
    finally:
        conn.close()


def build_and_materialize_kmeans_plan(
    *,
    cluster_count: int = 30,
    private_cluster_count: Optional[int] = None,
    shared_cluster_count: int = 5,
    shared_score_ratio: float = 0.10,
    shared_route_limit: int = 3,
    private_replication_budget_ratio: float = 0.0,
    ef_search: int = 120,
    embedding_dim: Optional[int] = None,
    document_limit: Optional[int] = None,
    query_dataset_path: Optional[str] = None,
    create_indexes: bool = False,
    index_type: str = "squidhnsw",
    show_progress: bool = True,
    enable_split: bool = True,
    private_edge_top_d: int = 32,
    db_connection_factory=_default_db_connection_factory,
) -> KMeansPlan:
    print("[kmeans][planner] loading ACL rows...", flush=True)
    repository = KMeansRepository(db_connection_factory=db_connection_factory)
    acl_rows = repository.fetch_acl_rows(document_limit=document_limit)
    print(f"[kmeans][planner] loaded {len(acl_rows)} ACL rows", flush=True)
    effective_private_cluster_count = int(private_cluster_count) if private_cluster_count is not None else int(cluster_count)
    planner = HybridACLKMeansPlanner()
    print(
        "[kmeans][planner] building cost-guided two-zone split plan: "
        f"private_clusters={int(effective_private_cluster_count)}, "
        f"shared_clusters={int(shared_cluster_count)}, "
        f"shared_score_ratio={float(shared_score_ratio):.4f}, "
        f"shared_route_limit={int(shared_route_limit)}, "
        f"private_replication_budget_ratio={float(private_replication_budget_ratio):.4f}, "
        f"ef_search={int(ef_search)}, "
        f"enable_split={bool(enable_split)}, "
        f"private_edge_top_d={int(private_edge_top_d)}",
        flush=True,
    )
    plan = planner.build_plan(
        acl_rows,
        private_cluster_count=int(effective_private_cluster_count),
        shared_cluster_count=int(shared_cluster_count),
        shared_score_ratio=float(shared_score_ratio),
        shared_route_limit=int(shared_route_limit),
        private_replication_budget_ratio=float(private_replication_budget_ratio),
        ef_search=int(ef_search),
        embedding_dim=embedding_dim,
        query_dataset_path=query_dataset_path,
        show_progress=show_progress,
        enable_split=bool(enable_split),
        private_edge_top_d=int(private_edge_top_d),
    )
    print(
        f"[kmeans][planner] built {len(plan.partitions)} partitions; "
        f"memory_replication_factor={float(plan.metadata.get('memory_replication_factor', 0.0)):.4f}",
        flush=True,
    )
    return materialize_plan(
        plan,
        create_indexes=create_indexes,
        index_type=index_type,
        show_progress=show_progress,
        db_connection_factory=db_connection_factory,
    )
