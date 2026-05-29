from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import time
from typing import Iterable, Optional

from psycopg2 import sql
from psycopg2.extras import execute_values

from controller.dynamic_partition.load_result_to_database import _configure_index_session
from services.config import get_db_connection, get_document_vector_dimension, get_maintenance_settings

from .common import (
    PARTITION_DOCUMENT_TABLE,
    PARTITION_TABLE,
    PLAN_TABLE,
    WorkloadAwarePartition,
    WorkloadAwarePlan,
    get_partition_table_name,
)
from .planner import WorkloadAwarePlanner
from .repository import WorkloadAwareRepository
from .workload import load_workload_queries

_CACHED_PLAN_SUMMARY: Optional[dict] = None
_CACHED_PARTITIONS: Optional[list[WorkloadAwarePartition]] = None
_EXPECTED_PLAN_COLUMNS = {
    "plan_id",
    "logical_pattern_count",
    "dag_node_count",
    "partition_count",
    "document_count",
    "metadata",
    "created_at",
}
_EXPECTED_PARTITION_COLUMNS = {
    "partition_id",
    "plan_id",
    "table_name",
    "document_count",
    "vector_count",
    "tenant_ids",
    "logical_pattern_ids",
    "metadata",
}
_PARTITION_METADATA_BATCH_SIZE = 1024
_PARTITION_DOCUMENT_BATCH_SIZE = 8192
_DEFAULT_MATERIALIZE_MAX_WORKERS = 8
_DEFAULT_INDEX_MAX_WORKERS = 6


def _default_db_connection_factory():
    return get_db_connection()


def _recommended_worker_count(
    task_count: int,
    *,
    max_workers: Optional[int] = None,
    hard_cap: int,
) -> int:
    if task_count <= 1:
        return 1
    if max_workers is not None:
        return max(1, min(int(max_workers), int(task_count)))
    cpu_count = max(1, int(os.cpu_count() or 1))
    return max(1, min(int(task_count), int(hard_cap), max(1, cpu_count // 4)))


def invalidate_cached_plan_metadata() -> None:
    global _CACHED_PLAN_SUMMARY, _CACHED_PARTITIONS
    _CACHED_PLAN_SUMMARY = None
    _CACHED_PARTITIONS = None


def _plan_progress(message: str) -> None:
    print(str(message), flush=True)


def _table_columns(cur, table_name: str) -> set[str]:
    cur.execute(
        """
        SELECT column_name
        FROM information_schema.columns
        WHERE table_name = %s;
        """,
        [table_name],
    )
    return {str(row[0]) for row in cur.fetchall()}


def initialize_plan_schema(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute('SELECT to_regclass(%s);', [PLAN_TABLE])
            plan_exists = cur.fetchone()[0] is not None
            cur.execute('SELECT to_regclass(%s);', [PARTITION_TABLE])
            partition_exists = cur.fetchone()[0] is not None
            if plan_exists and _table_columns(cur, PLAN_TABLE) != _EXPECTED_PLAN_COLUMNS:
                cur.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE;').format(sql.Identifier(PARTITION_DOCUMENT_TABLE)))
                cur.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE;').format(sql.Identifier(PARTITION_TABLE)))
                cur.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE;').format(sql.Identifier(PLAN_TABLE)))
                plan_exists = False
                partition_exists = False
            if partition_exists and _table_columns(cur, PARTITION_TABLE) != _EXPECTED_PARTITION_COLUMNS:
                cur.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE;').format(sql.Identifier(PARTITION_DOCUMENT_TABLE)))
                cur.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE;').format(sql.Identifier(PARTITION_TABLE)))
                cur.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE;').format(sql.Identifier(PLAN_TABLE)))

            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {PLAN_TABLE} (
                    plan_id BIGSERIAL PRIMARY KEY,
                    logical_pattern_count INTEGER NOT NULL,
                    dag_node_count INTEGER NOT NULL,
                    partition_count INTEGER NOT NULL,
                    document_count BIGINT NOT NULL,
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
                    table_name TEXT NOT NULL,
                    document_count BIGINT NOT NULL,
                    vector_count BIGINT NOT NULL,
                    tenant_ids BIGINT[] NOT NULL,
                    logical_pattern_ids INTEGER[] NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb
                );
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {PARTITION_DOCUMENT_TABLE} (
                    partition_id TEXT NOT NULL REFERENCES {PARTITION_TABLE}(partition_id) ON DELETE CASCADE,
                    document_id BIGINT NOT NULL,
                    PRIMARY KEY (partition_id, document_id)
                );
                """
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (tenant_ids);"
                ).format(
                    sql.Identifier(f"idx_{PARTITION_TABLE}_tenant"),
                    sql.Identifier(PARTITION_TABLE),
                )
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} (document_id);"
                ).format(
                    sql.Identifier(f"idx_{PARTITION_DOCUMENT_TABLE}_document"),
                    sql.Identifier(PARTITION_DOCUMENT_TABLE),
                )
            )
        conn.commit()
    finally:
        conn.close()


def clear_current_plan(*, db_connection_factory=_default_db_connection_factory) -> None:
    invalidate_cached_plan_metadata()
    initialize_plan_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL('DELETE FROM {};').format(sql.Identifier(PLAN_TABLE)))
        conn.commit()
    finally:
        conn.close()


def save_plan_result(
    plan: WorkloadAwarePlan,
    *,
    db_connection_factory=_default_db_connection_factory,
) -> int:
    invalidate_cached_plan_metadata()
    initialize_plan_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL('DELETE FROM {};').format(sql.Identifier(PLAN_TABLE)))
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (
                        logical_pattern_count,
                        dag_node_count,
                        partition_count,
                        document_count,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    RETURNING plan_id;
                    """
                ).format(sql.Identifier(PLAN_TABLE)),
                [
                    int(plan.metadata.get("logical_pattern_count", len(plan.logical_patterns))),
                    int(plan.metadata.get("dag_node_count", len(plan.dag_nodes))),
                    int(plan.metadata.get("partition_count", len(plan.partitions))),
                    int(plan.metadata.get("document_count", 0)),
                    json.dumps(plan.metadata),
                ],
            )
            plan_id = int(cur.fetchone()[0])
            partition_rows = [
                (
                    partition.partition_id,
                    plan_id,
                    partition.table_name,
                    int(partition.document_count),
                    int(partition.vector_count),
                    list(partition.tenant_ids),
                    list(partition.logical_pattern_ids),
                    json.dumps(partition.metadata),
                )
                for partition in plan.partitions
            ]
            if partition_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {PARTITION_TABLE} (
                        partition_id,
                        plan_id,
                        table_name,
                        document_count,
                        vector_count,
                        tenant_ids,
                        logical_pattern_ids,
                        metadata
                    )
                    VALUES %s;
                    """,
                    partition_rows,
                    template="(%s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                    page_size=_PARTITION_METADATA_BATCH_SIZE,
                )

            partition_document_rows = [
                (partition.partition_id, int(document_id))
                for partition in plan.partitions
                for document_id in partition.document_ids
            ]
            if partition_document_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {PARTITION_DOCUMENT_TABLE} (
                        partition_id,
                        document_id
                    )
                    VALUES %s;
                    """,
                    partition_document_rows,
                    page_size=_PARTITION_DOCUMENT_BATCH_SIZE,
                )
        conn.commit()
        return plan_id
    finally:
        conn.close()


def get_current_plan_summary(
    *,
    refresh: bool = False,
    db_connection_factory=_default_db_connection_factory,
) -> Optional[dict]:
    global _CACHED_PLAN_SUMMARY
    if not refresh and _CACHED_PLAN_SUMMARY is not None:
        return _CACHED_PLAN_SUMMARY

    initialize_plan_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        plan_id,
                        logical_pattern_count,
                        dag_node_count,
                        partition_count,
                        document_count,
                        metadata
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
        "logical_pattern_count": int(row[1]),
        "dag_node_count": int(row[2]),
        "partition_count": int(row[3]),
        "document_count": int(row[4]),
        "metadata": dict(row[5] or {}),
    }
    return _CACHED_PLAN_SUMMARY


def load_current_partitions(
    *,
    refresh: bool = False,
    db_connection_factory=_default_db_connection_factory,
) -> list[WorkloadAwarePartition]:
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
                    SELECT
                        partition_id,
                        table_name,
                        document_count,
                        vector_count,
                        tenant_ids,
                        logical_pattern_ids,
                        metadata
                    FROM {}
                    WHERE plan_id = %s
                    ORDER BY partition_id;
                    """
                ).format(sql.Identifier(PARTITION_TABLE)),
                [int(plan_summary["plan_id"])],
            )
            partition_rows = cur.fetchall()
            if partition_rows:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT partition_id, document_id
                        FROM {}
                        WHERE partition_id = ANY(%s)
                        ORDER BY partition_id, document_id;
                        """
                    ).format(sql.Identifier(PARTITION_DOCUMENT_TABLE)),
                    [[row[0] for row in partition_rows]],
                )
                document_rows = cur.fetchall()
            else:
                document_rows = []
    finally:
        conn.close()

    documents_by_partition: dict[str, list[int]] = defaultdict(list)
    for partition_id, document_id in document_rows:
        documents_by_partition[str(partition_id)].append(int(document_id))

    partitions: list[WorkloadAwarePartition] = []
    for partition_id, table_name, _, vector_count, tenant_ids, logical_pattern_ids, metadata in partition_rows:
        partitions.append(
            WorkloadAwarePartition(
                partition_id=str(partition_id),
                table_name=str(table_name),
                document_ids=tuple(documents_by_partition.get(str(partition_id), ())),
                tenant_ids=tuple(int(tenant_id) for tenant_id in (tenant_ids or ())),
                vector_count=int(vector_count),
                logical_pattern_ids=tuple(int(pattern_id) for pattern_id in (logical_pattern_ids or ())),
                metadata=dict(metadata or {}),
            )
        )
    _CACHED_PARTITIONS = partitions
    return partitions


def list_materialized_partition_tables(*, db_connection_factory=_default_db_connection_factory) -> list[str]:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE tablename LIKE %s
                ORDER BY tablename;
                """,
                ["workload_documentblocks_partition_%"],
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def drop_materialized_partitions(
    *,
    valid_partition_ids: Optional[Iterable[str]] = None,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    valid_table_names = {get_partition_table_name(partition_id) for partition_id in valid_partition_ids} if valid_partition_ids is not None else set()
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE tablename LIKE %s;
                """,
                ["workload_documentblocks_partition_%"],
            )
            existing_table_names = [str(row[0]) for row in cur.fetchall()]
        for table_name in existing_table_names:
            if valid_table_names and table_name in valid_table_names:
                continue
            with conn.cursor() as cur:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name)))
            conn.commit()
    finally:
        conn.close()


def materialize_partition(
    partition: WorkloadAwarePartition,
    *,
    db_connection_factory=_default_db_connection_factory,
) -> str:
    conn = db_connection_factory()
    vector_dimension = get_document_vector_dimension()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        block_id BIGINT NOT NULL,
                        document_id INT NOT NULL REFERENCES Documents(document_id),
                        block_content BYTEA NOT NULL,
                        vector VECTOR({dimension}),
                        PRIMARY KEY (block_id, document_id)
                    );
                    """
                ).format(
                    sql.Identifier(partition.table_name),
                    dimension=sql.SQL(str(vector_dimension)),
                )
            )
            cur.execute(sql.SQL("TRUNCATE TABLE {};").format(sql.Identifier(partition.table_name)))
            if partition.document_count > 0:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (block_id, document_id, block_content, vector)
                        SELECT db.block_id, db.document_id, db.block_content, db.vector
                        FROM documentblocks db
                        JOIN {} pd
                          ON pd.document_id = db.document_id
                        WHERE pd.partition_id = %s;
                        """
                    ).format(
                        sql.Identifier(partition.table_name),
                        sql.Identifier(PARTITION_DOCUMENT_TABLE),
                    ),
                    [partition.partition_id],
                )
        conn.commit()
        return partition.table_name
    finally:
        conn.close()


def _partition_materialization_matches(left: WorkloadAwarePartition, right: WorkloadAwarePartition) -> bool:
    return (
        str(left.table_name) == str(right.table_name)
        and tuple(int(document_id) for document_id in left.document_ids)
        == tuple(int(document_id) for document_id in right.document_ids)
        and int(left.vector_count) == int(right.vector_count)
        and tuple(int(tenant_id) for tenant_id in left.tenant_ids)
        == tuple(int(tenant_id) for tenant_id in right.tenant_ids)
    )


def _materialize_partition_timed(partition: WorkloadAwarePartition) -> tuple[str, float]:
    started_at = time.time()
    table_name = materialize_partition(partition)
    return table_name, time.time() - started_at


def materialize_planner_result(
    plan: WorkloadAwarePlan,
    *,
    create_indexes: bool = False,
    index_type: str = "hnsw",
    db_connection_factory=_default_db_connection_factory,
) -> WorkloadAwarePlan:
    existing_partitions = {
        partition.partition_id: partition
        for partition in load_current_partitions(refresh=True, db_connection_factory=db_connection_factory)
    }
    existing_table_names = set(list_materialized_partition_tables(db_connection_factory=db_connection_factory))

    reusable_partition_ids: list[str] = []
    partitions_to_materialize: list[WorkloadAwarePartition] = []
    for partition in plan.partitions:
        existing_partition = existing_partitions.get(partition.partition_id)
        if (
            existing_partition is not None
            and partition.table_name in existing_table_names
            and _partition_materialization_matches(partition, existing_partition)
        ):
            reusable_partition_ids.append(str(partition.partition_id))
            continue
        partitions_to_materialize.append(partition)

    _plan_progress(f"[plan][materialize] saving metadata for {len(plan.partitions)} partitions...")
    save_plan_result(plan, db_connection_factory=db_connection_factory)
    drop_materialized_partitions(
        valid_partition_ids=[partition.partition_id for partition in plan.partitions],
        db_connection_factory=db_connection_factory,
    )

    if reusable_partition_ids:
        _plan_progress(
            f"[plan][materialize] reusing {len(reusable_partition_ids)} unchanged partition tables"
        )

    partition_count = len(partitions_to_materialize)
    if partition_count == 0:
        _plan_progress("[plan][materialize] no partition tables needed rewriting")
    else:
        _plan_progress(
            f"[plan][materialize] materializing {partition_count} partition tables "
            f"(reused {len(reusable_partition_ids)})..."
        )
        report_every = max(1, partition_count // 20)
        worker_count = _recommended_worker_count(
            partition_count,
            hard_cap=_DEFAULT_MATERIALIZE_MAX_WORKERS,
        )
        ordered_partitions = sorted(
            partitions_to_materialize,
            key=lambda partition: (-int(partition.vector_count), -int(partition.document_count), str(partition.partition_id)),
        )
        if partition_count > 1 and db_connection_factory is _default_db_connection_factory and worker_count > 1:
            _plan_progress(f"[plan][materialize] parallel materialization with {worker_count} workers...")
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                future_to_partition = {
                    executor.submit(_materialize_partition_timed, partition): partition
                    for partition in ordered_partitions
                }
                completed = 0
                for future in as_completed(future_to_partition):
                    completed += 1
                    table_name, elapsed = future.result()
                    if completed == 1 or completed % report_every == 0 or completed == partition_count:
                        _plan_progress(
                            f"[plan][materialize] [{completed}/{partition_count}] materialized {table_name} in {elapsed:.2f}s"
                        )
        else:
            for index, partition in enumerate(ordered_partitions, start=1):
                started_at = time.time()
                materialize_partition(partition, db_connection_factory=db_connection_factory)
                elapsed = time.time() - started_at
                if index == 1 or index % report_every == 0 or index == partition_count:
                    _plan_progress(
                        f"[plan][materialize] [{index}/{partition_count}] materialized {partition.table_name} in {elapsed:.2f}s"
                    )
    if create_indexes:
        _plan_progress(f"[plan][materialize] creating {index_type} indexes...")
        create_indexes_for_materialized_partitions(index_type=index_type, db_connection_factory=db_connection_factory)
    _plan_progress("[plan] completed")
    return plan


def build_and_materialize_workload_aware_plan(
    *,
    min_pattern_support: int = 16,
    min_pattern_query_mass: float = 0.0,
    safe_density_threshold: float = 0.35,
    supplemental_edge_penalty: float = 0.25,
    supplemental_edge_gain_threshold: float = 0.0,
    target_partition_count: Optional[int] = None,
    max_partition_vector_count: Optional[int] = None,
    query_dataset_path: Optional[str] = None,
    workload_limit: Optional[int] = None,
    document_limit: Optional[int] = None,
    create_indexes: bool = False,
    index_type: str = "hnsw",
    db_connection_factory=_default_db_connection_factory,
) -> WorkloadAwarePlan:
    started_at = time.time()
    repository = WorkloadAwareRepository(db_connection_factory=db_connection_factory)

    _plan_progress("[plan][1/5] loading document access records...")
    records = repository.fetch_document_access_records(limit=document_limit)
    if not records:
        raise RuntimeError("No document access records found; cannot build ACL Prefix-DAG partitions.")
    _plan_progress(f"[plan][1/5] loaded {len(records)} document access records")

    _plan_progress("[plan][2/5] loading document block counts...")
    block_counts = repository.fetch_document_block_counts(record.document_id for record in records)
    _plan_progress(f"[plan][2/5] loaded block counts for {len(block_counts)} documents")

    _plan_progress("[plan][3/5] loading workload queries...")
    queries, tenant_query_weights = load_workload_queries(
        query_dataset_path=query_dataset_path,
        limit=workload_limit,
    )
    if not tenant_query_weights:
        tenant_query_weights = {
            int(tenant_id): 1.0
            for tenant_id in sorted({tenant_id for record in records for tenant_id in record.tenant_ids})
        }
    _plan_progress(f"[plan][3/5] loaded {len(queries)} workload queries and {len(tenant_query_weights)} tenant weights")

    _plan_progress("[plan][4/5] building ACL Prefix-DAG logical plan...")
    planner = WorkloadAwarePlanner()
    plan = planner.build_plan(
        records,
        document_block_counts=block_counts,
        queries=queries,
        tenant_query_weights=tenant_query_weights,
        min_pattern_support=min_pattern_support,
        min_pattern_query_mass=min_pattern_query_mass,
        safe_density_threshold=safe_density_threshold,
        supplemental_edge_penalty=supplemental_edge_penalty,
        supplemental_edge_gain_threshold=supplemental_edge_gain_threshold,
        target_partition_count=target_partition_count,
        max_partition_vector_count=max_partition_vector_count,
        progress_fn=_plan_progress,
    )
    _plan_progress(
        f"[plan][4/5] logical plan ready: acl_patterns={len(plan.logical_patterns)}, partitions={len(plan.partitions)}"
    )

    _plan_progress("[plan][5/5] materializing plan to database...")
    result = materialize_planner_result(
        plan,
        create_indexes=create_indexes,
        index_type=index_type,
        db_connection_factory=db_connection_factory,
    )
    elapsed = time.time() - started_at
    _plan_progress(f"[plan] total elapsed {elapsed:.2f}s")
    return result


def create_index_for_partition(
    table_name: str,
    index_type: str = "hnsw",
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
            _configure_index_session(cur, disable_sync_commit=disable_sync_commit, hnsw_threads=hnsw_threads)
            if index_type.lower() == "hnsw":
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {}
                        ON {} USING hnsw (vector vector_l2_ops)
                        WITH (m = {m}, ef_construction = {ef});
                        """
                    ).format(
                        sql.Identifier(f"{table_name}_vector_idx"),
                        sql.Identifier(table_name),
                        m=sql.Literal(int(hnsw_m)),
                        ef=sql.Literal(int(hnsw_ef_construction)),
                    )
                )
            elif index_type.lower() == "ivfflat":
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {}
                        ON {} USING ivfflat (vector vector_l2_ops);
                        """
                    ).format(
                        sql.Identifier(f"{table_name}_vector_idx"),
                        sql.Identifier(table_name),
                    )
                )
            else:
                raise ValueError(f"Unsupported index_type: {index_type}")
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
    index_type: str = "hnsw",
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
        "PostgreSQL parameters set: maintenance_work_mem = "
        f"{maintenance_settings['maintenance_work_mem_gb']}GB, "
        f"max_parallel_maintenance_workers = {maintenance_settings['max_parallel_maintenance_workers']}",
        flush=True,
    )
    if not table_names:
        print("ACL Prefix-DAG index build: no materialized partitions found. Skipping.", flush=True)
        return

    print(
        f"ACL Prefix-DAG index build: creating {index_type} indexes for {len(table_names)} partitions...",
        flush=True,
    )
    worker_count = _recommended_worker_count(
        len(table_names),
        max_workers=max_workers,
        hard_cap=_DEFAULT_INDEX_MAX_WORKERS,
    )
    ordered_table_names = sorted(table_names)
    if parallel and worker_count > 1 and db_connection_factory is _default_db_connection_factory:
        print(f"ACL Prefix-DAG index build: parallel mode with {worker_count} workers.", flush=True)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
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
                for table_name in ordered_table_names
            }
            completed = 0
            for future in as_completed(future_to_table):
                completed += 1
                table_name, elapsed = future.result()
                print(
                    f"ACL Prefix-DAG index build: [{completed}/{len(ordered_table_names)}] finished {table_name} in {elapsed:.2f}s",
                    flush=True,
                )
        return

    print("ACL Prefix-DAG index build: sequential mode.", flush=True)
    for index, table_name in enumerate(ordered_table_names, start=1):
        started_at = time.time()
        create_index_for_partition(
            table_name,
            index_type=index_type,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_threads=hnsw_threads,
            disable_sync_commit=disable_sync_commit,
            db_connection_factory=db_connection_factory,
        )
        elapsed = time.time() - started_at
        print(
            f"ACL Prefix-DAG index build: [{index}/{len(ordered_table_names)}] finished {table_name} in {elapsed:.2f}s",
            flush=True,
        )


def drop_indexes_for_materialized_partitions(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            for table_name in list_materialized_partition_tables(db_connection_factory=db_connection_factory):
                cur.execute(
                    sql.SQL("DROP INDEX IF EXISTS {} CASCADE;").format(sql.Identifier(f"{table_name}_vector_idx"))
                )
        conn.commit()
    finally:
        conn.close()
