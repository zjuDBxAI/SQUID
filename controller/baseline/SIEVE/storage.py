from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import time
from typing import Optional

from psycopg2 import sql
from psycopg2.extras import execute_values
from tqdm import tqdm

from controller.dynamic_partition.load_result_to_database import _configure_index_session
from services.config import get_db_connection, get_document_vector_dimension, get_maintenance_settings

from .common import (
    SIEVE_CANDIDATE_TABLE,
    SIEVE_EDGE_TABLE,
    SIEVE_HASSE_EDGE_TABLE,
    SIEVE_PARTITION_TABLE,
    SIEVE_PARTITION_TABLE_PREFIX,
    SIEVE_PLAN_TABLE,
    SIEVE_ROOT_TABLE,
    SieveCandidate,
    SievePartition,
    SievePlan,
    normalize_int_tuple,
)
from .cost_model import SieveCostModel
from .optimizer import SieveOptimizer
from .predicates import build_historical_role_candidate_records, build_role_position_map, count_historical_user_queries, load_historical_user_ids
from .repository import SieveRepository

_DEFAULT_INDEX_MAX_WORKERS = 6
_POSTGRES_IDENTIFIER_LIMIT = 63
_MATERIALIZE_ADVISORY_LOCK_KEY = 2026052401
_MATERIALIZE_BATCH_SIZE = 8192

_CACHED_PLAN_SUMMARY: Optional[dict[str, object]] = None
_CACHED_PARTITIONS: Optional[list[SievePartition]] = None
_CACHED_CANDIDATES: Optional[list[SieveCandidate]] = None
_CACHED_ROLE_POSITIONS: Optional[dict[int, int]] = None
_CACHED_USER_ROLES: Optional[dict[int, tuple[int, ...]]] = None
_CACHED_HASSE_CHILDREN: Optional[dict[str, tuple[str, ...]]] = None


def _default_db_connection_factory():
    return get_db_connection()


def invalidate_cache() -> None:
    global _CACHED_PLAN_SUMMARY, _CACHED_PARTITIONS, _CACHED_CANDIDATES, _CACHED_ROLE_POSITIONS, _CACHED_USER_ROLES, _CACHED_HASSE_CHILDREN
    _CACHED_PLAN_SUMMARY = None
    _CACHED_PARTITIONS = None
    _CACHED_CANDIDATES = None
    _CACHED_ROLE_POSITIONS = None
    _CACHED_USER_ROLES = None
    _CACHED_HASSE_CHILDREN = None


def _safe_index_name(table_name: str, suffix: str) -> str:
    candidate = f"{table_name}_{suffix}"
    if len(candidate) <= _POSTGRES_IDENTIFIER_LIMIT:
        return candidate
    digest = hashlib.blake2b(candidate.encode("utf-8"), digest_size=8).hexdigest()
    compact_suffix = suffix.replace("_", "")
    return f"idx_{digest}_{compact_suffix}"[:_POSTGRES_IDENTIFIER_LIMIT]


def _recommended_worker_count(task_count: int, *, max_workers: Optional[int] = None) -> int:
    if task_count <= 1:
        return 1
    if max_workers is not None:
        return max(1, min(int(max_workers), int(task_count)))
    cpu_count = max(1, int(os.cpu_count() or 1))
    return max(1, min(int(task_count), _DEFAULT_INDEX_MAX_WORKERS, max(1, cpu_count // 4)))


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


def get_partition_table_name(partition_index: int) -> str:
    return f"{SIEVE_PARTITION_TABLE_PREFIX}{int(partition_index)}"


def initialize_schema(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        plan_id BIGSERIAL PRIMARY KEY,
                        dataset_size BIGINT NOT NULL,
                        document_count BIGINT NOT NULL,
                        role_count INTEGER NOT NULL,
                        candidate_count INTEGER NOT NULL,
                        partition_count INTEGER NOT NULL,
                        root_table_name TEXT NOT NULL,
                        role_positions JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                ).format(sql.Identifier(SIEVE_PLAN_TABLE))
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        plan_id BIGINT NOT NULL REFERENCES {}(plan_id) ON DELETE CASCADE,
                        candidate_id INTEGER NOT NULL,
                        role_ids BIGINT[] NOT NULL,
                        role_mask NUMERIC NOT NULL,
                        query_count BIGINT NOT NULL,
                        cardinality BIGINT NOT NULL,
                        scaled_m INTEGER NOT NULL,
                        scaled_size BIGINT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        PRIMARY KEY (plan_id, candidate_id)
                    );
                    """
                ).format(sql.Identifier(SIEVE_CANDIDATE_TABLE), sql.Identifier(SIEVE_PLAN_TABLE))
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        plan_id BIGINT NOT NULL REFERENCES {}(plan_id) ON DELETE CASCADE,
                        partition_id TEXT NOT NULL,
                        candidate_id INTEGER NOT NULL,
                        partition_kind TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        role_ids BIGINT[] NOT NULL,
                        role_mask NUMERIC NOT NULL,
                        cardinality BIGINT NOT NULL,
                        vector_count BIGINT NOT NULL,
                        m INTEGER NOT NULL,
                        ef_construction INTEGER NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        PRIMARY KEY (plan_id, partition_id)
                    );
                    """
                ).format(sql.Identifier(SIEVE_PARTITION_TABLE), sql.Identifier(SIEVE_PLAN_TABLE))
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        plan_id BIGINT NOT NULL REFERENCES {}(plan_id) ON DELETE CASCADE,
                        child_candidate_id INTEGER NOT NULL,
                        parent_candidate_id INTEGER NOT NULL,
                        PRIMARY KEY (plan_id, child_candidate_id, parent_candidate_id)
                    );
                    """
                ).format(sql.Identifier(SIEVE_EDGE_TABLE), sql.Identifier(SIEVE_PLAN_TABLE))
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        plan_id BIGINT NOT NULL REFERENCES {}(plan_id) ON DELETE CASCADE,
                        child_partition_id TEXT NOT NULL,
                        parent_partition_id TEXT NOT NULL,
                        PRIMARY KEY (plan_id, child_partition_id, parent_partition_id)
                    );
                    """
                ).format(sql.Identifier(SIEVE_HASSE_EDGE_TABLE), sql.Identifier(SIEVE_PLAN_TABLE))
            )
            cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (role_ids);").format(
                sql.Identifier(f"idx_{SIEVE_CANDIDATE_TABLE}_role_ids"),
                sql.Identifier(SIEVE_CANDIDATE_TABLE),
            ))
            cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (role_ids);").format(
                sql.Identifier(f"idx_{SIEVE_PARTITION_TABLE}_role_ids"),
                sql.Identifier(SIEVE_PARTITION_TABLE),
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
            cur.execute(sql.SQL("DELETE FROM {};").format(sql.Identifier(SIEVE_PLAN_TABLE)))
        conn.commit()
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
                  AND (tablename LIKE %s OR tablename = %s)
                ORDER BY tablename;
                """,
                [f"{SIEVE_PARTITION_TABLE_PREFIX}%", SIEVE_ROOT_TABLE],
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def list_current_plan_partition_tables(*, include_root: bool = True, db_connection_factory=_default_db_connection_factory) -> list[str]:
    plan_summary = get_current_plan_summary(db_connection_factory=db_connection_factory)
    if plan_summary is None:
        return []
    table_names: list[str] = []
    if include_root:
        table_names.append(str(plan_summary.get("root_table_name") or SIEVE_ROOT_TABLE))
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT table_name
                    FROM {}
                    WHERE plan_id = %s
                    ORDER BY partition_id;
                    """
                ).format(sql.Identifier(SIEVE_PARTITION_TABLE)),
                [int(plan_summary["plan_id"])],
            )
            table_names.extend(str(row[0]) for row in cur.fetchall())
    finally:
        conn.close()
    return table_names


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
                    SELECT plan_id, dataset_size, document_count, role_count, candidate_count,
                           partition_count, root_table_name, role_positions, metadata
                    FROM {}
                    ORDER BY plan_id DESC
                    LIMIT 1;
                    """
                ).format(sql.Identifier(SIEVE_PLAN_TABLE))
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        _CACHED_PLAN_SUMMARY = None
        return None
    _CACHED_PLAN_SUMMARY = {
        "plan_id": int(row[0]),
        "dataset_size": int(row[1]),
        "document_count": int(row[2]),
        "role_count": int(row[3]),
        "candidate_count": int(row[4]),
        "partition_count": int(row[5]),
        "root_table_name": str(row[6]),
        "role_positions": {int(k): int(v) for k, v in dict(row[7] or {}).items()},
        "metadata": dict(row[8] or {}),
    }
    return _CACHED_PLAN_SUMMARY


def load_role_positions(*, refresh: bool = False, db_connection_factory=_default_db_connection_factory) -> dict[int, int]:
    global _CACHED_ROLE_POSITIONS
    if not refresh and _CACHED_ROLE_POSITIONS is not None:
        return _CACHED_ROLE_POSITIONS
    summary = get_current_plan_summary(refresh=refresh, db_connection_factory=db_connection_factory)
    _CACHED_ROLE_POSITIONS = dict(summary.get("role_positions", {}) if summary else {})
    return _CACHED_ROLE_POSITIONS


def load_current_user_roles(*, refresh: bool = False, db_connection_factory=_default_db_connection_factory) -> dict[int, tuple[int, ...]]:
    global _CACHED_USER_ROLES
    if not refresh and _CACHED_USER_ROLES is not None:
        return _CACHED_USER_ROLES
    repository = SieveRepository(db_connection_factory=db_connection_factory)
    _CACHED_USER_ROLES = repository.fetch_user_roles()
    return _CACHED_USER_ROLES


def save_plan(plan: SievePlan, *, role_positions: dict[int, int], db_connection_factory=_default_db_connection_factory) -> int:
    invalidate_cache()
    initialize_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {};").format(sql.Identifier(SIEVE_PLAN_TABLE)))
            metadata = dict(plan.metadata or {})
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (
                        dataset_size, document_count, role_count, candidate_count,
                        partition_count, root_table_name, role_positions, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                    RETURNING plan_id;
                    """
                ).format(sql.Identifier(SIEVE_PLAN_TABLE)),
                [
                    int(metadata.get("dataset_size", 0)),
                    int(metadata.get("document_count", 0)),
                    int(len(role_positions)),
                    int(len(plan.candidates)),
                    int(len(plan.partitions)),
                    SIEVE_ROOT_TABLE,
                    json.dumps({str(k): int(v) for k, v in role_positions.items()}),
                    json.dumps(metadata),
                ],
            )
            plan_id = int(cur.fetchone()[0])
            candidate_rows = [
                (
                    plan_id,
                    int(candidate.candidate_id),
                    list(candidate.role_ids),
                    str(int(candidate.role_mask)),
                    int(candidate.query_count),
                    int(candidate.cardinality),
                    int(candidate.scaled_m),
                    int(candidate.scaled_size),
                    json.dumps(candidate.metadata),
                )
                for candidate in plan.candidates
            ]
            if candidate_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {SIEVE_CANDIDATE_TABLE} (
                        plan_id, candidate_id, role_ids, role_mask, query_count,
                        cardinality, scaled_m, scaled_size, metadata
                    )
                    VALUES %s;
                    """,
                    candidate_rows,
                    template="(%s, %s, %s, %s::numeric, %s, %s, %s, %s, %s::jsonb)",
                    page_size=_MATERIALIZE_BATCH_SIZE,
                )
            partition_rows = [
                (
                    plan_id,
                    str(partition.partition_id),
                    int(partition.candidate_id),
                    str(partition.partition_kind),
                    str(partition.table_name),
                    list(partition.role_ids),
                    str(int(partition.role_mask)),
                    int(partition.cardinality),
                    int(partition.vector_count),
                    int(partition.m),
                    int(partition.ef_construction),
                    json.dumps(partition.metadata),
                )
                for partition in plan.partitions
            ]
            if partition_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {SIEVE_PARTITION_TABLE} (
                        plan_id, partition_id, candidate_id, partition_kind, table_name,
                        role_ids, role_mask, cardinality, vector_count, m, ef_construction, metadata
                    )
                    VALUES %s;
                    """,
                    partition_rows,
                    template="(%s, %s, %s, %s, %s, %s, %s::numeric, %s, %s, %s, %s, %s::jsonb)",
                    page_size=_MATERIALIZE_BATCH_SIZE,
                )
            edge_rows = []
            for left, right in (plan.dag_edges or []):
                edge_rows.append((plan_id, int(left), int(right)))
            if edge_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {SIEVE_EDGE_TABLE} (plan_id, child_candidate_id, parent_candidate_id)
                    VALUES %s;
                    """,
                    edge_rows,
                    page_size=_MATERIALIZE_BATCH_SIZE,
                )
            hasse_rows = []
            for child, parent in (plan.hasse_edges or []):
                hasse_rows.append((plan_id, str(child), str(parent)))
            if hasse_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {SIEVE_HASSE_EDGE_TABLE} (plan_id, child_partition_id, parent_partition_id)
                    VALUES %s;
                    """,
                    hasse_rows,
                    page_size=_MATERIALIZE_BATCH_SIZE,
                )
        conn.commit()
        return plan_id
    finally:
        conn.close()


def load_current_candidates(*, refresh: bool = False, db_connection_factory=_default_db_connection_factory) -> list[SieveCandidate]:
    global _CACHED_CANDIDATES
    if not refresh and _CACHED_CANDIDATES is not None:
        return _CACHED_CANDIDATES
    summary = get_current_plan_summary(refresh=refresh, db_connection_factory=db_connection_factory)
    if summary is None:
        _CACHED_CANDIDATES = []
        return []
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT candidate_id, role_ids, role_mask, query_count, cardinality,
                           scaled_m, scaled_size, metadata
                    FROM {}
                    WHERE plan_id = %s
                    ORDER BY cardinality, candidate_id;
                    """
                ).format(sql.Identifier(SIEVE_CANDIDATE_TABLE)),
                [int(summary["plan_id"])],
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    _CACHED_CANDIDATES = [
        SieveCandidate(
            candidate_id=int(row[0]),
            role_ids=normalize_int_tuple(row[1] or ()),
            role_mask=int(row[2]),
            query_count=int(row[3]),
            cardinality=int(row[4]),
            scaled_m=int(row[5]),
            scaled_size=int(row[6]),
            metadata=dict(row[7] or {}),
        )
        for row in rows
    ]
    return _CACHED_CANDIDATES


def load_current_partitions(*, refresh: bool = False, db_connection_factory=_default_db_connection_factory) -> list[SievePartition]:
    global _CACHED_PARTITIONS
    if not refresh and _CACHED_PARTITIONS is not None:
        return _CACHED_PARTITIONS
    summary = get_current_plan_summary(refresh=refresh, db_connection_factory=db_connection_factory)
    if summary is None:
        _CACHED_PARTITIONS = []
        return []
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT partition_id, candidate_id, partition_kind, table_name,
                           role_ids, role_mask, cardinality, vector_count,
                           m, ef_construction, metadata
                    FROM {}
                    WHERE plan_id = %s
                    ORDER BY cardinality, partition_id;
                    """
                ).format(sql.Identifier(SIEVE_PARTITION_TABLE)),
                [int(summary["plan_id"])],
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    _CACHED_PARTITIONS = [
        SievePartition(
            partition_id=str(row[0]),
            candidate_id=int(row[1]),
            partition_kind=str(row[2]),
            table_name=str(row[3]),
            role_ids=normalize_int_tuple(row[4] or ()),
            role_mask=int(row[5]),
            cardinality=int(row[6]),
            vector_count=int(row[7]),
            m=int(row[8]),
            ef_construction=int(row[9]),
            metadata=dict(row[10] or {}),
        )
        for row in rows
    ]
    return _CACHED_PARTITIONS


def load_current_hasse_children(*, refresh: bool = False, db_connection_factory=_default_db_connection_factory) -> dict[str, tuple[str, ...]]:
    """
    Load the current plan's Hasse diagram as parent -> direct children.

    The edges are stored as (child_partition_id, parent_partition_id), so the
    query path reverses them into a parent -> children adjacency list.
    """
    global _CACHED_HASSE_CHILDREN
    if not refresh and _CACHED_HASSE_CHILDREN is not None:
        return _CACHED_HASSE_CHILDREN
    summary = get_current_plan_summary(refresh=refresh, db_connection_factory=db_connection_factory)
    if summary is None:
        _CACHED_HASSE_CHILDREN = {}
        return {}

    children: dict[str, set[str]] = {}
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT child_partition_id, parent_partition_id
                    FROM {}
                    WHERE plan_id = %s;
                    """
                ).format(sql.Identifier(SIEVE_HASSE_EDGE_TABLE)),
                [int(summary["plan_id"])],
            )
            for child_partition_id, parent_partition_id in cur.fetchall():
                children.setdefault(str(parent_partition_id), set()).add(str(child_partition_id))
    finally:
        conn.close()

    _CACHED_HASSE_CHILDREN = {
        parent: tuple(sorted(child_ids))
        for parent, child_ids in children.items()
    }
    return _CACHED_HASSE_CHILDREN


def _create_vector_table(cur, table_name: str, *, vector_dimension: int) -> None:
    cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name)))
    cur.execute(
        sql.SQL(
            """
            CREATE TABLE {} (
                block_id BIGINT NOT NULL,
                document_id INT NOT NULL REFERENCES Documents(document_id),
                role_ids BIGINT[] NOT NULL,
                block_content BYTEA NOT NULL,
                vector VECTOR({dimension}),
                PRIMARY KEY (block_id, document_id)
            );
            """
        ).format(sql.Identifier(table_name), dimension=sql.SQL(str(int(vector_dimension))))
    )
    cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (role_ids);").format(
        sql.Identifier(_safe_index_name(table_name, "role_ids_idx")),
        sql.Identifier(table_name),
    ))


def materialize_root_table(*, db_connection_factory=_default_db_connection_factory) -> str:
    conn = db_connection_factory()
    vector_dimension = get_document_vector_dimension()
    try:
        with conn.cursor() as cur:
            _create_vector_table(cur, SIEVE_ROOT_TABLE, vector_dimension=vector_dimension)
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (block_id, document_id, role_ids, block_content, vector)
                    SELECT db.block_id, db.document_id,
                           COALESCE(roles.role_ids, ARRAY[]::BIGINT[]) AS role_ids,
                           db.block_content, db.vector
                    FROM documentblocks db
                    LEFT JOIN (
                        SELECT pa.document_id, array_agg(DISTINCT pa.role_id ORDER BY pa.role_id)::BIGINT[] AS role_ids
                        FROM permissionassignment pa
                        GROUP BY pa.document_id
                    ) roles ON roles.document_id = db.document_id;
                    """
                ).format(sql.Identifier(SIEVE_ROOT_TABLE))
            )
        conn.commit()
    finally:
        conn.close()
    return SIEVE_ROOT_TABLE


def materialize_partition(partition: SievePartition, *, db_connection_factory=_default_db_connection_factory) -> str:
    conn = db_connection_factory()
    vector_dimension = get_document_vector_dimension()
    try:
        with conn.cursor() as cur:
            _create_vector_table(cur, partition.table_name, vector_dimension=vector_dimension)
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (block_id, document_id, role_ids, block_content, vector)
                    SELECT db.block_id, db.document_id,
                           COALESCE(roles.role_ids, ARRAY[]::BIGINT[]) AS role_ids,
                           db.block_content, db.vector
                    FROM documentblocks db
                    JOIN (
                        SELECT pa.document_id, array_agg(DISTINCT pa.role_id ORDER BY pa.role_id)::BIGINT[] AS role_ids
                        FROM permissionassignment pa
                        GROUP BY pa.document_id
                    ) roles ON roles.document_id = db.document_id
                    WHERE roles.role_ids && %s::BIGINT[];
                    """
                ).format(sql.Identifier(partition.table_name)),
                [list(partition.role_ids)],
            )
        conn.commit()
    finally:
        conn.close()
    return partition.table_name


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
    plan: SievePlan,
    *,
    role_positions: dict[int, int],
    create_indexes: bool = False,
    index_type: str = "hnsw",
    show_progress: bool = True,
    db_connection_factory=_default_db_connection_factory,
) -> SievePlan:
    lock_conn = _acquire_materialize_lock(db_connection_factory=db_connection_factory)
    try:
        assigned_partitions: list[SievePartition] = []
        for index, partition in enumerate(plan.partitions, start=1):
            assigned_partitions.append(
                SievePartition(
                    partition_id=partition.partition_id,
                    candidate_id=partition.candidate_id,
                    partition_kind=partition.partition_kind,
                    table_name=get_partition_table_name(index),
                    role_ids=partition.role_ids,
                    role_mask=partition.role_mask,
                    cardinality=partition.cardinality,
                    vector_count=partition.vector_count,
                    m=partition.m,
                    ef_construction=partition.ef_construction,
                    metadata=dict(partition.metadata),
                )
            )
        materialized_plan = SievePlan(
            partitions=assigned_partitions,
            candidates=plan.candidates,
            dag_edges=plan.dag_edges,
            hasse_edges=plan.hasse_edges,
            metadata=dict(plan.metadata),
        )
        save_plan(materialized_plan, role_positions=role_positions, db_connection_factory=db_connection_factory)
        valid_tables = {SIEVE_ROOT_TABLE} | {partition.table_name for partition in assigned_partitions}
        drop_stale_materialized_partitions(valid_tables, db_connection_factory=db_connection_factory)

        print("[sieve][materialize] materializing root table...", flush=True)
        materialize_root_table(db_connection_factory=db_connection_factory)
        iterator = assigned_partitions
        if show_progress:
            iterator = tqdm(assigned_partitions, desc="SIEVE materialize", unit="partition")
        for partition in iterator:
            materialize_partition(partition, db_connection_factory=db_connection_factory)

        invalidate_cache()
        if create_indexes:
            create_indexes_for_materialized_partitions(index_type=index_type, db_connection_factory=db_connection_factory)
        return materialized_plan
    finally:
        _release_materialize_lock(lock_conn)


def create_index_for_partition(
    table_name: str,
    *,
    index_type: str = "hnsw",
    hnsw_m: Optional[int] = None,
    hnsw_ef_construction: Optional[int] = None,
    hnsw_threads: Optional[int] = None,
    disable_sync_commit: bool = True,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            _configure_index_session(cur, disable_sync_commit=disable_sync_commit, hnsw_threads=hnsw_threads)
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
                if str(index_name).endswith("role_ids_idx"):
                    continue
                cur.execute(sql.SQL("DROP INDEX IF EXISTS {} CASCADE;").format(sql.Identifier(str(index_name))))

            if index_type.lower() == "hnsw":
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {}
                        ON {} USING hnsw (vector vector_l2_ops)
                        WITH (m = {m}, ef_construction = {ef});
                        """
                    ).format(
                        sql.Identifier(_safe_index_name(table_name, "vector_idx")),
                        sql.Identifier(table_name),
                        m=sql.Literal(int(hnsw_m or 16)),
                        ef=sql.Literal(int(hnsw_ef_construction or 64)),
                    )
                )
            elif index_type.lower() == "ivfflat":
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


def _create_index_for_partition_timed(table_name: str, index_type: str, hnsw_m: int, hnsw_ef_construction: int) -> tuple[str, float]:
    started_at = time.time()
    create_index_for_partition(table_name, index_type=index_type, hnsw_m=hnsw_m, hnsw_ef_construction=hnsw_ef_construction)
    return table_name, time.time() - started_at


def create_indexes_for_materialized_partitions(
    index_type: str = "hnsw",
    *,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    maintenance_settings = get_maintenance_settings()
    table_names = list_current_plan_partition_tables(include_root=True, db_connection_factory=db_connection_factory)
    if not table_names:
        table_names = list_materialized_partition_tables(db_connection_factory=db_connection_factory)
    print(
        "SIEVE index build: PostgreSQL parameters set: maintenance_work_mem = "
        f"{maintenance_settings['maintenance_work_mem_gb']}GB, "
        f"max_parallel_maintenance_workers = {maintenance_settings['max_parallel_maintenance_workers']}",
        flush=True,
    )
    if not table_names:
        print("SIEVE index build: no materialized partitions found. Skipping.", flush=True)
        return

    partitions_by_table = {partition.table_name: partition for partition in load_current_partitions(db_connection_factory=db_connection_factory)}
    plan_summary = get_current_plan_summary(db_connection_factory=db_connection_factory) or {}
    plan_metadata = dict(plan_summary.get("metadata", {}) or {})
    root_hnsw_m = int(plan_metadata.get("m", 16))
    root_hnsw_ef = int(plan_metadata.get("ef_construction", 64))
    worker_count = _recommended_worker_count(len(table_names), max_workers=max_workers)
    ordered_table_names = sorted(table_names)
    if parallel and worker_count > 1 and db_connection_factory is _default_db_connection_factory:
        print(f"SIEVE index build: creating {index_type} indexes with {worker_count} workers.", flush=True)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {}
            for table_name in ordered_table_names:
                partition = partitions_by_table.get(table_name)
                hnsw_m = int(partition.m) if partition else root_hnsw_m
                hnsw_ef = int(partition.ef_construction) if partition else root_hnsw_ef
                futures[executor.submit(_create_index_for_partition_timed, table_name, index_type, hnsw_m, hnsw_ef)] = table_name
            for index, future in enumerate(as_completed(futures), start=1):
                table_name, elapsed = future.result()
                print(f"SIEVE index build: [{index}/{len(ordered_table_names)}] finished {table_name} in {elapsed:.2f}s", flush=True)
        return

    for index, table_name in enumerate(ordered_table_names, start=1):
        started_at = time.time()
        partition = partitions_by_table.get(table_name)
        create_index_for_partition(
            table_name,
            index_type=index_type,
            hnsw_m=int(partition.m) if partition else root_hnsw_m,
            hnsw_ef_construction=int(partition.ef_construction) if partition else root_hnsw_ef,
            db_connection_factory=db_connection_factory,
        )
        print(f"SIEVE index build: [{index}/{len(ordered_table_names)}] finished {table_name} in {time.time() - started_at:.2f}s", flush=True)


def drop_indexes_for_materialized_partitions(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            table_names = list_current_plan_partition_tables(include_root=True, db_connection_factory=db_connection_factory)
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


def build_sieve_plan(
    *,
    query_dataset_path: str,
    historical_filters_percentage: float = 0.25,
    workload_window_size: int = 100000,
    index_budget: float = 2.0,
    bitvector_cutoff: int = 1000,
    m: int = 16,
    ef_construction: int = 40,
    ef_search: int = 10,
    heterogeneous_indexing: bool = True,
    heterogeneous_search: bool = True,
    enable_multipartition_search: bool = False,
    document_limit: Optional[int] = None,
    db_connection_factory=_default_db_connection_factory,
) -> tuple[SievePlan, dict[int, int]]:
    repository = SieveRepository(db_connection_factory=db_connection_factory)
    print("[sieve][planner] loading document role records...", flush=True)
    document_records = repository.fetch_document_roles(document_limit=document_limit)

    historical_user_ids = load_historical_user_ids(
        query_dataset_path,
        historical_filters_percentage=float(historical_filters_percentage),
        workload_window_size=int(workload_window_size),
    )
    historical_user_queries = count_historical_user_queries(historical_user_ids)
    historical_user_roles = repository.fetch_user_roles()

    role_universe = set(repository.fetch_role_universe())
    for record in document_records:
        role_universe.update(record.role_ids)
    for role_ids in historical_user_roles.values():
        role_universe.update(role_ids)
    role_positions = build_role_position_map(role_universe)

    print(f"[sieve][planner] loaded {len(document_records)} documents and {len(role_positions)} roles", flush=True)

    historical_role_records = build_historical_role_candidate_records(
        document_records,
        historical_user_queries,
        historical_user_roles,
        role_positions,
        bitvector_cutoff=int(bitvector_cutoff),
    )
    cardinalities = {normalize_int_tuple(record.role_ids): int(record.cardinality) for record in historical_role_records}
    print(f"[sieve][planner] historical role-set candidate predicates after cutoff: {len(historical_role_records)}", flush=True)

    dataset_size = repository.fetch_document_vector_count()
    total_documents = repository.fetch_total_document_count()
    cost_model = SieveCostModel(
        dataset_size=dataset_size,
        m=int(m),
        bitvector_cutoff=int(bitvector_cutoff),
        ef_search=int(ef_search),
        k=10,
        heterogeneous_indexing=bool(heterogeneous_indexing),
        heterogeneous_search=bool(heterogeneous_search),
    )
    optimizer = SieveOptimizer(cost_model=cost_model, role_positions=role_positions)
    candidates = optimizer.build_candidates(historical_role_records, document_block_counts=cardinalities)
    plan = optimizer.build_plan(candidates, index_budget=float(index_budget))
    plan.metadata.update(
        {
            "dataset_size": int(dataset_size),
            "document_count": int(total_documents),
            "historical_query_count": int(len(historical_user_ids)),
            "candidate_count": int(len(candidates)),
            "partition_count": int(len(plan.partitions)),
            "historical_filters_percentage": float(historical_filters_percentage),
            "workload_window_size": int(workload_window_size),
            "index_budget": float(index_budget),
            "bitvector_cutoff": int(bitvector_cutoff),
            "m": int(m),
            "ef_construction": int(ef_construction),
            "ef_search": int(ef_search),
            "heterogeneous_indexing": bool(heterogeneous_indexing),
            "heterogeneous_search": bool(heterogeneous_search),
            "enable_multipartition_search": bool(enable_multipartition_search),
            "predicate_space": "role_set",
            "predicate_source": "historical_user_roles",
        }
    )
    fixed_partitions = []
    for partition in plan.partitions:
        fixed_partitions.append(
            SievePartition(
                partition_id=partition.partition_id,
                candidate_id=partition.candidate_id,
                partition_kind=partition.partition_kind,
                table_name=partition.table_name,
                role_ids=partition.role_ids,
                role_mask=partition.role_mask,
                cardinality=partition.cardinality,
                vector_count=partition.vector_count,
                m=partition.m,
                ef_construction=int(ef_construction),
                metadata=partition.metadata,
            )
        )
    plan.partitions[:] = fixed_partitions
    print(f"[sieve][planner] selected {len(plan.partitions)} SIEVE partitions", flush=True)
    return plan, role_positions


def build_and_materialize_sieve_plan(
    *,
    query_dataset_path: str,
    historical_filters_percentage: float = 0.25,
    workload_window_size: int = 100000,
    index_budget: float = 2.0,
    bitvector_cutoff: int = 1000,
    m: int = 16,
    ef_construction: int = 40,
    ef_search: int = 10,
    heterogeneous_indexing: bool = True,
    heterogeneous_search: bool = True,
    enable_multipartition_search: bool = False,
    document_limit: Optional[int] = None,
    create_indexes: bool = False,
    index_type: str = "hnsw",
    show_progress: bool = True,
    db_connection_factory=_default_db_connection_factory,
) -> SievePlan:
    plan, role_positions = build_sieve_plan(
        query_dataset_path=query_dataset_path,
        historical_filters_percentage=historical_filters_percentage,
        workload_window_size=workload_window_size,
        index_budget=index_budget,
        bitvector_cutoff=bitvector_cutoff,
        m=m,
        ef_construction=ef_construction,
        ef_search=ef_search,
        heterogeneous_indexing=heterogeneous_indexing,
        heterogeneous_search=heterogeneous_search,
        enable_multipartition_search=enable_multipartition_search,
        document_limit=document_limit,
        db_connection_factory=db_connection_factory,
    )
    return materialize_plan(
        plan,
        role_positions=role_positions,
        create_indexes=create_indexes,
        index_type=index_type,
        show_progress=show_progress,
        db_connection_factory=db_connection_factory,
    )
