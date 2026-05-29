from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
import time
from typing import Iterable, Optional

from psycopg2 import sql
from psycopg2.extras import execute_values

from controller.dynamic_partition.load_result_to_database import _configure_index_session
from services.config import get_db_connection, get_document_vector_dimension, get_maintenance_settings

from .common import (
    DAG_NODE_TABLE,
    LOGICAL_PATTERN_TABLE,
    PARTITION_DOCUMENT_TABLE,
    PARTITION_TABLE,
    PLAN_TABLE,
    PersistedDagNode,
    PersistedLogicalPattern,
    WORKLOAD_AWARE_ACCESS_OVERLAY_TABLE_PREFIX,
    WORKLOAD_AWARE_OVERLAY_TABLE_PREFIX,
    WorkloadAwarePartition,
    WorkloadAwarePlan,
    get_partition_table_name,
)
from .planner import WorkloadAwarePlanner
from .repository import WorkloadAwareRepository
from .workload import load_workload_queries

_CACHED_PLAN_SUMMARY: Optional[dict] = None
_CACHED_PARTITIONS: Optional[list[WorkloadAwarePartition]] = None
_CACHED_LOGICAL_PATTERNS: Optional[list[PersistedLogicalPattern]] = None
_CACHED_DAG_NODES: Optional[list[PersistedDagNode]] = None
_CACHED_TENANT_OVERLAYS: Optional[list[dict[str, object]]] = None
_CACHED_ACCESS_OVERLAYS: Optional[list[dict[str, object]]] = None
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
_EXPECTED_LOGICAL_PATTERN_COLUMNS = {
    "pattern_id",
    "plan_id",
    "partition_id",
    "tenant_ids",
    "ordered_tenant_ids",
    "entry_tenant_ids",
    "document_count",
    "vector_count",
    "metadata",
}
_EXPECTED_DAG_NODE_COLUMNS = {
    "node_id",
    "plan_id",
    "prefix_tenants",
    "children",
    "terminal_pattern_ids",
    "supplemental_pattern_ids",
    "document_count",
    "terminal_document_count",
    "metadata",
}
_PARTITION_METADATA_BATCH_SIZE = 1024
_PARTITION_DOCUMENT_BATCH_SIZE = 8192
_LOGICAL_PATTERN_BATCH_SIZE = 2048
_DAG_NODE_BATCH_SIZE = 2048
_DEFAULT_MATERIALIZE_MAX_WORKERS = 8
_DEFAULT_INDEX_MAX_WORKERS = 6
_MATERIALIZE_ADVISORY_LOCK_KEY = 2026042501
_ACCELERATOR_TABLE_PREFIX = "workload_documentblocks_partition_accel_"
_OVERLAY_TABLE_PREFIX = WORKLOAD_AWARE_OVERLAY_TABLE_PREFIX
_ACCESS_OVERLAY_TABLE_PREFIX = WORKLOAD_AWARE_ACCESS_OVERLAY_TABLE_PREFIX
_POSTGRES_IDENTIFIER_LIMIT = 63


def _default_db_connection_factory():
    return get_db_connection()


def _safe_index_name(table_name: str, suffix: str) -> str:
    candidate = f"{table_name}_{suffix}"
    if len(candidate) <= _POSTGRES_IDENTIFIER_LIMIT:
        return candidate
    digest = hashlib.blake2b(candidate.encode("utf-8"), digest_size=8).hexdigest()
    compact_suffix = suffix.replace("_", "")
    return f"idx_{digest}_{compact_suffix}"[:_POSTGRES_IDENTIFIER_LIMIT]


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
    global _CACHED_PLAN_SUMMARY, _CACHED_PARTITIONS, _CACHED_LOGICAL_PATTERNS, _CACHED_DAG_NODES, _CACHED_TENANT_OVERLAYS, _CACHED_ACCESS_OVERLAYS
    _CACHED_PLAN_SUMMARY = None
    _CACHED_PARTITIONS = None
    _CACHED_LOGICAL_PATTERNS = None
    _CACHED_DAG_NODES = None
    _CACHED_TENANT_OVERLAYS = None
    _CACHED_ACCESS_OVERLAYS = None


def _drop_plan_schema(cur) -> None:
    for table_name in (
        DAG_NODE_TABLE,
        LOGICAL_PATTERN_TABLE,
        PARTITION_DOCUMENT_TABLE,
        PARTITION_TABLE,
        PLAN_TABLE,
    ):
        cur.execute(sql.SQL('DROP TABLE IF EXISTS {} CASCADE;').format(sql.Identifier(table_name)))


def _plan_progress(message: str) -> None:
    print(str(message), flush=True)


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
            table_expectations = {
                PLAN_TABLE: _EXPECTED_PLAN_COLUMNS,
                PARTITION_TABLE: _EXPECTED_PARTITION_COLUMNS,
                LOGICAL_PATTERN_TABLE: _EXPECTED_LOGICAL_PATTERN_COLUMNS,
                DAG_NODE_TABLE: _EXPECTED_DAG_NODE_COLUMNS,
            }
            schema_valid = True
            for table_name, expected_columns in table_expectations.items():
                cur.execute('SELECT to_regclass(%s);', [table_name])
                exists = cur.fetchone()[0] is not None
                if exists and _table_columns(cur, table_name) != expected_columns:
                    schema_valid = False
                    break
            if not schema_valid:
                _drop_plan_schema(cur)

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
                f"""
                CREATE TABLE IF NOT EXISTS {LOGICAL_PATTERN_TABLE} (
                    pattern_id INTEGER NOT NULL,
                    plan_id BIGINT NOT NULL REFERENCES {PLAN_TABLE}(plan_id) ON DELETE CASCADE,
                    partition_id TEXT NOT NULL REFERENCES {PARTITION_TABLE}(partition_id) ON DELETE CASCADE,
                    tenant_ids BIGINT[] NOT NULL,
                    ordered_tenant_ids BIGINT[] NOT NULL,
                    entry_tenant_ids BIGINT[] NOT NULL,
                    document_count BIGINT NOT NULL,
                    vector_count BIGINT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    PRIMARY KEY (plan_id, pattern_id)
                );
                """
            )
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {DAG_NODE_TABLE} (
                    node_id BIGINT NOT NULL,
                    plan_id BIGINT NOT NULL REFERENCES {PLAN_TABLE}(plan_id) ON DELETE CASCADE,
                    prefix_tenants BIGINT[] NOT NULL,
                    children JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    terminal_pattern_ids INTEGER[] NOT NULL,
                    supplemental_pattern_ids INTEGER[] NOT NULL,
                    document_count BIGINT NOT NULL,
                    terminal_document_count BIGINT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    PRIMARY KEY (plan_id, node_id)
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
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (entry_tenant_ids);"
                ).format(
                    sql.Identifier(f"idx_{LOGICAL_PATTERN_TABLE}_entry_tenant"),
                    sql.Identifier(LOGICAL_PATTERN_TABLE),
                )
            )
            cur.execute(
                sql.SQL(
                    "CREATE INDEX IF NOT EXISTS {} ON {} (partition_id);"
                ).format(
                    sql.Identifier(f"idx_{LOGICAL_PATTERN_TABLE}_partition"),
                    sql.Identifier(LOGICAL_PATTERN_TABLE),
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

            pattern_partition_ids = {
                int(pattern_id): str(partition.partition_id)
                for partition in plan.partitions
                for pattern_id in partition.logical_pattern_ids
            }
            logical_pattern_rows = [
                (
                    int(pattern.pattern_id),
                    plan_id,
                    pattern_partition_ids[int(pattern.pattern_id)],
                    list(pattern.tenant_ids),
                    list(pattern.ordered_tenant_ids),
                    list(pattern.entry_tenant_ids),
                    int(pattern.document_count),
                    int(pattern.vector_count),
                    json.dumps(pattern.metadata),
                )
                for pattern in plan.logical_patterns
                if int(pattern.pattern_id) in pattern_partition_ids
            ]
            if logical_pattern_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {LOGICAL_PATTERN_TABLE} (
                        pattern_id,
                        plan_id,
                        partition_id,
                        tenant_ids,
                        ordered_tenant_ids,
                        entry_tenant_ids,
                        document_count,
                        vector_count,
                        metadata
                    )
                    VALUES %s;
                    """,
                    logical_pattern_rows,
                    template="(%s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)",
                    page_size=_LOGICAL_PATTERN_BATCH_SIZE,
                )

            dag_node_rows = [
                (
                    int(node.node_id),
                    plan_id,
                    list(node.prefix_tenants),
                    json.dumps({str(int(tenant_id)): int(child_id) for tenant_id, child_id in sorted(node.children.items())}),
                    list(sorted(int(pattern_id) for pattern_id in node.terminal_pattern_ids)),
                    list(sorted(int(pattern_id) for pattern_id in node.supplemental_pattern_ids)),
                    int(node.document_count),
                    int(node.terminal_document_count),
                    json.dumps({"prefix_length": len(node.prefix_tenants)}),
                )
                for node in plan.dag_nodes
            ]
            if dag_node_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {DAG_NODE_TABLE} (
                        node_id,
                        plan_id,
                        prefix_tenants,
                        children,
                        terminal_pattern_ids,
                        supplemental_pattern_ids,
                        document_count,
                        terminal_document_count,
                        metadata
                    )
                    VALUES %s;
                    """,
                    dag_node_rows,
                    template="(%s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s::jsonb)",
                    page_size=_DAG_NODE_BATCH_SIZE,
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


def load_current_logical_patterns(
    *,
    refresh: bool = False,
    db_connection_factory=_default_db_connection_factory,
) -> list[PersistedLogicalPattern]:
    global _CACHED_LOGICAL_PATTERNS
    if not refresh and _CACHED_LOGICAL_PATTERNS is not None:
        return _CACHED_LOGICAL_PATTERNS

    plan_summary = get_current_plan_summary(refresh=refresh, db_connection_factory=db_connection_factory)
    if plan_summary is None:
        _CACHED_LOGICAL_PATTERNS = []
        return []

    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        pattern_id,
                        partition_id,
                        tenant_ids,
                        ordered_tenant_ids,
                        entry_tenant_ids,
                        document_count,
                        vector_count,
                        metadata
                    FROM {}
                    WHERE plan_id = %s
                    ORDER BY pattern_id;
                    """
                ).format(sql.Identifier(LOGICAL_PATTERN_TABLE)),
                [int(plan_summary["plan_id"])],
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    logical_patterns = [
        PersistedLogicalPattern(
            pattern_id=int(pattern_id),
            partition_id=str(partition_id),
            tenant_ids=tuple(int(value) for value in (tenant_ids or ())),
            ordered_tenant_ids=tuple(int(value) for value in (ordered_tenant_ids or ())),
            entry_tenant_ids=tuple(int(value) for value in (entry_tenant_ids or ())),
            document_count=int(document_count),
            vector_count=int(vector_count),
            metadata=dict(metadata or {}),
        )
        for pattern_id, partition_id, tenant_ids, ordered_tenant_ids, entry_tenant_ids, document_count, vector_count, metadata in rows
    ]
    _CACHED_LOGICAL_PATTERNS = logical_patterns
    return logical_patterns


def load_current_dag_nodes(
    *,
    refresh: bool = False,
    db_connection_factory=_default_db_connection_factory,
) -> list[PersistedDagNode]:
    global _CACHED_DAG_NODES
    if not refresh and _CACHED_DAG_NODES is not None:
        return _CACHED_DAG_NODES

    plan_summary = get_current_plan_summary(refresh=refresh, db_connection_factory=db_connection_factory)
    if plan_summary is None:
        _CACHED_DAG_NODES = []
        return []

    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT
                        node_id,
                        prefix_tenants,
                        children,
                        terminal_pattern_ids,
                        supplemental_pattern_ids,
                        document_count,
                        terminal_document_count,
                        metadata
                    FROM {}
                    WHERE plan_id = %s
                    ORDER BY node_id;
                    """
                ).format(sql.Identifier(DAG_NODE_TABLE)),
                [int(plan_summary["plan_id"])],
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    dag_nodes = []
    for node_id, prefix_tenants, children, terminal_pattern_ids, supplemental_pattern_ids, document_count, terminal_document_count, metadata in rows:
        normalized_children = {
            int(tenant_id): int(child_node_id)
            for tenant_id, child_node_id in dict(children or {}).items()
        }
        dag_nodes.append(
            PersistedDagNode(
                node_id=int(node_id),
                prefix_tenants=tuple(int(value) for value in (prefix_tenants or ())),
                children=normalized_children,
                terminal_pattern_ids=tuple(int(value) for value in (terminal_pattern_ids or ())),
                supplemental_pattern_ids=tuple(int(value) for value in (supplemental_pattern_ids or ())),
                document_count=int(document_count),
                terminal_document_count=int(terminal_document_count),
                metadata=dict(metadata or {}),
            )
        )
    _CACHED_DAG_NODES = dag_nodes
    return dag_nodes


def load_current_tenant_overlays(
    *,
    refresh: bool = False,
    db_connection_factory=_default_db_connection_factory,
) -> list[dict[str, object]]:
    global _CACHED_TENANT_OVERLAYS
    if not refresh and _CACHED_TENANT_OVERLAYS is not None:
        return _CACHED_TENANT_OVERLAYS

    plan_summary = get_current_plan_summary(refresh=refresh, db_connection_factory=db_connection_factory)
    if plan_summary is None:
        _CACHED_TENANT_OVERLAYS = []
        return []

    raw_overlays = (plan_summary.get("metadata", {}) or {}).get("tenant_overlays", []) or []
    overlays: list[dict[str, object]] = []
    for raw_overlay in raw_overlays:
        overlay = dict(raw_overlay or {})
        if "tenant_id" not in overlay or "table_name" not in overlay:
            continue
        overlay["tenant_id"] = int(overlay["tenant_id"])
        overlay["table_name"] = str(overlay["table_name"])
        overlay["document_ids"] = [int(document_id) for document_id in (overlay.get("document_ids", []) or [])]
        overlay["pattern_ids"] = [int(pattern_id) for pattern_id in (overlay.get("pattern_ids", []) or [])]
        overlay["document_pattern_pairs"] = [
            [int(document_id), int(pattern_id)]
            for document_id, pattern_id in (overlay.get("document_pattern_pairs", []) or [])
        ]
        overlay["document_count"] = int(overlay.get("document_count", len(overlay["document_ids"])) or 0)
        overlay["vector_count"] = int(overlay.get("vector_count", 0) or 0)
        overlays.append(overlay)
    _CACHED_TENANT_OVERLAYS = overlays
    return overlays


def load_current_access_overlays(
    *,
    refresh: bool = False,
    db_connection_factory=_default_db_connection_factory,
) -> list[dict[str, object]]:
    global _CACHED_ACCESS_OVERLAYS
    if not refresh and _CACHED_ACCESS_OVERLAYS is not None:
        return _CACHED_ACCESS_OVERLAYS

    plan_summary = get_current_plan_summary(refresh=refresh, db_connection_factory=db_connection_factory)
    if plan_summary is None:
        _CACHED_ACCESS_OVERLAYS = []
        return []

    raw_overlays = (plan_summary.get("metadata", {}) or {}).get("access_overlays", []) or []
    overlays: list[dict[str, object]] = []
    for raw_overlay in raw_overlays:
        overlay = dict(raw_overlay or {})
        if "tenant_id" not in overlay or "partition_id" not in overlay or "table_name" not in overlay:
            continue
        overlay["tenant_id"] = int(overlay["tenant_id"])
        overlay["partition_id"] = str(overlay["partition_id"])
        overlay["partition_ids"] = [
            str(partition_id)
            for partition_id in (overlay.get("partition_ids", []) or [overlay["partition_id"]])
        ]
        overlay["table_name"] = str(overlay["table_name"])
        overlay["document_ids"] = [int(document_id) for document_id in (overlay.get("document_ids", []) or [])]
        overlay["pattern_ids"] = [int(pattern_id) for pattern_id in (overlay.get("pattern_ids", []) or [])]
        overlay["document_pattern_pairs"] = [
            [int(document_id), int(pattern_id)]
            for document_id, pattern_id in (overlay.get("document_pattern_pairs", []) or [])
        ]
        overlay["document_count"] = int(overlay.get("document_count", len(overlay["document_ids"])) or 0)
        overlay["vector_count"] = int(overlay.get("vector_count", 0) or 0)
        overlays.append(overlay)
    _CACHED_ACCESS_OVERLAYS = overlays
    return overlays


def list_materialized_partition_tables(*, db_connection_factory=_default_db_connection_factory) -> list[str]:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE tablename LIKE %s
                  AND tablename NOT LIKE %s
                ORDER BY tablename;
                """,
                ["workload_documentblocks_partition_%", f"{_ACCELERATOR_TABLE_PREFIX}%"],
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def list_materialized_accelerator_tables(*, db_connection_factory=_default_db_connection_factory) -> list[str]:
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
                [f"{_ACCELERATOR_TABLE_PREFIX}%"],
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def list_materialized_overlay_tables(*, db_connection_factory=_default_db_connection_factory) -> list[str]:
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
                [f"{_OVERLAY_TABLE_PREFIX}%"],
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def list_materialized_access_overlay_tables(*, db_connection_factory=_default_db_connection_factory) -> list[str]:
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
                [f"{_ACCESS_OVERLAY_TABLE_PREFIX}%"],
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
                WHERE tablename LIKE %s
                  AND tablename NOT LIKE %s;
                """,
                ["workload_documentblocks_partition_%", f"{_ACCELERATOR_TABLE_PREFIX}%"],
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


def drop_materialized_accelerators(
    *,
    valid_table_names: Optional[Iterable[str]] = None,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    valid_table_name_set = {str(table_name) for table_name in valid_table_names} if valid_table_names is not None else set()
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE tablename LIKE %s;
                """,
                [f"{_ACCELERATOR_TABLE_PREFIX}%"],
            )
            existing_table_names = [str(row[0]) for row in cur.fetchall()]
        for table_name in existing_table_names:
            if valid_table_name_set and table_name in valid_table_name_set:
                continue
            with conn.cursor() as cur:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name)))
            conn.commit()
    finally:
        conn.close()


def drop_materialized_overlays(
    *,
    valid_table_names: Optional[Iterable[str]] = None,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    valid_table_name_set = {str(table_name) for table_name in valid_table_names} if valid_table_names is not None else set()
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE tablename LIKE %s;
                """,
                [f"{_OVERLAY_TABLE_PREFIX}%"],
            )
            existing_table_names = [str(row[0]) for row in cur.fetchall()]
        for table_name in existing_table_names:
            if valid_table_name_set and table_name in valid_table_name_set:
                continue
            with conn.cursor() as cur:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name)))
            conn.commit()
    finally:
        conn.close()


def drop_materialized_access_overlays(
    *,
    valid_table_names: Optional[Iterable[str]] = None,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    valid_table_name_set = {str(table_name) for table_name in valid_table_names} if valid_table_names is not None else set()
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE tablename LIKE %s;
                """,
                [f"{_ACCESS_OVERLAY_TABLE_PREFIX}%"],
            )
            existing_table_names = [str(row[0]) for row in cur.fetchall()]
        for table_name in existing_table_names:
            if valid_table_name_set and table_name in valid_table_name_set:
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
    raw_document_pattern_pairs = partition.metadata.get("document_pattern_pairs", []) or []
    document_pattern_pairs = [
        (int(document_id), int(pattern_id))
        for document_id, pattern_id in raw_document_pattern_pairs
    ]
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        block_id BIGINT NOT NULL,
                        document_id INT NOT NULL REFERENCES Documents(document_id),
                        pattern_id INT NOT NULL,
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
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s;
                """,
                [partition.table_name],
            )
            existing_columns = {str(row[0]) for row in cur.fetchall()}
            if "pattern_id" not in existing_columns:
                cur.execute(
                    sql.SQL(
                        "ALTER TABLE {} ADD COLUMN pattern_id INT NOT NULL DEFAULT -1;"
                    ).format(sql.Identifier(partition.table_name))
                )
            cur.execute(sql.SQL("TRUNCATE TABLE {};").format(sql.Identifier(partition.table_name)))
            if document_pattern_pairs:
                cur.execute(
                    """
                    CREATE TEMP TABLE IF NOT EXISTS temp_method_partition_document_patterns (
                        document_id BIGINT NOT NULL,
                        pattern_id INTEGER NOT NULL
                    ) ON COMMIT DROP;
                    """
                )
                cur.execute("TRUNCATE TABLE temp_method_partition_document_patterns;")
                execute_values(
                    cur,
                    """
                    INSERT INTO temp_method_partition_document_patterns (document_id, pattern_id)
                    VALUES %s;
                    """,
                    document_pattern_pairs,
                    page_size=_PARTITION_DOCUMENT_BATCH_SIZE,
                )
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (block_id, document_id, pattern_id, block_content, vector)
                        SELECT db.block_id, db.document_id, tdp.pattern_id, db.block_content, db.vector
                        FROM documentblocks db
                        JOIN temp_method_partition_document_patterns tdp
                          ON tdp.document_id = db.document_id;
                        """
                    ).format(sql.Identifier(partition.table_name))
                )
            elif partition.document_count > 0:
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (block_id, document_id, pattern_id, block_content, vector)
                        SELECT db.block_id, db.document_id, -1, db.block_content, db.vector
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


def materialize_accelerator_pattern(
    partition: WorkloadAwarePartition,
    accelerator_pattern: dict[str, object],
    *,
    db_connection_factory=_default_db_connection_factory,
) -> str:
    conn = db_connection_factory()
    vector_dimension = get_document_vector_dimension()
    pattern_id = int(accelerator_pattern["pattern_id"])
    table_name = str(accelerator_pattern["table_name"])
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        block_id BIGINT NOT NULL,
                        document_id INT NOT NULL REFERENCES Documents(document_id),
                        pattern_id INT NOT NULL,
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
            cur.execute(sql.SQL("TRUNCATE TABLE {};").format(sql.Identifier(table_name)))
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (block_id, document_id, pattern_id, block_content, vector)
                    SELECT block_id, document_id, pattern_id, block_content, vector
                    FROM {}
                    WHERE pattern_id = %s;
                    """
                ).format(
                    sql.Identifier(table_name),
                    sql.Identifier(partition.table_name),
                ),
                [int(pattern_id)],
            )
        conn.commit()
        return table_name
    finally:
        conn.close()


def materialize_tenant_overlay(
    overlay: dict[str, object],
    *,
    db_connection_factory=_default_db_connection_factory,
) -> str:
    conn = db_connection_factory()
    vector_dimension = get_document_vector_dimension()
    table_name = str(overlay["table_name"])
    document_pattern_pairs = [
        (int(document_id), int(pattern_id))
        for document_id, pattern_id in (overlay.get("document_pattern_pairs", []) or [])
    ]
    if not document_pattern_pairs:
        document_pattern_pairs = [
            (int(document_id), -1)
            for document_id in (overlay.get("document_ids", []) or [])
        ]
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        block_id BIGINT NOT NULL,
                        document_id INT NOT NULL REFERENCES Documents(document_id),
                        pattern_id INT NOT NULL,
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
            cur.execute(sql.SQL("TRUNCATE TABLE {};").format(sql.Identifier(table_name)))
            if document_pattern_pairs:
                cur.execute(
                    """
                    CREATE TEMP TABLE IF NOT EXISTS temp_method_overlay_document_patterns (
                        document_id BIGINT NOT NULL,
                        pattern_id INTEGER NOT NULL
                    ) ON COMMIT DROP;
                    """
                )
                cur.execute("TRUNCATE TABLE temp_method_overlay_document_patterns;")
                execute_values(
                    cur,
                    """
                    INSERT INTO temp_method_overlay_document_patterns (document_id, pattern_id)
                    VALUES %s;
                    """,
                    document_pattern_pairs,
                    page_size=_PARTITION_DOCUMENT_BATCH_SIZE,
                )
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (block_id, document_id, pattern_id, block_content, vector)
                        SELECT db.block_id, db.document_id, todp.pattern_id, db.block_content, db.vector
                        FROM documentblocks db
                        JOIN temp_method_overlay_document_patterns todp
                          ON todp.document_id = db.document_id;
                        """
                    ).format(sql.Identifier(table_name))
                )
        conn.commit()
        return table_name
    finally:
        conn.close()


def _partition_materialization_matches(left: WorkloadAwarePartition, right: WorkloadAwarePartition) -> bool:
    return (
        str(left.table_name) == str(right.table_name)
        and tuple(int(document_id) for document_id in left.document_ids)
        == tuple(int(document_id) for document_id in right.document_ids)
        and tuple(int(pattern_id) for pattern_id in left.logical_pattern_ids)
        == tuple(int(pattern_id) for pattern_id in right.logical_pattern_ids)
        and tuple(
            (int(document_id), int(pattern_id))
            for document_id, pattern_id in (left.metadata.get("document_pattern_pairs", []) or [])
        )
        == tuple(
            (int(document_id), int(pattern_id))
            for document_id, pattern_id in (right.metadata.get("document_pattern_pairs", []) or [])
        )
        and int(left.vector_count) == int(right.vector_count)
        and tuple(int(tenant_id) for tenant_id in left.tenant_ids)
        == tuple(int(tenant_id) for tenant_id in right.tenant_ids)
        and int(left.metadata.get("storage_layout_version", 1) or 1)
        == int(right.metadata.get("storage_layout_version", 1) or 1)
    )


def _overlay_materialization_matches(left: dict[str, object], right: dict[str, object]) -> bool:
    return (
        int(left.get("tenant_id", -1)) == int(right.get("tenant_id", -2))
        and str(left.get("table_name", "")) == str(right.get("table_name", ""))
        and str(left.get("partition_id", "")) == str(right.get("partition_id", ""))
        and tuple(str(partition_id) for partition_id in (left.get("partition_ids", []) or []))
        == tuple(str(partition_id) for partition_id in (right.get("partition_ids", []) or []))
        and tuple(int(pattern_id) for pattern_id in (left.get("pattern_ids", []) or []))
        == tuple(int(pattern_id) for pattern_id in (right.get("pattern_ids", []) or []))
        and tuple(int(document_id) for document_id in (left.get("document_ids", []) or []))
        == tuple(int(document_id) for document_id in (right.get("document_ids", []) or []))
        and tuple(
            (int(document_id), int(pattern_id))
            for document_id, pattern_id in (left.get("document_pattern_pairs", []) or [])
        )
        == tuple(
            (int(document_id), int(pattern_id))
            for document_id, pattern_id in (right.get("document_pattern_pairs", []) or [])
        )
        and int(left.get("vector_count", 0) or 0) == int(right.get("vector_count", 0) or 0)
        and str(left.get("overlay_type", "") or "") == str(right.get("overlay_type", "") or "")
        and bool(left.get("requires_pattern_filter", False)) == bool(right.get("requires_pattern_filter", False))
    )


def _materialize_partition_timed(partition: WorkloadAwarePartition) -> tuple[str, float]:
    started_at = time.time()
    table_name = materialize_partition(partition)
    return table_name, time.time() - started_at


def _materialize_accelerator_timed(partition: WorkloadAwarePartition, accelerator_pattern: dict[str, object]) -> tuple[str, float]:
    started_at = time.time()
    table_name = materialize_accelerator_pattern(partition, accelerator_pattern)
    return table_name, time.time() - started_at


def _materialize_overlay_timed(overlay: dict[str, object]) -> tuple[str, float]:
    started_at = time.time()
    table_name = materialize_tenant_overlay(overlay)
    return table_name, time.time() - started_at


def materialize_planner_result(
    plan: WorkloadAwarePlan,
    *,
    create_indexes: bool = False,
    index_type: str = "hnsw",
    db_connection_factory=_default_db_connection_factory,
) -> WorkloadAwarePlan:
    _plan_progress("[plan][materialize] waiting for global materialization lock...")
    lock_conn = _acquire_materialize_lock(db_connection_factory=db_connection_factory)
    _plan_progress("[plan][materialize] acquired global materialization lock")
    try:
        existing_partitions = {
            partition.partition_id: partition
            for partition in load_current_partitions(refresh=True, db_connection_factory=db_connection_factory)
        }
        existing_overlays = {
            int(overlay["tenant_id"]): overlay
            for overlay in load_current_tenant_overlays(refresh=True, db_connection_factory=db_connection_factory)
        }
        existing_access_overlays = {
            (int(overlay["tenant_id"]), str(overlay["partition_id"])): overlay
            for overlay in load_current_access_overlays(refresh=True, db_connection_factory=db_connection_factory)
        }
        existing_table_names = set(list_materialized_partition_tables(db_connection_factory=db_connection_factory))
        existing_overlay_table_names = set(list_materialized_overlay_tables(db_connection_factory=db_connection_factory))
        existing_access_overlay_table_names = set(list_materialized_access_overlay_tables(db_connection_factory=db_connection_factory))
        tenant_overlays = [dict(overlay) for overlay in (plan.metadata.get("tenant_overlays", []) or [])]
        access_overlays = [dict(overlay) for overlay in (plan.metadata.get("access_overlays", []) or [])]

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

        reusable_overlay_tenant_ids: list[int] = []
        overlays_to_materialize: list[dict[str, object]] = []
        for overlay in tenant_overlays:
            tenant_id = int(overlay.get("tenant_id", -1))
            existing_overlay = existing_overlays.get(tenant_id)
            if (
                existing_overlay is not None
                and str(overlay.get("table_name")) in existing_overlay_table_names
                and _overlay_materialization_matches(overlay, existing_overlay)
            ):
                reusable_overlay_tenant_ids.append(int(tenant_id))
                continue
            overlays_to_materialize.append(overlay)

        reusable_access_overlay_keys: list[tuple[int, str]] = []
        access_overlays_to_materialize: list[dict[str, object]] = []
        for overlay in access_overlays:
            tenant_id = int(overlay.get("tenant_id", -1))
            partition_id = str(overlay.get("partition_id", ""))
            existing_overlay = existing_access_overlays.get((tenant_id, partition_id))
            if (
                existing_overlay is not None
                and str(overlay.get("table_name")) in existing_access_overlay_table_names
                and _overlay_materialization_matches(overlay, existing_overlay)
            ):
                reusable_access_overlay_keys.append((int(tenant_id), str(partition_id)))
                continue
            access_overlays_to_materialize.append(overlay)

        _plan_progress(f"[plan][materialize] saving metadata for {len(plan.partitions)} partitions...")
        save_plan_result(plan, db_connection_factory=db_connection_factory)
        drop_materialized_partitions(
            valid_partition_ids=[partition.partition_id for partition in plan.partitions],
            db_connection_factory=db_connection_factory,
        )
        valid_accelerator_tables = [
            str(accelerator_pattern["table_name"])
            for partition in plan.partitions
            for accelerator_pattern in (partition.metadata.get("accelerator_patterns", []) or [])
        ]
        drop_materialized_accelerators(
            valid_table_names=valid_accelerator_tables,
            db_connection_factory=db_connection_factory,
        )
        valid_overlay_tables = [
            str(overlay["table_name"])
            for overlay in tenant_overlays
            if overlay.get("table_name")
        ]
        drop_materialized_overlays(
            valid_table_names=valid_overlay_tables,
            db_connection_factory=db_connection_factory,
        )
        valid_access_overlay_tables = [
            str(overlay["table_name"])
            for overlay in access_overlays
            if overlay.get("table_name")
        ]
        drop_materialized_access_overlays(
            valid_table_names=valid_access_overlay_tables,
            db_connection_factory=db_connection_factory,
        )

        if reusable_partition_ids:
            _plan_progress(
                f"[plan][materialize] reusing {len(reusable_partition_ids)} unchanged partition tables"
            )
        if reusable_overlay_tenant_ids:
            _plan_progress(
                f"[plan][materialize] reusing {len(reusable_overlay_tenant_ids)} unchanged overlay tables"
            )
        if reusable_access_overlay_keys:
            _plan_progress(
                f"[plan][materialize] reusing {len(reusable_access_overlay_keys)} unchanged access overlay tables"
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
        accelerator_jobs = [
            (partition, dict(accelerator_pattern))
            for partition in plan.partitions
            for accelerator_pattern in (partition.metadata.get("accelerator_patterns", []) or [])
        ]
        if accelerator_jobs:
            _plan_progress(
                f"[plan][materialize] materializing {len(accelerator_jobs)} accelerator tables..."
            )
            accelerator_report_every = max(1, len(accelerator_jobs) // 20)
            accelerator_worker_count = _recommended_worker_count(
                len(accelerator_jobs),
                hard_cap=_DEFAULT_MATERIALIZE_MAX_WORKERS,
            )
            if len(accelerator_jobs) > 1 and db_connection_factory is _default_db_connection_factory and accelerator_worker_count > 1:
                with ThreadPoolExecutor(max_workers=accelerator_worker_count) as executor:
                    future_to_job = {
                        executor.submit(_materialize_accelerator_timed, partition, accelerator_pattern): (partition, accelerator_pattern)
                        for partition, accelerator_pattern in accelerator_jobs
                    }
                    completed = 0
                    for future in as_completed(future_to_job):
                        completed += 1
                        table_name, elapsed = future.result()
                        if completed == 1 or completed % accelerator_report_every == 0 or completed == len(accelerator_jobs):
                            _plan_progress(
                                f"[plan][materialize] [accel {completed}/{len(accelerator_jobs)}] materialized {table_name} in {elapsed:.2f}s"
                            )
            else:
                for index, (partition, accelerator_pattern) in enumerate(accelerator_jobs, start=1):
                    started_at = time.time()
                    table_name = materialize_accelerator_pattern(
                        partition,
                        accelerator_pattern,
                        db_connection_factory=db_connection_factory,
                    )
                    elapsed = time.time() - started_at
                    if index == 1 or index % accelerator_report_every == 0 or index == len(accelerator_jobs):
                        _plan_progress(
                            f"[plan][materialize] [accel {index}/{len(accelerator_jobs)}] materialized {table_name} in {elapsed:.2f}s"
                        )
        overlay_count = len(overlays_to_materialize)
        if overlay_count:
            _plan_progress(
                f"[plan][materialize] materializing {overlay_count} tenant overlay tables..."
            )
            overlay_report_every = max(1, overlay_count // 20)
            overlay_worker_count = _recommended_worker_count(
                overlay_count,
                hard_cap=_DEFAULT_MATERIALIZE_MAX_WORKERS,
            )
            ordered_overlays = sorted(
                overlays_to_materialize,
                key=lambda overlay: (
                    -int(overlay.get("vector_count", 0) or 0),
                    -float(overlay.get("query_mass", 0.0) or 0.0),
                    int(overlay.get("tenant_id", 0) or 0),
                ),
            )
            if overlay_count > 1 and db_connection_factory is _default_db_connection_factory and overlay_worker_count > 1:
                with ThreadPoolExecutor(max_workers=overlay_worker_count) as executor:
                    future_to_overlay = {
                        executor.submit(_materialize_overlay_timed, overlay): overlay
                        for overlay in ordered_overlays
                    }
                    completed = 0
                    for future in as_completed(future_to_overlay):
                        completed += 1
                        table_name, elapsed = future.result()
                        if completed == 1 or completed % overlay_report_every == 0 or completed == overlay_count:
                            _plan_progress(
                                f"[plan][materialize] [overlay {completed}/{overlay_count}] materialized {table_name} in {elapsed:.2f}s"
                            )
            else:
                for index, overlay in enumerate(ordered_overlays, start=1):
                    started_at = time.time()
                    table_name = materialize_tenant_overlay(
                        overlay,
                        db_connection_factory=db_connection_factory,
                    )
                    elapsed = time.time() - started_at
                    if index == 1 or index % overlay_report_every == 0 or index == overlay_count:
                        _plan_progress(
                            f"[plan][materialize] [overlay {index}/{overlay_count}] materialized {table_name} in {elapsed:.2f}s"
                        )
        access_overlay_count = len(access_overlays_to_materialize)
        if access_overlay_count:
            unique_access_overlays_to_materialize = {
                str(overlay.get("table_name")): overlay
                for overlay in access_overlays_to_materialize
                if overlay.get("table_name")
            }
            access_overlays_to_materialize = list(unique_access_overlays_to_materialize.values())
            access_overlay_count = len(access_overlays_to_materialize)
            _plan_progress(
                f"[plan][materialize] materializing {access_overlay_count} access overlay tables..."
            )
            access_overlay_report_every = max(1, access_overlay_count // 20)
            access_overlay_worker_count = _recommended_worker_count(
                access_overlay_count,
                hard_cap=_DEFAULT_MATERIALIZE_MAX_WORKERS,
            )
            ordered_access_overlays = sorted(
                access_overlays_to_materialize,
                key=lambda overlay: (
                    -int(overlay.get("vector_count", 0) or 0),
                    -float(overlay.get("benefit_density", 0.0) or 0.0),
                    int(overlay.get("tenant_id", 0) or 0),
                    str(overlay.get("partition_id", "")),
                ),
            )
            if access_overlay_count > 1 and db_connection_factory is _default_db_connection_factory and access_overlay_worker_count > 1:
                with ThreadPoolExecutor(max_workers=access_overlay_worker_count) as executor:
                    future_to_overlay = {
                        executor.submit(_materialize_overlay_timed, overlay): overlay
                        for overlay in ordered_access_overlays
                    }
                    completed = 0
                    for future in as_completed(future_to_overlay):
                        completed += 1
                        table_name, elapsed = future.result()
                        if completed == 1 or completed % access_overlay_report_every == 0 or completed == access_overlay_count:
                            _plan_progress(
                                f"[plan][materialize] [access-overlay {completed}/{access_overlay_count}] materialized {table_name} in {elapsed:.2f}s"
                            )
            else:
                for index, overlay in enumerate(ordered_access_overlays, start=1):
                    started_at = time.time()
                    table_name = materialize_tenant_overlay(
                        overlay,
                        db_connection_factory=db_connection_factory,
                    )
                    elapsed = time.time() - started_at
                    if index == 1 or index % access_overlay_report_every == 0 or index == access_overlay_count:
                        _plan_progress(
                            f"[plan][materialize] [access-overlay {index}/{access_overlay_count}] materialized {table_name} in {elapsed:.2f}s"
                        )
        if create_indexes:
            _plan_progress(f"[plan][materialize] creating {index_type} indexes...")
            create_indexes_for_materialized_partitions(index_type=index_type, db_connection_factory=db_connection_factory)
        _plan_progress("[plan] completed")
        return plan
    finally:
        _release_materialize_lock(lock_conn)


def build_and_materialize_workload_aware_plan(
    *,
    min_pattern_support: int = 16,
    min_pattern_query_mass: float = 0.0,
    safe_density_threshold: float = 0.35,
    supplemental_edge_penalty: float = 0.25,
    supplemental_edge_gain_threshold: float = 0.0,
    target_partition_count: Optional[int] = None,
    max_partition_vector_count: Optional[int] = None,
    overlay_space_ratio: float = 0.25,
    protection_overlay_space_ratio: Optional[float] = None,
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
        overlay_space_ratio=overlay_space_ratio,
        protection_overlay_space_ratio=protection_overlay_space_ratio,
        progress_fn=_plan_progress,
    )
    _plan_progress(
        f"[plan][4/5] logical plan ready: acl_patterns={len(plan.logical_patterns)}, "
        f"partitions={len(plan.partitions)}, overlays={len(plan.metadata.get('tenant_overlays', []) or [])}, "
        f"access_overlays={len(plan.metadata.get('access_overlays', []) or [])}"
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
    include_pattern_indexes: bool = True,
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
            cur.execute(
                """
                SELECT column_name
                FROM information_schema.columns
                WHERE table_name = %s;
                """,
                [table_name],
            )
            existing_columns = {str(row[0]) for row in cur.fetchall()}
            if "pattern_id" not in existing_columns:
                cur.execute(
                    sql.SQL(
                        "ALTER TABLE {} ADD COLUMN pattern_id INT NOT NULL DEFAULT -1;"
                    ).format(sql.Identifier(table_name))
                )
            if include_pattern_indexes:
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {}
                        ON {} (pattern_id);
                        """
                    ).format(
                        sql.Identifier(_safe_index_name(table_name, "pattern_idx")),
                        sql.Identifier(table_name),
                    )
                )
                cur.execute(
                    sql.SQL(
                        """
                        CREATE INDEX IF NOT EXISTS {}
                        ON {} (pattern_id, document_id);
                        """
                    ).format(
                        sql.Identifier(_safe_index_name(table_name, "pattern_document_idx")),
                        sql.Identifier(table_name),
                    )
                )
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
                        sql.Identifier(_safe_index_name(table_name, "vector_idx")),
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
    include_pattern_indexes: bool,
    hnsw_m: int,
    hnsw_ef_construction: int,
    hnsw_threads: Optional[int],
    disable_sync_commit: bool,
) -> tuple[str, float]:
    started_at = time.time()
    create_index_for_partition(
        table_name,
        index_type=index_type,
        include_pattern_indexes=include_pattern_indexes,
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
    table_names = (
        list_materialized_partition_tables(db_connection_factory=db_connection_factory)
        + list_materialized_accelerator_tables(db_connection_factory=db_connection_factory)
        + list_materialized_overlay_tables(db_connection_factory=db_connection_factory)
        + list_materialized_access_overlay_tables(db_connection_factory=db_connection_factory)
    )
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
                    include_pattern_indexes=not (
                        str(table_name).startswith(_OVERLAY_TABLE_PREFIX)
                        or str(table_name).startswith(_ACCESS_OVERLAY_TABLE_PREFIX)
                    ),
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
            include_pattern_indexes=not (
                str(table_name).startswith(_OVERLAY_TABLE_PREFIX)
                or str(table_name).startswith(_ACCESS_OVERLAY_TABLE_PREFIX)
            ),
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
            for table_name in (
                list_materialized_partition_tables(db_connection_factory=db_connection_factory)
                + list_materialized_accelerator_tables(db_connection_factory=db_connection_factory)
                + list_materialized_overlay_tables(db_connection_factory=db_connection_factory)
                + list_materialized_access_overlay_tables(db_connection_factory=db_connection_factory)
            ):
                cur.execute(
                    """
                    SELECT index_class.relname
                    FROM pg_index idx
                    JOIN pg_class table_class
                      ON table_class.oid = idx.indrelid
                    JOIN pg_namespace ns
                      ON ns.oid = table_class.relnamespace
                    JOIN pg_class index_class
                      ON index_class.oid = idx.indexrelid
                    WHERE ns.nspname = current_schema()
                      AND table_class.relname = %s
                      AND NOT idx.indisprimary
                      AND NOT idx.indisunique;
                    """,
                    [str(table_name)],
                )
                index_names = [str(row[0]) for row in cur.fetchall()]
                for index_name in index_names:
                    cur.execute(
                        sql.SQL("DROP INDEX IF EXISTS {} CASCADE;").format(sql.Identifier(index_name))
                    )
        conn.commit()
    finally:
        conn.close()
