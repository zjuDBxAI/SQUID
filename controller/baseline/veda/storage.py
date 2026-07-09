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
    VEDA_NODE_TABLE,
    VEDA_NODE_TABLE_PREFIX,
    VEDA_PATTERN_TABLE,
    VEDA_PLAN_TABLE,
    get_node_table_prefix,
    VEDA_ROLE_PLAN_TABLE,
    VEDA_ROUTE_TABLE,
    VedaNode,
    VedaPattern,
    VedaPlan,
    VedaRoute,
    normalize_algorithm,
    normalize_int_tuple,
)
from .planner import VedaPlanner
from .repository import VedaRepository


_DEFAULT_INDEX_MAX_WORKERS = 6
_MATERIALIZE_BATCH_SIZE = 8192
_MATERIALIZE_ADVISORY_LOCK_KEY = 2026053001
_POSTGRES_IDENTIFIER_LIMIT = 63

_CACHED_PLAN_SUMMARY: dict[str, dict[str, object] | None] = {}
_CACHED_NODES: dict[str, list[VedaNode]] = {}
_CACHED_ROUTES: dict[tuple[str, int], list[VedaRoute]] = {}


def _default_db_connection_factory():
    return get_db_connection()


def _configured_algorithm(default: str = "effveda") -> str:
    try:
        from basic_benchmark import efconfig  # type: ignore
    except Exception:
        efconfig = None
    value = getattr(efconfig, "veda_algorithm", default) if efconfig is not None else default
    return normalize_algorithm(str(value or default))


def invalidate_cache() -> None:
    global _CACHED_PLAN_SUMMARY, _CACHED_NODES, _CACHED_ROUTES
    _CACHED_PLAN_SUMMARY = {}
    _CACHED_NODES = {}
    _CACHED_ROUTES = {}


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


def initialize_schema(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        plan_id BIGSERIAL PRIMARY KEY,
                        algorithm TEXT NOT NULL,
                        role_count INTEGER NOT NULL,
                        pattern_count INTEGER NOT NULL,
                        node_count INTEGER NOT NULL,
                        index_node_count INTEGER NOT NULL,
                        leftover_node_count INTEGER NOT NULL,
                        document_count BIGINT NOT NULL,
                        original_vector_count BIGINT NOT NULL,
                        materialized_vector_count BIGINT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                ).format(sql.Identifier(VEDA_PLAN_TABLE))
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        plan_id BIGINT NOT NULL REFERENCES {}(plan_id) ON DELETE CASCADE,
                        pattern_id BIGINT NOT NULL,
                        role_ids BIGINT[] NOT NULL,
                        document_ids BIGINT[] NOT NULL,
                        document_count BIGINT NOT NULL,
                        vector_count BIGINT NOT NULL,
                        PRIMARY KEY (plan_id, pattern_id)
                    );
                    """
                ).format(sql.Identifier(VEDA_PATTERN_TABLE), sql.Identifier(VEDA_PLAN_TABLE))
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        plan_id BIGINT NOT NULL REFERENCES {}(plan_id) ON DELETE CASCADE,
                        node_id TEXT NOT NULL,
                        role_ids BIGINT[] NOT NULL,
                        pattern_ids BIGINT[] NOT NULL,
                        document_ids BIGINT[] NOT NULL,
                        document_pattern_pairs JSONB NOT NULL DEFAULT '[]'::jsonb,
                        vector_count BIGINT NOT NULL,
                        node_kind TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        PRIMARY KEY (plan_id, node_id)
                    );
                    """
                ).format(sql.Identifier(VEDA_NODE_TABLE), sql.Identifier(VEDA_PLAN_TABLE))
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        plan_id BIGINT NOT NULL REFERENCES {}(plan_id) ON DELETE CASCADE,
                        role_id BIGINT NOT NULL,
                        node_ids TEXT[] NOT NULL,
                        PRIMARY KEY (plan_id, role_id)
                    );
                    """
                ).format(sql.Identifier(VEDA_ROLE_PLAN_TABLE), sql.Identifier(VEDA_PLAN_TABLE))
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        plan_id BIGINT NOT NULL REFERENCES {}(plan_id) ON DELETE CASCADE,
                        user_id BIGINT NOT NULL,
                        node_id TEXT NOT NULL,
                        table_name TEXT NOT NULL,
                        route_kind TEXT NOT NULL,
                        pattern_ids BIGINT[] NOT NULL,
                        node_vector_count BIGINT NOT NULL,
                        accessible_vector_count BIGINT NOT NULL,
                        impurity_factor DOUBLE PRECISION NOT NULL,
                        PRIMARY KEY (plan_id, user_id, node_id)
                    );
                    """
                ).format(sql.Identifier(VEDA_ROUTE_TABLE), sql.Identifier(VEDA_PLAN_TABLE))
            )
            cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (user_id);").format(
                sql.Identifier(f"idx_{VEDA_ROUTE_TABLE}_user"),
                sql.Identifier(VEDA_ROUTE_TABLE),
            ))
            cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (role_ids);").format(
                sql.Identifier(f"idx_{VEDA_PATTERN_TABLE}_role_ids"),
                sql.Identifier(VEDA_PATTERN_TABLE),
            ))
            cur.execute(sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (pattern_ids);").format(
                sql.Identifier(f"idx_{VEDA_NODE_TABLE}_pattern_ids"),
                sql.Identifier(VEDA_NODE_TABLE),
            ))
        conn.commit()
    finally:
        conn.close()


def clear_current_plan(*, algorithm: str | None = None, db_connection_factory=_default_db_connection_factory) -> None:
    invalidate_cache()
    initialize_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            if algorithm is None:
                cur.execute(sql.SQL("DELETE FROM {};").format(sql.Identifier(VEDA_PLAN_TABLE)))
            else:
                cur.execute(
                    sql.SQL("DELETE FROM {} WHERE algorithm = %s;").format(sql.Identifier(VEDA_PLAN_TABLE)),
                    [normalize_algorithm(algorithm)],
                )
        conn.commit()
    finally:
        conn.close()


def save_plan(plan: VedaPlan, *, db_connection_factory=_default_db_connection_factory) -> int:
    invalidate_cache()
    initialize_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL("DELETE FROM {} WHERE algorithm = %s;").format(sql.Identifier(VEDA_PLAN_TABLE)),
                [normalize_algorithm(plan.algorithm)],
            )
            metadata = dict(plan.metadata or {})
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (
                        algorithm, role_count, pattern_count, node_count,
                        index_node_count, leftover_node_count,
                        document_count, original_vector_count,
                        materialized_vector_count, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb)
                    RETURNING plan_id;
                    """
                ).format(sql.Identifier(VEDA_PLAN_TABLE)),
                [
                    str(plan.algorithm),
                    int(metadata.get("role_count", 0)),
                    int(len(plan.patterns)),
                    int(len(plan.nodes)),
                    int(metadata.get("index_node_count", 0)),
                    int(metadata.get("leftover_node_count", 0)),
                    int(metadata.get("document_count", 0)),
                    int(metadata.get("original_vector_count", 0)),
                    int(metadata.get("materialized_vector_count", 0)),
                    json.dumps(metadata),
                ],
            )
            plan_id = int(cur.fetchone()[0])

            pattern_rows = [
                (
                    plan_id,
                    int(pattern.pattern_id),
                    list(pattern.role_ids),
                    list(pattern.document_ids),
                    int(pattern.document_count),
                    int(pattern.vector_count),
                )
                for pattern in plan.patterns
            ]
            if pattern_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {VEDA_PATTERN_TABLE} (
                        plan_id, pattern_id, role_ids, document_ids,
                        document_count, vector_count
                    )
                    VALUES %s;
                    """,
                    pattern_rows,
                    page_size=_MATERIALIZE_BATCH_SIZE,
                )

            node_rows = [
                (
                    plan_id,
                    str(node.node_id),
                    list(node.role_ids),
                    list(node.pattern_ids),
                    list(node.document_ids),
                    json.dumps([[int(document_id), int(pattern_id)] for document_id, pattern_id in node.document_pattern_pairs]),
                    int(node.vector_count),
                    str(node.node_kind),
                    str(node.table_name),
                    json.dumps(node.metadata),
                )
                for node in plan.nodes
            ]
            if node_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {VEDA_NODE_TABLE} (
                        plan_id, node_id, role_ids, pattern_ids, document_ids,
                        document_pattern_pairs, vector_count, node_kind,
                        table_name, metadata
                    )
                    VALUES %s;
                    """,
                    node_rows,
                    template="(%s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s::jsonb)",
                    page_size=_MATERIALIZE_BATCH_SIZE,
                )

            role_plan_rows = [
                (plan_id, int(role_id), list(node_ids))
                for role_id, node_ids in sorted(plan.role_plans.items())
            ]
            if role_plan_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {VEDA_ROLE_PLAN_TABLE} (plan_id, role_id, node_ids)
                    VALUES %s;
                    """,
                    role_plan_rows,
                    page_size=_MATERIALIZE_BATCH_SIZE,
                )

            route_rows = [
                (
                    plan_id,
                    int(route.user_id),
                    str(route.node_id),
                    str(route.table_name),
                    str(route.route_kind),
                    list(route.pattern_ids),
                    int(route.node_vector_count),
                    int(route.accessible_vector_count),
                    float(route.impurity_factor),
                )
                for route in plan.user_routes
            ]
            if route_rows:
                execute_values(
                    cur,
                    f"""
                    INSERT INTO {VEDA_ROUTE_TABLE} (
                        plan_id, user_id, node_id, table_name, route_kind,
                        pattern_ids, node_vector_count, accessible_vector_count,
                        impurity_factor
                    )
                    VALUES %s;
                    """,
                    route_rows,
                    page_size=_MATERIALIZE_BATCH_SIZE,
                )
        conn.commit()
        return plan_id
    finally:
        conn.close()


def get_current_plan_summary(*, algorithm: str | None = None, refresh: bool = False, db_connection_factory=_default_db_connection_factory) -> Optional[dict[str, object]]:
    active_algorithm = _configured_algorithm() if algorithm is None else normalize_algorithm(algorithm)
    global _CACHED_PLAN_SUMMARY
    if not refresh and active_algorithm in _CACHED_PLAN_SUMMARY:
        return _CACHED_PLAN_SUMMARY[active_algorithm]
    initialize_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT plan_id, algorithm, role_count, pattern_count, node_count,
                           index_node_count, leftover_node_count, document_count,
                           original_vector_count, materialized_vector_count, metadata
                    FROM {}
                    WHERE algorithm = %s
                    ORDER BY plan_id DESC
                    LIMIT 1;
                    """
                ).format(sql.Identifier(VEDA_PLAN_TABLE)),
                [active_algorithm],
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        _CACHED_PLAN_SUMMARY[active_algorithm] = None
        return None
    summary = {
        "plan_id": int(row[0]),
        "algorithm": str(row[1]),
        "role_count": int(row[2]),
        "pattern_count": int(row[3]),
        "node_count": int(row[4]),
        "index_node_count": int(row[5]),
        "leftover_node_count": int(row[6]),
        "document_count": int(row[7]),
        "original_vector_count": int(row[8]),
        "materialized_vector_count": int(row[9]),
        "metadata": dict(row[10] or {}),
    }
    _CACHED_PLAN_SUMMARY[active_algorithm] = summary
    return summary


def load_current_nodes(*, algorithm: str | None = None, refresh: bool = False, db_connection_factory=_default_db_connection_factory) -> list[VedaNode]:
    active_algorithm = _configured_algorithm() if algorithm is None else normalize_algorithm(algorithm)
    global _CACHED_NODES
    if not refresh and active_algorithm in _CACHED_NODES:
        return _CACHED_NODES[active_algorithm]
    summary = get_current_plan_summary(algorithm=active_algorithm, refresh=refresh, db_connection_factory=db_connection_factory)
    if summary is None:
        _CACHED_NODES[active_algorithm] = []
        return []
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT node_id, role_ids, pattern_ids, document_ids,
                           document_pattern_pairs, vector_count, node_kind,
                           table_name, metadata
                    FROM {}
                    WHERE plan_id = %s
                    ORDER BY node_id;
                    """
                ).format(sql.Identifier(VEDA_NODE_TABLE)),
                [int(summary["plan_id"])],
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    nodes = [
        VedaNode(
            node_id=str(row[0]),
            role_ids=normalize_int_tuple(row[1] or ()),
            pattern_ids=normalize_int_tuple(row[2] or ()),
            document_ids=normalize_int_tuple(row[3] or ()),
            document_pattern_pairs=tuple((int(left), int(right)) for left, right in (row[4] or [])),
            vector_count=int(row[5]),
            node_kind=str(row[6]),
            table_name=str(row[7]),
            metadata=dict(row[8] or {}),
        )
        for row in rows
    ]
    _CACHED_NODES[active_algorithm] = nodes
    return nodes


def load_current_partitions(*, algorithm: str | None = None, refresh: bool = False, db_connection_factory=_default_db_connection_factory) -> list[VedaNode]:
    return load_current_nodes(algorithm=algorithm, refresh=refresh, db_connection_factory=db_connection_factory)


def load_user_routes(user_id: int, *, algorithm: str | None = None, refresh: bool = False, db_connection_factory=_default_db_connection_factory) -> list[VedaRoute]:
    active_algorithm = _configured_algorithm() if algorithm is None else normalize_algorithm(algorithm)
    global _CACHED_ROUTES
    user_id = int(user_id)
    cache_key = (active_algorithm, user_id)
    if not refresh and cache_key in _CACHED_ROUTES:
        return _CACHED_ROUTES[cache_key]
    summary = get_current_plan_summary(algorithm=active_algorithm, refresh=refresh, db_connection_factory=db_connection_factory)
    if summary is None:
        _CACHED_ROUTES[cache_key] = []
        return []
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT user_id, node_id, table_name, route_kind, pattern_ids,
                           node_vector_count, accessible_vector_count, impurity_factor
                    FROM {}
                    WHERE plan_id = %s
                      AND user_id = %s
                    ORDER BY route_kind, node_id;
                    """
                ).format(sql.Identifier(VEDA_ROUTE_TABLE)),
                [int(summary["plan_id"]), int(user_id)],
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    routes = [
        VedaRoute(
            user_id=int(row[0]),
            node_id=str(row[1]),
            table_name=str(row[2]),
            route_kind=str(row[3]),
            pattern_ids=normalize_int_tuple(row[4] or ()),
            node_vector_count=int(row[5] or 0),
            accessible_vector_count=int(row[6] or 0),
            impurity_factor=float(row[7] or 1.0),
        )
        for row in rows
    ]
    _CACHED_ROUTES[cache_key] = routes
    return routes


def list_materialized_node_tables(*, algorithm: str | None = None, db_connection_factory=_default_db_connection_factory) -> list[str]:
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
                [f"{get_node_table_prefix(algorithm)}%"],
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def list_materialized_partition_tables(*, algorithm: str | None = None, db_connection_factory=_default_db_connection_factory) -> list[str]:
    return list_materialized_node_tables(algorithm=algorithm, db_connection_factory=db_connection_factory)


def list_current_plan_partition_tables(*, algorithm: str | None = None, db_connection_factory=_default_db_connection_factory) -> list[str]:
    active_algorithm = _configured_algorithm() if algorithm is None else normalize_algorithm(algorithm)
    summary = get_current_plan_summary(algorithm=active_algorithm, db_connection_factory=db_connection_factory)
    if summary is None:
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
                    ORDER BY node_id;
                    """
                ).format(sql.Identifier(VEDA_NODE_TABLE)),
                [int(summary["plan_id"])],
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def _pattern_role_map(patterns: list[VedaPattern]) -> dict[int, tuple[int, ...]]:
    return {int(pattern.pattern_id): tuple(pattern.role_ids) for pattern in patterns}


def materialize_node(
    node: VedaNode,
    *,
    pattern_roles: dict[int, tuple[int, ...]],
    db_connection_factory=_default_db_connection_factory,
) -> str:
    conn = db_connection_factory()
    vector_dimension = get_document_vector_dimension()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(node.table_name)))
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE {} (
                        block_id BIGINT NOT NULL,
                        document_id INT NOT NULL REFERENCES Documents(document_id),
                        pattern_id BIGINT NOT NULL,
                        role_ids BIGINT[] NOT NULL,
                        block_content BYTEA NOT NULL,
                        vector VECTOR({dimension}),
                        PRIMARY KEY (block_id, document_id, pattern_id)
                    );
                    """
                ).format(sql.Identifier(node.table_name), dimension=sql.SQL(str(int(vector_dimension))))
            )
            cur.execute(
                """
                CREATE TEMP TABLE IF NOT EXISTS temp_veda_node_document_patterns (
                    document_id BIGINT NOT NULL,
                    pattern_id BIGINT NOT NULL,
                    role_ids BIGINT[] NOT NULL
                ) ON COMMIT DROP;
                """
            )
            cur.execute("TRUNCATE TABLE temp_veda_node_document_patterns;")
            rows = [
                (int(document_id), int(pattern_id), list(pattern_roles[int(pattern_id)]))
                for document_id, pattern_id in node.document_pattern_pairs
            ]
            if rows:
                execute_values(
                    cur,
                    """
                    INSERT INTO temp_veda_node_document_patterns (document_id, pattern_id, role_ids)
                    VALUES %s;
                    """,
                    rows,
                    page_size=_MATERIALIZE_BATCH_SIZE,
                )
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (block_id, document_id, pattern_id, role_ids, block_content, vector)
                        SELECT db.block_id, db.document_id, tvp.pattern_id, tvp.role_ids,
                               db.block_content, db.vector
                        FROM documentblocks db
                        JOIN temp_veda_node_document_patterns tvp
                          ON tvp.document_id = db.document_id;
                        """
                    ).format(sql.Identifier(node.table_name))
                )
        conn.commit()
        return node.table_name
    finally:
        conn.close()


def drop_stale_materialized_nodes(valid_table_names: set[str], *, algorithm: str | None = None, db_connection_factory=_default_db_connection_factory) -> None:
    existing = set(list_materialized_node_tables(algorithm=algorithm, db_connection_factory=db_connection_factory))
    for table_name in sorted(existing - set(valid_table_names)):
        conn = db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name)))
            conn.commit()
        finally:
            conn.close()


def build_veda_plan(
    *,
    algorithm: str = "effveda",
    indexing_threshold: int = 1000,
    storage_amplification: float = 1.2,
    ef_search: int = 100,
    document_limit: Optional[int] = None,
    show_progress: bool = True,
    db_connection_factory=_default_db_connection_factory,
) -> VedaPlan:
    repository = VedaRepository(db_connection_factory=db_connection_factory)
    patterns = repository.fetch_exclusive_patterns(document_limit=document_limit)
    role_ids = repository.fetch_role_universe()
    user_roles = repository.fetch_user_roles()
    planner = VedaPlanner(
        indexing_threshold=int(indexing_threshold),
        storage_amplification=float(storage_amplification),
        ef_search=int(ef_search),
    )
    return planner.build_plan(
        patterns,
        role_ids=role_ids,
        user_roles=user_roles,
        algorithm=normalize_algorithm(algorithm),
        show_progress=show_progress,
    )


def materialize_plan(
    plan: VedaPlan,
    *,
    create_indexes: bool = False,
    index_type: str = "hnsw",
    show_progress: bool = True,
    db_connection_factory=_default_db_connection_factory,
) -> VedaPlan:
    print("[veda][materialize] waiting for global materialization lock...", flush=True)
    lock_conn = _acquire_materialize_lock(db_connection_factory=db_connection_factory)
    print("[veda][materialize] acquired global materialization lock", flush=True)
    try:
        print(f"[veda][materialize] saving metadata for {len(plan.nodes)} nodes...", flush=True)
        save_plan(plan, db_connection_factory=db_connection_factory)
        valid_table_names = {str(node.table_name) for node in plan.nodes}
        drop_stale_materialized_nodes(valid_table_names, algorithm=plan.algorithm, db_connection_factory=db_connection_factory)
        pattern_roles = _pattern_role_map(plan.patterns)
        iterator = tqdm(
            list(enumerate(plan.nodes, start=1)),
            desc="Veda materialize nodes",
            unit="node",
            disable=not show_progress,
        )
        for index, node in iterator:
            started_at = time.time()
            table_name = materialize_node(node, pattern_roles=pattern_roles, db_connection_factory=db_connection_factory)
            elapsed = time.time() - started_at
            print(f"[veda][materialize] [{index}/{len(plan.nodes)}] materialized {table_name} in {elapsed:.2f}s", flush=True)
        if create_indexes:
            create_indexes_for_materialized_partitions(index_type=index_type, algorithm=plan.algorithm, db_connection_factory=db_connection_factory)
        invalidate_cache()
        return plan
    finally:
        _release_materialize_lock(lock_conn)


def build_and_materialize_veda_plan(
    *,
    algorithm: str = "effveda",
    indexing_threshold: int = 1000,
    storage_amplification: float = 1.2,
    ef_search: int = 100,
    document_limit: Optional[int] = None,
    create_indexes: bool = False,
    index_type: str = "hnsw",
    show_progress: bool = True,
    db_connection_factory=_default_db_connection_factory,
) -> VedaPlan:
    plan = build_veda_plan(
        algorithm=algorithm,
        indexing_threshold=int(indexing_threshold),
        storage_amplification=float(storage_amplification),
        ef_search=int(ef_search),
        document_limit=document_limit,
        show_progress=show_progress,
        db_connection_factory=db_connection_factory,
    )
    return materialize_plan(
        plan,
        create_indexes=create_indexes,
        index_type=index_type,
        show_progress=show_progress,
        db_connection_factory=db_connection_factory,
    )


def create_index_for_partition(
    table_name: str,
    *,
    node_kind: str = "index",
    index_type: str = "hnsw",
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
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (role_ids);").format(
                    sql.Identifier(_safe_index_name(table_name, "role_ids_idx")),
                    sql.Identifier(table_name),
                )
            )
            if str(node_kind) != "index":
                conn.commit()
                return
            normalized_index_type = index_type.lower()
            if normalized_index_type in {"hnsw", "vedahnsw"}:
                include_clause = sql.SQL(" INCLUDE (pattern_id)") if normalized_index_type == "vedahnsw" else sql.SQL("")
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


def _create_index_for_node_timed(table_name: str, node_kind: str, index_type: str) -> tuple[str, float]:
    started_at = time.time()
    create_index_for_partition(table_name, node_kind=node_kind, index_type=index_type)
    return table_name, time.time() - started_at


def create_indexes_for_materialized_partitions(
    index_type: str = "hnsw",
    *,
    algorithm: str | None = None,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    maintenance_settings = get_maintenance_settings()
    nodes = load_current_nodes(algorithm=algorithm, refresh=True, db_connection_factory=db_connection_factory)
    if not nodes:
        print("Veda index build: no materialized nodes found. Skipping.", flush=True)
        return
    print(
        "Veda index build: PostgreSQL parameters set: maintenance_work_mem = "
        f"{maintenance_settings['maintenance_work_mem_gb']}GB, "
        f"max_parallel_maintenance_workers = {maintenance_settings['max_parallel_maintenance_workers']}",
        flush=True,
    )
    node_tasks = [(node.table_name, node.node_kind) for node in sorted(nodes, key=lambda item: item.table_name)]
    worker_count = _recommended_worker_count(len(node_tasks), max_workers=max_workers)
    if parallel and worker_count > 1 and db_connection_factory is _default_db_connection_factory:
        print(f"Veda index build: creating auxiliary and ANN indexes with {worker_count} workers.", flush=True)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_create_index_for_node_timed, table_name, node_kind, index_type): table_name
                for table_name, node_kind in node_tasks
            }
            for index, future in enumerate(as_completed(futures), start=1):
                table_name, elapsed = future.result()
                print(f"Veda index build: [{index}/{len(node_tasks)}] finished {table_name} in {elapsed:.2f}s", flush=True)
        return
    for index, (table_name, node_kind) in enumerate(node_tasks, start=1):
        started_at = time.time()
        create_index_for_partition(table_name, node_kind=node_kind, index_type=index_type, db_connection_factory=db_connection_factory)
        elapsed = time.time() - started_at
        print(f"Veda index build: [{index}/{len(node_tasks)}] finished {table_name} in {elapsed:.2f}s", flush=True)


def drop_indexes_for_materialized_partitions(*, algorithm: str | None = None, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            table_names = list_current_plan_partition_tables(algorithm=algorithm, db_connection_factory=db_connection_factory)
            if not table_names:
                table_names = list_materialized_node_tables(algorithm=algorithm, db_connection_factory=db_connection_factory)
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
