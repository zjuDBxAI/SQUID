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

from services.config import get_db_connection, get_document_vector_dimension, get_maintenance_settings


ACL_PLAN_TABLE = "acl_partition_current_plan"
ACL_PATTERN_TABLE = "acl_partition_current_patterns"
ACL_ROUTE_TABLE = "acl_partition_current_routes"
ACL_PARTITION_PREFIX = "acl_documentblocks_partition_"

_DEFAULT_INDEX_MAX_WORKERS = 6
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


def _recommended_worker_count(task_count: int, *, max_workers: Optional[int] = None) -> int:
    if task_count <= 1:
        return 1
    if max_workers is not None:
        return max(1, min(int(max_workers), int(task_count)))
    cpu_count = max(1, int(os.cpu_count() or 1))
    return max(1, min(int(task_count), _DEFAULT_INDEX_MAX_WORKERS, max(1, cpu_count // 4)))


def _configure_index_session(cur, *, disable_sync_commit: bool = True, hnsw_threads: Optional[int] = None) -> None:
    maintenance_settings = get_maintenance_settings()
    cur.execute(f"SET maintenance_work_mem = '{int(maintenance_settings['maintenance_work_mem_gb'])}GB';")
    cur.execute(f"SET max_parallel_maintenance_workers = {int(maintenance_settings['max_parallel_maintenance_workers'])};")
    if disable_sync_commit:
        cur.execute("SET synchronous_commit = OFF;")
    if hnsw_threads:
        cur.execute(f"SET hnsw.threads = {int(hnsw_threads)};")


def initialize_schema(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        plan_id BIGSERIAL PRIMARY KEY,
                        pattern_count INTEGER NOT NULL,
                        document_count BIGINT NOT NULL,
                        original_vector_count BIGINT NOT NULL,
                        materialized_vector_count BIGINT NOT NULL,
                        metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                ).format(sql.Identifier(ACL_PLAN_TABLE))
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        plan_id BIGINT NOT NULL REFERENCES {}(plan_id) ON DELETE CASCADE,
                        pattern_id BIGINT NOT NULL,
                        role_ids BIGINT[] NOT NULL,
                        document_count BIGINT NOT NULL,
                        vector_count BIGINT NOT NULL,
                        table_name TEXT NOT NULL,
                        PRIMARY KEY (plan_id, pattern_id)
                    );
                    """
                ).format(sql.Identifier(ACL_PATTERN_TABLE), sql.Identifier(ACL_PLAN_TABLE))
            )
            cur.execute(
                sql.SQL(
                    """
                    CREATE TABLE IF NOT EXISTS {} (
                        plan_id BIGINT NOT NULL REFERENCES {}(plan_id) ON DELETE CASCADE,
                        user_id BIGINT NOT NULL,
                        pattern_id BIGINT NOT NULL,
                        table_name TEXT NOT NULL,
                        vector_count BIGINT NOT NULL,
                        PRIMARY KEY (plan_id, user_id, pattern_id)
                    );
                    """
                ).format(sql.Identifier(ACL_ROUTE_TABLE), sql.Identifier(ACL_PLAN_TABLE))
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (role_ids);").format(
                    sql.Identifier(f"idx_{ACL_PATTERN_TABLE}_role_ids"),
                    sql.Identifier(ACL_PATTERN_TABLE),
                )
            )
            cur.execute(
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (user_id);").format(
                    sql.Identifier(f"idx_{ACL_ROUTE_TABLE}_user_id"),
                    sql.Identifier(ACL_ROUTE_TABLE),
                )
            )
        conn.commit()
    finally:
        conn.close()


def clear_current_plan(*, db_connection_factory=_default_db_connection_factory) -> None:
    initialize_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {};").format(sql.Identifier(ACL_PLAN_TABLE)))
        conn.commit()
    finally:
        conn.close()


def _fetch_acl_patterns(cur, *, document_limit: Optional[int] = None) -> list[dict[str, object]]:
    limit_clause = sql.SQL("")
    params: list[object] = []
    if document_limit is not None:
        limit_clause = sql.SQL(
            """
            JOIN (
                SELECT DISTINCT document_id
                FROM documentblocks
                ORDER BY document_id
                LIMIT %s
            ) limited_docs ON limited_docs.document_id = pa.document_id
            """
        )
        params.append(int(document_limit))

    cur.execute(
        sql.SQL(
            """
            WITH doc_acl AS (
                SELECT
                    pa.document_id::bigint AS document_id,
                    array_agg(DISTINCT pa.role_id::bigint ORDER BY pa.role_id::bigint) AS role_ids
                FROM PermissionAssignment pa
                {limit_clause}
                GROUP BY pa.document_id
            ),
            pattern_docs AS (
                SELECT
                    role_ids,
                    COUNT(*)::bigint AS document_count
                FROM doc_acl
                GROUP BY role_ids
            ),
            pattern_vectors AS (
                SELECT
                    pd.role_ids,
                    pd.document_count,
                    COUNT(db.block_id)::bigint AS vector_count
                FROM pattern_docs pd
                JOIN doc_acl da ON da.role_ids = pd.role_ids
                JOIN documentblocks db ON db.document_id = da.document_id
                GROUP BY pd.role_ids, pd.document_count
            )
            SELECT
                ROW_NUMBER() OVER (ORDER BY role_ids)::bigint AS pattern_id,
                role_ids,
                document_count,
                vector_count
            FROM pattern_vectors
            ORDER BY pattern_id;
            """
        ).format(limit_clause=limit_clause),
        params,
    )
    patterns = []
    for pattern_id, role_ids, document_count, vector_count in cur.fetchall():
        patterns.append(
            {
                "pattern_id": int(pattern_id),
                "role_ids": [int(role_id) for role_id in role_ids],
                "document_count": int(document_count),
                "vector_count": int(vector_count),
                "table_name": f"{ACL_PARTITION_PREFIX}{int(pattern_id)}",
            }
        )
    return patterns


def _save_plan(
    patterns: list[dict[str, object]],
    *,
    metadata: Optional[dict[str, object]] = None,
    db_connection_factory=_default_db_connection_factory,
) -> int:
    initialize_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DELETE FROM {};").format(sql.Identifier(ACL_PLAN_TABLE)))
            original_vector_count = sum(int(pattern["vector_count"]) for pattern in patterns)
            document_count = sum(int(pattern["document_count"]) for pattern in patterns)
            plan_metadata = dict(metadata or {})
            plan_metadata.setdefault("partitioning", "one_distinct_acl_role_set_per_partition")
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (
                        pattern_count, document_count, original_vector_count,
                        materialized_vector_count, metadata
                    )
                    VALUES (%s, %s, %s, %s, %s::jsonb)
                    RETURNING plan_id;
                    """
                ).format(sql.Identifier(ACL_PLAN_TABLE)),
                [
                    int(len(patterns)),
                    int(document_count),
                    int(original_vector_count),
                    int(original_vector_count),
                    json.dumps(plan_metadata),
                ],
            )
            plan_id = int(cur.fetchone()[0])
            pattern_rows = [
                (
                    plan_id,
                    int(pattern["pattern_id"]),
                    list(pattern["role_ids"]),
                    int(pattern["document_count"]),
                    int(pattern["vector_count"]),
                    str(pattern["table_name"]),
                )
                for pattern in patterns
            ]
            if pattern_rows:
                execute_values(
                    cur,
                    sql.SQL(
                        """
                        INSERT INTO {} (
                            plan_id, pattern_id, role_ids, document_count,
                            vector_count, table_name
                        )
                        VALUES %s;
                        """
                    ).format(sql.Identifier(ACL_PATTERN_TABLE)).as_string(conn),
                    pattern_rows,
                    page_size=1000,
                )
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (plan_id, user_id, pattern_id, table_name, vector_count)
                    SELECT
                        p.plan_id,
                        ur.user_id,
                        p.pattern_id,
                        p.table_name,
                        p.vector_count
                    FROM (
                        SELECT user_id::bigint AS user_id,
                               array_agg(role_id::bigint ORDER BY role_id::bigint) AS role_ids
                        FROM UserRoles
                        GROUP BY user_id
                    ) ur
                    JOIN {} p
                      ON p.plan_id = %s
                     AND p.role_ids && ur.role_ids;
                    """
                ).format(sql.Identifier(ACL_ROUTE_TABLE), sql.Identifier(ACL_PATTERN_TABLE)),
                [plan_id],
            )
        conn.commit()
        return plan_id
    finally:
        conn.close()


def list_current_plan_partition_tables(*, db_connection_factory=_default_db_connection_factory) -> list[str]:
    initialize_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                sql.SQL(
                    """
                    SELECT table_name
                    FROM {}
                    WHERE plan_id = (SELECT MAX(plan_id) FROM {})
                    ORDER BY pattern_id;
                    """
                ).format(sql.Identifier(ACL_PATTERN_TABLE), sql.Identifier(ACL_PLAN_TABLE))
            )
            return [str(row[0]) for row in cur.fetchall()]
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
                [f"{ACL_PARTITION_PREFIX}%"],
            )
            return [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()


def _drop_stale_materialized_partitions(
    valid_table_names: set[str],
    *,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    existing = set(list_materialized_partition_tables(db_connection_factory=db_connection_factory))
    stale = sorted(existing - set(valid_table_names))
    if not stale:
        return
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            for table_name in stale:
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name)))
        conn.commit()
    finally:
        conn.close()


def build_acl_partition_plan(
    *,
    document_limit: Optional[int] = None,
    db_connection_factory=_default_db_connection_factory,
) -> list[dict[str, object]]:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            return _fetch_acl_patterns(cur, document_limit=document_limit)
    finally:
        conn.close()


def materialize_plan(
    patterns: list[dict[str, object]],
    *,
    create_indexes: bool = False,
    index_type: str = "hnsw",
    show_progress: bool = True,
    db_connection_factory=_default_db_connection_factory,
) -> list[dict[str, object]]:
    plan_id = _save_plan(patterns, metadata={"index_type": str(index_type)}, db_connection_factory=db_connection_factory)
    valid_table_names = {str(pattern["table_name"]) for pattern in patterns}
    _drop_stale_materialized_partitions(valid_table_names, db_connection_factory=db_connection_factory)

    vector_dimension = get_document_vector_dimension()
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TEMP TABLE temp_acl_document_patterns AS
                SELECT
                    pa.document_id::bigint AS document_id,
                    array_agg(DISTINCT pa.role_id::bigint ORDER BY pa.role_id::bigint) AS role_ids
                FROM PermissionAssignment pa
                GROUP BY pa.document_id;
                """
            )
            cur.execute("CREATE INDEX temp_acl_document_patterns_role_ids_idx ON temp_acl_document_patterns (role_ids);")
            iterator = tqdm(
                patterns,
                desc="ACL partition materialize",
                unit="partition",
                disable=not show_progress,
            )
            for pattern in iterator:
                table_name = str(pattern["table_name"])
                role_ids = list(pattern["role_ids"])
                started_at = time.time()
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name)))
                cur.execute(
                    sql.SQL(
                        """
                        CREATE TABLE {} (
                            block_id BIGINT NOT NULL,
                            document_id BIGINT NOT NULL,
                            pattern_id BIGINT NOT NULL,
                            role_ids BIGINT[] NOT NULL,
                            block_content BYTEA NOT NULL,
                            hash_value BYTEA,
                            vector VECTOR({dimension}),
                            PRIMARY KEY (block_id, document_id)
                        );
                        """
                    ).format(sql.Identifier(table_name), dimension=sql.SQL(str(int(vector_dimension))))
                )
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (
                            block_id, document_id, pattern_id, role_ids,
                            block_content, hash_value, vector
                        )
                        SELECT
                            db.block_id, db.document_id, %s, %s::bigint[],
                            db.block_content, db.hash_value, db.vector
                        FROM documentblocks db
                        JOIN temp_acl_document_patterns acl
                          ON acl.document_id = db.document_id
                        WHERE acl.role_ids = %s::bigint[];
                        """
                    ).format(sql.Identifier(table_name)),
                    [int(pattern["pattern_id"]), role_ids, role_ids],
                )
                elapsed = time.time() - started_at
                print(f"[acl_partition] materialized {table_name} in {elapsed:.2f}s", flush=True)
        conn.commit()
    finally:
        conn.close()

    if create_indexes:
        create_indexes_for_materialized_partitions(index_type=index_type, db_connection_factory=db_connection_factory)
    print(
        f"[acl_partition] built {len(patterns)} partitions; plan_id={plan_id}; "
        f"memory_replication_factor=1.0000",
        flush=True,
    )
    return patterns


def build_and_materialize_acl_partition_plan(
    *,
    document_limit: Optional[int] = None,
    create_indexes: bool = False,
    index_type: str = "hnsw",
    show_progress: bool = True,
    db_connection_factory=_default_db_connection_factory,
) -> list[dict[str, object]]:
    patterns = build_acl_partition_plan(document_limit=document_limit, db_connection_factory=db_connection_factory)
    return materialize_plan(
        patterns,
        create_indexes=create_indexes,
        index_type=index_type,
        show_progress=show_progress,
        db_connection_factory=db_connection_factory,
    )


def create_index_for_partition(
    table_name: str,
    *,
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
                sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (document_id);").format(
                    sql.Identifier(_safe_index_name(table_name, "document_id_idx")),
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
    index_type: str = "hnsw",
    *,
    parallel: bool = True,
    max_workers: Optional[int] = None,
    db_connection_factory=_default_db_connection_factory,
) -> None:
    table_names = list_current_plan_partition_tables(db_connection_factory=db_connection_factory)
    if not table_names:
        table_names = list_materialized_partition_tables(db_connection_factory=db_connection_factory)
    if not table_names:
        print("ACL partition index build: no materialized partitions found. Skipping.", flush=True)
        return

    worker_count = _recommended_worker_count(len(table_names), max_workers=max_workers)
    if parallel and worker_count > 1 and db_connection_factory is _default_db_connection_factory:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = {
                executor.submit(_create_index_for_partition_timed, table_name, index_type): table_name
                for table_name in table_names
            }
            for index, future in enumerate(as_completed(futures), start=1):
                table_name, elapsed = future.result()
                print(f"ACL partition index build: [{index}/{len(table_names)}] finished {table_name} in {elapsed:.2f}s", flush=True)
        return

    for index, table_name in enumerate(table_names, start=1):
        started_at = time.time()
        create_index_for_partition(table_name, index_type=index_type, db_connection_factory=db_connection_factory)
        elapsed = time.time() - started_at
        print(f"ACL partition index build: [{index}/{len(table_names)}] finished {table_name} in {elapsed:.2f}s", flush=True)


def drop_indexes_for_materialized_partitions(*, db_connection_factory=_default_db_connection_factory) -> None:
    table_names = list_current_plan_partition_tables(db_connection_factory=db_connection_factory)
    if not table_names:
        table_names = list_materialized_partition_tables(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
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
