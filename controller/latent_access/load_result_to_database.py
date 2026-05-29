"""Persistence and materialization helpers for latent access partitions."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import os
import numpy as np
import re
import time
from typing import Iterable, Optional

from psycopg2 import sql

from controller.dynamic_partition.load_result_to_database import _configure_index_session
from services.config import get_db_connection, get_document_vector_dimension, get_maintenance_settings

from .planner import LatentAccessPartition, LatentAccessPlan, LatentAccessPlanner
from .repository import LatentAccessRepository
from .trainer import LatentAccessTrainingConfig, PrototypeLatentAccessTrainer

LATENT_ACCESS_PARTITION_TABLE_PREFIX = "latent_documentblocks_partition_"

_CACHED_PARTITIONS: Optional[list[LatentAccessPartition]] = None
_CACHED_PLAN_SUMMARY: Optional[dict] = None
_CACHED_ATOM_TENANT_WEIGHTS: Optional[dict[int, dict[int, float]]] = None
_MAX_PARALLEL_WORKERS_CAP = 8


def _default_db_connection_factory():
    return get_db_connection()


def _sanitize_partition_id(partition_id: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(partition_id))
    return sanitized.strip("_") or "default"


def get_partition_table_name(partition_id: str) -> str:
    return f"{LATENT_ACCESS_PARTITION_TABLE_PREFIX}{_sanitize_partition_id(partition_id)}"


def _default_parallel_worker_count(max_workers: Optional[int] = None) -> int:
    if max_workers is not None:
        return max(1, int(max_workers))
    cpu_count = os.cpu_count() or 1
    return max(1, min(_MAX_PARALLEL_WORKERS_CAP, cpu_count // 2 or 1))


def invalidate_cached_plan_metadata() -> None:
    global _CACHED_PARTITIONS, _CACHED_PLAN_SUMMARY, _CACHED_ATOM_TENANT_WEIGHTS
    _CACHED_PARTITIONS = None
    _CACHED_PLAN_SUMMARY = None
    _CACHED_ATOM_TENANT_WEIGHTS = None


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).ravel()
    norm = float(np.linalg.norm(array))
    if norm <= 0:
        return array.astype(np.float32, copy=False)
    return (array / norm).astype(np.float32, copy=False)


def _summarize_partition_block_vectors(
    block_vectors: list[np.ndarray],
    *,
    anchor_count: int = 4,
) -> tuple[np.ndarray, np.ndarray]:
    if not block_vectors:
        return np.zeros(0, dtype=np.float32), np.zeros((0, 0), dtype=np.float32)

    normalized = np.vstack([_normalize_vector(vector) for vector in block_vectors]).astype(np.float32, copy=False)
    centroid = _normalize_vector(np.mean(normalized, axis=0))
    if normalized.shape[0] <= anchor_count:
        return centroid, normalized

    centroid_scores = normalized @ centroid
    chosen = [int(np.argmax(centroid_scores))]
    min_distances = 1.0 - (normalized @ normalized[chosen[0]])
    while len(chosen) < anchor_count:
        next_index = int(np.argmax(min_distances))
        if next_index in chosen:
            break
        chosen.append(next_index)
        min_distances = np.minimum(min_distances, 1.0 - (normalized @ normalized[next_index]))
    anchors = normalized[np.asarray(chosen, dtype=np.int32)]
    return centroid.astype(np.float32, copy=False), anchors.astype(np.float32, copy=False)


def _enrich_plan_with_block_semantics(
    plan: LatentAccessPlan,
    repository: LatentAccessRepository,
    *,
    anchor_count: int = 4,
) -> LatentAccessPlan:
    enriched_partitions: list[LatentAccessPartition] = []
    for partition in plan.partitions:
        block_vectors = repository.fetch_block_vectors_for_documents(partition.document_ids)
        metadata = dict(partition.metadata)
        if block_vectors:
            centroid, anchors = _summarize_partition_block_vectors(block_vectors, anchor_count=anchor_count)
            metadata['semantic_centroid'] = centroid.astype(float).tolist()
            metadata['semantic_anchor_vectors'] = anchors.astype(float).tolist()
            metadata['semantic_anchor_count'] = int(anchors.shape[0])
            metadata['semantic_source'] = 'block_vectors'
        enriched_partitions.append(
            LatentAccessPartition(
                partition_id=partition.partition_id,
                semantic_cell_id=partition.semantic_cell_id,
                latent_atom_id=partition.latent_atom_id,
                residual_flag=partition.residual_flag,
                document_ids=partition.document_ids,
                tenant_ids=partition.tenant_ids,
                document_count=partition.document_count,
                vector_count=partition.vector_count,
                metadata=metadata,
            )
        )

    enriched_metadata = dict(plan.metadata)
    enriched_metadata['semantic_metadata_mode'] = 'block_anchors'
    enriched_metadata['semantic_anchor_count'] = int(anchor_count)
    return LatentAccessPlan(
        partitions=enriched_partitions,
        semantic_centroids=plan.semantic_centroids,
        semantic_assignments=plan.semantic_assignments,
        model=plan.model,
        metadata=enriched_metadata,
    )


def initialize_plan_schema(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS latent_access_current_plan (
                    plan_id BIGSERIAL PRIMARY KEY,
                    atom_count INTEGER NOT NULL,
                    semantic_cell_count INTEGER NOT NULL,
                    residual_quantile DOUBLE PRECISION NOT NULL,
                    document_count BIGINT NOT NULL,
                    partition_count BIGINT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS latent_access_current_partitions (
                    partition_id TEXT PRIMARY KEY,
                    plan_id BIGINT NOT NULL REFERENCES latent_access_current_plan(plan_id) ON DELETE CASCADE,
                    table_name TEXT NOT NULL,
                    semantic_cell_id INTEGER NOT NULL,
                    latent_atom_id INTEGER,
                    residual_flag BOOLEAN NOT NULL DEFAULT FALSE,
                    document_count BIGINT NOT NULL,
                    vector_count BIGINT NOT NULL,
                    tenant_ids BIGINT[] NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{}'::jsonb
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS latent_access_current_partition_documents (
                    partition_id TEXT NOT NULL REFERENCES latent_access_current_partitions(partition_id) ON DELETE CASCADE,
                    document_id BIGINT NOT NULL,
                    PRIMARY KEY (partition_id, document_id)
                );
                """
            )
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS latent_access_current_atom_tenants (
                    plan_id BIGINT NOT NULL REFERENCES latent_access_current_plan(plan_id) ON DELETE CASCADE,
                    atom_id INTEGER NOT NULL,
                    tenant_id BIGINT NOT NULL,
                    weight DOUBLE PRECISION NOT NULL,
                    PRIMARY KEY (plan_id, atom_id, tenant_id)
                );
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_latent_access_partitions_cell_atom
                ON latent_access_current_partitions (semantic_cell_id, latent_atom_id, residual_flag);
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_latent_access_partition_documents_doc
                ON latent_access_current_partition_documents (document_id);
                """
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
            cur.execute("DELETE FROM latent_access_current_plan;")
        conn.commit()
    finally:
        conn.close()


def save_plan_result(
    plan: LatentAccessPlan,
    *,
    db_connection_factory=_default_db_connection_factory,
) -> int:
    invalidate_cached_plan_metadata()
    initialize_plan_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM latent_access_current_plan;")
            cur.execute(
                """
                INSERT INTO latent_access_current_plan (
                    atom_count,
                    semantic_cell_count,
                    residual_quantile,
                    document_count,
                    partition_count,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s::jsonb)
                RETURNING plan_id;
                """,
                [
                    int(plan.model.training_metadata.get("atom_count", plan.model.config.atom_count)),
                    int(plan.model.config.semantic_cell_count),
                    float(plan.model.config.residual_quantile),
                    int(plan.metadata.get("document_count", 0)),
                    int(plan.metadata.get("partition_count", 0)),
                    json.dumps(plan.metadata),
                ],
            )
            plan_id = int(cur.fetchone()[0])

            for partition in plan.partitions:
                table_name = get_partition_table_name(partition.partition_id)
                partition_metadata = dict(partition.metadata)
                partition_metadata["table_name"] = table_name
                cur.execute(
                    """
                    INSERT INTO latent_access_current_partitions (
                        partition_id,
                        plan_id,
                        table_name,
                        semantic_cell_id,
                        latent_atom_id,
                        residual_flag,
                        document_count,
                        vector_count,
                        tenant_ids,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb);
                    """,
                    [
                        partition.partition_id,
                        plan_id,
                        table_name,
                        int(partition.semantic_cell_id),
                        partition.latent_atom_id,
                        bool(partition.residual_flag),
                        int(partition.document_count),
                        int(partition.vector_count),
                        list(partition.tenant_ids),
                        json.dumps(partition_metadata),
                    ],
                )
                for document_id in partition.document_ids:
                    cur.execute(
                        """
                        INSERT INTO latent_access_current_partition_documents (partition_id, document_id)
                        VALUES (%s, %s)
                        ON CONFLICT (partition_id, document_id) DO NOTHING;
                        """,
                        [partition.partition_id, int(document_id)],
                    )

            atom_weights = plan.model.atom_tenant_weights
            for atom_id in range(atom_weights.shape[0]):
                for tenant_index, tenant_id in enumerate(plan.model.tenant_ids):
                    weight = float(atom_weights[atom_id, tenant_index])
                    if weight <= 0:
                        continue
                    cur.execute(
                        """
                        INSERT INTO latent_access_current_atom_tenants (plan_id, atom_id, tenant_id, weight)
                        VALUES (%s, %s, %s, %s)
                        ON CONFLICT (plan_id, atom_id, tenant_id)
                        DO UPDATE SET weight = EXCLUDED.weight;
                        """,
                        [plan_id, atom_id, int(tenant_id), weight],
                    )
        conn.commit()
        return plan_id
    finally:
        conn.close()


def load_current_partitions(
    *,
    db_connection_factory=_default_db_connection_factory,
    refresh: bool = False,
) -> list[LatentAccessPartition]:
    global _CACHED_PARTITIONS
    if not refresh and _CACHED_PARTITIONS is not None:
        return _CACHED_PARTITIONS

    initialize_plan_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    p.partition_id,
                    p.semantic_cell_id,
                    p.latent_atom_id,
                    p.residual_flag,
                    p.document_count,
                    p.vector_count,
                    p.tenant_ids,
                    p.metadata,
                    COALESCE(array_agg(d.document_id ORDER BY d.document_id)
                             FILTER (WHERE d.document_id IS NOT NULL), ARRAY[]::BIGINT[]) AS document_ids
                FROM latent_access_current_partitions p
                LEFT JOIN latent_access_current_partition_documents d
                  ON d.partition_id = p.partition_id
                GROUP BY p.partition_id, p.semantic_cell_id, p.latent_atom_id, p.residual_flag,
                         p.document_count, p.vector_count, p.tenant_ids, p.metadata
                ORDER BY p.partition_id;
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    partitions: list[LatentAccessPartition] = []
    for (
        partition_id,
        semantic_cell_id,
        latent_atom_id,
        residual_flag,
        document_count,
        vector_count,
        tenant_ids,
        metadata,
        document_ids,
    ) in rows:
        payload = metadata or {}
        payload.setdefault("table_name", get_partition_table_name(partition_id))
        partitions.append(
            LatentAccessPartition(
                partition_id=str(partition_id),
                semantic_cell_id=int(semantic_cell_id),
                latent_atom_id=None if latent_atom_id is None else int(latent_atom_id),
                residual_flag=bool(residual_flag),
                document_ids=tuple(int(document_id) for document_id in (document_ids or ())),
                tenant_ids=tuple(int(tenant_id) for tenant_id in (tenant_ids or ())),
                document_count=int(document_count),
                vector_count=int(vector_count),
                metadata=payload,
            )
        )
    _CACHED_PARTITIONS = partitions
    return partitions


def load_current_atom_tenant_weights(
    *,
    db_connection_factory=_default_db_connection_factory,
    refresh: bool = False,
) -> dict[int, dict[int, float]]:
    global _CACHED_ATOM_TENANT_WEIGHTS
    if not refresh and _CACHED_ATOM_TENANT_WEIGHTS is not None:
        return _CACHED_ATOM_TENANT_WEIGHTS

    initialize_plan_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT atom_id, tenant_id, weight
                FROM latent_access_current_atom_tenants
                WHERE plan_id = (
                    SELECT plan_id
                    FROM latent_access_current_plan
                    ORDER BY plan_id DESC
                    LIMIT 1
                );
                """
            )
            rows = cur.fetchall()
    finally:
        conn.close()

    weight_map: dict[int, dict[int, float]] = {}
    for atom_id, tenant_id, weight in rows:
        weight_map.setdefault(int(atom_id), {})[int(tenant_id)] = float(weight)
    _CACHED_ATOM_TENANT_WEIGHTS = weight_map
    return weight_map


def get_current_plan_summary(
    *,
    db_connection_factory=_default_db_connection_factory,
    refresh: bool = False,
) -> Optional[dict]:
    global _CACHED_PLAN_SUMMARY
    if not refresh and _CACHED_PLAN_SUMMARY is not None:
        return _CACHED_PLAN_SUMMARY

    initialize_plan_schema(db_connection_factory=db_connection_factory)
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT plan_id, atom_count, semantic_cell_count, residual_quantile,
                       document_count, partition_count, metadata, created_at
                FROM latent_access_current_plan
                ORDER BY plan_id DESC
                LIMIT 1;
                """
            )
            row = cur.fetchone()
    finally:
        conn.close()

    if row is None:
        _CACHED_PLAN_SUMMARY = None
        return None
    _CACHED_PLAN_SUMMARY = {
        "plan_id": int(row[0]),
        "atom_count": int(row[1]),
        "semantic_cell_count": int(row[2]),
        "residual_quantile": float(row[3]),
        "document_count": int(row[4]),
        "partition_count": int(row[5]),
        "metadata": row[6] or {},
        "created_at": row[7],
    }
    return _CACHED_PLAN_SUMMARY


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
                [f"{LATENT_ACCESS_PARTITION_TABLE_PREFIX}%"],
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
                [f"{LATENT_ACCESS_PARTITION_TABLE_PREFIX}%"],
            )
            existing_tables = [row[0] for row in cur.fetchall()]
            for table_name in existing_tables:
                if valid_table_names is not None and table_name in valid_table_names:
                    continue
                cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE;").format(sql.Identifier(table_name)))
        conn.commit()
    finally:
        conn.close()


def materialize_partition(
    partition: LatentAccessPartition,
    *,
    db_connection_factory=_default_db_connection_factory,
) -> str:
    table_name = get_partition_table_name(partition.partition_id)
    vector_dimension = get_document_vector_dimension()
    conn = db_connection_factory()
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
                    sql.Identifier(table_name),
                    dimension=sql.SQL(str(vector_dimension)),
                )
            )
            cur.execute(sql.SQL("TRUNCATE TABLE {};").format(sql.Identifier(table_name)))
            cur.execute(
                sql.SQL(
                    """
                    INSERT INTO {} (block_id, document_id, block_content, vector)
                    SELECT block_id, document_id, block_content, vector
                    FROM documentblocks
                    WHERE document_id = ANY(%s);
                    """
                ).format(sql.Identifier(table_name)),
                [list(partition.document_ids)],
            )
        conn.commit()
        return table_name
    finally:
        conn.close()


def materialize_planner_result(
    plan: LatentAccessPlan,
    *,
    create_indexes: bool = False,
    index_type: str = "hnsw",
    db_connection_factory=_default_db_connection_factory,
) -> LatentAccessPlan:
    save_plan_result(plan, db_connection_factory=db_connection_factory)
    drop_materialized_partitions(
        valid_partition_ids=[partition.partition_id for partition in plan.partitions],
        db_connection_factory=db_connection_factory,
    )
    for partition in plan.partitions:
        materialize_partition(partition, db_connection_factory=db_connection_factory)
    if create_indexes:
        create_indexes_for_materialized_partitions(index_type=index_type, db_connection_factory=db_connection_factory)
    return plan


def build_and_materialize_latent_access_plan(
    *,
    atom_count: int = 32,
    semantic_cell_count: int = 64,
    residual_quantile: float = 0.9,
    access_weight: float = 1.0,
    semantic_weight: float = 0.35,
    semantic_knn: int = 8,
    semantic_knn_weight: float = 0.2,
    max_atoms_per_semantic_cell: int = 4,
    min_partition_documents: int = 4,
    sparsity: int = 2,
    max_iterations: int = 25,
    z_inner_iterations: int = 4,
    momentum_weight: float = 0.1,
    min_atom_support: float = 1.0,
    revive_every: int = 3,
    revive_residual_quantile: float = 0.85,
    training_limit: Optional[int] = None,
    create_indexes: bool = False,
    index_type: str = "hnsw",
    db_connection_factory=_default_db_connection_factory,
) -> LatentAccessPlan:
    repository = LatentAccessRepository(db_connection_factory=db_connection_factory)
    training_records = repository.fetch_document_access_records(limit=training_limit)
    if not training_records:
        raise RuntimeError("No document access records found; cannot build latent access plan.")
    planner_records = (
        training_records
        if training_limit is None
        else repository.fetch_document_access_records(limit=None)
    )
    if not planner_records:
        raise RuntimeError("No planner records found; cannot build latent access plan.")
    block_counts = repository.fetch_document_block_counts(record.document_id for record in planner_records)
    trainer = PrototypeLatentAccessTrainer(
        LatentAccessTrainingConfig(
            atom_count=atom_count,
            semantic_cell_count=semantic_cell_count,
            residual_quantile=residual_quantile,
            access_weight=access_weight,
            semantic_weight=semantic_weight,
            semantic_knn=semantic_knn,
            semantic_knn_weight=semantic_knn_weight,
            max_atoms_per_semantic_cell=max_atoms_per_semantic_cell,
            min_partition_documents=min_partition_documents,
            sparsity=sparsity,
            max_iterations=max_iterations,
            z_inner_iterations=z_inner_iterations,
            momentum_weight=momentum_weight,
            min_atom_support=min_atom_support,
            revive_every=revive_every,
            revive_residual_quantile=revive_residual_quantile,
        )
    )
    trained_model = trainer.fit(training_records)
    model = trained_model if training_limit is None else trainer.infer(planner_records, trained_model)
    planner = LatentAccessPlanner()
    plan = planner.build_plan(planner_records, model, document_block_counts=block_counts)
    plan = _enrich_plan_with_block_semantics(plan, repository, anchor_count=4)
    return materialize_planner_result(
        plan,
        create_indexes=create_indexes,
        index_type=index_type,
        db_connection_factory=db_connection_factory,
    )


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
            _configure_index_session(
                cur,
                disable_sync_commit=disable_sync_commit,
                hnsw_threads=hnsw_threads,
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
        print("LatentAccess index build: no materialized partitions found. Skipping.", flush=True)
        return
    print(
        f"LatentAccess index build: creating {index_type} indexes for {len(table_names)} materialized partitions...",
        flush=True,
    )
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
                    f"LatentAccess index build: [{completed}/{len(table_names)}] finished {table_name} in {elapsed:.2f}s",
                    flush=True,
                )
        return

    for table_name in table_names:
        create_index_for_partition(
            table_name,
            index_type=index_type,
            hnsw_m=hnsw_m,
            hnsw_ef_construction=hnsw_ef_construction,
            hnsw_threads=hnsw_threads,
            disable_sync_commit=disable_sync_commit,
            db_connection_factory=db_connection_factory,
        )


def drop_indexes_for_materialized_partitions(*, db_connection_factory=_default_db_connection_factory) -> None:
    conn = db_connection_factory()
    try:
        with conn.cursor() as cur:
            for table_name in list_materialized_partition_tables(db_connection_factory=db_connection_factory):
                cur.execute(
                    sql.SQL("DROP INDEX IF EXISTS {} CASCADE;").format(
                        sql.Identifier(f"{table_name}_vector_idx")
                    )
                )
        conn.commit()
    finally:
        conn.close()
