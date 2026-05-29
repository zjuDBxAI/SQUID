"""Search helpers for latent access partitions."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from psycopg2 import sql

from services.config import get_db_connection

try:
    import efconfig
except Exception:  # pragma: no cover - benchmark-only optional module
    efconfig = None

from .load_result_to_database import (
    get_current_plan_summary,
    get_partition_table_name,
    load_current_atom_tenant_weights,
    load_current_partitions,
)

_CACHED_PARTITION_VERSION: Optional[int] = None
_CACHED_PARTITIONS = None
_CACHED_ATOM_WEIGHTS = None


@dataclass(slots=True)
class LatentPartitionRoute:
    tenant_id: int
    partition_ids: tuple[str, ...]
    partition_count: int
    metadata: dict = field(default_factory=dict)


def _parse_vector(raw_vector) -> np.ndarray:
    if isinstance(raw_vector, np.ndarray):
        vector = raw_vector
    elif isinstance(raw_vector, str):
        payload = raw_vector.strip().strip("[]")
        vector = np.asarray([float(item) for item in payload.split(",") if item], dtype=np.float32)
    elif hasattr(raw_vector, "tolist"):
        vector = np.asarray(raw_vector.tolist(), dtype=np.float32)
    else:
        vector = np.asarray(raw_vector, dtype=np.float32)
    if vector.ndim != 1:
        vector = vector.ravel()
    vector = vector.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(vector))
    if norm > 0:
        vector = vector / norm
    return vector


def _load_cached_plan_state(*, refresh: bool = False):
    global _CACHED_PARTITION_VERSION, _CACHED_PARTITIONS, _CACHED_ATOM_WEIGHTS
    plan_summary = get_current_plan_summary(refresh=refresh)
    plan_id = int(plan_summary["plan_id"]) if plan_summary is not None else 0
    if refresh or _CACHED_PARTITIONS is None or _CACHED_PARTITION_VERSION != plan_id:
        _CACHED_PARTITIONS = load_current_partitions(refresh=refresh)
        _CACHED_ATOM_WEIGHTS = load_current_atom_tenant_weights(refresh=refresh)
        _CACHED_PARTITION_VERSION = plan_id
    return _CACHED_PARTITIONS or [], _CACHED_ATOM_WEIGHTS or {}, plan_summary


def _configured_fixed_ef_search() -> Optional[int]:
    configured = getattr(efconfig, "ef_search", None) if efconfig is not None else None
    if configured is None:
        return None
    if isinstance(configured, str):
        normalized = configured.strip().lower()
        if normalized in {"", "adaptive", "auto", "none"}:
            return None
        return max(1, int(float(normalized)))
    return max(1, int(configured))


def _configured_int(name: str, default: int, *, minimum: int = 1) -> int:
    configured = getattr(efconfig, name, None) if efconfig is not None else None
    if configured is None:
        return max(minimum, int(default))
    return max(minimum, int(configured))


def _configured_float(name: str, default: float) -> float:
    configured = getattr(efconfig, name, None) if efconfig is not None else None
    if configured is None:
        return float(default)
    return float(configured)


def _partition_centroid(partition) -> np.ndarray:
    return _parse_vector(partition.metadata.get("semantic_centroid", []))


def _partition_anchor_vectors(partition) -> np.ndarray:
    raw_anchors = partition.metadata.get("semantic_anchor_vectors", [])
    if not raw_anchors:
        centroid = _partition_centroid(partition)
        if centroid.size == 0:
            return np.zeros((0, 0), dtype=np.float32)
        return centroid.reshape(1, -1).astype(np.float32, copy=False)
    anchors = np.asarray(raw_anchors, dtype=np.float32)
    if anchors.ndim == 1:
        anchors = anchors.reshape(1, -1)
    return anchors.astype(np.float32, copy=False)


def _semantic_partition_score(partition, query_vector: np.ndarray) -> float:
    anchors = _partition_anchor_vectors(partition)
    if anchors.size == 0:
        return -1e9
    return float(np.max(anchors @ query_vector))


def _semantic_cell_score(query_vector: np.ndarray, partitions) -> float:
    if not partitions:
        return -1e9
    return max(_semantic_partition_score(partition, query_vector) for partition in partitions)


def _score_partition(partition, tenant_id: int, query_vector: np.ndarray, atom_weights: dict[int, dict[int, float]]) -> float:
    semantic_weight = _configured_float("latent_route_semantic_weight", 1.0)
    access_weight = _configured_float("latent_route_access_weight", 0.2)
    route_prior_weight = _configured_float("latent_route_prior_weight", 0.05)
    residual_penalty_weight = _configured_float("latent_route_residual_penalty", 0.02)

    semantic_score = _semantic_partition_score(partition, query_vector)
    if semantic_score <= -1e8:
        semantic_score = 0.0
    if partition.residual_flag or partition.latent_atom_id is None:
        access_score = 0.0
    else:
        access_score = float(atom_weights.get(int(partition.latent_atom_id), {}).get(int(tenant_id), 0.0))
    route_prior = float(partition.metadata.get("route_prior", 0.0) or 0.0)
    residual_penalty = residual_penalty_weight if partition.residual_flag else 0.0
    return semantic_weight * semantic_score + access_weight * access_score + route_prior_weight * route_prior - residual_penalty


def _group_partitions_by_cell(partitions) -> dict[int, list]:
    grouped: dict[int, list] = {}
    for partition in partitions:
        grouped.setdefault(int(partition.semantic_cell_id), []).append(partition)
    return grouped


def _cell_partition_queue(cell_partitions, tenant_id: int, query_vector: np.ndarray, atom_weights: dict[int, dict[int, float]]):
    residuals = [partition for partition in cell_partitions if partition.residual_flag]
    atoms = [partition for partition in cell_partitions if not partition.residual_flag]
    atoms.sort(
        key=lambda partition: _score_partition(partition, tenant_id, query_vector, atom_weights),
        reverse=True,
    )
    queue = []
    if residuals:
        queue.append(residuals[0])
    queue.extend(atoms)
    return queue


def get_tenant_partition_route(
    tenant_id: int,
    query_vector,
    *,
    route_limit: Optional[int] = None,
) -> LatentPartitionRoute:
    partitions, atom_weights, _ = _load_cached_plan_state()
    if not partitions:
        return LatentPartitionRoute(tenant_id=int(tenant_id), partition_ids=(), partition_count=0, metadata={"found": False})

    effective_tenant_id = int(tenant_id)
    query_array = _parse_vector(query_vector)
    explicit_route_limit = route_limit is not None
    effective_route_limit = route_limit
    if effective_route_limit is None:
        effective_route_limit = _configured_int("latent_route_limit", 16)
    else:
        effective_route_limit = max(1, int(effective_route_limit))
    base_route_limit = int(effective_route_limit)

    tenant_partitions = [partition for partition in partitions if effective_tenant_id in partition.tenant_ids]
    if not tenant_partitions:
        return LatentPartitionRoute(tenant_id=effective_tenant_id, partition_ids=(), partition_count=0, metadata={"found": False})

    partitions_by_cell = _group_partitions_by_cell(tenant_partitions)
    cell_scores = sorted(
        (
            (cell_id, _semantic_cell_score(query_array, cell_partitions))
            for cell_id, cell_partitions in partitions_by_cell.items()
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    adaptive_score_margin = None
    adaptive_expanded = False
    if (not explicit_route_limit) and len(cell_scores) > 1:
        adaptive_route_limit_max = min(
            len(tenant_partitions),
            _configured_int("latent_route_limit_max", max(base_route_limit, 24)),
        )
        comparison_index = min(len(cell_scores) - 1, max(1, base_route_limit) - 1)
        adaptive_score_margin = float(cell_scores[0][1] - cell_scores[comparison_index][1])
        expansion_margin = _configured_float("latent_route_expansion_margin", 0.15)
        if adaptive_score_margin < expansion_margin and adaptive_route_limit_max > effective_route_limit:
            effective_route_limit = int(adaptive_route_limit_max)
            adaptive_expanded = True

    default_cell_limit = max(1, (effective_route_limit * 3 + 3) // 4)
    semantic_cell_limit = min(
        len(partitions_by_cell),
        _configured_int("latent_semantic_cell_limit", default_cell_limit),
    )
    ordered_cells = [cell_id for cell_id, _ in cell_scores[:semantic_cell_limit]]

    cell_queues = {
        cell_id: _cell_partition_queue(partitions_by_cell[cell_id], effective_tenant_id, query_array, atom_weights)
        for cell_id in ordered_cells
    }

    selected = []
    selected_ids = set()
    made_progress = True
    while len(selected) < effective_route_limit and made_progress:
        made_progress = False
        for cell_id in ordered_cells:
            queue = cell_queues[cell_id]
            while queue and queue[0].partition_id in selected_ids:
                queue.pop(0)
            if not queue:
                continue
            partition = queue.pop(0)
            selected.append(partition)
            selected_ids.add(partition.partition_id)
            made_progress = True
            if len(selected) >= effective_route_limit:
                break

    if not selected:
        selected = sorted(
            tenant_partitions,
            key=lambda partition: _score_partition(partition, effective_tenant_id, query_array, atom_weights),
            reverse=True,
        )[:effective_route_limit]

    return LatentPartitionRoute(
        tenant_id=effective_tenant_id,
        partition_ids=tuple(partition.partition_id for partition in selected),
        partition_count=len(selected),
        metadata={
            "found": bool(selected),
            "route_limit": int(effective_route_limit),
            "base_route_limit": int(base_route_limit),
            "adaptive_route_limit_applied": bool(adaptive_expanded),
            "adaptive_score_margin": adaptive_score_margin,
            "tenant_partition_count": len(tenant_partitions),
            "semantic_cell_limit": int(semantic_cell_limit),
            "selected_semantic_cells": [int(cell_id) for cell_id in ordered_cells],
            "table_names": [partition.metadata.get("table_name", get_partition_table_name(partition.partition_id)) for partition in selected],
        },
    )


def latent_access_search(user_id: int, query_vector, topk: int = 5, statistics_type: str = "sql"):
    if statistics_type == "sql":
        return latent_access_search_statistics_sql(user_id, query_vector, topk)
    if statistics_type == "system":
        return latent_access_search_statistics_system(user_id, query_vector, topk)
    raise ValueError(f"Unknown statistics type: {statistics_type}")


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
        if len(unique_results) == topk:
            break
    return unique_results


def _build_partition_union_query(
    route: LatentPartitionRoute,
    *,
    user_id: int,
    query_vector,
    per_partition_limit: int,
    topk: int,
):
    branch_queries = []
    params = []
    for partition_id in route.partition_ids:
        table_name = get_partition_table_name(partition_id)
        branch_queries.append(
            sql.SQL(
                """
                SELECT p.block_id, p.document_id, p.block_content,
                       p.vector <-> %s::vector AS distance
                FROM {} p
                WHERE EXISTS (
                    SELECT 1
                    FROM PermissionAssignment pa
                    JOIN UserRoles ur ON ur.role_id = pa.role_id
                    WHERE ur.user_id = %s
                      AND pa.document_id = p.document_id
                )
                ORDER BY distance
                LIMIT %s
                """
            ).format(sql.Identifier(table_name))
        )
        params.extend([query_vector, int(user_id), int(per_partition_limit)])

    union_query = sql.SQL(" UNION ALL ").join(
        sql.SQL("({})").format(branch_query) for branch_query in branch_queries
    )
    query = sql.SQL(
        """
        SELECT block_id, document_id, block_content, distance
        FROM ({}) AS routed_partitions
        ORDER BY distance
        LIMIT %s;
        """
    ).format(union_query)
    params.append(int(topk))
    return query, params


def latent_access_search_statistics_sql(user_id: int, query_vector, topk: int = 5):
    route = get_tenant_partition_route(user_id, query_vector)
    if not route.partition_ids:
        return [], 0.0

    fixed_ef = _configured_fixed_ef_search()
    fetch_multiplier = _configured_int("latent_partition_fetch_multiplier", 3)
    per_partition_limit = max(int(topk), int(topk) * fetch_multiplier)
    query, params = _build_partition_union_query(
        route,
        user_id=int(user_id),
        query_vector=query_vector,
        per_partition_limit=int(per_partition_limit),
        topk=int(topk),
    )
    conn = get_db_connection()
    total_query_time = 0.0
    all_results = []
    try:
        with conn.cursor() as cur:
            cur.execute("SET max_parallel_workers_per_gather = 0;")
            cur.execute("SET jit = off;")
            if fixed_ef is not None:
                cur.execute(f"SET hnsw.ef_search = {fixed_ef};")

            explain_query = sql.SQL("EXPLAIN ANALYZE {}" ).format(query)
            cur.execute(explain_query, params)
            for (line,) in cur.fetchall():
                if "Execution Time" in line:
                    total_query_time += float(line.split()[-2]) / 1000.0

            cur.execute(query, params)
            all_results.extend(cur.fetchall())
    finally:
        conn.close()

    return _merge_results(all_results, topk), total_query_time


def latent_access_search_statistics_system(user_id: int, query_vector, topk: int = 5):
    route = get_tenant_partition_route(user_id, query_vector)
    if not route.partition_ids:
        return [], 0.0

    fetch_multiplier = _configured_int("latent_partition_fetch_multiplier", 3)
    per_partition_limit = max(int(topk), int(topk) * fetch_multiplier)
    query, params = _build_partition_union_query(
        route,
        user_id=int(user_id),
        query_vector=query_vector,
        per_partition_limit=int(per_partition_limit),
        topk=int(topk),
    )
    conn = get_db_connection()
    started_at = time.time()
    all_results = []
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            all_results.extend(cur.fetchall())
    finally:
        conn.close()

    return _merge_results(all_results, topk), time.time() - started_at
