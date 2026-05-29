from __future__ import annotations

from dataclasses import dataclass, field
import importlib
import math
import sys
import time
from typing import Optional

import numpy as np
from psycopg2 import sql

from services.config import get_db_connection

from .common import WorkloadAwarePartition, _normalize_vector, _parse_vector, get_partition_table_name
from .storage import load_current_partitions



def _resolve_efconfig_module():
    for module_name in ("basic_benchmark.efconfig", "efconfig"):
        module = sys.modules.get(module_name)
        if module is not None:
            return module
    for module_name in ("basic_benchmark.efconfig", "efconfig"):
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    return None


def _configured_fixed_ef_search() -> Optional[int]:
    efconfig = _resolve_efconfig_module()
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
    efconfig = _resolve_efconfig_module()
    configured = getattr(efconfig, name, None) if efconfig is not None else None
    if configured is None:
        return max(minimum, int(default))
    return max(minimum, int(configured))


def _configured_float(name: str, default: float) -> float:
    efconfig = _resolve_efconfig_module()
    configured = getattr(efconfig, name, None) if efconfig is not None else None
    if configured is None:
        return float(default)
    return float(configured)


@dataclass(slots=True)
class WorkloadAwareRoute:
    tenant_id: int
    partition_ids: tuple[str, ...]
    partition_count: int
    metadata: dict[str, object] = field(default_factory=dict)


def _tenant_density(partition: WorkloadAwarePartition, tenant_id: int) -> float:
    densities = partition.metadata.get("tenant_densities", {}) or {}
    return float(densities.get(str(int(tenant_id)), 0.0) or 0.0)


def _tenant_query_mass(partition: WorkloadAwarePartition, tenant_id: int) -> float:
    weights = partition.metadata.get("tenant_query_mass", {}) or {}
    return float(weights.get(str(int(tenant_id)), 0.0) or 0.0)


def _score_partition(partition: WorkloadAwarePartition, tenant_id: int, query_vector: np.ndarray) -> float:
    density_weight = _configured_float("dynamic_partition_route_density_weight", 1.0)
    workload_weight = _configured_float("dynamic_partition_route_workload_weight", 0.2)
    prior_weight = _configured_float("dynamic_partition_route_prior_weight", 0.05)
    semantic_weight = _configured_float("dynamic_partition_route_semantic_weight", 0.0)
    pattern_penalty = _configured_float("dynamic_partition_route_pattern_penalty", 0.01)

    density_score = _tenant_density(partition, tenant_id)
    workload_score = math.log1p(_tenant_query_mass(partition, tenant_id))
    route_prior = float(partition.metadata.get("route_prior", 0.0) or 0.0)
    logical_pattern_count = int(partition.metadata.get("logical_pattern_count", len(partition.logical_pattern_ids)) or 0)

    centroid = _parse_vector(partition.metadata.get("representative_centroid", []))
    semantic_score = float(np.dot(_normalize_vector(centroid), query_vector)) if centroid.size and query_vector.size else 0.0
    return (
        density_weight * density_score
        + workload_weight * workload_score
        + prior_weight * route_prior
        + semantic_weight * semantic_score
        - pattern_penalty * max(0, logical_pattern_count - 1)
    )


def get_tenant_partition_route(
    tenant_id: int,
    query_vector,
    *,
    route_limit: Optional[int] = None,
) -> WorkloadAwareRoute:
    partitions = load_current_partitions()
    if not partitions:
        return WorkloadAwareRoute(tenant_id=int(tenant_id), partition_ids=(), partition_count=0, metadata={"found": False})

    effective_route_limit = route_limit if route_limit is not None else _configured_int("dynamic_partition_route_limit", 64)
    effective_route_limit = max(1, int(effective_route_limit))
    query_array = _normalize_vector(_parse_vector(query_vector))

    candidate_partitions = [
        partition
        for partition in partitions
        if int(tenant_id) in set(int(value) for value in (partition.metadata.get("entry_tenant_ids", []) or []))
    ]
    fallback_used = False
    if not candidate_partitions:
        fallback_used = True
        candidate_partitions = [partition for partition in partitions if int(tenant_id) in partition.tenant_ids]
    if not candidate_partitions:
        return WorkloadAwareRoute(tenant_id=int(tenant_id), partition_ids=(), partition_count=0, metadata={"found": False})

    ranked = sorted(candidate_partitions, key=lambda partition: _score_partition(partition, int(tenant_id), query_array), reverse=True)
    selected = ranked[:effective_route_limit]
    return WorkloadAwareRoute(
        tenant_id=int(tenant_id),
        partition_ids=tuple(partition.partition_id for partition in selected),
        partition_count=len(selected),
        metadata={
            "found": bool(selected),
            "route_limit": int(effective_route_limit),
            "tenant_partition_count": len(candidate_partitions),
            "table_names": [partition.table_name for partition in selected],
            "fallback_used": bool(fallback_used),
        },
    )


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
    route: WorkloadAwareRoute,
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

    union_query = sql.SQL(" UNION ALL ").join(sql.SQL("({})").format(branch_query) for branch_query in branch_queries)
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


def dynamic_partition_search(user_id: int, query_vector, topk: int = 5, statistics_type: str = "sql"):
    if statistics_type == "sql":
        return dynamic_partition_search_statistics_sql(user_id, query_vector, topk)
    if statistics_type == "system":
        return dynamic_partition_search_statistics_system(user_id, query_vector, topk)
    raise ValueError(f"Unknown statistics type: {statistics_type}")


def dynamic_partition_search_statistics_sql(user_id: int, query_vector, topk: int = 5):
    route = get_tenant_partition_route(user_id, query_vector)
    if not route.partition_ids:
        return [], 0.0

    fetch_multiplier = _configured_int("dynamic_partition_partition_fetch_multiplier", 4)
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
            fixed_ef = _configured_fixed_ef_search()
            if fixed_ef is not None:
                cur.execute(f"SET hnsw.ef_search = {fixed_ef};")

            explain_query = sql.SQL("EXPLAIN ANALYZE {}").format(query)
            cur.execute(explain_query, params)
            for (line,) in cur.fetchall():
                if "Execution Time" in line:
                    total_query_time += float(line.split()[-2]) / 1000.0

            cur.execute(query, params)
            all_results.extend(cur.fetchall())
    finally:
        conn.close()

    return _merge_results(all_results, topk), total_query_time


def dynamic_partition_search_statistics_system(user_id: int, query_vector, topk: int = 5):
    route = get_tenant_partition_route(user_id, query_vector)
    if not route.partition_ids:
        return [], 0.0

    fetch_multiplier = _configured_int("dynamic_partition_partition_fetch_multiplier", 4)
    per_partition_limit = max(int(topk), int(topk) * fetch_multiplier)
    query, params = _build_partition_union_query(
        route,
        user_id=int(user_id),
        query_vector=query_vector,
        per_partition_limit=int(per_partition_limit),
        topk=int(topk),
    )
    started_at = time.time()
    conn = get_db_connection()
    all_results = []
    try:
        with conn.cursor() as cur:
            cur.execute(query, params)
            all_results.extend(cur.fetchall())
    finally:
        conn.close()
    return _merge_results(all_results, topk), time.time() - started_at
