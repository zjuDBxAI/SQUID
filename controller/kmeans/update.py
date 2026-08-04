from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import hashlib
import heapq
import json
import math
import time
from typing import Iterable, Optional

from psycopg2 import sql
from psycopg2.extras import execute_values

from services.config import get_db_connection

from .common import (
    ACLPattern,
    KMeansPartition,
    KMeansPlan,
    TenantRoute,
    PARTITION_TABLE,
    PATTERN_TABLE,
    PLAN_TABLE,
    ROUTE_TABLE,
    UPDATE_ACL_ROLE_TABLE,
    UPDATE_BATCH_TABLE,
    UPDATE_TOMBSTONE_TABLE,
    get_partition_table_name,
)
from .cost_model import estimate_partition_query_cost
from .storage import (
    create_indexes_for_partitions,
    drop_vector_indexes_below_threshold,
    drop_stale_materialized_partitions,
    get_current_plan_summary,
    initialize_schema,
    invalidate_cache,
    load_current_partitions,
    materialize_partition,
    save_plan,
)


@dataclass(slots=True)
class KMeansUpdateItem:
    operation: str
    document_id: int
    tenant_ids: tuple[int, ...] = ()
    role_ids: tuple[int, ...] = ()
    vectors: tuple[list[float], ...] = ()
    block_ids: tuple[int, ...] = ()
    source_document_id: Optional[int] = None
    document_name: Optional[str] = None
    metadata: dict[str, object] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, payload: dict) -> "KMeansUpdateItem":
        operation = str(payload.get("operation") or payload.get("op") or payload.get("type") or "upsert").lower()
        document_id = int(payload["document_id"])
        tenant_ids = tuple(sorted({int(value) for value in payload.get("tenant_ids", payload.get("acl", [])) or []}))
        role_ids = tuple(sorted({int(value) for value in payload.get("role_ids", []) or []}))
        raw_vectors = payload.get("vectors")
        if raw_vectors is None and payload.get("vector") is not None:
            raw_vectors = [payload.get("vector")]
        vectors = tuple([float(x) for x in vector] for vector in (raw_vectors or []))
        block_ids = tuple(int(value) for value in (payload.get("block_ids", []) or []))
        source_document_id = payload.get("source_document_id")
        return cls(
            operation=operation,
            document_id=document_id,
            tenant_ids=tenant_ids,
            role_ids=role_ids,
            vectors=vectors,
            block_ids=block_ids,
            source_document_id=None if source_document_id is None else int(source_document_id),
            document_name=None if payload.get("document_name") is None else str(payload.get("document_name")),
            metadata={k: v for k, v in payload.items() if k not in {"operation", "op", "type"}},
        )


@dataclass(slots=True)
class KMeansMaintenanceCandidate:
    op_type: str
    partition_ids: tuple[str, ...]
    delta_memory: int
    delta_latency: float
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class KMeansUpdateResult:
    batch_id: int
    applied_count: int
    affected_partition_ids: tuple[str, ...]
    rewritten_partition_ids: tuple[str, ...]
    accepted_operations: tuple[KMeansMaintenanceCandidate, ...]
    metadata: dict[str, object]


def _default_db_connection_factory():
    return get_db_connection()


def _acl_key(tenant_ids: Iterable[int]) -> str:
    return ",".join(str(int(value)) for value in sorted({int(value) for value in tenant_ids}))


def _pattern_score(vector_count: int, acl_tenant_count: int) -> float:
    return float(math.log1p(max(0, int(vector_count)) * max(0, int(acl_tenant_count) - 1)))


def _pattern_weight(vector_count: int, acl_tenant_count: int, total_tenant_count: int) -> float:
    acl_tenant_count = max(1, int(acl_tenant_count))
    return float(math.log1p(max(0, int(vector_count))) * math.log1p(float(max(1, int(total_tenant_count))) / float(acl_tenant_count)))


def _partition_table_id(partition_id: str) -> str:
    return str(partition_id)


def _next_private_partition_id(partitions: Iterable[KMeansPartition]) -> str:
    used = {str(partition.partition_id) for partition in partitions}
    index = 0
    while f"private_{index}" in used:
        index += 1
    return f"private_{index}"


def _partition_live_vectors(pattern_ids: Iterable[int], patterns_by_id: dict[int, ACLPattern]) -> int:
    return int(sum(int(patterns_by_id[int(pattern_id)].vector_count) for pattern_id in set(map(int, pattern_ids)) if int(pattern_id) in patterns_by_id))


def _tenant_patterns_for_partition(partition: KMeansPartition, patterns_by_id: dict[int, ACLPattern]) -> dict[int, set[int]]:
    explicit = partition.metadata.get("tenant_patterns", {}) or {}
    if explicit:
        return {
            int(tenant_id): {int(pattern_id) for pattern_id in pattern_values if int(pattern_id) in patterns_by_id}
            for tenant_id, pattern_values in dict(explicit).items()
        }
    result: dict[int, set[int]] = defaultdict(set)
    for pattern_id in partition.pattern_ids:
        pattern = patterns_by_id.get(int(pattern_id))
        if pattern is None:
            continue
        for tenant_id in pattern.tenant_ids:
            result[int(tenant_id)].add(int(pattern_id))
    return dict(result)


class _PatternBitsetContext:
    def __init__(self, patterns_by_id: dict[int, ACLPattern]) -> None:
        self.patterns_by_id = patterns_by_id
        ordered_pattern_ids = tuple(
            sorted(
                int(pattern_id)
                for pattern_id, pattern in patterns_by_id.items()
                if int(pattern.vector_count) > 0
            )
        )
        self.pattern_bit_index = {int(pattern_id): index for index, pattern_id in enumerate(ordered_pattern_ids)}
        self.bit_index_pattern_ids = {index: int(pattern_id) for pattern_id, index in self.pattern_bit_index.items()}
        self.bit_index_weights = tuple(int(patterns_by_id[int(pattern_id)].vector_count) for pattern_id in ordered_pattern_ids)
        self.single_pattern_bits = {int(pattern_id): int(1 << index) for pattern_id, index in self.pattern_bit_index.items()}
        self.bit_weight_cache: dict[int, int] = {0: 0}

    def pattern_bits_for(self, pattern_ids: Iterable[int]) -> int:
        bits = 0
        for pattern_id in pattern_ids:
            bits |= int(self.single_pattern_bits.get(int(pattern_id), 0))
        return int(bits)

    def pattern_ids_from_bits(self, pattern_bits: int) -> set[int]:
        bits = int(pattern_bits)
        result: set[int] = set()
        while bits:
            lowest_bit = bits & -bits
            index = int(lowest_bit.bit_length() - 1)
            pattern_id = self.bit_index_pattern_ids.get(int(index))
            if pattern_id is not None:
                result.add(int(pattern_id))
            bits ^= lowest_bit
        return result

    def vector_count_bits(self, pattern_bits: int) -> int:
        bits = int(pattern_bits)
        cached = self.bit_weight_cache.get(bits)
        if cached is not None:
            return int(cached)
        original_bits = bits
        total = 0
        while bits:
            lowest_bit = bits & -bits
            index = int(lowest_bit.bit_length() - 1)
            if 0 <= index < len(self.bit_index_weights):
                total += int(self.bit_index_weights[int(index)])
            bits ^= lowest_bit
        self.bit_weight_cache[int(original_bits)] = int(total)
        return int(total)

    def normalize_tenant_bits(self, tenant_bits: dict[int, int]) -> dict[int, int]:
        return {int(tenant_id): int(bits) for tenant_id, bits in tenant_bits.items() if int(bits) != 0}

    def tenant_bits_for_partition(self, partition: KMeansPartition) -> dict[int, int]:
        tenant_patterns = _tenant_patterns_for_partition(partition, self.patterns_by_id)
        partition_bits = self.pattern_bits_for(partition.pattern_ids)
        result = {
            int(tenant_id): int(self.pattern_bits_for(pattern_ids)) & int(partition_bits)
            for tenant_id, pattern_ids in tenant_patterns.items()
        }
        return self.normalize_tenant_bits(result)

    def tenant_patterns_from_bits(self, tenant_bits: dict[int, int]) -> dict[int, set[int]]:
        result: dict[int, set[int]] = {}
        for tenant_id, bits in self.normalize_tenant_bits(tenant_bits).items():
            pattern_ids = self.pattern_ids_from_bits(int(bits))
            if pattern_ids:
                result[int(tenant_id)] = pattern_ids
        return result

    def union_tenant_bits(self, *maps: dict[int, int]) -> dict[int, int]:
        result: dict[int, int] = {}
        for mapping in maps:
            for tenant_id, bits in mapping.items():
                bits = int(bits)
                if bits:
                    result[int(tenant_id)] = int(result.get(int(tenant_id), 0)) | int(bits)
        return self.normalize_tenant_bits(result)

    def remove_bits_from_tenants(self, tenant_bits: dict[int, int], removed_bits: int) -> dict[int, int]:
        removed_bits = int(removed_bits)
        return self.normalize_tenant_bits({
            int(tenant_id): int(bits) & ~removed_bits
            for tenant_id, bits in tenant_bits.items()
        })

    def mask_tenant_bits(self, tenant_bits: dict[int, int], mask_bits: int) -> dict[int, int]:
        mask_bits = int(mask_bits)
        if mask_bits == 0:
            return {}
        return self.normalize_tenant_bits({
            int(tenant_id): int(bits) & int(mask_bits)
            for tenant_id, bits in tenant_bits.items()
        })


def _item_has_acl_payload(item: KMeansUpdateItem) -> bool:
    if item.role_ids or item.tenant_ids:
        return True
    if str(item.operation).lower() in {"acl_update", "update_acl"}:
        return True
    return any(key in item.metadata for key in ("tenant_ids", "acl", "role_ids"))


def _make_partition(
    partition_id: str,
    *,
    cluster_id: int,
    pattern_ids: Iterable[int],
    tenant_patterns: dict[int, set[int]],
    patterns_by_id: dict[int, ACLPattern],
) -> Optional[KMeansPartition]:
    normalized_pattern_ids = tuple(sorted({int(pattern_id) for pattern_id in pattern_ids if int(pattern_id) in patterns_by_id and int(patterns_by_id[int(pattern_id)].vector_count) > 0}))
    normalized_tenant_patterns = {
        int(tenant_id): {int(pattern_id) for pattern_id in pattern_values if int(pattern_id) in normalized_pattern_ids}
        for tenant_id, pattern_values in tenant_patterns.items()
    }
    normalized_tenant_patterns = {tenant_id: values for tenant_id, values in normalized_tenant_patterns.items() if values}
    if not normalized_pattern_ids or not normalized_tenant_patterns:
        return None
    document_pattern_pairs = tuple(
        (int(document_id), int(pattern_id))
        for pattern_id in normalized_pattern_ids
        for document_id in patterns_by_id[int(pattern_id)].document_ids
    )
    document_ids = tuple(sorted({int(document_id) for document_id, _ in document_pattern_pairs}))
    tenant_ids = tuple(sorted(normalized_tenant_patterns))
    vector_count = _partition_live_vectors(normalized_pattern_ids, patterns_by_id)
    return KMeansPartition(
        partition_id=str(partition_id),
        cluster_id=int(cluster_id),
        partition_kind="private",
        table_name=get_partition_table_name(str(partition_id)),
        tenant_ids=tenant_ids,
        pattern_ids=normalized_pattern_ids,
        document_ids=document_ids,
        document_pattern_pairs=document_pattern_pairs,
        vector_count=int(vector_count),
        metadata={
            "partition_kind": "private",
            "pattern_count": int(len(normalized_pattern_ids)),
            "pattern_tenants": {
                str(int(pattern_id)): [int(tenant_id) for tenant_id in patterns_by_id[int(pattern_id)].tenant_ids]
                for pattern_id in normalized_pattern_ids
            },
            "tenant_patterns": {
                str(int(tenant_id)): sorted(int(pattern_id) for pattern_id in pattern_values)
                for tenant_id, pattern_values in normalized_tenant_patterns.items()
            },
        },
    )


def _build_routes(partitions: Iterable[KMeansPartition]) -> list[TenantRoute]:
    routes: list[TenantRoute] = []
    for partition in partitions:
        tenant_patterns = partition.metadata.get("tenant_patterns", {}) or {}
        for tenant_id_text, pattern_values in dict(tenant_patterns).items():
            pattern_ids = tuple(sorted({int(pattern_id) for pattern_id in pattern_values}))
            if not pattern_ids:
                continue
            routes.append(
                TenantRoute(
                    tenant_id=int(tenant_id_text),
                    partition_id=str(partition.partition_id),
                    table_name=str(partition.table_name),
                    route_kind="private",
                    cluster_id=int(partition.cluster_id),
                    pattern_ids=pattern_ids,
                )
            )
    return routes


def _tenant_requirements(patterns_by_id: dict[int, ACLPattern]) -> dict[int, set[int]]:
    requirements: dict[int, set[int]] = defaultdict(set)
    for pattern in patterns_by_id.values():
        if int(pattern.vector_count) <= 0:
            continue
        for tenant_id in pattern.tenant_ids:
            requirements[int(tenant_id)].add(int(pattern.pattern_id))
    return dict(requirements)


def _routes_from_partition_metadata(
    partitions: Iterable[KMeansPartition],
    patterns_by_id: dict[int, ACLPattern],
) -> dict[int, dict[str, set[int]]]:
    routes: dict[int, dict[str, set[int]]] = defaultdict(dict)
    partition_ids = {str(partition.partition_id) for partition in partitions}
    requirements = _tenant_requirements(patterns_by_id)
    for partition in partitions:
        partition_patterns = set(map(int, partition.pattern_ids))
        tenant_patterns = _tenant_patterns_for_partition(partition, patterns_by_id)
        for tenant_id, pattern_ids in tenant_patterns.items():
            allowed = requirements.get(int(tenant_id), set())
            kept = set(map(int, pattern_ids)) & partition_patterns & allowed
            if kept and str(partition.partition_id) in partition_ids:
                routes[int(tenant_id)][str(partition.partition_id)] = kept
    return {int(tenant_id): dict(by_partition) for tenant_id, by_partition in routes.items()}


def _routes_from_existing_routes(
    routes: Iterable[TenantRoute],
    partitions: Iterable[KMeansPartition],
    patterns_by_id: dict[int, ACLPattern],
) -> dict[int, dict[str, set[int]]]:
    partitions_by_id = {str(partition.partition_id): partition for partition in partitions}
    requirements = _tenant_requirements(patterns_by_id)
    result: dict[int, dict[str, set[int]]] = defaultdict(dict)
    for route in routes:
        partition = partitions_by_id.get(str(route.partition_id))
        if partition is None:
            continue
        allowed = requirements.get(int(route.tenant_id), set())
        kept = set(map(int, route.pattern_ids)) & set(map(int, partition.pattern_ids)) & allowed
        if kept:
            result[int(route.tenant_id)][str(route.partition_id)] = kept
    return {int(tenant_id): dict(by_partition) for tenant_id, by_partition in result.items()}


def _tenants_touching_partitions(
    partitions: Iterable[KMeansPartition],
    partition_ids: Iterable[str],
    patterns_by_id: dict[int, ACLPattern],
) -> set[int]:
    target = {str(value) for value in partition_ids}
    tenants: set[int] = set()
    for partition in partitions:
        if str(partition.partition_id) not in target:
            continue
        tenants.update(int(tenant_id) for tenant_id in _tenant_patterns_for_partition(partition, patterns_by_id))
    return tenants


def _repair_routes_greedy(
    partitions: list[KMeansPartition],
    patterns_by_id: dict[int, ACLPattern],
    *,
    affected_tenant_ids: set[int],
    existing_routes: Iterable[TenantRoute] = (),
) -> tuple[list[KMeansPartition], list[TenantRoute], dict[str, int]]:
    requirements = _tenant_requirements(patterns_by_id)
    partitions_by_id = {str(partition.partition_id): partition for partition in partitions}
    existing_route_list = list(existing_routes)
    existing_route_map = (
        _routes_from_existing_routes(existing_route_list, partitions, patterns_by_id)
        if existing_route_list
        else {}
    )
    metadata_route_map = _routes_from_partition_metadata(partitions, patterns_by_id)
    route_map: dict[int, dict[str, set[int]]] = {}
    repaired_count = 0
    fallback_repaired_count = 0
    affected_preserved_count = 0
    affected_metadata_rerouted_count = 0

    def route_cover(by_partition: dict[str, set[int]]) -> set[int]:
        covered: set[int] = set()
        for pattern_ids in by_partition.values():
            covered.update(int(pattern_id) for pattern_id in pattern_ids)
        return covered

    def greedy_cover(tenant_id: int, *, existing: dict[str, set[int]] | None = None) -> dict[str, set[int]]:
        required = set(requirements.get(int(tenant_id), set()))
        repaired: dict[str, set[int]] = {
            str(partition_id): set(pattern_ids) & required
            for partition_id, pattern_ids in (existing or {}).items()
            if str(partition_id) in partitions_by_id and set(pattern_ids) & required
        }
        uncovered = set(required) - route_cover(repaired)
        while uncovered:
            best_partition = None
            best_route_patterns: set[int] = set()
            best_rank = None
            for partition in partitions:
                partition_patterns = set(map(int, partition.pattern_ids))
                covered_now = partition_patterns & uncovered
                if not covered_now:
                    continue
                route_patterns = partition_patterns & required
                accessible_vectors = _partition_live_vectors(route_patterns, patterns_by_id)
                selectivity = float(accessible_vectors) / float(max(1, int(partition.vector_count)))
                rank = (
                    int(len(covered_now)),
                    float(selectivity),
                    -int(partition.vector_count),
                    -int(len(route_patterns)),
                    str(partition.partition_id),
                )
                if best_rank is None or rank > best_rank:
                    best_rank = rank
                    best_partition = partition
                    best_route_patterns = set(route_patterns)
            if best_partition is None:
                raise RuntimeError(f"Unable to repair kmeans route for tenant {tenant_id}: missing patterns {sorted(uncovered)}")
            repaired[str(best_partition.partition_id)] = set(best_route_patterns)
            uncovered -= set(best_route_patterns)
        return repaired

    for tenant_id, required in sorted(requirements.items()):
        existing = {
            str(partition_id): set(pattern_ids) & set(required)
            for partition_id, pattern_ids in existing_route_map.get(int(tenant_id), {}).items()
            if str(partition_id) in partitions_by_id and set(pattern_ids) & set(required)
        }
        metadata_route = {
            str(partition_id): set(pattern_ids) & set(required)
            for partition_id, pattern_ids in metadata_route_map.get(int(tenant_id), {}).items()
            if str(partition_id) in partitions_by_id and set(pattern_ids) & set(required)
        }
        existing_cover = route_cover(existing)
        route_complete = existing_cover == set(required)
        metadata_cover = route_cover(metadata_route)
        metadata_complete = metadata_cover == set(required)
        affected = int(tenant_id) in affected_tenant_ids
        if affected:
            if metadata_complete:
                if metadata_route == existing:
                    affected_preserved_count += 1
                else:
                    affected_metadata_rerouted_count += 1
                route_map[int(tenant_id)] = metadata_route
                continue
            route_map[int(tenant_id)] = greedy_cover(int(tenant_id), existing=metadata_route)
            repaired_count += 1
            continue
        if route_complete:
            route_map[int(tenant_id)] = existing
            continue
        if metadata_complete:
            route_map[int(tenant_id)] = metadata_route
            continue
        fallback_repaired_count += 1
        route_map[int(tenant_id)] = greedy_cover(int(tenant_id), existing=existing or metadata_route)
        repaired_count += 1

    by_partition: dict[str, dict[int, set[int]]] = defaultdict(dict)
    for tenant_id, partition_patterns in route_map.items():
        if int(tenant_id) not in requirements:
            continue
        for partition_id, pattern_ids in partition_patterns.items():
            kept = set(pattern_ids) & set(requirements[int(tenant_id)])
            if kept and str(partition_id) in partitions_by_id:
                by_partition[str(partition_id)][int(tenant_id)] = kept

    repaired_partitions: list[KMeansPartition] = []
    routes: list[TenantRoute] = []
    for partition in partitions:
        tenant_patterns = by_partition.get(str(partition.partition_id), {})
        metadata = dict(partition.metadata or {})
        metadata["tenant_patterns"] = {
            str(int(tenant_id)): sorted(int(pattern_id) for pattern_id in pattern_ids)
            for tenant_id, pattern_ids in sorted(tenant_patterns.items())
            if pattern_ids
        }
        metadata["pattern_tenants"] = {
            str(int(pattern_id)): [int(tenant_id) for tenant_id in patterns_by_id[int(pattern_id)].tenant_ids]
            for pattern_id in partition.pattern_ids
            if int(pattern_id) in patterns_by_id
        }
        tenant_ids = tuple(sorted(int(tenant_id) for tenant_id, pattern_ids in tenant_patterns.items() if pattern_ids))
        repaired_partition = KMeansPartition(
            partition_id=str(partition.partition_id),
            cluster_id=int(partition.cluster_id),
            partition_kind=str(partition.partition_kind),
            table_name=str(partition.table_name),
            tenant_ids=tenant_ids,
            pattern_ids=partition.pattern_ids,
            document_ids=partition.document_ids,
            document_pattern_pairs=partition.document_pattern_pairs,
            vector_count=int(partition.vector_count),
            metadata=metadata,
        )
        repaired_partitions.append(repaired_partition)
        for tenant_id, pattern_ids in sorted(tenant_patterns.items()):
            if not pattern_ids:
                continue
            routes.append(
                TenantRoute(
                    tenant_id=int(tenant_id),
                    partition_id=str(partition.partition_id),
                    table_name=str(partition.table_name),
                    route_kind="private",
                    cluster_id=int(partition.cluster_id),
                    pattern_ids=tuple(sorted(int(pattern_id) for pattern_id in pattern_ids)),
                )
            )
    return repaired_partitions, routes, {
        "route_repair_tenant_count": int(repaired_count),
        "route_repair_fallback_tenant_count": int(fallback_repaired_count),
        "route_repair_affected_tenant_count": int(len(affected_tenant_ids)),
        "route_repair_affected_preserved_count": int(affected_preserved_count),
        "route_repair_affected_metadata_rerouted_count": int(affected_metadata_rerouted_count),
        "route_repair_route_count": int(len(routes)),
    }


class KMeansUpdateRepository:
    def __init__(self, *, db_connection_factory=_default_db_connection_factory) -> None:
        self.db_connection_factory = db_connection_factory

    def create_batch(self, items: list[KMeansUpdateItem]) -> int:
        initialize_schema(db_connection_factory=self.db_connection_factory)
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("INSERT INTO {} (update_count, metadata) VALUES (%s, %s::jsonb) RETURNING batch_id;").format(
                        sql.Identifier(UPDATE_BATCH_TABLE)
                    ),
                    [len(items), json.dumps({"source": "controller.kmeans.update"})],
                )
                batch_id = int(cur.fetchone()[0])
            conn.commit()
            return int(batch_id)
        finally:
            conn.close()

    def current_plan_id(self) -> int:
        summary = get_current_plan_summary(refresh=True, db_connection_factory=self.db_connection_factory)
        if summary is None:
            raise RuntimeError("No kmeans plan found. Initialize kmeans partitions before applying updates.")
        return int(summary["plan_id"])

    def fetch_current_patterns(self) -> list[ACLPattern]:
        plan_id = self.current_plan_id()
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT pattern_id, tenant_ids, document_ids, document_count, vector_count, weight, metadata
                        FROM {}
                        WHERE plan_id = %s
                        ORDER BY pattern_id;
                        """
                    ).format(sql.Identifier(PATTERN_TABLE)),
                    [int(plan_id)],
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [
            ACLPattern(
                pattern_id=int(row[0]),
                tenant_ids=tuple(int(value) for value in (row[1] or ())),
                document_ids=tuple(int(value) for value in (row[2] or ())),
                document_count=int(row[3]),
                vector_count=int(row[4]),
                weight=float(row[5]),
                score=float((row[6] or {}).get("score", 0.0)),
                zone=str((row[6] or {}).get("zone", "private")),
            )
            for row in rows
        ]

    def fetch_current_routes(self) -> list[TenantRoute]:
        plan_id = self.current_plan_id()
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT tenant_id, partition_id, table_name, route_kind, cluster_id, pattern_ids
                        FROM {}
                        WHERE plan_id = %s
                        ORDER BY tenant_id, partition_id;
                        """
                    ).format(sql.Identifier(ROUTE_TABLE)),
                    [int(plan_id)],
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return [
            TenantRoute(
                tenant_id=int(row[0]),
                partition_id=str(row[1]),
                table_name=str(row[2]),
                route_kind=str(row[3]),
                cluster_id=int(row[4]),
                pattern_ids=tuple(int(value) for value in (row[5] or ())),
            )
            for row in rows
        ]

    def fetch_document_tenant_ids(self, document_ids: Iterable[int]) -> dict[int, tuple[int, ...]]:
        normalized = sorted({int(document_id) for document_id in document_ids})
        if not normalized:
            return {}
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT pa.document_id, array_agg(DISTINCT ur.user_id ORDER BY ur.user_id)
                    FROM PermissionAssignment pa
                    JOIN UserRoles ur ON ur.role_id = pa.role_id
                    WHERE pa.document_id = ANY(%s)
                    GROUP BY pa.document_id;
                    """,
                    [normalized],
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return {int(document_id): tuple(int(value) for value in (tenant_ids or ())) for document_id, tenant_ids in rows}

    def fetch_document_blocks(self, document_ids: Iterable[int]) -> dict[int, list[tuple[int, bytes, object]]]:
        normalized = sorted({int(document_id) for document_id in document_ids})
        if not normalized:
            return {}
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT document_id, block_id, block_content, vector
                    FROM documentblocks
                    WHERE document_id = ANY(%s)
                    ORDER BY document_id, block_id;
                    """,
                    [normalized],
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        by_doc: dict[int, list[tuple[int, bytes, object]]] = defaultdict(list)
        for document_id, block_id, block_content, vector in rows:
            by_doc[int(document_id)].append((int(block_id), block_content, vector))
        return dict(by_doc)

    def fetch_all_acl_groups(self) -> dict[tuple[int, ...], tuple[tuple[int, ...], int, int]]:
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH document_tenants AS (
                        SELECT pa.document_id, array_agg(DISTINCT ur.user_id ORDER BY ur.user_id) AS tenant_ids
                        FROM PermissionAssignment pa
                        JOIN UserRoles ur ON ur.role_id = pa.role_id
                        GROUP BY pa.document_id
                    ),
                    document_block_counts AS (
                        SELECT document_id, COUNT(*)::BIGINT AS vector_count
                        FROM documentblocks
                        GROUP BY document_id
                    )
                    SELECT dt.tenant_ids, array_agg(dt.document_id ORDER BY dt.document_id), COUNT(*)::BIGINT,
                           COALESCE(SUM(dbc.vector_count), 0)::BIGINT
                    FROM document_tenants dt
                    JOIN document_block_counts dbc ON dbc.document_id = dt.document_id
                    GROUP BY dt.tenant_ids;
                    """
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        result: dict[tuple[int, ...], tuple[tuple[int, ...], int, int]] = {}
        for tenant_ids, document_ids, document_count, vector_count in rows:
            key = tuple(sorted(int(value) for value in (tenant_ids or ())))
            if key:
                result[key] = (tuple(int(value) for value in (document_ids or ())), int(document_count), int(vector_count))
        return result

    def record_tombstones_for_documents(self, partition_ids: Iterable[str], document_ids: Iterable[int], *, batch_id: int) -> int:
        normalized_partitions = sorted({str(value) for value in partition_ids})
        normalized_docs = sorted({int(value) for value in document_ids})
        if not normalized_partitions or not normalized_docs:
            return 0
        rows: list[tuple[str, int, int, int, int]] = []
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                for partition_id in normalized_partitions:
                    table_name = get_partition_table_name(str(partition_id))
                    cur.execute(
                        sql.SQL(
                            """
                            SELECT %s, block_id, document_id, pattern_id, %s
                            FROM {}
                            WHERE document_id = ANY(%s);
                            """
                        ).format(sql.Identifier(table_name)),
                        [str(partition_id), int(batch_id), normalized_docs],
                    )
                    rows.extend((str(row[0]), int(row[1]), int(row[2]), int(row[3]), int(row[4])) for row in cur.fetchall())
                if rows:
                    execute_values(
                        cur,
                        f"""
                        INSERT INTO {UPDATE_TOMBSTONE_TABLE} (partition_id, block_id, document_id, pattern_id, batch_id)
                        VALUES %s
                        ON CONFLICT (partition_id, block_id, document_id)
                        DO UPDATE SET pattern_id = EXCLUDED.pattern_id, batch_id = EXCLUDED.batch_id, created_at = NOW();
                        """,
                        rows,
                    )
            conn.commit()
            return int(len(rows))
        finally:
            conn.close()

    def tombstone_counts(self) -> dict[str, int]:
        initialize_schema(db_connection_factory=self.db_connection_factory)
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("SELECT partition_id, COUNT(*) FROM {} GROUP BY partition_id;").format(
                        sql.Identifier(UPDATE_TOMBSTONE_TABLE)
                    )
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return {str(partition_id): int(count) for partition_id, count in rows}

    def clear_tombstones(self, partition_ids: Iterable[str]) -> None:
        normalized = sorted({str(value) for value in partition_ids})
        if not normalized:
            return
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    sql.SQL("DELETE FROM {} WHERE partition_id = ANY(%s);").format(
                        sql.Identifier(UPDATE_TOMBSTONE_TABLE)
                    ),
                    [normalized],
                )
            conn.commit()
        finally:
            conn.close()

    def ensure_role_for_tenants(self, tenant_ids: Iterable[int]) -> int:
        tenants = tuple(sorted({int(value) for value in tenant_ids}))
        if not tenants:
            raise ValueError("tenant_ids cannot be empty when creating an ACL role")
        key = _acl_key(tenants)
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("SELECT role_id FROM {} WHERE acl_key = %s;").format(sql.Identifier(UPDATE_ACL_ROLE_TABLE)), [key])
                row = cur.fetchone()
                if row is not None:
                    return int(row[0])
                cur.execute("SELECT COALESCE(MAX(role_id), 0) + 1 FROM Roles;")
                role_id = int(cur.fetchone()[0])
                digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).hexdigest()
                cur.execute("INSERT INTO Roles (role_id, role_name) VALUES (%s, %s) ON CONFLICT (role_id) DO NOTHING;", [role_id, f"kmeans_update_acl_{digest}"])
                execute_values(
                    cur,
                    "INSERT INTO UserRoles (user_id, role_id) VALUES %s ON CONFLICT DO NOTHING;",
                    [(int(tenant_id), int(role_id)) for tenant_id in tenants],
                )
                cur.execute(
                    sql.SQL("INSERT INTO {} (acl_key, tenant_ids, role_id) VALUES (%s, %s, %s);").format(sql.Identifier(UPDATE_ACL_ROLE_TABLE)),
                    [key, list(tenants), int(role_id)],
                )
            conn.commit()
            return int(role_id)
        finally:
            conn.close()

    def replace_document_acl(self, document_id: int, *, role_ids: Iterable[int] = (), tenant_ids: Iterable[int] = ()) -> None:
        roles = tuple(sorted({int(value) for value in role_ids}))
        if not roles:
            tenants = tuple(sorted({int(value) for value in tenant_ids}))
            if tenants:
                roles = (self.ensure_role_for_tenants(tenants),)
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM PermissionAssignment WHERE document_id = %s;", [int(document_id)])
                if roles:
                    execute_values(
                        cur,
                        "INSERT INTO PermissionAssignment (role_id, document_id) VALUES %s;",
                        [(int(role_id), int(document_id)) for role_id in roles],
                    )
            conn.commit()
        finally:
            conn.close()

    def grant_document_roles(self, document_id: int, role_ids: Iterable[int]) -> None:
        """Add direct role permissions without changing users or role membership.

        ``tenant_ids`` in SQUID are *effective users*, not RBAC roles.  A
        permission-workload grant must consequently operate on the explicit
        ``role_id`` values in ``PermissionAssignment`` and must not call
        :meth:`ensure_role_for_tenants`.
        """
        roles = tuple(sorted({int(value) for value in role_ids}))
        if not roles:
            raise ValueError("acl_grant requires at least one role_id")
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO PermissionAssignment (role_id, document_id)
                    SELECT requested.role_id, %s
                    FROM unnest(%s::BIGINT[]) AS requested(role_id)
                    WHERE NOT EXISTS (
                        SELECT 1
                        FROM PermissionAssignment current_assignment
                        WHERE current_assignment.role_id = requested.role_id
                          AND current_assignment.document_id = %s
                    );
                    """,
                    [int(document_id), list(roles), int(document_id)],
                )
            conn.commit()
        finally:
            conn.close()

    def revoke_document_roles(self, document_id: int, role_ids: Iterable[int]) -> None:
        """Remove direct role permissions without deleting the document/vector."""
        roles = tuple(sorted({int(value) for value in role_ids}))
        if not roles:
            raise ValueError("acl_revoke requires at least one role_id")
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    DELETE FROM PermissionAssignment
                    WHERE document_id = %s
                      AND role_id = ANY(%s::BIGINT[]);
                    """,
                    [int(document_id), list(roles)],
                )
            conn.commit()
        finally:
            conn.close()

    def upsert_document_blocks(self, item: KMeansUpdateItem) -> None:
        if not item.vectors and item.source_document_id is None:
            return
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO Documents (document_id, document_name, created_at, updated_at)
                    VALUES (%s, %s, NOW(), NOW())
                    ON CONFLICT (document_id)
                    DO UPDATE SET document_name = EXCLUDED.document_name, updated_at = NOW();
                    """,
                    [int(item.document_id), item.document_name or f"Document {int(item.document_id)}"],
                )
                if item.source_document_id is not None and not item.vectors:
                    cur.execute("DELETE FROM documentblocks WHERE document_id = %s;", [int(item.document_id)])
                    cur.execute(
                        """
                        INSERT INTO documentblocks (block_id, document_id, block_content, hash_value, vector)
                        SELECT (SELECT COALESCE(MAX(block_id), 0) FROM documentblocks) + row_number() OVER (),
                               %s, block_content, hash_value, vector
                        FROM documentblocks
                        WHERE document_id = %s
                        ORDER BY block_id;
                        """,
                        [int(item.document_id), int(item.source_document_id)],
                    )
                elif item.vectors:
                    cur.execute("DELETE FROM documentblocks WHERE document_id = %s;", [int(item.document_id)])
                    if item.block_ids and len(item.block_ids) != len(item.vectors):
                        raise ValueError("block_ids length must match vectors length")
                    if item.block_ids:
                        block_ids = list(item.block_ids)
                    else:
                        cur.execute("SELECT COALESCE(MAX(block_id), 0) FROM documentblocks;")
                        start = int(cur.fetchone()[0]) + 1
                        block_ids = list(range(start, start + len(item.vectors)))
                    rows = [
                        (int(block_id), int(item.document_id), b"", b"", list(vector))
                        for block_id, vector in zip(block_ids, item.vectors)
                    ]
                    execute_values(
                        cur,
                        """
                        INSERT INTO documentblocks (block_id, document_id, block_content, hash_value, vector)
                        VALUES %s;
                        """,
                        rows,
                    )
            conn.commit()
        finally:
            conn.close()

    def delete_main_document_data(self, document_id: int) -> None:
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM PermissionAssignment WHERE document_id = %s;", [int(document_id)])
                cur.execute("DELETE FROM documentblocks WHERE document_id = %s;", [int(document_id)])
            conn.commit()
        finally:
            conn.close()

    def insert_document_into_partition(self, partition: KMeansPartition, pattern_id: int, document_id: int) -> None:
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(sql.SQL("DELETE FROM {} WHERE document_id = %s;").format(sql.Identifier(partition.table_name)), [int(document_id)])
                cur.execute(
                    sql.SQL(
                        """
                        INSERT INTO {} (block_id, document_id, pattern_id, block_content, vector)
                        SELECT block_id, document_id, %s, block_content, vector
                        FROM documentblocks
                        WHERE document_id = %s;
                        """
                    ).format(sql.Identifier(partition.table_name)),
                    [int(pattern_id), int(document_id)],
                )
                cur.execute(
                    sql.SQL("DELETE FROM {} WHERE partition_id = %s AND document_id = %s;").format(sql.Identifier(UPDATE_TOMBSTONE_TABLE)),
                    [str(partition.partition_id), int(document_id)],
                )
            conn.commit()
        finally:
            conn.close()


class _LocalPlanner:
    def __init__(self, *, ef_search: int, memory_budget: int, tau_del: float, max_operations: int) -> None:
        self.ef_search = int(ef_search)
        self.memory_budget = int(memory_budget)
        self.tau_del = float(tau_del)
        self.max_operations = int(max_operations)
        self._bit_context_cache_key: tuple[int, int] | None = None
        self._bit_context_cache: _PatternBitsetContext | None = None

    def _bit_context(self, patterns_by_id: dict[int, ACLPattern]) -> _PatternBitsetContext:
        cache_key = (id(patterns_by_id), len(patterns_by_id))
        if self._bit_context_cache is None or self._bit_context_cache_key != cache_key:
            self._bit_context_cache = _PatternBitsetContext(patterns_by_id)
            self._bit_context_cache_key = cache_key
        return self._bit_context_cache

    def cost(self, partition: KMeansPartition, patterns_by_id: dict[int, ACLPattern]) -> float:
        bit_context = self._bit_context(patterns_by_id)
        partition_bits = bit_context.pattern_bits_for(partition.pattern_ids)
        partition_vectors = max(1, int(bit_context.vector_count_bits(partition_bits)))
        total = 0.0
        for _tenant_id, tenant_bits in bit_context.tenant_bits_for_partition(partition).items():
            accessible = bit_context.vector_count_bits(int(tenant_bits) & int(partition_bits))
            if accessible <= 0:
                continue
            total += estimate_partition_query_cost(
                partition_vectors=int(partition_vectors),
                accessible_vectors=int(accessible),
                tenant_weight=1.0,
                ef_search=int(self.ef_search),
                use_adaptive_ef=True,
            )
        return float(total)

    def _operation_bit_specs(
        self,
        left_pattern_bits: int,
        left_tenant_bits: dict[int, int],
        right_pattern_bits: int,
        right_tenant_bits: dict[int, int],
        bit_context: _PatternBitsetContext,
        operation: str,
    ) -> list[tuple[int, dict[int, int]]]:
        left_pattern_bits = int(left_pattern_bits)
        right_pattern_bits = int(right_pattern_bits)
        overlap_bits = int(left_pattern_bits) & int(right_pattern_bits)
        if int(overlap_bits) == 0:
            return []
        if operation == "full":
            return [
                (
                    int(left_pattern_bits | right_pattern_bits),
                    bit_context.union_tenant_bits(left_tenant_bits, right_tenant_bits),
                )
            ]
        if operation == "move_left":
            moved = bit_context.mask_tenant_bits(left_tenant_bits, overlap_bits)
            return [
                (
                    int(left_pattern_bits & ~overlap_bits),
                    bit_context.remove_bits_from_tenants(left_tenant_bits, overlap_bits),
                ),
                (
                    int(right_pattern_bits),
                    bit_context.union_tenant_bits(right_tenant_bits, moved),
                ),
            ]
        if operation == "move_right":
            moved = bit_context.mask_tenant_bits(right_tenant_bits, overlap_bits)
            return [
                (
                    int(left_pattern_bits),
                    bit_context.union_tenant_bits(left_tenant_bits, moved),
                ),
                (
                    int(right_pattern_bits & ~overlap_bits),
                    bit_context.remove_bits_from_tenants(right_tenant_bits, overlap_bits),
                ),
            ]
        if operation == "split_overlap":
            return [
                (
                    int(left_pattern_bits & ~overlap_bits),
                    bit_context.remove_bits_from_tenants(left_tenant_bits, overlap_bits),
                ),
                (
                    int(overlap_bits),
                    bit_context.union_tenant_bits(
                        bit_context.mask_tenant_bits(left_tenant_bits, overlap_bits),
                        bit_context.mask_tenant_bits(right_tenant_bits, overlap_bits),
                    ),
                ),
                (
                    int(right_pattern_bits & ~overlap_bits),
                    bit_context.remove_bits_from_tenants(right_tenant_bits, overlap_bits),
                ),
            ]
        if operation == "merge_extract_overlap":
            return [
                (
                    int((left_pattern_bits | right_pattern_bits) & ~overlap_bits),
                    bit_context.union_tenant_bits(
                        bit_context.remove_bits_from_tenants(left_tenant_bits, overlap_bits),
                        bit_context.remove_bits_from_tenants(right_tenant_bits, overlap_bits),
                    ),
                ),
                (
                    int(overlap_bits),
                    bit_context.union_tenant_bits(
                        bit_context.mask_tenant_bits(left_tenant_bits, overlap_bits),
                        bit_context.mask_tenant_bits(right_tenant_bits, overlap_bits),
                    ),
                ),
            ]
        raise ValueError(f"unknown operation: {operation}")

    def _operation_specs(
        self,
        left: KMeansPartition,
        right: KMeansPartition,
        patterns_by_id: dict[int, ACLPattern],
        operation: str,
    ) -> list[tuple[set[int], dict[int, set[int]]]]:
        left_stored = set(map(int, left.pattern_ids))
        right_stored = set(map(int, right.pattern_ids))
        overlap = left_stored & right_stored
        if not overlap:
            return []
        left_tenants = _tenant_patterns_for_partition(left, patterns_by_id)
        right_tenants = _tenant_patterns_for_partition(right, patterns_by_id)

        def mask(mapping: dict[int, set[int]], keep: set[int]) -> dict[int, set[int]]:
            return {tenant: set(values) & set(keep) for tenant, values in mapping.items() if set(values) & set(keep)}

        def remove(mapping: dict[int, set[int]], drop: set[int]) -> dict[int, set[int]]:
            return {tenant: set(values) - set(drop) for tenant, values in mapping.items() if set(values) - set(drop)}

        def union(*maps: dict[int, set[int]]) -> dict[int, set[int]]:
            result: dict[int, set[int]] = defaultdict(set)
            for mapping in maps:
                for tenant, values in mapping.items():
                    result[int(tenant)].update(int(value) for value in values)
            return dict(result)

        if operation == "full":
            return [(left_stored | right_stored, union(left_tenants, right_tenants))]
        if operation == "move_left":
            moved = mask(left_tenants, overlap)
            return [(left_stored - overlap, remove(left_tenants, overlap)), (right_stored, union(right_tenants, moved))]
        if operation == "move_right":
            moved = mask(right_tenants, overlap)
            return [(left_stored, union(left_tenants, moved)), (right_stored - overlap, remove(right_tenants, overlap))]
        if operation == "split_overlap":
            return [
                (left_stored - overlap, remove(left_tenants, overlap)),
                (set(overlap), union(mask(left_tenants, overlap), mask(right_tenants, overlap))),
                (right_stored - overlap, remove(right_tenants, overlap)),
            ]
        if operation == "merge_extract_overlap":
            return [
                ((left_stored | right_stored) - overlap, union(remove(left_tenants, overlap), remove(right_tenants, overlap))),
                (set(overlap), union(mask(left_tenants, overlap), mask(right_tenants, overlap))),
            ]
        raise ValueError(f"unknown operation: {operation}")

    def best_pair_candidate(
        self,
        left: KMeansPartition,
        right: KMeansPartition,
        patterns_by_id: dict[int, ACLPattern],
        tombstones_by_partition: dict[str, int],
        *,
        allowed_operations: Iterable[str] | None = None,
    ) -> Optional[KMeansMaintenanceCandidate]:
        before_cost = self.cost(left, patterns_by_id) + self.cost(right, patterns_by_id)
        before_memory = int(left.vector_count) + int(right.vector_count)
        before_actual_memory = before_memory + int(tombstones_by_partition.get(str(left.partition_id), 0)) + int(tombstones_by_partition.get(str(right.partition_id), 0))
        best: Optional[KMeansMaintenanceCandidate] = None
        best_rank: tuple[float, int, int] | None = None
        op_rank = {"split_overlap": 0, "merge_extract_overlap": 1, "move_left": 2, "move_right": 3, "full": 4}
        operations = tuple(str(value) for value in (allowed_operations or ("full", "move_left", "move_right", "split_overlap", "merge_extract_overlap")))
        for operation in operations:
            specs = self._operation_specs(left, right, patterns_by_id, operation)
            new_partitions = []
            for index, (pattern_ids, tenant_patterns) in enumerate(specs):
                candidate = _make_partition(
                    f"{left.partition_id}__{right.partition_id}__{operation}_{index}",
                    cluster_id=int(left.cluster_id),
                    pattern_ids=pattern_ids,
                    tenant_patterns=tenant_patterns,
                    patterns_by_id=patterns_by_id,
                )
                if candidate is not None:
                    new_partitions.append(candidate)
            if not new_partitions:
                continue
            after_memory = sum(int(partition.vector_count) for partition in new_partitions)
            max_partition_size = max(int(partition.vector_count) for partition in new_partitions)
            memory_saved = int(before_actual_memory) - int(after_memory)
            delta_latency = sum(self.cost(partition, patterns_by_id) for partition in new_partitions) - float(before_cost)
            if memory_saved <= 0:
                continue
            operation_rank = int(op_rank[operation])
            payload = {
                "operation": operation,
                "new_partitions": new_partitions,
                "op_rank": int(operation_rank),
                "max_result_partition_size": int(max_partition_size),
            }
            candidate = KMeansMaintenanceCandidate(
                op_type=str(operation),
                partition_ids=(str(left.partition_id), str(right.partition_id)),
                delta_memory=-int(memory_saved),
                delta_latency=float(delta_latency),
                payload=payload,
            )
            rank = (float(candidate.delta_latency), int(max_partition_size), int(operation_rank))
            if best is None or best_rank is None or rank < best_rank:
                best = candidate
                best_rank = rank
        return best

    def core_star_pair_candidates(
        self,
        partitions: list[KMeansPartition],
        patterns_by_id: dict[int, ACLPattern],
        tombstones_by_partition: dict[str, int],
        *,
        top_d: int,
        allowed_operations: Iterable[str] | None = None,
    ) -> tuple[list[KMeansMaintenanceCandidate], dict[str, int]]:
        partitions_by_id = {str(partition.partition_id): partition for partition in partitions}
        if len(partitions_by_id) <= 1:
            return [], {
                "core_star_candidate_edges": 0,
                "core_star_heap_entries": 0,
                "core_star_cache_hits": 0,
                "core_star_cache_misses": 0,
            }

        owners_by_pattern: dict[int, set[str]] = defaultdict(set)
        for partition in partitions:
            for pattern_id in partition.pattern_ids:
                if int(pattern_id) in patterns_by_id:
                    owners_by_pattern[int(pattern_id)].add(str(partition.partition_id))

        def pattern_core_key(pattern_id: int, partition_id: str) -> tuple[int, float, int, int, str]:
            partition = partitions_by_id[str(partition_id)]
            pattern_vectors = max(0, int(patterns_by_id[int(pattern_id)].vector_count))
            partition_vectors = max(1, int(partition.vector_count))
            acl_share = float(pattern_vectors) / float(partition_vectors)
            return (
                int(len(partition.pattern_ids)),
                -float(acl_share),
                int(partition_vectors),
                int(len(partition.tenant_ids)),
                str(partition_id),
            )

        edge_acl_counts: Counter[tuple[str, str]] = Counter()
        edge_signal_scores: dict[tuple[str, str], float] = defaultdict(float)
        for pattern_id, owner_ids in owners_by_pattern.items():
            live_owner_ids = sorted(str(owner_id) for owner_id in owner_ids if str(owner_id) in partitions_by_id)
            if len(live_owner_ids) <= 1:
                continue
            core_id = min(live_owner_ids, key=lambda owner_id: pattern_core_key(int(pattern_id), str(owner_id)))
            signal = float(patterns_by_id[int(pattern_id)].vector_count)
            if signal <= 0.0:
                continue
            for owner_id in live_owner_ids:
                if str(owner_id) == str(core_id):
                    continue
                edge = tuple(sorted((str(core_id), str(owner_id))))
                edge_acl_counts[edge] += 1
                edge_signal_scores[edge] += float(signal)

        incident_edges: dict[str, list[tuple[float, float, tuple[str, str]]]] = defaultdict(list)
        for edge, shared_acl_count in edge_acl_counts.items():
            left, right = edge
            if left not in partitions_by_id or right not in partitions_by_id:
                continue
            left_acl_count = max(1, len(partitions_by_id[left].pattern_ids))
            right_acl_count = max(1, len(partitions_by_id[right].pattern_ids))
            rank_score = float(shared_acl_count) / math.sqrt(float(left_acl_count) * float(right_acl_count))
            signal = float(edge_signal_scores.get(edge, 0.0))
            incident_edges[left].append((float(rank_score), float(signal), edge))
            incident_edges[right].append((float(rank_score), float(signal), edge))

        selected_edges: set[tuple[str, str]] = set()
        for partition_id, edges in incident_edges.items():
            edges.sort(key=lambda item: (-float(item[0]), -float(item[1]), item[2][0], item[2][1]))
            for _rank_score, _signal, edge in edges[: max(1, int(top_d))]:
                selected_edges.add(edge)

        operation_rank = {"split_overlap": 0, "merge_extract_overlap": 1, "move_left": 2, "move_right": 3, "full": 4}
        candidate_cache: dict[tuple[str, str], KMeansMaintenanceCandidate | None] = {}
        cache_hits = 0
        cache_misses = 0
        heap: list[tuple[object, ...]] = []

        def candidate_for_edge(edge: tuple[str, str]) -> KMeansMaintenanceCandidate | None:
            nonlocal cache_hits, cache_misses
            cache_key = tuple(sorted((str(edge[0]), str(edge[1]))))
            if cache_key in candidate_cache:
                cache_hits += 1
                return candidate_cache[cache_key]
            cache_misses += 1
            left = partitions_by_id.get(cache_key[0])
            right = partitions_by_id.get(cache_key[1])
            if left is None or right is None:
                candidate_cache[cache_key] = None
                return None
            candidate = self.best_pair_candidate(
                left,
                right,
                patterns_by_id,
                tombstones_by_partition,
                allowed_operations=allowed_operations,
            )
            candidate_cache[cache_key] = candidate
            return candidate

        for edge in sorted(selected_edges):
            candidate = candidate_for_edge(edge)
            if candidate is None:
                continue
            memory_saved = abs(int(candidate.delta_memory))
            if float(candidate.delta_latency) <= 0.0:
                latency_class = 0
                heap_score = -float(memory_saved)
            else:
                latency_class = 1
                heap_score = float(candidate.delta_latency) / float(max(1, memory_saved))
            heapq.heappush(
                heap,
                (
                    int(latency_class),
                    float(heap_score),
                    float(candidate.delta_latency),
                    -int(memory_saved),
                    int(operation_rank.get(str(candidate.op_type), 999)),
                    str(edge[0]),
                    str(edge[1]),
                ),
            )

        ordered_candidates: list[KMeansMaintenanceCandidate] = []
        seen_edges: set[tuple[str, str]] = set()
        while heap:
            _latency_class, _heap_score, _delta_latency, _neg_memory_saved, _op_rank, left_id, right_id = heapq.heappop(heap)
            edge = tuple(sorted((str(left_id), str(right_id))))
            if edge in seen_edges:
                continue
            seen_edges.add(edge)
            candidate = candidate_for_edge(edge)
            if candidate is not None:
                ordered_candidates.append(candidate)

        return ordered_candidates, {
            "core_star_candidate_edges": int(len(selected_edges)),
            "core_star_heap_entries": int(len(heap) + len(ordered_candidates)),
            "core_star_cache_hits": int(cache_hits),
            "core_star_cache_misses": int(cache_misses),
            "core_star_owner_patterns": int(sum(1 for owners in owners_by_pattern.values() if len(owners) > 1)),
        }

    def compact_candidates(self, partitions: list[KMeansPartition], tombstones_by_partition: dict[str, int]) -> list[KMeansMaintenanceCandidate]:
        result = []
        for partition in partitions:
            tombstones = int(tombstones_by_partition.get(str(partition.partition_id), 0) or 0)
            if tombstones <= 0:
                continue
            ratio = float(tombstones) / float(max(1, int(partition.vector_count) + int(tombstones)))
            if ratio < self.tau_del:
                continue
            result.append(
                KMeansMaintenanceCandidate(
                    op_type="compact",
                    partition_ids=(str(partition.partition_id),),
                    delta_memory=-int(tombstones),
                    delta_latency=0.0,
                    payload={"tombstone_ratio": ratio},
                )
            )
        return result

    def selectivity_candidates(
        self,
        partitions: list[KMeansPartition],
        patterns_by_id: dict[int, ACLPattern],
    ) -> list[KMeansMaintenanceCandidate]:
        result = []
        ranked_partitions: list[tuple[float, int, str, KMeansPartition, dict[int, set[int]], int]] = []
        for partition in partitions:
            tenant_patterns = _tenant_patterns_for_partition(partition, patterns_by_id)
            if len(tenant_patterns) <= 1:
                continue
            partition_vectors = max(1, int(partition.vector_count))
            worst_tenant = None
            worst_selectivity = 1.0
            for tenant_id, pattern_ids in tenant_patterns.items():
                accessible = _partition_live_vectors(pattern_ids, patterns_by_id)
                selectivity = float(accessible) / float(partition_vectors)
                if selectivity < worst_selectivity:
                    worst_selectivity = float(selectivity)
                    worst_tenant = int(tenant_id)
            if worst_tenant is None or worst_selectivity >= 1.0:
                continue
            ranked_partitions.append(
                (
                    float(worst_selectivity),
                    -int(partition_vectors),
                    str(partition.partition_id),
                    partition,
                    tenant_patterns,
                    int(worst_tenant),
                )
            )

        ranked_partitions.sort(key=lambda item: (float(item[0]), int(item[1]), str(item[2])))

        for _worst_selectivity, _neg_partition_vectors, _partition_id, partition, tenant_patterns, worst_tenant in ranked_partitions:
            extract_patterns = set(tenant_patterns[int(worst_tenant)])
            remain_tenants = {tenant: set(values) - extract_patterns for tenant, values in tenant_patterns.items()}
            extract_tenants = {tenant: set(values) & extract_patterns for tenant, values in tenant_patterns.items()}
            specs = [
                (set(partition.pattern_ids) - extract_patterns, remain_tenants),
                (extract_patterns, extract_tenants),
            ]
            new_partitions = []
            for index, (pattern_ids, route_map) in enumerate(specs):
                candidate = _make_partition(
                    f"{partition.partition_id}__selectivity_{index}",
                    cluster_id=int(partition.cluster_id),
                    pattern_ids=pattern_ids,
                    tenant_patterns=route_map,
                    patterns_by_id=patterns_by_id,
                )
                if candidate is not None:
                    new_partitions.append(candidate)
            if len(new_partitions) < 2:
                continue
            delta_latency = sum(self.cost(value, patterns_by_id) for value in new_partitions) - self.cost(partition, patterns_by_id)
            delta_memory = sum(int(value.vector_count) for value in new_partitions) - int(partition.vector_count)
            if delta_latency < 0.0:
                result.append(
                    KMeansMaintenanceCandidate(
                        op_type="selectivity_split",
                        partition_ids=(str(partition.partition_id),),
                        delta_memory=int(delta_memory),
                        delta_latency=float(delta_latency),
                        payload={"new_partitions": new_partitions, "worst_tenant": int(worst_tenant)},
                    )
                )
        return result

    def _selectivity_profile(self, partition: KMeansPartition, patterns_by_id: dict[int, ACLPattern]) -> dict[str, object] | None:
        partition_vectors = int(partition.vector_count)
        if partition_vectors <= 0:
            return None
        tenant_patterns = _tenant_patterns_for_partition(partition, patterns_by_id)
        if not tenant_patterns:
            return None
        total_selectivity = 0.0
        worst_tenant = None
        worst_access = 0
        worst_selectivity = 1.0
        worst_rank: tuple[float, int, int] | None = None
        is_pure = True
        live_tenant_count = 0
        for tenant_id, pattern_ids in tenant_patterns.items():
            accessible = _partition_live_vectors(pattern_ids, patterns_by_id)
            if int(accessible) <= 0:
                continue
            live_tenant_count += 1
            capped_access = min(int(partition_vectors), max(0, int(accessible)))
            selectivity = float(capped_access) / float(partition_vectors)
            total_selectivity += float(selectivity)
            if int(capped_access) < int(partition_vectors):
                is_pure = False
            rank = (float(selectivity), int(capped_access), int(tenant_id))
            if worst_rank is None or rank < worst_rank:
                worst_rank = rank
                worst_tenant = int(tenant_id)
                worst_access = int(capped_access)
                worst_selectivity = float(selectivity)
        if worst_tenant is None or live_tenant_count <= 0:
            return None
        return {
            "partition_id": str(partition.partition_id),
            "partition_vectors": int(partition_vectors),
            "tenant_count": int(live_tenant_count),
            "avg_selectivity": float(total_selectivity) / float(max(1, int(live_tenant_count))),
            "worst_selectivity": float(worst_selectivity),
            "worst_tenant": int(worst_tenant),
            "worst_access": int(worst_access),
            "is_pure": bool(is_pure),
        }

    def _selectivity_refine_candidate(
        self,
        partition: KMeansPartition,
        patterns_by_id: dict[int, ACLPattern],
        profile: dict[str, object],
    ) -> tuple[KMeansMaintenanceCandidate | None, str]:
        if bool(profile.get("is_pure", False)):
            return None, "worst_group_pure"
        tenant_patterns = _tenant_patterns_for_partition(partition, patterns_by_id)
        worst_tenant = int(profile["worst_tenant"])
        stored_patterns = set(map(int, partition.pattern_ids))
        extract_patterns = set(tenant_patterns.get(int(worst_tenant), set())) & stored_patterns
        if not extract_patterns or extract_patterns == stored_patterns:
            return None, "no_extractable_worst_tenant_bits"
        remain_tenants = {tenant: set(values) - extract_patterns for tenant, values in tenant_patterns.items()}
        extract_tenants = {tenant: set(values) & extract_patterns for tenant, values in tenant_patterns.items()}
        specs = [
            (stored_patterns - extract_patterns, remain_tenants),
            (extract_patterns, extract_tenants),
        ]
        new_partitions = []
        for index, (pattern_ids, route_map) in enumerate(specs):
            candidate = _make_partition(
                f"{partition.partition_id}__selectivity_{index}",
                cluster_id=int(partition.cluster_id),
                pattern_ids=pattern_ids,
                tenant_patterns=route_map,
                patterns_by_id=patterns_by_id,
            )
            if candidate is not None:
                new_partitions.append(candidate)
        if len(new_partitions) < 2:
            return None, "invalid_extract_specs"
        before_cost = self.cost(partition, patterns_by_id)
        before_memory = int(partition.vector_count)
        after_cost = sum(self.cost(value, patterns_by_id) for value in new_partitions)
        after_memory = sum(int(value.vector_count) for value in new_partitions)
        if int(after_memory) != int(before_memory):
            return None, "invalid_extract_specs"
        delta_latency = float(after_cost) - float(before_cost)
        if float(delta_latency) >= 0.0:
            return None, "worst_group_not_beneficial"
        return (
            KMeansMaintenanceCandidate(
                op_type="selectivity_split",
                partition_ids=(str(partition.partition_id),),
                delta_memory=0,
                delta_latency=float(delta_latency),
                payload={
                    "new_partitions": new_partitions,
                    "worst_tenant": int(worst_tenant),
                    "avg_selectivity": float(profile["avg_selectivity"]),
                    "worst_selectivity": float(profile["worst_selectivity"]),
                },
            ),
            "refine_candidate_pending",
        )

    def _route_aware_cost(self, partitions: list[KMeansPartition], patterns_by_id: dict[int, ACLPattern]) -> float:
        route_options: dict[int, dict[str, set[int]]] = defaultdict(dict)
        partitions_by_id = {str(partition.partition_id): partition for partition in partitions}
        for partition in partitions:
            for tenant_id, pattern_ids in _tenant_patterns_for_partition(partition, patterns_by_id).items():
                kept = {int(pattern_id) for pattern_id in pattern_ids if int(pattern_id) in patterns_by_id}
                if kept:
                    route_options[int(tenant_id)][str(partition.partition_id)] = kept

        total = 0.0
        for _tenant_id, by_partition in route_options.items():
            uncovered = set()
            for pattern_ids in by_partition.values():
                uncovered.update(int(pattern_id) for pattern_id in pattern_ids)
            while uncovered:
                best_partition_id = None
                best_patterns: set[int] = set()
                best_rank = None
                for partition_id, pattern_ids in by_partition.items():
                    covered = set(pattern_ids) & uncovered
                    if not covered:
                        continue
                    partition = partitions_by_id.get(str(partition_id))
                    if partition is None:
                        continue
                    accessible = _partition_live_vectors(covered, patterns_by_id)
                    if int(accessible) <= 0:
                        continue
                    route_cost = estimate_partition_query_cost(
                        partition_vectors=max(1, int(partition.vector_count)),
                        accessible_vectors=int(accessible),
                        tenant_weight=1.0,
                        ef_search=int(self.ef_search),
                        use_adaptive_ef=True,
                    )
                    rank = (
                        -int(len(covered)),
                        float(route_cost),
                        int(partition.vector_count),
                        str(partition_id),
                    )
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_partition_id = str(partition_id)
                        best_patterns = set(covered)
                if best_partition_id is None:
                    break
                partition = partitions_by_id[str(best_partition_id)]
                accessible = _partition_live_vectors(best_patterns, patterns_by_id)
                total += estimate_partition_query_cost(
                    partition_vectors=max(1, int(partition.vector_count)),
                    accessible_vectors=int(accessible),
                    tenant_weight=1.0,
                    ef_search=int(self.ef_search),
                    use_adaptive_ef=True,
                )
                uncovered -= set(best_patterns)
        return float(total)

    def _replicated_selectivity_refine_candidate(
        self,
        partition: KMeansPartition,
        patterns_by_id: dict[int, ACLPattern],
        profile: dict[str, object],
    ) -> tuple[KMeansMaintenanceCandidate | None, str]:
        if bool(profile.get("is_pure", False)):
            return None, "worst_group_pure"
        tenant_patterns = _tenant_patterns_for_partition(partition, patterns_by_id)
        if len(tenant_patterns) <= 1:
            return None, "no_refine_group"
        stored_patterns = {int(pattern_id) for pattern_id in partition.pattern_ids if int(pattern_id) in patterns_by_id}
        if len(stored_patterns) <= 1:
            return None, "no_extractable_worst_tenant_bits"

        partition_vectors = max(1, int(partition.vector_count))
        tenant_rank: list[tuple[float, float, int, int]] = []
        tenant_importance: dict[int, float] = {}
        for tenant_id, pattern_ids in tenant_patterns.items():
            kept_patterns = {int(pattern_id) for pattern_id in pattern_ids if int(pattern_id) in stored_patterns}
            if not kept_patterns:
                continue
            accessible = _partition_live_vectors(kept_patterns, patterns_by_id)
            if int(accessible) <= 0:
                continue
            selectivity = float(min(int(accessible), int(partition_vectors))) / float(partition_vectors)
            importance = float(accessible) * max(0.0, 1.0 - float(selectivity))
            if importance <= 0.0:
                importance = float(accessible)
            tenant_importance[int(tenant_id)] = float(importance)
            tenant_rank.append((-float(importance), float(selectivity), int(tenant_id), int(accessible)))
        tenant_rank.sort()
        important_tenants = [int(tenant_id) for _neg_importance, _selectivity, tenant_id, _accessible in tenant_rank[:64]]
        if len(important_tenants) < 2:
            return None, "no_refine_group"

        def pattern_weight(pattern_id: int) -> int:
            return max(0, int(patterns_by_id[int(pattern_id)].vector_count))

        def weighted_overlap(left_tenant: int, right_tenant: int) -> int:
            return sum(pattern_weight(pattern_id) for pattern_id in set(tenant_patterns[left_tenant]) & set(tenant_patterns[right_tenant]) & stored_patterns)

        best_seed_pair = None
        best_seed_rank = None
        for left_tenant in important_tenants:
            left_patterns = set(tenant_patterns[int(left_tenant)]) & stored_patterns
            if not left_patterns:
                continue
            for right_tenant in important_tenants:
                if int(left_tenant) >= int(right_tenant):
                    continue
                right_patterns = set(tenant_patterns[int(right_tenant)]) & stored_patterns
                if not right_patterns:
                    continue
                union_weight = sum(pattern_weight(pattern_id) for pattern_id in left_patterns | right_patterns)
                if int(union_weight) <= 0:
                    continue
                overlap_weight = weighted_overlap(int(left_tenant), int(right_tenant))
                overlap_ratio = float(overlap_weight) / float(max(1, int(union_weight)))
                rank = (
                    float(overlap_ratio),
                    int(overlap_weight),
                    -float(tenant_importance.get(int(left_tenant), 0.0) + tenant_importance.get(int(right_tenant), 0.0)),
                    int(left_tenant),
                    int(right_tenant),
                )
                if best_seed_rank is None or rank < best_seed_rank:
                    best_seed_rank = rank
                    best_seed_pair = (int(left_tenant), int(right_tenant))
        if best_seed_pair is None:
            return None, "no_refine_group"

        left_tenants = {int(best_seed_pair[0])}
        right_tenants = {int(best_seed_pair[1])}
        left_union = set(tenant_patterns[int(best_seed_pair[0])]) & stored_patterns
        right_union = set(tenant_patterns[int(best_seed_pair[1])]) & stored_patterns
        for tenant_id in important_tenants:
            if int(tenant_id) in left_tenants or int(tenant_id) in right_tenants:
                continue
            current_patterns = set(tenant_patterns[int(tenant_id)]) & stored_patterns
            left_overlap = sum(pattern_weight(pattern_id) for pattern_id in current_patterns & left_union)
            right_overlap = sum(pattern_weight(pattern_id) for pattern_id in current_patterns & right_union)
            if int(left_overlap) > int(right_overlap):
                left_tenants.add(int(tenant_id))
                left_union.update(current_patterns)
            elif int(right_overlap) > int(left_overlap):
                right_tenants.add(int(tenant_id))
                right_union.update(current_patterns)
            else:
                left_weight = sum(float(tenant_importance.get(int(value), 0.0)) for value in left_tenants)
                right_weight = sum(float(tenant_importance.get(int(value), 0.0)) for value in right_tenants)
                if float(left_weight) <= float(right_weight):
                    left_tenants.add(int(tenant_id))
                    left_union.update(current_patterns)
                else:
                    right_tenants.add(int(tenant_id))
                    right_union.update(current_patterns)

        tenant_side: dict[int, str] = {int(tenant_id): "left" for tenant_id in left_tenants}
        tenant_side.update({int(tenant_id): "right" for tenant_id in right_tenants})
        for tenant_id, pattern_ids in tenant_patterns.items():
            if int(tenant_id) in tenant_side:
                continue
            current_patterns = set(pattern_ids) & stored_patterns
            left_overlap = sum(pattern_weight(pattern_id) for pattern_id in current_patterns & left_union)
            right_overlap = sum(pattern_weight(pattern_id) for pattern_id in current_patterns & right_union)
            if int(left_overlap) > int(right_overlap):
                tenant_side[int(tenant_id)] = "left"
            elif int(right_overlap) > int(left_overlap):
                tenant_side[int(tenant_id)] = "right"
            else:
                left_weight = sum(float(tenant_importance.get(int(value), 0.0)) for value in left_tenants)
                right_weight = sum(float(tenant_importance.get(int(value), 0.0)) for value in right_tenants)
                tenant_side[int(tenant_id)] = "left" if float(left_weight) <= float(right_weight) else "right"

        left_patterns: set[int] = set()
        right_patterns: set[int] = set()
        replicated_patterns: set[int] = set()
        left_vectors = 0
        right_vectors = 0
        for pattern_id in sorted(stored_patterns):
            pattern_tenants = {int(tenant_id) for tenant_id in patterns_by_id[int(pattern_id)].tenant_ids}
            left_score = sum(float(tenant_importance.get(int(tenant_id), 0.0)) for tenant_id in pattern_tenants & left_tenants)
            right_score = sum(float(tenant_importance.get(int(tenant_id), 0.0)) for tenant_id in pattern_tenants & right_tenants)
            vectors = pattern_weight(int(pattern_id))
            if left_score > 0.0 and right_score > 0.0:
                left_patterns.add(int(pattern_id))
                right_patterns.add(int(pattern_id))
                replicated_patterns.add(int(pattern_id))
                left_vectors += int(vectors)
                right_vectors += int(vectors)
            elif left_score > 0.0:
                left_patterns.add(int(pattern_id))
                left_vectors += int(vectors)
            elif right_score > 0.0:
                right_patterns.add(int(pattern_id))
                right_vectors += int(vectors)
            elif int(left_vectors) <= int(right_vectors):
                left_patterns.add(int(pattern_id))
                left_vectors += int(vectors)
            else:
                right_patterns.add(int(pattern_id))
                right_vectors += int(vectors)

        if not left_patterns or not right_patterns:
            return None, "invalid_replicated_split_specs"
        if left_patterns == stored_patterns and right_patterns == stored_patterns:
            return None, "invalid_replicated_split_specs"

        def route_map(pattern_ids: set[int], side: str) -> dict[int, set[int]]:
            result: dict[int, set[int]] = defaultdict(set)
            for pattern_id in pattern_ids:
                pattern = patterns_by_id.get(int(pattern_id))
                if pattern is None:
                    continue
                is_replicated = int(pattern_id) in replicated_patterns
                for tenant_id in pattern.tenant_ids:
                    if is_replicated and tenant_side.get(int(tenant_id), str(side)) != str(side):
                        continue
                    result[int(tenant_id)].add(int(pattern_id))
            return dict(result)

        specs = [
            (left_patterns, route_map(left_patterns, "left")),
            (right_patterns, route_map(right_patterns, "right")),
        ]
        new_partitions = []
        for index, (pattern_ids, tenant_route_map) in enumerate(specs):
            candidate = _make_partition(
                f"{partition.partition_id}__replicated_selectivity_{index}",
                cluster_id=int(partition.cluster_id),
                pattern_ids=pattern_ids,
                tenant_patterns=tenant_route_map,
                patterns_by_id=patterns_by_id,
            )
            if candidate is not None:
                new_partitions.append(candidate)
        if len(new_partitions) < 2:
            return None, "invalid_replicated_split_specs"

        before_cost = self._route_aware_cost([partition], patterns_by_id)
        after_cost = self._route_aware_cost(new_partitions, patterns_by_id)
        delta_latency = float(after_cost) - float(before_cost)
        delta_memory = sum(int(value.vector_count) for value in new_partitions) - int(partition.vector_count)
        if float(delta_latency) >= 0.0:
            return None, "replicated_split_not_beneficial"
        if int(delta_memory) < 0:
            delta_memory = 0
        return (
            KMeansMaintenanceCandidate(
                op_type="replicated_selectivity_split",
                partition_ids=(str(partition.partition_id),),
                delta_memory=int(delta_memory),
                delta_latency=float(delta_latency),
                payload={
                    "new_partitions": new_partitions,
                    "left_tenant_count": int(len(left_tenants)),
                    "right_tenant_count": int(len(right_tenants)),
                    "replicated_pattern_count": int(len(replicated_patterns)),
                    "replicated_vector_count": int(sum(pattern_weight(pattern_id) for pattern_id in replicated_patterns)),
                    "avg_selectivity": float(profile["avg_selectivity"]),
                    "worst_selectivity": float(profile["worst_selectivity"]),
                    "worst_tenant": int(profile["worst_tenant"]),
                },
            ),
            "replicated_refine_candidate_pending",
        )

    def plan_query_refinement(
        self,
        partitions: list[KMeansPartition],
        patterns_by_id: dict[int, ACLPattern],
        tombstones_by_partition: dict[str, int],
        *,
        memory_now: int,
        top_d: int,
        used_partition_ids: set[str],
        optimization_partition_ids: set[str] | None = None,
        compact_partition_ids: set[str] | None = None,
    ) -> tuple[list[KMeansMaintenanceCandidate], dict[str, int | float | str]]:
        current = list(partitions)
        optimization_ids = {str(value) for value in (optimization_partition_ids or {str(partition.partition_id) for partition in current})}
        compact_ids = {str(value) for value in (compact_partition_ids or optimization_ids)}
        local_tombstones = {str(key): int(value) for key, value in tombstones_by_partition.items()}
        used_ids = set(str(value) for value in used_partition_ids)
        accepted: list[KMeansMaintenanceCandidate] = []
        metadata: dict[str, int | float | str] = {
            "compact_candidate_count": 0,
            "pair_candidate_count": 0,
            "core_star_candidate_edges": 0,
            "core_star_heap_entries": 0,
            "core_star_cache_hits": 0,
            "core_star_cache_misses": 0,
            "core_star_owner_patterns": 0,
        }
        stop_reason = "memory_budget_satisfied"
        last_avg_selectivity = None
        last_worst_selectivity = None
        last_worst_tenant = None
        last_partition_id = None
        total_delta_latency = 0.0
        last_selectivity_stop_reason = ""

        def best_selectivity_candidate() -> tuple[KMeansMaintenanceCandidate | None, dict[str, int | float | str]]:
            current_by_id = {str(partition.partition_id): partition for partition in current if str(partition.partition_id) in optimization_ids}
            selectivity_heap: list[tuple[float, float, int, str]] = []
            profile_count = 0
            for partition in current_by_id.values():
                profile = self._selectivity_profile(partition, patterns_by_id)
                if profile is None:
                    continue
                profile_count += 1
                heapq.heappush(
                    selectivity_heap,
                    (
                        float(profile["avg_selectivity"]),
                        float(profile["worst_selectivity"]),
                        -int(profile["partition_vectors"]),
                        str(partition.partition_id),
                    ),
                )
            if not selectivity_heap:
                return None, {
                    "selectivity_candidate_count": 0,
                    "selectivity_profile_count": int(profile_count),
                    "selectivity_extract_stop_reason": "no_refine_group",
                }
            while selectivity_heap:
                _avg_selectivity, _worst_selectivity, _negative_vectors, partition_id = heapq.heappop(selectivity_heap)
                partition = current_by_id.get(str(partition_id))
                if partition is None:
                    continue
                profile = self._selectivity_profile(partition, patterns_by_id)
                if profile is None:
                    continue
                candidate, reason = self._replicated_selectivity_refine_candidate(partition, patterns_by_id, profile)
                if (
                    candidate is not None
                    and float(candidate.delta_latency) < 0.0
                    and int(memory_now) + int(candidate.delta_memory) <= int(self.memory_budget)
                ):
                    return candidate, {
                        "selectivity_candidate_count": 1,
                        "selectivity_profile_count": int(profile_count),
                        "selectivity_extract_stop_reason": str(reason),
                        "selectivity_extract_last_group_id": str(partition.partition_id),
                        "selectivity_extract_last_worst_tenant": int(profile["worst_tenant"]),
                        "selectivity_extract_last_avg_selectivity": float(profile["avg_selectivity"]),
                        "selectivity_extract_last_worst_selectivity": float(profile["worst_selectivity"]),
                    }
            return None, {
                "selectivity_candidate_count": 0,
                "selectivity_profile_count": int(profile_count),
                "selectivity_extract_stop_reason": "no_refine_group",
            }

        while True:
            compact_candidates = self.compact_candidates([partition for partition in current if str(partition.partition_id) in compact_ids], local_tombstones)
            metadata["compact_candidate_count"] = int(metadata.get("compact_candidate_count", 0) or 0) + int(len(compact_candidates))
            if compact_candidates:
                metadata["update_candidate_count"] = int(metadata.get("update_candidate_count", 0) or 0) + int(len(compact_candidates))
                candidate = min(
                    compact_candidates,
                    key=lambda item: (
                        -float(item.payload.get("tombstone_ratio", 0.0) or 0.0),
                        int(item.delta_memory),
                        tuple(str(value) for value in item.partition_ids),
                    ),
                )
                accepted.append(candidate)
                memory_now += int(candidate.delta_memory)
                current, _rewritten = _apply_candidates(current, [candidate])
                for partition_id in candidate.partition_ids:
                    local_tombstones[str(partition_id)] = 0
                stop_reason = "compact_candidate_pending"
                continue

            selectivity_candidate, selectivity_metadata = best_selectivity_candidate()
            if selectivity_metadata.get("selectivity_extract_stop_reason"):
                last_selectivity_stop_reason = str(selectivity_metadata["selectivity_extract_stop_reason"])
            metadata["selectivity_candidate_count"] = int(metadata.get("selectivity_candidate_count", 0) or 0) + int(selectivity_metadata.get("selectivity_candidate_count", 0) or 0)
            metadata["selectivity_profile_count"] = int(metadata.get("selectivity_profile_count", 0) or 0) + int(selectivity_metadata.get("selectivity_profile_count", 0) or 0)
            if (
                selectivity_candidate is None
                or float(selectivity_candidate.delta_latency) >= 0.0
                or int(memory_now) + int(selectivity_candidate.delta_memory) > int(self.memory_budget)
            ):
                stop_reason = "no_feasible_query_candidate"
                break
            candidate = selectivity_candidate
            candidate = _normalize_candidate_new_partition_ids(candidate, used_ids=used_ids)
            accepted.append(candidate)
            memory_now += int(candidate.delta_memory)
            total_delta_latency += float(candidate.delta_latency)
            if str(candidate.op_type) in {"selectivity_split", "replicated_selectivity_split"}:
                last_partition_id = str(candidate.partition_ids[0]) if candidate.partition_ids else None
                last_worst_tenant = int(candidate.payload.get("worst_tenant", -1))
                last_avg_selectivity = float(candidate.payload.get("avg_selectivity", -1.0))
                last_worst_selectivity = float(candidate.payload.get("worst_selectivity", -1.0))
            current, _rewritten = _apply_candidates(current, [candidate])
            for partition_id in candidate.partition_ids:
                local_tombstones.pop(str(partition_id), None)
            for new_partition in candidate.payload.get("new_partitions", []) or []:
                local_tombstones[str(new_partition.partition_id)] = 0
            stop_reason = "query_candidate_pending"
        metadata.update(
            {
                "accepted_operation_count": int(len(accepted)),
                "selectivity_extract_count": int(sum(1 for candidate in accepted if str(candidate.op_type) in {"selectivity_split", "replicated_selectivity_split"})),
                "selectivity_extract_cost_delta": float(total_delta_latency),
                "query_optimization_stop_reason": str(stop_reason),
                "selectivity_extract_stop_reason": str(last_selectivity_stop_reason or stop_reason),
                "selectivity_extract_last_group_id": "" if last_partition_id is None else str(last_partition_id),
                "selectivity_extract_last_worst_tenant": -1 if last_worst_tenant is None else int(last_worst_tenant),
                "selectivity_extract_last_avg_selectivity": -1.0 if last_avg_selectivity is None else float(last_avg_selectivity),
                "selectivity_extract_last_worst_selectivity": -1.0 if last_worst_selectivity is None else float(last_worst_selectivity),
                "under_budget_pair_operation_rule": "disabled_replicated_selectivity_only",
                "selectivity_only_query_refinement": 1,
                "local_maintenance_stop_reason_code": {
                    "no_refine_group": 3,
                    "worst_group_pure": 4,
                    "no_extractable_worst_tenant_bits": 5,
                    "invalid_extract_specs": 6,
                    "worst_group_not_beneficial": 7,
                    "invalid_replicated_split_specs": 12,
                    "replicated_split_not_beneficial": 13,
                    "replicated_refine_candidate_pending": 14,
                    "query_candidate_pending": 8,
                    "no_feasible_query_candidate": 9,
                    "max_operations": 10,
                    "compact_candidate_pending": 11,
                }.get(str(stop_reason), 0),
            }
        )
        return accepted, metadata

    def _choose_next_candidate(
        self,
        candidates: list[KMeansMaintenanceCandidate],
        *,
        memory_now: int,
    ) -> KMeansMaintenanceCandidate | None:
        compact_candidates = [candidate for candidate in candidates if candidate.op_type == "compact"]
        if compact_candidates:
            return min(
                compact_candidates,
                key=lambda item: (
                    -float(item.payload.get("tombstone_ratio", 0.0) or 0.0),
                    int(item.delta_memory),
                    tuple(str(value) for value in item.partition_ids),
                ),
            )
        if int(memory_now) <= int(self.memory_budget):
            feasible = [
                candidate
                for candidate in candidates
                if float(candidate.delta_latency) < 0.0
                and int(memory_now) + int(candidate.delta_memory) <= int(self.memory_budget)
            ]
            if not feasible:
                return None
            return min(
                feasible,
                key=lambda item: (
                    float(item.delta_latency),
                    int(item.delta_memory),
                    str(item.op_type),
                    tuple(str(value) for value in item.partition_ids),
                ),
            )
        feasible = [candidate for candidate in candidates if int(candidate.delta_memory) < 0]
        if not feasible:
            return None
        return min(
            feasible,
            key=lambda item: (
                float(item.delta_latency) / float(max(1, abs(int(item.delta_memory)))),
                float(item.delta_latency),
                int(item.delta_memory),
                str(item.op_type),
                tuple(str(value) for value in item.partition_ids),
            ),
        )

    def plan_local_maintenance(
        self,
        partitions: list[KMeansPartition],
        patterns_by_id: dict[int, ACLPattern],
        tombstones_by_partition: dict[str, int],
        *,
        memory_now: int,
        top_d: int,
        used_partition_ids: set[str],
        optimization_partition_ids: set[str] | None = None,
        compact_partition_ids: set[str] | None = None,
    ) -> tuple[list[KMeansMaintenanceCandidate], dict[str, int]]:
        current = list(partitions)
        optimization_ids = {str(value) for value in (optimization_partition_ids or {str(partition.partition_id) for partition in current})}
        compact_ids = {str(value) for value in (compact_partition_ids or optimization_ids)}
        local_tombstones = {str(key): int(value) for key, value in tombstones_by_partition.items()}
        used_ids = set(str(value) for value in used_partition_ids)
        accepted: list[KMeansMaintenanceCandidate] = []
        metadata = Counter()
        stop_reason = "memory_budget_satisfied"
        memory_pair_state: _CoreStarPairState | None = None
        query_pair_state: _CoreStarPairState | None = None

        while int(memory_now) > int(self.memory_budget):
            compact_candidates = self.compact_candidates([partition for partition in current if str(partition.partition_id) in compact_ids], local_tombstones)
            selectivity_candidates: list[KMeansMaintenanceCandidate] = []
            metadata["compact_candidate_count"] += int(len(compact_candidates))
            metadata["selectivity_candidate_count"] += int(len(selectivity_candidates))
            pair_candidate = None
            if compact_candidates:
                compact_candidates.sort(
                    key=lambda item: (
                        -float(item.payload.get("tombstone_ratio", 0.0) or 0.0),
                        int(item.delta_memory),
                        tuple(str(value) for value in item.partition_ids),
                    )
                )
                metadata["update_candidate_count"] += int(len(compact_candidates))
                for candidate in compact_candidates:
                    if int(memory_now) <= int(self.memory_budget):
                        break
                    candidate = _normalize_candidate_new_partition_ids(candidate, used_ids=used_ids)
                    accepted.append(candidate)
                    memory_now += int(candidate.delta_memory)
                    current, _rewritten = _apply_candidates(current, [candidate])
                    for partition_id in candidate.partition_ids:
                        local_tombstones[str(partition_id)] = 0
                    replacement_partitions = list(candidate.payload.get("new_partitions", []) or [])
                    if memory_pair_state is not None:
                        memory_pair_state.apply_candidate(candidate, replacement_partitions=replacement_partitions)
                    if query_pair_state is not None:
                        query_pair_state.apply_candidate(candidate, replacement_partitions=replacement_partitions)
                continue
            elif int(memory_now) > int(self.memory_budget):
                if memory_pair_state is None:
                    memory_pair_state = _CoreStarPairState(
                        planner=self,
                        partitions=[partition for partition in current if str(partition.partition_id) in optimization_ids],
                        patterns_by_id=patterns_by_id,
                        tombstones_by_partition=local_tombstones,
                        top_d=int(top_d),
                        allowed_operations=None,
                        mode="memory",
                    )
                pair_candidate = memory_pair_state.pop_best(memory_now=int(memory_now))
            if not compact_candidates:
                candidates = []
                if pair_candidate is not None:
                    candidates.append(pair_candidate)
            metadata["update_candidate_count"] += int(len(compact_candidates) + len(selectivity_candidates))
            candidate = self._choose_next_candidate(candidates, memory_now=int(memory_now))
            if candidate is None:
                stop_reason = "no_feasible_candidate"
                break
            if pair_candidate is not None and candidate is not pair_candidate:
                if memory_pair_state is not None and int(memory_now) > int(self.memory_budget):
                    memory_pair_state.return_candidate(pair_candidate)
                elif query_pair_state is not None:
                    query_pair_state.return_candidate(pair_candidate)
            candidate = _normalize_candidate_new_partition_ids(candidate, used_ids=used_ids)
            accepted.append(candidate)
            memory_now += int(candidate.delta_memory)
            current, _rewritten = _apply_candidates(current, [candidate])
            if candidate.op_type == "compact":
                for partition_id in candidate.partition_ids:
                    local_tombstones[str(partition_id)] = 0
            else:
                for partition_id in candidate.partition_ids:
                    local_tombstones.pop(str(partition_id), None)
                for partition in candidate.payload.get("new_partitions", []) or []:
                    local_tombstones[str(partition.partition_id)] = 0
            replacement_partitions = list(candidate.payload.get("new_partitions", []) or [])
            if memory_pair_state is not None:
                memory_pair_state.apply_candidate(candidate, replacement_partitions=replacement_partitions)
            if query_pair_state is not None:
                query_pair_state.apply_candidate(candidate, replacement_partitions=replacement_partitions)
        else:
            stop_reason = "memory_budget_satisfied"

        pair_metadata = Counter()
        for state in (memory_pair_state, query_pair_state):
            if state is None:
                continue
            for key, value in state.metadata().items():
                pair_metadata[str(key)] += int(value)
        for key, value in pair_metadata.items():
            metadata[str(key)] += int(value)
        metadata["pair_candidate_count"] = int(pair_metadata.get("pair_candidate_count", 0))
        metadata["update_candidate_count"] += int(pair_metadata.get("pair_candidate_count", 0))
        metadata["accepted_operation_count"] = int(len(accepted))
        metadata["max_operations"] = int(self.max_operations)
        metadata["memory_budget_satisfied"] = int(int(memory_now) <= int(self.memory_budget))
        metadata["memory_after_selection"] = int(memory_now)
        metadata["max_operations_ignored_until_budget"] = 1
        metadata["local_maintenance_stop_reason_code"] = {
            "max_operations": 1,
            "no_feasible_candidate": 2,
            "memory_budget_satisfied_and_max_operations": 3,
            "memory_budget_satisfied": 4,
        }.get(stop_reason, 0)
        return accepted, {str(key): int(value) for key, value in metadata.items()}


class _CoreStarPairState:
    def __init__(
        self,
        *,
        planner: _LocalPlanner,
        partitions: list[KMeansPartition],
        patterns_by_id: dict[int, ACLPattern],
        tombstones_by_partition: dict[str, int],
        top_d: int,
        allowed_operations: Iterable[str] | None,
        mode: str,
    ) -> None:
        self.planner = planner
        self.patterns_by_id = patterns_by_id
        self.bit_context = planner._bit_context(patterns_by_id)
        self.tombstones_by_partition = {str(key): int(value) for key, value in tombstones_by_partition.items()}
        self.top_d = max(1, int(top_d))
        self.allowed_operations = None if allowed_operations is None else tuple(str(value) for value in allowed_operations)
        self.mode = str(mode)
        self.partitions_by_id = {str(partition.partition_id): partition for partition in partitions}
        self.versions: Counter[str] = Counter({str(partition.partition_id): 0 for partition in partitions})
        self.candidate_cache: dict[tuple[str, str, int, int], KMeansMaintenanceCandidate | None] = {}
        self.candidate_cache_partitions: dict[str, set[tuple[str, str, int, int]]] = defaultdict(set)
        self.partition_cost_cache: dict[tuple[str, int], float] = {}
        self.partition_bit_state_cache: dict[tuple[str, int], tuple[int, dict[int, int]]] = {}
        self.owners_by_pattern: dict[int, set[str]] = defaultdict(set)
        self.pattern_star_edges: dict[int, dict[tuple[str, str], float]] = defaultdict(dict)
        self.edge_refcounts: Counter[tuple[str, str]] = Counter()
        self.edge_signal_scores: dict[tuple[str, str], float] = defaultdict(float)
        self.adjacency: dict[str, set[str]] = defaultdict(set)
        self.heap: list[tuple[object, ...]] = []
        self.stats: Counter[str] = Counter()
        self.operation_rank = {"selectivity_split": 0, "split_overlap": 1, "merge_extract_overlap": 2, "move_left": 3, "move_right": 4, "full": 5}
        for partition in partitions:
            self._register_partition(partition)
        self._build_initial_heap()

    def metadata(self) -> dict[str, int]:
        result = {
            "core_star_candidate_edges": int(self.stats.get("core_star_candidate_edges", 0)),
            "core_star_heap_entries": int(self.stats.get("core_star_heap_entries", 0)),
            "core_star_cache_hits": int(self.stats.get("core_star_cache_hits", 0)),
            "core_star_cache_misses": int(self.stats.get("core_star_cache_misses", 0)),
            "core_star_owner_patterns": int(self.stats.get("core_star_owner_patterns", 0)),
            "core_star_stale_pops": int(self.stats.get("core_star_stale_pops", 0)),
            "core_star_local_refresh_edges": int(self.stats.get("core_star_local_refresh_edges", 0)),
            "pair_candidate_count": int(self.stats.get("pair_candidate_count", 0)),
            "core_star_cache_prunes": int(self.stats.get("core_star_cache_prunes", 0)),
            "partition_cost_cache_hits": int(self.stats.get("partition_cost_cache_hits", 0)),
            "partition_cost_cache_misses": int(self.stats.get("partition_cost_cache_misses", 0)),
            "edge_refcount_entries": int(len(self.edge_refcounts)),
            "adjacency_entries": int(sum(len(values) for values in self.adjacency.values())),
            "pattern_star_edge_entries": int(sum(len(values) for values in self.pattern_star_edges.values())),
        }
        return result

    def _edge_key(self, left_id: str, right_id: str) -> tuple[str, str]:
        left_id = str(left_id)
        right_id = str(right_id)
        return (left_id, right_id) if left_id < right_id else (right_id, left_id)

    def _register_partition(self, partition: KMeansPartition) -> None:
        partition_id = str(partition.partition_id)
        for pattern_id in partition.pattern_ids:
            if int(pattern_id) in self.patterns_by_id:
                self.owners_by_pattern[int(pattern_id)].add(partition_id)

    def _unregister_partition(self, partition: KMeansPartition) -> None:
        partition_id = str(partition.partition_id)
        for pattern_id in partition.pattern_ids:
            owners = self.owners_by_pattern.get(int(pattern_id))
            if owners is None:
                continue
            owners.discard(partition_id)
            if not owners:
                self.owners_by_pattern.pop(int(pattern_id), None)

    def _prune_candidate_cache(self, partition_ids: set[str]) -> None:
        removed = 0
        for partition_id in sorted(str(value) for value in partition_ids):
            cache_keys = self.candidate_cache_partitions.pop(str(partition_id), set())
            for cache_key in list(cache_keys):
                if cache_key in self.candidate_cache:
                    self.candidate_cache.pop(cache_key, None)
                    removed += 1
                other_partition_id = str(cache_key[1]) if str(cache_key[0]) == str(partition_id) else str(cache_key[0])
                other_keys = self.candidate_cache_partitions.get(other_partition_id)
                if other_keys is not None:
                    other_keys.discard(cache_key)
                    if not other_keys:
                        self.candidate_cache_partitions.pop(other_partition_id, None)
            self.partition_cost_cache.pop((str(partition_id), int(self.versions[str(partition_id)])), None)
            self.partition_bit_state_cache.pop((str(partition_id), int(self.versions[str(partition_id)])), None)
        self.stats["core_star_cache_prunes"] += int(removed)

    def _store_candidate_cache(self, cache_key: tuple[str, str, int, int], candidate: KMeansMaintenanceCandidate | None) -> None:
        self.candidate_cache[cache_key] = candidate
        self.candidate_cache_partitions[str(cache_key[0])].add(cache_key)
        self.candidate_cache_partitions[str(cache_key[1])].add(cache_key)

    def _partition_cost(self, partition: KMeansPartition) -> float:
        partition_id = str(partition.partition_id)
        cache_key = (partition_id, int(self.versions[partition_id]))
        cached = self.partition_cost_cache.get(cache_key)
        if cached is not None:
            self.stats["partition_cost_cache_hits"] += 1
            return float(cached)
        self.stats["partition_cost_cache_misses"] += 1
        pattern_bits, tenant_bits = self._partition_bit_state(partition)
        value = float(self._cost_from_bits(int(pattern_bits), tenant_bits))
        self.partition_cost_cache[cache_key] = float(value)
        return float(value)

    def _partition_bit_state(self, partition: KMeansPartition) -> tuple[int, dict[int, int]]:
        partition_id = str(partition.partition_id)
        cache_key = (partition_id, int(self.versions[partition_id]))
        cached = self.partition_bit_state_cache.get(cache_key)
        if cached is not None:
            return int(cached[0]), dict(cached[1])
        pattern_bits = self.bit_context.pattern_bits_for(partition.pattern_ids)
        tenant_bits = self.bit_context.tenant_bits_for_partition(partition)
        tenant_bits = {
            int(tenant_id): int(bits) & int(pattern_bits)
            for tenant_id, bits in tenant_bits.items()
            if int(bits) & int(pattern_bits)
        }
        self.partition_bit_state_cache[cache_key] = (int(pattern_bits), dict(tenant_bits))
        return int(pattern_bits), dict(tenant_bits)

    def _cost_from_bits(self, pattern_bits: int, tenant_bits: dict[int, int]) -> float:
        pattern_bits = int(pattern_bits)
        partition_vectors = self.bit_context.vector_count_bits(pattern_bits)
        if partition_vectors <= 0:
            return 0.0
        total = 0.0
        for _tenant_id, bits in tenant_bits.items():
            accessible = self.bit_context.vector_count_bits(int(bits) & int(pattern_bits))
            if accessible <= 0:
                continue
            total += estimate_partition_query_cost(
                partition_vectors=int(partition_vectors),
                accessible_vectors=int(accessible),
                tenant_weight=1.0,
                ef_search=int(self.planner.ef_search),
                use_adaptive_ef=True,
            )
        return float(total)

    def _pattern_core_key(self, pattern_id: int, partition_id: str) -> tuple[int, float, int, int, str]:
        partition = self.partitions_by_id[str(partition_id)]
        pattern_vectors = max(0, int(self.patterns_by_id[int(pattern_id)].vector_count))
        partition_vectors = max(1, int(partition.vector_count))
        acl_share = float(pattern_vectors) / float(partition_vectors)
        return (
            int(len(partition.pattern_ids)),
            -float(acl_share),
            int(partition_vectors),
            int(len(partition.tenant_ids)),
            str(partition_id),
        )

    def _owners_by_pattern(self) -> dict[int, set[str]]:
        return {int(pattern_id): set(owner_ids) for pattern_id, owner_ids in self.owners_by_pattern.items()}

    def _selected_edges_from_owners(self, owners_by_pattern: dict[int, set[str]]) -> set[tuple[str, str]]:
        edge_acl_counts: Counter[tuple[str, str]] = Counter()
        edge_signal_scores: dict[tuple[str, str], float] = defaultdict(float)
        for pattern_id, owner_ids in owners_by_pattern.items():
            live_owner_ids = sorted(str(owner_id) for owner_id in owner_ids if str(owner_id) in self.partitions_by_id)
            if len(live_owner_ids) <= 1:
                continue
            core_id = min(live_owner_ids, key=lambda owner_id: self._pattern_core_key(int(pattern_id), str(owner_id)))
            signal = float(self.patterns_by_id[int(pattern_id)].vector_count)
            if signal <= 0.0:
                continue
            for owner_id in live_owner_ids:
                if str(owner_id) == str(core_id):
                    continue
                edge = tuple(sorted((str(core_id), str(owner_id))))
                edge_acl_counts[edge] += 1
                edge_signal_scores[edge] += float(signal)

        incident_edges: dict[str, list[tuple[float, float, tuple[str, str]]]] = defaultdict(list)
        for edge, shared_acl_count in edge_acl_counts.items():
            left, right = edge
            if left not in self.partitions_by_id or right not in self.partitions_by_id:
                continue
            left_acl_count = max(1, len(self.partitions_by_id[left].pattern_ids))
            right_acl_count = max(1, len(self.partitions_by_id[right].pattern_ids))
            rank_score = float(shared_acl_count) / math.sqrt(float(left_acl_count) * float(right_acl_count))
            signal = float(edge_signal_scores.get(edge, 0.0))
            incident_edges[left].append((float(rank_score), float(signal), edge))
            incident_edges[right].append((float(rank_score), float(signal), edge))

        selected_edges: set[tuple[str, str]] = set()
        for edges in incident_edges.values():
            edges.sort(key=lambda item: (-float(item[0]), -float(item[1]), item[2][0], item[2][1]))
            for _rank_score, _signal, edge in edges[: self.top_d]:
                selected_edges.add(edge)
        return selected_edges

    def _selected_star_edges_by_pattern(self, pattern_ids: set[int]) -> dict[int, dict[tuple[str, str], float]]:
        raw_edges_by_pattern: dict[int, dict[tuple[str, str], float]] = {}
        edge_acl_counts: Counter[tuple[str, str]] = Counter()
        edge_signal_scores: dict[tuple[str, str], float] = defaultdict(float)
        target_partition_ids: set[str] = set()
        for pattern_id in sorted(int(value) for value in pattern_ids):
            if int(pattern_id) not in self.patterns_by_id:
                continue
            owners = {str(owner_id) for owner_id in self.owners_by_pattern.get(int(pattern_id), set()) if str(owner_id) in self.partitions_by_id}
            if len(owners) <= 1:
                continue
            core_id = min(sorted(owners), key=lambda owner_id: self._pattern_core_key(int(pattern_id), str(owner_id)))
            signal = float(self.patterns_by_id[int(pattern_id)].vector_count)
            if signal <= 0.0:
                continue
            pattern_edges: dict[tuple[str, str], float] = {}
            for owner_id in sorted(owners):
                if str(owner_id) == str(core_id):
                    continue
                edge = self._edge_key(str(core_id), str(owner_id))
                pattern_edges[edge] = float(signal)
                edge_acl_counts[edge] += 1
                edge_signal_scores[edge] += float(signal)
                target_partition_ids.add(str(edge[0]))
                target_partition_ids.add(str(edge[1]))
            if pattern_edges:
                raw_edges_by_pattern[int(pattern_id)] = pattern_edges

        incident_edges: dict[str, list[tuple[float, float, tuple[str, str]]]] = defaultdict(list)
        for edge, shared_acl_count in edge_acl_counts.items():
            left, right = edge
            if left not in self.partitions_by_id or right not in self.partitions_by_id:
                continue
            left_acl_count = max(1, len(self.partitions_by_id[left].pattern_ids))
            right_acl_count = max(1, len(self.partitions_by_id[right].pattern_ids))
            rank_score = float(shared_acl_count) / math.sqrt(float(left_acl_count) * float(right_acl_count))
            signal = float(edge_signal_scores.get(edge, 0.0))
            incident_edges[left].append((float(rank_score), float(signal), edge))
            incident_edges[right].append((float(rank_score), float(signal), edge))

        selected_edges: set[tuple[str, str]] = set()
        for partition_id in sorted(target_partition_ids):
            edges = incident_edges.get(str(partition_id), [])
            edges.sort(key=lambda item: (-float(item[0]), -float(item[1]), item[2][0], item[2][1]))
            for _rank_score, _signal, edge in edges[: self.top_d]:
                selected_edges.add(edge)

        return {
            int(pattern_id): {edge: float(signal) for edge, signal in edges.items() if edge in selected_edges}
            for pattern_id, edges in raw_edges_by_pattern.items()
            if any(edge in selected_edges for edge in edges)
        }

    def _add_edge_reference(self, edge: tuple[str, str], signal: float) -> None:
        edge = self._edge_key(edge[0], edge[1])
        if edge[0] == edge[1] or edge[0] not in self.partitions_by_id or edge[1] not in self.partitions_by_id:
            return
        was_inactive = int(self.edge_refcounts.get(edge, 0)) <= 0
        self.edge_refcounts[edge] += 1
        self.edge_signal_scores[edge] = float(self.edge_signal_scores.get(edge, 0.0)) + float(signal)
        if was_inactive:
            self.adjacency[edge[0]].add(edge[1])
            self.adjacency[edge[1]].add(edge[0])
            self._push_edge(edge)

    def _remove_edge_reference(self, edge: tuple[str, str], signal: float) -> None:
        edge = self._edge_key(edge[0], edge[1])
        current = int(self.edge_refcounts.get(edge, 0))
        if current <= 1:
            self.edge_refcounts.pop(edge, None)
            self.edge_signal_scores.pop(edge, None)
            self.adjacency[edge[0]].discard(edge[1])
            self.adjacency[edge[1]].discard(edge[0])
            return
        self.edge_refcounts[edge] = int(current - 1)
        self.edge_signal_scores[edge] = max(0.0, float(self.edge_signal_scores.get(edge, 0.0)) - float(signal))

    def _build_initial_heap(self) -> None:
        if len(self.partitions_by_id) <= 1:
            return
        owners_by_pattern = self._owners_by_pattern()
        self.stats["core_star_owner_patterns"] += int(sum(1 for owners in owners_by_pattern.values() if len(owners) > 1))
        self._refresh_edges_for_patterns(set(owners_by_pattern))
        self.stats["core_star_candidate_edges"] += int(len(self.edge_refcounts))

    def _candidate_for_edge(self, edge: tuple[str, str]) -> KMeansMaintenanceCandidate | None:
        left_id, right_id = tuple(sorted((str(edge[0]), str(edge[1]))))
        left = self.partitions_by_id.get(left_id)
        right = self.partitions_by_id.get(right_id)
        if left is None or right is None:
            return None
        cache_key = (left_id, right_id, int(self.versions[left_id]), int(self.versions[right_id]))
        if cache_key in self.candidate_cache:
            self.stats["core_star_cache_hits"] += 1
            return self.candidate_cache[cache_key]
        self.stats["core_star_cache_misses"] += 1
        candidate = self._best_pair_candidate(left, right)
        self._store_candidate_cache(cache_key, candidate)
        if candidate is not None:
            self.stats["pair_candidate_count"] += 1
        return candidate

    def _best_pair_candidate(self, left: KMeansPartition, right: KMeansPartition) -> KMeansMaintenanceCandidate | None:
        before_cost = self._partition_cost(left) + self._partition_cost(right)
        before_memory = int(left.vector_count) + int(right.vector_count)
        before_actual_memory = (
            int(before_memory)
            + int(self.tombstones_by_partition.get(str(left.partition_id), 0))
            + int(self.tombstones_by_partition.get(str(right.partition_id), 0))
        )
        left_pattern_bits, left_tenant_bits = self._partition_bit_state(left)
        right_pattern_bits, right_tenant_bits = self._partition_bit_state(right)
        if int(left_pattern_bits) & int(right_pattern_bits) == 0:
            return None

        def spec_cost_memory(pattern_bits: int, tenant_bits: dict[int, int]) -> tuple[float, int, bool]:
            pattern_bits = int(pattern_bits)
            if pattern_bits == 0:
                return 0.0, 0, False
            partition_vectors = self.bit_context.vector_count_bits(pattern_bits)
            if partition_vectors <= 0:
                return 0.0, 0, False
            total = 0.0
            live_tenant = False
            for tenant_id, bits in tenant_bits.items():
                accessible = self.bit_context.vector_count_bits(int(bits) & int(pattern_bits))
                if accessible <= 0:
                    continue
                live_tenant = True
                total += estimate_partition_query_cost(
                    partition_vectors=int(partition_vectors),
                    accessible_vectors=int(accessible),
                    tenant_weight=1.0,
                    ef_search=int(self.planner.ef_search),
                    use_adaptive_ef=True,
                )
            return float(total), int(partition_vectors), bool(live_tenant)

        best: tuple[tuple[float, int, int], str, float, int, int, list[tuple[int, dict[int, int]]]] | None = None
        operations = tuple(str(value) for value in (self.allowed_operations or ("full", "move_left", "move_right", "split_overlap", "merge_extract_overlap")))
        for operation in operations:
            specs = self.planner._operation_bit_specs(
                int(left_pattern_bits),
                left_tenant_bits,
                int(right_pattern_bits),
                right_tenant_bits,
                self.bit_context,
                operation,
            )
            after_cost = 0.0
            after_memory = 0
            max_partition_size = 0
            live_count = 0
            for pattern_bits, tenant_bits in specs:
                spec_cost, spec_memory, live = spec_cost_memory(int(pattern_bits), tenant_bits)
                if live:
                    after_cost += float(spec_cost)
                    after_memory += int(spec_memory)
                    max_partition_size = max(int(max_partition_size), int(spec_memory))
                    live_count += 1
            if live_count <= 0:
                continue
            memory_saved = int(before_actual_memory) - int(after_memory)
            if memory_saved <= 0:
                continue
            delta_latency = float(after_cost) - float(before_cost)
            rank = (float(delta_latency), int(max_partition_size), int(self.operation_rank.get(operation, 999)))
            if best is None or rank < best[0]:
                best = (rank, str(operation), float(delta_latency), int(memory_saved), int(max_partition_size), specs)
        if best is None:
            return None
        rank, operation, delta_latency, memory_saved, max_partition_size, specs = best
        new_partitions: list[KMeansPartition] = []
        for index, (pattern_bits, tenant_bits) in enumerate(specs):
            candidate_partition = _make_partition(
                f"{left.partition_id}__{right.partition_id}__{operation}_{index}",
                cluster_id=int(left.cluster_id),
                pattern_ids=self.bit_context.pattern_ids_from_bits(int(pattern_bits)),
                tenant_patterns=self.bit_context.tenant_patterns_from_bits(tenant_bits),
                patterns_by_id=self.patterns_by_id,
            )
            if candidate_partition is not None:
                new_partitions.append(candidate_partition)
        if not new_partitions:
            return None
        payload = {
            "operation": operation,
            "new_partitions": new_partitions,
            "op_rank": int(rank[2]),
            "max_result_partition_size": int(max_partition_size),
        }
        return KMeansMaintenanceCandidate(
            op_type=str(operation),
            partition_ids=(str(left.partition_id), str(right.partition_id)),
            delta_memory=-int(memory_saved),
            delta_latency=float(delta_latency),
            payload=payload,
        )

    def _rank(self, candidate: KMeansMaintenanceCandidate) -> tuple[object, ...] | None:
        memory_saved = abs(int(candidate.delta_memory))
        op_rank = int(self.operation_rank.get(str(candidate.op_type), 999))
        ids = tuple(str(value) for value in candidate.partition_ids)
        if self.mode == "memory":
            if int(candidate.delta_memory) >= 0:
                return None
            if float(candidate.delta_latency) <= 0.0:
                latency_class = 0
                heap_score = -float(memory_saved)
            else:
                latency_class = 1
                heap_score = float(candidate.delta_latency) / float(max(1, memory_saved))
            return (
                int(latency_class),
                float(heap_score),
                float(candidate.delta_latency),
                int(candidate.payload.get("max_result_partition_size", 0) or 0),
                -int(memory_saved),
                int(op_rank),
                int(candidate.delta_memory),
                str(candidate.op_type),
                ids,
            )
        if self.mode == "query":
            if str(candidate.op_type) == "full" or float(candidate.delta_latency) >= 0.0:
                return None
            return (
                float(candidate.delta_latency),
                int(op_rank),
                int(candidate.delta_memory),
                ids,
            )
        if self.mode == "query_all":
            if float(candidate.delta_latency) >= 0.0:
                return None
            return (
                float(candidate.delta_latency),
                int(candidate.delta_memory),
                str(candidate.op_type),
                ids,
            )
        raise ValueError(f"unknown core-star heap mode: {self.mode}")

    def _push_edge(self, edge: tuple[str, str]) -> None:
        left_id, right_id = tuple(sorted((str(edge[0]), str(edge[1]))))
        if left_id not in self.partitions_by_id or right_id not in self.partitions_by_id:
            return
        candidate = self._candidate_for_edge((left_id, right_id))
        if candidate is None:
            return
        rank = self._rank(candidate)
        if rank is None:
            return
        heapq.heappush(
            self.heap,
            (
                *rank,
                str(left_id),
                str(right_id),
                int(self.versions[left_id]),
                int(self.versions[right_id]),
                candidate,
            ),
        )
        self.stats["core_star_heap_entries"] += 1

    def return_candidate(self, candidate: KMeansMaintenanceCandidate) -> None:
        if len(candidate.partition_ids) != 2:
            return
        self._push_edge((str(candidate.partition_ids[0]), str(candidate.partition_ids[1])))

    def pop_best(self, *, memory_now: int) -> KMeansMaintenanceCandidate | None:
        while self.heap:
            entry = heapq.heappop(self.heap)
            left_id = str(entry[-5])
            right_id = str(entry[-4])
            left_version = int(entry[-3])
            right_version = int(entry[-2])
            candidate = entry[-1]
            if (
                left_id not in self.partitions_by_id
                or right_id not in self.partitions_by_id
                or int(self.versions[left_id]) != int(left_version)
                or int(self.versions[right_id]) != int(right_version)
            ):
                self.stats["core_star_stale_pops"] += 1
                continue
            if not isinstance(candidate, KMeansMaintenanceCandidate):
                self.stats["core_star_stale_pops"] += 1
                continue
            if self.mode in {"query", "query_all"} and int(memory_now) + int(candidate.delta_memory) > int(self.planner.memory_budget):
                self.stats["core_star_stale_pops"] += 1
                continue
            if right_id not in self.adjacency.get(left_id, set()):
                self.stats["core_star_stale_pops"] += 1
                continue
            if not (set(self.partitions_by_id[left_id].pattern_ids) & set(self.partitions_by_id[right_id].pattern_ids)):
                self.stats["core_star_stale_pops"] += 1
                continue
            return candidate
        return None

    def _refresh_edges_for_patterns(self, pattern_ids: set[int]) -> None:
        if not pattern_ids or len(self.partitions_by_id) <= 1:
            return
        normalized_pattern_ids = {int(pattern_id) for pattern_id in pattern_ids if int(pattern_id) in self.patterns_by_id}
        if not normalized_pattern_ids:
            return
        new_edges_by_pattern = self._selected_star_edges_by_pattern(normalized_pattern_ids)
        refreshed_edges: set[tuple[str, str]] = set()
        for pattern_id in sorted(normalized_pattern_ids):
            old_edges = dict(self.pattern_star_edges.get(int(pattern_id), {}))
            new_edges = dict(new_edges_by_pattern.get(int(pattern_id), {}))
            if old_edges == new_edges:
                continue
            for edge, signal in old_edges.items():
                if edge not in new_edges:
                    self._remove_edge_reference(edge, float(signal))
                    refreshed_edges.add(edge)
            for edge, signal in new_edges.items():
                if edge not in old_edges:
                    self._add_edge_reference(edge, float(signal))
                    refreshed_edges.add(edge)
                else:
                    old_signal = float(old_edges.get(edge, 0.0))
                    if abs(float(signal) - old_signal) > 1e-12:
                        self.edge_signal_scores[edge] = max(0.0, float(self.edge_signal_scores.get(edge, 0.0)) - old_signal + float(signal))
                        refreshed_edges.add(edge)
            if new_edges:
                self.pattern_star_edges[int(pattern_id)] = new_edges
            else:
                self.pattern_star_edges.pop(int(pattern_id), None)
        for edge in sorted(refreshed_edges):
            if edge[0] in self.partitions_by_id and edge[1] in self.partitions_by_id and edge[1] in self.adjacency.get(edge[0], set()):
                self._push_edge(edge)
        self.stats["core_star_local_refresh_edges"] += int(len(refreshed_edges))

    def apply_candidate(
        self,
        candidate: KMeansMaintenanceCandidate,
        *,
        replacement_partitions: list[KMeansPartition],
    ) -> None:
        if str(candidate.op_type) == "compact":
            touched_patterns: set[int] = set()
            touched_ids = {str(value) for value in candidate.partition_ids}
            self._prune_candidate_cache(touched_ids)
            for partition_id in candidate.partition_ids:
                partition_key = str(partition_id)
                self.tombstones_by_partition[partition_key] = 0
                self.versions[partition_key] += 1
                partition = self.partitions_by_id.get(partition_key)
                if partition is not None:
                    touched_patterns.update(int(pattern_id) for pattern_id in partition.pattern_ids)
            self._refresh_edges_for_patterns(touched_patterns)
            return

        touched_ids = {str(value) for value in candidate.partition_ids}
        touched_patterns: set[int] = set()
        self._prune_candidate_cache(touched_ids)
        for partition_id in touched_ids:
            partition = self.partitions_by_id.pop(str(partition_id), None)
            self.versions[str(partition_id)] += 1
            self.tombstones_by_partition.pop(str(partition_id), None)
            if partition is not None:
                self._unregister_partition(partition)
                touched_patterns.update(int(pattern_id) for pattern_id in partition.pattern_ids)

        for partition in replacement_partitions:
            partition_id = str(partition.partition_id)
            self.partitions_by_id[partition_id] = partition
            self.versions.setdefault(partition_id, 0)
            self.tombstones_by_partition[partition_id] = 0
            self._register_partition(partition)
            touched_patterns.update(int(pattern_id) for pattern_id in partition.pattern_ids)

        self._refresh_edges_for_patterns(touched_patterns)


def _refresh_patterns(repository: KMeansUpdateRepository, old_patterns: list[ACLPattern]) -> list[ACLPattern]:
    old_by_acl = {tuple(pattern.tenant_ids): int(pattern.pattern_id) for pattern in old_patterns}
    old_by_id = {int(pattern.pattern_id): pattern for pattern in old_patterns}
    next_pattern_id = max(old_by_id or {0: None}) + 1
    groups = repository.fetch_all_acl_groups()
    total_tenant_count = max(1, len({int(tenant_id) for tenant_ids in groups for tenant_id in tenant_ids}))
    result = []
    for tenant_ids, (document_ids, document_count, vector_count) in sorted(groups.items(), key=lambda item: old_by_acl.get(item[0], 10**12)):
        pattern_id = old_by_acl.get(tuple(tenant_ids))
        if pattern_id is None:
            pattern_id = int(next_pattern_id)
            next_pattern_id += 1
        acl_tenant_count = max(1, len(tuple(tenant_ids)))
        result.append(
            ACLPattern(
                pattern_id=int(pattern_id),
                tenant_ids=tuple(int(value) for value in tenant_ids),
                document_ids=tuple(int(value) for value in document_ids),
                vector_count=int(vector_count),
                document_count=int(document_count),
                weight=_pattern_weight(int(vector_count), int(acl_tenant_count), int(total_tenant_count)),
                score=_pattern_score(int(vector_count), int(acl_tenant_count)),
                zone="private",
            )
        )
    return result


def _document_pattern_map(patterns: Iterable[ACLPattern]) -> dict[int, int]:
    result = {}
    for pattern in patterns:
        for document_id in pattern.document_ids:
            result[int(document_id)] = int(pattern.pattern_id)
    return result


def _core_star_top_d_edges(
    partitions: list[KMeansPartition],
    patterns_by_id: dict[int, ACLPattern],
    *,
    top_d: int,
) -> tuple[set[tuple[str, str]], dict[str, int]]:
    partitions_by_id = {str(partition.partition_id): partition for partition in partitions}
    if len(partitions_by_id) <= 1:
        return set(), {
            "affected_core_star_edges": 0,
            "affected_core_star_owner_patterns": 0,
        }

    owners_by_pattern: dict[int, set[str]] = defaultdict(set)
    for partition in partitions:
        for pattern_id in partition.pattern_ids:
            if int(pattern_id) in patterns_by_id:
                owners_by_pattern[int(pattern_id)].add(str(partition.partition_id))

    def pattern_core_key(pattern_id: int, partition_id: str) -> tuple[int, float, int, int, str]:
        partition = partitions_by_id[str(partition_id)]
        pattern_vectors = max(0, int(patterns_by_id[int(pattern_id)].vector_count))
        partition_vectors = max(1, int(partition.vector_count))
        acl_share = float(pattern_vectors) / float(partition_vectors)
        return (
            int(len(partition.pattern_ids)),
            -float(acl_share),
            int(partition_vectors),
            int(len(partition.tenant_ids)),
            str(partition_id),
        )

    edge_acl_counts: Counter[tuple[str, str]] = Counter()
    edge_signal_scores: dict[tuple[str, str], float] = defaultdict(float)
    for pattern_id, owner_ids in owners_by_pattern.items():
        live_owner_ids = sorted(str(owner_id) for owner_id in owner_ids if str(owner_id) in partitions_by_id)
        if len(live_owner_ids) <= 1:
            continue
        core_id = min(live_owner_ids, key=lambda owner_id: pattern_core_key(int(pattern_id), str(owner_id)))
        signal = float(patterns_by_id[int(pattern_id)].vector_count)
        if signal <= 0.0:
            continue
        for owner_id in live_owner_ids:
            if str(owner_id) == str(core_id):
                continue
            edge = tuple(sorted((str(core_id), str(owner_id))))
            edge_acl_counts[edge] += 1
            edge_signal_scores[edge] += float(signal)

    incident_edges: dict[str, list[tuple[float, float, tuple[str, str]]]] = defaultdict(list)
    for edge, shared_acl_count in edge_acl_counts.items():
        left, right = edge
        if left not in partitions_by_id or right not in partitions_by_id:
            continue
        left_acl_count = max(1, len(partitions_by_id[left].pattern_ids))
        right_acl_count = max(1, len(partitions_by_id[right].pattern_ids))
        rank_score = float(shared_acl_count) / math.sqrt(float(left_acl_count) * float(right_acl_count))
        signal = float(edge_signal_scores.get(edge, 0.0))
        incident_edges[left].append((float(rank_score), float(signal), edge))
        incident_edges[right].append((float(rank_score), float(signal), edge))

    selected_edges: set[tuple[str, str]] = set()
    for edges in incident_edges.values():
        edges.sort(key=lambda item: (-float(item[0]), -float(item[1]), item[2][0], item[2][1]))
        for _rank_score, _signal, edge in edges[: max(1, int(top_d))]:
            selected_edges.add(edge)

    return selected_edges, {
        "affected_core_star_edges": int(len(selected_edges)),
        "affected_core_star_owner_patterns": int(sum(1 for owners in owners_by_pattern.values() if len(owners) > 1)),
    }


def _affected_region(
    partitions: list[KMeansPartition],
    patterns_by_id: dict[int, ACLPattern],
    direct_partition_ids: set[str],
    *,
    top_d: int,
) -> tuple[set[str], dict[str, int]]:
    partition_ids = {str(partition.partition_id) for partition in partitions}
    direct = {str(value) for value in direct_partition_ids if str(value) in partition_ids}
    affected = set(direct)
    selected_edges, metadata = _core_star_top_d_edges(partitions, patterns_by_id, top_d=int(top_d))
    for left, right in selected_edges:
        if str(left) in direct:
            affected.add(str(right))
        if str(right) in direct:
            affected.add(str(left))
    metadata.update(
        {
            "affected_direct_partition_count": int(len(direct)),
            "affected_partition_count": int(len(affected)),
        }
    )
    return affected, metadata


def _rebuild_partitions_with_patterns(
    old_partitions: list[KMeansPartition],
    patterns_by_id: dict[int, ACLPattern],
    *,
    changed_pattern_ids: set[int] | None = None,
) -> list[KMeansPartition]:
    result: list[KMeansPartition] = []
    changed_patterns = None if changed_pattern_ids is None else {int(pattern_id) for pattern_id in changed_pattern_ids}
    for partition in old_partitions:
        if changed_patterns is not None and not (set(map(int, partition.pattern_ids)) & changed_patterns):
            result.append(partition)
            continue
        tenant_patterns = _tenant_patterns_for_partition(partition, patterns_by_id)
        rebuilt = _make_partition(
            str(partition.partition_id),
            cluster_id=int(partition.cluster_id),
            pattern_ids=partition.pattern_ids,
            tenant_patterns=tenant_patterns,
            patterns_by_id=patterns_by_id,
        )
        if rebuilt is not None:
            result.append(rebuilt)
    return result


def _assign_new_patterns(
    partitions: list[KMeansPartition],
    new_pattern_ids: set[int],
    patterns_by_id: dict[int, ACLPattern],
    *,
    ef_search: int,
    memory_budget: int,
    max_partition_links_per_pattern: int = 2,
) -> tuple[list[KMeansPartition], set[str], dict[str, int | str]]:
    changed: set[str] = set()
    current = list(partitions)
    bit_context = _PatternBitsetContext(patterns_by_id)
    max_links = max(1, int(max_partition_links_per_pattern))
    metadata: dict[str, int | str] = {
        "new_pattern_assignment_policy": "capped_best_partitions",
        "new_pattern_assignment_max_partition_links": int(max_links),
        "new_pattern_assignment_count": int(len(new_pattern_ids)),
        "new_pattern_assignment_total_tenants": 0,
        "new_pattern_assignment_partition_links": 0,
        "new_pattern_assignment_created_partitions": 0,
        "new_pattern_assignment_existing_partition_changes": 0,
        "new_pattern_assignment_replication_avoided": 0,
    }

    def route_cost(partition_vectors: int, accessible_vectors: int) -> float:
        accessible = int(accessible_vectors)
        if int(accessible) <= 0:
            return 0.0
        return estimate_partition_query_cost(
            partition_vectors=max(1, int(partition_vectors)),
            accessible_vectors=int(accessible),
            tenant_weight=1.0,
            ef_search=int(ef_search),
            use_adaptive_ef=True,
        )

    for pattern_id in sorted(new_pattern_ids):
        pattern = patterns_by_id[int(pattern_id)]
        pattern_bit = bit_context.pattern_bits_for((int(pattern_id),))
        pattern_tenants = {int(tenant_id) for tenant_id in pattern.tenant_ids}
        metadata["new_pattern_assignment_total_tenants"] = int(metadata["new_pattern_assignment_total_tenants"]) + int(len(pattern_tenants))

        tenant_partition_ids: dict[int, list[str]] = defaultdict(list)
        partition_bits_by_id: dict[str, int] = {}
        tenant_bits_by_partition_id: dict[str, dict[int, int]] = {}
        partitions_by_id: dict[str, KMeansPartition] = {}
        for partition in current:
            partition_id = str(partition.partition_id)
            partitions_by_id[partition_id] = partition
            partition_bits = bit_context.pattern_bits_for(partition.pattern_ids)
            partition_bits_by_id[partition_id] = int(partition_bits)
            tenant_bits = bit_context.tenant_bits_for_partition(partition)
            tenant_bits_by_partition_id[partition_id] = tenant_bits
            for tenant_id in partition.tenant_ids:
                tenant_partition_ids[int(tenant_id)].append(partition_id)

        def tenant_partition_rank(tenant_id: int, partition_id: str) -> tuple[float, int, str]:
            partition_bits = int(partition_bits_by_id.get(str(partition_id), 0))
            tenant_bits = tenant_bits_by_partition_id.get(str(partition_id), {})
            before_bits = int(tenant_bits.get(int(tenant_id), 0)) & int(partition_bits)
            after_bits = int(before_bits) | int(pattern_bit)
            after_partition_bits = int(partition_bits) | int(pattern_bit)
            before_accessible = bit_context.vector_count_bits(int(before_bits))
            after_accessible = bit_context.vector_count_bits(int(after_bits))
            before_vectors = bit_context.vector_count_bits(int(partition_bits))
            after_vectors = bit_context.vector_count_bits(int(after_partition_bits))
            delta_cost = route_cost(after_vectors, after_accessible) - route_cost(before_vectors, before_accessible)
            return (float(delta_cost), int(after_vectors), str(partition_id))

        choices_by_tenant: dict[int, list[tuple[tuple[float, int, str], str]]] = {}
        grouped_best_tenants: dict[str, set[int]] = defaultdict(set)
        grouped_best_cost: dict[str, float] = defaultdict(float)
        for tenant_id in sorted(pattern_tenants):
            choices: list[tuple[tuple[float, int, str], str]] = []
            for partition_id in tenant_partition_ids.get(int(tenant_id), []):
                if str(partition_id) not in partitions_by_id:
                    continue
                rank = tenant_partition_rank(int(tenant_id), str(partition_id))
                choices.append((rank, str(partition_id)))
            choices.sort(key=lambda item: item[0])
            if choices:
                choices_by_tenant[int(tenant_id)] = choices
                best_rank, best_partition_id = choices[0]
                grouped_best_tenants[str(best_partition_id)].add(int(tenant_id))
                grouped_best_cost[str(best_partition_id)] += float(best_rank[0])

        selected_partition_ids = [
            partition_id
            for partition_id, _tenant_ids in sorted(
                grouped_best_tenants.items(),
                key=lambda item: (
                    -int(len(item[1])),
                    float(grouped_best_cost.get(str(item[0]), 0.0)),
                    int(bit_context.vector_count_bits(int(partition_bits_by_id.get(str(item[0]), 0)) | int(pattern_bit))),
                    str(item[0]),
                ),
            )[:max_links]
        ]

        selected_tenants_by_partition: dict[str, set[int]] = defaultdict(set)
        for tenant_id in sorted(pattern_tenants):
            if selected_partition_ids:
                best_selected_partition_id = min(
                    selected_partition_ids,
                    key=lambda partition_id: tenant_partition_rank(int(tenant_id), str(partition_id)),
                )
                selected_tenants_by_partition[str(best_selected_partition_id)].add(int(tenant_id))

        if not selected_tenants_by_partition:
            best_partition_id = None
            best_rank = None
            for partition in current:
                partition_id = str(partition.partition_id)
                partition_bits = int(partition_bits_by_id.get(partition_id, 0))
                after_vectors = bit_context.vector_count_bits(int(partition_bits) | int(pattern_bit))
                rank = (
                    int(after_vectors),
                    int(len(partition.pattern_ids)),
                    str(partition.partition_id),
                )
                if best_rank is None or rank < best_rank:
                    best_rank = rank
                    best_partition_id = str(partition.partition_id)
            if best_partition_id is not None:
                selected_tenants_by_partition[str(best_partition_id)] = set(pattern_tenants)

        updated = []
        assigned_existing_count = 0
        for partition in current:
            assigned_tenants = selected_tenants_by_partition.get(str(partition.partition_id), set())
            if not assigned_tenants:
                updated.append(partition)
                continue
            tenant_patterns = _tenant_patterns_for_partition(partition, patterns_by_id)
            for tenant_id in assigned_tenants:
                tenant_patterns.setdefault(int(tenant_id), set()).add(int(pattern_id))
            pattern_ids = set(partition.pattern_ids)
            pattern_ids.add(int(pattern_id))
            rebuilt = _make_partition(
                str(partition.partition_id),
                cluster_id=int(partition.cluster_id),
                pattern_ids=pattern_ids,
                tenant_patterns=tenant_patterns,
                patterns_by_id=patterns_by_id,
            )
            if rebuilt is not None:
                updated.append(rebuilt)
                changed.add(str(rebuilt.partition_id))
                assigned_existing_count += 1
                metadata["new_pattern_assignment_existing_partition_changes"] = int(metadata["new_pattern_assignment_existing_partition_changes"]) + 1
                metadata["new_pattern_assignment_partition_links"] = int(metadata["new_pattern_assignment_partition_links"]) + 1
            else:
                updated.append(partition)
        current = updated

        if assigned_existing_count > 0:
            metadata["new_pattern_assignment_replication_avoided"] = (
                int(metadata["new_pattern_assignment_replication_avoided"]) + max(0, int(len(pattern_tenants)) - int(assigned_existing_count))
            )
            continue

        if not assigned_existing_count:
            partition_id = _next_private_partition_id(current)
            cluster_id = max([int(partition.cluster_id) for partition in current] or [-1]) + 1
            tenant_patterns = {int(tenant_id): {int(pattern_id)} for tenant_id in pattern_tenants}
            new_partition = _make_partition(
                partition_id,
                cluster_id=int(cluster_id),
                pattern_ids={int(pattern_id)},
                tenant_patterns=tenant_patterns,
                patterns_by_id=patterns_by_id,
            )
            if new_partition is not None:
                current.append(new_partition)
                changed.add(str(new_partition.partition_id))
                metadata["new_pattern_assignment_created_partitions"] = int(metadata["new_pattern_assignment_created_partitions"]) + 1
                metadata["new_pattern_assignment_partition_links"] = int(metadata["new_pattern_assignment_partition_links"]) + 1
                metadata["new_pattern_assignment_replication_avoided"] = (
                    int(metadata["new_pattern_assignment_replication_avoided"]) + max(0, int(len(pattern_tenants)) - 1)
                )
    return current, changed, metadata


def _copy_partition_with_id(partition: KMeansPartition, partition_id: str) -> KMeansPartition:
    return KMeansPartition(
        partition_id=str(partition_id),
        cluster_id=int(partition.cluster_id),
        partition_kind=str(partition.partition_kind),
        table_name=get_partition_table_name(str(partition_id)),
        tenant_ids=partition.tenant_ids,
        pattern_ids=partition.pattern_ids,
        document_ids=partition.document_ids,
        document_pattern_pairs=partition.document_pattern_pairs,
        vector_count=int(partition.vector_count),
        metadata=dict(partition.metadata or {}),
    )


def _next_update_partition_id(used_ids: set[str]) -> str:
    index = 0
    while f"private_u{index}" in used_ids:
        index += 1
    partition_id = f"private_u{index}"
    used_ids.add(partition_id)
    return partition_id


def _normalize_candidate_new_partition_ids(
    candidate: KMeansMaintenanceCandidate,
    *,
    used_ids: set[str],
) -> KMeansMaintenanceCandidate:
    if candidate.op_type == "compact":
        return candidate
    new_partitions = list(candidate.payload.get("new_partitions", []) or [])
    if not new_partitions:
        return candidate
    normalized = []
    for partition in new_partitions:
        requested_id = str(partition.partition_id)
        if requested_id in used_ids or "__" in requested_id or len(get_partition_table_name(requested_id)) > 55:
            partition_id = _next_update_partition_id(used_ids)
        else:
            partition_id = requested_id
            used_ids.add(partition_id)
        normalized.append(_copy_partition_with_id(partition, partition_id))
    payload = dict(candidate.payload)
    payload["new_partitions"] = normalized
    return KMeansMaintenanceCandidate(
        op_type=str(candidate.op_type),
        partition_ids=tuple(str(value) for value in candidate.partition_ids),
        delta_memory=int(candidate.delta_memory),
        delta_latency=float(candidate.delta_latency),
        payload=payload,
    )


def _apply_candidates(
    partitions: list[KMeansPartition],
    candidates: list[KMeansMaintenanceCandidate],
) -> tuple[list[KMeansPartition], set[str]]:
    current = list(partitions)
    rewritten: set[str] = set()
    for candidate in candidates:
        touched = set(str(value) for value in candidate.partition_ids)
        if candidate.op_type == "compact":
            rewritten.update(touched)
            continue
        new_partitions = list(candidate.payload.get("new_partitions", []) or [])
        current = [partition for partition in current if str(partition.partition_id) not in touched]
        used_ids = {str(partition.partition_id) for partition in current}
        for new_partition in new_partitions:
            base_id = str(new_partition.partition_id)
            partition_id = base_id
            suffix = 0
            while partition_id in used_ids:
                suffix += 1
                partition_id = f"{base_id}_{suffix}"
            used_ids.add(partition_id)
            current.append(
                KMeansPartition(
                    partition_id=partition_id,
                    cluster_id=int(new_partition.cluster_id),
                    partition_kind="private",
                    table_name=get_partition_table_name(partition_id),
                    tenant_ids=new_partition.tenant_ids,
                    pattern_ids=new_partition.pattern_ids,
                    document_ids=new_partition.document_ids,
                    document_pattern_pairs=new_partition.document_pattern_pairs,
                    vector_count=int(new_partition.vector_count),
                    metadata=new_partition.metadata,
                )
            )
            rewritten.add(partition_id)
        rewritten.update(touched)
    return current, rewritten


def _memory_budget(plan_summary: dict[str, object], patterns: list[ACLPattern]) -> int:
    metadata = dict(plan_summary.get("metadata", {}) or {})
    ratio = float(metadata.get("private_replication_budget_ratio", 2.0) or 0.0)
    current_vector_count = sum(int(pattern.vector_count) for pattern in patterns)
    return int(math.floor(float(current_vector_count) * (1.0 + max(0.0, ratio))))


def _ef_search(plan_summary: dict[str, object]) -> int:
    metadata = dict(plan_summary.get("metadata", {}) or {})
    return int(metadata.get("ef_search_for_cost", metadata.get("ef_search", 40)) or 40)


def _private_edge_top_d(plan_summary: dict[str, object]) -> int:
    metadata = dict(plan_summary.get("metadata", {}) or {})
    private_meta = dict(metadata.get("private_cluster_metadata", {}) or {})
    return int(metadata.get("private_edge_top_d", private_meta.get("private_edge_top_d", 32)) or 32)


def _save_and_materialize_changed(
    plan: KMeansPlan,
    *,
    changed_partition_ids: set[str],
    removed_table_names: set[str],
    create_indexes: bool,
    index_type: str,
    vector_index_min_vectors: int,
    repository: KMeansUpdateRepository,
) -> dict[str, float]:
    timing: dict[str, float] = {
        "partition_metadata_save_seconds": 0.0,
        "partition_materialize_seconds": 0.0,
        "index_build_seconds": 0.0,
        "stale_partition_drop_seconds": 0.0,
        "tombstone_clear_seconds": 0.0,
    }
    started_at = time.perf_counter()
    save_plan(plan, db_connection_factory=repository.db_connection_factory)
    timing["partition_metadata_save_seconds"] = time.perf_counter() - started_at
    valid_table_names = {str(partition.table_name) for partition in plan.partitions}
    started_at = time.perf_counter()
    drop_stale_materialized_partitions(valid_table_names | set(removed_table_names), db_connection_factory=repository.db_connection_factory)
    timing["stale_partition_drop_seconds"] += time.perf_counter() - started_at
    changed = [partition for partition in plan.partitions if str(partition.partition_id) in changed_partition_ids or str(partition.table_name) in removed_table_names]
    timing["materialized_partition_count"] = float(len(changed))
    timing["requested_changed_partition_id_count"] = float(len(changed_partition_ids))
    for partition in changed:
        started_at = time.perf_counter()
        materialize_partition(partition, db_connection_factory=repository.db_connection_factory)
        timing["partition_materialize_seconds"] += time.perf_counter() - started_at
    if create_indexes and changed:
        started_at = time.perf_counter()
        create_indexes_for_partitions(
            changed,
            index_type=index_type,
            vector_index_min_vectors=int(vector_index_min_vectors),
            parallel=True,
            db_connection_factory=repository.db_connection_factory,
        )
        timing["index_build_seconds"] += time.perf_counter() - started_at
    if create_indexes:
        started_at = time.perf_counter()
        drop_vector_indexes_below_threshold(
            plan.partitions,
            vector_index_min_vectors=int(vector_index_min_vectors),
            db_connection_factory=repository.db_connection_factory,
        )
        timing["index_build_seconds"] += time.perf_counter() - started_at
    if removed_table_names:
        started_at = time.perf_counter()
        drop_stale_materialized_partitions(valid_table_names, db_connection_factory=repository.db_connection_factory)
        timing["stale_partition_drop_seconds"] += time.perf_counter() - started_at
    started_at = time.perf_counter()
    repository.clear_tombstones(changed_partition_ids)
    timing["tombstone_clear_seconds"] = time.perf_counter() - started_at
    invalidate_cache()
    return timing


def apply_kmeans_update_batch(
    updates: Iterable[KMeansUpdateItem | dict],
    *,
    tau_del: float = 0.2,
    max_operations: int = 8,
    max_new_pattern_partitions: int = 2,
    enable_maintenance: bool = True,
    create_indexes: bool = True,
    index_type: str = "squidhnsw",
    vector_index_min_vectors: int = 1,
    db_connection_factory=None,
) -> KMeansUpdateResult:
    total_started_at = time.perf_counter()
    items = [item if isinstance(item, KMeansUpdateItem) else KMeansUpdateItem.from_mapping(dict(item)) for item in updates]
    if not items:
        raise ValueError("update batch is empty")
    repository = KMeansUpdateRepository(db_connection_factory=db_connection_factory or _default_db_connection_factory)
    batch_id = repository.create_batch(items)

    plan_summary = get_current_plan_summary(refresh=True, db_connection_factory=repository.db_connection_factory)
    if plan_summary is None:
        raise RuntimeError("No kmeans plan found")
    old_patterns = repository.fetch_current_patterns()
    old_patterns_by_id = {int(pattern.pattern_id): pattern for pattern in old_patterns}
    old_routes = repository.fetch_current_routes()
    old_partitions = load_current_partitions(refresh=True, db_connection_factory=repository.db_connection_factory)
    old_doc_to_pattern = _document_pattern_map(old_patterns)

    touched_docs = {int(item.document_id) for item in items}
    old_doc_partitions = {
        int(document_id): [
            str(partition.partition_id)
            for partition in old_partitions
            if int(document_id) in set(map(int, partition.document_ids))
        ]
        for document_id in touched_docs
    }
    tombstone_partition_ids = {partition_id for values in old_doc_partitions.values() for partition_id in values}
    tombstone_count = repository.record_tombstones_for_documents(tombstone_partition_ids, touched_docs, batch_id=int(batch_id))

    apply_started_at = time.perf_counter()
    operation_apply_times: dict[str, list[float]] = {"insert": [], "delete": [], "update": []}
    for item in items:
        op = str(item.operation).lower()
        op_key = "delete" if op in {"delete", "remove"} else "insert" if op in {"insert", "upsert"} else "update"
        item_apply_started_at = time.perf_counter()
        if op in {"delete", "remove"}:
            repository.delete_main_document_data(int(item.document_id))
            operation_apply_times[op_key].append(time.perf_counter() - item_apply_started_at)
            continue
        if op in {"insert", "upsert", "vector_update", "update_vector"}:
            repository.upsert_document_blocks(item)
        if op in {"insert", "upsert", "acl_update", "update_acl", "vector_update", "update_vector"}:
            if _item_has_acl_payload(item):
                repository.replace_document_acl(int(item.document_id), role_ids=item.role_ids, tenant_ids=item.tenant_ids)
        elif op in {"acl_grant", "grant"}:
            # ``role_ids`` is intentionally a delta here, unlike acl_update
            # where it represents the complete replacement ACL.
            repository.grant_document_roles(int(item.document_id), item.role_ids)
        elif op in {"acl_revoke", "revoke"}:
            # This removes only the listed role-document edges.  It neither
            # touches Users/UserRoles nor deletes Documents/DocumentBlocks.
            repository.revoke_document_roles(int(item.document_id), item.role_ids)
        operation_apply_times[op_key].append(time.perf_counter() - item_apply_started_at)
    update_apply_seconds = time.perf_counter() - apply_started_at
    operation_apply_stats = {
        f"avg_{key}_seconds": (sum(values) / len(values) if values else 0.0)
        for key, values in operation_apply_times.items()
    }
    operation_apply_stats.update({f"{key}_count": len(values) for key, values in operation_apply_times.items()})
    print(f"[kmeans-update] batch {batch_id}: main-table updates applied in {update_apply_seconds:.4f}s", flush=True)

    maintenance_started_at = time.perf_counter()
    timing: dict[str, float] = {}
    phase_started_at = time.perf_counter()
    refreshed_patterns = _refresh_patterns(repository, old_patterns)
    patterns_by_id = {int(pattern.pattern_id): pattern for pattern in refreshed_patterns}
    refreshed_doc_to_pattern = _document_pattern_map(refreshed_patterns)
    changed_pattern_ids = {
        int(old_doc_to_pattern[doc])
        for doc in touched_docs
        if int(doc) in old_doc_to_pattern
    } | {
        int(refreshed_doc_to_pattern[doc])
        for doc in touched_docs
        if int(doc) in refreshed_doc_to_pattern
    }
    new_pattern_ids = {int(pattern.pattern_id) for pattern in refreshed_patterns if int(pattern.pattern_id) not in old_patterns_by_id}
    timing["pattern_refresh_seconds"] = time.perf_counter() - phase_started_at

    phase_started_at = time.perf_counter()
    refreshed_partitions = _rebuild_partitions_with_patterns(
        old_partitions,
        patterns_by_id,
        changed_pattern_ids=set(changed_pattern_ids),
    )
    partition_ids_before_new_assignment = {str(partition.partition_id) for partition in refreshed_partitions}
    memory_budget = _memory_budget(plan_summary, refreshed_patterns)
    top_d = _private_edge_top_d(plan_summary)
    refreshed_partitions, new_assignment_partitions, new_assignment_metadata = _assign_new_patterns(
        refreshed_partitions,
        new_pattern_ids,
        patterns_by_id,
        ef_search=_ef_search(plan_summary),
        memory_budget=int(memory_budget),
        max_partition_links_per_pattern=int(max_new_pattern_partitions),
    )
    new_physical_partition_ids = {
        str(partition.partition_id)
        for partition in refreshed_partitions
        if str(partition.partition_id) not in partition_ids_before_new_assignment
    }
    timing["partition_rebuild_assignment_seconds"] = time.perf_counter() - phase_started_at

    phase_started_at = time.perf_counter()
    query_direct_partition_ids = {
        str(partition.partition_id)
        for partition in refreshed_partitions
        if set(map(int, partition.pattern_ids)) & changed_pattern_ids
    } | set(new_assignment_partitions)
    query_affected_ids, affected_region_metadata = _affected_region(
        refreshed_partitions,
        patterns_by_id,
        query_direct_partition_ids,
        top_d=int(top_d),
    )
    tombstones_by_partition = repository.tombstone_counts()
    global_compact_ids = {
        str(partition.partition_id)
        for partition in refreshed_partitions
        if int(tombstones_by_partition.get(str(partition.partition_id), 0) or 0) > 0
        and (
            float(tombstones_by_partition.get(str(partition.partition_id), 0) or 0)
            / float(max(1, int(partition.vector_count) + int(tombstones_by_partition.get(str(partition.partition_id), 0) or 0)))
        )
        >= float(tau_del)
    }
    compact_ids = set(tombstone_partition_ids) | set(global_compact_ids)
    memory_now = sum(int(partition.vector_count) + int(tombstones_by_partition.get(str(partition.partition_id), 0)) for partition in refreshed_partitions)
    selection_mode = "memory_reduction" if int(memory_now) > int(memory_budget) else "query_cost_reduction"
    memory_pair_seed_ids = set(compact_ids) | set(tombstone_partition_ids) | set(new_assignment_partitions)
    if not memory_pair_seed_ids:
        memory_pair_seed_ids = set(query_direct_partition_ids)
    memory_pair_region_ids, memory_pair_region_raw_metadata = _affected_region(
        refreshed_partitions,
        patterns_by_id,
        memory_pair_seed_ids,
        top_d=int(top_d),
    )
    memory_pair_region_metadata = {
        "memory_pair_seed_partition_count": int(len({str(value) for value in memory_pair_seed_ids})),
        "memory_pair_region_partition_count": int(len(memory_pair_region_ids)),
        "memory_pair_region_core_star_edges": int(memory_pair_region_raw_metadata.get("affected_core_star_edges", 0)),
        "memory_pair_region_owner_patterns": int(memory_pair_region_raw_metadata.get("affected_core_star_owner_patterns", 0)),
    }
    affected_ids = set(query_affected_ids) | set(compact_ids)
    maintenance_scope_ids = set(affected_ids)
    if str(selection_mode) == "memory_reduction":
        maintenance_scope_ids.update(memory_pair_region_ids)
        affected_ids.update(memory_pair_region_ids)
    affected_region_metadata.update(
        {
            "query_affected_partition_count": int(len(query_affected_ids)),
            "query_direct_partition_count": int(len(query_direct_partition_ids)),
            "tombstone_compact_partition_count": int(len(tombstone_partition_ids)),
            "global_compact_partition_count": int(len(global_compact_ids)),
            "compact_partition_count": int(len(compact_ids)),
            "affected_partition_count": int(len(affected_ids)),
            "maintenance_scope_partition_count": int(len(maintenance_scope_ids)),
            **memory_pair_region_metadata,
        }
    )
    affected_partitions = [partition for partition in refreshed_partitions if str(partition.partition_id) in maintenance_scope_ids]
    timing["affected_region_seconds"] = time.perf_counter() - phase_started_at

    phase_started_at = time.perf_counter()
    if not bool(enable_maintenance):
        accepted = []
        maintenance_selection_metadata = {
            "maintenance_disabled": 1,
            "maintenance_skipped_under_budget": 1 if int(memory_now) <= int(memory_budget) else 0,
            "under_budget_query_optimization": 0,
            "accepted_operation_count": 0,
            "update_candidate_count": 0,
            "pair_candidate_count": 0,
            "compact_candidate_count": 0,
            "selectivity_candidate_count": 0,
            "memory_budget_satisfied": int(int(memory_now) <= int(memory_budget)),
            "memory_after_selection": int(memory_now),
            "local_maintenance_stop_reason_code": 12,
        }
    elif int(memory_now) > int(memory_budget):
        planner = _LocalPlanner(
            ef_search=_ef_search(plan_summary),
            memory_budget=int(memory_budget),
            tau_del=float(tau_del),
            max_operations=int(max_operations),
        )
        accepted, maintenance_selection_metadata = planner.plan_local_maintenance(
            affected_partitions,
            patterns_by_id,
            tombstones_by_partition,
            memory_now=int(memory_now),
            top_d=int(top_d),
            used_partition_ids={str(partition.partition_id) for partition in refreshed_partitions},
            optimization_partition_ids=set(memory_pair_region_ids),
            compact_partition_ids=set(compact_ids),
        )
    else:
        planner = _LocalPlanner(
            ef_search=_ef_search(plan_summary),
            memory_budget=int(memory_budget),
            tau_del=float(tau_del),
            max_operations=int(max_operations),
        )
        accepted, maintenance_selection_metadata = planner.plan_query_refinement(
            affected_partitions,
            patterns_by_id,
            tombstones_by_partition,
            memory_now=int(memory_now),
            top_d=int(top_d),
            used_partition_ids={str(partition.partition_id) for partition in refreshed_partitions},
            optimization_partition_ids=set(query_affected_ids),
            compact_partition_ids=set(compact_ids),
        )
        maintenance_selection_metadata.update(
            {
                "maintenance_skipped_under_budget": 0,
                "under_budget_query_optimization": 1,
            }
        )
    timing["maintenance_planning_seconds"] = time.perf_counter() - phase_started_at
    print(
        f"[kmeans-update] batch {batch_id}: maintenance selected {len(accepted)} ops "
        f"from {maintenance_selection_metadata.get('update_candidate_count', 0)} candidates",
        flush=True,
    )
    phase_started_at = time.perf_counter()
    planned_partitions, rewritten_partition_ids = _apply_candidates(refreshed_partitions, accepted)
    rewritten_partition_ids.update(new_physical_partition_ids)
    timing["apply_candidates_seconds"] = time.perf_counter() - phase_started_at

    phase_started_at = time.perf_counter()
    affected_tenant_ids = {
        int(tenant_id)
        for pattern_id in (set(changed_pattern_ids) | set(new_pattern_ids))
        if int(pattern_id) in patterns_by_id
        for tenant_id in patterns_by_id[int(pattern_id)].tenant_ids
    }
    for candidate in accepted:
        if candidate.payload.get("worst_tenant") is not None:
            affected_tenant_ids.add(int(candidate.payload["worst_tenant"]))
        for partition in candidate.payload.get("new_partitions", []) or []:
            affected_tenant_ids.update(int(tenant_id) for tenant_id in partition.tenant_ids)
    planned_partitions, routes, route_repair_metadata = _repair_routes_greedy(
        planned_partitions,
        patterns_by_id,
        affected_tenant_ids=affected_tenant_ids,
        existing_routes=old_routes,
    )
    timing["route_repair_seconds"] = time.perf_counter() - phase_started_at
    print(
        f"[kmeans-update] batch {batch_id}: repaired {route_repair_metadata.get('route_repair_tenant_count', 0)} tenants, "
        f"routes={route_repair_metadata.get('route_repair_route_count', 0)}",
        flush=True,
    )

    phase_started_at = time.perf_counter()
    removed_table_names = {str(partition.table_name) for partition in old_partitions} - {str(partition.table_name) for partition in planned_partitions}
    metadata = dict(plan_summary.get("metadata", {}) or {})
    metadata.update(
        {
            "algorithm": "private_core_star_split_merge_v16",
            "update_enabled": True,
            "update_maintenance_enabled": bool(enable_maintenance),
            "update_last_batch_id": int(batch_id),
            "update_last_batch_size": int(len(items)),
            "update_memory_budget": int(memory_budget),
            "update_memory_before_selection": int(memory_now),
            "update_selection_mode": str(selection_mode),
            **new_assignment_metadata,
            **affected_region_metadata,
            **maintenance_selection_metadata,
            **route_repair_metadata,
            "partition_count": int(len(planned_partitions)),
            "pattern_count": int(len(refreshed_patterns)),
            "document_count": int(len({doc for pattern in refreshed_patterns for doc in pattern.document_ids})),
            "partition_vector_count": int(sum(int(partition.vector_count) for partition in planned_partitions)),
            **timing,
        }
    )
    plan = KMeansPlan(
        partitions=planned_partitions,
        tenant_routes=routes,
        tenant_to_cluster={int(route.tenant_id): int(route.cluster_id) for route in routes},
        patterns=refreshed_patterns,
        metadata=metadata,
    )
    save_materialize_timing = _save_and_materialize_changed(
        plan,
        changed_partition_ids=rewritten_partition_ids,
        removed_table_names=removed_table_names,
        create_indexes=bool(create_indexes),
        index_type=str(index_type),
        vector_index_min_vectors=int(vector_index_min_vectors),
        repository=repository,
    )
    timing.update(save_materialize_timing)
    timing["save_materialize_seconds"] = time.perf_counter() - phase_started_at
    print(
        f"[kmeans-update] batch {batch_id}: materialized "
        f"{int(save_materialize_timing.get('materialized_partition_count', 0.0) or 0.0)} final changed partitions "
        f"({len(rewritten_partition_ids)} rewritten ids)",
        flush=True,
    )

    # Incremental inserts into unchanged partitions that gained newly inserted documents.
    phase_started_at = time.perf_counter()
    for item in items:
        pattern_id = refreshed_doc_to_pattern.get(int(item.document_id))
        if pattern_id is None:
            continue
        for partition in planned_partitions:
            if int(pattern_id) not in set(map(int, partition.pattern_ids)):
                continue
            if str(partition.partition_id) in rewritten_partition_ids:
                continue
            repository.insert_document_into_partition(partition, int(pattern_id), int(item.document_id))
    timing["incremental_insert_seconds"] = time.perf_counter() - phase_started_at

    maintenance_seconds = time.perf_counter() - maintenance_started_at
    total_seconds = time.perf_counter() - total_started_at
    print(
        f"[kmeans-update] batch {batch_id}: maintenance timing "
        f"planning={timing.get('maintenance_planning_seconds', 0.0):.4f}s, "
        f"pattern_refresh={timing.get('pattern_refresh_seconds', 0.0):.4f}s, "
        f"partition_rebuild_assignment={timing.get('partition_rebuild_assignment_seconds', 0.0):.4f}s, "
        f"route_repair={timing.get('route_repair_seconds', 0.0):.4f}s, "
        f"metadata_save={timing.get('partition_metadata_save_seconds', 0.0):.4f}s, "
        f"materialize={timing.get('partition_materialize_seconds', 0.0):.4f}s, "
        f"index_build={timing.get('index_build_seconds', 0.0):.4f}s, "
        f"drop_stale={timing.get('stale_partition_drop_seconds', 0.0):.4f}s, "
        f"tombstone_clear={timing.get('tombstone_clear_seconds', 0.0):.4f}s, "
        f"save_materialize_total={timing.get('save_materialize_seconds', 0.0):.4f}s, "
        f"incremental_insert={timing.get('incremental_insert_seconds', 0.0):.4f}s, "
        f"maintenance_total={maintenance_seconds:.4f}s",
        flush=True,
    )
    return KMeansUpdateResult(
        batch_id=int(batch_id),
        applied_count=int(len(items)),
        affected_partition_ids=tuple(sorted(affected_ids)),
        rewritten_partition_ids=tuple(sorted(rewritten_partition_ids)),
        accepted_operations=tuple(accepted),
        metadata={
            "update_apply_seconds": float(update_apply_seconds),
            **operation_apply_stats,
            "maintenance_seconds": float(maintenance_seconds),
            "total_update_batch_seconds": float(total_seconds),
            "vector_index_min_vectors": int(vector_index_min_vectors),
            "tombstone_count_recorded": int(tombstone_count),
            "candidate_count": int(maintenance_selection_metadata.get("update_candidate_count", 0)),
            **new_assignment_metadata,
            **affected_region_metadata,
            **maintenance_selection_metadata,
            **route_repair_metadata,
            **timing,
            "memory_budget": int(memory_budget),
            "memory_before_selection": int(memory_now),
            "selection_mode": str(selection_mode),
        },
    )
