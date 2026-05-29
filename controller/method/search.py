from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import importlib
import math
import sys
import time
from typing import Optional

import numpy as np
from psycopg2 import sql

from services.config import get_db_connection

from .common import PersistedDagNode, PersistedLogicalPattern, WorkloadAwarePartition, _normalize_vector, _parse_vector, get_partition_table_name
from .storage import (
    get_current_plan_summary,
    load_current_access_overlays,
    load_current_dag_nodes,
    load_current_logical_patterns,
    load_current_partitions,
    load_current_tenant_overlays,
)


_CACHED_ROUTE_INDEX: Optional["RouteIndex"] = None


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


def _configured_setting(*names: str):
    efconfig = _resolve_efconfig_module()
    if efconfig is None:
        return None
    for name in names:
        if hasattr(efconfig, name):
            return getattr(efconfig, name)
    return None


def _configured_fixed_ef_search() -> Optional[int]:
    configured = _configured_setting("method_ef_search", "dynamic_partition_ef_search", "ef_search")
    if configured is None:
        return None
    if isinstance(configured, str):
        normalized = configured.strip().lower()
        if normalized in {"", "adaptive", "auto", "none"}:
            return None
        return max(1, int(float(normalized)))
    return max(1, int(configured))


def _configured_int(primary_name: str, default: int, *, minimum: int = 1, aliases: tuple[str, ...] = ()) -> int:
    configured = _configured_setting(primary_name, *aliases)
    if configured is None:
        return max(minimum, int(default))
    return max(minimum, int(configured))


def _configured_float(primary_name: str, default: float, *, aliases: tuple[str, ...] = ()) -> float:
    configured = _configured_setting(primary_name, *aliases)
    if configured is None:
        return float(default)
    return float(configured)


@dataclass(slots=True)
class WorkloadAwareRoute:
    tenant_id: int
    partition_ids: tuple[str, ...]
    partition_count: int
    selected_candidates: tuple["RouteCandidate", ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class RouteCandidate:
    partition_id: str
    table_name: str
    base_score: float
    representative_centroid: np.ndarray
    prototype_centroids: tuple[np.ndarray, ...]
    matched_pattern_ids: tuple[int, ...]
    accelerator_patterns: tuple[dict[str, object], ...]
    matched_document_count: int
    matched_vector_count: int
    matched_query_mass: float
    partition_document_count: int
    partition_vector_count: int
    storage_layout_version: int
    source: str


@dataclass(slots=True)
class TenantOverlay:
    tenant_id: int
    table_name: str
    document_count: int
    vector_count: int
    query_mass: float
    covered_partition_count: int = 1
    estimated_saved_cost: float = 0.0
    benefit_density: float = 0.0


@dataclass(slots=True)
class AccessOverlay:
    tenant_id: int
    partition_id: str
    partition_ids: tuple[str, ...]
    table_name: str
    pattern_ids: tuple[int, ...]
    document_count: int
    vector_count: int
    query_mass: float
    tenant_vector_count: int = 0
    covered_partition_count: int = 1
    estimated_saved_cost: float = 0.0
    benefit_density: float = 0.0
    overlay_type: str = ""
    requires_pattern_filter: bool = False


@dataclass(slots=True)
class RouteIndex:
    plan_id: int
    entry_candidates_by_tenant: dict[int, tuple[RouteCandidate, ...]]
    fallback_candidates_by_tenant: dict[int, tuple[RouteCandidate, ...]]
    singleton_node_ids_by_tenant: dict[int, int]
    tenant_pattern_counts: dict[int, int]
    tenant_overlays_by_tenant: dict[int, TenantOverlay]
    access_overlays_by_key: dict[tuple[int, str], AccessOverlay]


def _tenant_density(partition: WorkloadAwarePartition, tenant_id: int) -> float:
    densities = partition.metadata.get("tenant_densities", {}) or {}
    return float(densities.get(str(int(tenant_id)), 0.0) or 0.0)


def _tenant_query_mass(partition: WorkloadAwarePartition, tenant_id: int) -> float:
    weights = partition.metadata.get("tenant_query_mass", {}) or {}
    return float(weights.get(str(int(tenant_id)), 0.0) or 0.0)


def _static_partition_score(partition: WorkloadAwarePartition, tenant_id: int) -> float:
    density_weight = _configured_float(
        "method_route_density_weight",
        1.0,
        aliases=("dynamic_partition_route_density_weight",),
    )
    workload_weight = _configured_float(
        "method_route_workload_weight",
        0.2,
        aliases=("dynamic_partition_route_workload_weight",),
    )
    prior_weight = _configured_float(
        "method_route_prior_weight",
        0.05,
        aliases=("dynamic_partition_route_prior_weight",),
    )
    pattern_penalty = _configured_float(
        "method_route_pattern_penalty",
        0.01,
        aliases=("dynamic_partition_route_pattern_penalty",),
    )

    density_score = _tenant_density(partition, tenant_id)
    workload_score = math.log1p(_tenant_query_mass(partition, tenant_id))
    route_prior = float(partition.metadata.get("route_prior", 0.0) or 0.0)
    logical_pattern_count = int(partition.metadata.get("logical_pattern_count", len(partition.logical_pattern_ids)) or 0)
    return (
        density_weight * density_score
        + workload_weight * workload_score
        + prior_weight * route_prior
        - pattern_penalty * max(0, logical_pattern_count - 1)
    )


def _candidate_sort_key(candidate: RouteCandidate) -> tuple[float, str]:
    return (-float(candidate.base_score), str(candidate.partition_id))


def _normalize_score_values(values: list[float]) -> list[float]:
    if not values:
        return []
    minimum = min(float(value) for value in values)
    maximum = max(float(value) for value in values)
    span = max(float(maximum - minimum), 1e-6)
    return [float((float(value) - minimum) / span) for value in values]


def _semantic_score_for_candidate(candidate: RouteCandidate, query_array: np.ndarray) -> float:
    if not query_array.size:
        return 0.0
    if candidate.prototype_centroids:
        return max(float(np.dot(centroid, query_array)) for centroid in candidate.prototype_centroids)
    if candidate.representative_centroid.size:
        return float(np.dot(candidate.representative_centroid, query_array))
    return 0.0


def _route_physical_key(
    tenant_id: int,
    candidate: RouteCandidate,
    access_overlays_by_key: dict[tuple[int, str], AccessOverlay],
) -> tuple[str, str]:
    access_overlay = access_overlays_by_key.get((int(tenant_id), str(candidate.partition_id)))
    if access_overlay is not None:
        return ("access_overlay", str(access_overlay.table_name))
    return ("partition", str(candidate.partition_id))


def _covered_partition_ids_for_candidate(
    tenant_id: int,
    candidate: RouteCandidate,
    access_overlays_by_key: dict[tuple[int, str], AccessOverlay],
) -> tuple[str, ...]:
    access_overlay = access_overlays_by_key.get((int(tenant_id), str(candidate.partition_id)))
    if access_overlay is not None and access_overlay.partition_ids:
        return tuple(str(partition_id) for partition_id in access_overlay.partition_ids)
    return (str(candidate.partition_id),)


def _selected_accessible_vector_count(
    candidates: list[RouteCandidate],
    *,
    tenant_id: int,
    candidates_by_partition: dict[str, RouteCandidate],
    access_overlays_by_key: dict[tuple[int, str], AccessOverlay],
) -> int:
    selected_partition_ids: set[str] = set()
    total = 0
    for candidate in candidates:
        for partition_id in _covered_partition_ids_for_candidate(
            int(tenant_id),
            candidate,
            access_overlays_by_key,
        ):
            if str(partition_id) in selected_partition_ids:
                continue
            covered_candidate = candidates_by_partition.get(str(partition_id))
            if covered_candidate is None:
                continue
            selected_partition_ids.add(str(partition_id))
            total += max(0, int(covered_candidate.matched_vector_count))
    return int(total)


def _unique_access_overlays_for_tenant(
    tenant_id: int,
    access_overlays_by_key: dict[tuple[int, str], AccessOverlay],
) -> tuple[AccessOverlay, ...]:
    overlays_by_table: dict[str, AccessOverlay] = {}
    for (overlay_tenant_id, _), overlay in access_overlays_by_key.items():
        if int(overlay_tenant_id) != int(tenant_id):
            continue
        overlays_by_table[str(overlay.table_name)] = overlay
    return tuple(
        sorted(
            overlays_by_table.values(),
            key=lambda overlay: (
                -int(overlay.covered_partition_count),
                -float(overlay.benefit_density),
                -float(overlay.estimated_saved_cost),
                str(overlay.table_name),
            ),
        )
    )


def _collect_dag_pattern_ids(node_by_id: dict[int, PersistedDagNode], start_node_id: int) -> set[int]:
    stack = [int(start_node_id)]
    visited = set()
    pattern_ids: set[int] = set()
    while stack:
        node_id = int(stack.pop())
        if node_id in visited:
            continue
        visited.add(node_id)
        node = node_by_id.get(node_id)
        if node is None:
            continue
        pattern_ids.update(int(pattern_id) for pattern_id in node.terminal_pattern_ids)
        pattern_ids.update(int(pattern_id) for pattern_id in node.supplemental_pattern_ids)
        for child_node_id in node.children.values():
            if int(child_node_id) not in visited:
                stack.append(int(child_node_id))
    return pattern_ids


def _pattern_tenant_query_mass(pattern: PersistedLogicalPattern, tenant_id: int) -> float:
    masses = pattern.metadata.get("tenant_query_mass", {}) or {}
    return float(
        masses.get(str(int(tenant_id)), masses.get(int(tenant_id), 0.0)) or 0.0
    )


def _pattern_centroid(pattern: PersistedLogicalPattern) -> np.ndarray:
    return _normalize_vector(_parse_vector(pattern.metadata.get("representative_centroid", [])))


def _build_route_candidate(
    *,
    tenant_id: int,
    partition: WorkloadAwarePartition,
    matched_pattern_ids: tuple[int, ...],
    patterns_by_id: dict[int, PersistedLogicalPattern],
    base_score: float,
    source: str,
) -> Optional[RouteCandidate]:
    if not matched_pattern_ids:
        return None

    matched_patterns = [
        patterns_by_id[int(pattern_id)]
        for pattern_id in matched_pattern_ids
        if int(pattern_id) in patterns_by_id
    ]
    if not matched_patterns:
        return None

    ordered_patterns = sorted(
        matched_patterns,
        key=lambda pattern: (
            -float(_pattern_tenant_query_mass(pattern, tenant_id)),
            -int(pattern.document_count),
            int(pattern.pattern_id),
        ),
    )
    accelerator_entries = {
        int(entry.get("pattern_id", -1)): dict(entry)
        for entry in (partition.metadata.get("accelerator_patterns", []) or [])
    }
    partition_pattern_document_counts = {
        int(pattern_id): int(count)
        for pattern_id, count in (partition.metadata.get("pattern_document_counts", {}) or {}).items()
    }
    partition_pattern_vector_counts = {
        int(pattern_id): int(count)
        for pattern_id, count in (partition.metadata.get("pattern_vector_counts", {}) or {}).items()
    }
    semantic_atoms_by_pattern: dict[int, list[dict[str, object]]] = defaultdict(list)
    for atom in partition.metadata.get("semantic_atoms", []) or []:
        try:
            semantic_atoms_by_pattern[int(atom.get("pattern_id", -1))].append(dict(atom))
        except Exception:
            continue

    prototype_centroids: list[np.ndarray] = []
    weighted_centroid_numerator = None
    total_weight = 0.0
    matched_document_count = 0
    matched_vector_count = 0
    matched_query_mass = 0.0
    for pattern in ordered_patterns:
        centroid = _pattern_centroid(pattern)
        weight = max(
            1.0,
            float(_pattern_tenant_query_mass(pattern, tenant_id)),
            float(partition_pattern_document_counts.get(int(pattern.pattern_id), pattern.document_count)),
        )
        atom_centroids = []
        for atom in semantic_atoms_by_pattern.get(int(pattern.pattern_id), ()):
            atom_centroid = _normalize_vector(_parse_vector(atom.get("centroid", [])))
            if atom_centroid.size:
                atom_centroids.append((atom_centroid, int(atom.get("vector_count", 0) or 0)))
        if atom_centroids:
            for atom_centroid, atom_vector_count in atom_centroids:
                prototype_centroids.append(atom_centroid)
                atom_weight = max(1.0, float(atom_vector_count), weight)
                numerator = atom_centroid * atom_weight
                weighted_centroid_numerator = (
                    numerator
                    if weighted_centroid_numerator is None
                    else weighted_centroid_numerator + numerator
                )
                total_weight += atom_weight
        elif centroid.size:
            prototype_centroids.append(centroid)
            numerator = centroid * weight
            weighted_centroid_numerator = (
                numerator
                if weighted_centroid_numerator is None
                else weighted_centroid_numerator + numerator
            )
            total_weight += weight
        pattern_document_count = int(partition_pattern_document_counts.get(int(pattern.pattern_id), pattern.document_count))
        pattern_vector_count = int(partition_pattern_vector_counts.get(int(pattern.pattern_id), pattern.vector_count))
        matched_document_count += int(pattern_document_count)
        matched_vector_count += int(pattern_vector_count)
        matched_query_mass += float(_pattern_tenant_query_mass(pattern, tenant_id)) * (
            float(pattern_document_count) / max(float(pattern.document_count), 1.0)
        )

    representative_centroid = np.zeros(0, dtype=np.float32)
    if weighted_centroid_numerator is not None and total_weight > 0.0:
        representative_centroid = _normalize_vector(weighted_centroid_numerator / total_weight)

    return RouteCandidate(
        partition_id=str(partition.partition_id),
        table_name=str(partition.table_name),
        base_score=float(base_score),
        representative_centroid=representative_centroid,
        prototype_centroids=tuple(prototype_centroids),
        matched_pattern_ids=tuple(sorted(int(pattern_id) for pattern_id in matched_pattern_ids)),
        accelerator_patterns=tuple(
            accelerator_entries[int(pattern.pattern_id)]
            for pattern in ordered_patterns
            if int(pattern.pattern_id) in accelerator_entries
        ),
        matched_document_count=int(matched_document_count),
        matched_vector_count=int(matched_vector_count),
        matched_query_mass=float(matched_query_mass),
        partition_document_count=int(partition.document_count),
        partition_vector_count=int(partition.vector_count),
        storage_layout_version=int(partition.metadata.get("storage_layout_version", 1) or 1),
        source=str(source),
    )


def _build_entry_candidates(
    tenant_id: int,
    pattern_ids: set[int],
    patterns_by_id: dict[int, PersistedLogicalPattern],
    partitions_by_id: dict[str, WorkloadAwarePartition],
    partitions_by_pattern: dict[int, tuple[WorkloadAwarePartition, ...]],
) -> tuple[RouteCandidate, ...]:
    candidates_by_partition: dict[str, dict[str, object]] = {}
    for pattern_id in sorted(int(value) for value in pattern_ids):
        pattern = patterns_by_id.get(int(pattern_id))
        if pattern is None or int(tenant_id) not in set(pattern.tenant_ids):
            continue
        matched_partitions = partitions_by_pattern.get(int(pattern_id), ())
        if not matched_partitions:
            fallback_partition = partitions_by_id.get(str(pattern.partition_id))
            matched_partitions = (fallback_partition,) if fallback_partition is not None else ()
        if not matched_partitions:
            continue
        if int(tenant_id) in set(pattern.entry_tenant_ids):
            source = "route_core" if pattern.ordered_tenant_ids and int(pattern.ordered_tenant_ids[0]) == int(tenant_id) else "route_entry"
            bonus = 0.05 if source == "route_core" else 0.02
        else:
            source = "acl_membership"
            bonus = 0.0
        for partition in matched_partitions:
            base_score = _static_partition_score(partition, int(tenant_id)) + bonus
            entry = candidates_by_partition.get(str(partition.partition_id))
            if entry is None:
                entry = {
                    "partition": partition,
                    "base_score": float(base_score),
                    "matched_pattern_ids": [int(pattern_id)],
                    "source": source,
                }
                candidates_by_partition[str(partition.partition_id)] = entry
            else:
                entry["matched_pattern_ids"].append(int(pattern_id))
                if float(base_score) > float(entry["base_score"]):
                    entry["base_score"] = float(base_score)
                    entry["source"] = source

    candidates = []
    for entry in candidates_by_partition.values():
        candidate = _build_route_candidate(
            tenant_id=int(tenant_id),
            partition=entry["partition"],
            matched_pattern_ids=tuple(int(pattern_id) for pattern_id in entry["matched_pattern_ids"]),
            patterns_by_id=patterns_by_id,
            base_score=float(entry["base_score"]),
            source=str(entry["source"]),
        )
        if candidate is None:
            continue
        candidates.append(candidate)
    candidates.sort(key=_candidate_sort_key)
    return tuple(candidates)


def _build_fallback_candidates(
    partitions: list[WorkloadAwarePartition],
    *,
    patterns_by_id: dict[int, PersistedLogicalPattern],
) -> dict[int, tuple[RouteCandidate, ...]]:
    fallback_candidates: dict[int, list[RouteCandidate]] = defaultdict(list)
    for partition in partitions:
        centroid = _normalize_vector(_parse_vector(partition.metadata.get("representative_centroid", [])))
        for tenant_id in partition.tenant_ids:
            matched_pattern_ids = []
            for pattern_id in partition.logical_pattern_ids:
                pattern = patterns_by_id.get(int(pattern_id))
                if pattern is None:
                    continue
                if int(tenant_id) in set(pattern.tenant_ids):
                    matched_pattern_ids.append(int(pattern_id))
            candidate = _build_route_candidate(
                tenant_id=int(tenant_id),
                partition=partition,
                matched_pattern_ids=tuple(matched_pattern_ids),
                patterns_by_id=patterns_by_id,
                base_score=float(_static_partition_score(partition, int(tenant_id))),
                source="fallback_tenant_membership",
            )
            if candidate is None:
                continue
            if not candidate.representative_centroid.size and centroid.size:
                candidate.representative_centroid = centroid
            fallback_candidates[int(tenant_id)].append(candidate)
    return {
        int(tenant_id): tuple(sorted(candidates, key=_candidate_sort_key))
        for tenant_id, candidates in fallback_candidates.items()
    }


def _build_route_index(*, refresh: bool = False) -> RouteIndex:
    plan_summary = get_current_plan_summary(refresh=refresh)
    if plan_summary is None:
        return RouteIndex(
            plan_id=0,
            entry_candidates_by_tenant={},
            fallback_candidates_by_tenant={},
            singleton_node_ids_by_tenant={},
            tenant_pattern_counts={},
            tenant_overlays_by_tenant={},
            access_overlays_by_key={},
        )

    partitions = load_current_partitions(refresh=refresh)
    logical_patterns = load_current_logical_patterns(refresh=refresh)
    tenant_overlays = load_current_tenant_overlays(refresh=refresh)
    access_overlays = load_current_access_overlays(refresh=refresh)
    partitions_by_id = {str(partition.partition_id): partition for partition in partitions}
    patterns_by_id = {int(pattern.pattern_id): pattern for pattern in logical_patterns}
    partitions_by_pattern: dict[int, list[WorkloadAwarePartition]] = defaultdict(list)
    for partition in partitions:
        for pattern_id in partition.logical_pattern_ids:
            partitions_by_pattern[int(pattern_id)].append(partition)
    partitions_by_pattern_tuple = {
        int(pattern_id): tuple(
            sorted(
                pattern_partitions,
                key=lambda partition: (
                    int(partition.vector_count),
                    str(partition.partition_id),
                ),
            )
        )
        for pattern_id, pattern_partitions in partitions_by_pattern.items()
    }
    singleton_node_ids_by_tenant: dict[int, int] = {}

    tenant_pattern_ids: dict[int, set[int]] = defaultdict(set)
    for pattern in logical_patterns:
        for tenant_id in pattern.tenant_ids:
            tenant_pattern_ids[int(tenant_id)].add(int(pattern.pattern_id))

    entry_candidates_by_tenant = {
        int(tenant_id): _build_entry_candidates(
            int(tenant_id),
            pattern_ids,
            patterns_by_id,
            partitions_by_id,
            partitions_by_pattern_tuple,
        )
        for tenant_id, pattern_ids in tenant_pattern_ids.items()
        if pattern_ids
    }
    fallback_candidates_by_tenant = _build_fallback_candidates(
        partitions,
        patterns_by_id=patterns_by_id,
    )
    tenant_pattern_counts = {int(tenant_id): len(pattern_ids) for tenant_id, pattern_ids in tenant_pattern_ids.items()}
    tenant_overlays_by_tenant = {
        int(overlay["tenant_id"]): TenantOverlay(
            tenant_id=int(overlay["tenant_id"]),
            table_name=str(overlay["table_name"]),
            document_count=int(overlay.get("document_count", 0) or 0),
            vector_count=int(overlay.get("vector_count", 0) or 0),
            query_mass=float(overlay.get("query_mass", 0.0) or 0.0),
            covered_partition_count=int(overlay.get("covered_partition_count", 1) or 1),
            estimated_saved_cost=float(overlay.get("estimated_saved_cost", 0.0) or 0.0),
            benefit_density=float(overlay.get("benefit_density", 0.0) or 0.0),
        )
        for overlay in tenant_overlays
    }
    access_overlays_by_key: dict[tuple[int, str], AccessOverlay] = {}
    for overlay in access_overlays:
        partition_ids = tuple(
            str(partition_id)
            for partition_id in (overlay.get("partition_ids", []) or [overlay["partition_id"]])
        )
        access_overlay = AccessOverlay(
            tenant_id=int(overlay["tenant_id"]),
            partition_id=str(overlay["partition_id"]),
            partition_ids=partition_ids,
            table_name=str(overlay["table_name"]),
            pattern_ids=tuple(int(pattern_id) for pattern_id in (overlay.get("pattern_ids", []) or [])),
            document_count=int(overlay.get("document_count", 0) or 0),
            vector_count=int(overlay.get("vector_count", 0) or 0),
            tenant_vector_count=int(overlay.get("tenant_vector_count", 0) or 0),
            query_mass=float(overlay.get("query_mass", 0.0) or 0.0),
            covered_partition_count=int(overlay.get("covered_partition_count", len(partition_ids)) or len(partition_ids)),
            estimated_saved_cost=float(overlay.get("estimated_saved_cost", 0.0) or 0.0),
            benefit_density=float(overlay.get("benefit_density", 0.0) or 0.0),
            overlay_type=str(overlay.get("overlay_type", "") or ""),
            requires_pattern_filter=bool(overlay.get("requires_pattern_filter", False)),
        )
        for partition_id in partition_ids:
            access_overlays_by_key[(int(overlay["tenant_id"]), str(partition_id))] = access_overlay
    return RouteIndex(
        plan_id=int(plan_summary["plan_id"]),
        entry_candidates_by_tenant=entry_candidates_by_tenant,
        fallback_candidates_by_tenant=fallback_candidates_by_tenant,
        singleton_node_ids_by_tenant=singleton_node_ids_by_tenant,
        tenant_pattern_counts=tenant_pattern_counts,
        tenant_overlays_by_tenant=tenant_overlays_by_tenant,
        access_overlays_by_key=access_overlays_by_key,
    )


def _get_route_index(*, refresh: bool = False) -> RouteIndex:
    global _CACHED_ROUTE_INDEX
    plan_summary = get_current_plan_summary(refresh=refresh)
    plan_id = int(plan_summary["plan_id"]) if plan_summary is not None else 0
    if not refresh and _CACHED_ROUTE_INDEX is not None and int(_CACHED_ROUTE_INDEX.plan_id) == plan_id:
        return _CACHED_ROUTE_INDEX
    _CACHED_ROUTE_INDEX = _build_route_index(refresh=refresh)
    return _CACHED_ROUTE_INDEX


def get_tenant_partition_route(
    tenant_id: int,
    query_vector,
    *,
    route_limit: Optional[int] = None,
    topk: Optional[int] = None,
) -> WorkloadAwareRoute:
    route_index = _get_route_index()
    tenant_id = int(tenant_id)
    effective_route_limit = route_limit if route_limit is not None else _configured_int(
        "method_route_limit",
        64,
        aliases=("dynamic_partition_route_limit",),
    )
    effective_route_limit = max(1, int(effective_route_limit))

    candidates = route_index.entry_candidates_by_tenant.get(tenant_id, ())
    fallback_used = False
    if not candidates:
        fallback_used = True
        candidates = route_index.fallback_candidates_by_tenant.get(tenant_id, ())
    if not candidates:
        return WorkloadAwareRoute(
            tenant_id=tenant_id,
            partition_ids=(),
            partition_count=0,
            selected_candidates=(),
            metadata={"found": False},
        )

    query_array = _normalize_vector(_parse_vector(query_vector))
    semantic_setting = _configured_setting(
        "method_route_semantic_weight",
        "dynamic_partition_route_semantic_weight",
    )
    if query_array.size:
        ranked_candidates = []
        base_scores = [float(candidate.base_score) for candidate in candidates]
        coverage_scores = [
            math.log1p(max(0, int(candidate.matched_vector_count)))
            for candidate in candidates
        ]
        workload_scores = [
            math.log1p(max(0.0, float(candidate.matched_query_mass)))
            for candidate in candidates
        ]
        normalized_base_scores = _normalize_score_values(base_scores)
        normalized_coverage_scores = _normalize_score_values(coverage_scores)
        normalized_workload_scores = _normalize_score_values(workload_scores)
        semantic_weight = _configured_float(
            "method_route_semantic_weight",
            0.4,
            aliases=("dynamic_partition_route_semantic_weight",),
        )
        base_weight = _configured_float(
            "method_route_base_weight",
            0.1,
            aliases=("dynamic_partition_route_base_weight",),
        )
        coverage_weight = _configured_float(
            "method_route_coverage_weight",
            0.4,
            aliases=("dynamic_partition_route_coverage_weight",),
        )
        workload_route_weight = _configured_float(
            "method_route_candidate_workload_weight",
            0.1,
            aliases=("dynamic_partition_route_candidate_workload_weight",),
        )
        for candidate in candidates:
            index = len(ranked_candidates)
            semantic_score = _semantic_score_for_candidate(candidate, query_array)
            normalized_semantic_score = float((semantic_score + 1.0) / 2.0)
            if semantic_setting is None:
                combined_score = (
                    float(semantic_weight) * normalized_semantic_score
                    + float(base_weight) * float(normalized_base_scores[index])
                    + float(coverage_weight) * float(normalized_coverage_scores[index])
                    + float(workload_route_weight) * float(normalized_workload_scores[index])
                )
            else:
                combined_score = (
                    float(candidate.base_score)
                    + float(semantic_weight) * float(semantic_score)
                    + float(coverage_weight) * float(normalized_coverage_scores[index])
                    + float(workload_route_weight) * float(normalized_workload_scores[index])
                )
            ranked_candidates.append(
                (
                    float(combined_score),
                    candidate.partition_id,
                    candidate,
                    float(semantic_score),
                )
            )
        ranked_candidates.sort(key=lambda item: (-float(item[0]), str(item[1])))
        ordered_candidates = [candidate for _, _, candidate, _ in ranked_candidates]
    else:
        ordered_candidates = list(candidates)

    candidates_by_partition = {
        str(candidate.partition_id): candidate
        for candidate in ordered_candidates
    }
    tenant_access_overlays = route_index.access_overlays_by_key
    selected_candidates: list[RouteCandidate] = []
    selected_partition_ids: set[str] = set()
    selected_physical_keys: set[tuple[str, str]] = set()

    def add_candidate_with_overlay(candidate: RouteCandidate) -> None:
        physical_key = _route_physical_key(int(tenant_id), candidate, tenant_access_overlays)
        if physical_key in selected_physical_keys:
            return
        selected_physical_keys.add(physical_key)
        selected_candidates.append(candidate)
        selected_partition_ids.add(str(candidate.partition_id))
        access_overlay = tenant_access_overlays.get((int(tenant_id), str(candidate.partition_id)))
        if access_overlay is None:
            return
        for partition_id in access_overlay.partition_ids:
            covered_candidate = candidates_by_partition.get(str(partition_id))
            if covered_candidate is None or str(partition_id) in selected_partition_ids:
                continue
            selected_candidates.append(covered_candidate)
            selected_partition_ids.add(str(partition_id))

    for access_overlay in _unique_access_overlays_for_tenant(int(tenant_id), tenant_access_overlays):
        if len(selected_physical_keys) >= effective_route_limit:
            break
        seed_candidate = None
        for partition_id in access_overlay.partition_ids:
            seed_candidate = candidates_by_partition.get(str(partition_id))
            if seed_candidate is not None:
                break
        if seed_candidate is None:
            continue
        add_candidate_with_overlay(seed_candidate)

    for candidate in ordered_candidates:
        if len(selected_physical_keys) >= effective_route_limit:
            break
        if str(candidate.partition_id) in selected_partition_ids:
            continue
        add_candidate_with_overlay(candidate)

    total_accessible_vectors = sum(max(0, int(candidate.matched_vector_count)) for candidate in ordered_candidates)
    selected_accessible_vectors = _selected_accessible_vector_count(
        selected_candidates,
        tenant_id=int(tenant_id),
        candidates_by_partition=candidates_by_partition,
        access_overlays_by_key=tenant_access_overlays,
    )
    base_selected_accessible_vectors = int(selected_accessible_vectors)
    coverage_guard_used = False

    physical_selected_candidates: list[RouteCandidate] = []
    seen_physical_keys: set[tuple[str, str]] = set()
    for candidate in selected_candidates:
        physical_key = _route_physical_key(int(tenant_id), candidate, tenant_access_overlays)
        if physical_key in seen_physical_keys:
            continue
        seen_physical_keys.add(physical_key)
        physical_selected_candidates.append(candidate)
    selected_candidates = physical_selected_candidates

    return WorkloadAwareRoute(
        tenant_id=tenant_id,
        partition_ids=tuple(candidate.partition_id for candidate in selected_candidates),
        partition_count=len(selected_candidates),
        selected_candidates=tuple(selected_candidates),
        metadata={
            "found": bool(selected_candidates),
            "route_limit": int(effective_route_limit),
            "route_coverage_target": None,
            "route_coverage_guard_used": bool(coverage_guard_used),
            "route_semantic_guard_used": False,
            "selected_accessible_vector_count": int(selected_accessible_vectors),
            "total_accessible_vector_count": int(total_accessible_vectors),
            "base_selected_accessible_vector_coverage": (
                float(base_selected_accessible_vectors / float(total_accessible_vectors))
                if total_accessible_vectors > 0
                else 0.0
            ),
            "selected_accessible_vector_coverage": (
                float(selected_accessible_vectors / float(total_accessible_vectors))
                if total_accessible_vectors > 0
                else 0.0
            ),
            "candidate_partition_count": len(candidates),
            "candidate_pattern_count": int(route_index.tenant_pattern_counts.get(tenant_id, 0)),
            "singleton_node_id": route_index.singleton_node_ids_by_tenant.get(tenant_id),
            "table_names": [candidate.table_name for candidate in selected_candidates],
            "sources": [candidate.source for candidate in selected_candidates],
            "matched_pattern_counts": [len(candidate.matched_pattern_ids) for candidate in selected_candidates],
            "matched_document_counts": [int(candidate.matched_document_count) for candidate in selected_candidates],
            "matched_vector_counts": [int(candidate.matched_vector_count) for candidate in selected_candidates],
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


def _candidate_fetch_limit(
    candidate: RouteCandidate,
    *,
    topk: int,
    fetch_multiplier: int,
) -> int:
    base_limit = max(int(topk), int(topk) * max(1, int(fetch_multiplier)))
    partition_vector_count = max(int(candidate.partition_vector_count), 1)
    accessible_vector_count = max(1, int(candidate.matched_vector_count))
    selectivity = min(1.0, float(accessible_vector_count) / float(partition_vector_count))
    recall_guard = math.sqrt(1.0 / max(selectivity, 1e-9))
    inflation = max(1.0, min(recall_guard, 8.0))
    computed_limit = int(math.ceil(float(base_limit) * inflation))
    return max(base_limit, min(computed_limit, max(int(candidate.partition_vector_count), base_limit)))


def _use_exact_filtered_branch(candidate: RouteCandidate, *, topk: int) -> bool:
    effective_topk = max(1, int(topk))
    accessible_vector_count = max(1, int(candidate.matched_vector_count))
    partition_vector_count = max(1, int(candidate.partition_vector_count))
    selectivity = float(accessible_vector_count) / float(partition_vector_count)
    exact_vector_limit = int(effective_topk ** 3)
    selectivity_limit = float(1.0 / float(effective_topk))
    return accessible_vector_count <= exact_vector_limit and selectivity <= selectivity_limit


def _build_partition_union_query(
    route: WorkloadAwareRoute,
    *,
    query_vector,
    fetch_multiplier: int,
    topk: int,
):
    branch_queries = []
    params = []
    selected_candidates = route.selected_candidates or tuple(
        RouteCandidate(
            partition_id=str(partition_id),
            table_name=get_partition_table_name(partition_id),
            base_score=0.0,
            representative_centroid=np.zeros(0, dtype=np.float32),
            prototype_centroids=(),
            matched_pattern_ids=(),
            accelerator_patterns=(),
            matched_document_count=0,
            matched_vector_count=0,
            matched_query_mass=0.0,
            partition_document_count=0,
            partition_vector_count=0,
            storage_layout_version=1,
            source="legacy_route",
        )
        for partition_id in route.partition_ids
    )
    emitted_access_overlay_tables: set[str] = set()
    for candidate in selected_candidates:
        branch_limit = _candidate_fetch_limit(
            candidate,
            topk=int(topk),
            fetch_multiplier=int(fetch_multiplier),
        )
        access_overlay = _get_access_overlay(int(route.tenant_id), str(candidate.partition_id))
        if access_overlay is not None:
            if str(access_overlay.table_name) in emitted_access_overlay_tables:
                continue
            emitted_access_overlay_tables.add(str(access_overlay.table_name))
            overlay_limit = min(
                max(int(branch_limit), int(topk) * max(1, int(fetch_multiplier))),
                max(int(access_overlay.vector_count), int(topk)),
            )
            if access_overlay.requires_pattern_filter and access_overlay.tenant_vector_count > 0:
                overlay_selectivity = max(
                    float(access_overlay.tenant_vector_count) / float(max(int(access_overlay.vector_count), 1)),
                    1e-9,
                )
                filtered_limit = int(math.ceil(float(overlay_limit) * math.sqrt(1.0 / overlay_selectivity)))
                overlay_limit = min(
                    max(int(filtered_limit), int(topk) * max(1, int(fetch_multiplier))),
                    max(int(access_overlay.vector_count), int(topk)),
                )
            if access_overlay.requires_pattern_filter and access_overlay.pattern_ids:
                branch_queries.append(
                    sql.SQL(
                        """
                        SELECT p.block_id, p.document_id, p.block_content,
                        p.vector <-> %s::vector AS distance
                        FROM {} p
                        WHERE p.pattern_id = ANY(%s)
                        ORDER BY distance
                        LIMIT %s
                        """
                    ).format(sql.Identifier(access_overlay.table_name))
                )
                params.extend([
                    query_vector,
                    list(int(pattern_id) for pattern_id in access_overlay.pattern_ids),
                    int(overlay_limit),
                ])
            else:
                branch_queries.append(
                    sql.SQL(
                        """
                        SELECT p.block_id, p.document_id, p.block_content,
                        p.vector <-> %s::vector AS distance
                        FROM {} p
                        ORDER BY distance
                        LIMIT %s
                        """
                    ).format(sql.Identifier(access_overlay.table_name))
                )
                params.extend([
                    query_vector,
                    int(overlay_limit),
                ])
            continue

        accelerator_pattern_ids = {
            int(entry["pattern_id"])
            for entry in candidate.accelerator_patterns
            if "pattern_id" in entry and "table_name" in entry
        }
        for accelerator_pattern in candidate.accelerator_patterns:
            accelerator_table_name = str(accelerator_pattern["table_name"])
            accelerator_pattern_id = int(accelerator_pattern["pattern_id"])
            accelerator_limit = max(
                int(topk),
                min(
                    int(branch_limit),
                    max(int(accelerator_pattern.get("vector_count", branch_limit) or branch_limit), int(topk)),
                ),
            )
            branch_queries.append(
                sql.SQL(
                    """
                    SELECT p.block_id, p.document_id, p.block_content,
                    p.vector <-> %s::vector AS distance
                    FROM {} p
                    ORDER BY distance
                    LIMIT %s
                    """
                ).format(sql.Identifier(accelerator_table_name))
            )
            params.extend([
                query_vector,
                int(accelerator_limit),
            ])

        remaining_pattern_ids = [
            int(pattern_id)
            for pattern_id in candidate.matched_pattern_ids
            if int(pattern_id) not in accelerator_pattern_ids
        ]
        if int(candidate.storage_layout_version) >= 2 and remaining_pattern_ids:
            exact_branch = _use_exact_filtered_branch(candidate, topk=int(topk))
            if exact_branch:
                branch_queries.append(
                    sql.SQL(
                        """
                        SELECT p.block_id, p.document_id, p.block_content,
                        p.vector <-> %s::vector AS distance
                        FROM {} p
                        WHERE p.pattern_id = ANY(%s)
                        ORDER BY (p.vector <-> %s::vector) + 0
                        LIMIT %s
                        """
                    ).format(sql.Identifier(candidate.table_name))
                )
                params.extend([
                    query_vector,
                    list(int(pattern_id) for pattern_id in remaining_pattern_ids),
                    query_vector,
                    int(topk),
                ])
            else:
                branch_queries.append(
                    sql.SQL(
                        """
                        SELECT p.block_id, p.document_id, p.block_content,
                        p.vector <-> %s::vector AS distance
                        FROM {} p
                        WHERE p.pattern_id = ANY(%s)
                        ORDER BY distance
                        LIMIT %s
                        """
                    ).format(sql.Identifier(candidate.table_name))
                )
                params.extend([
                    query_vector,
                    list(int(pattern_id) for pattern_id in remaining_pattern_ids),
                    int(branch_limit),
                ])
        elif not accelerator_pattern_ids:
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
                ).format(sql.Identifier(candidate.table_name))
            )
            params.extend([
                query_vector,
                int(route.tenant_id),
                int(branch_limit),
            ])

    if not branch_queries:
        return None, []

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


def _get_tenant_overlay(user_id: int) -> Optional[TenantOverlay]:
    route_index = _get_route_index()
    return route_index.tenant_overlays_by_tenant.get(int(user_id))


def _build_overlay_query(
    overlay: TenantOverlay,
    *,
    query_vector,
    topk: int,
):
    return (
        sql.SQL(
            """
            SELECT routed.block_id, routed.document_id, routed.block_content, routed.distance
            FROM (
                SELECT p.block_id, p.document_id, p.block_content,
                p.vector <-> %s::vector AS distance
                FROM {} p
                ORDER BY distance
                LIMIT %s
            ) AS routed
            ORDER BY distance
            LIMIT %s;
            """
        ).format(sql.Identifier(overlay.table_name)),
        [query_vector, int(topk), int(topk)],
    )


def _get_access_overlay(tenant_id: int, partition_id: str) -> Optional[AccessOverlay]:
    route_index = _get_route_index()
    return route_index.access_overlays_by_key.get((int(tenant_id), str(partition_id)))


def _configure_search_session(cur) -> None:
    cur.execute("SET max_parallel_workers_per_gather = 0;")
    cur.execute("SET jit = off;")
    fixed_ef = _configured_fixed_ef_search()
    if fixed_ef is not None:
        cur.execute(f"SET hnsw.ef_search = {fixed_ef};")


def dynamic_partition_search(user_id: int, query_vector, topk: int = 5, statistics_type: str = "sql"):
    if statistics_type == "sql":
        return dynamic_partition_search_statistics_sql(user_id, query_vector, topk)
    if statistics_type == "system":
        return dynamic_partition_search_statistics_system(user_id, query_vector, topk)
    raise ValueError(f"Unknown statistics type: {statistics_type}")


def dynamic_partition_search_statistics_sql(user_id: int, query_vector, topk: int = 5):
    overlay = _get_tenant_overlay(int(user_id))
    if overlay is not None:
        query, params = _build_overlay_query(
            overlay,
            query_vector=query_vector,
            topk=int(topk),
        )
        conn = get_db_connection()
        total_query_time = 0.0
        all_results = []
        try:
            with conn.cursor() as cur:
                _configure_search_session(cur)
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

    route = get_tenant_partition_route(user_id, query_vector, topk=int(topk))
    if not route.partition_ids:
        return [], 0.0

    fetch_multiplier = _configured_int(
        "method_partition_fetch_multiplier",
        4,
        aliases=("dynamic_partition_partition_fetch_multiplier",),
    )
    query, params = _build_partition_union_query(
        route,
        query_vector=query_vector,
        fetch_multiplier=int(fetch_multiplier),
        topk=int(topk),
    )
    if query is None:
        return [], 0.0
    conn = get_db_connection()
    total_query_time = 0.0
    all_results = []
    try:
        with conn.cursor() as cur:
            _configure_search_session(cur)
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
    conn = get_db_connection()
    started_at = time.time()
    overlay = _get_tenant_overlay(int(user_id))
    if overlay is not None:
        query, params = _build_overlay_query(
            overlay,
            query_vector=query_vector,
            topk=int(topk),
        )
        all_results = []
        try:
            with conn.cursor() as cur:
                _configure_search_session(cur)
                cur.execute(query, params)
                all_results.extend(cur.fetchall())
        finally:
            conn.close()
        return _merge_results(all_results, topk), time.time() - started_at

    route = get_tenant_partition_route(user_id, query_vector, topk=int(topk))
    if not route.partition_ids:
        conn.close()
        return [], 0.0

    fetch_multiplier = _configured_int(
        "method_partition_fetch_multiplier",
        4,
        aliases=("dynamic_partition_partition_fetch_multiplier",),
    )
    query, params = _build_partition_union_query(
        route,
        query_vector=query_vector,
        fetch_multiplier=int(fetch_multiplier),
        topk=int(topk),
    )
    if query is None:
        conn.close()
        return [], 0.0
    all_results = []
    try:
        with conn.cursor() as cur:
            _configure_search_session(cur)
            cur.execute(query, params)
            all_results.extend(cur.fetchall())
    finally:
        conn.close()
    return _merge_results(all_results, topk), time.time() - started_at
