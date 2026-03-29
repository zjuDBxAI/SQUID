"""Honeybee-aligned and tenant-aware cost model helpers for adaptive_tenant.

This module keeps the original Honeybee analytical helpers while adding the
extra signals needed by the adaptive tenant design:

- memory-bounded planning
- overlap-aware merge scoring
- tenant-level split gain estimation
- pollution-aware ef_search correction for shared partitions
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence


DEFAULT_PARAMETER_FILE = (
    Path(__file__).resolve().parents[1] / 'dynamic_partition' / 'hnsw' / 'parameter_hnsw.json'
)
DEFAULT_POLLUTION_WEIGHT = 0.35
DEFAULT_SENSITIVITY_FLOOR = 1e-6
DEFAULT_REBUILD_WEIGHT = 0.25
DEFAULT_QUERY_RATE_FALLBACK = 1.0


@dataclass(slots=True)
class HoneybeeHNSWParameters:
    k: float
    beta: float
    a: float
    b: float
    join_times: float = 0.0


@dataclass(slots=True)
class HoneybeeQueryCostEstimate:
    ef_search: float
    query_time: float
    weighted_partition_count: float
    average_partition_load: float
    selectivity: float


@dataclass(slots=True)
class AdaptiveCostWeights:
    lambda_memory: float = 1.0
    lambda_overlap: float = 1.0
    lambda_rebuild: float = DEFAULT_REBUILD_WEIGHT
    pollution_weight: float = DEFAULT_POLLUTION_WEIGHT


@dataclass(slots=True)
class AdaptivePartitionCostEstimate:
    ef_search: float
    query_cost: float
    memory_cost: float
    selectivity: float
    pollution: float
    query_rate: float
    vector_count: int
    document_count: int


@dataclass(slots=True)
class AdaptiveTenantCostEstimate:
    tenant_id: int
    partition_id: str
    ef_search: float
    query_cost: float
    sensitivity: float
    pollution: float
    query_rate: float
    vector_count: int
    partition_vector_count: int


@dataclass(slots=True)
class MergeScoreEstimate:
    score: float
    overlap_ratio: float
    delta_query_cost: float
    delta_memory: float
    merged_query_cost: float
    merged_memory: float


@dataclass(slots=True)
class SplitScoreEstimate:
    score: float
    gain: float
    rebuild_cost: float
    delta_query_cost: float
    delta_memory: float
    target_selectivity: float
    target_pollution: float


def _load_parameter_file(parameter_file: Optional[str | Path] = None) -> dict:
    path = Path(parameter_file) if parameter_file is not None else DEFAULT_PARAMETER_FILE
    with path.open('r', encoding='utf-8') as fh:
        return json.load(fh)


def load_honeybee_hnsw_parameters(
    parameter_file: Optional[str | Path] = None,
    *,
    regenerate_if_missing: bool = False,
) -> HoneybeeHNSWParameters:
    path = Path(parameter_file) if parameter_file is not None else DEFAULT_PARAMETER_FILE
    if regenerate_if_missing and not path.exists():
        from controller.dynamic_partition.get_parameter import save_parameter_to_json

        save_parameter_to_json(index_type='hnsw')

    payload = _load_parameter_file(path)
    return HoneybeeHNSWParameters(
        k=float(payload['k']),
        beta=float(payload['beta']),
        a=float(payload['a']),
        b=float(payload['b']),
        join_times=float(payload.get('join_times', 0.0) or 0.0),
    )


def predict_honeybee_ef_search(
    *,
    sel_whole: float,
    topk: int,
    k: float,
    beta: float,
    recall: Optional[float] = None,
) -> float:
    """Reuse Honeybee's HNSW ef_search prediction formula."""
    if recall is None:
        x = 3
        while (1 + x / 10) - k >= 1:
            x -= 1
        dynamic_value = 1 + x / 10
    else:
        dynamic_value = recall + 1 / 2

    safe_sel = max(float(sel_whole), DEFAULT_SENSITIVITY_FLOOR)
    delta = max(dynamic_value - float(k), DEFAULT_SENSITIVITY_FLOOR)
    inner = 1 / delta - 1
    if inner <= 0:
        inner = DEFAULT_SENSITIVITY_FLOOR
    safe_beta = beta if abs(beta) > DEFAULT_SENSITIVITY_FLOOR else DEFAULT_SENSITIVITY_FLOOR

    return math.log(inner) / (-4 * safe_beta * safe_sel) * topk + k * topk / safe_sel


def estimate_partition_memory(vector_count: int) -> float:
    n = max(int(vector_count), 0)
    if n <= 0:
        return 0.0
    return float(n) * math.log1p(float(n))


def estimate_memory_budget(total_vector_count: int, alpha: float) -> float:
    if alpha <= 0:
        raise ValueError('alpha must be positive')
    return float(alpha) * estimate_partition_memory(total_vector_count)


def compute_document_overlap_ratio(
    left_document_ids: Iterable[int],
    right_document_ids: Iterable[int],
) -> float:
    left = set(left_document_ids)
    right = set(right_document_ids)
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


def compute_tenant_sensitivity(
    *,
    tenant_cardinality: int,
    partition_cardinality: int,
    probe_tenant_hits: Optional[int] = None,
    probe_total_hits: Optional[int] = None,
) -> float:
    if probe_tenant_hits is not None and probe_total_hits is not None and probe_total_hits > 0:
        return max(float(probe_tenant_hits) / float(probe_total_hits), DEFAULT_SENSITIVITY_FLOOR)
    if partition_cardinality <= 0:
        return 1.0
    return max(min(float(tenant_cardinality) / float(partition_cardinality), 1.0), DEFAULT_SENSITIVITY_FLOOR)


def compute_partition_pollution(
    *,
    tenant_cardinalities: Mapping[int, int] | Sequence[int],
    partition_cardinality: int,
) -> float:
    if partition_cardinality <= 0:
        return 0.0
    if isinstance(tenant_cardinalities, Mapping):
        values = [max(int(value), 0) for value in tenant_cardinalities.values()]
    else:
        values = [max(int(value), 0) for value in tenant_cardinalities]
    if not values:
        return 0.0
    dominant_share = max(min(float(value) / float(partition_cardinality), 1.0) for value in values)
    return max(0.0, 1.0 - dominant_share)


def predict_adaptive_ef_search(
    *,
    selectivity: float,
    topk: int,
    parameters: HoneybeeHNSWParameters,
    recall_target: Optional[float] = None,
    pollution: float = 0.0,
    pollution_weight: float = DEFAULT_POLLUTION_WEIGHT,
) -> float:
    base = predict_honeybee_ef_search(
        sel_whole=selectivity,
        topk=topk,
        k=parameters.k,
        beta=parameters.beta,
        recall=recall_target,
    )
    multiplier = 1.0 + max(float(pollution), 0.0) * max(float(pollution_weight), 0.0)
    return base * multiplier


def compute_adaptive_partition_query_cost(
    *,
    vector_count: int,
    document_count: int,
    query_rate: float,
    selectivity: float,
    parameters: HoneybeeHNSWParameters,
    topk: int,
    recall_target: Optional[float] = None,
    pollution: float = 0.0,
    memory_cost: Optional[float] = None,
    pollution_weight: float = DEFAULT_POLLUTION_WEIGHT,
) -> AdaptivePartitionCostEstimate:
    safe_vectors = max(int(vector_count), 1)
    safe_documents = max(int(document_count), 1)
    safe_query_rate = max(float(query_rate), DEFAULT_QUERY_RATE_FALLBACK)
    safe_selectivity = max(float(selectivity), DEFAULT_SENSITIVITY_FLOOR)
    ef_search = predict_adaptive_ef_search(
        selectivity=safe_selectivity,
        topk=topk,
        parameters=parameters,
        recall_target=recall_target,
        pollution=pollution,
        pollution_weight=pollution_weight,
    )
    query_cost = math.log(float(safe_vectors)) * (parameters.a * ef_search + parameters.b) * safe_query_rate
    return AdaptivePartitionCostEstimate(
        ef_search=ef_search,
        query_cost=query_cost,
        memory_cost=estimate_partition_memory(safe_vectors) if memory_cost is None else float(memory_cost),
        selectivity=safe_selectivity,
        pollution=max(float(pollution), 0.0),
        query_rate=safe_query_rate,
        vector_count=safe_vectors,
        document_count=safe_documents,
    )


def compute_adaptive_tenant_partition_cost(
    *,
    tenant_id: int,
    partition_id: str,
    tenant_vector_count: int,
    partition_vector_count: int,
    tenant_document_count: int,
    partition_document_count: int,
    tenant_query_rate: float,
    parameters: HoneybeeHNSWParameters,
    topk: int,
    recall_target: Optional[float] = None,
    sensitivity: Optional[float] = None,
    pollution: float = 0.0,
    pollution_weight: float = DEFAULT_POLLUTION_WEIGHT,
) -> AdaptiveTenantCostEstimate:
    safe_partition_vectors = max(int(partition_vector_count), 1)
    safe_tenant_vectors = max(int(tenant_vector_count), 1)
    safe_tenant_documents = max(int(tenant_document_count), 1)
    safe_partition_documents = max(int(partition_document_count), 1)
    tenant_sensitivity = compute_tenant_sensitivity(
        tenant_cardinality=safe_tenant_documents,
        partition_cardinality=safe_partition_documents,
    ) if sensitivity is None else max(float(sensitivity), DEFAULT_SENSITIVITY_FLOOR)
    ef_search = predict_adaptive_ef_search(
        selectivity=tenant_sensitivity,
        topk=topk,
        parameters=parameters,
        recall_target=recall_target,
        pollution=pollution,
        pollution_weight=pollution_weight,
    )
    query_cost = (
        math.log(float(safe_tenant_vectors))
        * (parameters.a * ef_search + parameters.b)
        * max(float(tenant_query_rate), DEFAULT_QUERY_RATE_FALLBACK)
        / max(tenant_sensitivity, DEFAULT_SENSITIVITY_FLOOR)
    )
    return AdaptiveTenantCostEstimate(
        tenant_id=int(tenant_id),
        partition_id=str(partition_id),
        ef_search=ef_search,
        query_cost=query_cost,
        sensitivity=tenant_sensitivity,
        pollution=max(float(pollution), 0.0),
        query_rate=max(float(tenant_query_rate), DEFAULT_QUERY_RATE_FALLBACK),
        vector_count=safe_tenant_vectors,
        partition_vector_count=safe_partition_vectors,
    )


def score_merge_candidate(
    *,
    source_vectors: int,
    source_documents: int,
    source_query_rate: float,
    candidate_vectors: int,
    candidate_documents: int,
    candidate_query_rate: float,
    merged_vectors: int,
    merged_documents: int,
    merged_query_rate: float,
    overlap_ratio: float,
    merged_selectivity: float,
    merged_pollution: float,
    parameters: HoneybeeHNSWParameters,
    topk: int,
    recall_target: Optional[float] = None,
    weights: Optional[AdaptiveCostWeights] = None,
) -> MergeScoreEstimate:
    weights = weights or AdaptiveCostWeights()
    left = compute_adaptive_partition_query_cost(
        vector_count=source_vectors,
        document_count=source_documents,
        query_rate=source_query_rate,
        selectivity=1.0,
        parameters=parameters,
        topk=topk,
        recall_target=recall_target,
        pollution=0.0,
        pollution_weight=weights.pollution_weight,
    )
    right = compute_adaptive_partition_query_cost(
        vector_count=candidate_vectors,
        document_count=candidate_documents,
        query_rate=candidate_query_rate,
        selectivity=1.0,
        parameters=parameters,
        topk=topk,
        recall_target=recall_target,
        pollution=0.0,
        pollution_weight=weights.pollution_weight,
    )
    merged = compute_adaptive_partition_query_cost(
        vector_count=merged_vectors,
        document_count=merged_documents,
        query_rate=merged_query_rate,
        selectivity=merged_selectivity,
        parameters=parameters,
        topk=topk,
        recall_target=recall_target,
        pollution=merged_pollution,
        pollution_weight=weights.pollution_weight,
    )
    delta_query_cost = merged.query_cost - left.query_cost - right.query_cost
    delta_memory = merged.memory_cost - left.memory_cost - right.memory_cost
    score = (
        delta_query_cost
        + weights.lambda_memory * delta_memory
        - weights.lambda_overlap * max(float(overlap_ratio), 0.0)
    )
    return MergeScoreEstimate(
        score=score,
        overlap_ratio=max(float(overlap_ratio), 0.0),
        delta_query_cost=delta_query_cost,
        delta_memory=delta_memory,
        merged_query_cost=merged.query_cost,
        merged_memory=merged.memory_cost,
    )


def score_split_candidate(
    *,
    source_before_cost: float,
    source_after_cost: float,
    target_after_cost: float,
    current_memory: float,
    future_memory: float,
    moved_vector_count: int,
    target_selectivity: float,
    target_pollution: float,
    weights: Optional[AdaptiveCostWeights] = None,
) -> SplitScoreEstimate:
    weights = weights or AdaptiveCostWeights()
    delta_query_cost = (source_after_cost + target_after_cost) - source_before_cost
    delta_memory = float(future_memory) - float(current_memory)
    rebuild_cost = weights.lambda_rebuild * estimate_partition_memory(moved_vector_count)
    gain = -delta_query_cost - rebuild_cost
    score = gain - weights.lambda_memory * max(delta_memory, 0.0)
    return SplitScoreEstimate(
        score=score,
        gain=gain,
        rebuild_cost=rebuild_cost,
        delta_query_cost=delta_query_cost,
        delta_memory=delta_memory,
        target_selectivity=max(float(target_selectivity), DEFAULT_SENSITIVITY_FLOOR),
        target_pollution=max(float(target_pollution), 0.0),
    )


def compute_honeybee_partition_query_time(
    *,
    partition_loads: Mapping[int, int] | Iterable[int],
    sel_whole: float,
    topk: int,
    parameters: HoneybeeHNSWParameters,
    partition_weights: Optional[Iterable[float]] = None,
    recall: Optional[float] = None,
) -> HoneybeeQueryCostEstimate:
    """Compute Honeybee-style partition query cost from partition sizes."""
    if isinstance(partition_loads, Mapping):
        loads = [int(value) for value in partition_loads.values() if int(value) > 0]
    else:
        loads = [int(value) for value in partition_loads if int(value) > 0]

    if not loads:
        ef = predict_honeybee_ef_search(
            sel_whole=sel_whole,
            topk=topk,
            k=parameters.k,
            beta=parameters.beta,
            recall=recall,
        )
        return HoneybeeQueryCostEstimate(
            ef_search=ef,
            query_time=0.0,
            weighted_partition_count=0.0,
            average_partition_load=0.0,
            selectivity=max(float(sel_whole), 0.0),
        )

    if partition_weights is None:
        weights = [1.0] * len(loads)
    else:
        weights = [float(weight) for weight in partition_weights]
        if len(weights) != len(loads):
            raise ValueError('partition_weights must match the number of partition loads')

    ef = predict_honeybee_ef_search(
        sel_whole=sel_whole,
        topk=topk,
        k=parameters.k,
        beta=parameters.beta,
        recall=recall,
    )

    total_query_time = 0.0
    total_weight = 0.0
    weighted_partition_count = 0.0
    weighted_load_sum = 0.0
    for load, weight in zip(loads, weights):
        if weight <= 0:
            continue
        total_query_time += weight * math.log(load) * (parameters.a * ef + parameters.b)
        total_weight += weight
        weighted_partition_count += weight
        weighted_load_sum += weight * load

    average_partition_load = weighted_load_sum / total_weight if total_weight > 0 else 0.0
    return HoneybeeQueryCostEstimate(
        ef_search=ef,
        query_time=total_query_time,
        weighted_partition_count=weighted_partition_count,
        average_partition_load=average_partition_load,
        selectivity=max(float(sel_whole), 0.0),
    )


def compute_honeybee_selectivity(
    *,
    tracked_partitions: Iterable[int],
    partition_loads: Mapping[int, int],
    accessible_document_ids: Iterable[int],
    partition_assignment: Mapping[int, Iterable[int]],
) -> float:
    """Reuse Honeybee's average partition selectivity idea for one access scope."""
    role_docs = set(accessible_document_ids)
    partition_sels: list[float] = []
    for partition_id in tracked_partitions:
        partition_docs = int(partition_loads.get(partition_id, 0) or 0)
        if partition_docs <= 0:
            continue
        assigned_docs = set(partition_assignment.get(partition_id, ()))
        sel = len(role_docs & assigned_docs) / partition_docs
        partition_sels.append(sel)
    return sum(partition_sels) / len(partition_sels) if partition_sels else 0.0


def compute_honeybee_weighted_selectivity(
    *,
    comb_to_partitions: Mapping[tuple[int, ...], Iterable[int]],
    partition_loads: Mapping[int, int],
    comb_to_documents: Mapping[tuple[int, ...], Iterable[int]],
    partition_assignment: Mapping[int, Iterable[int]],
    combination_weights: Optional[Mapping[tuple[int, ...], float]] = None,
) -> float:
    """Honeybee-style weighted overall selectivity across combinations."""
    total_weighted_sel = 0.0
    total_weight = 0.0

    for comb, tracked_partitions in comb_to_partitions.items():
        avg_sel = compute_honeybee_selectivity(
            tracked_partitions=tracked_partitions,
            partition_loads=partition_loads,
            accessible_document_ids=comb_to_documents.get(comb, ()),
            partition_assignment=partition_assignment,
        )
        weight = 1.0
        if combination_weights is not None:
            weight = float(combination_weights.get(comb, 0.0) or 0.0)
            if weight == 0.0 and comb:
                weight = 1.0
        total_weighted_sel += avg_sel * weight
        total_weight += weight

    return total_weighted_sel / total_weight if total_weight > 0 else 0.0
