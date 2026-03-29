"""Tenant-aware memory-bounded planner for adaptive_tenant.

This planner implements the control-plane policy behind the current idea:

- maintain a global memory budget ``alpha``
- prefer dedicated partitions when budget allows
- otherwise merge tenants into shared partitions with overlap-aware scoring
- periodically split expensive shared partitions when the budget has slack
- apply simple hysteresis via split/merge cooldown windows
"""

from __future__ import annotations

from dataclasses import dataclass, field
import itertools
from typing import Iterable, Optional, Sequence

from .cost_model import (
    AdaptiveCostWeights,
    DEFAULT_QUERY_RATE_FALLBACK,
    compute_adaptive_partition_query_cost,
    compute_document_overlap_ratio,
    compute_partition_pollution,
    compute_tenant_sensitivity,
    estimate_partition_memory,
    load_honeybee_hnsw_parameters,
    score_merge_candidate,
    score_split_candidate,
)
from .tenant_state import TenantStateRepository, TenantStateSnapshot


@dataclass(slots=True)
class PlannedPartition:
    partition_id: str
    tenant_ids: tuple[int, ...]
    tenant_names: tuple[str, ...]
    document_count: int
    vector_count: int
    estimated_memory: float
    query_rate: float
    write_rate: float
    recall_target: float
    tenant_document_counts: dict[int, int]
    tenant_vector_counts: dict[int, int]
    document_ids: frozenset[int] = field(default_factory=frozenset)
    metadata: dict[str, float | int | str] = field(default_factory=dict)


@dataclass(slots=True)
class PlannerBudget:
    alpha: float
    baseline_memory: float
    memory_limit: float
    current_memory: float
    total_vector_count: int
    total_query_rate: float


@dataclass(slots=True)
class MergeStep:
    left_partition_id: str
    right_partition_id: str
    merged_partition_id: str
    merged_tenant_ids: tuple[int, ...]
    overlap_ratio: float
    delta_query_cost: float
    delta_memory: float
    merged_memory: float
    score: float
    reason: str
    window_marker: int


@dataclass(slots=True)
class SplitStep:
    source_partition_id: str
    target_partition_id: str
    tenant_id: int
    source_tenant_ids_after: tuple[int, ...]
    target_tenant_ids_after: tuple[int, ...]
    score: float
    gain: float
    delta_query_cost: float
    delta_memory: float
    mode: str
    reason: str
    window_marker: int


@dataclass(slots=True)
class PlannerResult:
    action: str
    new_tenant_id: int
    new_partition_id: str
    partitions: list[PlannedPartition]
    budget: PlannerBudget
    total_query_cost: float
    merge_steps: list[MergeStep] = field(default_factory=list)
    split_steps: list[SplitStep] = field(default_factory=list)
    current_epoch: int = 0


class AdaptiveTenantPlanner:
    """Adaptive planner for tenant-aware dedicated/shared partitioning."""

    def __init__(
        self,
        *,
        alpha: float,
        tenant_state_repository: Optional[TenantStateRepository] = None,
        topk: int = 10,
        prefer_dedicated_on_budget: bool = True,
        split_threshold: float = 0.0,
        merge_threshold: float = 0.0,
        split_cooldown_windows: int = 1,
        merge_cooldown_windows: int = 1,
        candidate_merge_limit: int = 8,
        max_split_actions: int = 1,
        cost_weights: Optional[AdaptiveCostWeights] = None,
    ) -> None:
        if alpha <= 0:
            raise ValueError('alpha must be positive')
        if topk <= 0:
            raise ValueError('topk must be positive')
        self.alpha = float(alpha)
        self.topk = int(topk)
        self.prefer_dedicated_on_budget = bool(prefer_dedicated_on_budget)
        self.split_threshold = float(split_threshold)
        self.merge_threshold = float(merge_threshold)
        self.split_cooldown_windows = int(split_cooldown_windows)
        self.merge_cooldown_windows = int(merge_cooldown_windows)
        self.candidate_merge_limit = max(2, int(candidate_merge_limit))
        self.max_split_actions = max(0, int(max_split_actions))
        self.tenant_state_repository = tenant_state_repository or TenantStateRepository()
        self.cost_weights = cost_weights or AdaptiveCostWeights()
        self.parameters = load_honeybee_hnsw_parameters()
        self._partition_counter = itertools.count(1)

    def estimate_partition_memory(self, vector_count: int) -> float:
        return estimate_partition_memory(vector_count)

    def compute_budget(self, tenant_states: Sequence[TenantStateSnapshot]) -> PlannerBudget:
        baseline_memory = sum(self.estimate_partition_memory(state.size.vector_count) for state in tenant_states)
        memory_limit = self.alpha * baseline_memory
        total_vector_count = sum(int(state.size.vector_count) for state in tenant_states)
        total_query_rate = sum(max(float(state.query_rate_ema), DEFAULT_QUERY_RATE_FALLBACK) for state in tenant_states)
        return PlannerBudget(
            alpha=self.alpha,
            baseline_memory=baseline_memory,
            memory_limit=memory_limit,
            current_memory=baseline_memory,
            total_vector_count=total_vector_count,
            total_query_rate=total_query_rate,
        )

    def initialize_plan(
        self,
        *,
        tenant_ids: Optional[Iterable[int]] = None,
        window_limit: int = 10,
    ) -> PlannerResult:
        tenant_states = self.tenant_state_repository.get_all_tenant_states(
            tenant_ids=tenant_ids,
            window_limit=window_limit,
        )
        state_by_id, document_cache, document_block_counts, current_epoch = self._prepare_state_maps(tenant_states)
        partitions = self.build_singleton_partitions(
            tenant_states,
            document_cache=document_cache,
            document_block_counts=document_block_counts,
        )
        budget = self.compute_budget(tenant_states)
        partitions, merge_steps = self._reduce_to_budget(
            partitions,
            state_by_id=state_by_id,
            document_cache=document_cache,
            document_block_counts=document_block_counts,
            memory_limit=budget.memory_limit,
            current_epoch=current_epoch,
            reason='initialize under budget',
        )
        budget.current_memory = self._total_memory(partitions)
        total_query_cost = self._total_query_cost(partitions, state_by_id)
        return PlannerResult(
            action='initialize',
            new_tenant_id=-1,
            new_partition_id='',
            partitions=self._sort_partitions(partitions),
            budget=budget,
            total_query_cost=total_query_cost,
            merge_steps=merge_steps,
            current_epoch=current_epoch,
        )

    def build_singleton_partitions(
        self,
        tenant_states: Sequence[TenantStateSnapshot],
        *,
        document_cache: Optional[dict[int, set[int]]] = None,
        document_block_counts: Optional[dict[int, int]] = None,
    ) -> list[PlannedPartition]:
        state_by_id = {state.tenant_id: state for state in tenant_states}
        if document_cache is None:
            document_cache = self.tenant_state_repository.get_many_accessible_document_ids(state_by_id)
        if document_block_counts is None:
            all_document_ids = set().union(*(document_cache.get(tenant_id, set()) for tenant_id in state_by_id))
            document_block_counts = self.tenant_state_repository.get_document_block_counts(all_document_ids)
        return [
            self._build_partition_from_tenant_ids(
                (state.tenant_id,),
                state_by_id,
                document_cache,
                document_block_counts,
            )
            for state in tenant_states
        ]

    def place_new_tenant(
        self,
        new_tenant_state: TenantStateSnapshot,
        *,
        current_tenant_states: Sequence[TenantStateSnapshot],
        current_partitions: Optional[Sequence[PlannedPartition]] = None,
    ) -> PlannerResult:
        all_states = list(current_tenant_states) + [new_tenant_state]
        state_by_id, document_cache, document_block_counts, current_epoch = self._prepare_state_maps(all_states)
        budget = self.compute_budget(all_states)
        partitions = self._clone_partitions(
            current_partitions
            if current_partitions is not None
            else self.build_singleton_partitions(
                current_tenant_states,
                document_cache=document_cache,
                document_block_counts=document_block_counts,
            )
        )
        merge_steps: list[MergeStep] = []

        new_partition = self._build_partition_from_tenant_ids(
            (new_tenant_state.tenant_id,),
            state_by_id,
            document_cache,
            document_block_counts,
            partition_id=self._next_partition_id(prefix='tenant'),
        )
        tentative = partitions + [new_partition]

        if self._total_memory(tentative) <= budget.memory_limit and self.prefer_dedicated_on_budget:
            budget.current_memory = self._total_memory(tentative)
            return PlannerResult(
                action='create_partition',
                new_tenant_id=new_tenant_state.tenant_id,
                new_partition_id=new_partition.partition_id,
                partitions=self._sort_partitions(tentative),
                budget=budget,
                total_query_cost=self._total_query_cost(tentative, state_by_id),
                current_epoch=current_epoch,
            )

        if not partitions:
            budget.current_memory = self._total_memory([new_partition])
            return PlannerResult(
                action='create_partition_over_budget',
                new_tenant_id=new_tenant_state.tenant_id,
                new_partition_id=new_partition.partition_id,
                partitions=[new_partition],
                budget=budget,
                total_query_cost=self._total_query_cost([new_partition], state_by_id),
                current_epoch=current_epoch,
            )

        merge_choice = self._pick_best_merge_for_source(
            new_partition,
            partitions,
            state_by_id=state_by_id,
            document_cache=document_cache,
            document_block_counts=document_block_counts,
            current_epoch=current_epoch,
            respect_cooldown=True,
        )
        if merge_choice is None:
            merge_choice = self._pick_best_merge_for_source(
                new_partition,
                partitions,
                state_by_id=state_by_id,
                document_cache=document_cache,
                document_block_counts=document_block_counts,
                current_epoch=current_epoch,
                respect_cooldown=False,
            )
        if merge_choice is None:
            tentative = partitions + [new_partition]
            budget.current_memory = self._total_memory(tentative)
            return PlannerResult(
                action='create_partition_no_merge_candidate',
                new_tenant_id=new_tenant_state.tenant_id,
                new_partition_id=new_partition.partition_id,
                partitions=self._sort_partitions(tentative),
                budget=budget,
                total_query_cost=self._total_query_cost(tentative, state_by_id),
                current_epoch=current_epoch,
            )

        merged_partition, merge_step = merge_choice
        partitions = [partition for partition in partitions if partition.partition_id != merge_step.right_partition_id]
        partitions.append(merged_partition)
        merge_steps.append(merge_step)
        action = 'merge_with_candidate'

        if self._total_memory(partitions) > budget.memory_limit:
            partitions, recursive_steps = self._reduce_to_budget(
                partitions,
                state_by_id=state_by_id,
                document_cache=document_cache,
                document_block_counts=document_block_counts,
                memory_limit=budget.memory_limit,
                current_epoch=current_epoch,
                reason='recursive merge after tenant placement',
            )
            merge_steps.extend(recursive_steps)
            action = 'merge_with_candidate_then_recursive'

        budget.current_memory = self._total_memory(partitions)
        target_partition = self._find_partition_for_tenant(partitions, new_tenant_state.tenant_id)
        return PlannerResult(
            action=action,
            new_tenant_id=new_tenant_state.tenant_id,
            new_partition_id=target_partition.partition_id,
            partitions=self._sort_partitions(partitions),
            budget=budget,
            total_query_cost=self._total_query_cost(partitions, state_by_id),
            merge_steps=merge_steps,
            current_epoch=current_epoch,
        )

    def place_new_tenant_by_id(
        self,
        tenant_id: int,
        *,
        current_tenant_ids: Optional[Iterable[int]] = None,
        current_partitions: Optional[Sequence[PlannedPartition]] = None,
        window_limit: int = 10,
    ) -> PlannerResult:
        new_tenant_state = self.tenant_state_repository.get_tenant_state(
            tenant_id,
            window_limit=window_limit,
        )
        current_states = self.tenant_state_repository.get_all_tenant_states(
            tenant_ids=current_tenant_ids,
            window_limit=window_limit,
        )
        current_states = [state for state in current_states if state.tenant_id != tenant_id]
        return self.place_new_tenant(
            new_tenant_state,
            current_tenant_states=current_states,
            current_partitions=current_partitions,
        )

    def rebalance_partitions(
        self,
        partitions: Sequence[PlannedPartition],
        *,
        tenant_states: Sequence[TenantStateSnapshot],
        max_split_actions: Optional[int] = None,
    ) -> PlannerResult:
        state_by_id, document_cache, document_block_counts, current_epoch = self._prepare_state_maps(tenant_states)
        working = self._clone_partitions(partitions)
        budget = self.compute_budget(tenant_states)
        budget.current_memory = self._total_memory(working)
        split_steps: list[SplitStep] = []
        action = 'rebalance_noop'

        action_limit = max_split_actions if max_split_actions is not None else self.max_split_actions
        for _ in range(max(0, int(action_limit))):
            candidate = self._find_best_split_candidate(
                working,
                state_by_id=state_by_id,
                document_cache=document_cache,
                document_block_counts=document_block_counts,
                current_epoch=current_epoch,
                memory_limit=budget.memory_limit,
            )
            if candidate is None:
                break
            split_step, updated_partitions = candidate
            if split_step.score <= self.split_threshold:
                break
            working = updated_partitions
            split_steps.append(split_step)
            action = 'rebalance_split'

        budget.current_memory = self._total_memory(working)
        total_query_cost = self._total_query_cost(working, state_by_id)
        return PlannerResult(
            action=action,
            new_tenant_id=-1,
            new_partition_id='',
            partitions=self._sort_partitions(working),
            budget=budget,
            total_query_cost=total_query_cost,
            split_steps=split_steps,
            current_epoch=current_epoch,
        )

    def _prepare_state_maps(
        self,
        tenant_states: Sequence[TenantStateSnapshot],
    ) -> tuple[dict[int, TenantStateSnapshot], dict[int, set[int]], dict[int, int], int]:
        state_by_id = {state.tenant_id: state for state in tenant_states}
        document_cache = self.tenant_state_repository.get_many_accessible_document_ids(state_by_id)
        all_document_ids = set().union(*(document_cache.get(tenant_id, set()) for tenant_id in state_by_id))
        document_block_counts = self.tenant_state_repository.get_document_block_counts(all_document_ids)
        current_epoch = 0
        for state in tenant_states:
            current_epoch = max(current_epoch, self._latest_window_id(state))
        return state_by_id, document_cache, document_block_counts, current_epoch

    def _latest_window_id(self, state: TenantStateSnapshot) -> int:
        if not state.windows:
            return 0
        return max(int(window.window_id) for window in state.windows)

    def _build_partition_from_tenant_ids(
        self,
        tenant_ids: Iterable[int],
        state_by_id: dict[int, TenantStateSnapshot],
        document_cache: dict[int, set[int]],
        document_block_counts: dict[int, int],
        *,
        partition_id: Optional[str] = None,
    ) -> PlannedPartition:
        sorted_ids = tuple(sorted({int(tenant_id) for tenant_id in tenant_ids}))
        document_ids = frozenset().union(*(document_cache.get(tenant_id, set()) for tenant_id in sorted_ids))
        tenant_document_counts = {
            tenant_id: len(document_cache.get(tenant_id, set())) for tenant_id in sorted_ids
        }
        tenant_vector_counts = {
            tenant_id: int(state_by_id[tenant_id].size.vector_count) for tenant_id in sorted_ids
        }
        query_rate = sum(max(float(state_by_id[tenant_id].query_rate_ema), DEFAULT_QUERY_RATE_FALLBACK) for tenant_id in sorted_ids)
        write_rate = sum(float(state_by_id[tenant_id].write_rate_ema) for tenant_id in sorted_ids)
        recall_target = max((float(state_by_id[tenant_id].recall_target) for tenant_id in sorted_ids), default=0.95)
        document_count = len(document_ids)
        vector_count = self._estimate_vectors_for_documents(
            document_ids,
            document_block_counts,
            fallback=sum(tenant_vector_counts.values()),
        )
        estimated_memory = self.estimate_partition_memory(vector_count)
        pollution = compute_partition_pollution(
            tenant_cardinalities=tenant_document_counts,
            partition_cardinality=max(document_count, 1),
        )
        role_count = sum(int(state_by_id[tenant_id].size.role_count) for tenant_id in sorted_ids)
        return PlannedPartition(
            partition_id=partition_id or self._next_partition_id(prefix='p'),
            tenant_ids=sorted_ids,
            tenant_names=tuple(state_by_id[tenant_id].tenant_name for tenant_id in sorted_ids),
            document_count=document_count,
            vector_count=vector_count,
            estimated_memory=estimated_memory,
            query_rate=query_rate,
            write_rate=write_rate,
            recall_target=recall_target,
            tenant_document_counts=tenant_document_counts,
            tenant_vector_counts=tenant_vector_counts,
            document_ids=document_ids,
            metadata={
                'pollution': pollution,
                'role_count': role_count,
            },
        )

    def _estimate_vectors_for_documents(
        self,
        document_ids: Iterable[int],
        document_block_counts: dict[int, int],
        *,
        fallback: int = 0,
    ) -> int:
        vector_count = sum(int(document_block_counts.get(int(document_id), 0)) for document_id in document_ids)
        if vector_count > 0:
            return vector_count
        return int(fallback)

    def _partition_selectivity(self, partition: PlannedPartition, state_by_id: dict[int, TenantStateSnapshot]) -> float:
        if partition.document_count <= 0:
            return 1.0
        total_weight = 0.0
        weighted_selectivity = 0.0
        for tenant_id in partition.tenant_ids:
            weight = max(float(state_by_id[tenant_id].query_rate_ema), DEFAULT_QUERY_RATE_FALLBACK)
            selectivity = compute_tenant_sensitivity(
                tenant_cardinality=partition.tenant_document_counts.get(tenant_id, 0),
                partition_cardinality=partition.document_count,
            )
            weighted_selectivity += weight * selectivity
            total_weight += weight
        if total_weight <= 0:
            return max(
                (
                    compute_tenant_sensitivity(
                        tenant_cardinality=partition.tenant_document_counts.get(tenant_id, 0),
                        partition_cardinality=partition.document_count,
                    )
                    for tenant_id in partition.tenant_ids
                ),
                default=1.0,
            )
        return weighted_selectivity / total_weight

    def _partition_pollution(self, partition: PlannedPartition) -> float:
        return compute_partition_pollution(
            tenant_cardinalities=partition.tenant_document_counts,
            partition_cardinality=max(partition.document_count, 1),
        )

    def _estimate_partition_cost(
        self,
        partition: PlannedPartition,
        state_by_id: dict[int, TenantStateSnapshot],
    ) -> float:
        estimate = compute_adaptive_partition_query_cost(
            vector_count=partition.vector_count,
            document_count=partition.document_count,
            query_rate=partition.query_rate,
            selectivity=self._partition_selectivity(partition, state_by_id),
            parameters=self.parameters,
            topk=self.topk,
            recall_target=partition.recall_target,
            pollution=self._partition_pollution(partition),
            pollution_weight=self.cost_weights.pollution_weight,
        )
        return estimate.query_cost

    def _total_query_cost(
        self,
        partitions: Sequence[PlannedPartition],
        state_by_id: dict[int, TenantStateSnapshot],
    ) -> float:
        return sum(self._estimate_partition_cost(partition, state_by_id) for partition in partitions)

    def _pick_best_merge_for_source(
        self,
        source_partition: PlannedPartition,
        candidates: Sequence[PlannedPartition],
        *,
        state_by_id: dict[int, TenantStateSnapshot],
        document_cache: dict[int, set[int]],
        document_block_counts: dict[int, int],
        current_epoch: int,
        respect_cooldown: bool,
    ) -> Optional[tuple[PlannedPartition, MergeStep]]:
        ordered_candidates = sorted(
            (
                candidate for candidate in candidates
                if candidate.partition_id != source_partition.partition_id
            ),
            key=lambda candidate: (
                -compute_document_overlap_ratio(source_partition.document_ids, candidate.document_ids),
                candidate.estimated_memory,
                candidate.partition_id,
            ),
        )[: self.candidate_merge_limit]

        best_choice: Optional[tuple[PlannedPartition, MergeStep]] = None
        best_score: Optional[float] = None
        for candidate in ordered_candidates:
            merged_tenant_ids = tuple(sorted(set(source_partition.tenant_ids + candidate.tenant_ids)))
            if respect_cooldown and not self._merge_allowed(merged_tenant_ids, state_by_id, current_epoch):
                continue
            merged_partition = self._build_partition_from_tenant_ids(
                merged_tenant_ids,
                state_by_id,
                document_cache,
                document_block_counts,
                partition_id=self._next_partition_id(prefix='merged'),
            )
            overlap_ratio = compute_document_overlap_ratio(source_partition.document_ids, candidate.document_ids)
            merge_estimate = score_merge_candidate(
                source_vectors=source_partition.vector_count,
                source_documents=source_partition.document_count,
                source_query_rate=source_partition.query_rate,
                candidate_vectors=candidate.vector_count,
                candidate_documents=candidate.document_count,
                candidate_query_rate=candidate.query_rate,
                merged_vectors=merged_partition.vector_count,
                merged_documents=merged_partition.document_count,
                merged_query_rate=merged_partition.query_rate,
                overlap_ratio=overlap_ratio,
                merged_selectivity=self._partition_selectivity(merged_partition, state_by_id),
                merged_pollution=self._partition_pollution(merged_partition),
                parameters=self.parameters,
                topk=self.topk,
                recall_target=merged_partition.recall_target,
                weights=self.cost_weights,
            )
            if best_score is None or merge_estimate.score < best_score:
                best_score = merge_estimate.score
                best_choice = (
                    merged_partition,
                    MergeStep(
                        left_partition_id=source_partition.partition_id,
                        right_partition_id=candidate.partition_id,
                        merged_partition_id=merged_partition.partition_id,
                        merged_tenant_ids=merged_partition.tenant_ids,
                        overlap_ratio=merge_estimate.overlap_ratio,
                        delta_query_cost=merge_estimate.delta_query_cost,
                        delta_memory=merge_estimate.delta_memory,
                        merged_memory=merge_estimate.merged_memory,
                        score=merge_estimate.score,
                        reason='merge source into best candidate',
                        window_marker=current_epoch,
                    ),
                )
        return best_choice

    def _reduce_to_budget(
        self,
        partitions: Sequence[PlannedPartition],
        *,
        state_by_id: dict[int, TenantStateSnapshot],
        document_cache: dict[int, set[int]],
        document_block_counts: dict[int, int],
        memory_limit: float,
        current_epoch: int,
        reason: str,
    ) -> tuple[list[PlannedPartition], list[MergeStep]]:
        working = self._clone_partitions(partitions)
        merge_steps: list[MergeStep] = []
        while len(working) >= 2 and self._total_memory(working) > memory_limit:
            best: Optional[tuple[int, int, PlannedPartition, MergeStep]] = None
            best_score: Optional[float] = None
            for left_index, right_index in self._candidate_pairs_for_budget(working):
                left = working[left_index]
                right = working[right_index]
                merged_tenant_ids = tuple(sorted(set(left.tenant_ids + right.tenant_ids)))
                merged_partition = self._build_partition_from_tenant_ids(
                    merged_tenant_ids,
                    state_by_id,
                    document_cache,
                    document_block_counts,
                    partition_id=self._next_partition_id(prefix='merged'),
                )
                overlap_ratio = compute_document_overlap_ratio(left.document_ids, right.document_ids)
                merge_estimate = score_merge_candidate(
                    source_vectors=left.vector_count,
                    source_documents=left.document_count,
                    source_query_rate=left.query_rate,
                    candidate_vectors=right.vector_count,
                    candidate_documents=right.document_count,
                    candidate_query_rate=right.query_rate,
                    merged_vectors=merged_partition.vector_count,
                    merged_documents=merged_partition.document_count,
                    merged_query_rate=merged_partition.query_rate,
                    overlap_ratio=overlap_ratio,
                    merged_selectivity=self._partition_selectivity(merged_partition, state_by_id),
                    merged_pollution=self._partition_pollution(merged_partition),
                    parameters=self.parameters,
                    topk=self.topk,
                    recall_target=merged_partition.recall_target,
                    weights=self.cost_weights,
                )
                if best_score is None or merge_estimate.score < best_score:
                    best_score = merge_estimate.score
                    best = (
                        left_index,
                        right_index,
                        merged_partition,
                        MergeStep(
                            left_partition_id=left.partition_id,
                            right_partition_id=right.partition_id,
                            merged_partition_id=merged_partition.partition_id,
                            merged_tenant_ids=merged_partition.tenant_ids,
                            overlap_ratio=merge_estimate.overlap_ratio,
                            delta_query_cost=merge_estimate.delta_query_cost,
                            delta_memory=merge_estimate.delta_memory,
                            merged_memory=merge_estimate.merged_memory,
                            score=merge_estimate.score,
                            reason=reason,
                            window_marker=current_epoch,
                        ),
                    )
            if best is None:
                break
            left_index, right_index, merged_partition, merge_step = best
            for index in sorted((left_index, right_index), reverse=True):
                working.pop(index)
            working.append(merged_partition)
            merge_steps.append(merge_step)
        return self._sort_partitions(working), merge_steps

    def _candidate_pairs_for_budget(self, partitions: Sequence[PlannedPartition]) -> list[tuple[int, int]]:
        ordered_indices = sorted(
            range(len(partitions)),
            key=lambda index: (
                partitions[index].estimated_memory,
                partitions[index].partition_id,
            ),
        )
        if len(ordered_indices) < 2:
            return []

        primary_size = min(len(ordered_indices), self.candidate_merge_limit)
        extended_size = min(len(ordered_indices), max(primary_size + 1, self.candidate_merge_limit * 2))
        primary = ordered_indices[:primary_size]
        extended = ordered_indices[:extended_size]

        pair_keys: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for left_pos in range(len(primary)):
            for right_pos in range(left_pos + 1, len(primary)):
                pair = (primary[left_pos], primary[right_pos])
                seen.add(pair)
                pair_keys.append(pair)

        smallest = primary[0]
        for other in extended[1:]:
            pair = (min(smallest, other), max(smallest, other))
            if pair in seen:
                continue
            seen.add(pair)
            pair_keys.append(pair)
        return pair_keys

    def _find_best_split_candidate(
        self,
        partitions: Sequence[PlannedPartition],
        *,
        state_by_id: dict[int, TenantStateSnapshot],
        document_cache: dict[int, set[int]],
        document_block_counts: dict[int, int],
        current_epoch: int,
        memory_limit: float,
    ) -> Optional[tuple[SplitStep, list[PlannedPartition]]]:
        current_memory = self._total_memory(partitions)
        best_plan: Optional[tuple[SplitStep, list[PlannedPartition]]] = None
        best_score: Optional[float] = None

        for source in partitions:
            if len(source.tenant_ids) <= 1:
                continue
            source_before_cost = self._estimate_partition_cost(source, state_by_id)
            other_partitions = [partition for partition in partitions if partition.partition_id != source.partition_id]

            for tenant_id in source.tenant_ids:
                state = state_by_id[tenant_id]
                if not self._split_allowed(state, current_epoch):
                    continue
                remaining_tenant_ids = tuple(sorted(tid for tid in source.tenant_ids if tid != tenant_id))
                source_after = self._build_partition_from_tenant_ids(
                    remaining_tenant_ids,
                    state_by_id,
                    document_cache,
                    document_block_counts,
                    partition_id=source.partition_id,
                )
                dedicated_target = self._build_partition_from_tenant_ids(
                    (tenant_id,),
                    state_by_id,
                    document_cache,
                    document_block_counts,
                    partition_id=self._next_partition_id(prefix='split'),
                )
                future_memory = current_memory - source.estimated_memory + source_after.estimated_memory + dedicated_target.estimated_memory
                if future_memory <= memory_limit:
                    split_estimate = score_split_candidate(
                        source_before_cost=source_before_cost,
                        source_after_cost=self._estimate_partition_cost(source_after, state_by_id),
                        target_after_cost=self._estimate_partition_cost(dedicated_target, state_by_id),
                        current_memory=current_memory,
                        future_memory=future_memory,
                        moved_vector_count=state.size.vector_count,
                        target_selectivity=self._partition_selectivity(dedicated_target, state_by_id),
                        target_pollution=self._partition_pollution(dedicated_target),
                        weights=self.cost_weights,
                    )
                    if best_score is None or split_estimate.score > best_score:
                        updated = [partition for partition in partitions if partition.partition_id != source.partition_id]
                        if source_after.tenant_ids:
                            updated.append(source_after)
                        updated.append(dedicated_target)
                        best_score = split_estimate.score
                        best_plan = (
                            SplitStep(
                                source_partition_id=source.partition_id,
                                target_partition_id=dedicated_target.partition_id,
                                tenant_id=tenant_id,
                                source_tenant_ids_after=source_after.tenant_ids,
                                target_tenant_ids_after=dedicated_target.tenant_ids,
                                score=split_estimate.score,
                                gain=split_estimate.gain,
                                delta_query_cost=split_estimate.delta_query_cost,
                                delta_memory=split_estimate.delta_memory,
                                mode='dedicated',
                                reason='split hot tenant into dedicated partition',
                                window_marker=current_epoch,
                            ),
                            self._sort_partitions(updated),
                        )

                candidate_targets = sorted(
                    other_partitions,
                    key=lambda partition: (
                        -compute_document_overlap_ratio(document_cache.get(tenant_id, set()), partition.document_ids),
                        partition.estimated_memory,
                        partition.partition_id,
                    ),
                )[: self.candidate_merge_limit]
                for target in candidate_targets:
                    if not self._merge_allowed(target.tenant_ids + (tenant_id,), state_by_id, current_epoch):
                        continue
                    target_after = self._build_partition_from_tenant_ids(
                        tuple(sorted(set(target.tenant_ids + (tenant_id,)))),
                        state_by_id,
                        document_cache,
                        document_block_counts,
                        partition_id=target.partition_id,
                    )
                    future_memory = (
                        current_memory
                        - source.estimated_memory
                        - target.estimated_memory
                        + source_after.estimated_memory
                        + target_after.estimated_memory
                    )
                    if future_memory > memory_limit:
                        continue
                    split_estimate = score_split_candidate(
                        source_before_cost=source_before_cost + self._estimate_partition_cost(target, state_by_id),
                        source_after_cost=self._estimate_partition_cost(source_after, state_by_id),
                        target_after_cost=self._estimate_partition_cost(target_after, state_by_id),
                        current_memory=current_memory,
                        future_memory=future_memory,
                        moved_vector_count=state.size.vector_count,
                        target_selectivity=self._partition_selectivity(target_after, state_by_id),
                        target_pollution=self._partition_pollution(target_after),
                        weights=self.cost_weights,
                    )
                    if best_score is None or split_estimate.score > best_score:
                        updated = [
                            partition for partition in partitions
                            if partition.partition_id not in {source.partition_id, target.partition_id}
                        ]
                        if source_after.tenant_ids:
                            updated.append(source_after)
                        updated.append(target_after)
                        best_score = split_estimate.score
                        best_plan = (
                            SplitStep(
                                source_partition_id=source.partition_id,
                                target_partition_id=target.partition_id,
                                tenant_id=tenant_id,
                                source_tenant_ids_after=source_after.tenant_ids,
                                target_tenant_ids_after=target_after.tenant_ids,
                                score=split_estimate.score,
                                gain=split_estimate.gain,
                                delta_query_cost=split_estimate.delta_query_cost,
                                delta_memory=split_estimate.delta_memory,
                                mode='shared',
                                reason='split hot tenant into a better shared partition',
                                window_marker=current_epoch,
                            ),
                            self._sort_partitions(updated),
                        )
        return best_plan

    def _split_allowed(self, state: TenantStateSnapshot, current_epoch: int) -> bool:
        last_split = int(state.metadata.get('last_split_window', 0) or 0)
        last_merge = int(state.metadata.get('last_merge_window', 0) or 0)
        return current_epoch - max(last_split, last_merge) >= self.split_cooldown_windows

    def _merge_allowed(
        self,
        tenant_ids: Iterable[int],
        state_by_id: dict[int, TenantStateSnapshot],
        current_epoch: int,
    ) -> bool:
        for tenant_id in tenant_ids:
            state = state_by_id.get(int(tenant_id))
            if state is None:
                continue
            last_merge = int(state.metadata.get('last_merge_window', 0) or 0)
            last_split = int(state.metadata.get('last_split_window', 0) or 0)
            if current_epoch - max(last_merge, last_split) < self.merge_cooldown_windows:
                return False
        return True

    def _find_partition_for_tenant(
        self,
        partitions: Sequence[PlannedPartition],
        tenant_id: int,
    ) -> PlannedPartition:
        for partition in partitions:
            if tenant_id in partition.tenant_ids:
                return partition
        raise ValueError(f'tenant {tenant_id} not found in any partition')

    def _clone_partitions(self, partitions: Sequence[PlannedPartition]) -> list[PlannedPartition]:
        return [
            PlannedPartition(
                partition_id=partition.partition_id,
                tenant_ids=tuple(partition.tenant_ids),
                tenant_names=tuple(partition.tenant_names),
                document_count=partition.document_count,
                vector_count=partition.vector_count,
                estimated_memory=partition.estimated_memory,
                query_rate=partition.query_rate,
                write_rate=partition.write_rate,
                recall_target=partition.recall_target,
                tenant_document_counts=dict(partition.tenant_document_counts),
                tenant_vector_counts=dict(partition.tenant_vector_counts),
                document_ids=frozenset(partition.document_ids),
                metadata=dict(partition.metadata),
            )
            for partition in partitions
        ]

    def _sort_partitions(self, partitions: Sequence[PlannedPartition]) -> list[PlannedPartition]:
        return sorted(partitions, key=lambda partition: (partition.estimated_memory, partition.partition_id))

    def _total_memory(self, partitions: Sequence[PlannedPartition]) -> float:
        return sum(partition.estimated_memory for partition in partitions)

    def _next_partition_id(self, *, prefix: str) -> str:
        return f'{prefix}_{next(self._partition_counter)}'
