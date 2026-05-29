from __future__ import annotations

from collections import defaultdict
import heapq
from typing import Iterable

from .common import (
    HistoricalPredicateRecord,
    SIEVE_ROOT_TABLE,
    SieveCandidate,
    SievePartition,
    SievePlan,
    normalize_int_tuple,
)
from .cost_model import SieveCostModel
from .predicates import mask_is_subset, roles_to_mask


def _candidate_key(role_ids: tuple[int, ...]) -> tuple[int, ...]:
    return normalize_int_tuple(role_ids)


class SieveOptimizer:
    def __init__(
        self,
        *,
        cost_model: SieveCostModel,
        role_positions: dict[int, int],
    ) -> None:
        self.cost_model = cost_model
        self.role_positions = role_positions

    def build_candidates(
        self,
        historical_records: Iterable[HistoricalPredicateRecord],
        *,
        document_block_counts: dict[tuple[int, ...], int],
    ) -> list[SieveCandidate]:
        candidates: list[SieveCandidate] = []
        for candidate_id, record in enumerate(historical_records, start=1):
            role_ids = normalize_int_tuple(record.role_ids)
            if not role_ids:
                continue
            cardinality = int(record.cardinality)
            scaled_m = self.cost_model.downscaled_m(cardinality)
            scaled_size = self.cost_model.scaled_partition_size(cardinality)
            candidates.append(
                SieveCandidate(
                    candidate_id=int(candidate_id),
                    role_ids=role_ids,
                    role_mask=roles_to_mask(role_ids, self.role_positions),
                    query_count=int(record.query_count),
                    cardinality=cardinality,
                    scaled_m=scaled_m,
                    scaled_size=scaled_size,
                    metadata={
                        "query_count": int(record.query_count),
                        "cardinality": cardinality,
                    },
                )
            )

        candidates.sort(key=lambda candidate: (candidate.cardinality, candidate.role_mask, candidate.candidate_id))
        return candidates

    def build_subset_graph(self, candidates: list[SieveCandidate]) -> tuple[dict[int, set[int]], dict[int, set[int]], list[tuple[int, int]]]:
        parent_set: dict[int, set[int]] = defaultdict(set)
        child_set: dict[int, set[int]] = defaultdict(set)
        edges: list[tuple[int, int]] = []

        for i, left in enumerate(candidates):
            for j in range(i, len(candidates)):
                right = candidates[j]
                if mask_is_subset(left.role_mask, right.role_mask):
                    parent_set[i].add(j)
                    child_set[j].add(i)
                    edges.append((i, j))
        return parent_set, child_set, edges

    def select_partitions(self, candidates: list[SieveCandidate], *, index_budget: float) -> list[int]:
        if not candidates:
            return []

        parent_set, child_set, _ = self.build_subset_graph(candidates)
        total_budget = self.cost_model.budget_units(float(index_budget))
        if total_budget <= 0:
            total_budget = sum(self.cost_model.scaled_partition_size(candidate.cardinality) for candidate in candidates)

        best_costs = [
            min(
                self.cost_model.bf_search_cost(candidate.cardinality),
                self.cost_model.root_search_cost(candidate.cardinality),
            )
            for candidate in candidates
        ]
        dirty = [False] * len(candidates)
        selected: set[int] = set()
        total_vecs = 0

        def score_node(node: int) -> float:
            ratio_sum = 0.0
            for child in child_set.get(node, set()):
                benefit = best_costs[child] - self.cost_model.upward_search_cost(
                    candidates[child].cardinality,
                    candidates[node].cardinality,
                )
                ratio_sum += max(0.0, float(benefit)) * float(candidates[child].query_count)
            scaled = self.cost_model.scaled_partition_size(candidates[node].cardinality)
            return ratio_sum / float(scaled) if scaled > 0 else 0.0

        queue = [(-score_node(i), i) for i in range(len(candidates))]
        heapq.heapify(queue)

        while total_vecs < total_budget and len(selected) < len(candidates) and queue:
            _, node = heapq.heappop(queue)
            if node in selected:
                continue
            if dirty[node]:
                heapq.heappush(queue, (-score_node(node), node))
                dirty[node] = False
                continue

            for child in child_set.get(node, set()):
                if child not in selected:
                    dirty[child] = True
                    best_costs[child] = min(
                        best_costs[child],
                        self.cost_model.upward_search_cost(
                            candidates[child].cardinality,
                            candidates[node].cardinality,
                        ),
                    )
            for parent in parent_set.get(node, set()):
                if parent not in selected:
                    dirty[parent] = True

            selected.add(node)
            total_vecs += self.cost_model.scaled_partition_size(candidates[node].cardinality)

        return sorted(selected)

    def build_plan(self, candidates: list[SieveCandidate], *, index_budget: float) -> SievePlan:
        selected_ids = self.select_partitions(candidates, index_budget=float(index_budget))
        _, _, dag_edges_by_index = self.build_subset_graph(candidates)
        partitions: list[SievePartition] = []
        selected_index_to_partition_id: dict[int, str] = {}
        for partition_index, candidate_index in enumerate(selected_ids, start=1):
            candidate = candidates[candidate_index]
            partition_id = f"sieve_{partition_index}"
            selected_index_to_partition_id[candidate_index] = partition_id
            partitions.append(
                SievePartition(
                    partition_id=partition_id,
                    candidate_id=candidate.candidate_id,
                    partition_kind="selected",
                    table_name="",
                    role_ids=candidate.role_ids,
                    role_mask=candidate.role_mask,
                    cardinality=candidate.cardinality,
                    vector_count=candidate.cardinality,
                    m=candidate.scaled_m,
                    ef_construction=self.cost_model.ef_search,
                    metadata=dict(candidate.metadata),
                )
            )

        dag_edges = [
            (int(candidates[child].candidate_id), int(candidates[parent].candidate_id))
            for child, parent in dag_edges_by_index
        ]
        hasse_edges = self._build_selected_hasse_edges(candidates, selected_ids, selected_index_to_partition_id)
        return SievePlan(
            partitions=partitions,
            candidates=candidates,
            dag_edges=dag_edges,
            hasse_edges=hasse_edges,
            metadata={
                "selected_candidate_indices": selected_ids,
                "dataset_size": self.cost_model.dataset_size,
            },
        )

    def _build_selected_hasse_edges(
        self,
        candidates: list[SieveCandidate],
        selected_ids: list[int],
        selected_index_to_partition_id: dict[int, str],
    ) -> list[tuple[str, str]]:
        if not selected_ids:
            return []

        selected = sorted(selected_ids, key=lambda idx: (candidates[idx].cardinality, candidates[idx].role_mask, idx))
        parent_set: dict[int, set[int]] = defaultdict(set)
        for left_pos, left_idx in enumerate(selected):
            for right_idx in selected[left_pos + 1:]:
                if mask_is_subset(candidates[left_idx].role_mask, candidates[right_idx].role_mask):
                    parent_set[left_idx].add(right_idx)

        hasse_edges: list[tuple[str, str]] = []
        for child_idx in selected:
            direct_parents: list[int] = []
            for parent_idx in sorted(parent_set.get(child_idx, set()), key=lambda idx: (candidates[idx].cardinality, idx)):
                direct = True
                for middle_idx in selected:
                    if middle_idx in (child_idx, parent_idx):
                        continue
                    if (
                        mask_is_subset(candidates[child_idx].role_mask, candidates[middle_idx].role_mask)
                        and mask_is_subset(candidates[middle_idx].role_mask, candidates[parent_idx].role_mask)
                    ):
                        direct = False
                        break
                if direct:
                    direct_parents.append(parent_idx)

            child_partition_id = selected_index_to_partition_id[child_idx]
            if direct_parents:
                for parent_idx in direct_parents:
                    hasse_edges.append((child_partition_id, selected_index_to_partition_id[parent_idx]))
            else:
                hasse_edges.append((child_partition_id, SIEVE_ROOT_TABLE))

        return hasse_edges

