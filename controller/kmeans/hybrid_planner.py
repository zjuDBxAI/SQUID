from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import json
import math
from pathlib import Path

from typing import Optional

from tqdm import tqdm

from .cost_model import DEFAULT_COST_MODEL, DEFAULT_COST_TOPK, cost_model_metadata, estimate_partition_query_cost

from .common import ACLPattern, KMeansPartition, KMeansPlan, TenantRoute, get_partition_table_name


_COST_TOPK = int(DEFAULT_COST_TOPK)
_PRIVATE_MERGE_NEIGHBOR_LIMIT = 128
_PRIVATE_FALLBACK_POOL_LIMIT = 128
_SHARED_MERGE_NEIGHBOR_LIMIT = 128
_SHARED_TENANT_PATTERN_CAP = 512
_SHARED_FALLBACK_POOL_LIMIT = 128


_PRIVATE_PLANNER_TRACE_HOOK = None


def _fixed_ef_for_cost(ef_search: Optional[int]) -> Optional[int]:
    return None if ef_search is None else int(ef_search)


def _shared_pattern_count_at_threshold(patterns: list[ACLPattern], threshold: float) -> int:
    return int(sum(1 for pattern in patterns if float(pattern.score) >= float(threshold)))


def _binary_search_shared_threshold(patterns: list[ACLPattern], target_ratio: float) -> float:
    if not patterns:
        return 0.0

    normalized_ratio = min(max(float(target_ratio), 0.0), 1.0)
    if normalized_ratio <= 0.0:
        return math.inf

    unique_scores = sorted({float(pattern.score) for pattern in patterns})
    total_patterns = max(1, int(len(patterns)))
    if normalized_ratio >= 1.0:
        return float(unique_scores[0])

    left = 0
    right = len(unique_scores) - 1
    best_index = 0
    found = False
    while left <= right:
        mid = (left + right) // 2
        threshold = float(unique_scores[mid])
        shared_ratio = float(_shared_pattern_count_at_threshold(patterns, threshold) / float(total_patterns))
        if shared_ratio >= normalized_ratio:
            best_index = mid
            found = True
            left = mid + 1
        else:
            right = mid - 1
    if not found:
        return float(unique_scores[0])
    return float(unique_scores[best_index])


def _load_workload_frequencies(query_dataset_path: Optional[str]) -> dict[int, float]:
    if not query_dataset_path:
        return {}
    path = Path(query_dataset_path)
    if not path.exists():
        return {}
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    frequencies: dict[int, float] = Counter()
    for row in rows if isinstance(rows, list) else []:
        if not isinstance(row, dict) or "user_id" not in row:
            continue
        frequencies[int(row["user_id"])] += 1.0
    return {int(tenant_id): float(value) for tenant_id, value in frequencies.items()}


def _weighted_jaccard(left: set[int], right: set[int], weights: dict[int, int]) -> float:
    if not left and not right:
        return 1.0
    if not left or not right:
        return 0.0
    intersection_weight = float(sum(int(weights.get(item, 1)) for item in left & right))
    union_weight = float(sum(int(weights.get(item, 1)) for item in left | right))
    return float(intersection_weight / max(union_weight, 1.0))


class HybridACLKMeansPlanner:
    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = int(random_state)
        self._search_index_model = "hnsw"
        self._last_private_metadata: dict[str, object] = {}
        self._last_shared_metadata: dict[str, object] = {}
        self._last_split_metadata: dict[str, object] = {}
        self._last_private_groups: list[dict[str, object]] = []

    def build_plan(
        self,
        acl_rows: list[tuple[int, tuple[int, ...], tuple[int, ...], int]],
        *,
        private_cluster_count: int,
        shared_cluster_count: int,
        shared_score_ratio: float,
        shared_route_limit: int = 3,
        private_replication_budget_ratio: float = 0.0,
        embedding_dim: Optional[int] = None,
        query_dataset_path: Optional[str] = None,
        ef_search: Optional[int] = None,
        show_progress: bool = True,
        enable_split: bool = True,
        private_edge_top_d: int = 32,
        index_type: str = "hnsw",
    ) -> KMeansPlan:
        if not acl_rows:
            raise ValueError("Cannot build kmeans ACL partitions without ACL rows")
        self._search_index_model = str(index_type or "hnsw").strip().lower()

        with tqdm(total=7, desc="Cost split planner", unit="stage", disable=not show_progress) as progress:
            patterns = self._build_patterns(acl_rows, show_progress=show_progress)
            progress.update(1)
            progress.set_description("Cost split planner: built ACL patterns")

            tenant_ids = tuple(sorted({tenant_id for pattern in patterns for tenant_id in pattern.tenant_ids}))
            if not tenant_ids:
                raise ValueError("No tenants found in ACL rows")
            tenant_vector_counts = self._tenant_vector_counts(patterns)
            total_original_vectors = int(sum(int(pattern.vector_count) for pattern in patterns))
            tenant_query_weights = {int(tenant_id): 1.0 for tenant_id in tenant_ids}
            split_tenant_query_weights = dict(tenant_query_weights)
            progress.update(1)
            progress.set_description("Cost split planner: splitting shared/private ACLs")

            shared_patterns, private_patterns, shared_threshold = self._split_patterns(
                patterns,
                shared_score_ratio=float(shared_score_ratio),
                private_cluster_count=int(private_cluster_count),
                tenant_ids=tenant_ids,
                tenant_vector_counts=tenant_vector_counts,
                tenant_query_weights=split_tenant_query_weights,
                total_original_vectors=total_original_vectors,
                ef_search=_fixed_ef_for_cost(ef_search),
                private_replication_budget_ratio=float(private_replication_budget_ratio),
                enable_split=bool(enable_split),
            )
            progress.update(1)
            progress.set_description("Cost split planner: split shared/private ACLs")
            shared_assignments = self._assign_shared_groups_by_cost_split(
                shared_patterns,
                cluster_count=int(shared_cluster_count),
                route_limit=int(shared_route_limit),
                tenant_query_weights=tenant_query_weights,
                total_original_vectors=total_original_vectors,
                ef_search=_fixed_ef_for_cost(ef_search),
                show_progress=show_progress,
            )
            progress.update(1)
            progress.set_description("Cost split planner: built shared splits")

            shared_vector_count = int(sum(int(pattern.vector_count) for pattern in shared_patterns))
            tenant_to_private_cluster = self._cluster_private_tenants_by_cost_split(
                private_patterns,
                tenant_ids=tenant_ids,
                cluster_count=int(private_cluster_count),
                replication_budget_ratio=float(private_replication_budget_ratio),
                tenant_query_weights=tenant_query_weights,
                total_original_vectors=total_original_vectors,
                shared_vector_count=int(shared_vector_count),
                ef_search=_fixed_ef_for_cost(ef_search),
                show_progress=show_progress,
                private_edge_top_d=int(private_edge_top_d),
            )
            progress.update(1)
            progress.set_description("Cost split planner: built private splits")

            partitions = self._build_partitions(
                shared_patterns=shared_patterns,
                private_patterns=private_patterns,
                shared_assignments=shared_assignments,
                tenant_to_private_cluster=tenant_to_private_cluster,
                private_cluster_count=int(private_cluster_count),
                shared_cluster_count=int(shared_cluster_count),
                show_progress=show_progress,
            )
            progress.update(1)
            progress.set_description("Cost split planner: built partitions")

            routes = self._build_routes(partitions, tenant_to_private_cluster=tenant_to_private_cluster)
            progress.update(1)
            progress.set_description("Cost split planner: built tenant routes")

        total_original_documents = int(sum(int(pattern.document_count) for pattern in patterns))
        total_partition_vectors = int(sum(int(partition.vector_count) for partition in partitions))
        route_counts = Counter(int(route.tenant_id) for route in routes)
        shared_route_counts = Counter(int(route.tenant_id) for route in routes if str(route.route_kind) == "shared")
        private_route_counts = Counter(int(route.tenant_id) for route in routes if str(route.route_kind) == "private")
        route_count_values = [int(value) for value in route_counts.values()]
        effective_private_cluster_count = len({int(cluster_id) for cluster_id in tenant_to_private_cluster.values()})
        effective_shared_cluster_count = len({int(cluster_id) for cluster_id in shared_assignments.values()})
        metadata = {
            "algorithm": "private_core_star_split_merge_v16",
            "private_cluster_count": int(private_cluster_count),
            "effective_private_cluster_count": int(effective_private_cluster_count),
            "shared_cluster_count": int(shared_cluster_count),
            "effective_shared_cluster_count": int(effective_shared_cluster_count),
            "cluster_count": int(effective_private_cluster_count + effective_shared_cluster_count),
            "tenant_count": int(len(tenant_ids)),
            "pattern_count": int(len(patterns)),
            "shared_pattern_count": int(len(shared_patterns)),
            "private_pattern_count": int(len(private_patterns)),
            "document_count": int(total_original_documents),
            "original_vector_count": int(total_original_vectors),
            "partition_vector_count": int(total_partition_vectors),
            "memory_replication_factor": float(total_partition_vectors / max(total_original_vectors, 1)),
            "shared_score_ratio": float(shared_score_ratio),
            "shared_score_threshold": None if shared_threshold is None else float(shared_threshold),
            "shared_route_limit": int(shared_route_limit),
            "private_replication_budget_ratio": float(private_replication_budget_ratio),
            "enable_split": bool(enable_split),
            "private_edge_top_d": int(private_edge_top_d),
            "ef_search_for_cost": None if ef_search is None else int(ef_search),
            "ef_search_for_cost_mode": "trained_adaptive" if ef_search is None else "fixed_override",
            "search_index_model": str(self._search_index_model),
            "topk_for_cost": int(_COST_TOPK),
            **cost_model_metadata(DEFAULT_COST_MODEL),
            "shared_cost_model": "bottom-up ACL group merge using exact pre/post adaptive-ef latency cost",
            "private_cost_model": "v16 core-star private merge: edge first chooses the memory-saving operation with minimum delta_latency; heap first takes delta_latency<=0 memory-saving candidates, then ranks positive-loss candidates by delta_latency/memory_saved",
            "shared_score_rule": "adaptive ACL marginal cost split with reference private groups and storage multiplier",
            "query_weight_rule": "all tenants use q_t=1; query_dataset workload frequencies are ignored",
            "split_query_weight_rule": "all tenants use q_t=1 in shared/private split",
            "private_query_weight_rule": "all tenants use q_t=1 in private merge",
            "shared_query_weight_rule": "q_t fixed to 1 in shared cost model",
            "shared_vector_ratio_actual": float(
                sum(int(pattern.vector_count) for pattern in shared_patterns) / max(total_original_vectors, 1)
            ),
            "shared_pattern_ratio_actual": float(len(shared_patterns) / max(len(patterns), 1)),
            "shared_vector_count": int(sum(int(pattern.vector_count) for pattern in shared_patterns)),
            "shared_storage_if_private": int(sum(int(pattern.vector_count) * len(pattern.tenant_ids) for pattern in shared_patterns)),
            "shared_storage_saved_by_split": int(
                sum(int(pattern.vector_count) * max(0, len(pattern.tenant_ids) - 1) for pattern in shared_patterns)
            ),
            "private_objective": "v16 core-star split-merge: start from one private group per tenant and compress ACL copies until total storage satisfies private_replication_budget_ratio",
            "shared_objective": "bottom-up shared ACL global merge while total pre/post query-cost gain is positive",
            "embedding_dim_status": "ignored_by_cost_guided_two_zone_split_v12",
            "tenant_vector_count_min": int(min(tenant_vector_counts.values())) if tenant_vector_counts else 0,
            "tenant_vector_count_max": int(max(tenant_vector_counts.values())) if tenant_vector_counts else 0,
            "private_cluster_metadata": dict(self._last_private_metadata),
            "shared_cluster_metadata": dict(self._last_shared_metadata),
            "split_metadata": dict(self._last_split_metadata),
            "route_count_min": int(min(route_count_values)) if route_count_values else 0,
            "route_count_mean": float(sum(route_count_values) / len(route_count_values)) if route_count_values else 0.0,
            "route_count_max": int(max(route_count_values)) if route_count_values else 0,
            "private_route_count_max": int(max(private_route_counts.values())) if private_route_counts else 0,
            "shared_route_count_max": int(max(shared_route_counts.values())) if shared_route_counts else 0,
            "partition_sizes": {str(partition.partition_id): int(partition.vector_count) for partition in partitions},
        }
        return KMeansPlan(
            partitions=partitions,
            tenant_routes=routes,
            tenant_to_cluster=tenant_to_private_cluster,
            patterns=patterns,
            metadata=metadata,
        )

    def _build_patterns(
        self,
        acl_rows: list[tuple[int, tuple[int, ...], tuple[int, ...], int]],
        *,
        show_progress: bool,
    ) -> list[ACLPattern]:
        tenant_count = max(1, len({tenant_id for _, tenants, _, _ in acl_rows for tenant_id in tenants}))
        patterns: list[ACLPattern] = []
        iterator = tqdm(acl_rows, desc="ACL score", unit="acl", leave=False, disable=not show_progress)
        for pattern_id, tenants, document_ids, vector_count in iterator:
            acl_tenant_count = max(1, len(tenants))
            score = float(math.log1p(max(0, int(vector_count)) * max(0, int(acl_tenant_count) - 1)))
            weight = float(math.log1p(max(0, int(vector_count))) * math.log1p(float(tenant_count) / float(acl_tenant_count)))
            patterns.append(
                ACLPattern(
                    pattern_id=int(pattern_id),
                    tenant_ids=tuple(int(tenant_id) for tenant_id in tenants),
                    document_ids=tuple(int(document_id) for document_id in document_ids),
                    vector_count=int(vector_count),
                    document_count=int(len(document_ids)),
                    weight=weight,
                    score=score,
                    zone="private",
                )
            )
        return patterns

    def _split_patterns(
        self,
        patterns: list[ACLPattern],
        *,
        shared_score_ratio: float,
        private_cluster_count: int,
        tenant_ids: tuple[int, ...],
        tenant_vector_counts: dict[int, int],
        tenant_query_weights: dict[int, float],
        total_original_vectors: int,
        ef_search: Optional[int],
        private_replication_budget_ratio: float,
        enable_split: bool = True,
    ) -> tuple[list[ACLPattern], list[ACLPattern], Optional[float]]:
        if not patterns:
            self._last_split_metadata = {
                "enabled": False,
                "reason": "no_patterns",
                "shared_score_ratio_input": float(shared_score_ratio),
            }
            return [], [], None

        if not bool(enable_split):
            private = []
            for pattern in patterns:
                pattern.zone = "private"
                private.append(pattern)
            total_original = int(sum(int(pattern.vector_count) for pattern in patterns))
            base_private_storage = int(sum(int(pattern.vector_count) * len(pattern.tenant_ids) for pattern in patterns))
            allowed_storage = int(
                math.floor(
                    float(max(1, int(total_original_vectors)))
                    * (1.0 + max(0.0, float(private_replication_budget_ratio)))
                )
            )
            self._last_split_metadata = {
                "enabled": False,
                "reason": "disabled_by_enable_split_flag",
                "rule": "split_disabled_all_acl_private",
                "shared_score_ratio_input": float(shared_score_ratio),
                "shared_score_ratio_used": False,
                "shared_count": 0,
                "private_count": int(len(private)),
                "actual_shared_ratio": 0.0,
                "selected_acl_count": 0,
                "private_replication_budget_ratio": float(private_replication_budget_ratio),
                "estimated_allowed_storage": int(allowed_storage),
                "estimated_base_private_storage": int(base_private_storage),
                "estimated_selected_storage": int(base_private_storage),
                "estimated_all_shared_storage": int(total_original),
                "estimated_storage_budget_satisfied": bool(int(base_private_storage) <= int(allowed_storage)),
                "private_cluster_count_hint": int(private_cluster_count),
            }
            return [], private, None

        tenant_ids = tuple(sorted(int(tenant_id) for tenant_id in tenant_ids))
        pattern_weights = {int(pattern.pattern_id): max(0, int(pattern.vector_count)) for pattern in patterns}

        route_startup_samples = sorted(
            self._partition_query_cost(
                partition_vectors=max(1, int(pattern.vector_count)),
                accessible_vectors=max(1, int(pattern.vector_count)),
                tenant_weight=1.0,
                total_vectors=int(total_original_vectors),
                ef_search=_fixed_ef_for_cost(ef_search),
            )
            for pattern in patterns
            if int(pattern.vector_count) > 0
        )
        route_startup_cost = (
            float(route_startup_samples[len(route_startup_samples) // 2])
            if route_startup_samples
            else float(_COST_TOPK)
        )

        def split_query_cost(partition_vectors: int, accessible_vectors: int) -> float:
            if int(partition_vectors) <= 0 or int(accessible_vectors) <= 0:
                return 0.0
            return self._partition_query_cost(
                partition_vectors=int(partition_vectors),
                accessible_vectors=int(accessible_vectors),
                tenant_weight=1.0,
                total_vectors=int(total_original_vectors),
                ef_search=_fixed_ef_for_cost(ef_search),
            )

        # Same split shape as SQUID: build reference private groups only for deciding
        # whether an ACL should be admitted to shared. These are not final partitions.
        tenant_owner_rank: dict[int, tuple[float, int, int, int]] = {}
        tenant_owner_pattern: dict[int, int] = {}
        for pattern in patterns:
            pattern_id = int(pattern.pattern_id)
            pattern_vector_count = max(0, int(pattern.vector_count))
            pattern_tenant_count = max(1, len(pattern.tenant_ids))
            owner_score = float(pattern_vector_count) / float(pattern_tenant_count)
            rank = (float(owner_score), int(pattern_vector_count), -int(pattern_tenant_count), -int(pattern_id))
            for tenant_id in pattern.tenant_ids:
                tenant_id = int(tenant_id)
                if tenant_id not in tenant_owner_rank or rank > tenant_owner_rank[tenant_id]:
                    tenant_owner_rank[tenant_id] = rank
                    tenant_owner_pattern[tenant_id] = pattern_id

        owner_ids = sorted({int(pattern_id) for pattern_id in tenant_owner_pattern.values()})
        owner_to_group = {int(owner_id): index for index, owner_id in enumerate(owner_ids)}
        tenant_group: dict[int, int] = {
            int(tenant_id): int(owner_to_group[int(owner_pattern)])
            for tenant_id, owner_pattern in tenant_owner_pattern.items()
        }
        missing_tenants = [int(tenant_id) for tenant_id in tenant_ids if int(tenant_id) not in tenant_group]
        next_group_id = len(owner_to_group)
        for tenant_id in missing_tenants:
            tenant_group[int(tenant_id)] = int(next_group_id)
            next_group_id += 1

        group_tenants: dict[int, set[int]] = defaultdict(set)
        for tenant_id, group_id in tenant_group.items():
            group_tenants[int(group_id)].add(int(tenant_id))

        group_vector_counts: Counter = Counter()
        group_tenant_access: dict[int, Counter] = defaultdict(Counter)
        pattern_groups: dict[int, set[int]] = {}
        for pattern in patterns:
            pattern_id = int(pattern.pattern_id)
            pattern_vector_count = max(0, int(pattern.vector_count))
            touched_groups = {
                int(tenant_group[int(tenant_id)])
                for tenant_id in pattern.tenant_ids
                if int(tenant_id) in tenant_group
            }
            pattern_groups[pattern_id] = set(touched_groups)
            for group_id in touched_groups:
                group_vector_counts[int(group_id)] += int(pattern_vector_count)
            for tenant_id in pattern.tenant_ids:
                tenant_id = int(tenant_id)
                if tenant_id not in tenant_group:
                    continue
                group_tenant_access[int(tenant_group[tenant_id])][tenant_id] += int(pattern_vector_count)

        def group_query_cost(vector_count: int, tenant_access: Counter) -> float:
            if int(vector_count) <= 0:
                return 0.0
            total = 0.0
            for _tenant_id, accessible_vectors in tenant_access.items():
                total += split_query_cost(int(vector_count), int(accessible_vectors))
            return float(total)

        group_costs = {
            int(group_id): float(group_query_cost(int(vector_count), group_tenant_access[int(group_id)]))
            for group_id, vector_count in group_vector_counts.items()
        }
        base_private_storage = int(sum(int(value) for value in group_vector_counts.values()))
        allowed_storage = int(
            math.floor(
                float(max(1, int(total_original_vectors)))
                * (1.0 + max(0.0, float(private_replication_budget_ratio)))
            )
        )

        split_rows: list[dict[str, object]] = []
        for pattern in patterns:
            pattern_id = int(pattern.pattern_id)
            pattern_vector_count = max(0, int(pattern.vector_count))
            pattern_tenant_set = {int(tenant_id) for tenant_id in pattern.tenant_ids}
            touched_groups = set(pattern_groups.get(pattern_id, set()))
            private_saving = 0.0
            for group_id in touched_groups:
                before_vectors = int(group_vector_counts.get(int(group_id), 0))
                after_vectors = max(0, int(before_vectors) - int(pattern_vector_count))
                before_cost = float(group_costs.get(int(group_id), 0.0))
                before_access = group_tenant_access[int(group_id)]
                after_access = Counter()
                for tenant_id, accessible_vectors in before_access.items():
                    next_access = int(accessible_vectors)
                    if int(tenant_id) in pattern_tenant_set:
                        next_access -= int(pattern_vector_count)
                    if next_access > 0:
                        after_access[int(tenant_id)] = int(next_access)
                after_cost = float(group_query_cost(int(after_vectors), after_access))
                private_saving += float(before_cost - after_cost)

            shared_cost = 0.0
            for _tenant_id in pattern_tenant_set:
                shared_cost += split_query_cost(
                    max(1, int(pattern_vector_count)),
                    max(1, int(pattern_vector_count)),
                )
            touched_group_count = max(1, len(touched_groups))
            private_storage = int(pattern_vector_count) * int(touched_group_count)
            shared_storage = int(pattern_vector_count)
            storage_delta = int(shared_storage) - int(private_storage)
            base_score = float(shared_cost) - float(private_saving)
            split_rows.append(
                {
                    "pattern": pattern,
                    "base_score": float(base_score),
                    "private_saving": float(private_saving),
                    "shared_cost": float(shared_cost),
                    "storage_delta": int(storage_delta),
                    "touched_group_count": int(touched_group_count),
                    "tenant_count": int(len(pattern_tenant_set)),
                }
            )

        shared_admission_unit_cost = float(route_startup_cost)

        def selected_rows(lambda_value: float) -> list[dict[str, object]]:
            candidates: list[tuple[float, int, dict[str, object]]] = []
            for row in split_rows:
                adjusted_score = float(row["base_score"]) + float(lambda_value) * float(row["storage_delta"])
                gain = float(-adjusted_score)
                if gain > 0.0:
                    pattern = row["pattern"]
                    candidates.append((float(-gain), int(pattern.pattern_id), row))
            candidates.sort(key=lambda item: (item[0], item[1]))

            selected: list[dict[str, object]] = []
            selected_tenant_memberships = 0
            for negative_gain, _pattern_id, row in candidates:
                gain = float(-negative_gain)
                tenant_count = max(1, int(row.get("tenant_count", 1)))
                admission_work = int(len(selected)) * int(tenant_count) + int(selected_tenant_memberships)
                admission_cost = float(shared_admission_unit_cost) * float(admission_work)
                if gain <= admission_cost:
                    break
                selected.append(row)
                selected_tenant_memberships += int(tenant_count)
            return selected

        def estimate_storage(lambda_value: float) -> int:
            storage = int(base_private_storage)
            for row in selected_rows(float(lambda_value)):
                storage += int(row["storage_delta"])
            return int(storage)

        touched_group_counts_for_price = [max(1, int(row["touched_group_count"])) for row in split_rows]
        average_touched_group_count = (
            float(sum(touched_group_counts_for_price) / len(touched_group_counts_for_price))
            if touched_group_counts_for_price
            else 1.0
        )
        base_private_query_cost = float(sum(float(value) for value in group_costs.values()))
        average_query_cost_per_replicated_vector = float(
            base_private_query_cost / float(max(1, int(base_private_storage)))
        )
        budget_pressure = max(
            0.0,
            float(base_private_storage) / float(max(1, int(allowed_storage))) - 1.0,
        )
        lambda_value = float(
            average_query_cost_per_replicated_vector
            * float(average_touched_group_count)
            * float(budget_pressure)
        )
        memory_budget_active = int(base_private_storage) > int(allowed_storage)

        selected_row_ids = {id(row) for row in selected_rows(float(lambda_value))}
        shared: list[ACLPattern] = []
        private: list[ACLPattern] = []
        selected_scores: list[float] = []
        selected_gains: list[float] = []
        base_scores = [float(row["base_score"]) for row in split_rows]
        storage_deltas = [int(row["storage_delta"]) for row in split_rows]
        touched_group_counts = [int(row["touched_group_count"]) for row in split_rows]
        beneficial_count = 0
        shared_by_query_gain = 0
        shared_by_storage_pressure = 0
        for row in split_rows:
            pattern = row["pattern"]
            adjusted_score = float(row["base_score"]) + float(lambda_value) * float(row["storage_delta"])
            gain = float(-adjusted_score)
            if gain > 0.0:
                beneficial_count += 1
            if id(row) in selected_row_ids:
                pattern.zone = "shared"
                shared.append(pattern)
                selected_scores.append(float(adjusted_score))
                selected_gains.append(float(gain))
                if float(row["base_score"]) < 0.0:
                    shared_by_query_gain += 1
                else:
                    shared_by_storage_pressure += 1
            else:
                pattern.zone = "private"
                private.append(pattern)

        shared.sort(key=lambda pattern: int(pattern.pattern_id))
        private.sort(key=lambda pattern: int(pattern.pattern_id))
        selected_storage = int(estimate_storage(float(lambda_value)))
        reference_group_sizes = [len(value) for value in group_tenants.values()]
        reference_group_vectors = [int(value) for value in group_vector_counts.values()]
        self._last_split_metadata = {
            "enabled": True,
            "rule": "squid_reference_group_adaptive_acl_marginal_cost_v1",
            "shared_score_ratio_input": float(shared_score_ratio),
            "shared_score_ratio_used": False,
            "shared_count": int(len(shared)),
            "private_count": int(len(private)),
            "actual_shared_ratio": float(len(shared) / max(len(patterns), 1)),
            "score_threshold": None,
            "lambda": float(lambda_value),
            "lambda_rule": "avg_private_query_cost_per_replicated_vector * avg_acl_reference_group_span * max(base_storage/budget - 1, 0)",
            "memory_budget_active": bool(memory_budget_active),
            "budget_pressure": float(budget_pressure),
            "average_query_cost_per_replicated_vector": float(average_query_cost_per_replicated_vector),
            "average_touched_group_count_for_price": float(average_touched_group_count),
            "estimated_base_private_query_cost": float(base_private_query_cost),
            "private_replication_budget_ratio": float(private_replication_budget_ratio),
            "estimated_allowed_storage": int(allowed_storage),
            "estimated_base_private_storage": int(base_private_storage),
            "estimated_selected_storage": int(selected_storage),
            "estimated_all_shared_storage": int(sum(int(pattern.vector_count) for pattern in patterns)),
            "private_cluster_count_hint": int(private_cluster_count),
            "shared_beneficial_candidate_count": int(beneficial_count),
            "selected_acl_count": int(len(shared)),
            "shared_admission_unit_cost": float(shared_admission_unit_cost),
            "shared_admission_rule": "sort by positive net gain and admit ACL a only if gain(a) > R0 * sum_{b in S}(|T_a| + |T_b|)",
            "estimated_storage_budget_satisfied": bool(int(selected_storage) <= int(allowed_storage)),
            "private_reference_group_count": int(len(group_tenants)),
            "private_reference_owner_rule": "owner(t)=argmax_a n_a/|T_a|, tie by larger n_a, smaller |T_a|, smaller pattern_id",
            "route_startup_cost": float(route_startup_cost),
            "query_weight_rule": "q_t fixed to 1.0 per tenant for split estimation; workload frequencies are ignored; VEDA cost parameters are unchanged",
            "shared_by_query_gain": int(shared_by_query_gain),
            "shared_by_storage_pressure": int(shared_by_storage_pressure),
            "base_score_min": float(min(base_scores)) if base_scores else None,
            "base_score_max": float(max(base_scores)) if base_scores else None,
            "selected_adjusted_score_min": float(min(selected_scores)) if selected_scores else None,
            "selected_adjusted_score_max": float(max(selected_scores)) if selected_scores else None,
            "selected_gain_min": float(min(selected_gains)) if selected_gains else None,
            "selected_gain_max": float(max(selected_gains)) if selected_gains else None,
            "storage_delta_min": int(min(storage_deltas)) if storage_deltas else 0,
            "storage_delta_max": int(max(storage_deltas)) if storage_deltas else 0,
            "touched_group_count_min": int(min(touched_group_counts)) if touched_group_counts else 0,
            "touched_group_count_mean": float(sum(touched_group_counts) / len(touched_group_counts)) if touched_group_counts else 0.0,
            "touched_group_count_max": int(max(touched_group_counts)) if touched_group_counts else 0,
            "reference_group_tenant_min": int(min(reference_group_sizes)) if reference_group_sizes else 0,
            "reference_group_tenant_mean": float(sum(reference_group_sizes) / len(reference_group_sizes)) if reference_group_sizes else 0.0,
            "reference_group_tenant_max": int(max(reference_group_sizes)) if reference_group_sizes else 0,
            "reference_group_vector_min": int(min(reference_group_vectors)) if reference_group_vectors else 0,
            "reference_group_vector_mean": float(sum(reference_group_vectors) / len(reference_group_vectors)) if reference_group_vectors else 0.0,
            "reference_group_vector_max": int(max(reference_group_vectors)) if reference_group_vectors else 0,
            "cost_model": str(cost_model_metadata(DEFAULT_COST_MODEL)["cost_model"]),
            "private_cost_aggregation": "average_over_served_tenants",
        }
        return shared, private, None

    def _tenant_vector_counts(self, patterns: list[ACLPattern]) -> dict[int, int]:
        counts: dict[int, int] = Counter()
        for pattern in patterns:
            for tenant_id in pattern.tenant_ids:
                counts[int(tenant_id)] += int(pattern.vector_count)
        return counts

    def _tenant_query_weights(
        self,
        tenant_ids: tuple[int, ...],
        workload_frequencies: dict[int, float],
    ) -> dict[int, float]:
        return {int(tenant_id): 1.0 for tenant_id in tenant_ids}

    def _partition_query_cost(
        self,
        *,
        partition_vectors: int,
        accessible_vectors: int,
        tenant_weight: float,
        total_vectors: int,
        ef_search: Optional[int],
    ) -> float:
        if int(partition_vectors) <= 0 or int(accessible_vectors) <= 0 or float(tenant_weight) <= 0.0:
            return 0.0
        return estimate_partition_query_cost(
            partition_vectors=max(1, int(partition_vectors)),
            accessible_vectors=max(1, int(accessible_vectors)),
            tenant_weight=float(tenant_weight),
            ef_search=_fixed_ef_for_cost(ef_search),
            topk=int(_COST_TOPK),
            use_adaptive_ef=True,
            index_type=str(self._search_index_model),
        )

    def _assign_shared_groups_by_cost_split(
        self,
        patterns: list[ACLPattern],
        *,
        cluster_count: int,
        route_limit: int,
        tenant_query_weights: dict[int, float],
        total_original_vectors: int,
        ef_search: Optional[int],
        show_progress: bool,
    ) -> dict[int, int]:
        if not patterns:
            self._last_shared_metadata = {"enabled": False, "reason": "no_shared_patterns"}
            return {}

        target_cluster_count = max(1, min(int(cluster_count), len(patterns)))
        pattern_by_id = {int(pattern.pattern_id): pattern for pattern in patterns}
        pattern_weights = {int(pattern.pattern_id): int(pattern.vector_count) for pattern in patterns}
        route_startup_samples = sorted(
            self._partition_query_cost(
                partition_vectors=max(1, int(pattern.vector_count)),
                accessible_vectors=max(1, int(pattern.vector_count)),
                tenant_weight=1.0,
                total_vectors=int(total_original_vectors),
                ef_search=_fixed_ef_for_cost(ef_search),
            )
            for pattern in patterns
            if int(pattern.vector_count) > 0
        )
        route_startup_cost = (
            float(route_startup_samples[len(route_startup_samples) // 2])
            if route_startup_samples
            else float(_COST_TOPK)
        )

        def tenant_weight(tenant_id: int) -> float:
            return 1.0

        def tenant_query_cost(vector_count: int, accessible_vectors: int, tenant_id: int) -> float:
            return self._partition_query_cost(
                partition_vectors=int(vector_count),
                accessible_vectors=int(accessible_vectors),
                tenant_weight=float(tenant_weight(int(tenant_id))),
                total_vectors=int(total_original_vectors),
                ef_search=_fixed_ef_for_cost(ef_search),
            )

        def group_query_cost_from_access(vector_count: int, tenant_access: Counter) -> float:
            vector_count = int(vector_count)
            if vector_count <= 0:
                return 0.0
            total = 0.0
            for tenant_id, accessible_vectors in tenant_access.items():
                accessible_vectors = int(accessible_vectors)
                if accessible_vectors <= 0:
                    continue
                total += float(tenant_query_cost(int(vector_count), int(accessible_vectors), int(tenant_id)))
            return float(total)

        def group_cost_denominator(tenant_access: Counter) -> float:
            return float(max(1, sum(1 for value in tenant_access.values() if int(value) > 0)))

        def group_cost_from_access(vector_count: int, tenant_access: Counter) -> float:
            query_cost = float(group_query_cost_from_access(int(vector_count), tenant_access))
            denominator = float(group_cost_denominator(tenant_access))
            if denominator <= 0.0:
                return 0.0
            return float(query_cost / denominator)

        def make_group(group_id: int, pattern_ids: set[int], tenant_access: Optional[Counter] = None) -> dict[str, object]:
            normalized_patterns = set(int(pattern_id) for pattern_id in pattern_ids)
            vector_count = int(sum(int(pattern_weights.get(int(pattern_id), 0)) for pattern_id in normalized_patterns))
            if tenant_access is None:
                tenant_access = Counter()
                for pattern_id in normalized_patterns:
                    pattern = pattern_by_id[int(pattern_id)]
                    pattern_vectors = int(pattern_weights.get(int(pattern_id), 0))
                    for tenant_id in pattern.tenant_ids:
                        tenant_access[int(tenant_id)] += int(pattern_vectors)
            else:
                tenant_access = Counter({int(tenant_id): int(value) for tenant_id, value in tenant_access.items() if int(value) > 0})
            query_cost = float(group_query_cost_from_access(int(vector_count), tenant_access))
            cost_denominator = float(group_cost_denominator(tenant_access))
            return {
                "group_id": int(group_id),
                "pattern_ids": normalized_patterns,
                "vector_count": int(vector_count),
                "tenant_access": tenant_access,
                "query_cost": float(query_cost),
                "cost_denominator": float(cost_denominator),
                "cost": float(query_cost),
                "version": 0,
            }

        def overlap_weight(left: dict[str, object], right: dict[str, object]) -> float:
            left_access: Counter = left["tenant_access"]  # type: ignore[assignment]
            right_access: Counter = right["tenant_access"]  # type: ignore[assignment]
            if len(left_access) > len(right_access):
                left_access, right_access = right_access, left_access
            return float(sum(float(tenant_weight(int(tenant_id))) for tenant_id in left_access if tenant_id in right_access))

        def merged_group_access(left: dict[str, object], right: dict[str, object]) -> Counter:
            left_access: Counter = left["tenant_access"]  # type: ignore[assignment]
            right_access: Counter = right["tenant_access"]  # type: ignore[assignment]
            tenant_access = Counter(left_access)
            tenant_access.update(right_access)
            return Counter({int(tenant_id): int(value) for tenant_id, value in tenant_access.items() if int(value) > 0})

        def merged_group_query_cost(left: dict[str, object], right: dict[str, object], merged_vectors: int) -> float:
            merged_vectors = int(merged_vectors)
            if merged_vectors <= 0:
                return 0.0
            return float(group_query_cost_from_access(int(merged_vectors), merged_group_access(left, right)))

        def affected_average_cost(query_cost: float, tenant_access: Counter) -> float:
            denominator = float(group_cost_denominator(tenant_access))
            if denominator <= 0.0:
                return 0.0
            return float(query_cost / denominator)

        def merge_groups(new_group_id: int, left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
            left_patterns = left["pattern_ids"]  # type: ignore[assignment]
            right_patterns = right["pattern_ids"]  # type: ignore[assignment]
            left_access: Counter = left["tenant_access"]  # type: ignore[assignment]
            right_access: Counter = right["tenant_access"]  # type: ignore[assignment]
            tenant_access = Counter(left_access)
            tenant_access.update(right_access)
            return make_group(int(new_group_id), set(left_patterns) | set(right_patterns), tenant_access)

        groups: dict[int, dict[str, object]] = {
            int(pattern.pattern_id): make_group(int(pattern.pattern_id), {int(pattern.pattern_id)})
            for pattern in patterns
        }
        next_group_id = max(groups) + 1 if groups else 1
        initial_cost = float(sum(float(group["cost"]) for group in groups.values()))
        current_cost = float(initial_cost)

        candidate_heap: list[tuple[float, float, float, int, int, int, int, int]] = []
        candidate_evaluations = 0
        stale_candidates = 0
        fallback_merge_count = 0
        positive_overlap_merge_count = 0
        zero_overlap_merge_count = 0
        rejected_by_cost_model = 0
        total_cost_increase = 0.0
        total_cost_gain = 0.0
        total_overlap_weight = 0.0
        initial_candidate_edges = 0
        capped_tenant_count = 0
        skipped_pattern_memberships_by_cap = 0

        def candidate_entry(left_id: int, right_id: int) -> tuple[float, float, float, int, int, int, int, int] | None:
            nonlocal candidate_evaluations
            left_id = int(left_id)
            right_id = int(right_id)
            if left_id == right_id or left_id not in groups or right_id not in groups:
                return None
            if left_id > right_id:
                left_id, right_id = right_id, left_id
            left = groups[left_id]
            right = groups[right_id]
            merged_vectors = int(left["vector_count"]) + int(right["vector_count"])
            merged_access = merged_group_access(left, right)
            before_query_cost = float(left["query_cost"]) + float(right["query_cost"])
            after_query_cost = float(merged_group_query_cost(left, right, int(merged_vectors)))
            pre_cost = float(before_query_cost)
            merged_cost_value = float(after_query_cost)
            cost_increase = float(merged_cost_value - pre_cost)
            gain = float(pre_cost - merged_cost_value)
            if not math.isfinite(float(gain)):
                return None
            overlap = float(overlap_weight(left, right))
            candidate_evaluations += 1
            return (
                float(-gain),
                float(cost_increase),
                float(-overlap),
                int(merged_vectors),
                int(left_id),
                int(right_id),
                int(left["version"]),
                int(right["version"]),
            )

        def push_candidate(left_id: int, right_id: int) -> bool:
            entry = candidate_entry(int(left_id), int(right_id))
            if entry is None:
                return False
            heapq.heappush(candidate_heap, entry)
            return True

        tenant_to_group_ids: dict[int, list[int]] = defaultdict(list)
        for pattern in patterns:
            pattern_id = int(pattern.pattern_id)
            for tenant_id in pattern.tenant_ids:
                tenant_to_group_ids[int(tenant_id)].append(int(pattern_id))

        initial_neighbor_weights: dict[int, Counter] = defaultdict(Counter)
        for _tenant_id, group_ids in tenant_to_group_ids.items():
            ordered_group_ids = sorted(
                {int(group_id) for group_id in group_ids},
                key=lambda group_id: (-int(pattern_weights.get(int(group_id), 0)), int(group_id)),
            )
            if len(ordered_group_ids) > int(_SHARED_TENANT_PATTERN_CAP):
                capped_tenant_count += 1
                skipped_pattern_memberships_by_cap += len(ordered_group_ids) - int(_SHARED_TENANT_PATTERN_CAP)
                ordered_group_ids = ordered_group_ids[: int(_SHARED_TENANT_PATTERN_CAP)]
            for index, left_id in enumerate(ordered_group_ids):
                for right_id in ordered_group_ids[index + 1:]:
                    initial_neighbor_weights[int(left_id)][int(right_id)] += 1
                    initial_neighbor_weights[int(right_id)][int(left_id)] += 1

        initial_candidate_pairs: set[tuple[int, int]] = set()
        for left_id, neighbor_weights in initial_neighbor_weights.items():
            for right_id, _weight in neighbor_weights.most_common(int(_SHARED_MERGE_NEIGHBOR_LIMIT)):
                edge = (int(left_id), int(right_id)) if int(left_id) < int(right_id) else (int(right_id), int(left_id))
                initial_candidate_pairs.add(edge)
        for left_id, right_id in sorted(initial_candidate_pairs):
            if push_candidate(int(left_id), int(right_id)):
                initial_candidate_edges += 1

        def pop_valid_candidate() -> tuple[int, int, float, float, float] | None:
            nonlocal stale_candidates
            while candidate_heap:
                negative_gain, cost_increase, negative_overlap, _merged_vectors, left_id, right_id, left_version, right_version = heapq.heappop(candidate_heap)
                if int(left_id) not in groups or int(right_id) not in groups:
                    stale_candidates += 1
                    continue
                if int(groups[int(left_id)]["version"]) != int(left_version) or int(groups[int(right_id)]["version"]) != int(right_version):
                    stale_candidates += 1
                    continue
                gain = float(-negative_gain)
                if not math.isfinite(float(gain)):
                    stale_candidates += 1
                    continue
                overlap = float(-negative_overlap)
                return int(left_id), int(right_id), float(gain), float(cost_increase), float(overlap)
            return None

        progress = tqdm(
            total=max(0, len(groups) - 1),
            desc="Shared ACL global merge",
            unit="merge",
            leave=False,
            disable=not show_progress,
        )
        merge_count = 0
        stop_reason = "single_group" if len(groups) <= 1 else "not_started"
        last_gain = None
        last_cost_increase = None
        while len(groups) > 1:
            candidate = pop_valid_candidate()
            if candidate is None:
                stop_reason = "no_candidate"
                break

            left_id, right_id, gain, cost_increase, overlap = candidate
            last_gain = float(gain)
            last_cost_increase = float(cost_increase)
            if float(gain) <= 0.0:
                rejected_by_cost_model += 1
                stop_reason = "global_cost_model_stop"
                break
            if int(left_id) not in groups or int(right_id) not in groups:
                continue
            left = groups.pop(int(left_id))
            right = groups.pop(int(right_id))
            new_group = merge_groups(int(next_group_id), left, right)
            new_group_id = int(next_group_id)
            next_group_id += 1
            groups[new_group_id] = new_group

            actual_cost_increase = float(cost_increase)
            actual_gain = float(gain)
            current_cost = float(sum(float(group["cost"]) for group in groups.values()))
            total_cost_increase += float(actual_cost_increase)
            total_cost_gain += float(actual_gain)
            total_overlap_weight += float(overlap)
            merge_count += 1
            if overlap > 0.0:
                positive_overlap_merge_count += 1
            else:
                zero_overlap_merge_count += 1

            refreshed_neighbors: list[tuple[float, int, int]] = []
            for other_id, other in groups.items():
                if int(other_id) == int(new_group_id):
                    continue
                other_overlap = float(overlap_weight(new_group, other))
                if other_overlap <= 0.0:
                    continue
                refreshed_neighbors.append((-float(other_overlap), int(other["vector_count"]), int(other_id)))
            for _negative_overlap, _other_vectors, other_id in sorted(refreshed_neighbors)[: int(_SHARED_MERGE_NEIGHBOR_LIMIT)]:
                push_candidate(int(new_group_id), int(other_id))

            if progress.n < progress.total:
                progress.update(1)
            stop_reason = "single_group" if len(groups) <= 1 else "global_cost_model_candidate_pending"
            if show_progress:
                progress.set_postfix(
                    {
                        "groups": int(len(groups)),
                        "gain": f"{float(gain):.4f}",
                        "delta": f"{float(cost_increase):.4f}",
                        "overlap": f"{float(overlap):.4f}",
                    }
                )
        progress.close()

        compact_ids = {group_id: index for index, group_id in enumerate(sorted(groups))}
        assignments: dict[int, int] = {}
        for group_id, group in groups.items():
            compact_id = int(compact_ids[int(group_id)])
            for pattern_id in group["pattern_ids"]:
                assignments[int(pattern_id)] = compact_id

        tenant_route_counts: Counter = Counter()
        group_vector_counts = [int(group["vector_count"]) for group in groups.values()]
        group_tenant_counts = [int(len(group["tenant_access"])) for group in groups.values()]
        group_pattern_counts = [int(len(group["pattern_ids"])) for group in groups.values()]
        for group in groups.values():
            tenant_access: Counter = group["tenant_access"]  # type: ignore[assignment]
            for tenant_id in tenant_access:
                tenant_route_counts[int(tenant_id)] += 1

        initial_route_count = int(sum(len(pattern.tenant_ids) for pattern in patterns))
        final_route_count = int(sum(int(len(group["tenant_access"])) for group in groups.values()))
        self._last_shared_metadata = {
            "enabled": True,
            "objective": "bottom-up shared ACL merge by neighbor-limited exact pre/post tenant query-cost gain",
            "target_cluster_count": int(target_cluster_count),
            "target_cluster_count_enforced": False,
            "route_limit": int(route_limit),
            "route_limit_enforced": False,
            "initial_group_count": int(len(patterns)),
            "final_group_count": int(len(groups)),
            "merge_count": int(merge_count),
            "positive_route_merge_count": int(positive_overlap_merge_count),
            "zero_route_merge_count": int(zero_overlap_merge_count),
            "positive_overlap_merge_count": int(positive_overlap_merge_count),
            "zero_overlap_merge_count": int(zero_overlap_merge_count),
            "fallback_merge_count": int(fallback_merge_count),
            "rejected_by_cost_model": int(rejected_by_cost_model),
            "stop_reason": str(stop_reason),
            "last_gain": None if last_gain is None else float(last_gain),
            "last_cost_increase": None if last_cost_increase is None else float(last_cost_increase),
            "last_unit_cost": None,
            "last_adaptive_threshold": None,
            "candidate_evaluations": int(candidate_evaluations),
            "active_heap_entries": int(len(candidate_heap)),
            "edge_cache_size": 0,
            "edge_cache_hits": 0,
            "edge_cache_misses": 0,
            "initial_candidate_edges": int(initial_candidate_edges),
            "stale_candidates": int(stale_candidates),
            "tenant_pattern_cap": int(_SHARED_TENANT_PATTERN_CAP),
            "capped_tenant_count": int(capped_tenant_count),
            "skipped_pattern_memberships_by_cap": int(skipped_pattern_memberships_by_cap),
            "neighbor_limit": int(_SHARED_MERGE_NEIGHBOR_LIMIT),
            "fallback_pool_limit": None,
            "global_candidate_graph": False,
            "cost_initial": float(initial_cost),
            "cost_final": float(current_cost),
            "total_cost_increase": float(total_cost_increase),
            "total_cost_gain": float(total_cost_gain),
            "total_route_benefit": float(total_overlap_weight),
            "total_overlap_weight": float(total_overlap_weight),
            "initial_route_count": int(initial_route_count),
            "final_route_count": int(final_route_count),
            "route_count_min": int(min(tenant_route_counts.values())) if tenant_route_counts else 0,
            "route_count_mean": float(sum(tenant_route_counts.values()) / len(tenant_route_counts)) if tenant_route_counts else 0.0,
            "route_count_max": int(max(tenant_route_counts.values())) if tenant_route_counts else 0,
            "min_group_vectors": int(min(group_vector_counts)) if group_vector_counts else 0,
            "max_group_vectors": int(max(group_vector_counts)) if group_vector_counts else 0,
            "min_group_tenants": int(min(group_tenant_counts)) if group_tenant_counts else 0,
            "max_group_tenants": int(max(group_tenant_counts)) if group_tenant_counts else 0,
            "min_group_patterns": int(min(group_pattern_counts)) if group_pattern_counts else 0,
            "max_group_patterns": int(max(group_pattern_counts)) if group_pattern_counts else 0,
            "merge_score_rule": "choose best available neighbor candidate by total Gain=C_before-C_after; merge only when Gain>0",
            "route_startup_cost": float(route_startup_cost),
            "candidate_rule": "tenant co-access neighbor heap capped by neighbor_limit; after each merge refresh only positive-overlap neighbors of the new group",
        }
        return assignments

    def _cluster_private_tenants_by_cost_split(
        self,
        patterns: list[ACLPattern],
        *,
        tenant_ids: tuple[int, ...],
        cluster_count: int,
        replication_budget_ratio: float,
        tenant_query_weights: dict[int, float],
        total_original_vectors: int,
        shared_vector_count: int,
        ef_search: Optional[int],
        show_progress: bool,
        private_edge_top_d: int = 32,
    ) -> dict[int, int]:
        return self._cluster_private_tenants_by_core_star_v16(
            patterns,
            tenant_ids=tenant_ids,
            cluster_count=int(cluster_count),
            replication_budget_ratio=float(replication_budget_ratio),
            tenant_query_weights=tenant_query_weights,
            total_original_vectors=int(total_original_vectors),
            shared_vector_count=int(shared_vector_count),
            ef_search=_fixed_ef_for_cost(ef_search),
            show_progress=bool(show_progress),
            private_edge_top_d=int(private_edge_top_d),
        )

    def _cluster_private_tenants_by_core_star_v16(
        self,
        patterns: list[ACLPattern],
        *,
        tenant_ids: tuple[int, ...],
        cluster_count: int,
        replication_budget_ratio: float,
        tenant_query_weights: dict[int, float],
        total_original_vectors: int,
        shared_vector_count: int,
        ef_search: Optional[int],
        show_progress: bool,
        private_edge_top_d: int = 32,
    ) -> dict[int, int]:
        tenant_ids = tuple(sorted(int(tenant_id) for tenant_id in tenant_ids))
        tenant_set = set(int(tenant_id) for tenant_id in tenant_ids)
        tenant_count = int(len(tenant_ids))
        top_d = max(1, int(private_edge_top_d))
        self._last_private_groups = []
        if tenant_count == 0:
            self._last_private_metadata = {"enabled": False, "reason": "empty_tenant_input", "planner": "private_core_star_split_merge_v16"}
            return {}

        target_cluster_count = max(1, min(int(cluster_count), tenant_count))
        pattern_weights = {int(pattern.pattern_id): max(0, int(pattern.vector_count)) for pattern in patterns}
        tenant_bit_index = {int(tenant_id): index for index, tenant_id in enumerate(tenant_ids)}
        pattern_tenant_masks: dict[int, int] = {}
        for pattern in patterns:
            tenant_mask = 0
            for tenant_id in pattern.tenant_ids:
                tenant_index = tenant_bit_index.get(int(tenant_id))
                if tenant_index is not None:
                    tenant_mask |= int(1 << int(tenant_index))
            pattern_tenant_masks[int(pattern.pattern_id)] = int(tenant_mask)
        private_unique_vectors = int(sum(int(value) for value in pattern_weights.values()))
        if private_unique_vectors <= 0:
            self._last_private_metadata = {
                "enabled": False,
                "reason": "no_private_patterns",
                "planner": "private_core_star_split_merge_v16",
                "target_cluster_count": int(target_cluster_count),
                "private_unique_vectors": 0,
                "private_edge_top_d": int(top_d),
            }
            self._last_private_groups = []
            return {int(tenant_id): 0 for tenant_id in tenant_ids}

        allowed_total_storage = int(
            math.floor(float(max(1, total_original_vectors)) * (1.0 + max(0.0, float(replication_budget_ratio))))
        )
        allowed_private_storage = max(
            int(private_unique_vectors),
            int(allowed_total_storage) - int(shared_vector_count),
        )
        base_all_private_storage_for_patterns = int(
            sum(int(pattern_weights.get(int(pattern.pattern_id), 0)) * len(pattern.tenant_ids) for pattern in patterns)
        )
        shared_adjusted_private_budget = int(allowed_private_storage)

        ordered_pattern_ids = tuple(sorted(int(pattern_id) for pattern_id in pattern_weights))
        pattern_bit_index = {int(pattern_id): index for index, pattern_id in enumerate(ordered_pattern_ids)}
        bit_index_pattern_ids = {index: int(pattern_id) for pattern_id, index in pattern_bit_index.items()}
        bit_index_weights = tuple(int(pattern_weights[int(pattern_id)]) for pattern_id in ordered_pattern_ids)
        single_pattern_bits = {int(pattern_id): int(1 << index) for pattern_id, index in pattern_bit_index.items()}
        bit_weight_cache: dict[int, int] = {0: 0}

        def pattern_bits_for(pattern_ids: set[int] | frozenset[int] | tuple[int, ...] | list[int]) -> int:
            bits = 0
            for pattern_id in pattern_ids:
                bits |= int(single_pattern_bits.get(int(pattern_id), 0))
            return int(bits)

        def pattern_ids_from_bits(pattern_bits: int) -> set[int]:
            bits = int(pattern_bits)
            result: set[int] = set()
            while bits:
                lowest_bit = bits & -bits
                index = int(lowest_bit.bit_length() - 1)
                result.add(int(bit_index_pattern_ids[int(index)]))
                bits ^= lowest_bit
            return result

        def vector_count_bits(pattern_bits: int) -> int:
            bits = int(pattern_bits)
            cached = bit_weight_cache.get(bits)
            if cached is not None:
                return int(cached)
            original_bits = bits
            total = 0
            while bits:
                lowest_bit = bits & -bits
                index = int(lowest_bit.bit_length() - 1)
                total += int(bit_index_weights[int(index)])
                bits ^= lowest_bit
            bit_weight_cache[int(original_bits)] = int(total)
            return int(total)

        def bit_count(value: int) -> int:
            return int(int(value).bit_count())

        def normalize_tenant_bits(tenant_bits: dict[int, int]) -> dict[int, int]:
            return {int(tenant_id): int(bits) for tenant_id, bits in tenant_bits.items() if int(bits) != 0}

        def data_bits_from_tenant_bits(tenant_bits: dict[int, int]) -> int:
            bits = 0
            for value in tenant_bits.values():
                bits |= int(value)
            return int(bits)

        def tenant_access_from_bits(tenant_bits: dict[int, int], data_bits: int | None = None) -> dict[int, int]:
            mask = None if data_bits is None else int(data_bits)
            result: dict[int, int] = {}
            for tenant_id, bits in tenant_bits.items():
                kept = int(bits) if mask is None else int(bits) & int(mask)
                if kept:
                    access = int(vector_count_bits(kept))
                    if access > 0:
                        result[int(tenant_id)] = int(access)
            return result

        def group_cost_from_access(tenant_access: dict[int, int], partition_vectors: int) -> float:
            partition_vectors = int(partition_vectors)
            if partition_vectors <= 0:
                return 0.0
            partition_vectors = max(1, int(partition_vectors))
            total = 0.0
            for tenant_id, accessible_vectors in tenant_access.items():
                accessible_vectors = int(accessible_vectors)
                tenant_weight = float(tenant_query_weights.get(int(tenant_id), 1.0))
                if accessible_vectors <= 0 or tenant_weight <= 0.0:
                    continue
                total += estimate_partition_query_cost(
                    partition_vectors=int(partition_vectors),
                    accessible_vectors=int(accessible_vectors),
                    tenant_weight=1.0,
                    ef_search=_fixed_ef_for_cost(ef_search),
                    topk=int(_COST_TOPK),
                    use_adaptive_ef=True,
                    index_type=str(self._search_index_model),
                )
            return float(total)

        def tenant_bits_to_pattern_map(tenant_bits: dict[int, int]) -> dict[int, set[int]]:
            result: dict[int, set[int]] = {}
            for tenant_id, bits in normalize_tenant_bits(tenant_bits).items():
                pattern_ids = pattern_ids_from_bits(int(bits))
                if pattern_ids:
                    result[int(tenant_id)] = pattern_ids
            return result

        def make_group_from_bits(
            group_id: int,
            tenant_bits: dict[int, int],
            *,
            stored_bits: int | None = None,
            version: int = 0,
        ) -> dict[str, object]:
            service_bits = normalize_tenant_bits(tenant_bits)
            service_data_bits = int(data_bits_from_tenant_bits(service_bits)) if service_bits else 0
            data_bits = int(service_data_bits) if stored_bits is None else int(stored_bits)
            # The physical ACLs in a private partition and the tenants routed to
            # that partition are separate concepts. Cost and route generation use
            # service_bits only; data_bits is the physical storage used by memory
            # accounting and graph overlap.
            data_bits &= int(service_data_bits)
            tenant_access = tenant_access_from_bits(service_bits, data_bits)
            vectors = int(vector_count_bits(data_bits))
            pattern_ids = pattern_ids_from_bits(data_bits)
            tenant_patterns = tenant_bits_to_pattern_map(service_bits)
            tenant_count_for_core = int(len(service_bits))
            return {
                "group_id": int(group_id),
                "pattern_ids": pattern_ids,
                "pattern_bits": int(data_bits),
                "tenant_patterns": tenant_patterns,
                "service_tenant_patterns": tenant_patterns,
                "tenant_pattern_bits": service_bits,
                "service_tenant_pattern_bits": service_bits,
                "tenant_access": tenant_access,
                "vector_count": int(vectors),
                "pattern_count": int(len(pattern_ids)),
                "tenant_count": int(tenant_count_for_core),
                "cost": float(group_cost_from_access(tenant_access, int(vectors))),
                "version": int(version),
            }

        def is_live_group(group: dict[str, object]) -> bool:
            service_bits = group.get("service_tenant_pattern_bits", group.get("tenant_pattern_bits"))
            return bool(int(group.get("vector_count", 0)) > 0 and service_bits)

        def group_tenant_bits(group: dict[str, object]) -> dict[int, int]:
            return dict(group.get("service_tenant_pattern_bits", group.get("tenant_pattern_bits", {})) or {})  # type: ignore[arg-type]

        def group_pattern_bits(group: dict[str, object]) -> int:
            return int(group.get("pattern_bits", 0))

        def union_tenant_bits(*maps: dict[int, int]) -> dict[int, int]:
            result: dict[int, int] = {}
            for mapping in maps:
                for tenant_id, bits in mapping.items():
                    bits = int(bits)
                    if bits:
                        result[int(tenant_id)] = int(result.get(int(tenant_id), 0)) | bits
            return normalize_tenant_bits(result)

        def remove_bits_from_tenants(tenant_bits: dict[int, int], removed_bits: int) -> dict[int, int]:
            removed_bits = int(removed_bits)
            result: dict[int, int] = {}
            for tenant_id, bits in tenant_bits.items():
                kept = int(bits) & ~removed_bits
                if kept:
                    result[int(tenant_id)] = int(kept)
            return result

        def mask_tenant_bits(tenant_bits: dict[int, int], mask_bits: int) -> dict[int, int]:
            mask_bits = int(mask_bits)
            if mask_bits == 0:
                return {}
            result: dict[int, int] = {}
            for tenant_id, bits in tenant_bits.items():
                kept = int(bits) & mask_bits
                if kept:
                    result[int(tenant_id)] = int(kept)
            return result

        def normalize_spec(spec: tuple[dict[int, int], int | None]) -> tuple[dict[int, int], int]:
            tenant_bits, stored_bits = spec
            normalized = normalize_tenant_bits(tenant_bits)
            data_bits = int(data_bits_from_tenant_bits(normalized)) if stored_bits is None else int(stored_bits)
            data_bits &= int(data_bits_from_tenant_bits(normalized)) if normalized else 0
            return normalized, int(data_bits)

        def specs_cost_memory(specs: list[tuple[dict[int, int], int | None]]) -> tuple[float, int, int, int]:
            after_cost = 0.0
            after_memory = 0
            max_partition_size = 0
            live_count = 0
            for spec in specs:
                normalized, data_bits = normalize_spec(spec)
                if not normalized or data_bits == 0:
                    continue
                vectors = int(vector_count_bits(data_bits))
                if vectors <= 0:
                    continue
                tenant_access = tenant_access_from_bits(normalized, data_bits)
                if not tenant_access:
                    continue
                after_memory += int(vectors)
                after_cost += float(group_cost_from_access(tenant_access, int(vectors)))
                max_partition_size = max(int(max_partition_size), int(vectors))
                live_count += 1
            return float(after_cost), int(after_memory), int(max_partition_size), int(live_count)

        def operation_specs(left: dict[str, object], right: dict[str, object], operation: str) -> list[tuple[dict[int, int], int | None]]:
            left_bits = group_tenant_bits(left)
            right_bits = group_tenant_bits(right)
            left_stored_bits = int(group_pattern_bits(left))
            right_stored_bits = int(group_pattern_bits(right))
            raw_overlap_bits = int(left_stored_bits & right_stored_bits)
            overlap_bits = int(raw_overlap_bits)
            if overlap_bits == 0:
                return []
            if operation == "full":
                merged_bits = union_tenant_bits(left_bits, right_bits)
                return [(merged_bits, int(left_stored_bits | right_stored_bits))]
            if operation == "move_left":
                moved = mask_tenant_bits(left_bits, overlap_bits)
                new_left = remove_bits_from_tenants(left_bits, overlap_bits)
                new_right = union_tenant_bits(right_bits, moved)
                return [(new_left, int(left_stored_bits & ~overlap_bits)), (new_right, int(right_stored_bits))]
            if operation == "move_right":
                moved = mask_tenant_bits(right_bits, overlap_bits)
                new_left = union_tenant_bits(left_bits, moved)
                new_right = remove_bits_from_tenants(right_bits, overlap_bits)
                return [(new_left, int(left_stored_bits)), (new_right, int(right_stored_bits & ~overlap_bits))]
            if operation == "split_overlap":
                left_remain = remove_bits_from_tenants(left_bits, overlap_bits)
                overlap = union_tenant_bits(mask_tenant_bits(left_bits, overlap_bits), mask_tenant_bits(right_bits, overlap_bits))
                right_remain = remove_bits_from_tenants(right_bits, overlap_bits)
                return [
                    (left_remain, int(left_stored_bits & ~overlap_bits)),
                    (overlap, int(overlap_bits)),
                    (right_remain, int(right_stored_bits & ~overlap_bits)),
                ]
            if operation == "merge_extract_overlap":
                left_remain = remove_bits_from_tenants(left_bits, overlap_bits)
                right_remain = remove_bits_from_tenants(right_bits, overlap_bits)
                merged_remain = union_tenant_bits(left_remain, right_remain)
                overlap = union_tenant_bits(mask_tenant_bits(left_bits, overlap_bits), mask_tenant_bits(right_bits, overlap_bits))
                return [
                    (merged_remain, int((left_stored_bits | right_stored_bits) & ~overlap_bits)),
                    (overlap, int(overlap_bits)),
                ]
            raise ValueError(f"Unknown private core-star operation: {operation}")

        operation_rank = {
            "split_overlap": 0,
            "merge_extract_overlap": 1,
            "move_left": 2,
            "move_right": 3,
            "full": 4,
        }
        operations = ("full", "move_left", "move_right", "merge_extract_overlap", "split_overlap")

        def group_selectivity_profile(group_id: int, group: dict[str, object]) -> dict[str, object] | None:
            partition_vectors = int(group.get("vector_count", 0))
            if partition_vectors <= 0:
                return None
            raw_tenant_access = group.get("tenant_access", {}) or {}
            tenant_access = {
                int(tenant_id): int(accessible_vectors)
                for tenant_id, accessible_vectors in dict(raw_tenant_access).items()
                if int(accessible_vectors) > 0
            }
            if not tenant_access:
                return None
            total_selectivity = 0.0
            worst_tenant = None
            worst_access = 0
            worst_selectivity = 1.0
            worst_rank: tuple[float, int, int] | None = None
            is_pure = True
            for tenant_id, accessible_vectors in tenant_access.items():
                capped_access = min(int(partition_vectors), max(0, int(accessible_vectors)))
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
            if worst_tenant is None:
                return None
            avg_selectivity = float(total_selectivity) / float(max(1, len(tenant_access)))
            return {
                "group_id": int(group_id),
                "partition_vectors": int(partition_vectors),
                "tenant_count": int(len(tenant_access)),
                "avg_selectivity": float(avg_selectivity),
                "worst_selectivity": float(worst_selectivity),
                "worst_tenant": int(worst_tenant),
                "worst_access": int(worst_access),
                "is_pure": bool(is_pure),
            }

        def selectivity_extract_specs(
            group: dict[str, object],
            worst_tenant: int,
        ) -> list[tuple[dict[int, int], int | None]]:
            tenant_bits = group_tenant_bits(group)
            stored_bits = int(group_pattern_bits(group))
            extract_bits = int(tenant_bits.get(int(worst_tenant), 0)) & int(stored_bits)
            if int(extract_bits) == 0 or int(extract_bits) == int(stored_bits):
                return []
            remain_bits = remove_bits_from_tenants(tenant_bits, int(extract_bits))
            extract_tenant_bits = mask_tenant_bits(tenant_bits, int(extract_bits))
            return [
                (remain_bits, int(stored_bits & ~int(extract_bits))),
                (extract_tenant_bits, int(extract_bits)),
            ]

        tenant_pattern_bits: dict[int, int] = {int(tenant_id): 0 for tenant_id in tenant_ids}
        for pattern in sorted(patterns, key=lambda item: int(item.pattern_id)):
            pattern_id = int(pattern.pattern_id)
            if int(pattern_weights.get(pattern_id, 0)) <= 0:
                continue
            bit = int(single_pattern_bits.get(pattern_id, 0))
            if bit == 0:
                continue
            for tenant_id in pattern.tenant_ids:
                tenant_id = int(tenant_id)
                if tenant_id in tenant_set:
                    tenant_pattern_bits[int(tenant_id)] = int(tenant_pattern_bits.get(int(tenant_id), 0)) | int(bit)

        groups: dict[int, dict[str, object]] = {}
        next_group_id = 0
        for tenant_id in tenant_ids:
            bits = int(tenant_pattern_bits.get(int(tenant_id), 0))
            if bits == 0:
                continue
            group = make_group_from_bits(int(next_group_id), {int(tenant_id): int(bits)})
            if is_live_group(group):
                groups[int(next_group_id)] = group
                next_group_id += 1
        initial_group_count = int(len(groups))

        if not groups:
            self._last_private_metadata = {"enabled": False, "reason": "empty_private_groups", "planner": "private_core_star_split_merge_v16"}
            self._last_private_groups = []
            return {int(tenant_id): 0 for tenant_id in tenant_ids}

        private_current_storage = int(sum(int(group["vector_count"]) for group in groups.values()))
        total_current_storage = int(shared_vector_count) + int(private_current_storage)
        initial_private_storage = int(private_current_storage)
        initial_total_storage = int(total_current_storage)
        initial_cost = float(sum(float(group["cost"]) for group in groups.values()))
        current_cost = float(initial_cost)

        pattern_group_ids: dict[int, set[int]] = defaultdict(set)
        pattern_star_edges: dict[int, dict[tuple[int, int], float]] = defaultdict(dict)
        pattern_core_ids: dict[int, int] = {}
        edge_refcounts: Counter = Counter()
        edge_signal_scores: dict[tuple[int, int], float] = defaultdict(float)
        adjacency: dict[int, set[int]] = defaultdict(set)
        candidate_heap: list[tuple[object, ...]] = []
        edge_heap_tokens: dict[tuple[int, int], int] = {}
        edge_candidate_cache: dict[tuple[int, int, int, int], dict[str, object] | None] = {}
        edge_candidate_cache_groups: dict[int, set[tuple[int, int, int, int]]] = defaultdict(set)
        edge_cache_hits = 0
        edge_cache_misses = 0
        edge_cache_prune_count = 0
        next_heap_token = 0
        candidate_evaluations = 0
        stale_candidates = 0
        heap_push_count = 0
        heap_rebuild_count = 0
        graph_rebuild_count = 0
        operation_counts: Counter = Counter()
        initial_candidate_edges = 0
        refreshed_edge_count = 0
        incremental_group_refresh_count = 0
        incremental_pattern_refresh_count = 0
        incremental_edge_add_count = 0
        incremental_edge_remove_count = 0
        rejected_no_overlap_candidates = 0
        rejected_no_saving_candidates = 0
        total_storage_reduction = 0
        total_latency_delta = 0.0
        last_operation = None
        last_candidate_score = None
        last_candidate_delta_latency = None
        last_candidate_memory_saved = None
        selectivity_refine_count = 0
        selectivity_refine_cost_delta = 0.0
        selectivity_refine_stop_reason = "not_started"
        selectivity_refine_last_group_id = None
        selectivity_refine_last_worst_tenant = None
        selectivity_refine_last_avg_selectivity = None
        selectivity_refine_last_worst_selectivity = None

        def edge_key(left_id: int, right_id: int) -> tuple[int, int]:
            left_id = int(left_id)
            right_id = int(right_id)
            return (left_id, right_id) if left_id < right_id else (right_id, left_id)

        def pattern_core_key(pattern_id: int, group_id: int) -> tuple[int, float, int, int, int]:
            group = groups[int(group_id)]
            group_vectors = max(1, int(group.get("vector_count", 0)))
            acl_vectors = max(0, int(pattern_weights.get(int(pattern_id), 0)))
            acl_share = float(acl_vectors) / float(group_vectors)
            pattern_count = int(group.get("pattern_count", bit_count(group_pattern_bits(group))))
            tenant_count = int(group.get("tenant_count", len(group_tenant_bits(group))))
            return (
                int(pattern_count),
                -float(acl_share),
                int(group_vectors),
                int(tenant_count),
                int(group_id),
            )

        def register_group(group_id: int) -> None:
            group = groups.get(int(group_id))
            if group is None:
                return
            for pattern_id in group.get("pattern_ids", set()):  # type: ignore[union-attr]
                pattern_id = int(pattern_id)
                pattern_group_ids[pattern_id].add(int(group_id))
                cached_core = pattern_core_ids.get(pattern_id)
                if cached_core is not None and cached_core in groups:
                    if pattern_core_key(pattern_id, int(group_id)) < pattern_core_key(pattern_id, int(cached_core)):
                        pattern_core_ids[pattern_id] = int(group_id)

        def unregister_group(group_id: int) -> None:
            group = groups.get(int(group_id))
            if group is None:
                return
            for pattern_id in group.get("pattern_ids", set()):  # type: ignore[union-attr]
                pattern_id = int(pattern_id)
                owners = pattern_group_ids.get(pattern_id)
                if owners is not None:
                    owners.discard(int(group_id))
                    if not owners:
                        pattern_group_ids.pop(pattern_id, None)
                        pattern_core_ids.pop(pattern_id, None)
                if int(pattern_core_ids.get(pattern_id, -1)) == int(group_id):
                    pattern_core_ids.pop(pattern_id, None)

        def choose_core_group(pattern_id: int, owner_ids: set[int]) -> int | None:
            live_owner_ids = [int(group_id) for group_id in owner_ids if int(group_id) in groups]
            if not live_owner_ids:
                return None
            return min(live_owner_ids, key=lambda group_id: pattern_core_key(int(pattern_id), int(group_id)))

        def compute_pattern_star_edges(
            pattern_id: int,
            *,
            focus_owner_ids: set[int] | None = None,
        ) -> dict[tuple[int, int], float]:
            pattern_id = int(pattern_id)
            owners = {int(group_id) for group_id in pattern_group_ids.get(pattern_id, set()) if int(group_id) in groups}
            if len(owners) <= 1:
                pattern_core_ids.pop(pattern_id, None)
                return {}
            core_id = pattern_core_ids.get(pattern_id)
            if core_id not in owners or int(core_id) not in groups:
                core_id = choose_core_group(pattern_id, owners)
                if core_id is None:
                    pattern_core_ids.pop(pattern_id, None)
                    return {}
                pattern_core_ids[pattern_id] = int(core_id)
            elif focus_owner_ids:
                for focus_id in sorted(int(group_id) for group_id in focus_owner_ids if int(group_id) in owners):
                    if pattern_core_key(pattern_id, int(focus_id)) < pattern_core_key(pattern_id, int(core_id)):
                        core_id = int(focus_id)
                pattern_core_ids[pattern_id] = int(core_id)
            signal = float(pattern_weights.get(pattern_id, 0))
            if signal <= 0.0:
                return {}
            if focus_owner_ids and int(core_id) not in focus_owner_ids:
                target_owner_ids = {
                    int(group_id)
                    for group_id in focus_owner_ids
                    if int(group_id) in owners and int(group_id) != int(core_id)
                }
            else:
                target_owner_ids = {int(group_id) for group_id in owners if int(group_id) != int(core_id)}
            return {
                edge_key(int(core_id), int(owner_id)): float(signal)
                for owner_id in sorted(target_owner_ids)
            }

        def store_edge_candidate_cache(cache_key: tuple[int, int, int, int], value: dict[str, object] | None) -> None:
            edge_candidate_cache[cache_key] = None if value is None else dict(value)
            edge_candidate_cache_groups[int(cache_key[0])].add(cache_key)
            edge_candidate_cache_groups[int(cache_key[2])].add(cache_key)

        def prune_edge_candidate_cache(group_ids: set[int]) -> int:
            nonlocal edge_cache_prune_count
            normalized_group_ids = {int(group_id) for group_id in group_ids}
            if not normalized_group_ids:
                return 0
            removed = 0
            for group_id in sorted(normalized_group_ids):
                cache_keys = edge_candidate_cache_groups.pop(int(group_id), set())
                for cache_key in list(cache_keys):
                    if cache_key in edge_candidate_cache:
                        edge_candidate_cache.pop(cache_key, None)
                        removed += 1
                    other_group_id = int(cache_key[2]) if int(cache_key[0]) == int(group_id) else int(cache_key[0])
                    other_keys = edge_candidate_cache_groups.get(int(other_group_id))
                    if other_keys is not None:
                        other_keys.discard(cache_key)
                        if not other_keys:
                            edge_candidate_cache_groups.pop(int(other_group_id), None)
            edge_cache_prune_count += int(removed)
            return int(removed)

        def add_edge_reference(edge: tuple[int, int], signal: float, *, push: bool = True) -> bool:
            edge = edge_key(int(edge[0]), int(edge[1]))
            if edge[0] == edge[1] or edge[0] not in groups or edge[1] not in groups:
                return False
            was_inactive = int(edge_refcounts.get(edge, 0)) <= 0
            edge_refcounts[edge] += 1
            edge_signal_scores[edge] = float(edge_signal_scores.get(edge, 0.0)) + float(signal)
            if was_inactive:
                adjacency[edge[0]].add(edge[1])
                adjacency[edge[1]].add(edge[0])
                if push:
                    push_candidate(edge[0], edge[1])
                return True
            if push and edge not in edge_heap_tokens:
                push_candidate(edge[0], edge[1])
            return False

        def remove_edge_reference(edge: tuple[int, int], signal: float) -> bool:
            edge = edge_key(int(edge[0]), int(edge[1]))
            current = int(edge_refcounts.get(edge, 0))
            if current <= 0:
                edge_refcounts.pop(edge, None)
                edge_signal_scores.pop(edge, None)
                return False
            if current <= 1:
                edge_refcounts.pop(edge, None)
                edge_signal_scores.pop(edge, None)
                adjacency[edge[0]].discard(edge[1])
                adjacency[edge[1]].discard(edge[0])
                edge_heap_tokens.pop(edge, None)
                return True
            edge_refcounts[edge] = current - 1
            edge_signal_scores[edge] = max(0.0, float(edge_signal_scores.get(edge, 0.0)) - float(signal))
            return False

        def refresh_star_edges_for_patterns(
            pattern_ids: set[int],
            *,
            push: bool = True,
            focus_group_ids: set[int] | None = None,
        ) -> tuple[int, int, int]:
            nonlocal incremental_pattern_refresh_count, incremental_edge_add_count, incremental_edge_remove_count
            normalized_pattern_ids = {int(pattern_id) for pattern_id in pattern_ids}
            if not normalized_pattern_ids:
                return 0, 0, 0
            focus_ids = {int(group_id) for group_id in (focus_group_ids or set()) if int(group_id) in groups}

            all_new_edges_by_pattern: dict[int, dict[tuple[int, int], float]] = {}
            edge_scores: dict[tuple[int, int], float] = defaultdict(float)
            edge_acl_counts: Counter = Counter()
            target_group_ids: set[int] = set()
            full_refresh_patterns: set[int] = set()
            for pattern_id in sorted(normalized_pattern_ids):
                pattern_id = int(pattern_id)
                old_core_id = pattern_core_ids.get(pattern_id)
                focus_for_pattern = focus_ids if focus_ids else None
                new_edges = compute_pattern_star_edges(
                    pattern_id,
                    focus_owner_ids=focus_for_pattern,
                )
                new_core_id = pattern_core_ids.get(pattern_id)
                if focus_ids and old_core_id != new_core_id:
                    new_edges = compute_pattern_star_edges(pattern_id, focus_owner_ids=None)
                    full_refresh_patterns.add(pattern_id)
                all_new_edges_by_pattern[pattern_id] = new_edges
                for edge, signal in new_edges.items():
                    if edge[0] not in groups or edge[1] not in groups:
                        continue
                    edge_scores[edge] += float(signal)
                    edge_acl_counts[edge] += 1
                    target_group_ids.add(int(edge[0]))
                    target_group_ids.add(int(edge[1]))
                for edge in pattern_star_edges.get(int(pattern_id), {}):
                    target_group_ids.add(int(edge[0]))
                    target_group_ids.add(int(edge[1]))

            if focus_ids:
                target_group_ids = set(focus_ids)

            incident_edges_by_group: dict[int, list[tuple[float, tuple[int, int]]]] = defaultdict(list)
            for edge, _score in edge_scores.items():
                if edge[0] not in groups or edge[1] not in groups:
                    continue
                left_acl_count = max(1, len(groups[int(edge[0])].get("pattern_ids", set())))
                right_acl_count = max(1, len(groups[int(edge[1])].get("pattern_ids", set())))
                shared_acl_count = max(1, int(edge_acl_counts.get(edge, 1)))
                edge_rank_score = float(shared_acl_count) / math.sqrt(float(left_acl_count) * float(right_acl_count))
                incident_edges_by_group[int(edge[0])].append((float(edge_rank_score), edge))
                incident_edges_by_group[int(edge[1])].append((float(edge_rank_score), edge))

            selected_edges: set[tuple[int, int]] = set()
            for group_id in sorted(group_id for group_id in target_group_ids if int(group_id) in groups):
                incident_edges = incident_edges_by_group.get(int(group_id), [])
                incident_edges.sort(key=lambda item: (-float(item[0]), int(item[1][0]), int(item[1][1])))
                for _score, edge in incident_edges[: int(top_d)]:
                    selected_edges.add(edge)

            refreshed_patterns = 0
            added_edges = 0
            removed_edges = 0
            for pattern_id in sorted(normalized_pattern_ids):
                old_edges = dict(pattern_star_edges.get(int(pattern_id), {}))
                new_edges = {
                    edge: float(signal)
                    for edge, signal in all_new_edges_by_pattern.get(int(pattern_id), {}).items()
                    if edge in selected_edges
                }
                if focus_ids and int(pattern_id) not in full_refresh_patterns:
                    preserved_edges = {
                        edge: float(signal)
                        for edge, signal in old_edges.items()
                        if edge[0] in groups
                        and edge[1] in groups
                        and edge[0] not in focus_ids
                        and edge[1] not in focus_ids
                    }
                    preserved_edges.update(new_edges)
                    new_edges = preserved_edges
                if old_edges == new_edges:
                    continue
                refreshed_patterns += 1
                for edge, signal in old_edges.items():
                    if edge not in new_edges:
                        if remove_edge_reference(edge, float(signal)):
                            removed_edges += 1
                for edge, signal in new_edges.items():
                    if edge not in old_edges:
                        if add_edge_reference(edge, float(signal), push=push):
                            added_edges += 1
                    else:
                        old_signal = float(old_edges.get(edge, 0.0))
                        if abs(float(signal) - old_signal) > 1e-12:
                            edge_signal_scores[edge] = max(
                                0.0,
                                float(edge_signal_scores.get(edge, 0.0)) - old_signal + float(signal),
                            )
                if new_edges:
                    pattern_star_edges[int(pattern_id)] = dict(new_edges)
                else:
                    pattern_star_edges.pop(int(pattern_id), None)

            incremental_pattern_refresh_count += int(refreshed_patterns)
            incremental_edge_add_count += int(added_edges)
            incremental_edge_remove_count += int(removed_edges)
            return int(refreshed_patterns), int(added_edges), int(removed_edges)

        def candidate_for_edge(left_id: int, right_id: int, *, include_specs: bool = False) -> dict[str, object] | None:
            nonlocal candidate_evaluations, edge_cache_hits, edge_cache_misses, rejected_no_overlap_candidates, rejected_no_saving_candidates
            left_id = int(left_id)
            right_id = int(right_id)
            if left_id == right_id or left_id not in groups or right_id not in groups:
                return None
            if left_id > right_id:
                left_id, right_id = right_id, left_id
            left = groups[left_id]
            right = groups[right_id]
            cache_key = (int(left_id), int(left["version"]), int(right_id), int(right["version"]))
            cached = edge_candidate_cache.get(cache_key, "__missing__")
            if cached != "__missing__":
                edge_cache_hits += 1
                if cached is None:
                    return None
                candidate = dict(cached)
                if include_specs and not candidate.get("result_specs"):
                    specs = operation_specs(left, right, str(candidate["operation"]))
                    if not specs:
                        return None
                    candidate["result_specs"] = specs
                return candidate

            edge_cache_misses += 1
            candidate_evaluations += 1
            overlap_bits = int(group_pattern_bits(left) & group_pattern_bits(right))
            if overlap_bits == 0:
                rejected_no_overlap_candidates += 1
                store_edge_candidate_cache(cache_key, None)
                return None
            before_cost = float(left["cost"]) + float(right["cost"])
            before_memory = int(left["vector_count"]) + int(right["vector_count"])
            best: tuple[tuple[float, int, int], str, float, int, int, list[tuple[dict[int, int], int | None]]] | None = None
            for operation in operations:
                specs = operation_specs(left, right, str(operation))
                if not specs:
                    continue
                after_cost, after_memory, max_partition_size, live_count = specs_cost_memory(specs)
                if int(live_count) <= 0:
                    continue
                memory_saved = int(before_memory) - int(after_memory)
                if int(memory_saved) <= 0:
                    rejected_no_saving_candidates += 1
                    continue
                delta_latency = float(after_cost) - float(before_cost)
                op_rank = int(operation_rank[str(operation)])
                rank = (float(delta_latency), int(max_partition_size), int(op_rank))
                if best is None or rank < best[0]:
                    best = (rank, str(operation), float(delta_latency), int(memory_saved), int(max_partition_size), specs)
            if best is None:
                store_edge_candidate_cache(cache_key, None)
                return None
            _rank, operation, delta_latency, memory_saved, max_partition_size, _specs = best
            score_memory_gain = float(max(1, int(memory_saved)))
            score_memory_m0 = 0.0
            unit_latency_cost = float(delta_latency) / max(float(score_memory_gain), 1e-12)
            if float(delta_latency) <= 0.0:
                latency_class = 0
                memory_per_latency_gain = float(score_memory_gain)
                heap_score = -float(score_memory_gain)
            else:
                latency_class = 1
                memory_per_latency_gain = float(score_memory_gain) / max(float(delta_latency), 1e-12)
                heap_score = float(unit_latency_cost)
            candidate: dict[str, object] = {
                "left_id": int(left_id),
                "right_id": int(right_id),
                "operation": str(operation),
                "score": float(unit_latency_cost),
                "latency_class": int(latency_class),
                "heap_score": float(heap_score),
                "memory_per_latency_gain": float(memory_per_latency_gain),
                "unit_latency_cost": float(unit_latency_cost),
                "delta_latency": float(delta_latency),
                "memory_saved": int(memory_saved),
                "score_memory_gain": float(score_memory_gain),
                "score_memory_m0": float(score_memory_m0),
                "before_cost": float(before_cost),
                "before_memory": int(before_memory),
                "max_result_partition_size": int(max_partition_size),
                "left_version": int(left["version"]),
                "right_version": int(right["version"]),
                "result_specs": _specs,
            }
            store_edge_candidate_cache(cache_key, candidate)
            return candidate

        def push_candidate(left_id: int, right_id: int) -> bool:
            nonlocal next_heap_token, heap_push_count
            left_id = int(left_id)
            right_id = int(right_id)
            if left_id == right_id or left_id not in groups or right_id not in groups:
                return False
            edge = edge_key(left_id, right_id)
            candidate = candidate_for_edge(edge[0], edge[1], include_specs=False)
            if candidate is None:
                return False
            next_heap_token += 1
            edge_heap_tokens[edge] = int(next_heap_token)
            heap_push_count += 1
            heapq.heappush(
                candidate_heap,
                (
                    int(candidate["latency_class"]),
                    float(candidate["heap_score"]),
                    float(candidate["delta_latency"]),
                    int(candidate["max_result_partition_size"]),
                    -int(candidate["memory_saved"]),
                    int(operation_rank[str(candidate["operation"])]),
                    int(edge[0]),
                    int(edge[1]),
                    int(candidate["left_version"]),
                    int(candidate["right_version"]),
                    str(candidate["operation"]),
                    int(next_heap_token),
                ),
            )
            return True

        def add_edge(left_id: int, right_id: int, *, push: bool = True) -> bool:
            edge = edge_key(int(left_id), int(right_id))
            return bool(add_edge_reference(edge, 0.0, push=push))

        def remove_edge(left_id: int, right_id: int) -> None:
            edge = edge_key(left_id, right_id)
            adjacency[edge[0]].discard(edge[1])
            adjacency[edge[1]].discard(edge[0])
            edge_heap_tokens.pop(edge, None)
            edge_refcounts.pop(edge, None)
            edge_signal_scores.pop(edge, None)

        def remove_group_edges(group_id: int) -> None:
            group_id = int(group_id)
            for neighbor_id in list(adjacency.get(group_id, set())):
                remove_edge(group_id, int(neighbor_id))
            adjacency.pop(group_id, None)

        def rebuild_overlap_graph() -> int:
            nonlocal graph_rebuild_count, stale_candidates
            graph_rebuild_count += 1
            stale_candidates = 0
            adjacency.clear()
            candidate_heap.clear()
            edge_heap_tokens.clear()
            edge_candidate_cache.clear()
            edge_candidate_cache_groups.clear()
            pattern_star_edges.clear()
            pattern_core_ids.clear()
            edge_refcounts.clear()
            edge_signal_scores.clear()
            _refreshed, added_edges, _removed = refresh_star_edges_for_patterns(set(pattern_group_ids), push=True)
            return int(added_edges)

        def rebuild_candidate_heap() -> None:
            nonlocal heap_rebuild_count, stale_candidates
            heap_rebuild_count += 1
            stale_candidates = 0
            candidate_heap.clear()
            edge_heap_tokens.clear()
            for left_id in sorted(adjacency):
                for right_id in sorted(adjacency.get(left_id, set())):
                    if int(left_id) < int(right_id):
                        push_candidate(int(left_id), int(right_id))

        def refresh_neighbors_for_patterns(pattern_ids: set[int]) -> int:
            _refreshed, added_edges, _removed = refresh_star_edges_for_patterns(pattern_ids, push=True)
            return int(added_edges)

        def refresh_neighbors_for_groups(group_ids: set[int]) -> int:
            nonlocal incremental_group_refresh_count
            normalized_group_ids = {int(group_id) for group_id in group_ids if int(group_id) in groups}
            if not normalized_group_ids:
                return 0
            pattern_ids: set[int] = set()
            for group_id in sorted(normalized_group_ids):
                group = groups.get(int(group_id))
                if group is None:
                    continue
                pattern_ids.update(int(pattern_id) for pattern_id in group.get("pattern_ids", set()))
            incremental_group_refresh_count += int(len(normalized_group_ids))
            _refreshed, added_edges, _removed = refresh_star_edges_for_patterns(
                pattern_ids,
                push=True,
                focus_group_ids=normalized_group_ids,
            )
            return int(added_edges)

        def pop_valid_candidate() -> dict[str, object] | None:
            nonlocal stale_candidates
            while candidate_heap:
                (
                    _latency_class,
                    _heap_score,
                    _delta_latency,
                    _max_partition_size,
                    _negative_memory_saved,
                    _operation_rank,
                    left_id,
                    right_id,
                    left_version,
                    right_version,
                    _operation,
                    heap_token,
                ) = heapq.heappop(candidate_heap)
                left_id = int(left_id)
                right_id = int(right_id)
                edge = edge_key(left_id, right_id)
                if int(edge_heap_tokens.get(edge, -1)) != int(heap_token):
                    stale_candidates += 1
                    continue
                if left_id not in groups or right_id not in groups:
                    stale_candidates += 1
                    continue
                if int(groups[left_id]["version"]) != int(left_version) or int(groups[right_id]["version"]) != int(right_version):
                    stale_candidates += 1
                    continue
                if right_id not in adjacency.get(left_id, set()):
                    stale_candidates += 1
                    continue
                candidate = candidate_for_edge(left_id, right_id, include_specs=True)
                if candidate is None:
                    stale_candidates += 1
                    continue
                return candidate
            return None

        for group_id in sorted(groups):
            register_group(int(group_id))
        initial_candidate_edges = int(rebuild_overlap_graph())

        progress = tqdm(
            desc="Private core-star planner",
            unit="op",
            leave=False,
            disable=not show_progress,
        )
        stop_reason = "memory_satisfied" if int(total_current_storage) <= int(allowed_total_storage) else "not_started"
        merge_count = 0
        while int(total_current_storage) > int(allowed_total_storage):
            candidate = pop_valid_candidate()
            if candidate is None:
                rebuild_overlap_graph()
                candidate = pop_valid_candidate()
            if candidate is None:
                stop_reason = "no_storage_saving_candidate"
                break

            operation = str(candidate["operation"])
            left_id = int(candidate["left_id"])
            right_id = int(candidate["right_id"])
            if left_id not in groups or right_id not in groups:
                continue
            left = groups[left_id]
            right = groups[right_id]
            specs: list[tuple[dict[int, int], int | None]] = candidate.get("result_specs", [])  # type: ignore[assignment]
            if not specs:
                continue

            before_memory = int(left["vector_count"]) + int(right["vector_count"])
            before_cost = float(left["cost"]) + float(right["cost"])
            affected_patterns = set(int(pattern_id) for pattern_id in left.get("pattern_ids", set())) | set(
                int(pattern_id) for pattern_id in right.get("pattern_ids", set())
            )
            if _PRIVATE_PLANNER_TRACE_HOOK is not None:
                _PRIVATE_PLANNER_TRACE_HOOK(locals())

            prune_edge_candidate_cache({int(left_id), int(right_id)})
            unregister_group(left_id)
            unregister_group(right_id)
            groups.pop(left_id, None)
            groups.pop(right_id, None)

            new_group_ids: set[int] = set()
            after_memory = 0
            after_cost = 0.0
            for spec in specs:
                new_group_bits, new_group_stored_bits = normalize_spec(spec)
                new_group = make_group_from_bits(int(next_group_id), new_group_bits, stored_bits=new_group_stored_bits)
                if not is_live_group(new_group):
                    continue
                new_group_id = int(next_group_id)
                next_group_id += 1
                groups[new_group_id] = new_group
                register_group(new_group_id)
                affected_patterns.update(int(pattern_id) for pattern_id in new_group.get("pattern_ids", set()))
                new_group_ids.add(new_group_id)
                after_memory += int(new_group["vector_count"])
                after_cost += float(new_group["cost"])

            memory_saved = int(before_memory) - int(after_memory)
            delta_latency = float(after_cost) - float(before_cost)
            if int(memory_saved) <= 0 or not new_group_ids:
                stop_reason = "invalid_candidate_after_recompute"
                break

            private_current_storage -= int(memory_saved)
            total_current_storage -= int(memory_saved)
            current_cost += float(delta_latency)
            total_storage_reduction += int(memory_saved)
            total_latency_delta += float(delta_latency)
            operation_counts[str(operation)] += 1
            merge_count += 1
            last_operation = str(operation)
            last_candidate_score = float(candidate["score"])
            last_candidate_delta_latency = float(delta_latency)
            last_candidate_memory_saved = int(memory_saved)
            if new_group_ids:
                refreshed_edge_count += int(refresh_neighbors_for_groups(new_group_ids))
            else:
                refreshed_edge_count += int(refresh_neighbors_for_patterns(affected_patterns))

            progress.update(1)
            if show_progress:
                progress.set_postfix(
                    {
                        "groups": int(len(groups)),
                        "private_storage": int(private_current_storage),
                        "budget": int(allowed_private_storage),
                        "saved": int(memory_saved),
                        "op": str(operation),
                    }
                )
            if int(total_current_storage) <= int(allowed_total_storage):
                stop_reason = "memory_satisfied"
                break
            stop_reason = "memory_candidate_pending"
            if int(stale_candidates) > max(4096, int(len(candidate_heap) // 2)):
                rebuild_candidate_heap()
        progress.close()
        if int(total_current_storage) <= int(allowed_total_storage):
            stop_reason = "memory_satisfied"

        selectivity_refine_stop_reason = "not_started"
        selectivity_heap: list[tuple[float, float, int, int, int, int]] = []
        for group_id, group in groups.items():
            profile = group_selectivity_profile(int(group_id), group)
            if profile is None:
                continue
            heapq.heappush(
                selectivity_heap,
                (
                    float(profile["avg_selectivity"]),
                    float(profile["worst_selectivity"]),
                    -int(profile["partition_vectors"]),
                    int(group_id),
                    int(group.get("version", 0)),
                    int(profile["worst_tenant"]),
                ),
            )
        if not selectivity_heap:
            selectivity_refine_stop_reason = "no_refine_group"
        while selectivity_heap:
            (
                _avg_selectivity,
                _worst_selectivity,
                _negative_vectors,
                group_id,
                group_version,
                _worst_tenant_from_heap,
            ) = heapq.heappop(selectivity_heap)
            if int(group_id) not in groups:
                continue
            group = groups[int(group_id)]
            if int(group.get("version", 0)) != int(group_version):
                continue
            profile = group_selectivity_profile(int(group_id), group)
            if profile is None:
                continue
            selectivity_refine_last_group_id = int(group_id)
            selectivity_refine_last_worst_tenant = int(profile["worst_tenant"])
            selectivity_refine_last_avg_selectivity = float(profile["avg_selectivity"])
            selectivity_refine_last_worst_selectivity = float(profile["worst_selectivity"])
            if bool(profile["is_pure"]):
                selectivity_refine_stop_reason = "worst_group_pure"
                break
            refine_specs = selectivity_extract_specs(group, int(profile["worst_tenant"]))
            if not refine_specs:
                selectivity_refine_stop_reason = "no_extractable_worst_tenant_bits"
                break
            before_cost = float(group["cost"])
            before_memory = int(group["vector_count"])
            after_cost, after_memory, _max_partition_size, live_count = specs_cost_memory(refine_specs)
            if int(live_count) <= 1 or int(after_memory) != int(before_memory):
                selectivity_refine_stop_reason = "invalid_extract_specs"
                break
            delta_latency = float(after_cost) - float(before_cost)
            if float(delta_latency) >= 0.0:
                selectivity_refine_stop_reason = "worst_group_not_beneficial"
                break

            prune_edge_candidate_cache({int(group_id)})
            unregister_group(int(group_id))
            groups.pop(int(group_id), None)

            new_group_ids: set[int] = set()
            for spec in refine_specs:
                new_group_bits, new_group_stored_bits = normalize_spec(spec)
                new_group = make_group_from_bits(int(next_group_id), new_group_bits, stored_bits=new_group_stored_bits)
                if not is_live_group(new_group):
                    continue
                new_group_id = int(next_group_id)
                next_group_id += 1
                groups[new_group_id] = new_group
                new_group_ids.add(new_group_id)
                new_profile = group_selectivity_profile(int(new_group_id), new_group)
                if new_profile is not None:
                    heapq.heappush(
                        selectivity_heap,
                        (
                            float(new_profile["avg_selectivity"]),
                            float(new_profile["worst_selectivity"]),
                            -int(new_profile["partition_vectors"]),
                            int(new_group_id),
                            int(new_group.get("version", 0)),
                            int(new_profile["worst_tenant"]),
                        ),
                    )
            if not new_group_ids:
                selectivity_refine_stop_reason = "extract_created_no_live_group"
                break
            current_cost += float(delta_latency)
            selectivity_refine_cost_delta += float(delta_latency)
            selectivity_refine_count += 1
            operation_counts["selectivity_extract"] += 1
            selectivity_refine_stop_reason = "refine_candidate_pending"
        else:
            if selectivity_refine_stop_reason == "not_started" or selectivity_refine_stop_reason == "refine_candidate_pending":
                selectivity_refine_stop_reason = "no_refine_group"
        if int(selectivity_refine_count) > 0:
            adjacency.clear()
            candidate_heap.clear()
            edge_heap_tokens.clear()
            edge_candidate_cache.clear()
            edge_candidate_cache_groups.clear()
            pattern_star_edges.clear()
            pattern_core_ids.clear()
            edge_refcounts.clear()
            edge_signal_scores.clear()

        # Second-stage tenant-similarity merge is intentionally disabled. The
        # planner now relies only on the first-stage private core-star operations;
        # this keeps the partition state directly attributable to the Cost Model.
        tenant_similarity_initial_group_count = int(len(groups))
        tenant_similarity_candidate_count = 0
        tenant_similarity_merge_count = 0
        tenant_similarity_stale_candidate_count = 0
        tenant_similarity_heap_push_count = 0
        tenant_similarity_edge_refresh_count = 0
        tenant_similarity_total_gain = 0.0
        tenant_similarity_last_gain = None
        tenant_similarity_stop_reason = "disabled"
        tenant_similarity_top_d = 0
        tenant_similarity_min_partition_vectors = 0
        tenant_similarity_final_group_count = int(len(groups))

        compact_ids = {group_id: index for index, group_id in enumerate(sorted(groups))}
        private_groups_for_partitions: list[dict[str, object]] = []
        tenant_to_cluster: dict[int, int] = {}
        tenant_primary_rank: dict[int, tuple[int, int]] = {}
        for group_id in sorted(groups):
            compact_id = int(compact_ids[int(group_id)])
            group = groups[int(group_id)]
            tenant_pattern_map: dict[int, set[int]] = group.get("service_tenant_patterns", group.get("tenant_patterns", {}))  # type: ignore[assignment]
            compact_tenant_patterns = {
                int(tenant_id): set(int(pattern_id) for pattern_id in pattern_ids)
                for tenant_id, pattern_ids in tenant_pattern_map.items()
                if pattern_ids
            }
            if not compact_tenant_patterns:
                continue
            private_groups_for_partitions.append(
                {
                    "cluster_id": int(compact_id),
                    "pattern_ids": set(int(pattern_id) for pattern_id in group["pattern_ids"]),  # type: ignore[index]
                    "tenant_patterns": compact_tenant_patterns,
                    "service_tenant_patterns": compact_tenant_patterns,
                    "vector_count": int(group["vector_count"]),
                }
            )
            tenant_access: dict[int, int] = group.get("tenant_access", {})  # type: ignore[assignment]
            for tenant_id, pattern_ids in compact_tenant_patterns.items():
                cached_access = tenant_access.get(int(tenant_id))
                if cached_access is None:
                    accessible_vectors = int(vector_count_bits(pattern_bits_for(tuple(pattern_ids))))
                else:
                    accessible_vectors = int(cached_access)
                rank = (int(accessible_vectors), -int(compact_id))
                if int(tenant_id) not in tenant_primary_rank or rank > tenant_primary_rank[int(tenant_id)]:
                    tenant_primary_rank[int(tenant_id)] = rank
                    tenant_to_cluster[int(tenant_id)] = int(compact_id)
        for tenant_id in tenant_ids:
            tenant_to_cluster.setdefault(int(tenant_id), 0)
        self._last_private_groups = private_groups_for_partitions

        cluster_sizes = [int(len(group.get("tenant_patterns", {}))) for group in groups.values()]
        cluster_vector_counts = [int(group["vector_count"]) for group in groups.values()]
        active_adjacency_edges = int(sum(len(neighbors) for neighbors in adjacency.values()) // 2)
        adjacency_degrees = [int(len(neighbors)) for group_id, neighbors in adjacency.items() if int(group_id) in groups]
        route_counts: Counter = Counter()
        weighted_filter = 0.0
        for group in groups.values():
            group_size = int(group["vector_count"])
            tenant_access: dict[int, int] = group.get("tenant_access", {})  # type: ignore[assignment]
            for tenant_id, accessible_vectors in tenant_access.items():
                route_counts[int(tenant_id)] += 1
                weighted_filter += float(tenant_query_weights.get(int(tenant_id), 1.0)) * (float(group_size) / float(max(1, int(accessible_vectors))))

        self._last_private_metadata = {
            "enabled": True,
            "planner": "private_core_star_split_merge_v16",
            "objective": "v16 rollback: start from one private group per tenant; compress replicated ACLs with ACL-core star graph and five operations until total storage reaches budget",
            "target_cluster_count": int(target_cluster_count),
            "target_cluster_count_enforced": False,
            "replication_budget_ratio": float(max(0.0, float(replication_budget_ratio))),
            "allowed_total_storage": int(allowed_total_storage),
            "allowed_private_storage": int(allowed_private_storage),
            "shared_vector_count": int(shared_vector_count),
            "shared_adjusted_private_budget": int(shared_adjusted_private_budget),
            "base_all_private_storage_for_private_patterns": int(base_all_private_storage_for_patterns),
            "private_budget_rule": "allowed_private_storage = max(private_unique_vectors, allowed_total_storage - shared_vector_count); shared ACLs consume one copy each, so private compression only needs to fit the remaining budget",
            "initial_total_storage": int(initial_total_storage),
            "final_total_storage": int(total_current_storage),
            "private_unique_vectors": int(private_unique_vectors),
            "initial_private_storage": int(initial_private_storage),
            "final_private_storage": int(private_current_storage),
            "private_replication_factor": float(private_current_storage / max(1, private_unique_vectors)),
            "initial_group_count": int(initial_group_count),
            "final_group_count": int(len(groups)),
            "operation_count": int(merge_count),
            "full_merge_count": int(operation_counts.get("full", 0)),
            "move_left_count": int(operation_counts.get("move_left", 0)),
            "move_right_count": int(operation_counts.get("move_right", 0)),
            "split_overlap_count": int(operation_counts.get("split_overlap", 0)),
            "merge_extract_overlap_count": int(operation_counts.get("merge_extract_overlap", 0)),
            "tenant_similarity_merge_count": int(tenant_similarity_merge_count),
            "tenant_similarity_candidate_count": int(tenant_similarity_candidate_count),
            "tenant_similarity_heap_push_count": int(tenant_similarity_heap_push_count),
            "tenant_similarity_edge_refresh_count": int(tenant_similarity_edge_refresh_count),
            "tenant_similarity_stale_candidate_count": int(tenant_similarity_stale_candidate_count),
            "tenant_similarity_initial_group_count": int(tenant_similarity_initial_group_count),
            "tenant_similarity_final_group_count": int(tenant_similarity_final_group_count),
            "tenant_similarity_total_gain": float(tenant_similarity_total_gain),
            "tenant_similarity_last_gain": None if tenant_similarity_last_gain is None else float(tenant_similarity_last_gain),
            "tenant_similarity_stop_reason": str(tenant_similarity_stop_reason),
            "tenant_similarity_top_d": int(tenant_similarity_top_d),
            "tenant_similarity_min_partition_vectors": int(tenant_similarity_min_partition_vectors),
            "selectivity_extract_count": int(selectivity_refine_count),
            "selectivity_extract_cost_delta": float(selectivity_refine_cost_delta),
            "selectivity_extract_stop_reason": str(selectivity_refine_stop_reason),
            "selectivity_extract_last_group_id": None if selectivity_refine_last_group_id is None else int(selectivity_refine_last_group_id),
            "selectivity_extract_last_worst_tenant": None if selectivity_refine_last_worst_tenant is None else int(selectivity_refine_last_worst_tenant),
            "selectivity_extract_last_avg_selectivity": None if selectivity_refine_last_avg_selectivity is None else float(selectivity_refine_last_avg_selectivity),
            "selectivity_extract_last_worst_selectivity": None if selectivity_refine_last_worst_selectivity is None else float(selectivity_refine_last_worst_selectivity),
            "total_storage_reduction": int(total_storage_reduction),
            "total_latency_delta": float(total_latency_delta) + float(selectivity_refine_cost_delta) - float(tenant_similarity_total_gain),
            "cost_initial": float(initial_cost),
            "cost_final": float(current_cost),
            "candidate_score_rule": "for each edge choose op with min delta_latency among memory-saving operations; heap first takes delta_latency<=0 memory-saving candidates, then ranks positive-loss candidates by delta_latency/memory_saved",
            "cost_model": str(cost_model_metadata(DEFAULT_COST_MODEL)["cost_model"]),
            "graph_rule": "ACL-core star graph: for each ACL choose the owner with the fewest ACLs as core, then largest in-group vector share as tie-break and connect core to other owners; edge signal vector_count(a); top-d rank uses shared_acl_count/sqrt(|ACL(Gi)|*|ACL(Gj)|); each group keeps top-d incident candidates",
            "graph_update_rule": "v16 incremental core-star: maintain pattern_star_edges, pattern_core_ids, edge_refcounts, edge_signal_scores, and group-indexed candidate cache; after each operation refresh only new groups' ACL patterns and their top-d incident star edges; rebuild graph only as fallback when heap is empty",
            "operation_rule": "five operations: full keeps whole-group A+B merge; move_left, move_right, split_overlap, and merge_extract_overlap use I as the full ACL intersection; merge_extract_overlap produces (A-minus-I)+(B-minus-I) and I",
            "selectivity_refinement_rule": "after memory compression, repeatedly take the globally worst average-selectivity private group, extract the worst tenant route ACL block into a separate partition if route-level Cost Model decreases; stop immediately when that worst group is pure or not beneficial",
            "private_edge_top_d": int(top_d),
            "initial_candidate_edges": int(initial_candidate_edges),
            "active_adjacency_edges": int(active_adjacency_edges),
            "active_heap_entries": int(len(candidate_heap)),
            "heap_push_count": int(heap_push_count),
            "stale_candidates": int(stale_candidates),
            "heap_rebuild_count": int(heap_rebuild_count),
            "graph_rebuild_count": int(graph_rebuild_count),
            "candidate_evaluations": int(candidate_evaluations),
            "edge_cache_size": int(len(edge_candidate_cache)),
            "edge_cache_group_index_size": int(sum(len(values) for values in edge_candidate_cache_groups.values())),
            "edge_cache_hits": int(edge_cache_hits),
            "edge_cache_misses": int(edge_cache_misses),
            "edge_cache_prune_count": int(edge_cache_prune_count),
            "refreshed_edge_count": int(refreshed_edge_count),
            "incremental_group_refresh_count": int(incremental_group_refresh_count),
            "incremental_pattern_refresh_count": int(incremental_pattern_refresh_count),
            "incremental_edge_add_count": int(incremental_edge_add_count),
            "incremental_edge_remove_count": int(incremental_edge_remove_count),
            "dag_edge_count": int(active_adjacency_edges),
            "edge_refcount_entries": int(len(edge_refcounts)),
            "rejected_no_overlap_candidates": int(rejected_no_overlap_candidates),
            "rejected_no_saving_candidates": int(rejected_no_saving_candidates),
            "max_adjacency_degree": int(max(adjacency_degrees)) if adjacency_degrees else 0,
            "mean_adjacency_degree": float(sum(adjacency_degrees) / len(adjacency_degrees)) if adjacency_degrees else 0.0,
            "route_count_min": int(min(route_counts.values())) if route_counts else 0,
            "route_count_mean": float(sum(route_counts.values()) / len(route_counts)) if route_counts else 0.0,
            "route_count_max": int(max(route_counts.values())) if route_counts else 0,
            "weighted_filter_ratio": float(weighted_filter),
            "min_cluster_size": int(min(cluster_sizes)) if cluster_sizes else 0,
            "max_cluster_size": int(max(cluster_sizes)) if cluster_sizes else 0,
            "min_cluster_vectors": int(min(cluster_vector_counts)) if cluster_vector_counts else 0,
            "max_cluster_vectors": int(max(cluster_vector_counts)) if cluster_vector_counts else 0,
            "last_operation": None if last_operation is None else str(last_operation),
            "last_candidate_score": None if last_candidate_score is None else float(last_candidate_score),
            "last_candidate_delta_latency": None if last_candidate_delta_latency is None else float(last_candidate_delta_latency),
            "last_candidate_memory_saved": None if last_candidate_memory_saved is None else int(last_candidate_memory_saved),
            "stop_reason": str(stop_reason),
            "query_weight_rule": "all tenants use q_t=1 unless caller passes explicit tenant_query_weights; current build_plan passes q_t=1",
        }
        return tenant_to_cluster

    def _build_partitions(
        self,
        *,
        shared_patterns: list[ACLPattern],
        private_patterns: list[ACLPattern],
        shared_assignments: dict[int, int],
        tenant_to_private_cluster: dict[int, int],
        private_cluster_count: int,
        shared_cluster_count: int,
        show_progress: bool,
    ) -> list[KMeansPartition]:
        partitions: list[KMeansPartition] = []
        shared_by_cluster: dict[int, list[ACLPattern]] = defaultdict(list)
        for pattern in shared_patterns:
            shared_by_cluster[int(shared_assignments.get(int(pattern.pattern_id), 0))].append(pattern)
        shared_partition_index = 0
        for cluster_id in sorted(shared_by_cluster):
            cluster_patterns = shared_by_cluster.get(int(cluster_id), [])
            if cluster_patterns:
                partitions.append(
                    self._make_partition(
                        f"shared_{shared_partition_index}",
                        cluster_id,
                        "shared",
                        cluster_patterns,
                    )
                )
                shared_partition_index += 1

        private_group_records = list(getattr(self, "_last_private_groups", []) or [])
        pattern_by_id = {int(pattern.pattern_id): pattern for pattern in private_patterns}
        private_partition_index = 0
        if private_group_records:
            for group in sorted(private_group_records, key=lambda item: int(item.get("cluster_id", 0))):
                cluster_id = int(group.get("cluster_id", private_partition_index))
                pattern_ids = sorted(int(pattern_id) for pattern_id in group.get("pattern_ids", set()) if int(pattern_id) in pattern_by_id)
                cluster_patterns = [pattern_by_id[int(pattern_id)] for pattern_id in pattern_ids]
                raw_tenant_pattern_map = group.get("service_tenant_patterns", group.get("tenant_patterns", {}))
                tenant_pattern_map = {
                    int(tenant_id): {int(pattern_id) for pattern_id in pattern_values if int(pattern_id) in pattern_by_id}
                    for tenant_id, pattern_values in dict(raw_tenant_pattern_map).items()
                }
                tenant_pattern_map = {tenant_id: pattern_values for tenant_id, pattern_values in tenant_pattern_map.items() if pattern_values}
                if cluster_patterns and tenant_pattern_map:
                    partitions.append(
                        self._make_partition(
                            f"private_{private_partition_index}",
                            cluster_id,
                            "private",
                            cluster_patterns,
                            route_tenant_patterns=tenant_pattern_map,
                        )
                    )
                    private_partition_index += 1
        else:
            private_by_cluster: dict[int, dict[int, ACLPattern]] = defaultdict(dict)
            for pattern in tqdm(private_patterns, desc="Private cluster copies", unit="acl", leave=False, disable=not show_progress):
                owning_clusters = {
                    int(tenant_to_private_cluster[int(tenant_id)])
                    for tenant_id in pattern.tenant_ids
                    if int(tenant_id) in tenant_to_private_cluster
                }
                for cluster_id in owning_clusters:
                    private_by_cluster[int(cluster_id)][int(pattern.pattern_id)] = pattern
            for cluster_id in sorted(private_by_cluster):
                cluster_patterns = [private_by_cluster[int(cluster_id)][pattern_id] for pattern_id in sorted(private_by_cluster[int(cluster_id)])]
                if cluster_patterns:
                    partitions.append(
                        self._make_partition(
                            f"private_{private_partition_index}",
                            cluster_id,
                            "private",
                            cluster_patterns,
                        )
                    )
                    private_partition_index += 1
        return partitions

    def _make_partition(
        self,
        partition_id: str,
        cluster_id: int,
        partition_kind: str,
        patterns: list[ACLPattern],
        route_tenant_patterns: Optional[dict[int, set[int]]] = None,
    ) -> KMeansPartition:
        if route_tenant_patterns is not None:
            tenant_ids = tuple(sorted(int(tenant_id) for tenant_id, pattern_ids in route_tenant_patterns.items() if pattern_ids))
        else:
            tenant_ids = tuple(sorted({int(tenant_id) for pattern in patterns for tenant_id in pattern.tenant_ids}))
        document_pattern_pairs = tuple(
            (int(document_id), int(pattern.pattern_id))
            for pattern in patterns
            for document_id in pattern.document_ids
        )
        document_ids = tuple(sorted({int(document_id) for document_id, _ in document_pattern_pairs}))
        pattern_ids = tuple(sorted(int(pattern.pattern_id) for pattern in patterns))
        vector_count = int(sum(int(pattern.vector_count) for pattern in patterns))
        return KMeansPartition(
            partition_id=str(partition_id),
            cluster_id=int(cluster_id),
            partition_kind=str(partition_kind),
            table_name=get_partition_table_name(str(partition_id)),
            tenant_ids=tenant_ids,
            pattern_ids=pattern_ids,
            document_ids=document_ids,
            document_pattern_pairs=document_pattern_pairs,
            vector_count=vector_count,
            metadata={
                "partition_kind": str(partition_kind),
                "pattern_count": int(len(pattern_ids)),
                "pattern_tenants": {
                    str(int(pattern.pattern_id)): [int(tenant_id) for tenant_id in pattern.tenant_ids]
                    for pattern in patterns
                },
                "tenant_patterns": {
                    str(int(tenant_id)): sorted(int(pattern_id) for pattern_id in pattern_ids)
                    for tenant_id, pattern_ids in (route_tenant_patterns or {}).items()
                    if pattern_ids
                } if route_tenant_patterns is not None else {},
            },
        )

    def _build_routes(
        self,
        partitions: list[KMeansPartition],
        *,
        tenant_to_private_cluster: dict[int, int],
    ) -> list[TenantRoute]:
        routes: list[TenantRoute] = []
        for partition in partitions:
            tenant_to_patterns: dict[int, list[int]] = defaultdict(list)
            explicit_tenant_patterns = partition.metadata.get("tenant_patterns", {}) or {}
            if explicit_tenant_patterns:
                for tenant_id_text, pattern_values in dict(explicit_tenant_patterns).items():
                    tenant_to_patterns[int(tenant_id_text)].extend(int(pattern_id) for pattern_id in pattern_values)
            else:
                pattern_tenant_map = partition.metadata.get("pattern_tenants", {}) or {}
                if not pattern_tenant_map:
                    continue
                for pattern_id_text, tenant_values in pattern_tenant_map.items():
                    for tenant_id in tenant_values:
                        if (
                            str(partition.partition_kind) == "private"
                            and int(tenant_to_private_cluster.get(int(tenant_id), -1)) != int(partition.cluster_id)
                        ):
                            continue
                        tenant_to_patterns[int(tenant_id)].append(int(pattern_id_text))
            for tenant_id, pattern_ids in tenant_to_patterns.items():
                normalized_pattern_ids = tuple(sorted(set(int(pattern_id) for pattern_id in pattern_ids)))
                if not normalized_pattern_ids:
                    continue
                routes.append(
                    TenantRoute(
                        tenant_id=int(tenant_id),
                        partition_id=str(partition.partition_id),
                        table_name=str(partition.table_name),
                        route_kind=str(partition.partition_kind),
                        cluster_id=int(partition.cluster_id),
                        pattern_ids=normalized_pattern_ids,
                    )
                )
        return routes
