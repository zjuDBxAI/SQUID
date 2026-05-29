from __future__ import annotations

from collections import Counter, defaultdict
import heapq
import json
import math
from pathlib import Path
from typing import Optional

from tqdm import tqdm

from .common import ACLPattern, KMeansPartition, KMeansPlan, TenantRoute, get_partition_table_name


_COST_TOPK = 5
_COST_GAMMA = 1.0
_COST_EPS = 1e-6
_PRIVATE_MERGE_NEIGHBOR_LIMIT = 128
_PRIVATE_FALLBACK_POOL_LIMIT = 128
_SHARED_MERGE_NEIGHBOR_LIMIT = 128
_SHARED_TENANT_PATTERN_CAP = 512
_SHARED_FALLBACK_POOL_LIMIT = 128


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
        self._last_private_metadata: dict[str, object] = {}
        self._last_shared_metadata: dict[str, object] = {}
        self._last_split_metadata: dict[str, object] = {}

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
        ef_search: int = 120,
        show_progress: bool = True,
    ) -> KMeansPlan:
        if not acl_rows:
            raise ValueError("Cannot build kmeans ACL partitions without ACL rows")

        with tqdm(total=7, desc="Cost split planner", unit="stage", disable=not show_progress) as progress:
            patterns = self._build_patterns(acl_rows, show_progress=show_progress)
            progress.update(1)
            progress.set_description("Cost split planner: built ACL patterns")

            tenant_ids = tuple(sorted({tenant_id for pattern in patterns for tenant_id in pattern.tenant_ids}))
            if not tenant_ids:
                raise ValueError("No tenants found in ACL rows")
            tenant_vector_counts = self._tenant_vector_counts(patterns)
            total_original_vectors = int(sum(int(pattern.vector_count) for pattern in patterns))
            workload_frequencies = _load_workload_frequencies(query_dataset_path)
            tenant_query_weights = self._tenant_query_weights(tenant_ids, workload_frequencies)
            progress.update(1)
            progress.set_description("Cost split planner: loaded tenant stats")

            shared_patterns, private_patterns, shared_threshold = self._split_patterns(
                patterns,
                shared_score_ratio=float(shared_score_ratio),
                private_cluster_count=int(private_cluster_count),
                tenant_ids=tenant_ids,
                tenant_vector_counts=tenant_vector_counts,
                tenant_query_weights=tenant_query_weights,
                total_original_vectors=total_original_vectors,
                ef_search=int(ef_search),
                private_replication_budget_ratio=float(private_replication_budget_ratio),
            )
            progress.update(1)
            progress.set_description("Cost split planner: split shared/private ACLs")
            shared_assignments = self._assign_shared_groups_by_cost_split(
                shared_patterns,
                cluster_count=int(shared_cluster_count),
                route_limit=int(shared_route_limit),
                tenant_query_weights=tenant_query_weights,
                total_original_vectors=total_original_vectors,
                ef_search=int(ef_search),
                show_progress=show_progress,
            )
            progress.update(1)
            progress.set_description("Cost split planner: built shared splits")

            tenant_to_private_cluster = self._cluster_private_tenants_by_cost_split(
                private_patterns,
                tenant_ids=tenant_ids,
                cluster_count=int(private_cluster_count),
                replication_budget_ratio=float(private_replication_budget_ratio),
                tenant_query_weights=tenant_query_weights,
                total_original_vectors=total_original_vectors,
                shared_vector_count=int(sum(int(pattern.vector_count) for pattern in shared_patterns)),
                ef_search=int(ef_search),
                show_progress=show_progress,
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
            "algorithm": "cost_guided_two_zone_split_private_merge_v1",
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
            "ef_search_for_cost": int(ef_search),
            "topk_for_cost": int(_COST_TOPK),
            "cost_model": "shared uses bottom-up ACL merge by exact pre/post tenant query cost; private uses bottom-up ACL-overlap merge with HNSW/filter effort",
            "shared_cost_model": "bottom-up ACL group merge: C(t,G)=R0+log(1+V_G)*max(k, ef*log(1+V_G)/log(1+N), k/selectivity); candidate uses total affected query gain C_before-C_after",
            "private_cost_model": (
                "sum_t q_t * log(1+|P|) * max(k, ef*log(1+|P|)/log(1+N), "
                "k/max(A(t,P)/|P|, eps))"
            ),
            "shared_score_rule": "adaptive ACL marginal cost split with reference private groups and storage multiplier",
            "shared_query_weight_rule": "q_t fixed to 1 in shared cost model",
            "shared_vector_ratio_actual": float(
                sum(int(pattern.vector_count) for pattern in shared_patterns) / max(total_original_vectors, 1)
            ),
            "shared_pattern_ratio_actual": float(len(shared_patterns) / max(len(patterns), 1)),
            "private_objective": "bottom-up tenant merge minimizing query cost increase per storage saved under private storage budget",
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
        ef_search: int,
        private_replication_budget_ratio: float,
    ) -> tuple[list[ACLPattern], list[ACLPattern], Optional[float]]:
        if not patterns:
            self._last_split_metadata = {
                "enabled": False,
                "reason": "no_patterns",
                "shared_score_ratio_input": float(shared_score_ratio),
            }
            return [], [], None

        tenant_ids = tuple(sorted(int(tenant_id) for tenant_id in tenant_ids))
        pattern_weights = {int(pattern.pattern_id): max(0, int(pattern.vector_count)) for pattern in patterns}

        route_startup_samples = sorted(
            float(_COST_TOPK) * math.log1p(max(1, int(pattern.vector_count)))
            for pattern in patterns
            if int(pattern.vector_count) > 0
        )
        route_startup_cost = (
            float(route_startup_samples[len(route_startup_samples) // 2])
            if route_startup_samples
            else float(_COST_TOPK)
        )
        total_vector_scale = max(math.log1p(max(1, int(total_original_vectors))), 1.0)

        def split_query_cost(partition_vectors: int, accessible_vectors: int) -> float:
            partition_vectors = max(1, int(partition_vectors))
            accessible_vectors = int(accessible_vectors)
            if accessible_vectors <= 0:
                return 0.0
            selectivity = float(accessible_vectors) / float(partition_vectors)
            size_scaled_ef = float(max(1, int(ef_search))) * math.log1p(partition_vectors) / float(total_vector_scale)
            filter_scaled_ef = float(_COST_TOPK) / max(selectivity, _COST_EPS)
            effort = max(float(_COST_TOPK), float(size_scaled_ef), float(filter_scaled_ef))
            return float(route_startup_cost) + math.log1p(partition_vectors) * float(effort)

        # Reference private groups are only an estimator for shared/private split.
        # They avoid running the full private merge before the split decision.
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
            math.floor(float(max(1, int(total_original_vectors))) * (1.0 + max(0.0, float(private_replication_budget_ratio))))
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
        touched_group_counts = [int(row["touched_group_count"]) for row in split_rows]
        self._last_split_metadata = {
            "enabled": True,
            "rule": "adaptive_acl_marginal_cost_with_reference_private_groups",
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
            "shared_admission_unit_cost": float(shared_admission_unit_cost),
            "shared_admission_rule": "sort by positive net gain and admit ACL a only if gain(a) > R0 * sum_{b in S}(|T_a| + |T_b|)",
            "estimated_storage_budget_satisfied": bool(int(selected_storage) <= int(allowed_storage)),
            "private_reference_group_count": int(len(group_tenants)),
            "private_reference_owner_rule": "owner(t)=argmax_a n_a/|T_a|, tie by larger n_a, smaller |T_a|, smaller pattern_id",
            "route_startup_cost": float(route_startup_cost),
            "query_weight_rule": "q_t fixed to 1.0 per tenant for split estimation",
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
        raw = {int(tenant_id): max(0.0, float(workload_frequencies.get(int(tenant_id), 0.0))) for tenant_id in tenant_ids}
        total = float(sum(raw.values()))
        if total <= 0.0:
            uniform = 1.0 / float(max(1, len(tenant_ids)))
            return {int(tenant_id): float(uniform) for tenant_id in tenant_ids}
        return {int(tenant_id): float(value / total) for tenant_id, value in raw.items()}

    def _partition_query_cost(
        self,
        *,
        partition_vectors: int,
        accessible_vectors: int,
        tenant_weight: float,
        total_vectors: int,
        ef_search: int,
    ) -> float:
        if int(partition_vectors) <= 0 or int(accessible_vectors) <= 0 or float(tenant_weight) <= 0.0:
            return 0.0
        partition_vectors = max(1, int(partition_vectors))
        total_vectors = max(partition_vectors, int(total_vectors), 1)
        selectivity = float(accessible_vectors) / float(partition_vectors)
        size_scaled_ef = float(max(1, int(ef_search))) * math.log1p(partition_vectors) / max(math.log1p(total_vectors), 1.0)
        filter_scaled_ef = float(_COST_GAMMA * _COST_TOPK / max(selectivity, _COST_EPS))
        effort = max(float(_COST_TOPK), float(size_scaled_ef), float(filter_scaled_ef))
        return float(tenant_weight) * math.log1p(partition_vectors) * effort

    def _assign_shared_groups_by_cost_split(
        self,
        patterns: list[ACLPattern],
        *,
        cluster_count: int,
        route_limit: int,
        tenant_query_weights: dict[int, float],
        total_original_vectors: int,
        ef_search: int,
        show_progress: bool,
    ) -> dict[int, int]:
        if not patterns:
            self._last_shared_metadata = {"enabled": False, "reason": "no_shared_patterns"}
            return {}

        target_cluster_count = max(1, min(int(cluster_count), len(patterns)))
        pattern_by_id = {int(pattern.pattern_id): pattern for pattern in patterns}
        pattern_weights = {int(pattern.pattern_id): int(pattern.vector_count) for pattern in patterns}
        route_startup_samples = sorted(
            float(_COST_TOPK) * math.log1p(max(1, int(pattern.vector_count)))
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
            vector_count = int(vector_count)
            accessible_vectors = int(accessible_vectors)
            if vector_count <= 0 or accessible_vectors <= 0:
                return 0.0
            partition_vectors = max(1, int(vector_count))
            total_vectors = max(partition_vectors, int(total_original_vectors), 1)
            selectivity = float(accessible_vectors) / float(partition_vectors)
            size_scaled_ef = float(max(1, int(ef_search))) * math.log1p(partition_vectors) / max(
                math.log1p(total_vectors),
                1.0,
            )
            filter_scaled_ef = float(_COST_TOPK / max(selectivity, _COST_EPS))
            effort = max(float(_COST_TOPK), float(size_scaled_ef), float(filter_scaled_ef))
            return float(route_startup_cost) + float(tenant_weight(int(tenant_id))) * math.log1p(partition_vectors) * float(effort)

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

        initial_group_ids = sorted(int(group_id) for group_id in groups)
        for index, left_id in enumerate(initial_group_ids):
            for right_id in initial_group_ids[index + 1:]:
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

            for other_id in list(groups):
                if int(other_id) != int(new_group_id):
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
            "objective": "bottom-up shared ACL merge by global exact pre/post tenant query-cost gain",
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
            "initial_candidate_edges": int(initial_candidate_edges),
            "stale_candidates": int(stale_candidates),
            "tenant_pattern_cap": None,
            "capped_tenant_count": 0,
            "skipped_pattern_memberships_by_cap": 0,
            "neighbor_limit": None,
            "fallback_pool_limit": None,
            "global_candidate_graph": True,
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
            "merge_score_rule": "choose global candidate with max total Gain=C_before-C_after; merge only when Gain>0",
            "route_startup_cost": float(route_startup_cost),
            "candidate_rule": "global all-pair shared ACL group heap; after each merge refresh new group against every active group",
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
        ef_search: int,
        show_progress: bool,
    ) -> dict[int, int]:
        tenant_ids = tuple(sorted(int(tenant_id) for tenant_id in tenant_ids))
        tenant_count = int(len(tenant_ids))
        if tenant_count == 0:
            self._last_private_metadata = {"enabled": False, "reason": "empty_tenant_input"}
            return {}

        target_cluster_count = max(1, min(int(cluster_count), tenant_count))
        pattern_weights = {int(pattern.pattern_id): int(pattern.vector_count) for pattern in patterns}
        tenant_patterns: dict[int, set[int]] = {int(tenant_id): set() for tenant_id in tenant_ids}
        for pattern in patterns:
            pattern_id = int(pattern.pattern_id)
            for tenant_id in pattern.tenant_ids:
                tenant_id = int(tenant_id)
                if tenant_id in tenant_patterns:
                    tenant_patterns[tenant_id].add(pattern_id)

        tenant_private_vectors = {
            int(tenant_id): int(sum(pattern_weights.get(pattern_id, 0) for pattern_id in pattern_ids))
            for tenant_id, pattern_ids in tenant_patterns.items()
        }
        private_unique_vectors = int(sum(pattern_weights.values()))
        if private_unique_vectors <= 0:
            self._last_private_metadata = {
                "enabled": False,
                "reason": "no_private_patterns",
                "target_cluster_count": int(target_cluster_count),
                "private_unique_vectors": 0,
            }
            return {int(tenant_id): 0 for tenant_id in tenant_ids}

        allowed_total_storage = int(
            math.floor(float(max(1, total_original_vectors)) * (1.0 + max(0.0, float(replication_budget_ratio))))
        )
        allowed_private_storage = max(
            int(private_unique_vectors),
            int(allowed_total_storage) - int(shared_vector_count),
        )

        def vector_count(pattern_ids: set[int] | frozenset[int]) -> int:
            return int(sum(pattern_weights.get(int(pattern_id), 0) for pattern_id in pattern_ids))

        def group_cost_for(tenant_values: set[int] | frozenset[int], partition_vectors: int) -> float:
            partition_vectors = int(partition_vectors)
            if partition_vectors <= 0:
                return 0.0
            cost = 0.0
            for tenant_id in tenant_values:
                tenant_id = int(tenant_id)
                accessible_vectors = int(tenant_private_vectors.get(tenant_id, 0))
                cost += self._partition_query_cost(
                    partition_vectors=partition_vectors,
                    accessible_vectors=accessible_vectors,
                    tenant_weight=float(tenant_query_weights.get(tenant_id, 0.0)),
                    total_vectors=int(total_original_vectors),
                    ef_search=int(ef_search),
                )
            return float(cost)

        def make_group(group_id: int, tenant_values: set[int], pattern_ids: set[int]) -> dict[str, object]:
            group_vectors = int(vector_count(pattern_ids))
            return {
                "group_id": int(group_id),
                "tenant_ids": set(int(tenant_id) for tenant_id in tenant_values),
                "pattern_ids": set(int(pattern_id) for pattern_id in pattern_ids),
                "vector_count": int(group_vectors),
                "cost": float(group_cost_for(set(int(tenant_id) for tenant_id in tenant_values), int(group_vectors))),
                "version": 0,
            }

        def shared_weight(left: dict[str, object], right: dict[str, object]) -> int:
            left_patterns = left["pattern_ids"]  # type: ignore[assignment]
            right_patterns = right["pattern_ids"]  # type: ignore[assignment]
            if len(left_patterns) > len(right_patterns):
                left_patterns, right_patterns = right_patterns, left_patterns
            return int(sum(pattern_weights.get(int(pattern_id), 0) for pattern_id in left_patterns if pattern_id in right_patterns))

        def merged_group(next_group_id: int, left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
            left_tenants = left["tenant_ids"]  # type: ignore[assignment]
            right_tenants = right["tenant_ids"]  # type: ignore[assignment]
            left_patterns = left["pattern_ids"]  # type: ignore[assignment]
            right_patterns = right["pattern_ids"]  # type: ignore[assignment]
            return make_group(
                int(next_group_id),
                set(left_tenants) | set(right_tenants),
                set(left_patterns) | set(right_patterns),
            )

        groups: dict[int, dict[str, object]] = {}
        initial_group_id_by_tenant: dict[int, int] = {}
        for index, tenant_id in enumerate(tenant_ids):
            group_id = int(index)
            tenant_id = int(tenant_id)
            initial_group_id_by_tenant[tenant_id] = group_id
            groups[group_id] = make_group(group_id, {tenant_id}, set(tenant_patterns.get(tenant_id, set())))
        next_group_id = int(len(groups))
        private_current_storage = int(sum(int(group["vector_count"]) for group in groups.values()))
        total_current_storage = int(shared_vector_count) + int(private_current_storage)
        initial_private_storage = int(private_current_storage)
        initial_total_storage = int(total_current_storage)
        initial_cost = float(sum(float(group["cost"]) for group in groups.values()))
        current_cost = float(initial_cost)

        def should_continue() -> bool:
            return bool(
                len(groups) > int(target_cluster_count)
                or int(private_current_storage) > int(allowed_private_storage)
            )

        candidate_heap: list[tuple[float, float, int, int, int, int, int, int]] = []
        candidate_evaluations = 0
        stale_candidates = 0
        fallback_merge_count = 0
        positive_merge_count = 0
        zero_benefit_merge_count = 0
        total_storage_reduction = 0
        total_cost_increase = 0.0

        def push_candidate(left_id: int, right_id: int, precomputed_benefit: Optional[int] = None) -> None:
            nonlocal candidate_evaluations
            left_id = int(left_id)
            right_id = int(right_id)
            if left_id == right_id or left_id not in groups or right_id not in groups:
                return
            if left_id > right_id:
                left_id, right_id = right_id, left_id
            left = groups[left_id]
            right = groups[right_id]
            benefit = int(precomputed_benefit) if precomputed_benefit is not None else int(shared_weight(left, right))
            if benefit <= 0:
                return
            merged_vectors = int(left["vector_count"]) + int(right["vector_count"]) - int(benefit)
            left_tenants = left["tenant_ids"]  # type: ignore[assignment]
            right_tenants = right["tenant_ids"]  # type: ignore[assignment]
            merged_tenants = set(left_tenants) | set(right_tenants)
            merged_cost_value = float(group_cost_for(merged_tenants, int(merged_vectors)))
            cost_increase = float(merged_cost_value - float(left["cost"]) - float(right["cost"]))
            unit_cost = float(cost_increase / float(max(1, benefit)))
            candidate_evaluations += 1
            heapq.heappush(
                candidate_heap,
                (
                    float(unit_cost),
                    float(cost_increase),
                    -int(benefit),
                    int(merged_vectors),
                    int(left_id),
                    int(right_id),
                    int(left["version"]),
                    int(right["version"]),
                ),
            )

        initial_pair_weights: dict[int, Counter] = defaultdict(Counter)
        for pattern in patterns:
            weight = int(pattern_weights.get(int(pattern.pattern_id), 0))
            if weight <= 0:
                continue
            active_tenants = sorted(int(tenant_id) for tenant_id in pattern.tenant_ids if int(tenant_id) in initial_group_id_by_tenant)
            for index, left_tenant in enumerate(active_tenants):
                left_group_id = int(initial_group_id_by_tenant[int(left_tenant)])
                for right_tenant in active_tenants[index + 1:]:
                    right_group_id = int(initial_group_id_by_tenant[int(right_tenant)])
                    initial_pair_weights[left_group_id][right_group_id] += int(weight)
                    initial_pair_weights[right_group_id][left_group_id] += int(weight)

        initial_candidate_edges: set[tuple[int, int]] = set()
        for left_id, neighbor_weights in initial_pair_weights.items():
            for right_id, _weight in neighbor_weights.most_common(int(_PRIVATE_MERGE_NEIGHBOR_LIMIT)):
                edge = (int(left_id), int(right_id)) if int(left_id) < int(right_id) else (int(right_id), int(left_id))
                initial_candidate_edges.add(edge)
        for left_id, right_id in sorted(initial_candidate_edges):
            benefit = int(initial_pair_weights.get(int(left_id), {}).get(int(right_id), 0))
            push_candidate(int(left_id), int(right_id), precomputed_benefit=int(benefit))

        def pop_valid_candidate() -> tuple[int, int, int, float] | None:
            nonlocal stale_candidates
            while candidate_heap:
                unit_cost, cost_increase, negative_benefit, _merged_vectors, left_id, right_id, left_version, right_version = heapq.heappop(candidate_heap)
                if left_id not in groups or right_id not in groups:
                    stale_candidates += 1
                    continue
                if int(groups[left_id]["version"]) != int(left_version) or int(groups[right_id]["version"]) != int(right_version):
                    stale_candidates += 1
                    continue
                benefit = int(-negative_benefit)
                if benefit <= 0:
                    stale_candidates += 1
                    continue
                return int(left_id), int(right_id), int(benefit), float(cost_increase)
            return None

        def fallback_candidate() -> tuple[int, int, int, float] | None:
            active = sorted(
                groups,
                key=lambda group_id: (
                    int(groups[int(group_id)]["vector_count"]),
                    int(len(groups[int(group_id)]["tenant_ids"])),
                    int(group_id),
                ),
            )[: max(2, min(int(_PRIVATE_FALLBACK_POOL_LIMIT), len(groups)))]
            best = None
            for index, left_id in enumerate(active):
                for right_id in active[index + 1:]:
                    left = groups[int(left_id)]
                    right = groups[int(right_id)]
                    benefit = int(shared_weight(left, right))
                    merged_vectors = int(left["vector_count"]) + int(right["vector_count"]) - int(benefit)
                    left_tenants = left["tenant_ids"]  # type: ignore[assignment]
                    right_tenants = right["tenant_ids"]  # type: ignore[assignment]
                    merged_tenants = set(left_tenants) | set(right_tenants)
                    merged_cost_value = float(group_cost_for(merged_tenants, int(merged_vectors)))
                    cost_increase = float(merged_cost_value - float(left["cost"]) - float(right["cost"]))
                    rank = (
                        float(cost_increase),
                        int(merged_vectors),
                        int(len(merged_tenants)),
                        int(left_id),
                        int(right_id),
                    )
                    if best is None or rank < best[0]:
                        best = (rank, int(left_id), int(right_id), int(benefit), float(cost_increase))
            if best is None:
                return None
            _rank, left_id, right_id, benefit, cost_increase = best
            return int(left_id), int(right_id), int(benefit), float(cost_increase)

        progress = tqdm(
            total=max(0, tenant_count - target_cluster_count),
            desc="Private cost merge",
            unit="merge",
            leave=False,
            disable=not show_progress,
        )
        merge_count = 0
        while should_continue() and len(groups) > 1:
            candidate = pop_valid_candidate()
            from_fallback = False
            if candidate is None:
                candidate = fallback_candidate()
                from_fallback = True
            if candidate is None:
                break

            left_id, right_id, benefit, cost_increase = candidate
            if left_id not in groups or right_id not in groups:
                continue
            left = groups.pop(int(left_id))
            right = groups.pop(int(right_id))
            new_group = merged_group(int(next_group_id), left, right)
            new_group_id = int(next_group_id)
            next_group_id += 1
            groups[new_group_id] = new_group

            storage_reduction = int(int(left["vector_count"]) + int(right["vector_count"]) - int(new_group["vector_count"]))
            private_current_storage -= int(storage_reduction)
            total_current_storage -= int(storage_reduction)
            actual_cost_increase = float(float(new_group["cost"]) - float(left["cost"]) - float(right["cost"]))
            current_cost += float(actual_cost_increase)
            total_cost_increase += float(actual_cost_increase)
            total_storage_reduction += int(storage_reduction)
            merge_count += 1
            if int(benefit) > 0:
                positive_merge_count += 1
            else:
                zero_benefit_merge_count += 1
            if from_fallback:
                fallback_merge_count += 1

            for other_id in list(groups):
                if int(other_id) != int(new_group_id):
                    push_candidate(int(new_group_id), int(other_id))

            if progress.n < progress.total:
                progress.update(1)
            if show_progress:
                progress.set_postfix(
                    {
                        "groups": int(len(groups)),
                        "private_storage": int(private_current_storage),
                        "need": int(allowed_private_storage),
                        "saved": int(storage_reduction),
                    }
                )
        progress.close()

        compact_ids = {group_id: index for index, group_id in enumerate(sorted(groups))}
        tenant_to_cluster: dict[int, int] = {}
        for group_id in sorted(groups):
            compact_id = int(compact_ids[int(group_id)])
            for tenant_id in groups[int(group_id)]["tenant_ids"]:
                tenant_to_cluster[int(tenant_id)] = compact_id

        cluster_sizes = [int(len(group["tenant_ids"])) for group in groups.values()]
        cluster_vector_counts = [int(group["vector_count"]) for group in groups.values()]
        weighted_filter = 0.0
        for group in groups.values():
            group_size = int(group["vector_count"])
            for tenant_id in group["tenant_ids"]:
                tenant_size = max(1, int(tenant_private_vectors.get(int(tenant_id), 0)))
                if int(tenant_private_vectors.get(int(tenant_id), 0)) <= 0:
                    continue
                weighted_filter += float(tenant_query_weights.get(int(tenant_id), 0.0)) * (float(group_size) / float(tenant_size))

        self._last_private_metadata = {
            "enabled": True,
            "objective": "bottom-up tenant merge by ACL copy saving under private storage budget",
            "target_cluster_count": int(target_cluster_count),
            "replication_budget_ratio": float(max(0.0, float(replication_budget_ratio))),
            "allowed_total_storage": int(allowed_total_storage),
            "allowed_private_storage": int(allowed_private_storage),
            "initial_total_storage": int(initial_total_storage),
            "final_total_storage": int(total_current_storage),
            "private_unique_vectors": int(private_unique_vectors),
            "initial_private_storage": int(initial_private_storage),
            "final_private_storage": int(private_current_storage),
            "private_replication_factor": float(private_current_storage / max(1, private_unique_vectors)),
            "initial_group_count": int(tenant_count),
            "final_group_count": int(len(groups)),
            "merge_count": int(merge_count),
            "positive_merge_count": int(positive_merge_count),
            "zero_benefit_merge_count": int(zero_benefit_merge_count),
            "fallback_merge_count": int(fallback_merge_count),
            "stale_candidates": int(stale_candidates),
            "candidate_evaluations": int(candidate_evaluations),
            "cost_initial": float(initial_cost),
            "cost_final": float(current_cost),
            "total_cost_increase": float(total_cost_increase),
            "total_storage_reduction": int(total_storage_reduction),
            "candidate_rule": "initial top co-access neighbors from ACL graph, exact merged-group overlap refresh, fallback only when no positive overlap remains",
            "merge_score_rule": "minimize query_cost_increase / storage_saved; stop when private storage budget and cluster target are both satisfied",
            "weighted_filter_ratio": float(weighted_filter),
            "min_cluster_size": int(min(cluster_sizes)) if cluster_sizes else 0,
            "max_cluster_size": int(max(cluster_sizes)) if cluster_sizes else 0,
            "min_cluster_vectors": int(min(cluster_vector_counts)) if cluster_vector_counts else 0,
            "max_cluster_vectors": int(max(cluster_vector_counts)) if cluster_vector_counts else 0,
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

        private_by_cluster: dict[int, dict[int, ACLPattern]] = defaultdict(dict)
        for pattern in tqdm(private_patterns, desc="Private cluster copies", unit="acl", leave=False, disable=not show_progress):
            owning_clusters = {
                int(tenant_to_private_cluster[int(tenant_id)])
                for tenant_id in pattern.tenant_ids
                if int(tenant_id) in tenant_to_private_cluster
            }
            for cluster_id in owning_clusters:
                private_by_cluster[int(cluster_id)][int(pattern.pattern_id)] = pattern
        private_partition_index = 0
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
    ) -> KMeansPartition:
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
                routes.append(
                    TenantRoute(
                        tenant_id=int(tenant_id),
                        partition_id=str(partition.partition_id),
                        table_name=str(partition.table_name),
                        route_kind=str(partition.partition_kind),
                        cluster_id=int(partition.cluster_id),
                        pattern_ids=tuple(sorted(set(int(pattern_id) for pattern_id in pattern_ids))),
                    )
                )
        return routes
