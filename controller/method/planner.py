from __future__ import annotations

from collections import Counter, defaultdict
import math
import hashlib
from typing import Callable, Optional

import numpy as np

from .common import (
    ACLLogicalPattern,
    DocumentAccessRecord,
    PrefixDagNode,
    WorkloadAwarePartition,
    WorkloadAwarePlan,
    WorkloadQuery,
    _normalize_rows,
    _normalize_vector,
    _parse_vector,
    _weighted_jaccard_from_dicts,
    _weighted_jaccard_from_sets,
    get_access_overlay_group_table_name,
    get_access_signature_cover_table_name,
    get_access_overlay_table_name,
    get_overlay_table_name,
    get_partition_table_name,
)


def _emit_progress(progress_fn: Optional[Callable[[str], None]], message: str) -> None:
    if progress_fn is not None:
        progress_fn(str(message))


class WorkloadAwarePlanner:
    def __init__(self, *, random_state: int = 42) -> None:
        self.random_state = int(random_state)

    def build_plan(
        self,
        records: list[DocumentAccessRecord],
        *,
        document_block_counts: dict[int, int],
        queries: list[WorkloadQuery],
        tenant_query_weights: dict[int, float],
        min_pattern_support: int = 16,
        min_pattern_query_mass: float = 0.0,
        safe_density_threshold: float = 0.35,
        supplemental_edge_penalty: float = 0.25,
        supplemental_edge_gain_threshold: float = 0.0,
        target_partition_count: Optional[int] = None,
        max_partition_vector_count: Optional[int] = None,
        overlay_space_ratio: float = 0.25,
        protection_overlay_space_ratio: Optional[float] = None,
        progress_fn: Optional[Callable[[str], None]] = None,
    ) -> WorkloadAwarePlan:
        if not records:
            raise ValueError("Cannot build ACL planning-tree partitions without document records")

        _emit_progress(progress_fn, "[plan][planner] normalizing representative vectors...")
        normalized_vectors = _normalize_rows(np.vstack([record.vector for record in records]).astype(np.float32))

        _emit_progress(progress_fn, "[plan][planner] ordering tenants for ACL normalization...")
        tenant_order, tenant_weights, tenant_doc_counts, tenant_scores = self._build_tenant_order(
            records,
            tenant_query_weights,
        )   ## 可以多看下这个代码：租户排序，目前在频繁优化这点

        _emit_progress(progress_fn, "[plan][planner] collecting exact ACL patterns...")
        logical_patterns = self._build_logical_patterns(
            records,
            normalized_vectors=normalized_vectors,
            document_block_counts=document_block_counts,
            tenant_order=tenant_order,
            tenant_query_weights=tenant_weights,
        )
        _emit_progress(progress_fn, f"[plan][planner] produced {len(logical_patterns)} exact ACL patterns")

        effective_target_partition_count = self._resolve_target_partition_count(
            logical_patterns,
            target_partition_count=target_partition_count,
        )
        _emit_progress(
            progress_fn,
            f"[plan][planner] resolved target physical partitions={effective_target_partition_count}",
        )

        _emit_progress(progress_fn, "[plan][planner] building compact ACL route DAG...")
        dag_nodes, dag_metadata = self._build_compact_route_dag(
            logical_patterns,
            tenant_order=tenant_order,
        )
        _emit_progress(progress_fn, f"[plan][planner] built DAG with {len(dag_nodes)} nodes")

        _emit_progress(progress_fn, "[plan][planner] building ACL containment planning tree...")
        planning_tree, tree_metadata = self._build_acl_planning_tree(logical_patterns)
        _emit_progress(
            progress_fn,
            f"[plan][planner] built planning tree with {int(tree_metadata.get('planning_node_count', 0))} nodes",
        )

        _emit_progress(progress_fn, "[plan][planner] logically pruning planning tree...")
        pruned_tree, prune_metadata = self._prune_acl_tree(
            planning_tree,
            logical_patterns=logical_patterns,
        )
        _emit_progress(
            progress_fn,
            f"[plan][planner] pruned planning tree to {int(prune_metadata.get('pruned_planning_node_count', 0))} active nodes",
        )

        _emit_progress(progress_fn, "[plan][planner] solving K-cut DP for partition placement...")
        partitions, partition_metadata = self._dp_cut_acl_tree(
            logical_patterns=logical_patterns,
            pruned_tree=pruned_tree,
            tenant_weights=tenant_weights,
            queries=queries,
            target_partition_count=effective_target_partition_count,
            max_partition_vector_count=max_partition_vector_count,
        )
        _emit_progress(progress_fn, f"[plan][planner] merged into {len(partitions)} physical partitions")

        total_vectors = max(sum(partition.vector_count for partition in partitions), 1)
        for partition in partitions:
            partition.metadata["route_prior"] = float(partition.vector_count / total_vectors)

        _emit_progress(progress_fn, "[plan][planner] selecting tenant overlay fast paths...")
        effective_protection_overlay_space_ratio = (
            float(overlay_space_ratio)
            if protection_overlay_space_ratio is None
            else float(protection_overlay_space_ratio)
        )
        tenant_overlays, overlay_metadata = self._select_tenant_overlays(
            logical_patterns=logical_patterns,
            partitions=partitions,
            queries=queries,
            tenant_query_weights=tenant_weights,
            tenant_order=tenant_order,
            overlay_space_ratio=effective_protection_overlay_space_ratio,
        )
        _emit_progress(
            progress_fn,
            f"[plan][planner] selected {len(tenant_overlays)} tenant overlays and "
            f"{int(overlay_metadata.get('overlay_selected_access_count', 0))} access overlays",
        )

        for pattern in logical_patterns:
            pattern.metadata.pop("_member_document_vectors", None)

        # 返回分区安排
        return WorkloadAwarePlan(
            partitions=partitions,
            logical_patterns=logical_patterns,
            dag_nodes=dag_nodes,
            tenant_order=tenant_order,
            metadata={
                "document_count": len(records),
                "logical_pattern_count": len(logical_patterns),
                "dag_node_count": len(dag_nodes),
                "partition_count": len(partitions),
                "tenant_count": len(tenant_order),
                "query_count": len(queries),
                "safe_density_threshold": float(safe_density_threshold),
                "min_pattern_support": int(min_pattern_support),
                "min_pattern_query_mass": float(min_pattern_query_mass),
                "target_partition_count": int(effective_target_partition_count),
                "overlay_space_ratio": float(overlay_space_ratio),
                "protection_overlay_space_ratio": float(effective_protection_overlay_space_ratio),
                "tenant_overlays": tenant_overlays,
                "tenant_doc_counts": {str(k): int(v) for k, v in sorted(tenant_doc_counts.items())},
                "tenant_scores": {str(k): float(v) for k, v in sorted(tenant_scores.items())},
                **dag_metadata,
                **tree_metadata,
                **prune_metadata,
                **partition_metadata,
                **overlay_metadata,
            },
        )

    def _select_tenant_overlays(
        self,
        *,
        logical_patterns: list[ACLLogicalPattern],
        partitions: list[WorkloadAwarePartition],
        queries: list[WorkloadQuery],
        tenant_query_weights: dict[int, float],
        tenant_order: tuple[int, ...],
        overlay_space_ratio: float,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        total_vector_count = max(sum(int(pattern.vector_count) for pattern in logical_patterns), 1)
        budget_vectors = max(0, int(math.floor(total_vector_count * max(0.0, float(overlay_space_ratio)))))
        if budget_vectors <= 0:
            return [], {
                "overlay_budget_vectors": 0,
                "overlay_selected_vectors": 0,
                "overlay_selected_tenant_count": 0,
                "overlay_selected_access_count": 0,
                "access_signature_cover_count": 0,
            }

        return self._select_shared_protection_overlays(
            logical_patterns=logical_patterns,
            partitions=partitions,
            queries=queries,
            tenant_query_weights=tenant_query_weights,
            tenant_order=tenant_order,
            budget_vectors=budget_vectors,
            total_vector_count=total_vector_count,
        )

        query_mass_by_tenant: dict[int, float] = Counter()
        for query in queries:
            query_mass_by_tenant[int(query.tenant_id)] += float(query.weight)
        if not query_mass_by_tenant:
            query_mass_by_tenant.update({int(tenant_id): float(weight) for tenant_id, weight in tenant_query_weights.items()})

        patterns_by_tenant: dict[int, list[ACLLogicalPattern]] = defaultdict(list)
        for pattern in logical_patterns:
            for tenant_id in pattern.tenant_ids:
                patterns_by_tenant[int(tenant_id)].append(pattern)

        partition_by_pattern: dict[int, WorkloadAwarePartition] = {
            int(pattern_id): partition
            for partition in partitions
            for pattern_id in partition.logical_pattern_ids
        }

        total_query_mass = float(sum(float(value) for value in query_mass_by_tenant.values()))

        candidates: list[tuple[float, float, int, int, dict[str, object]]] = []
        access_candidates: list[tuple[float, float, int, int, str, dict[str, object]]] = []
        for tenant_id, tenant_patterns in patterns_by_tenant.items():
            query_mass = float(query_mass_by_tenant.get(int(tenant_id), 0.0))
            if query_mass <= 0.0:
                continue

            pattern_ids = tuple(sorted(int(pattern.pattern_id) for pattern in tenant_patterns))
            document_ids = tuple(
                sorted({int(document_id) for pattern in tenant_patterns for document_id in pattern.document_ids})
            )
            vector_count = int(sum(int(pattern.vector_count) for pattern in tenant_patterns))
            if vector_count <= 0 or vector_count > budget_vectors:
                continue

            matched_partitions = {
                str(partition_by_pattern[int(pattern_id)].partition_id): partition_by_pattern[int(pattern_id)]
                for pattern_id in pattern_ids
                if int(pattern_id) in partition_by_pattern
            }
            if not matched_partitions:
                continue

            patterns_by_partition: dict[str, list[ACLLogicalPattern]] = defaultdict(list)
            for pattern in tenant_patterns:
                partition = partition_by_pattern.get(int(pattern.pattern_id))
                if partition is not None:
                    patterns_by_partition[str(partition.partition_id)].append(pattern)

            fixed_index_cost = max(1, int(math.sqrt(float(total_vector_count))))
            tenant_access_items: list[dict[str, object]] = []
            for partition_id, partition_patterns in patterns_by_partition.items():
                partition = matched_partitions.get(str(partition_id))
                if partition is None:
                    continue
                access_pattern_ids = tuple(sorted(int(pattern.pattern_id) for pattern in partition_patterns))
                access_document_ids = tuple(
                    sorted({int(document_id) for pattern in partition_patterns for document_id in pattern.document_ids})
                )
                access_vector_count = int(sum(int(pattern.vector_count) for pattern in partition_patterns))
                if access_vector_count <= 0:
                    continue
                partition_vector_count = max(1, int(partition.vector_count))
                selectivity = float(access_vector_count) / float(partition_vector_count)
                access_saved_cost = max(0.0, math.log1p(partition_vector_count) - math.log1p(access_vector_count))
                if access_saved_cost <= 0.0:
                    continue
                access_document_pattern_pairs = tuple(
                    sorted(
                        (int(document_id), int(pattern.pattern_id))
                        for pattern in partition_patterns
                        for document_id in pattern.document_ids
                    )
                )
                access_benefit = float(query_mass * access_saved_cost)
                tenant_access_items.append(
                    {
                        "partition_id": str(partition_id),
                        "pattern_ids": tuple(int(pattern_id) for pattern_id in access_pattern_ids),
                        "document_ids": tuple(int(document_id) for document_id in access_document_ids),
                        "document_pattern_pairs": tuple(
                            (int(document_id), int(pattern_id))
                            for document_id, pattern_id in access_document_pattern_pairs
                        ),
                        "document_count": int(len(access_document_ids)),
                        "vector_count": int(access_vector_count),
                        "partition_vector_count": int(partition_vector_count),
                        "selectivity": float(selectivity),
                        "base_route_cost": float(math.log1p(partition_vector_count)),
                        "standalone_saved_cost": float(access_saved_cost),
                        "standalone_benefit": float(access_benefit),
                        "standalone_density": float(access_benefit / max(1, access_vector_count)),
                    }
                )

            if tenant_access_items:
                tenant_access_items.sort(
                    key=lambda item: (
                        -float(item["standalone_density"]),
                        -float(item["standalone_benefit"]),
                        str(item["partition_id"]),
                    )
                )
                best_group_spec: Optional[dict[str, object]] = None
                best_group_score: tuple[float, float, float, int, str] | None = None
                group_pattern_ids: set[int] = set()
                group_document_ids: set[int] = set()
                group_document_pattern_pairs: set[tuple[int, int]] = set()
                group_partition_ids: list[str] = []
                group_access_vector_count = 0
                group_partition_vector_count = 0
                group_base_route_cost = 0.0
                group_selectivity_weighted_sum = 0.0
                for item in tenant_access_items:
                    group_partition_ids.append(str(item["partition_id"]))
                    group_pattern_ids.update(int(pattern_id) for pattern_id in item["pattern_ids"])
                    group_document_ids.update(int(document_id) for document_id in item["document_ids"])
                    group_document_pattern_pairs.update(
                        (int(document_id), int(pattern_id))
                        for document_id, pattern_id in item["document_pattern_pairs"]
                    )
                    group_access_vector_count += int(item["vector_count"])
                    group_partition_vector_count += int(item["partition_vector_count"])
                    group_base_route_cost += float(item["base_route_cost"])
                    group_selectivity_weighted_sum += float(item["selectivity"]) * float(item["vector_count"])
                    if group_access_vector_count <= 0:
                        continue
                    materialization_cost = int(group_access_vector_count + fixed_index_cost)
                    if materialization_cost > budget_vectors:
                        continue
                    grouped_saved_cost = max(
                        0.0,
                        float(group_base_route_cost) - math.log1p(float(group_access_vector_count)),
                    )
                    if grouped_saved_cost <= 0.0:
                        continue
                    grouped_benefit = float(query_mass * grouped_saved_cost)
                    grouped_density = float(grouped_benefit / max(1, materialization_cost))
                    partition_count = max(1, len(group_partition_ids))
                    workload_coverage_score = float(
                        query_mass * math.log1p(float(partition_count)) / math.sqrt(max(1, materialization_cost))
                    )
                    score = (
                        float(workload_coverage_score),
                        float(grouped_density),
                        float(grouped_benefit),
                        int(partition_count),
                        ",".join(group_partition_ids),
                    )
                    if best_group_score is not None and score <= best_group_score:
                        continue
                    weighted_selectivity = (
                        float(group_selectivity_weighted_sum / float(group_access_vector_count))
                        if group_access_vector_count > 0
                        else 1.0
                    )
                    if partition_count == 1:
                        partition_id = str(group_partition_ids[0])
                        table_name = get_access_overlay_table_name(int(tenant_id), partition_id)
                    else:
                        partition_id = "__route__"
                        table_name = get_access_overlay_group_table_name(int(tenant_id))
                    best_group_spec = {
                        "tenant_id": int(tenant_id),
                        "partition_id": str(partition_id),
                        "partition_ids": [str(value) for value in group_partition_ids],
                        "table_name": table_name,
                        "pattern_ids": [int(pattern_id) for pattern_id in sorted(group_pattern_ids)],
                        "document_ids": [int(document_id) for document_id in sorted(group_document_ids)],
                        "document_pattern_pairs": [
                            [int(document_id), int(pattern_id)]
                            for document_id, pattern_id in sorted(group_document_pattern_pairs)
                        ],
                        "document_count": int(len(group_document_ids)),
                        "vector_count": int(group_access_vector_count),
                        "materialization_cost": int(materialization_cost),
                        "partition_vector_count": int(group_partition_vector_count),
                        "covered_partition_count": int(partition_count),
                        "selectivity": float(weighted_selectivity),
                        "query_mass": float(query_mass),
                        "query_share": float(query_mass / max(total_query_mass, 1e-9)),
                        "estimated_saved_cost": float(grouped_saved_cost),
                        "benefit_density": float(grouped_density),
                        "workload_coverage_score": float(workload_coverage_score),
                    }
                    best_group_score = score

                if best_group_spec is not None and best_group_score is not None:
                    access_candidates.append(
                        (
                            float(best_group_score[0]),
                            float(best_group_score[2]),
                            -int(best_group_spec["materialization_cost"]),
                            int(tenant_id),
                            str(best_group_spec["partition_id"]),
                            best_group_spec,
                        )
                    )

            document_pattern_pairs = tuple(
                sorted(
                    (int(document_id), int(pattern.pattern_id))
                    for pattern in tenant_patterns
                    for document_id in pattern.document_ids
                )
            )
            partition_route_cost = float(
                sum(math.log1p(max(1, int(partition.vector_count))) for partition in matched_partitions.values())
            )
            covered_partition_vector_count = int(
                sum(int(partition.vector_count) for partition in matched_partitions.values())
            )
            overlay_cost = float(math.log1p(max(1, vector_count)))
            saved_cost = max(0.0, partition_route_cost - overlay_cost)
            if saved_cost <= 0.0:
                continue

            benefit = float(query_mass * saved_cost)
            density = float(benefit / max(1, vector_count))
            workload_coverage_score = float(
                query_mass * math.log1p(float(len(matched_partitions))) / math.sqrt(max(1, vector_count))
            )
            query_share = float(query_mass / max(total_query_mass, 1e-9))
            overlay_spec = {
                "tenant_id": int(tenant_id),
                "table_name": get_overlay_table_name(int(tenant_id)),
                "pattern_ids": [int(pattern_id) for pattern_id in pattern_ids],
                "document_ids": [int(document_id) for document_id in document_ids],
                "document_pattern_pairs": [
                    [int(document_id), int(pattern_id)]
                    for document_id, pattern_id in document_pattern_pairs
                ],
                "document_count": int(len(document_ids)),
                "vector_count": int(vector_count),
                "query_mass": float(query_mass),
                "query_share": float(query_share),
                "route_partition_count": int(len(matched_partitions)),
                "covered_partition_vector_count": int(covered_partition_vector_count),
                "estimated_saved_cost": float(saved_cost),
                "benefit_density": float(density),
                "workload_coverage_score": float(workload_coverage_score),
            }
            candidates.append((float(workload_coverage_score), float(benefit), -int(vector_count), int(tenant_id), overlay_spec))

        selected: list[dict[str, object]] = []
        selected_access_overlays: list[dict[str, object]] = []
        used_vectors = 0
        combined_candidates: list[tuple[float, float, int, str, str, dict[str, object]]] = []
        combined_candidates.extend(
            (
                density,
                benefit,
                -neg_cost,
                "tenant",
                f"tenant:{int(overlay_spec['tenant_id'])}",
                overlay_spec,
            )
            for density, benefit, neg_cost, _, overlay_spec in candidates
        )
        combined_candidates.extend(
            (
                density,
                benefit,
                -neg_cost,
                "access",
                f"access:{int(overlay_spec['tenant_id'])}:{str(overlay_spec['partition_id'])}",
                overlay_spec,
            )
            for density, benefit, neg_cost, _, _, overlay_spec in access_candidates
        )
        covered_access_keys: set[tuple[int, str]] = set()
        covered_tenants: set[int] = set()
        access_covered_tenants: set[int] = set()
        for _, _, materialization_cost, overlay_type, _, overlay_spec in sorted(
            combined_candidates,
            key=lambda item: (-float(item[0]), -float(item[1]), int(item[2]), str(item[3]), str(item[4])),
        ):
            tenant_id = int(overlay_spec["tenant_id"])
            if overlay_type == "tenant":
                if tenant_id in covered_tenants or tenant_id in access_covered_tenants:
                    continue
                vector_count = int(overlay_spec["vector_count"])
                if used_vectors + vector_count > budget_vectors:
                    continue
                selected.append(overlay_spec)
                used_vectors += vector_count
                covered_tenants.add(tenant_id)
                continue

            partition_id = str(overlay_spec["partition_id"])
            access_key = (tenant_id, partition_id)
            if tenant_id in covered_tenants or tenant_id in access_covered_tenants or access_key in covered_access_keys:
                continue
            if used_vectors + int(materialization_cost) > budget_vectors:
                continue
            selected_access_overlays.append(overlay_spec)
            used_vectors += int(materialization_cost)
            covered_access_keys.add(access_key)
            access_covered_tenants.add(tenant_id)

        selected.sort(key=lambda item: (-float(item.get("query_mass", 0.0)), int(item.get("tenant_id", 0))))
        selected_access_overlays.sort(
            key=lambda item: (
                -float(item.get("benefit_density", 0.0)),
                int(item.get("tenant_id", 0)),
                str(item.get("partition_id", "")),
            )
        )
        return selected, {
            "access_overlays": selected_access_overlays,
            "overlay_budget_vectors": int(budget_vectors),
            "overlay_selected_vectors": int(used_vectors),
            "overlay_selected_tenant_count": int(len(selected)),
            "overlay_selected_access_count": int(len(selected_access_overlays)),
            "overlay_candidate_tenant_count": int(len(candidates)),
            "overlay_candidate_access_count": int(len(access_candidates)),
        }

    def _select_access_signature_covers(
        self,
        *,
        logical_patterns: list[ACLLogicalPattern],
        partitions: list[WorkloadAwarePartition],
        queries: list[WorkloadQuery],
        tenant_query_weights: dict[int, float],
        budget_vectors: int,
        total_vector_count: int,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        query_mass_by_tenant: dict[int, float] = Counter()
        for query in queries:
            query_mass_by_tenant[int(query.tenant_id)] += float(query.weight)
        has_workload_query_mass = bool(query_mass_by_tenant)
        if not query_mass_by_tenant:
            query_mass_by_tenant.update(
                {int(tenant_id): float(weight) for tenant_id, weight in tenant_query_weights.items()}
            )

        patterns_by_tenant: dict[int, list[ACLLogicalPattern]] = defaultdict(list)
        for pattern in logical_patterns:
            for tenant_id in pattern.tenant_ids:
                patterns_by_tenant[int(tenant_id)].append(pattern)

        partitions_by_pattern: dict[int, list[WorkloadAwarePartition]] = defaultdict(list)
        for partition in partitions:
            for pattern_id in partition.logical_pattern_ids:
                partitions_by_pattern[int(pattern_id)].append(partition)

        def _matched_partitions_for_patterns(pattern_ids: tuple[int, ...]) -> dict[str, WorkloadAwarePartition]:
            matched: dict[str, WorkloadAwarePartition] = {}
            for pattern_id in pattern_ids:
                for partition in partitions_by_pattern.get(int(pattern_id), ()):
                    matched[str(partition.partition_id)] = partition
            return matched

        pattern_vector_counts = {
            int(pattern.pattern_id): int(pattern.vector_count)
            for pattern in logical_patterns
        }

        partition_count = max(1, len(partitions))
        average_partition_vector_count = float(max(1, int(total_vector_count)) / float(partition_count))
        effective_budget_vectors = min(
            int(budget_vectors),
            max(1, int(2.0 * average_partition_vector_count)),
        )
        total_query_mass = float(sum(float(value) for value in query_mass_by_tenant.values()))
        signature_candidates: dict[tuple[int, ...], dict[str, object]] = {}
        for tenant_id, tenant_patterns in patterns_by_tenant.items():
            query_mass = float(query_mass_by_tenant.get(int(tenant_id), 0.0))
            if query_mass <= 0.0:
                continue

            pattern_ids = tuple(sorted(int(pattern.pattern_id) for pattern in tenant_patterns))
            if not pattern_ids:
                continue
            matched_partitions = _matched_partitions_for_patterns(pattern_ids)
            if not matched_partitions:
                continue

            document_ids = tuple(
                sorted({int(document_id) for pattern in tenant_patterns for document_id in pattern.document_ids})
            )
            vector_count = int(sum(int(pattern.vector_count) for pattern in tenant_patterns))
            materialization_cost = int(vector_count)
            covered_partition_count = max(1, len(matched_partitions))
            max_signature_vectors = int(
                math.ceil(2.0 * float(average_partition_vector_count))
            )
            if (
                vector_count <= 0
                or materialization_cost > int(effective_budget_vectors)
                or vector_count > max_signature_vectors
            ):
                continue

            route_cost = float(
                sum(math.log1p(max(1, int(partition.vector_count))) for partition in matched_partitions.values())
            )
            cover_cost = float(math.log1p(max(1, vector_count)))
            saved_cost = max(0.0, route_cost - cover_cost)
            if saved_cost <= 0.0:
                continue
            covered_partition_vector_count = int(
                sum(int(partition.vector_count) for partition in matched_partitions.values())
            )
            cover_selectivity = float(vector_count / max(1, covered_partition_vector_count))

            entry = signature_candidates.get(pattern_ids)
            if entry is None:
                signature_payload = ",".join(str(pattern_id) for pattern_id in pattern_ids).encode("utf-8")
                signature_id = hashlib.blake2b(signature_payload, digest_size=10).hexdigest()
                document_pattern_pairs = tuple(
                    sorted(
                        (int(document_id), int(pattern.pattern_id))
                        for pattern in tenant_patterns
                        for document_id in pattern.document_ids
                    )
                )
                partition_ids = tuple(sorted(str(partition_id) for partition_id in matched_partitions))
                entry = {
                    "signature_id": str(signature_id),
                    "tenant_ids": [],
                    "pattern_ids": [int(pattern_id) for pattern_id in pattern_ids],
                    "document_ids": [int(document_id) for document_id in document_ids],
                    "document_pattern_pairs": [
                        [int(document_id), int(pattern_id)]
                        for document_id, pattern_id in document_pattern_pairs
                    ],
                    "document_count": int(len(document_ids)),
                    "vector_count": int(vector_count),
                    "materialization_cost": int(materialization_cost),
                    "partition_ids": [str(partition_id) for partition_id in partition_ids],
                    "covered_partition_count": int(len(partition_ids)),
                    "route_partition_count": int(len(partition_ids)),
                    "covered_partition_vector_count": int(covered_partition_vector_count),
                    "cover_selectivity": float(cover_selectivity),
                    "estimated_saved_cost": float(saved_cost),
                    "query_mass": 0.0,
                }
                signature_candidates[pattern_ids] = entry
            entry["tenant_ids"].append(int(tenant_id))
            entry["query_mass"] = float(entry.get("query_mass", 0.0) or 0.0) + float(query_mass)

        ranked_candidates: list[tuple[float, float, int, tuple[int, ...], dict[str, object]]] = []
        for signature_key, entry in signature_candidates.items():
            query_mass = float(entry.get("query_mass", 0.0) or 0.0)
            materialization_cost = int(entry.get("materialization_cost", 0) or 0)
            if query_mass <= 0.0 or materialization_cost <= 0:
                continue
            covered_partition_count = max(1, int(entry.get("covered_partition_count", 1) or 1))
            cover_selectivity = min(1.0, max(0.0, float(entry.get("cover_selectivity", 1.0) or 1.0)))
            recall_risk = math.log1p(float(covered_partition_count)) * math.sqrt(max(0.0, 1.0 - cover_selectivity))
            benefit = float(query_mass * float(entry.get("estimated_saved_cost", 0.0) or 0.0) * max(recall_risk, 1e-9))
            density = float(benefit / max(1, materialization_cost))
            workload_coverage_score = float(
                benefit / math.sqrt(max(1, materialization_cost))
            )
            entry["benefit"] = float(benefit)
            entry["benefit_density"] = float(density)
            entry["recall_risk"] = float(recall_risk)
            entry["query_share"] = float(query_mass / max(total_query_mass, 1e-9))
            entry["workload_coverage_score"] = float(workload_coverage_score)
            ranked_candidates.append(
                (float(workload_coverage_score), float(benefit), int(materialization_cost), signature_key, entry)
            )

        selected_signatures: list[dict[str, object]] = []
        used_vectors = 0
        selected_pattern_ids: set[int] = set()
        for _, _, materialization_cost, signature_key, entry in sorted(
            ranked_candidates,
            key=lambda item: (-float(item[0]), -float(item[1]), int(item[2]), tuple(item[3])),
        ):
            signature_pattern_ids = set(int(pattern_id) for pattern_id in signature_key)
            new_vector_count = sum(
                int(pattern_vector_counts.get(int(pattern_id), 0))
                for pattern_id in signature_pattern_ids
                if int(pattern_id) not in selected_pattern_ids
            )
            covered_partition_count = max(1, int(entry.get("covered_partition_count", 1) or 1))
            if covered_partition_count > 1 and new_vector_count * covered_partition_count <= int(materialization_cost):
                continue
            if used_vectors + int(materialization_cost) > int(effective_budget_vectors):
                continue
            selected_signatures.append(entry)
            used_vectors += int(materialization_cost)
            selected_pattern_ids.update(signature_pattern_ids)

        access_overlays: list[dict[str, object]] = []
        for entry in selected_signatures:
            signature_id = str(entry["signature_id"])
            table_name = get_access_signature_cover_table_name(signature_id)
            tenant_ids = sorted(set(int(tenant_id) for tenant_id in (entry.get("tenant_ids", []) or [])))
            partition_ids = [str(partition_id) for partition_id in (entry.get("partition_ids", []) or [])]
            for tenant_id in tenant_ids:
                access_overlays.append(
                    {
                        "tenant_id": int(tenant_id),
                        "partition_id": "__cover__",
                        "partition_ids": [str(partition_id) for partition_id in partition_ids],
                        "table_name": table_name,
                        "signature_id": str(signature_id),
                        "cover_tenant_ids": [int(value) for value in tenant_ids],
                        "pattern_ids": [int(pattern_id) for pattern_id in (entry.get("pattern_ids", []) or [])],
                        "document_ids": [int(document_id) for document_id in (entry.get("document_ids", []) or [])],
                        "document_pattern_pairs": [
                            [int(document_id), int(pattern_id)]
                            for document_id, pattern_id in (entry.get("document_pattern_pairs", []) or [])
                        ],
                        "document_count": int(entry.get("document_count", 0) or 0),
                        "vector_count": int(entry.get("vector_count", 0) or 0),
                        "materialization_cost": int(entry.get("materialization_cost", 0) or 0),
                        "covered_partition_count": int(entry.get("covered_partition_count", len(partition_ids)) or len(partition_ids)),
                        "route_partition_count": int(entry.get("route_partition_count", len(partition_ids)) or len(partition_ids)),
                        "covered_partition_vector_count": int(entry.get("covered_partition_vector_count", 0) or 0),
                        "cover_selectivity": float(entry.get("cover_selectivity", 1.0) or 1.0),
                        "recall_risk": float(entry.get("recall_risk", 0.0) or 0.0),
                        "query_mass": float(query_mass_by_tenant.get(int(tenant_id), 0.0)),
                        "signature_query_mass": float(entry.get("query_mass", 0.0) or 0.0),
                        "query_share": float(entry.get("query_share", 0.0) or 0.0),
                        "estimated_saved_cost": float(entry.get("estimated_saved_cost", 0.0) or 0.0),
                        "benefit_density": float(entry.get("benefit_density", 0.0) or 0.0),
                        "workload_coverage_score": float(entry.get("workload_coverage_score", 0.0) or 0.0),
                        "overlay_type": "access_signature_cover",
                    }
                )

        access_overlays.sort(
            key=lambda item: (
                -float(item.get("workload_coverage_score", 0.0)),
                str(item.get("signature_id", "")),
                int(item.get("tenant_id", 0)),
            )
        )
        return [], {
            "access_overlays": access_overlays,
            "overlay_budget_vectors": int(budget_vectors),
            "access_signature_effective_budget_vectors": int(effective_budget_vectors),
            "overlay_selected_vectors": int(used_vectors),
            "overlay_selected_tenant_count": 0,
            "overlay_selected_access_count": int(len(access_overlays)),
            "overlay_candidate_tenant_count": 0,
            "overlay_candidate_access_count": int(len(ranked_candidates)),
            "access_signature_cover_count": int(len(selected_signatures)),
            "access_signature_cover_tenant_mapping_count": int(len(access_overlays)),
        }

    def _select_shared_protection_overlays(
        self,
        *,
        logical_patterns: list[ACLLogicalPattern],
        partitions: list[WorkloadAwarePartition],
        queries: list[WorkloadQuery],
        tenant_query_weights: dict[int, float],
        tenant_order: tuple[int, ...],
        budget_vectors: int,
        total_vector_count: int,
    ) -> tuple[list[dict[str, object]], dict[str, object]]:
        query_mass_by_tenant: dict[int, float] = Counter()
        for query in queries:
            query_mass_by_tenant[int(query.tenant_id)] += float(query.weight)
        has_workload_query_mass = bool(query_mass_by_tenant)
        if not query_mass_by_tenant:
            query_mass_by_tenant.update(
                {int(tenant_id): float(weight) for tenant_id, weight in tenant_query_weights.items()}
            )

        patterns_by_tenant: dict[int, set[int]] = defaultdict(set)
        documents_by_tenant: dict[int, set[int]] = defaultdict(set)
        tenant_vector_counts: dict[int, int] = Counter()
        pattern_by_id = {int(pattern.pattern_id): pattern for pattern in logical_patterns}
        pattern_vector_counts = {
            int(pattern.pattern_id): int(pattern.vector_count)
            for pattern in logical_patterns
        }
        for pattern in logical_patterns:
            pattern_id = int(pattern.pattern_id)
            for tenant_id in pattern.tenant_ids:
                tenant_id = int(tenant_id)
                patterns_by_tenant[tenant_id].add(pattern_id)
                documents_by_tenant[tenant_id].update(int(document_id) for document_id in pattern.document_ids)
                tenant_vector_counts[tenant_id] += int(pattern.vector_count)

        partition_by_pattern: dict[int, WorkloadAwarePartition] = {}
        for partition in partitions:
            for pattern_id in partition.logical_pattern_ids:
                partition_by_pattern[int(pattern_id)] = partition
        partition_by_id = {
            str(partition.partition_id): partition
            for partition in partitions
        }

        alpha = 1.0

        def _search_cost(table_vectors: int, accessible_vectors: int) -> float:
            table_vectors = max(1, int(table_vectors))
            accessible_vectors = max(1, min(int(accessible_vectors), table_vectors))
            selectivity = max(float(accessible_vectors) / float(table_vectors), 1e-9)
            return float(alpha + math.log1p(float(table_vectors)) * math.sqrt(1.0 / selectivity))

        def _protection_cost(table_vectors: int, accessible_vectors: int) -> float:
            table_vectors = max(1, int(table_vectors))
            accessible_vectors = max(1, min(int(accessible_vectors), table_vectors))
            selectivity = max(float(accessible_vectors) / float(table_vectors), 1e-9)
            # Shared protection overlays are filtered after ANN search by pattern_id.
            # Low tenant selectivity therefore hurts both latency and recall more than
            # it does in normal partition routing, so use a stronger risk penalty.
            return float(_search_cost(table_vectors, accessible_vectors) * math.sqrt(1.0 / selectivity))

        base_cost_by_tenant: dict[int, float] = {}
        base_branch_count_by_tenant: dict[int, int] = {}
        for tenant_id, pattern_ids in patterns_by_tenant.items():
            vectors_by_partition: dict[str, int] = Counter()
            for pattern_id in pattern_ids:
                partition = partition_by_pattern.get(int(pattern_id))
                if partition is None:
                    continue
                vectors_by_partition[str(partition.partition_id)] += int(pattern_vector_counts.get(int(pattern_id), 0))
            base_branch_count_by_tenant[int(tenant_id)] = int(len(vectors_by_partition))
            base_cost = 0.0
            for partition_id, matched_vectors in vectors_by_partition.items():
                partition = partition_by_id.get(str(partition_id))
                partition_vectors = int(partition.vector_count) if partition is not None else int(matched_vectors)
                base_cost += _search_cost(partition_vectors, int(matched_vectors))
            base_cost_by_tenant[int(tenant_id)] = float(base_cost)

        def _group_vector_count(pattern_ids: set[int]) -> int:
            return int(sum(int(pattern_vector_counts.get(int(pattern_id), 0)) for pattern_id in pattern_ids))

        def _group_benefit(tenant_ids: set[int], pattern_ids: set[int]) -> float:
            group_vectors = _group_vector_count(pattern_ids)
            if group_vectors <= 0:
                return 0.0
            benefit = 0.0
            for tenant_id in tenant_ids:
                tenant_vectors = int(tenant_vector_counts.get(int(tenant_id), 0))
                if tenant_vectors <= 0:
                    continue
                fallback_query_mass = 0.0 if has_workload_query_mass else float(tenant_query_weights.get(int(tenant_id), 1.0))
                query_mass = max(0.0, float(query_mass_by_tenant.get(int(tenant_id), fallback_query_mass)))
                base_cost = float(base_cost_by_tenant.get(int(tenant_id), 0.0))
                protect_cost = _protection_cost(group_vectors, tenant_vectors)
                benefit += query_mass * float(base_cost - protect_cost)
            return float(benefit)

        def _group_selectivity_stats(tenant_ids: set[int], group_vectors: int) -> tuple[float, float]:
            group_vectors = max(1, int(group_vectors))
            selectivities = [
                min(1.0, float(max(1, int(tenant_vector_counts.get(int(tenant_id), 0)))) / float(group_vectors))
                for tenant_id in tenant_ids
                if int(tenant_vector_counts.get(int(tenant_id), 0)) > 0
            ]
            if not selectivities:
                return 0.0, 0.0
            return float(min(selectivities)), float(sum(selectivities) / len(selectivities))

        def _tenant_query_mass(tenant_id: int) -> float:
            return max(
                0.0,
                float(
                    query_mass_by_tenant.get(
                        int(tenant_id),
                        0.0 if has_workload_query_mass else tenant_query_weights.get(int(tenant_id), 1.0),
                    )
                ),
            )

        candidate_groups_by_key: dict[tuple[int, ...], dict[str, object]] = {}
        skipped_non_positive = 0
        skipped_budget = 0
        skipped_low_fanout = 0
        skipped_join_non_positive = 0
        skipped_join_budget = 0
        skipped_join_density = 0
        skipped_join_low_overlap = 0

        ordered_tenants = [
            int(tenant_id)
            for tenant_id in tenant_order
            if int(tenant_id) in patterns_by_tenant and int(tenant_vector_counts.get(int(tenant_id), 0)) > 0
        ]
        tenant_order_rank = {
            int(tenant_id): int(index)
            for index, tenant_id in enumerate(ordered_tenants)
        }
        branch_counts = [
            int(base_branch_count_by_tenant.get(int(tenant_id), 0))
            for tenant_id in ordered_tenants
        ]
        positive_branch_counts = [count for count in branch_counts if count > 0]
        avg_branch_count = (
            float(sum(positive_branch_counts) / len(positive_branch_counts))
            if positive_branch_counts
            else 1.0
        )
        sorted_branch_counts = sorted(positive_branch_counts)
        p75_branch_count = (
            float(sorted_branch_counts[min(len(sorted_branch_counts) - 1, int(0.75 * (len(sorted_branch_counts) - 1)))])
            if sorted_branch_counts
            else 1.0
        )
        fanout_branch_floor = max(2.0, min(float(p75_branch_count), max(1.0, avg_branch_count)))
        min_protection_selectivity = max(0.10, min(0.25, 1.0 / max(fanout_branch_floor, 1.0)))
        partition_vector_counts = sorted(int(partition.vector_count) for partition in partitions)
        median_partition_vectors = (
            float(partition_vector_counts[len(partition_vector_counts) // 2])
            if partition_vector_counts
            else 1.0
        )
        small_partition_vector_limit = max(1.0, float(median_partition_vectors))
        tenant_entries: list[dict[str, object]] = []
        for tenant_id in ordered_tenants:
            tenant_patterns = set(int(pattern_id) for pattern_id in patterns_by_tenant[int(tenant_id)])
            tenant_vectors = int(tenant_vector_counts.get(int(tenant_id), 0))
            if tenant_vectors <= 0:
                continue
            vectors_by_partition: dict[str, int] = Counter()
            for pattern_id in tenant_patterns:
                partition = partition_by_pattern.get(int(pattern_id))
                if partition is None:
                    continue
                vectors_by_partition[str(partition.partition_id)] += int(pattern_vector_counts.get(int(pattern_id), 0))
            branch_count = int(len(vectors_by_partition))
            small_branch_count = int(
                sum(
                    1
                    for partition_id in vectors_by_partition
                    if partition_by_id.get(str(partition_id)) is not None
                    and int(partition_by_id[str(partition_id)].vector_count)
                    <= int(small_partition_vector_limit)
                )
            )
            small_branch_ratio = (
                float(small_branch_count / branch_count)
                if branch_count > 0
                else 0.0
            )

            workload_query_mass = _tenant_query_mass(int(tenant_id))
            if branch_count < int(math.ceil(fanout_branch_floor)) and not (
                workload_query_mass > 0.0 and branch_count > 1
            ):
                skipped_low_fanout += 1
                continue
            if tenant_vectors > int(budget_vectors):
                skipped_budget += 1
                continue
            singleton_benefit = _group_benefit({int(tenant_id)}, tenant_patterns)
            fanout_pain = float(
                workload_query_mass
                * max(1, int(branch_count) - 1)
                * max(1.0, float(base_cost_by_tenant.get(int(tenant_id), 0.0)))
            )
            if singleton_benefit <= 0.0:
                skipped_non_positive += 1
                singleton_benefit = float(fanout_pain / max(1, int(branch_count)))
            top_pattern_ids = tuple(
                int(pattern_id)
                for pattern_id in sorted(
                    tenant_patterns,
                    key=lambda pattern_id: (
                        -int(pattern_vector_counts.get(int(pattern_id), 0)),
                        int(pattern_id),
                    ),
                )[:3]
            )
            min_selectivity, avg_selectivity = _group_selectivity_stats({int(tenant_id)}, int(tenant_vectors))
            tenant_entries.append(
                {
                    "tenant_id": int(tenant_id),
                    "pattern_ids": tenant_patterns,
                    "pattern_key": tuple(sorted(int(pattern_id) for pattern_id in tenant_patterns)),
                    "vector_count": int(tenant_vectors),
                    "benefit": float(singleton_benefit),
                    "density": float(singleton_benefit / max(tenant_vectors, 1)),
                    "branch_count": int(branch_count),
                    "small_branch_count": int(small_branch_count),
                    "small_branch_ratio": float(small_branch_ratio),
                    "fanout_pain": float(fanout_pain),
                    "top_pattern_ids": top_pattern_ids,
                    "min_tenant_selectivity": float(min_selectivity),
                    "avg_tenant_selectivity": float(avg_selectivity),
                }
            )

        def _candidate_key(group: dict[str, object]) -> tuple[int, ...]:
            return tuple(sorted(int(value) for value in group["tenant_ids"]))

        def _record_candidate(group: dict[str, object]) -> None:
            tenant_ids = set(int(value) for value in group["tenant_ids"])
            pattern_ids = set(int(value) for value in group["pattern_ids"])
            vector_count = int(group["vector_count"])
            benefit = float(group["benefit"])
            if not tenant_ids or not pattern_ids or vector_count <= 0 or benefit <= 0.0:
                return
            min_selectivity, avg_selectivity = _group_selectivity_stats(tenant_ids, vector_count)
            payload = {
                "tenant_ids": tenant_ids,
                "pattern_ids": pattern_ids,
                "vector_count": int(vector_count),
                "benefit": float(benefit),
                "benefit_density": float(benefit / max(vector_count, 1)),
                "fanout_pain": float(group.get("fanout_pain", 0.0) or 0.0),
                "min_tenant_selectivity": float(min_selectivity),
                "avg_tenant_selectivity": float(avg_selectivity),
            }
            key = _candidate_key(payload)
            existing = candidate_groups_by_key.get(key)
            if existing is None:
                candidate_groups_by_key[key] = payload
                return
            existing_score = (
                float(existing.get("fanout_pain", 0.0) or 0.0),
                float(existing.get("benefit_density", 0.0) or 0.0),
                float(existing.get("benefit", 0.0) or 0.0),
                -int(existing.get("vector_count", 0) or 0),
            )
            payload_score = (
                float(payload.get("fanout_pain", 0.0) or 0.0),
                float(payload.get("benefit_density", 0.0) or 0.0),
                float(payload.get("benefit", 0.0) or 0.0),
                -int(payload.get("vector_count", 0) or 0),
            )
            if payload_score > existing_score:
                candidate_groups_by_key[key] = payload

        def _entry_sort_key(entry: dict[str, object], offset: int) -> tuple[object, ...]:
            top_pattern_ids = tuple(int(value) for value in (entry.get("top_pattern_ids", ()) or ()))
            if top_pattern_ids:
                rotated = top_pattern_ids[offset:] + top_pattern_ids[:offset]
            else:
                rotated = ()
            padded = rotated + tuple([-1] * max(0, 3 - len(rotated)))
            return (
                padded,
                -int(entry.get("branch_count", 0) or 0),
                -float(entry.get("fanout_pain", 0.0) or 0.0),
                -float(entry.get("small_branch_ratio", 0.0) or 0.0),
                -int(entry.get("vector_count", 0) or 0),
                -float(entry.get("density", 0.0) or 0.0),
                int(tenant_order_rank.get(int(entry["tenant_id"]), 0)),
            )

        def _singleton_group(entry: dict[str, object]) -> dict[str, object]:
            return {
                "tenant_ids": {int(entry["tenant_id"])},
                "pattern_ids": set(int(value) for value in entry["pattern_ids"]),
                "vector_count": int(entry["vector_count"]),
                "benefit": float(entry["benefit"]),
                "benefit_density": float(entry["density"]),
                "fanout_pain": float(entry.get("fanout_pain", 0.0) or 0.0),
                "min_tenant_selectivity": float(entry.get("min_tenant_selectivity", 1.0) or 1.0),
                "avg_tenant_selectivity": float(entry.get("avg_tenant_selectivity", 1.0) or 1.0),
            }

        def _try_merge_group(group: dict[str, object], entry: dict[str, object]) -> Optional[dict[str, object]]:
            tenant_id = int(entry["tenant_id"])
            current_tenant_ids = set(int(value) for value in group["tenant_ids"])
            if tenant_id in current_tenant_ids:
                return None
            current_pattern_ids = set(int(value) for value in group["pattern_ids"])
            tenant_patterns = set(int(value) for value in entry["pattern_ids"])
            tenant_vectors = int(entry["vector_count"])
            merged_pattern_ids = current_pattern_ids | tenant_patterns
            merged_vector_count = _group_vector_count(merged_pattern_ids)
            delta_vectors = max(0, int(merged_vector_count - int(group["vector_count"])))
            overlap_vectors = max(0, int(group["vector_count"]) + int(tenant_vectors) - int(merged_vector_count))
            if delta_vectors > 0 and overlap_vectors <= delta_vectors:
                return None
            merged_tenant_ids = current_tenant_ids | {tenant_id}
            old_benefit = float(group["benefit"])
            new_benefit = _group_benefit(merged_tenant_ids, merged_pattern_ids)
            delta_benefit = float(new_benefit - old_benefit)
            if delta_benefit <= 0.0:
                return None
            old_density = float(old_benefit / max(int(group["vector_count"]), 1))
            new_density = float(new_benefit / max(int(merged_vector_count), 1))
            separate_benefit = float(old_benefit + max(0.0, float(entry["benefit"])))
            separate_vectors = int(group["vector_count"]) + int(tenant_vectors)
            separate_density = float(separate_benefit / max(separate_vectors, 1))
            if new_density < max(old_density, separate_density):
                return None
            min_selectivity, avg_selectivity = _group_selectivity_stats(
                merged_tenant_ids,
                int(merged_vector_count),
            )
            if min_selectivity < float(min_protection_selectivity):
                return None
            return {
                "tenant_ids": merged_tenant_ids,
                "pattern_ids": merged_pattern_ids,
                "vector_count": int(merged_vector_count),
                "benefit": float(new_benefit),
                "benefit_density": float(new_density),
                "fanout_pain": float(group.get("fanout_pain", 0.0) or 0.0) + float(entry.get("fanout_pain", 0.0) or 0.0),
                "min_tenant_selectivity": float(min_selectivity),
                "avg_tenant_selectivity": float(avg_selectivity),
            }

        def _build_candidates_from_order(entries: list[dict[str, object]]) -> tuple[int, int, int]:
            local_low_overlap = 0
            local_non_positive = 0
            local_density = 0
            current: Optional[dict[str, object]] = None
            for entry in entries:
                _record_candidate(_singleton_group(entry))
                if current is None:
                    current = _singleton_group(entry)
                    continue
                merged = _try_merge_group(current, entry)
                if merged is None:
                    current_pattern_ids = set(int(value) for value in current["pattern_ids"])
                    tenant_patterns = set(int(value) for value in entry["pattern_ids"])
                    merged_pattern_ids = current_pattern_ids | tenant_patterns
                    merged_vector_count = _group_vector_count(merged_pattern_ids)
                    delta_vectors = max(0, int(merged_vector_count - int(current["vector_count"])))
                    overlap_vectors = max(0, int(current["vector_count"]) + int(entry["vector_count"]) - int(merged_vector_count))
                    if delta_vectors > 0 and overlap_vectors <= delta_vectors:
                        local_low_overlap += 1
                    else:
                        merged_tenant_ids = set(int(value) for value in current["tenant_ids"]) | {int(entry["tenant_id"])}
                        new_benefit = _group_benefit(merged_tenant_ids, merged_pattern_ids)
                        if new_benefit <= float(current["benefit"]):
                            local_non_positive += 1
                        else:
                            local_density += 1
                    _record_candidate(current)
                    current = _singleton_group(entry)
                    continue
                current = merged
            if current is not None:
                _record_candidate(current)
            return local_low_overlap, local_non_positive, local_density

        for offset in range(3):
            sorted_entries = sorted(
                tenant_entries,
                key=lambda entry, current_offset=offset: _entry_sort_key(entry, current_offset),
            )
            low_overlap_count, non_positive_count, density_count = _build_candidates_from_order(sorted_entries)
            skipped_join_low_overlap += int(low_overlap_count)
            skipped_join_non_positive += int(non_positive_count)
            skipped_join_density += int(density_count)

        exact_signature_candidate_count = 0
        exact_signature_entries: dict[tuple[int, ...], list[dict[str, object]]] = defaultdict(list)
        for entry in tenant_entries:
            pattern_key = tuple(int(pattern_id) for pattern_id in (entry.get("pattern_key", ()) or ()))
            if pattern_key:
                exact_signature_entries[pattern_key].append(entry)
        for pattern_key, entries in exact_signature_entries.items():
            if len(entries) <= 1:
                continue
            tenant_ids = {int(entry["tenant_id"]) for entry in entries}
            pattern_ids = set(int(pattern_id) for pattern_id in pattern_key)
            vector_count = _group_vector_count(pattern_ids)
            benefit = _group_benefit(tenant_ids, pattern_ids)
            if vector_count <= 0 or benefit <= 0.0:
                continue
            exact_signature_candidate_count += 1
            _record_candidate(
                {
                    "tenant_ids": tenant_ids,
                    "pattern_ids": pattern_ids,
                    "vector_count": int(vector_count),
                    "benefit": float(benefit),
                    "fanout_pain": float(
                        sum(float(entry.get("fanout_pain", 0.0) or 0.0) for entry in entries)
                    ),
                }
            )

        candidate_groups = list(candidate_groups_by_key.values())
        tenant_entry_by_id = {
            int(entry["tenant_id"]): entry
            for entry in tenant_entries
        }

        def _candidate_workload_fanout_gain(candidate: dict[str, object]) -> float:
            gain = 0.0
            for tenant_id in set(int(value) for value in candidate["tenant_ids"]):
                entry = tenant_entry_by_id.get(int(tenant_id), {})
                query_mass = _tenant_query_mass(int(tenant_id))
                branch_count = int(entry.get("branch_count", base_branch_count_by_tenant.get(int(tenant_id), 0)) or 0)
                gain += float(query_mass * max(0, branch_count - 1))
            return float(gain)

        for candidate in candidate_groups:
            workload_fanout_gain = _candidate_workload_fanout_gain(candidate)
            candidate["workload_fanout_gain"] = float(workload_fanout_gain)
            candidate["workload_fanout_density"] = float(
                workload_fanout_gain / max(int(candidate.get("vector_count", 0) or 0), 1)
            )
            candidate["positive_workload_tenant_count"] = int(
                sum(1 for tenant_id in set(int(value) for value in candidate["tenant_ids"]) if _tenant_query_mass(int(tenant_id)) > 0.0)
            )

        selectable_candidate_groups = [
            candidate
            for candidate in candidate_groups
            if float(candidate.get("workload_fanout_gain", 0.0) or 0.0) > 0.0
            and int(candidate.get("positive_workload_tenant_count", 0) or 0)
            == len(set(int(value) for value in candidate["tenant_ids"]))
        ]

        selectable_candidate_groups.sort(
            key=lambda candidate: (
                -float(candidate.get("workload_fanout_density", 0.0) or 0.0),
                -float(candidate.get("workload_fanout_gain", 0.0) or 0.0),
                -float(candidate.get("min_tenant_selectivity", 0.0) or 0.0),
                -float(candidate.get("avg_tenant_selectivity", 0.0) or 0.0),
                int(candidate.get("vector_count", 0) or 0),
            )
        )
        groups: list[dict[str, object]] = []
        used_vectors = 0
        used_tenant_ids: set[int] = set()
        for candidate in selectable_candidate_groups:
            tenant_ids = set(int(value) for value in candidate["tenant_ids"])
            if not tenant_ids or tenant_ids & used_tenant_ids:
                continue
            group_vectors = int(candidate["vector_count"])
            if used_vectors + group_vectors > int(budget_vectors):
                skipped_budget += 1
                continue
            groups.append(
                {
                    "group_id": f"protect_{len(groups)}",
                    "tenant_ids": tenant_ids,
                    "pattern_ids": set(int(value) for value in candidate["pattern_ids"]),
                    "vector_count": int(group_vectors),
                    "benefit": float(candidate["benefit"]),
                    "fanout_pain": float(candidate.get("fanout_pain", 0.0) or 0.0),
                    "workload_fanout_gain": float(candidate.get("workload_fanout_gain", 0.0) or 0.0),
                    "workload_fanout_density": float(candidate.get("workload_fanout_density", 0.0) or 0.0),
                    "min_tenant_selectivity": float(candidate.get("min_tenant_selectivity", 0.0) or 0.0),
                    "avg_tenant_selectivity": float(candidate.get("avg_tenant_selectivity", 0.0) or 0.0),
                }
            )
            used_tenant_ids.update(tenant_ids)
            used_vectors += int(group_vectors)

        created_group_count = int(sum(1 for group in groups if len(group["tenant_ids"]) == 1))
        joined_existing_count = int(sum(max(0, len(group["tenant_ids"]) - 1) for group in groups))

        access_overlays: list[dict[str, object]] = []
        protected_tenant_ids: set[int] = set()
        selected_group_payloads: list[dict[str, object]] = []
        for group_index, group in enumerate(groups):
            tenant_ids = tuple(sorted(int(value) for value in group["tenant_ids"]))
            pattern_ids = tuple(sorted(int(value) for value in group["pattern_ids"]))
            if not tenant_ids or not pattern_ids:
                continue
            document_pattern_pairs = sorted(
                {
                    (int(document_id), int(pattern_id))
                    for pattern_id in pattern_ids
                    for pattern in [pattern_by_id[int(pattern_id)]]
                    for document_id in pattern.document_ids
                }
            )
            document_ids = sorted({int(document_id) for document_id, _ in document_pattern_pairs})
            partition_ids = tuple(
                sorted(
                    {
                        str(partition_by_pattern[int(pattern_id)].partition_id)
                        for pattern_id in pattern_ids
                        if int(pattern_id) in partition_by_pattern
                    }
                )
            )
            group_payload = ",".join(str(tenant_id) for tenant_id in tenant_ids).encode("utf-8")
            group_id = hashlib.blake2b(group_payload, digest_size=10).hexdigest()
            table_name = get_access_signature_cover_table_name(f"protect_{group_id}")
            group_vector_count = int(group["vector_count"])
            group_benefit = float(group["benefit"])
            min_tenant_selectivity, avg_tenant_selectivity = _group_selectivity_stats(
                set(int(value) for value in tenant_ids),
                int(group_vector_count),
            )
            group_branch_counts = sorted(
                int(base_branch_count_by_tenant.get(int(tenant_id), 0))
                for tenant_id in tenant_ids
            )
            median_group_branch_count = (
                float(group_branch_counts[len(group_branch_counts) // 2])
                if group_branch_counts
                else 0.0
            )
            selected_group_payloads.append(
                {
                    "group_id": f"protect_{group_index}",
                    "signature_id": str(group_id),
                    "tenant_ids": [int(value) for value in tenant_ids],
                    "pattern_ids": [int(value) for value in pattern_ids],
                    "partition_ids": [str(value) for value in partition_ids],
                    "document_count": int(len(document_ids)),
                    "vector_count": int(group_vector_count),
                    "benefit": float(group_benefit),
                    "benefit_density": float(group_benefit / max(group_vector_count, 1)),
                    "fanout_pain": float(group.get("fanout_pain", 0.0) or 0.0),
                    "workload_fanout_gain": float(group.get("workload_fanout_gain", 0.0) or 0.0),
                    "workload_fanout_density": float(group.get("workload_fanout_density", 0.0) or 0.0),
                    "min_branch_count": int(group_branch_counts[0]) if group_branch_counts else 0,
                    "median_branch_count": float(median_group_branch_count),
                    "max_branch_count": int(group_branch_counts[-1]) if group_branch_counts else 0,
                    "min_tenant_selectivity": float(min_tenant_selectivity),
                    "avg_tenant_selectivity": float(avg_tenant_selectivity),
                    "table_name": table_name,
                }
            )
            for tenant_id in tenant_ids:
                protected_tenant_ids.add(int(tenant_id))
                tenant_pattern_ids = tuple(sorted(int(pattern_id) for pattern_id in patterns_by_tenant[int(tenant_id)]))
                tenant_vector_count = int(tenant_vector_counts.get(int(tenant_id), 0))
                requires_pattern_filter = set(int(pattern_id) for pattern_id in pattern_ids) != set(
                    int(pattern_id) for pattern_id in tenant_pattern_ids
                )
                access_overlays.append(
                    {
                        "tenant_id": int(tenant_id),
                        "partition_id": "__protection__",
                        "partition_ids": [str(partition_id) for partition_id in partition_ids],
                        "table_name": table_name,
                        "signature_id": str(group_id),
                        "protection_group_id": f"protect_{group_index}",
                        "cover_tenant_ids": [int(value) for value in tenant_ids],
                        "pattern_ids": [int(pattern_id) for pattern_id in tenant_pattern_ids],
                        "cover_pattern_ids": [int(pattern_id) for pattern_id in pattern_ids],
                        "document_ids": [int(document_id) for document_id in document_ids],
                        "document_pattern_pairs": [
                            [int(document_id), int(pattern_id)]
                            for document_id, pattern_id in document_pattern_pairs
                        ],
                        "document_count": int(len(document_ids)),
                        "vector_count": int(group_vector_count),
                        "tenant_vector_count": int(tenant_vector_count),
                        "materialization_cost": int(group_vector_count),
                        "covered_partition_count": int(len(partition_ids)),
                        "route_partition_count": int(base_branch_count_by_tenant.get(int(tenant_id), len(partition_ids))),
                        "query_mass": float(query_mass_by_tenant.get(int(tenant_id), tenant_query_weights.get(int(tenant_id), 1.0))),
                        "estimated_saved_cost": float(group_benefit),
                        "benefit_density": float(group_benefit / max(group_vector_count, 1)),
                        "workload_fanout_gain": float(group.get("workload_fanout_gain", 0.0) or 0.0),
                        "workload_fanout_density": float(group.get("workload_fanout_density", 0.0) or 0.0),
                        "min_tenant_selectivity": float(min_tenant_selectivity),
                        "avg_tenant_selectivity": float(avg_tenant_selectivity),
                        "overlay_type": "shared_protection_overlay",
                        "requires_pattern_filter": bool(requires_pattern_filter),
                    }
                )

        access_overlays.sort(
            key=lambda item: (
                str(item.get("table_name", "")),
                int(item.get("tenant_id", 0)),
            )
        )
        return [], {
            "access_overlays": access_overlays,
            "overlay_budget_vectors": int(budget_vectors),
            "protection_overlay_selected_vectors": int(used_vectors),
            "overlay_selected_vectors": int(used_vectors),
            "overlay_selected_tenant_count": 0,
            "overlay_selected_access_count": int(len(access_overlays)),
            "shared_protection_overlay_enabled": True,
            "shared_protection_group_count": int(len(selected_group_payloads)),
            "shared_protection_mapping_count": int(len(access_overlays)),
            "shared_protection_protected_tenant_count": int(len(protected_tenant_ids)),
            "shared_protection_joined_existing_count": int(joined_existing_count),
            "shared_protection_created_group_count": int(created_group_count),
            "shared_protection_skipped_non_positive_count": int(skipped_non_positive),
            "shared_protection_skipped_budget_count": int(skipped_budget),
            "shared_protection_skipped_low_fanout_count": int(skipped_low_fanout),
            "shared_protection_skipped_join_non_positive_count": int(skipped_join_non_positive),
            "shared_protection_skipped_join_budget_count": int(skipped_join_budget),
            "shared_protection_skipped_join_density_count": int(skipped_join_density),
            "shared_protection_skipped_join_low_overlap_count": int(skipped_join_low_overlap),
            "shared_protection_exact_signature_candidate_count": int(exact_signature_candidate_count),
            "shared_protection_no_filter_mapping_count": int(
                sum(1 for overlay in access_overlays if not bool(overlay.get("requires_pattern_filter", False)))
            ),
            "shared_protection_workload_fanout_gain": float(
                sum(float(group.get("workload_fanout_gain", 0.0) or 0.0) for group in groups)
            ),
            "shared_protection_fanout_branch_floor": float(fanout_branch_floor),
            "shared_protection_min_selectivity": float(min_protection_selectivity),
            "shared_protection_groups": selected_group_payloads,
            "access_signature_cover_count": int(len(selected_group_payloads)),
            "access_signature_cover_tenant_mapping_count": int(len(access_overlays)),
        }

    def _build_tenant_order(
        self,
        records: list[DocumentAccessRecord],
        tenant_query_weights: dict[int, float],
    ) -> tuple[tuple[int, ...], dict[int, float], dict[int, int], dict[int, float]]:
        tenant_doc_counts: dict[int, int] = Counter()
        tenant_cooccurrence: dict[int, int] = Counter()
        all_tenants = set(int(tenant_id) for record in records for tenant_id in record.tenant_ids)
        for record in records:
            tenants = tuple(sorted(int(tenant_id) for tenant_id in record.tenant_ids))
            for tenant_id in tenants:
                tenant_doc_counts[tenant_id] += 1
                tenant_cooccurrence[tenant_id] += max(0, len(tenants) - 1)

        tenant_weights: dict[int, float] = {}
        tenant_scores: dict[int, float] = {}
        for tenant_id in all_tenants:
            query_weight = float(tenant_query_weights.get(tenant_id, 1.0))
            tenant_weights[tenant_id] = query_weight
            tenant_scores[tenant_id] = (
                math.log1p(float(tenant_doc_counts.get(tenant_id, 0)))
                + math.log1p(query_weight)
                + math.log1p(float(tenant_cooccurrence.get(tenant_id, 0)))
            )

        ordered = tuple(
            sorted(
                all_tenants,
                key=lambda tenant_id: (
                    -tenant_scores[tenant_id],
                    -tenant_doc_counts.get(tenant_id, 0),
                    -tenant_weights[tenant_id],
                    int(tenant_id),
                ),
            )
        )
        return ordered, tenant_weights, dict(tenant_doc_counts), tenant_scores

    def _build_logical_patterns(
        self,
        records: list[DocumentAccessRecord],
        *,
        normalized_vectors: np.ndarray,
        document_block_counts: dict[int, int],
        tenant_order: tuple[int, ...],
        tenant_query_weights: dict[int, float],
    ) -> list[ACLLogicalPattern]:
        tenant_rank = {tenant_id: index for index, tenant_id in enumerate(tenant_order)}
        grouped: dict[tuple[int, ...], list[int]] = defaultdict(list)
        for index, record in enumerate(records):
            ordered_signature = tuple(
                sorted(
                    (int(tenant_id) for tenant_id in record.tenant_ids),
                    key=lambda tenant_id: (tenant_rank.get(tenant_id, len(tenant_rank)), int(tenant_id)),
                )
            )
            grouped[ordered_signature].append(index)

        logical_patterns: list[ACLLogicalPattern] = []
        ordered_groups = sorted(grouped.items(), key=lambda item: (-len(item[1]), item[0]))
        for pattern_id, (ordered_signature, member_indices) in enumerate(ordered_groups):
            member_records = [records[index] for index in member_indices]
            document_ids = tuple(sorted(record.document_id for record in member_records))
            document_count = len(document_ids)
            vector_count = sum(int(document_block_counts.get(document_id, 1)) for document_id in document_ids)
            tenant_ids = tuple(sorted(int(tenant_id) for tenant_id in ordered_signature))
            representative_vectors = normalized_vectors[np.asarray(member_indices, dtype=np.int32)]
            centroid = _normalize_vector(representative_vectors.mean(axis=0)) if representative_vectors.size else np.zeros(0, dtype=np.float32)
            member_document_vectors = [
                (
                    int(record.document_id),
                    int(document_block_counts.get(int(record.document_id), 1)),
                    normalized_vectors[int(index)].astype(np.float32, copy=False),
                )
                for record, index in zip(member_records, member_indices)
            ]
            tenant_query_mass = {
                int(tenant_id): float(tenant_query_weights.get(int(tenant_id), 1.0) * document_count)
                for tenant_id in tenant_ids
            }
            pattern_query_mass = float(sum(tenant_query_mass.values()) / max(len(tenant_query_mass), 1))
            logical_patterns.append(
                ACLLogicalPattern(
                    pattern_id=int(pattern_id),
                    tenant_ids=tenant_ids,
                    ordered_tenant_ids=tuple(int(tenant_id) for tenant_id in ordered_signature),
                    document_ids=document_ids,
                    vector_count=int(vector_count),
                    document_count=int(document_count),
                    metadata={
                        "representative_centroid": centroid.astype(float).tolist(),
                        "tenant_query_mass": tenant_query_mass,
                        "pattern_query_mass": pattern_query_mass,
                        "acl_size": len(tenant_ids),
                        "_member_document_vectors": member_document_vectors,
                    },
                )
            )
        return logical_patterns

    def _resolve_target_partition_count(
        self,
        logical_patterns: list[ACLLogicalPattern],
        *,
        target_partition_count: Optional[int],
    ) -> int:
        if target_partition_count is not None:
            return max(1, int(target_partition_count))
        pattern_count = max(1, len(logical_patterns))
        return max(1, int(round(math.sqrt(pattern_count) * 3.0)))

    def _build_compact_route_dag(
        self,
        logical_patterns: list[ACLLogicalPattern],
        *,
        tenant_order: tuple[int, ...],
    ) -> tuple[list[PrefixDagNode], dict[str, object]]:
        tenant_rank = {int(tenant_id): index for index, tenant_id in enumerate(tenant_order)}
        prefix_to_node_id: dict[tuple[int, ...], int] = {(): 0}
        nodes: dict[int, PrefixDagNode] = {
            0: PrefixDagNode(node_id=0, prefix_tenants=(), metadata={"route_core": [], "pattern_ids": []})
        }
        next_node_id = 1
        route_core_histogram: dict[int, int] = Counter()
        tenant_entry_pattern_counts: dict[int, int] = Counter()

        def ensure_node(prefix: tuple[int, ...]) -> int:
            nonlocal next_node_id
            node_id = prefix_to_node_id.get(prefix)
            if node_id is not None:
                return int(node_id)
            node_id = next_node_id
            next_node_id += 1
            prefix_to_node_id[prefix] = int(node_id)
            nodes[int(node_id)] = PrefixDagNode(
                node_id=int(node_id),
                prefix_tenants=tuple(int(value) for value in prefix),
                metadata={"route_core": [int(value) for value in prefix], "pattern_ids": []},
            )
            return int(node_id)

        for pattern in logical_patterns:
            pattern_weight = float(pattern.metadata.get("pattern_weight", pattern.document_count))
            ordered_tenants = tuple(int(value) for value in pattern.ordered_tenant_ids)
            if not ordered_tenants:
                pattern.entry_tenant_ids = ()
                continue
            if len(ordered_tenants) == 1:
                route_core = ordered_tenants
            else:
                route_core_len = min(3, len(ordered_tenants))
                route_core = tuple(ordered_tenants[:route_core_len])
            route_core_histogram[len(route_core)] += 1
            pattern.entry_tenant_ids = tuple(sorted(set(int(tenant_id) for tenant_id in route_core)))
            pattern.metadata["route_core"] = [int(tenant_id) for tenant_id in route_core]
            for tenant_id in pattern.entry_tenant_ids:
                tenant_entry_pattern_counts[int(tenant_id)] += 1

            current_node_id = 0
            nodes[current_node_id].document_count += int(pattern.document_count)
            prefix: list[int] = []
            for tenant_id in route_core:
                prefix.append(int(tenant_id))
                prefix_tuple = tuple(prefix)
                child_node_id = ensure_node(prefix_tuple)
                nodes[current_node_id].children[int(tenant_id)] = int(child_node_id)
                current_node_id = int(child_node_id)
                nodes[current_node_id].document_count += int(pattern.document_count)
            nodes[current_node_id].terminal_pattern_ids.add(int(pattern.pattern_id))
            nodes[current_node_id].terminal_document_count += int(pattern.document_count)
            nodes[current_node_id].metadata.setdefault("pattern_ids", []).append(int(pattern.pattern_id))
            nodes[current_node_id].metadata["pattern_weight"] = float(
                nodes[current_node_id].metadata.get("pattern_weight", 0.0) + pattern_weight
            )

        for tenant_id in tenant_order:
            tenant_id = int(tenant_id)
            if tenant_id not in nodes[0].children:
                node_id = ensure_node((tenant_id,))
                nodes[0].children[int(tenant_id)] = int(node_id)

        dag_nodes = [nodes[node_id] for node_id in sorted(nodes)]
        return dag_nodes, {
            "dag_root_children": len(nodes[0].children),
            "dag_main_edge_count": int(sum(len(node.children) for node in dag_nodes)),
            "dag_supplemental_edge_count": 0,
            "route_core_length_histogram": {
                str(length): int(count)
                for length, count in sorted(route_core_histogram.items())
            },
            "tenant_entry_pattern_counts": {
                str(tenant_id): int(count)
                for tenant_id, count in sorted(tenant_entry_pattern_counts.items())
            },
        }

    def _build_acl_planning_tree(
        self,
        logical_patterns: list[ACLLogicalPattern],
    ) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
        patterns_by_id = {int(pattern.pattern_id): pattern for pattern in logical_patterns}
        tree_nodes: dict[int, dict[str, object]] = {
            0: {
                "node_id": 0,
                "pattern_id": None,
                "acl": tuple(),
                "parent_id": None,
                "child_ids": [],
                "subtree_pattern_ids": [],
                "collapsed": False,
                "is_root": True,
                "node_type": "root",
            }
        }

        ordered_patterns = sorted(
            logical_patterns,
            key=lambda pattern: (
                len(pattern.tenant_ids),
                -float(pattern.metadata.get("pattern_weight", pattern.document_count)),
                int(pattern.pattern_id),
            ),
        )
        parent_by_pattern: dict[int, Optional[int]] = {}
        child_ids_by_pattern: dict[int, list[int]] = defaultdict(list)
        patterns_sorted_by_size = [
            (int(pattern.pattern_id), set(int(tenant_id) for tenant_id in pattern.tenant_ids))
            for pattern in ordered_patterns
        ]

        for index, (pattern_id, acl_set) in enumerate(patterns_sorted_by_size):
            best_parent_id = None
            best_delta = None
            for candidate_pattern_id, candidate_acl in patterns_sorted_by_size[index + 1:]:
                if not acl_set.issubset(candidate_acl):
                    continue
                delta = len(candidate_acl) - len(acl_set)
                if best_parent_id is None or delta < best_delta or (
                    delta == best_delta
                    and float(
                        patterns_by_id[int(candidate_pattern_id)].metadata.get(
                            "pattern_weight",
                            patterns_by_id[int(candidate_pattern_id)].document_count,
                        )
                    )
                    > float(
                        patterns_by_id[int(best_parent_id)].metadata.get(
                            "pattern_weight",
                            patterns_by_id[int(best_parent_id)].document_count,
                        )
                    )
                ):
                    best_parent_id = int(candidate_pattern_id)
                    best_delta = int(delta)
                if delta == 1:
                    break
            parent_by_pattern[int(pattern_id)] = best_parent_id
            if best_parent_id is not None:
                child_ids_by_pattern[int(best_parent_id)].append(int(pattern_id))

        internal_pattern_ids = {
            int(pattern_id) for pattern_id, child_ids in child_ids_by_pattern.items() if child_ids
        }
        group_node_id_by_pattern: dict[int, int] = {}
        leaf_node_id_by_pattern: dict[int, int] = {}
        next_node_id = 1

        for pattern in logical_patterns:
            pattern_id = int(pattern.pattern_id)
            if pattern_id in internal_pattern_ids:
                group_node_id_by_pattern[int(pattern_id)] = int(next_node_id)
                next_node_id += 1
            leaf_node_id_by_pattern[int(pattern_id)] = int(next_node_id)
            next_node_id += 1

        for pattern in logical_patterns:
            pattern_id = int(pattern.pattern_id)
            acl = tuple(int(tenant_id) for tenant_id in pattern.tenant_ids)
            if pattern_id in internal_pattern_ids:
                group_node_id = int(group_node_id_by_pattern[int(pattern_id)])
                tree_nodes[group_node_id] = {
                    "node_id": group_node_id,
                    "pattern_id": None,
                    "acl": acl,
                    "parent_id": None,
                    "child_ids": [],
                    "subtree_pattern_ids": [],
                    "collapsed": False,
                    "is_root": False,
                    "node_type": "group",
                    "anchor_pattern_id": int(pattern_id),
                }
            leaf_node_id = int(leaf_node_id_by_pattern[int(pattern_id)])
            tree_nodes[leaf_node_id] = {
                "node_id": leaf_node_id,
                "pattern_id": int(pattern_id),
                "acl": acl,
                "parent_id": None,
                "child_ids": [],
                "subtree_pattern_ids": [],
                "collapsed": False,
                "is_root": False,
                "node_type": "pattern_leaf",
            }

        for pattern in logical_patterns:
            pattern_id = int(pattern.pattern_id)
            parent_pattern_id = parent_by_pattern.get(pattern_id)
            current_node_id = (
                int(group_node_id_by_pattern[int(pattern_id)])
                if pattern_id in internal_pattern_ids
                else int(leaf_node_id_by_pattern[int(pattern_id)])
            )
            parent_node_id = 0
            if parent_pattern_id is not None:
                parent_node_id = int(group_node_id_by_pattern[int(parent_pattern_id)])
            tree_nodes[int(current_node_id)]["parent_id"] = int(parent_node_id)
            tree_nodes[int(parent_node_id)]["child_ids"].append(int(current_node_id))

            if pattern_id in internal_pattern_ids:
                leaf_node_id = int(leaf_node_id_by_pattern[int(pattern_id)])
                tree_nodes[int(leaf_node_id)]["parent_id"] = int(current_node_id)
                tree_nodes[int(current_node_id)]["child_ids"].append(int(leaf_node_id))
                for child_pattern_id in sorted(child_ids_by_pattern.get(pattern_id, ())):
                    child_node_id = (
                        int(group_node_id_by_pattern[int(child_pattern_id)])
                        if int(child_pattern_id) in internal_pattern_ids
                        else int(leaf_node_id_by_pattern[int(child_pattern_id)])
                    )
                    tree_nodes[int(child_node_id)]["parent_id"] = int(current_node_id)
                    tree_nodes[int(current_node_id)]["child_ids"].append(int(child_node_id))

        for node in tree_nodes.values():
            node["child_ids"] = sorted(set(int(child_id) for child_id in node["child_ids"]))

        def populate_subtree(node_id: int) -> list[int]:
            node = tree_nodes[int(node_id)]
            subtree_pattern_ids: list[int] = []
            if node.get("pattern_id") is not None:
                subtree_pattern_ids.append(int(node["pattern_id"]))
            for child_id in node["child_ids"]:
                subtree_pattern_ids.extend(populate_subtree(int(child_id)))
            node["subtree_pattern_ids"] = sorted(set(int(pattern_id) for pattern_id in subtree_pattern_ids))
            return node["subtree_pattern_ids"]

        populate_subtree(0)
        return tree_nodes, {
            "planning_node_count": int(len(tree_nodes)),
            "planning_root_children": int(len(tree_nodes[0]["child_ids"])),
            "planning_group_node_count": int(len(group_node_id_by_pattern)),
            "planning_pattern_leaf_count": int(len(leaf_node_id_by_pattern)),
        }

    def _prune_acl_tree(
        self,
        planning_tree: dict[int, dict[str, object]],
        *,
        logical_patterns: list[ACLLogicalPattern],
    ) -> tuple[dict[int, dict[str, object]], dict[str, object]]:
        patterns_by_id = {int(pattern.pattern_id): pattern for pattern in logical_patterns}
        pruned_tree = {
            int(node_id): {
                **node,
                "child_ids": [int(child_id) for child_id in node["child_ids"]],
                "subtree_pattern_ids": [int(pattern_id) for pattern_id in node["subtree_pattern_ids"]],
                "collapsed": False,
                "collapse_cost": 0.0,
                "split_cost": 0.0,
            }
            for node_id, node in planning_tree.items()
        }
        active_nodes: dict[int, bool] = {int(node_id): True for node_id in pruned_tree}

        def collapse_cost(node: dict[str, object]) -> float:
            subtree_pattern_ids = [int(pattern_id) for pattern_id in node.get("subtree_pattern_ids", ())]
            if not subtree_pattern_ids:
                return 0.0
            acls = [
                set(int(value) for value in patterns_by_id[int(pattern_id)].tenant_ids)
                for pattern_id in subtree_pattern_ids
                if int(pattern_id) in patterns_by_id
            ]
            if not acls:
                return 0.0
            common = set.intersection(*acls) if acls else set()
            cost = 0.0
            for acl in acls:
                cost += 1.0 - float(len(common) / max(len(acl), 1))
            return float(cost)

        def dfs(node_id: int) -> tuple[float, int]:
            node = pruned_tree[int(node_id)]
            child_ids = [int(child_id) for child_id in node["child_ids"]]
            if not child_ids:
                cost = float(collapse_cost(node))
                node["collapse_cost"] = cost
                node["split_cost"] = cost
                return cost, 1

            child_total_cost = 0.0
            child_active_count = 1
            for child_id in child_ids:
                sub_cost, sub_count = dfs(int(child_id))
                child_total_cost += float(sub_cost)
                child_active_count += int(sub_count)

            node_collapse_cost = float(collapse_cost(node))
            split_cost = float(child_total_cost + 0.5 * max(0, len(child_ids) - 1) * math.log1p(len(node["subtree_pattern_ids"])))
            node["collapse_cost"] = node_collapse_cost
            node["split_cost"] = split_cost
            if int(node_id) != 0 and node_collapse_cost <= split_cost:
                node["collapsed"] = True
                for child_id in child_ids:
                    active_nodes[int(child_id)] = False
                node["child_ids"] = []
                return node_collapse_cost, 1
            return split_cost, child_active_count

        _, active_count = dfs(0)
        for node_id, node in pruned_tree.items():
            node["active"] = bool(active_nodes.get(int(node_id), False))
        return pruned_tree, {
            "pruned_planning_node_count": int(active_count),
            "pruned_removed_planning_node_count": int(len(pruned_tree) - sum(1 for active in active_nodes.values() if active)),
        }

    def _dp_cut_acl_tree(
        self,
        *,
        logical_patterns: list[ACLLogicalPattern],
        pruned_tree: dict[int, dict[str, object]],
        tenant_weights: dict[int, float],
        queries: list[WorkloadQuery],
        target_partition_count: int,
        max_partition_vector_count: Optional[int],
    ) -> tuple[list[WorkloadAwarePartition], dict[str, object]]:
        patterns_by_id = {int(pattern.pattern_id): pattern for pattern in logical_patterns}
        active_node_ids = sorted(int(node_id) for node_id, node in pruned_tree.items() if bool(node.get("active", False)))
        if not active_node_ids:
            return [], {
                "dp_target_partition_count": int(target_partition_count),
                "dp_selected_cut_count": 0,
            }

        active_children: dict[int, list[int]] = {
            int(node_id): [
                int(child_id)
                for child_id in pruned_tree[int(node_id)]["child_ids"]
                if bool(pruned_tree.get(int(child_id), {}).get("active", False))
            ]
            for node_id in active_node_ids
        }
        frontier_node_ids = sorted(
            int(node_id) for node_id in active_node_ids if not active_children.get(int(node_id), [])
        )
        if not frontier_node_ids:
            frontier_node_ids = [0]

        all_pattern_ids = tuple(sorted(int(pattern.pattern_id) for pattern in logical_patterns))
        all_pattern_id_set = frozenset(int(pattern_id) for pattern_id in all_pattern_ids)
        total_vector_count = max(sum(int(pattern.vector_count) for pattern in logical_patterns), 1)
        average_target_vector_count = float(total_vector_count / max(int(target_partition_count), 1))
        requested_partition_count = max(1, int(target_partition_count))
        max_feasible_partition_count = max(1, len(frontier_node_ids))
        effective_partition_budget = min(requested_partition_count, max_feasible_partition_count)
        max_closed_components = int(effective_partition_budget)

        summary_cache: dict[tuple[int, ...], dict[str, object]] = {}

        def _aggregate_pattern_summary(pattern_ids: tuple[int, ...]) -> dict[str, object]:
            key = tuple(sorted(set(int(pattern_id) for pattern_id in pattern_ids)))
            cached = summary_cache.get(key)
            if cached is not None:
                return cached
            descendant_patterns = [patterns_by_id[int(pattern_id)] for pattern_id in key if int(pattern_id) in patterns_by_id]
            document_ids = sorted({int(document_id) for pattern in descendant_patterns for document_id in pattern.document_ids})
            tenant_ids = sorted({int(tenant_id) for pattern in descendant_patterns for tenant_id in pattern.tenant_ids})
            vector_count = sum(int(pattern.vector_count) for pattern in descendant_patterns)
            document_count = sum(int(pattern.document_count) for pattern in descendant_patterns)
            tenant_doc_counts: dict[int, int] = Counter()
            tenant_query_mass: dict[int, float] = Counter()
            centroids = []
            weighted_centroid_numerator = None
            for pattern in descendant_patterns:
                centroid = _parse_vector(pattern.metadata.get("representative_centroid", []))
                if centroid.size:
                    centroids.append(centroid)
                    numerator = centroid * max(int(pattern.document_count), 1)
                    weighted_centroid_numerator = numerator if weighted_centroid_numerator is None else weighted_centroid_numerator + numerator
                for tenant_id in pattern.tenant_ids:
                    tenant_doc_counts[int(tenant_id)] += int(pattern.document_count)
                for tenant_id, value in (pattern.metadata.get("tenant_query_mass", {}) or {}).items():
                    tenant_query_mass[int(tenant_id)] += float(value)
            centroid = np.zeros(0, dtype=np.float32)
            if weighted_centroid_numerator is not None and document_count > 0:
                centroid = _normalize_vector(weighted_centroid_numerator / float(max(document_count, 1)))
            dispersion = 0.0
            if centroids and centroid.size:
                dispersion = float(
                    sum(float(1.0 - np.dot(_normalize_vector(value), centroid)) for value in centroids) / max(len(centroids), 1)
                )
            acl_union = set(tenant_ids)
            impurity = 0.0
            total_weight = 0.0
            for pattern in descendant_patterns:
                weight = float(pattern.metadata.get("pattern_weight", pattern.document_count))
                total_weight += weight
                impurity += weight * float(len(acl_union.difference(set(int(value) for value in pattern.tenant_ids))) / max(len(acl_union), 1))
            if total_weight > 0.0:
                impurity = float(impurity / total_weight)
            size_penalty = float(max(0.0, float(vector_count) / max(average_target_vector_count, 1.0) - 1.0) ** 2)
            normalized_tenant_query_mass = {
                str(tenant_id): float(value)
                for tenant_id, value in sorted(tenant_query_mass.items())
            }
            route_core = tuple(
                int(tenant_id)
                for tenant_id in sorted(
                    tenant_ids,
                    key=lambda tenant_id: (
                        -float(tenant_query_mass.get(int(tenant_id), 0.0)),
                        -int(tenant_doc_counts.get(int(tenant_id), 0)),
                        int(tenant_id),
                    ),
                )[:3]
            )
            summary = {
                "pattern_ids": key,
                "pattern_id_set": frozenset(int(pattern_id) for pattern_id in key),
                "document_ids": tuple(document_ids),
                "tenant_ids": tuple(tenant_ids),
                "tenant_doc_counts": {int(key): int(value) for key, value in tenant_doc_counts.items()},
                "tenant_query_mass": {int(key): float(value) for key, value in tenant_query_mass.items()},
                "vector_count": int(vector_count),
                "document_count": int(document_count),
                "pattern_count": len(key),
                "representative_centroid": centroid.astype(float).tolist(),
                "centroid_vector": centroid,
                "centroid_dispersion": float(dispersion),
                "acl_impurity": float(impurity),
                "size_penalty": float(size_penalty),
                "entry_tenant_ids": tuple(sorted(tenant_ids)),
                "partition_query_mass": float(sum(float(value) for value in tenant_query_mass.values())),
                "normalized_tenant_query_mass": normalized_tenant_query_mass,
                "route_core": route_core,
            }
            summary_cache[key] = summary
            return summary

        global_summary = _aggregate_pattern_summary(all_pattern_ids)
        frontier_node_id_set = set(int(node_id) for node_id in frontier_node_ids)
        edge_cut_scores: dict[tuple[int, int], float] = {}

        def _target_size_fit(vector_count: int) -> float:
            ratio = float(max(int(vector_count), 1)) / max(float(average_target_vector_count), 1.0)
            if ratio <= 0.0:
                return 0.0
            return float(min(ratio, 1.0 / ratio))

        for parent_id in active_node_ids:
            for child_id in active_children.get(int(parent_id), []):
                child_pattern_ids = tuple(int(pattern_id) for pattern_id in pruned_tree[int(child_id)]["subtree_pattern_ids"])
                child_summary = _aggregate_pattern_summary(child_pattern_ids)
                rest_pattern_ids = tuple(
                    sorted(int(pattern_id) for pattern_id in (all_pattern_id_set - child_summary["pattern_id_set"]))
                )
                rest_summary = _aggregate_pattern_summary(rest_pattern_ids)
                total_vectors = float(int(child_summary["vector_count"]) + int(rest_summary["vector_count"]))
                child_share = float(child_summary["vector_count"]) / total_vectors if total_vectors > 0.0 else 0.0
                balance = 4.0 * child_share * max(0.0, 1.0 - child_share)
                acl_separation = 1.0 - _weighted_jaccard_from_sets(
                    child_summary["tenant_ids"],
                    rest_summary["tenant_ids"],
                    tenant_weights=tenant_weights,
                )
                workload_separation = 1.0 - _weighted_jaccard_from_dicts(
                    child_summary["tenant_query_mass"],
                    rest_summary["tenant_query_mass"],
                )
                semantic_separation = 0.0
                child_centroid = child_summary["centroid_vector"]
                rest_centroid = rest_summary["centroid_vector"]
                if child_centroid.size and rest_centroid.size:
                    semantic_separation = max(0.0, float(1.0 - np.dot(child_centroid, rest_centroid)))
                separation_score = float((acl_separation + workload_separation + semantic_separation) / 3.0)
                shared_workload_mass = sum(
                    min(
                        float(child_summary["tenant_query_mass"].get(int(tenant_id), 0.0)),
                        float(rest_summary["tenant_query_mass"].get(int(tenant_id), 0.0)),
                    )
                    for tenant_id in set(child_summary["tenant_query_mass"]) | set(rest_summary["tenant_query_mass"])
                )
                fanout_pressure = float(
                    shared_workload_mass / max(float(global_summary["partition_query_mass"]), 1e-9)
                )
                target_size_fit = _target_size_fit(int(child_summary["vector_count"]))
                edge_cut_scores[(int(parent_id), int(child_id))] = float(
                    separation_score
                    * math.sqrt(max(balance * target_size_fit, 0.0))
                    * max(0.0, 1.0 - fanout_pressure)
                )

        dp_tables: dict[int, dict[tuple[int, int], float]] = {}
        dp_backtrace: dict[int, tuple[list[int], list[dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int], str]]]]] = {}

        def solve(node_id: int) -> dict[tuple[int, int], float]:
            cached = dp_tables.get(int(node_id))
            if cached is not None:
                return cached
            children = active_children.get(int(node_id), [])
            initial_state = (0, 1 if int(node_id) in frontier_node_id_set else 0)
            current: dict[tuple[int, int], float] = {initial_state: 0.0}
            step_backtraces: list[dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int], str]]] = []
            for child_id in children:
                child_table = solve(int(child_id))
                next_current: dict[tuple[int, int], float] = {}
                step_trace: dict[tuple[int, int], tuple[tuple[int, int], tuple[int, int], str]] = {}
                for parent_state, parent_score in current.items():
                    parent_closed, parent_has_frontier = parent_state
                    for child_state, child_score in child_table.items():
                        child_closed, child_has_frontier = child_state
                        keep_closed = int(parent_closed + child_closed)
                        if keep_closed <= max_closed_components:
                            keep_state = (keep_closed, 1 if parent_has_frontier or child_has_frontier else 0)
                            keep_score = float(parent_score + child_score)
                            existing_keep_score = next_current.get(keep_state)
                            if existing_keep_score is None or keep_score > float(existing_keep_score):
                                next_current[keep_state] = keep_score
                                step_trace[keep_state] = (parent_state, child_state, "keep")

                        if int(child_has_frontier) == 1:
                            cut_closed = int(parent_closed + child_closed + 1)
                            if cut_closed <= max_closed_components:
                                cut_state = (cut_closed, int(parent_has_frontier))
                                cut_score = float(
                                    parent_score
                                    + child_score
                                    + edge_cut_scores.get((int(node_id), int(child_id)), 0.0)
                                )
                                existing_cut_score = next_current.get(cut_state)
                                if existing_cut_score is None or cut_score > float(existing_cut_score):
                                    next_current[cut_state] = cut_score
                                    step_trace[cut_state] = (parent_state, child_state, "cut")
                current = next_current
                step_backtraces.append(step_trace)

            dp_tables[int(node_id)] = current
            dp_backtrace[int(node_id)] = ([int(child_id) for child_id in children], step_backtraces)
            return current

        root_table = solve(0)
        feasible_root_states = [
            ((int(closed_count), int(has_frontier)), float(score))
            for (closed_count, has_frontier), score in root_table.items()
            if int(closed_count) + int(has_frontier) <= int(effective_partition_budget)
        ]
        if feasible_root_states:
            feasible_root_states.sort(
                key=lambda item: (
                    -(int(item[0][0]) + int(item[0][1])),
                    -float(item[1]),
                    int(item[0][1]),
                )
            )
            selected_root_state = feasible_root_states[0][0]
            best_root_score = float(feasible_root_states[0][1])
        else:
            selected_root_state = (0, 1)
            best_root_score = 0.0

        selected_cut_edges: set[tuple[int, int]] = set()

        def reconstruct(node_id: int, final_state: tuple[int, int]) -> None:
            child_ids, step_backtraces = dp_backtrace.get(int(node_id), ([], []))
            current_state = final_state
            for child_index in range(len(child_ids) - 1, -1, -1):
                child_id = int(child_ids[child_index])
                step_trace = step_backtraces[child_index]
                previous_state, child_state, action = step_trace[current_state]
                reconstruct(int(child_id), child_state)
                if action == "cut":
                    selected_cut_edges.add((int(node_id), int(child_id)))
                current_state = previous_state

        if selected_root_state in root_table:
            reconstruct(0, selected_root_state)

        component_root_ids = sorted(
            ({0} if int(selected_root_state[1]) == 1 else set()) | {int(child_id) for _, child_id in selected_cut_edges}
        )
        component_entries: list[dict[str, object]] = []
        assigned_nodes: set[int] = set()
        for component_root_id in component_root_ids:
            if int(component_root_id) in assigned_nodes:
                continue
            stack = [int(component_root_id)]
            component_nodes: list[int] = []
            while stack:
                node_id = int(stack.pop())
                if node_id in assigned_nodes:
                    continue
                assigned_nodes.add(node_id)
                component_nodes.append(node_id)
                for child_id in active_children.get(int(node_id), []):
                    if (int(node_id), int(child_id)) in selected_cut_edges:
                        continue
                    stack.append(int(child_id))
            frontier_nodes = [int(node_id) for node_id in component_nodes if int(node_id) in frontier_node_id_set]
            if not frontier_nodes:
                continue
            pattern_ids = tuple(
                sorted(
                    {
                        int(pattern_id)
                        for frontier_node_id in frontier_nodes
                        for pattern_id in pruned_tree[int(frontier_node_id)]["subtree_pattern_ids"]
                    }
                )
            )
            if pattern_ids:
                component_entries.append(
                    {
                        "component_root_id": int(component_root_id),
                        "frontier_node_ids": tuple(int(node_id) for node_id in sorted(frontier_nodes)),
                        "pattern_ids": pattern_ids,
                    }
                )

        if not component_entries:
            component_entries = [
                {
                    "component_root_id": 0,
                    "frontier_node_ids": tuple(int(node_id) for node_id in frontier_node_ids),
                    "pattern_ids": all_pattern_ids,
                }
            ]

        rebalance_initial_component_count = len(component_entries)
        rebalance_split_component_count = 0
        rebalance_merge_count = 0

        def _entry_summary(entry: dict[str, object]) -> dict[str, object]:
            return _aggregate_pattern_summary(tuple(int(pattern_id) for pattern_id in entry["pattern_ids"]))

        def _make_entry(
            *,
            component_root_id: int,
            frontier_node_ids: tuple[int, ...],
            pattern_ids: tuple[int, ...],
        ) -> dict[str, object]:
            summary = _aggregate_pattern_summary(pattern_ids)
            return {
                "component_root_id": int(component_root_id),
                "frontier_node_ids": tuple(int(node_id) for node_id in frontier_node_ids),
                "pattern_ids": tuple(int(pattern_id) for pattern_id in pattern_ids),
                "vector_count": int(summary["vector_count"]),
            }

        def _component_atoms(entry: dict[str, object]) -> list[dict[str, object]]:
            atoms: list[dict[str, object]] = []
            for frontier_node_id in entry.get("frontier_node_ids", ()) or ():
                frontier_pattern_ids = tuple(
                    int(pattern_id)
                    for pattern_id in pruned_tree[int(frontier_node_id)]["subtree_pattern_ids"]
                )
                if not frontier_pattern_ids:
                    continue
                frontier_summary = _aggregate_pattern_summary(frontier_pattern_ids)
                if (
                    int(frontier_summary["vector_count"]) > float(average_target_vector_count)
                    and len(frontier_pattern_ids) > 1
                ):
                    for pattern_id in frontier_pattern_ids:
                        atoms.append(
                            _make_entry(
                                component_root_id=int(frontier_node_id),
                                frontier_node_ids=(int(frontier_node_id),),
                                pattern_ids=(int(pattern_id),),
                            )
                        )
                    continue
                atoms.append(
                    _make_entry(
                        component_root_id=int(frontier_node_id),
                        frontier_node_ids=(int(frontier_node_id),),
                        pattern_ids=frontier_pattern_ids,
                    )
                )
            if atoms:
                return atoms
            return [
                _make_entry(
                    component_root_id=int(entry["component_root_id"]),
                    frontier_node_ids=tuple(int(node_id) for node_id in entry.get("frontier_node_ids", ()) or ()),
                    pattern_ids=tuple(int(pattern_id) for pattern_id in entry["pattern_ids"]),
                )
            ]

        def _pack_atoms(atoms: list[dict[str, object]]) -> list[dict[str, object]]:
            packed: list[dict[str, object]] = []
            ordered_atoms = sorted(
                atoms,
                key=lambda atom: (
                    -int(atom.get("vector_count", 0) or 0),
                    -len(atom.get("pattern_ids", ()) or ()),
                    int(atom.get("component_root_id", 0) or 0),
                    tuple(int(pattern_id) for pattern_id in atom.get("pattern_ids", ()) or ()),
                ),
            )
            for atom in ordered_atoms:
                atom_vector_count = int(atom.get("vector_count", 0) or 0)
                best_index: Optional[int] = None
                best_key: Optional[tuple[float, int, int]] = None
                for index, current in enumerate(packed):
                    projected = int(current.get("vector_count", 0) or 0) + atom_vector_count
                    if float(projected) > float(average_target_vector_count):
                        continue
                    key = (
                        float(average_target_vector_count - projected),
                        -int(current.get("vector_count", 0) or 0),
                        int(index),
                    )
                    if best_key is None or key < best_key:
                        best_key = key
                        best_index = int(index)
                if best_index is None:
                    packed.append(dict(atom))
                    continue
                current = packed[int(best_index)]
                merged_pattern_ids = tuple(
                    sorted(
                        set(int(pattern_id) for pattern_id in current.get("pattern_ids", ()) or ())
                        | set(int(pattern_id) for pattern_id in atom.get("pattern_ids", ()) or ())
                    )
                )
                merged_frontier_node_ids = tuple(
                    sorted(
                        set(int(node_id) for node_id in current.get("frontier_node_ids", ()) or ())
                        | set(int(node_id) for node_id in atom.get("frontier_node_ids", ()) or ())
                    )
                )
                packed[int(best_index)] = _make_entry(
                    component_root_id=min(int(current.get("component_root_id", 0) or 0), int(atom.get("component_root_id", 0) or 0)),
                    frontier_node_ids=merged_frontier_node_ids,
                    pattern_ids=merged_pattern_ids,
                )
            return packed

        rebalanced_entries: list[dict[str, object]] = []
        for entry in component_entries:
            summary = _entry_summary(entry)
            atoms = _component_atoms(entry)
            if int(summary["vector_count"]) > float(average_target_vector_count) and len(atoms) > 1:
                rebalance_split_component_count += 1
                rebalanced_entries.extend(_pack_atoms(atoms))
                continue
            rebalanced_entries.append(
                _make_entry(
                    component_root_id=int(entry["component_root_id"]),
                    frontier_node_ids=tuple(int(node_id) for node_id in entry.get("frontier_node_ids", ()) or ()),
                    pattern_ids=tuple(int(pattern_id) for pattern_id in entry["pattern_ids"]),
                )
            )

        def _merge_entries(left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
            pattern_ids = tuple(
                sorted(
                    set(int(pattern_id) for pattern_id in left.get("pattern_ids", ()) or ())
                    | set(int(pattern_id) for pattern_id in right.get("pattern_ids", ()) or ())
                )
            )
            frontier_node_ids = tuple(
                sorted(
                    set(int(node_id) for node_id in left.get("frontier_node_ids", ()) or ())
                    | set(int(node_id) for node_id in right.get("frontier_node_ids", ()) or ())
                )
            )
            return _make_entry(
                component_root_id=min(int(left.get("component_root_id", 0) or 0), int(right.get("component_root_id", 0) or 0)),
                frontier_node_ids=frontier_node_ids,
                pattern_ids=pattern_ids,
            )

        def _merge_score(left: dict[str, object], right: dict[str, object]) -> tuple[float, float, int, int]:
            left_summary = _entry_summary(left)
            right_summary = _entry_summary(right)
            combined_vectors = int(left_summary["vector_count"]) + int(right_summary["vector_count"])
            size_distance = abs(float(combined_vectors) - float(average_target_vector_count)) / max(float(average_target_vector_count), 1.0)
            acl_distance = 1.0 - _weighted_jaccard_from_sets(
                left_summary["tenant_ids"],
                right_summary["tenant_ids"],
                tenant_weights=tenant_weights,
            )
            workload_distance = 1.0 - _weighted_jaccard_from_dicts(
                left_summary["tenant_query_mass"],
                right_summary["tenant_query_mass"],
            )
            semantic_distance = 0.0
            left_centroid = left_summary["centroid_vector"]
            right_centroid = right_summary["centroid_vector"]
            semantic_affinity = 0.0
            if left_centroid.size and right_centroid.size:
                semantic_affinity = max(0.0, float((np.dot(left_centroid, right_centroid) + 1.0) / 2.0))
                semantic_distance = max(0.0, float(1.0 - semantic_affinity))
            structural_distance = float((acl_distance + workload_distance + semantic_distance) / 3.0)
            shared_workload_mass = sum(
                min(
                    float(left_summary["tenant_query_mass"].get(int(tenant_id), 0.0)),
                    float(right_summary["tenant_query_mass"].get(int(tenant_id), 0.0)),
                )
                for tenant_id in set(left_summary["tenant_query_mass"]) | set(right_summary["tenant_query_mass"])
            )
            route_fanout_gain = float(
                shared_workload_mass / max(float(global_summary["partition_query_mass"]), 1e-9)
            )
            workload_semantic_affinity = float(route_fanout_gain * semantic_affinity)
            semantic_locality_gain = float(
                math.log1p(shared_workload_mass) * semantic_affinity / math.log1p(max(float(global_summary["partition_query_mass"]), 1.0))
            )
            oversized_penalty = max(
                0.0,
                float(combined_vectors) / max(float(average_target_vector_count) * 1.25, 1.0) - 1.0,
            )
            unrelated_acl_penalty = float(acl_distance * max(0.0, 1.0 - route_fanout_gain))
            return (
                float(
                    0.45 * size_distance
                    + 0.20 * structural_distance
                    + 0.35 * unrelated_acl_penalty
                    + oversized_penalty
                    - 1.50 * workload_semantic_affinity
                    - 0.75 * semantic_locality_gain
                ),
                float(size_distance),
                int(combined_vectors),
                min(int(left.get("component_root_id", 0) or 0), int(right.get("component_root_id", 0) or 0)),
            )

        while len(rebalanced_entries) > int(effective_partition_budget):
            rebalanced_entries.sort(
                key=lambda entry: (
                    int(entry.get("vector_count", 0) or 0),
                    len(entry.get("pattern_ids", ()) or ()),
                    int(entry.get("component_root_id", 0) or 0),
                )
            )
            left = rebalanced_entries.pop(0)
            best_index = 0
            best_score: Optional[tuple[float, float, int, int]] = None
            for index, right in enumerate(rebalanced_entries):
                score = _merge_score(left, right)
                if best_score is None or score < best_score:
                    best_score = score
                    best_index = int(index)
            right = rebalanced_entries.pop(best_index)
            rebalanced_entries.append(_merge_entries(left, right))
            rebalance_merge_count += 1

        component_pattern_ids: list[tuple[int, tuple[int, ...]]] = [
            (
                int(entry.get("component_root_id", 0) or 0),
                tuple(int(pattern_id) for pattern_id in entry.get("pattern_ids", ()) or ()),
            )
            for entry in rebalanced_entries
            if entry.get("pattern_ids")
        ]
        semantic_bucket_count = min(int(requested_partition_count), len(all_pattern_ids))
        semantic_primary_used = False
        semantic_split_count = 0
        semantic_atom_count = 0
        semantic_atom_split_pattern_count = 0
        component_summaries_override: Optional[list[tuple[int, dict[str, object]]]] = None
        if semantic_bucket_count > 0:
            workload_direction = np.zeros(0, dtype=np.float32)
            weighted_query_sum = None
            total_query_weight = 0.0
            for query in queries:
                query_vector = _normalize_vector(_parse_vector(query.query_vector))
                if not query_vector.size:
                    continue
                weight = max(float(query.weight), 1e-9)
                weighted = query_vector * weight
                weighted_query_sum = weighted if weighted_query_sum is None else weighted_query_sum + weighted
                total_query_weight += weight
            if weighted_query_sum is not None and total_query_weight > 0.0:
                workload_direction = _normalize_vector(weighted_query_sum / float(total_query_weight))

            def _document_chunk_summary(
                docs: list[tuple[int, int, np.ndarray]],
            ) -> tuple[int, int, np.ndarray, float]:
                vector_count = int(sum(max(1, int(block_count)) for _, block_count, _ in docs))
                document_count = int(len(docs))
                weighted_centroid = None
                for _, block_count, vector in docs:
                    normalized = _normalize_vector(vector)
                    if not normalized.size:
                        continue
                    weighted = normalized * float(max(1, int(block_count)))
                    weighted_centroid = weighted if weighted_centroid is None else weighted_centroid + weighted
                centroid = np.zeros(0, dtype=np.float32)
                if weighted_centroid is not None and vector_count > 0:
                    centroid = _normalize_vector(weighted_centroid / float(vector_count))
                dispersion = 0.0
                if centroid.size and docs:
                    total_weight = 0.0
                    weighted_distance = 0.0
                    for _, block_count, vector in docs:
                        normalized = _normalize_vector(vector)
                        if not normalized.size:
                            continue
                        weight = float(max(1, int(block_count)))
                        total_weight += weight
                        weighted_distance += weight * max(0.0, float(1.0 - np.dot(normalized, centroid)))
                    if total_weight > 0.0:
                        dispersion = float(weighted_distance / total_weight)
                return int(document_count), int(vector_count), centroid, float(dispersion)

            def _split_document_chunk(
                docs: list[tuple[int, int, np.ndarray]],
            ) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
                if len(docs) <= 1:
                    return docs, []
                _, vector_count, centroid, _ = _document_chunk_summary(docs)
                direction = np.zeros(0, dtype=np.float32)
                if centroid.size:
                    farthest_vector = None
                    farthest_distance = -1.0
                    for _, _, vector in docs:
                        normalized = _normalize_vector(vector)
                        if not normalized.size:
                            continue
                        distance = max(0.0, float(1.0 - np.dot(normalized, centroid)))
                        if distance > farthest_distance:
                            farthest_distance = float(distance)
                            farthest_vector = normalized
                    if farthest_vector is not None:
                        direction = _normalize_vector(farthest_vector - centroid)
                if not direction.size and workload_direction.size:
                    direction = workload_direction
                if not direction.size and centroid.size:
                    direction = centroid

                if direction.size:
                    ordered_docs = sorted(
                        docs,
                        key=lambda item: (
                            float(np.dot(_normalize_vector(item[2]), direction)),
                            int(item[0]),
                        ),
                    )
                else:
                    ordered_docs = sorted(docs, key=lambda item: int(item[0]))

                half_vectors = max(1.0, float(vector_count) / 2.0)
                left: list[tuple[int, int, np.ndarray]] = []
                right: list[tuple[int, int, np.ndarray]] = []
                running_vectors = 0
                for item in ordered_docs:
                    if running_vectors < half_vectors or not left:
                        left.append(item)
                        running_vectors += max(1, int(item[1]))
                    else:
                        right.append(item)
                if not right and len(left) > 1:
                    right = left[len(left) // 2:]
                    left = left[:len(left) // 2]
                return left, right

            def _pattern_semantic_atoms(pattern: ACLLogicalPattern) -> list[dict[str, object]]:
                raw_docs = pattern.metadata.get("_member_document_vectors", []) or []
                docs: list[tuple[int, int, np.ndarray]] = [
                    (int(document_id), int(block_count), _normalize_vector(vector))
                    for document_id, block_count, vector in raw_docs
                ]
                centroid = _normalize_vector(_parse_vector(pattern.metadata.get("representative_centroid", [])))
                dispersion = 0.0
                if docs:
                    _, _, doc_centroid, doc_dispersion = _document_chunk_summary(docs)
                    if doc_centroid.size:
                        centroid = doc_centroid
                    dispersion = float(doc_dispersion)
                return [
                    {
                        "pattern_id": int(pattern.pattern_id),
                        "atom_id": f"{int(pattern.pattern_id)}:0",
                        "pattern_ids": (int(pattern.pattern_id),),
                        "document_ids": tuple(int(document_id) for document_id in pattern.document_ids),
                        "document_count": int(pattern.document_count),
                        "vector_count": int(pattern.vector_count),
                        "tenant_ids": tuple(int(tenant_id) for tenant_id in pattern.tenant_ids),
                        "ordered_tenant_ids": tuple(int(tenant_id) for tenant_id in pattern.ordered_tenant_ids),
                        "centroid": centroid,
                        "dispersion": float(dispersion),
                        "pattern_document_count": int(pattern.document_count),
                        "pattern_vector_count": int(pattern.vector_count),
                    }
                ]

            semantic_atoms: list[dict[str, object]] = []
            for pattern in logical_patterns:
                pattern_atoms = _pattern_semantic_atoms(pattern)
                if len(pattern_atoms) > 1:
                    semantic_atom_split_pattern_count += 1
                for atom in pattern_atoms:
                    centroid = atom["centroid"]
                    semantic_key = 0.0
                    if centroid.size and workload_direction.size:
                        semantic_key = float(np.dot(centroid, workload_direction))
                    elif centroid.size:
                        semantic_key = float(centroid[0])
                    atom["semantic_key"] = float(semantic_key)
                    semantic_atoms.append(atom)

            semantic_atom_count = int(len(semantic_atoms))
            semantic_bucket_count = min(int(requested_partition_count), len(semantic_atoms))
            semantic_atoms.sort(
                key=lambda atom: (
                    -float(atom.get("semantic_key", 0.0) or 0.0),
                    int(atom.get("pattern_id", 0) or 0),
                    str(atom.get("atom_id", "")),
                )
            )

            bucket_vectors = [0 for _ in range(int(semantic_bucket_count))]
            bucket_atoms: list[list[dict[str, object]]] = [[] for _ in range(int(semantic_bucket_count))]
            bucket_centroids: list[np.ndarray] = [np.zeros(0, dtype=np.float32) for _ in range(int(semantic_bucket_count))]
            bucket_tenant_ids: list[set[int]] = [set() for _ in range(int(semantic_bucket_count))]

            if semantic_atoms and semantic_bucket_count > 0:
                if semantic_bucket_count == 1:
                    seed_positions = [0]
                else:
                    seed_positions = [
                        int(round(float(index) * float(len(semantic_atoms) - 1) / float(max(1, semantic_bucket_count - 1))))
                        for index in range(int(semantic_bucket_count))
                    ]
                seeded_positions = set(seed_positions)
                for bucket_index, atom_index in enumerate(seed_positions):
                    atom = semantic_atoms[int(atom_index)]
                    bucket_atoms[int(bucket_index)].append(atom)
                    bucket_vectors[int(bucket_index)] += int(atom.get("vector_count", 0) or 0)
                    bucket_centroids[int(bucket_index)] = atom["centroid"]
                    bucket_tenant_ids[int(bucket_index)].update(int(tenant_id) for tenant_id in atom.get("tenant_ids", ()) or ())

                for atom_index, atom in enumerate(semantic_atoms):
                    if int(atom_index) in seeded_positions:
                        continue
                    atom_centroid = atom["centroid"]
                    atom_vectors = int(atom.get("vector_count", 0) or 0)
                    atom_tenant_ids = set(int(tenant_id) for tenant_id in atom.get("tenant_ids", ()) or ())
                    best_index = 0
                    best_score: Optional[tuple[float, float, int]] = None
                    for bucket_index in range(int(semantic_bucket_count)):
                        bucket_centroid = bucket_centroids[int(bucket_index)]
                        semantic_distance = 1.0
                        if atom_centroid.size and bucket_centroid.size:
                            semantic_distance = max(0.0, float(1.0 - np.dot(atom_centroid, bucket_centroid)))
                        acl_distance = 1.0
                        if bucket_tenant_ids[int(bucket_index)]:
                            acl_distance = 1.0 - _weighted_jaccard_from_sets(
                                atom_tenant_ids,
                                bucket_tenant_ids[int(bucket_index)],
                                tenant_weights=tenant_weights,
                            )
                        projected_vectors = int(bucket_vectors[int(bucket_index)] + atom_vectors)
                        size_distance = abs(float(projected_vectors) - float(average_target_vector_count)) / max(float(average_target_vector_count), 1.0)
                        overflow = max(0.0, float(projected_vectors) / max(float(average_target_vector_count) * 1.50, 1.0) - 1.0)
                        score = (
                            float(0.70 * semantic_distance + 0.20 * acl_distance + 0.10 * size_distance + overflow),
                            float(size_distance),
                            int(bucket_index),
                        )
                        if best_score is None or score < best_score:
                            best_score = score
                            best_index = int(bucket_index)
                    bucket_atoms[best_index].append(atom)
                    old_vectors = int(bucket_vectors[best_index])
                    new_vectors = old_vectors + atom_vectors
                    if atom_centroid.size:
                        old_centroid = bucket_centroids[best_index]
                        if old_centroid.size and old_vectors > 0:
                            bucket_centroids[best_index] = _normalize_vector(
                                (old_centroid * float(old_vectors) + atom_centroid * float(atom_vectors))
                                / float(max(new_vectors, 1))
                            )
                        else:
                            bucket_centroids[best_index] = atom_centroid
                    bucket_vectors[best_index] = int(new_vectors)
                    bucket_tenant_ids[best_index].update(atom_tenant_ids)

            def _aggregate_atom_summary(atoms: list[dict[str, object]]) -> dict[str, object]:
                pattern_ids = tuple(sorted(set(int(atom["pattern_id"]) for atom in atoms)))
                document_ids = tuple(sorted(set(int(document_id) for atom in atoms for document_id in atom.get("document_ids", ()) or ())))
                tenant_ids = tuple(sorted(set(int(tenant_id) for atom in atoms for tenant_id in atom.get("tenant_ids", ()) or ())))
                vector_count = int(sum(int(atom.get("vector_count", 0) or 0) for atom in atoms))
                document_count = int(len(document_ids))
                tenant_doc_counts: dict[int, int] = Counter()
                tenant_query_mass: dict[int, float] = Counter()
                pattern_document_counts: dict[int, int] = Counter()
                pattern_vector_counts: dict[int, int] = Counter()
                document_pattern_pairs: list[tuple[int, int]] = []
                centroid_numerator = None
                total_centroid_weight = 0.0
                atom_metadata: list[dict[str, object]] = []
                for atom in atoms:
                    pattern_id = int(atom["pattern_id"])
                    atom_document_count = int(atom.get("document_count", 0) or 0)
                    atom_vector_count = int(atom.get("vector_count", 0) or 0)
                    pattern = patterns_by_id[int(pattern_id)]
                    pattern_document_counts[int(pattern_id)] += int(atom_document_count)
                    pattern_vector_counts[int(pattern_id)] += int(atom_vector_count)
                    for document_id in atom.get("document_ids", ()) or ():
                        document_pattern_pairs.append((int(document_id), int(pattern_id)))
                    for tenant_id in pattern.tenant_ids:
                        tenant_doc_counts[int(tenant_id)] += int(atom_document_count)
                    mass_scale = float(atom_document_count) / max(float(pattern.document_count), 1.0)
                    for tenant_id, value in (pattern.metadata.get("tenant_query_mass", {}) or {}).items():
                        tenant_query_mass[int(tenant_id)] += float(value) * mass_scale
                    centroid = atom["centroid"]
                    if centroid.size and atom_vector_count > 0:
                        weighted = centroid * float(atom_vector_count)
                        centroid_numerator = weighted if centroid_numerator is None else centroid_numerator + weighted
                        total_centroid_weight += float(atom_vector_count)
                    atom_metadata.append(
                        {
                            "pattern_id": int(pattern_id),
                            "atom_id": str(atom.get("atom_id", f"{pattern_id}:0")),
                            "document_count": int(atom_document_count),
                            "vector_count": int(atom_vector_count),
                            "centroid": atom["centroid"].astype(float).tolist() if atom["centroid"].size else [],
                            "dispersion": float(atom.get("dispersion", 0.0) or 0.0),
                        }
                    )

                centroid = np.zeros(0, dtype=np.float32)
                if centroid_numerator is not None and total_centroid_weight > 0.0:
                    centroid = _normalize_vector(centroid_numerator / float(total_centroid_weight))
                dispersion = 0.0
                if centroid.size and atom_metadata:
                    weighted_distance = 0.0
                    total_weight = 0.0
                    for atom in atoms:
                        atom_centroid = atom["centroid"]
                        atom_vector_count = int(atom.get("vector_count", 0) or 0)
                        if not atom_centroid.size or atom_vector_count <= 0:
                            continue
                        total_weight += float(atom_vector_count)
                        weighted_distance += float(atom_vector_count) * max(0.0, float(1.0 - np.dot(atom_centroid, centroid)))
                    if total_weight > 0.0:
                        dispersion = float(weighted_distance / total_weight)

                acl_union = set(tenant_ids)
                impurity = 0.0
                total_weight = 0.0
                for atom in atoms:
                    pattern = patterns_by_id[int(atom["pattern_id"])]
                    weight = float(atom.get("vector_count", 0) or 0)
                    total_weight += weight
                    impurity += weight * float(
                        len(acl_union.difference(set(int(value) for value in pattern.tenant_ids)))
                        / max(len(acl_union), 1)
                    )
                if total_weight > 0.0:
                    impurity = float(impurity / total_weight)

                normalized_tenant_query_mass = {
                    str(tenant_id): float(value)
                    for tenant_id, value in sorted(tenant_query_mass.items())
                }
                route_core = tuple(
                    int(tenant_id)
                    for tenant_id in sorted(
                        tenant_ids,
                        key=lambda tenant_id: (
                            -float(tenant_query_mass.get(int(tenant_id), 0.0)),
                            -int(tenant_doc_counts.get(int(tenant_id), 0)),
                            int(tenant_id),
                        ),
                    )[:3]
                )
                return {
                    "pattern_ids": pattern_ids,
                    "pattern_id_set": frozenset(int(pattern_id) for pattern_id in pattern_ids),
                    "document_ids": document_ids,
                    "tenant_ids": tenant_ids,
                    "tenant_doc_counts": {int(key): int(value) for key, value in tenant_doc_counts.items()},
                    "tenant_query_mass": {int(key): float(value) for key, value in tenant_query_mass.items()},
                    "vector_count": int(vector_count),
                    "document_count": int(document_count),
                    "pattern_count": len(pattern_ids),
                    "representative_centroid": centroid.astype(float).tolist(),
                    "centroid_vector": centroid,
                    "centroid_dispersion": float(dispersion),
                    "acl_impurity": float(impurity),
                    "entry_tenant_ids": tuple(sorted(tenant_ids)),
                    "partition_query_mass": float(sum(float(value) for value in tenant_query_mass.values())),
                    "normalized_tenant_query_mass": normalized_tenant_query_mass,
                    "route_core": route_core,
                    "document_pattern_pairs": tuple(sorted(set(document_pattern_pairs))),
                    "pattern_document_counts": {str(key): int(value) for key, value in sorted(pattern_document_counts.items())},
                    "pattern_vector_counts": {str(key): int(value) for key, value in sorted(pattern_vector_counts.items())},
                    "semantic_atoms": atom_metadata,
                }

            semantic_component_summaries = [
                (int(index), _aggregate_atom_summary(atoms))
                for index, atoms in enumerate(bucket_atoms)
                if atoms
            ]
            if semantic_component_summaries:
                component_summaries_override = semantic_component_summaries
                component_pattern_ids = [
                    (
                        int(component_root_id),
                        tuple(int(pattern_id) for pattern_id in summary["pattern_ids"]),
                    )
                    for component_root_id, summary in semantic_component_summaries
                ]
                semantic_primary_used = True
                semantic_split_count = int(len(semantic_component_summaries))

        partitions: list[WorkloadAwarePartition] = []
        if component_summaries_override is not None:
            component_summaries = component_summaries_override
        else:
            component_summaries = [
                (int(component_root_id), _aggregate_pattern_summary(pattern_ids))
                for component_root_id, pattern_ids in component_pattern_ids
            ]
        component_summaries.sort(
            key=lambda item: (
                -int(item[1]["vector_count"]),
                -int(item[1]["pattern_count"]),
                int(item[0]),
            )
        )
        for partition_index, (component_root_id, summary) in enumerate(component_summaries):
            if (
                max_partition_vector_count is not None
                and int(summary["vector_count"]) > int(max_partition_vector_count)
                and len(summary["pattern_ids"]) > 1
            ):
                # Fallback safety: keep large components as-is for now, exact ACL filtering still preserves correctness.
                pass
            partition_id = f"p{partition_index}"
            # Fine-grained pattern accelerators fragment storage again; hot tenants are handled
            # by tenant-level overlays selected after the DP partitioning step.
            accelerator_patterns: list[dict[str, object]] = []
            document_pattern_pairs = sorted(
                (
                    int(document_id),
                    int(pattern_id),
                )
                for document_id, pattern_id in (
                    summary.get("document_pattern_pairs", ()) or ()
                )
            )
            if not document_pattern_pairs:
                document_pattern_pairs = sorted(
                    (
                        int(document_id),
                        int(pattern.pattern_id),
                    )
                    for pattern_id in summary["pattern_ids"]
                    for pattern in [patterns_by_id[int(pattern_id)]]
                    for document_id in pattern.document_ids
                )
            tenant_doc_counts = dict(summary["tenant_doc_counts"])
            total_documents = max(int(summary["document_count"]), 1)
            tenant_densities = {
                str(tenant_id): float(int(count) / total_documents)
                for tenant_id, count in sorted(tenant_doc_counts.items())
            }
            partitions.append(
                WorkloadAwarePartition(
                    partition_id=partition_id,
                    table_name=get_partition_table_name(partition_id),
                    document_ids=tuple(int(document_id) for document_id in summary["document_ids"]),
                    tenant_ids=tuple(int(tenant_id) for tenant_id in summary["tenant_ids"]),
                    vector_count=int(summary["vector_count"]),
                    logical_pattern_ids=tuple(int(pattern_id) for pattern_id in summary["pattern_ids"]),
                    metadata={
                        "tenant_densities": tenant_densities,
                        "tenant_query_mass": dict(summary["normalized_tenant_query_mass"]),
                        "entry_tenant_ids": [int(tenant_id) for tenant_id in summary["entry_tenant_ids"]],
                        "primary_anchor": int(summary["route_core"][0]) if summary["route_core"] else -1,
                        "logical_pattern_count": int(summary["pattern_count"]),
                        "pattern_document_count": int(summary["document_count"]),
                        "partition_query_mass": float(summary["partition_query_mass"]),
                        "representative_centroid": list(summary["representative_centroid"]),
                        "storage_layout_version": 3,
                        "document_pattern_pairs": [
                            [int(document_id), int(pattern_id)]
                            for document_id, pattern_id in document_pattern_pairs
                        ],
                        "accelerator_patterns": accelerator_patterns,
                        "ordered_acl_patterns": [
                            [int(tenant_id) for tenant_id in patterns_by_id[int(pattern_id)].ordered_tenant_ids]
                            for pattern_id in summary["pattern_ids"]
                        ],
                        "planning_node_id": int(component_root_id),
                        "acl_impurity": float(summary["acl_impurity"]),
                        "centroid_dispersion": float(summary["centroid_dispersion"]),
                        "route_core": [int(value) for value in summary["route_core"]],
                        "pattern_document_counts": dict(summary.get("pattern_document_counts", {}) or {}),
                        "pattern_vector_counts": dict(summary.get("pattern_vector_counts", {}) or {}),
                        "semantic_atoms": list(summary.get("semantic_atoms", []) or []),
                    },
                )
            )

        partition_by_pattern: dict[int, str] = {
            int(pattern_id): str(partition.partition_id)
            for partition in partitions
            for pattern_id in partition.logical_pattern_ids
        }
        for pattern in logical_patterns:
            pattern.metadata["partition_id"] = partition_by_pattern.get(int(pattern.pattern_id))

        return partitions, {
            "dp_target_partition_count": int(target_partition_count),
            "dp_selected_cut_count": int(len(partitions)),
            "dp_cut_edge_count": int(len(selected_cut_edges)),
            "dp_cut_edges": [
                [int(parent_id), int(child_id)]
                for parent_id, child_id in sorted(selected_cut_edges)
            ],
            "dp_average_target_vectors": float(average_target_vector_count),
            "dp_effective_partition_count": int(len(partitions)),
            "dp_best_root_score": float(best_root_score),
            "dp_max_feasible_partition_count": int(max_feasible_partition_count),
            "dp_frontier_node_count": int(len(frontier_node_ids)),
            "dp_selected_partition_budget": int(effective_partition_budget),
            "dp_selected_root_state": [int(selected_root_state[0]), int(selected_root_state[1])],
            "dp_rebalance_initial_component_count": int(rebalance_initial_component_count),
            "dp_rebalance_split_component_count": int(rebalance_split_component_count),
            "dp_rebalance_merge_count": int(rebalance_merge_count),
            "dp_rebalance_final_component_count": int(len(component_pattern_ids)),
            "semantic_primary_partitioning_used": bool(semantic_primary_used),
            "semantic_primary_partition_count": int(semantic_split_count),
            "semantic_atom_count": int(semantic_atom_count),
            "semantic_atom_split_pattern_count": int(semantic_atom_split_pattern_count),
        }

    def _build_prefix_dag(
        self,
        logical_patterns: list[ACLLogicalPattern],
        *,
        tenant_order: tuple[int, ...],
        tenant_query_weights: dict[int, float],
        supplemental_edge_penalty: float,
        supplemental_edge_gain_threshold: float,
    ) -> tuple[list[PrefixDagNode], dict[str, object]]:
        nodes: dict[int, PrefixDagNode] = {0: PrefixDagNode(node_id=0, prefix_tenants=())}
        prefix_to_node_id: dict[tuple[int, ...], int] = {(): 0}
        next_node_id = 1
        pattern_to_node_id: dict[int, int] = {}
        tenant_reachable_patterns: dict[int, set[int]] = defaultdict(set)
        supplemental_edge_count = 0

        for tenant_id in tenant_order:
            prefix = (int(tenant_id),)
            singleton_id = prefix_to_node_id.get(prefix)
            if singleton_id is None:
                singleton_id = next_node_id
                next_node_id += 1
                nodes[singleton_id] = PrefixDagNode(node_id=singleton_id, prefix_tenants=prefix)
                prefix_to_node_id[prefix] = singleton_id
            nodes[0].children[int(tenant_id)] = singleton_id

        for pattern in logical_patterns:
            current_node_id = 0
            nodes[current_node_id].document_count += int(pattern.document_count)
            prefix: list[int] = []
            for tenant_id in pattern.ordered_tenant_ids:
                prefix.append(int(tenant_id))
                prefix_tuple = tuple(prefix)
                next_node_id_for_prefix = prefix_to_node_id.get(prefix_tuple)
                if next_node_id_for_prefix is None:
                    next_node_id_for_prefix = next_node_id
                    next_node_id += 1
                    nodes[next_node_id_for_prefix] = PrefixDagNode(
                        node_id=next_node_id_for_prefix,
                        prefix_tenants=prefix_tuple,
                    )
                    prefix_to_node_id[prefix_tuple] = next_node_id_for_prefix
                nodes[current_node_id].children[int(tenant_id)] = next_node_id_for_prefix
                current_node_id = next_node_id_for_prefix
                nodes[current_node_id].document_count += int(pattern.document_count)
            nodes[current_node_id].terminal_pattern_ids.add(int(pattern.pattern_id))
            nodes[current_node_id].terminal_document_count += int(pattern.document_count)
            pattern_to_node_id[int(pattern.pattern_id)] = int(current_node_id)

        for pattern in logical_patterns:
            entry_tenants = set()
            first_tenant = int(pattern.ordered_tenant_ids[0])
            entry_tenants.add(first_tenant)
            tenant_reachable_patterns[first_tenant].add(int(pattern.pattern_id))
            for tenant_id in pattern.ordered_tenant_ids[1:]:
                singleton_node_id = prefix_to_node_id.get((int(tenant_id),))
                if singleton_node_id is None:
                    continue
                current_fanout = len(tenant_reachable_patterns[int(tenant_id)])
                gain = (
                    float(tenant_query_weights.get(int(tenant_id), 1.0))
                    * math.log1p(float(pattern.document_count))
                    / (1.0 + float(current_fanout))
                    - float(supplemental_edge_penalty)
                )
                if gain <= float(supplemental_edge_gain_threshold):
                    continue
                nodes[singleton_node_id].supplemental_pattern_ids.add(int(pattern.pattern_id))
                tenant_reachable_patterns[int(tenant_id)].add(int(pattern.pattern_id))
                entry_tenants.add(int(tenant_id))
                supplemental_edge_count += 1
            pattern.entry_tenant_ids = tuple(sorted(entry_tenants))
            pattern.metadata["dag_node_id"] = int(pattern_to_node_id[int(pattern.pattern_id)])
            pattern.metadata["entry_tenant_ids"] = [int(tenant_id) for tenant_id in pattern.entry_tenant_ids]

        dag_nodes = [nodes[node_id] for node_id in sorted(nodes)]
        main_edge_count = sum(len(node.children) for node in dag_nodes)
        return dag_nodes, {
            "dag_root_children": len(nodes[0].children),
            "dag_main_edge_count": int(main_edge_count),
            "dag_supplemental_edge_count": int(supplemental_edge_count),
            "tenant_entry_pattern_counts": {
                str(tenant_id): len(pattern_ids)
                for tenant_id, pattern_ids in sorted(tenant_reachable_patterns.items())
            },
        }

    def _pattern_importance(self, pattern: ACLLogicalPattern) -> float:
        return float(pattern.document_count) + 0.1 * float(pattern.metadata.get("pattern_query_mass", 0.0))

    def _cluster_from_pattern(self, pattern: ACLLogicalPattern) -> dict[str, object]:
        tenant_query_mass = {
            int(tenant_id): float(value)
            for tenant_id, value in (pattern.metadata.get("tenant_query_mass", {}) or {}).items()
        }
        return {
            "pattern_ids": {int(pattern.pattern_id)},
            "document_ids": set(int(document_id) for document_id in pattern.document_ids),
            "tenant_ids": set(int(tenant_id) for tenant_id in pattern.tenant_ids),
            "tenant_doc_counts": {int(tenant_id): int(pattern.document_count) for tenant_id in pattern.tenant_ids},
            "tenant_query_mass": tenant_query_mass,
            "entry_tenant_ids": set(int(tenant_id) for tenant_id in pattern.entry_tenant_ids),
            "vector_count": int(pattern.vector_count),
            "document_count": int(pattern.document_count),
            "pattern_count": 1,
            "primary_anchor": int(pattern.ordered_tenant_ids[0]),
            "representative_centroid": _parse_vector(pattern.metadata.get("representative_centroid", [])),
            "query_mass": float(pattern.metadata.get("pattern_query_mass", 0.0)),
        }

    def _merge_cluster_meta(self, left: dict[str, object], right: dict[str, object]) -> dict[str, object]:
        combined_document_count = int(left["document_count"]) + int(right["document_count"])
        combined_vector_count = int(left["vector_count"]) + int(right["vector_count"])
        tenant_doc_counts = dict(left["tenant_doc_counts"])
        for tenant_id, count in right["tenant_doc_counts"].items():
            tenant_doc_counts[int(tenant_id)] = int(tenant_doc_counts.get(int(tenant_id), 0)) + int(count)
        tenant_query_mass = dict(left["tenant_query_mass"])
        for tenant_id, value in right["tenant_query_mass"].items():
            tenant_query_mass[int(tenant_id)] = float(tenant_query_mass.get(int(tenant_id), 0.0) + float(value))
        left_centroid = _parse_vector(left.get("representative_centroid", []))
        right_centroid = _parse_vector(right.get("representative_centroid", []))
        if left_centroid.size and right_centroid.size:
            centroid = _normalize_vector(
                (left_centroid * max(int(left["document_count"]), 1) + right_centroid * max(int(right["document_count"]), 1))
                / max(combined_document_count, 1)
            )
        else:
            centroid = left_centroid if left_centroid.size else right_centroid
        if not centroid.size:
            centroid = np.zeros(0, dtype=np.float32)
        dominant_tenant = max(
            tenant_doc_counts,
            key=lambda tenant_id: (tenant_doc_counts[int(tenant_id)], tenant_query_mass.get(int(tenant_id), 0.0), -int(tenant_id)),
        )
        return {
            "pattern_ids": set(left["pattern_ids"]) | set(right["pattern_ids"]),
            "document_ids": set(left["document_ids"]) | set(right["document_ids"]),
            "tenant_ids": set(left["tenant_ids"]) | set(right["tenant_ids"]),
            "tenant_doc_counts": tenant_doc_counts,
            "tenant_query_mass": tenant_query_mass,
            "entry_tenant_ids": set(left["entry_tenant_ids"]) | set(right["entry_tenant_ids"]),
            "vector_count": combined_vector_count,
            "document_count": combined_document_count,
            "pattern_count": int(left["pattern_count"]) + int(right["pattern_count"]),
            "primary_anchor": int(dominant_tenant),
            "representative_centroid": centroid,
            "query_mass": float(left.get("query_mass", 0.0)) + float(right.get("query_mass", 0.0)),
        }

    def _cluster_densities(self, meta: dict[str, object]) -> dict[int, float]:
        total_documents = max(int(meta["document_count"]), 1)
        return {
            int(tenant_id): float(int(count) / total_documents)
            for tenant_id, count in sorted(meta["tenant_doc_counts"].items())
        }

    def _density_safe(self, meta: dict[str, object], threshold: float) -> bool:
        densities = self._cluster_densities(meta)
        return all(float(density) >= float(threshold) for density in densities.values())

    def _average_density(self, meta: dict[str, object], tenant_weights: dict[int, float]) -> float:
        densities = self._cluster_densities(meta)
        if not densities:
            return 0.0
        numerator = 0.0
        denominator = 0.0
        for tenant_id, density in densities.items():
            weight = float(tenant_weights.get(int(tenant_id), 1.0))
            numerator += weight * float(density)
            denominator += weight
        if denominator <= 0.0:
            return 0.0
        return float(numerator / denominator)

    def _can_merge_clusters(
        self,
        left: dict[str, object],
        right: dict[str, object],
        *,
        safe_density_threshold: float,
        max_partition_vector_count: Optional[int],
    ) -> tuple[bool, dict[str, object] | None]:
        merged = self._merge_cluster_meta(left, right)
        if max_partition_vector_count is not None and int(merged["vector_count"]) > int(max_partition_vector_count):
            return False, None
        if not self._density_safe(merged, safe_density_threshold):
            return False, None
        return True, merged

    def _cluster_merge_score(
        self,
        left: dict[str, object],
        right: dict[str, object],
        *,
        tenant_weights: dict[int, float],
    ) -> float:
        merged = self._merge_cluster_meta(left, right)
        tenant_overlap = _weighted_jaccard_from_sets(
            left["tenant_ids"],
            right["tenant_ids"],
            tenant_weights=tenant_weights,
        )
        query_overlap = _weighted_jaccard_from_dicts(
            left["tenant_query_mass"],
            right["tenant_query_mass"],
        )
        density_score = self._average_density(merged, tenant_weights)
        anchor_bonus = 0.15 if int(left["primary_anchor"]) == int(right["primary_anchor"]) else 0.0
        pattern_penalty = 0.01 * max(0, int(merged["pattern_count"]) - 4)
        return float(0.45 * tenant_overlap + 0.25 * query_overlap + 0.25 * density_score + anchor_bonus - pattern_penalty)

    def _merge_logical_patterns_to_partitions(
        self,
        *,
        logical_patterns: list[ACLLogicalPattern],
        tenant_weights: dict[int, float],
        min_pattern_support: int,
        min_pattern_query_mass: float,
        safe_density_threshold: float,
        target_partition_count: Optional[int],
        max_partition_vector_count: Optional[int],
    ) -> list[WorkloadAwarePartition]:
        if not logical_patterns:
            return []

        patterns_by_id = {int(pattern.pattern_id): pattern for pattern in logical_patterns}
        ordered_patterns = sorted(
            logical_patterns,
            key=lambda pattern: (-self._pattern_importance(pattern), len(pattern.tenant_ids), pattern.pattern_id),
        )

        seed_pattern_ids = {
            int(pattern.pattern_id)
            for pattern in ordered_patterns
            if int(pattern.document_count) >= int(min_pattern_support)
            or float(pattern.metadata.get("pattern_query_mass", 0.0)) >= float(min_pattern_query_mass)
        }
        if not seed_pattern_ids:
            seed_pattern_ids.add(int(ordered_patterns[0].pattern_id))

        clusters: list[dict[str, object]] = []
        assigned_pattern_ids: set[int] = set()
        for pattern in ordered_patterns:
            pattern_id = int(pattern.pattern_id)
            if pattern_id in assigned_pattern_ids:
                continue
            pattern_cluster = self._cluster_from_pattern(pattern)
            if pattern_id in seed_pattern_ids or not clusters:
                clusters.append(pattern_cluster)
                assigned_pattern_ids.add(pattern_id)
                continue

            best_cluster_index = None
            best_merged = None
            best_score = -1e18
            for cluster_index, cluster_meta in enumerate(clusters):
                can_merge, merged = self._can_merge_clusters(
                    cluster_meta,
                    pattern_cluster,
                    safe_density_threshold=float(safe_density_threshold),
                    max_partition_vector_count=max_partition_vector_count,
                )
                if not can_merge or merged is None:
                    continue
                score = self._cluster_merge_score(cluster_meta, pattern_cluster, tenant_weights=tenant_weights)
                if score > best_score:
                    best_score = score
                    best_cluster_index = cluster_index
                    best_merged = merged
            if best_cluster_index is None or best_merged is None:
                clusters.append(pattern_cluster)
            else:
                clusters[best_cluster_index] = best_merged
            assigned_pattern_ids.add(pattern_id)

        effective_target_partition_count = None
        if target_partition_count is not None:
            effective_target_partition_count = max(1, int(target_partition_count))

        while effective_target_partition_count is not None and len(clusters) > effective_target_partition_count:
            best_pair = None
            best_merged = None
            best_score = -1e18
            for left_index in range(len(clusters)):
                for right_index in range(left_index + 1, len(clusters)):
                    can_merge, merged = self._can_merge_clusters(
                        clusters[left_index],
                        clusters[right_index],
                        safe_density_threshold=float(safe_density_threshold),
                        max_partition_vector_count=max_partition_vector_count,
                    )
                    if not can_merge or merged is None:
                        continue
                    score = self._cluster_merge_score(
                        clusters[left_index],
                        clusters[right_index],
                        tenant_weights=tenant_weights,
                    )
                    if score > best_score:
                        best_pair = (left_index, right_index)
                        best_merged = merged
                        best_score = score
            if best_pair is None or best_merged is None:
                break
            left_index, right_index = best_pair
            clusters[left_index] = best_merged
            clusters.pop(right_index)

        partitions: list[WorkloadAwarePartition] = []
        sorted_clusters = sorted(
            clusters,
            key=lambda meta: (
                -int(meta["document_count"]),
                min(int(pattern_id) for pattern_id in meta["pattern_ids"]),
            ),
        )
        for partition_index, meta in enumerate(sorted_clusters):
            tenant_densities = {str(tenant_id): float(density) for tenant_id, density in self._cluster_densities(meta).items()}
            tenant_query_mass = {
                str(tenant_id): float(value)
                for tenant_id, value in sorted(meta["tenant_query_mass"].items())
            }
            ordered_pattern_ids = sorted(int(pattern_id) for pattern_id in meta["pattern_ids"])
            ordered_patterns_for_metadata = [
                [int(tenant_id) for tenant_id in patterns_by_id[pattern_id].ordered_tenant_ids]
                for pattern_id in ordered_pattern_ids
            ]
            partitions.append(
                WorkloadAwarePartition(
                    partition_id=f"p{partition_index}",
                    table_name=get_partition_table_name(f"p{partition_index}"),
                    document_ids=tuple(sorted(int(document_id) for document_id in meta["document_ids"])),
                    tenant_ids=tuple(sorted(int(tenant_id) for tenant_id in meta["tenant_ids"])),
                    vector_count=int(meta["vector_count"]),
                    logical_pattern_ids=tuple(ordered_pattern_ids),
                    metadata={
                        "tenant_densities": tenant_densities,
                        "tenant_query_mass": tenant_query_mass,
                        "entry_tenant_ids": sorted(int(tenant_id) for tenant_id in meta["entry_tenant_ids"]),
                        "primary_anchor": int(meta["primary_anchor"]),
                        "logical_pattern_count": int(meta["pattern_count"]),
                        "pattern_document_count": int(meta["document_count"]),
                        "partition_query_mass": float(meta.get("query_mass", 0.0)),
                        "representative_centroid": _normalize_vector(_parse_vector(meta.get("representative_centroid", []))).astype(float).tolist(),
                        "ordered_acl_patterns": ordered_patterns_for_metadata,
                    },
                )
            )
        return partitions
