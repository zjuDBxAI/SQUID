from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from controller.method.common import DocumentAccessRecord, WorkloadQuery
from controller.method.planner import WorkloadAwarePlanner


def _vector(values: list[float]) -> np.ndarray:
    return np.asarray(values, dtype=np.float32)


class PlannerTreeDpTest(unittest.TestCase):
    def setUp(self) -> None:
        self.planner = WorkloadAwarePlanner(random_state=42)

    def _build_plan(
        self,
        pattern_specs: list[tuple[tuple[int, ...], int, list[float]]],
        *,
        target_partition_count: int,
        query_weights: dict[int, float] | None = None,
        overlay_space_ratio: float = 0.0,
    ):
        records: list[DocumentAccessRecord] = []
        block_counts: dict[int, int] = {}
        tenant_ids: set[int] = set()
        document_id = 1

        for pattern_index, (acl, document_count, base_vector) in enumerate(pattern_specs):
            normalized_acl = tuple(sorted(int(tenant_id) for tenant_id in acl))
            base = _vector(base_vector)
            tenant_ids.update(normalized_acl)
            for local_index in range(int(document_count)):
                jitter = np.zeros_like(base)
                jitter[local_index % max(1, base.size)] = 0.01 * float(pattern_index + 1)
                records.append(
                    DocumentAccessRecord(
                        document_id=int(document_id),
                        representative_block_id=int(document_id),
                        vector=(base + jitter).astype(np.float32),
                        tenant_ids=normalized_acl,
                    )
                )
                block_counts[int(document_id)] = 1
                document_id += 1

        queries = [
            WorkloadQuery(
                tenant_id=int(tenant_id),
                query_vector=_vector([float(tenant_id), 1.0, 0.0, 0.0]),
                topk=10,
                weight=float((query_weights or {}).get(int(tenant_id), 1.0)),
            )
            for tenant_id in sorted(tenant_ids)
        ]
        tenant_query_weights = {int(query.tenant_id): float(query.weight) for query in queries}
        return self.planner.build_plan(
            records,
            document_block_counts=block_counts,
            queries=queries,
            tenant_query_weights=tenant_query_weights,
            target_partition_count=int(target_partition_count),
            overlay_space_ratio=float(overlay_space_ratio),
        )

    def test_dp_respects_budget_when_root_has_many_children(self) -> None:
        pattern_specs = [
            ((tenant_id,), 2, [float(tenant_id), 0.1 * float(tenant_id), 1.0, 0.5])
            for tenant_id in range(1, 13)
        ]

        plan = self._build_plan(pattern_specs, target_partition_count=4)
        assigned_pattern_ids = [
            int(pattern_id)
            for partition in plan.partitions
            for pattern_id in partition.logical_pattern_ids
        ]
        logical_pattern_ids = [int(pattern.pattern_id) for pattern in plan.logical_patterns]

        self.assertEqual(int(plan.metadata["logical_pattern_count"]), 12)
        self.assertEqual(int(plan.metadata["planning_root_children"]), 12)
        self.assertEqual(int(plan.metadata["partition_count"]), 4)
        self.assertEqual(int(plan.metadata["dp_effective_partition_count"]), 4)
        self.assertEqual(int(plan.metadata["dp_cut_edge_count"]), 3)
        self.assertLess(int(plan.metadata["partition_count"]), int(plan.metadata["logical_pattern_count"]))
        self.assertEqual(sorted(assigned_pattern_ids), sorted(logical_pattern_ids))
        self.assertEqual(len(assigned_pattern_ids), len(set(assigned_pattern_ids)))

    def test_tree_pruning_and_partition_merge_cover_all_patterns(self) -> None:
        pattern_specs = [
            ((1, 2, 3, 4), 3, [1.0, 0.0, 0.0, 0.0]),
            ((1, 2, 3), 3, [1.1, 0.1, 0.0, 0.0]),
            ((1, 2), 3, [1.2, 0.2, 0.0, 0.0]),
            ((5, 6, 7), 3, [0.0, 1.0, 0.0, 0.0]),
            ((5, 6), 3, [0.0, 1.1, 0.1, 0.0]),
            ((8, 9), 3, [0.0, 0.0, 1.0, 0.0]),
        ]

        plan = self._build_plan(pattern_specs, target_partition_count=3)
        assigned_pattern_ids = {
            int(pattern_id)
            for partition in plan.partitions
            for pattern_id in partition.logical_pattern_ids
        }
        logical_pattern_ids = {int(pattern.pattern_id) for pattern in plan.logical_patterns}

        self.assertGreater(int(plan.metadata["planning_group_node_count"]), 0)
        self.assertGreater(int(plan.metadata["pruned_removed_planning_node_count"]), 0)
        self.assertEqual(int(plan.metadata["partition_count"]), 3)
        self.assertLess(int(plan.metadata["partition_count"]), int(plan.metadata["logical_pattern_count"]))
        self.assertEqual(assigned_pattern_ids, logical_pattern_ids)
        self.assertTrue(any(len(partition.logical_pattern_ids) > 1 for partition in plan.partitions))
        self.assertTrue(all(partition.document_count > 0 for partition in plan.partitions))

    def test_hot_tenant_overlay_uses_space_budget_without_pattern_accelerators(self) -> None:
        pattern_specs = [
            ((1, 2), 5, [1.0, 0.0, 0.0, 0.0]),
            ((1, 3), 5, [0.0, 1.0, 0.0, 0.0]),
            ((1, 4), 5, [0.0, 0.0, 1.0, 0.0]),
            ((2, 5), 5, [0.0, 1.0, 0.0, 0.0]),
            ((3, 6), 5, [0.0, 0.0, 1.0, 0.0]),
            ((4, 7), 5, [1.0, 0.0, 0.0, 0.0]),
        ]

        plan = self._build_plan(
            pattern_specs,
            target_partition_count=3,
            query_weights={1: 100.0, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0, 6: 1.0, 7: 1.0},
            overlay_space_ratio=0.5,
        )
        overlays = list(plan.metadata.get("tenant_overlays", []) or [])
        access_overlays = list(plan.metadata.get("access_overlays", []) or [])

        hot_tenant_full_overlay = next((overlay for overlay in overlays if int(overlay["tenant_id"]) == 1), None)
        hot_tenant_access_overlay = next((overlay for overlay in access_overlays if int(overlay["tenant_id"]) == 1), None)
        self.assertTrue(hot_tenant_full_overlay is not None or hot_tenant_access_overlay is not None)
        if hot_tenant_full_overlay is not None:
            self.assertLess(
                int(hot_tenant_full_overlay["vector_count"]),
                int(hot_tenant_full_overlay["covered_partition_vector_count"]),
            )
            self.assertGreater(float(hot_tenant_full_overlay["estimated_saved_cost"]), 0.0)
        if hot_tenant_access_overlay is not None:
            self.assertGreaterEqual(int(hot_tenant_access_overlay["covered_partition_count"]), 1)
            self.assertGreater(float(hot_tenant_access_overlay["estimated_saved_cost"]), 0.0)
        self.assertLessEqual(
            int(plan.metadata["overlay_selected_vectors"]),
            int(plan.metadata["overlay_budget_vectors"]),
        )
        self.assertTrue(
            all(not (partition.metadata.get("accelerator_patterns", []) or []) for partition in plan.partitions)
        )
        self.assertFalse(
            {int(overlay["tenant_id"]) for overlay in overlays}
            & {int(overlay["tenant_id"]) for overlay in access_overlays}
        )


if __name__ == "__main__":
    unittest.main()
