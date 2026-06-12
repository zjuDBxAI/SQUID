from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import itertools
import math
from typing import Iterable, Optional

from tqdm import tqdm

from .common import (
    DEFAULT_COST_A,
    DEFAULT_COST_B,
    DEFAULT_COST_C,
    VedaNode,
    VedaPattern,
    VedaPlan,
    VedaRoute,
    get_node_table_name,
    normalize_algorithm,
    normalize_int_tuple,
    role_key,
)


@dataclass(slots=True)
class _LatticeNode:
    node_id: str
    role_ids: tuple[int, ...]
    pattern_ids: frozenset[int]
    virtual_components: tuple[str, ...] = field(default_factory=tuple)


class VedaPlanner:
    """Planner for the Veda and EffVeda access-aware lattice algorithms."""

    def __init__(
        self,
        *,
        indexing_threshold: int = 1000,
        storage_amplification: float = 1.2,
        ef_search: int = 100,
        cost_a: float = DEFAULT_COST_A,
        cost_b: float = DEFAULT_COST_B,
        cost_c: float = DEFAULT_COST_C,
        linear_scan_cost: float = 0.0005,
        max_veda_rounds: int = 8,
    ) -> None:
        self.indexing_threshold = max(1, int(indexing_threshold))
        self.storage_amplification = max(1.0, float(storage_amplification))
        self.ef_search = max(1, int(ef_search))
        self.cost_a = float(cost_a)
        self.cost_b = float(cost_b)
        self.cost_c = float(cost_c)
        self.linear_scan_cost = max(0.0, float(linear_scan_cost))
        self.max_veda_rounds = max(1, int(max_veda_rounds))

        self.patterns: dict[int, VedaPattern] = {}
        self.roles: tuple[int, ...] = tuple()
        self.exclusive_lattice: dict[str, _LatticeNode] = {}
        self.post_copy_lattice: dict[str, _LatticeNode] = {}
        self.node_by_roles: dict[tuple[int, ...], str] = {}
        self._generated_node_counter = 0

    def build_plan(
        self,
        patterns: list[VedaPattern],
        *,
        role_ids: Iterable[int],
        user_roles: dict[int, tuple[int, ...]],
        algorithm: str = "effveda",
        show_progress: bool = True,
    ) -> VedaPlan:
        if not patterns:
            raise ValueError("Cannot build a Veda plan without exclusive patterns")

        self.patterns = {int(pattern.pattern_id): pattern for pattern in patterns}
        self.roles = normalize_int_tuple(role_ids)
        if not self.roles:
            self.roles = normalize_int_tuple(role_id for pattern in patterns for role_id in pattern.role_ids)
        if not self.roles:
            raise ValueError("Cannot build a Veda plan without roles")

        self.exclusive_lattice = self._build_exclusive_lattice(patterns)
        self.node_by_roles = {
            node.role_ids: node_id
            for node_id, node in self.exclusive_lattice.items()
        }
        algorithm = normalize_algorithm(algorithm)
        lattice = self._clone_lattice(self.exclusive_lattice)

        if algorithm == "veda":
            lattice, operation_metadata = self._run_veda(lattice, show_progress=show_progress)
        else:
            lattice, operation_metadata = self._run_effveda(lattice, show_progress=show_progress)

        lattice, finalize_metadata = self._finalize_lattice(lattice)
        role_plans = self._build_role_query_plans(lattice, exact=True)
        if self._storage_amplification(lattice) < self.storage_amplification:
            lattice, role_plans, super_impure_metadata = self._handle_super_impure_nodes(lattice, role_plans)
        else:
            super_impure_metadata = {"enabled": False, "reason": "no_reclaimed_storage_budget"}

        nodes = self._materializable_nodes(lattice)
        node_id_map = {
            str(node.metadata.get("source_node_id", node.node_id)): node.node_id
            for node in nodes
        }
        role_plans = self._translate_role_plans(role_plans, node_id_map)
        routes = self._build_user_routes(nodes, role_plans, user_roles)

        original_vectors = self._exclusive_vector_count()
        materialized_vectors = sum(int(node.vector_count) for node in nodes)
        metadata = {
            "algorithm": algorithm,
            "paper": "Veda/EffVeda access-aware lattice",
            "indexing_threshold": int(self.indexing_threshold),
            "storage_amplification_budget": float(self.storage_amplification),
            "ef_search_for_cost": int(self.ef_search),
            "cost_model": "C_theta(|idx|, efs) = a*log2(|idx|+1) + b*efs + c; impure routes inflate efs by lambda",
            "cost_a": float(self.cost_a),
            "cost_b": float(self.cost_b),
            "cost_c": float(self.cost_c),
            "role_count": int(len(self.roles)),
            "pattern_count": int(len(patterns)),
            "node_count": int(len(nodes)),
            "index_node_count": int(sum(1 for node in nodes if node.node_kind == "index")),
            "leftover_node_count": int(sum(1 for node in nodes if node.node_kind == "leftover")),
            "document_count": int(sum(len(pattern.document_ids) for pattern in patterns)),
            "original_vector_count": int(original_vectors),
            "materialized_vector_count": int(materialized_vectors),
            "storage_amplification_actual": float(materialized_vectors / max(1, original_vectors)),
            "operation_metadata": operation_metadata,
            "finalize_metadata": finalize_metadata,
            "super_impure_metadata": super_impure_metadata,
        }
        return VedaPlan(
            algorithm=algorithm,
            patterns=list(sorted(patterns, key=lambda pattern: int(pattern.pattern_id))),
            nodes=nodes,
            role_plans=role_plans,
            user_routes=routes,
            metadata=metadata,
        )

    def _build_exclusive_lattice(self, patterns: list[VedaPattern]) -> dict[str, _LatticeNode]:
        lattice: dict[str, _LatticeNode] = {}
        for pattern in patterns:
            node_id = role_key(pattern.role_ids)
            lattice[node_id] = _LatticeNode(
                node_id=node_id,
                role_ids=pattern.role_ids,
                pattern_ids=frozenset({int(pattern.pattern_id)}),
                virtual_components=(node_id,),
            )
        return lattice

    def _clone_lattice(self, lattice: dict[str, _LatticeNode]) -> dict[str, _LatticeNode]:
        return {
            node_id: _LatticeNode(
                node_id=node.node_id,
                role_ids=tuple(node.role_ids),
                pattern_ids=frozenset(node.pattern_ids),
                virtual_components=tuple(node.virtual_components),
            )
            for node_id, node in lattice.items()
        }

    def _exclusive_vector_count(self) -> int:
        return int(sum(max(0, int(pattern.vector_count)) for pattern in self.patterns.values()))

    def _node_size(self, node: _LatticeNode | Iterable[int]) -> int:
        pattern_ids = node.pattern_ids if isinstance(node, _LatticeNode) else node
        return int(sum(max(0, int(self.patterns[int(pattern_id)].vector_count)) for pattern_id in pattern_ids))

    def _authorized_size(self, node: _LatticeNode, role_id: int) -> int:
        role_id = int(role_id)
        return int(
            sum(
                max(0, int(self.patterns[int(pattern_id)].vector_count))
                for pattern_id in node.pattern_ids
                if role_id in self.patterns[int(pattern_id)].role_ids
            )
        )

    def _node_cost_for_role(self, node: _LatticeNode, role_id: int, *, final: bool = False) -> float:
        node_size = max(1, int(self._node_size(node)))
        authorized_size = int(self._authorized_size(node, int(role_id)))
        if authorized_size <= 0:
            return float("inf")
        if final and node_size < self.indexing_threshold:
            return float(self.linear_scan_cost * authorized_size)
        impurity = float(node_size) / float(max(1, authorized_size))
        return float(self.cost_a * math.log2(node_size + 1) + self.cost_b * impurity * self.ef_search + self.cost_c)

    def _plan_cost(self, lattice: dict[str, _LatticeNode], role_plans: Optional[dict[int, tuple[str, ...]]] = None) -> float:
        plans = role_plans if role_plans is not None else self._build_role_query_plans(lattice)
        if not plans:
            return 0.0
        total = 0.0
        for role_id, node_ids in plans.items():
            for node_id in node_ids:
                node = lattice.get(node_id)
                if node is None:
                    continue
                total += self._node_cost_for_role(node, int(role_id))
        return float(total / max(1, len(plans)))

    def _storage_amplification(self, lattice: dict[str, _LatticeNode]) -> float:
        return float(sum(self._node_size(node) for node in lattice.values()) / max(1, self._exclusive_vector_count()))

    def _descendant_ancestor_pairs(self, lattice: dict[str, _LatticeNode]) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        nodes = list(lattice.values())
        for child in nodes:
            child_roles = set(child.role_ids)
            if not child_roles:
                continue
            for ancestor in nodes:
                if child.node_id == ancestor.node_id:
                    continue
                ancestor_roles = set(ancestor.role_ids)
                if ancestor_roles and ancestor_roles < child_roles:
                    pairs.append((child.node_id, ancestor.node_id))
        pairs.sort(key=lambda item: (len(lattice[item[0]].role_ids), len(lattice[item[1]].role_ids), item[0], item[1]), reverse=True)
        return pairs

    def _copy_delta_size(self, lattice: dict[str, _LatticeNode], child_id: str, ancestor_id: str) -> int:
        child_exclusive = self.exclusive_lattice.get(child_id)
        ancestor = lattice.get(ancestor_id)
        if child_exclusive is None or ancestor is None:
            return 0
        return int(self._node_size(child_exclusive.pattern_ids - ancestor.pattern_ids))

    def _simulate_copy(self, lattice: dict[str, _LatticeNode], child_id: str, ancestor_id: str) -> dict[str, _LatticeNode]:
        child_exclusive = self.exclusive_lattice[child_id]
        next_lattice = self._clone_lattice(lattice)
        ancestor = next_lattice[ancestor_id]
        next_lattice[ancestor_id] = _LatticeNode(
            node_id=ancestor.node_id,
            role_ids=ancestor.role_ids,
            pattern_ids=frozenset(set(ancestor.pattern_ids) | set(child_exclusive.pattern_ids)),
            virtual_components=tuple(dict.fromkeys((*ancestor.virtual_components, *child_exclusive.virtual_components))),
        )
        return next_lattice

    def _simulate_merge(self, lattice: dict[str, _LatticeNode], child_id: str, ancestor_id: str) -> dict[str, _LatticeNode]:
        next_lattice = self._clone_lattice(lattice)
        child = next_lattice[child_id]
        ancestor = next_lattice[ancestor_id]
        next_lattice[ancestor_id] = _LatticeNode(
            node_id=ancestor.node_id,
            role_ids=normalize_int_tuple(set(ancestor.role_ids) | set(child.role_ids)),
            pattern_ids=frozenset(set(ancestor.pattern_ids) | set(child.pattern_ids)),
            virtual_components=tuple(dict.fromkeys((*ancestor.virtual_components, *child.virtual_components))),
        )
        del next_lattice[child_id]
        return next_lattice

    def _run_veda(self, lattice: dict[str, _LatticeNode], *, show_progress: bool) -> tuple[dict[str, _LatticeNode], dict[str, object]]:
        budget_vectors = int(math.floor(self.storage_amplification * self._exclusive_vector_count()))
        copy_count = 0
        merge_count = 0
        rounds = 0
        pairs = self._descendant_ancestor_pairs(self.exclusive_lattice)
        iterator = range(self.max_veda_rounds)
        if show_progress:
            iterator = tqdm(iterator, desc="Veda greedy rounds", unit="round")
        for _ in iterator:
            rounds += 1
            copied = self._veda_copy_phase(lattice, pairs, budget_vectors)
            copy_count += copied
            merged = self._veda_merge_phase(lattice, pairs)
            merge_count += merged
            if copied <= 0 and merged <= 0:
                break
        return lattice, {
            "copy_operations": int(copy_count),
            "merge_operations": int(merge_count),
            "rounds": int(rounds),
            "storage_budget_vectors": int(budget_vectors),
            "benefit_function": "full AvgCost query-plan re-derivation over descendant-ancestor pairs",
        }

    def _veda_copy_phase(self, lattice: dict[str, _LatticeNode], pairs: list[tuple[str, str]], budget_vectors: int) -> int:
        applied = 0
        while True:
            current_storage = int(sum(self._node_size(node) for node in lattice.values()))
            buffer = int(budget_vectors - current_storage)
            if buffer <= 0:
                return applied
            current_cost = self._plan_cost(lattice)
            best = None
            best_benefit = 0.0
            for child_id, ancestor_id in pairs:
                if child_id not in lattice or ancestor_id not in lattice:
                    continue
                delta = self._copy_delta_size(lattice, child_id, ancestor_id)
                if delta <= 0 or delta > buffer:
                    continue
                next_lattice = self._simulate_copy(lattice, child_id, ancestor_id)
                benefit = (current_cost - self._plan_cost(next_lattice)) / float(delta + 1)
                if benefit > best_benefit:
                    best_benefit = float(benefit)
                    best = (child_id, ancestor_id, next_lattice)
            if best is None or best_benefit < 0.0:
                return applied
            _child_id, _ancestor_id, next_lattice = best
            lattice.clear()
            lattice.update(next_lattice)
            applied += 1

    def _veda_merge_phase(self, lattice: dict[str, _LatticeNode], pairs: list[tuple[str, str]]) -> int:
        applied = 0
        while True:
            current_cost = self._plan_cost(lattice)
            best = None
            best_benefit = 0.0
            for child_id, ancestor_id in pairs:
                if child_id not in lattice or ancestor_id not in lattice:
                    continue
                next_lattice = self._simulate_merge(lattice, child_id, ancestor_id)
                benefit = current_cost - self._plan_cost(next_lattice)
                if benefit > best_benefit:
                    best_benefit = float(benefit)
                    best = (child_id, ancestor_id, next_lattice)
            if best is None or best_benefit <= 0.0:
                return applied
            _child_id, _ancestor_id, next_lattice = best
            lattice.clear()
            lattice.update(next_lattice)
            applied += 1

    def _run_effveda(self, lattice: dict[str, _LatticeNode], *, show_progress: bool) -> tuple[dict[str, _LatticeNode], dict[str, object]]:
        copy_count = self._effveda_copy_phase(lattice, show_progress=show_progress)
        self._refresh_virtual_components(lattice)
        merge_count = self._effveda_merge_phase(lattice, show_progress=show_progress)
        return lattice, {
            "copy_operations": int(copy_count),
            "merge_operations": int(merge_count),
            "copy_goal": "bottom-up valid-partition full-node duplication",
            "merge_goal": "grow large unindexable nodes to the indexing threshold with virtual-decomposition impurity scoring",
        }

    def _effveda_copy_phase(self, lattice: dict[str, _LatticeNode], *, show_progress: bool) -> int:
        original_vectors = self._exclusive_vector_count()
        buffer = int(math.floor((self.storage_amplification - 1.0) * original_vectors))
        if buffer <= 0:
            return 0
        applied = 0
        max_depth = max((len(node.role_ids) for node in lattice.values()), default=0)
        layers = range(max_depth, 1, -1)
        if show_progress:
            layers = tqdm(layers, desc="EffVeda copy layers", unit="layer")
        for layer in layers:
            candidates: list[tuple[float, str, tuple[tuple[int, ...], ...], int]] = []
            role_index = self._role_index(lattice)
            for node in list(lattice.values()):
                if len(node.role_ids) != layer:
                    continue
                node_size = self._node_size(node)
                if node_size <= 0 or node_size > buffer:
                    continue
                ancestors = [
                    lattice[node_id]
                    for roles, node_id in role_index.items()
                    if set(roles) < set(node.role_ids)
                ]
                ancestors.sort(key=lambda item: (-len(item.role_ids), item.role_ids))
                partition, benefit = self._effveda_find_best_partition(node, ancestors, buffer)
                if partition:
                    delta = node_size * max(0, len(partition) - 1)
                    candidates.append((float(benefit), node.node_id, tuple(partition), int(delta)))
            candidates.sort(key=lambda item: (item[0], -item[3], item[1]), reverse=True)
            for _benefit, node_id, partition, delta in candidates:
                if node_id not in lattice or delta > buffer:
                    continue
                node = lattice[node_id]
                for target_roles in partition:
                    if set(target_roles) == set(node.role_ids):
                        continue
                    target_id = self._role_index(lattice).get(target_roles)
                    if target_id is None:
                        target_id = self._new_node_id("res", target_roles)
                        lattice[target_id] = _LatticeNode(
                            node_id=target_id,
                            role_ids=normalize_int_tuple(target_roles),
                            pattern_ids=frozenset(node.pattern_ids),
                            virtual_components=tuple(node.virtual_components),
                        )
                    else:
                        target = lattice[target_id]
                        lattice[target_id] = _LatticeNode(
                            node_id=target.node_id,
                            role_ids=target.role_ids,
                            pattern_ids=frozenset(set(target.pattern_ids) | set(node.pattern_ids)),
                            virtual_components=tuple(dict.fromkeys((*target.virtual_components, *node.virtual_components))),
                        )
                if node_id in lattice:
                    del lattice[node_id]
                buffer -= int(delta)
                applied += 1
        return applied

    def _effveda_find_best_partition(
        self,
        node: _LatticeNode,
        ancestors: list[_LatticeNode],
        buffer: int,
    ) -> tuple[list[tuple[int, ...]], float]:
        if not ancestors or self._node_size(node) > buffer:
            return [], 0.0
        ancestor_by_roles = {ancestor.role_ids: ancestor for ancestor in ancestors}
        best_two_way: list[tuple[int, ...]] = []
        best_two_way_delta = 0.0
        best_singleton: list[tuple[int, ...]] = []
        best_singleton_delta = 0.0
        node_roles = set(node.role_ids)

        for ancestor in ancestors:
            ancestor_roles = set(ancestor.role_ids)
            singleton_delta = len(ancestor.role_ids) * self._copy_pair_gain(node, ancestor)
            if singleton_delta > best_singleton_delta:
                best_singleton_delta = float(singleton_delta)
                best_singleton = [ancestor.role_ids]

            complement = normalize_int_tuple(node_roles - ancestor_roles)
            complement_node = ancestor_by_roles.get(complement)
            if complement_node is None:
                continue
            two_way_delta = singleton_delta + len(complement_node.role_ids) * self._copy_pair_gain(node, complement_node)
            if two_way_delta > best_two_way_delta:
                best_two_way_delta = float(two_way_delta)
                best_two_way = [ancestor.role_ids, complement_node.role_ids]

        node_size = max(1, self._node_size(node))
        if best_two_way:
            return best_two_way, float(best_two_way_delta / node_size)
        if not best_singleton:
            return [], 0.0

        selected = [best_singleton[0]]
        covered = set(best_singleton[0])
        residual = set(node.role_ids) - covered
        delta = float(best_singleton_delta)
        delta_storage = 0
        for ancestor in ancestors:
            ancestor_roles = set(ancestor.role_ids)
            if ancestor.role_ids in selected:
                continue
            if ancestor_roles and ancestor_roles <= residual:
                delta_storage += node_size
                if delta_storage > buffer:
                    break
                selected.append(ancestor.role_ids)
                residual -= ancestor_roles
                delta += len(ancestor.role_ids) * self._copy_pair_gain(node, ancestor)
                if not residual:
                    break
        if residual:
            selected.append(normalize_int_tuple(residual))
        denominator = node_size * max(1, len(selected) - 1)
        return selected, float(delta / denominator)

    def _copy_pair_gain(self, child: _LatticeNode, ancestor: _LatticeNode) -> float:
        child_size = max(1, self._node_size(child))
        ancestor_size = max(1, self._node_size(ancestor))
        before = self._hnsw_cost(ancestor_size, self.ef_search) + self._hnsw_cost(child_size, self.ef_search)
        after = self._hnsw_cost(ancestor_size + child_size, self.ef_search)
        return float(before - after)

    def _effveda_merge_phase(self, lattice: dict[str, _LatticeNode], *, show_progress: bool) -> int:
        node_order = sorted(lattice, key=lambda node_id: self._node_size(lattice[node_id]), reverse=True)
        applied = 0
        index = 0
        iterator_total = len(node_order)
        progress = tqdm(total=iterator_total, desc="EffVeda merge nodes", unit="node", disable=not show_progress)
        try:
            while index < len(node_order):
                node_id = node_order[index]
                if node_id not in lattice or self._node_size(lattice[node_id]) >= self.indexing_threshold:
                    index += 1
                    progress.update(1)
                    continue
                node = lattice[node_id]
                candidates = self._merge_candidates(node, lattice)
                scored = [
                    (self._eff_merge_benefit(node, candidate), candidate.node_id)
                    for candidate in candidates
                    if candidate.node_id in lattice
                ]
                scored.sort(key=lambda item: item[0], reverse=True)
                merged_this_node = False
                for initial_benefit, candidate_id in scored:
                    if initial_benefit <= 0.0 or node_id not in lattice or candidate_id not in lattice:
                        break
                    renewed = self._eff_merge_benefit(lattice[node_id], lattice[candidate_id])
                    if renewed <= 0.0:
                        continue
                    lattice[node_id] = self._merge_nodes(lattice[node_id], lattice[candidate_id])
                    del lattice[candidate_id]
                    applied += 1
                    merged_this_node = True
                    if self._node_size(lattice[node_id]) >= self.indexing_threshold:
                        break
                if node_id in lattice and self._node_size(lattice[node_id]) < self.indexing_threshold and merged_this_node:
                    continue
                index += 1
                progress.update(1)
        finally:
            progress.close()
        return applied

    def _refresh_virtual_components(self, lattice: dict[str, _LatticeNode]) -> None:
        self.post_copy_lattice = {
            node.node_id: _LatticeNode(
                node_id=node.node_id,
                role_ids=node.role_ids,
                pattern_ids=node.pattern_ids,
                virtual_components=(node.node_id,),
            )
            for node in lattice.values()
        }
        for node_id, node in list(lattice.items()):
            lattice[node_id] = _LatticeNode(
                node_id=node.node_id,
                role_ids=node.role_ids,
                pattern_ids=node.pattern_ids,
                virtual_components=(node.node_id,),
            )

    def _merge_candidates(self, node: _LatticeNode, lattice: dict[str, _LatticeNode]) -> list[_LatticeNode]:
        node_roles = set(node.role_ids)
        candidates = []
        for candidate in lattice.values():
            if candidate.node_id == node.node_id:
                continue
            candidate_roles = set(candidate.role_ids)
            if node_roles < candidate_roles or candidate_roles < node_roles:
                candidates.append(candidate)
            elif len(node_roles) == len(candidate_roles) and node_roles & candidate_roles:
                candidates.append(candidate)
        candidates.sort(key=lambda item: (self._node_size(item), len(item.role_ids)), reverse=True)
        return candidates

    def _eff_merge_benefit(self, left: _LatticeNode, right: _LatticeNode) -> float:
        merged = self._merge_nodes(left, right)
        return float(self._eff_h(left) + self._eff_h(right) - self._eff_h(merged))

    def _eff_h(self, node: _LatticeNode) -> float:
        roles = self._pi_roles(node)
        if not roles:
            return 0.0
        node_size = max(1, self._node_size(node))
        total = len(roles) * self.cost_a * math.log2(node_size + 1)
        for role_id in roles:
            omega = self._virtual_authorized_size(node, role_id)
            impurity = float(node_size) / float(max(1, omega))
            total += self.cost_b * impurity * self.ef_search + self.cost_c
        return float(total)

    def _component_node(self, component_id: str) -> _LatticeNode | None:
        component = self.post_copy_lattice.get(str(component_id))
        if component is not None:
            return component
        return self.exclusive_lattice.get(str(component_id))

    def _virtual_authorized_size(self, node: _LatticeNode, role_id: int) -> int:
        total = 0
        role_id = int(role_id)
        for component_id in node.virtual_components:
            component = self._component_node(str(component_id))
            if component is None:
                continue
            if role_id in component.role_ids:
                total += self._node_size(component)
        if total > 0:
            return int(total)
        return self._authorized_size(node, role_id)

    def _pi_roles(self, node: _LatticeNode) -> tuple[int, ...]:
        roles: set[int] = set()
        for component_id in node.virtual_components:
            component = self._component_node(str(component_id))
            if component is not None:
                roles.update(component.role_ids)
        if not roles:
            roles.update(node.role_ids)
        return normalize_int_tuple(roles)

    def _merge_nodes(self, left: _LatticeNode, right: _LatticeNode) -> _LatticeNode:
        return _LatticeNode(
            node_id=left.node_id,
            role_ids=normalize_int_tuple(set(left.role_ids) | set(right.role_ids)),
            pattern_ids=frozenset(set(left.pattern_ids) | set(right.pattern_ids)),
            virtual_components=tuple(dict.fromkeys((*left.virtual_components, *right.virtual_components))),
        )

    def _finalize_lattice(self, lattice: dict[str, _LatticeNode]) -> tuple[dict[str, _LatticeNode], dict[str, object]]:
        finalized: dict[str, _LatticeNode] = {}
        split_count = 0
        for node in lattice.values():
            if self._node_size(node) >= self.indexing_threshold:
                finalized[node.node_id] = node
                continue
            for pattern_id in sorted(node.pattern_ids):
                pattern = self.patterns[int(pattern_id)]
                node_id = self._new_node_id("leftover", pattern.role_ids, suffix=str(pattern_id))
                finalized[node_id] = _LatticeNode(
                    node_id=node_id,
                    role_ids=pattern.role_ids,
                    pattern_ids=frozenset({int(pattern_id)}),
                    virtual_components=(role_key(pattern.role_ids),),
                )
                split_count += 1
        return finalized, {
            "split_small_nodes_into_leftovers": True,
            "small_group_split_count": int(split_count),
        }

    def _standalone_node_for_pattern(self, lattice: dict[str, _LatticeNode], pattern_id: int) -> str | None:
        target = frozenset({int(pattern_id)})
        for node_id, node in lattice.items():
            if node.pattern_ids == target:
                return node_id
        return None

    def _separate_pure_pattern(
        self,
        lattice: dict[str, _LatticeNode],
        copied: dict[int, str],
        pattern_id: int,
    ) -> tuple[str, int]:
        pattern_id = int(pattern_id)
        standalone_id = self._standalone_node_for_pattern(lattice, pattern_id)
        if standalone_id is not None:
            return standalone_id, 0
        copied_id = copied.get(pattern_id)
        if copied_id is not None and copied_id in lattice:
            return copied_id, 0

        pattern = self.patterns[pattern_id]
        base_id = role_key(pattern.role_ids)
        if base_id not in lattice:
            node_id = base_id
        else:
            node_id = self._new_node_id("pure", pattern.role_ids, suffix=str(pattern_id))
        lattice[node_id] = _LatticeNode(
            node_id=node_id,
            role_ids=pattern.role_ids,
            pattern_ids=frozenset({pattern_id}),
            virtual_components=(role_key(pattern.role_ids),),
        )
        copied[pattern_id] = node_id
        return node_id, int(pattern.vector_count)

    def _pure_pattern_copy_size(
        self,
        lattice: dict[str, _LatticeNode],
        copied: dict[int, str],
        pattern_ids: Iterable[int],
    ) -> int:
        total = 0
        for pattern_id in pattern_ids:
            pattern_id = int(pattern_id)
            if self._standalone_node_for_pattern(lattice, pattern_id) is not None:
                continue
            copied_id = copied.get(pattern_id)
            if copied_id is not None and copied_id in lattice:
                continue
            total += int(self.patterns[pattern_id].vector_count)
        return int(total)

    def _handle_super_impure_nodes(
        self,
        lattice: dict[str, _LatticeNode],
        role_plans: dict[int, tuple[str, ...]],
    ) -> tuple[dict[str, _LatticeNode], dict[int, tuple[str, ...]], dict[str, object]]:
        budget_vectors = int(math.floor(self.storage_amplification * self._exclusive_vector_count()))
        current_storage = int(sum(self._node_size(node) for node in lattice.values()))
        buffer = max(0, budget_vectors - current_storage)
        if buffer <= 0:
            return lattice, role_plans, {"enabled": False, "reason": "no_buffer"}

        ref: dict[str, int] = defaultdict(int)
        for node_ids in role_plans.values():
            for node_id in set(node_ids):
                ref[str(node_id)] += 1

        candidates: list[tuple[float, int, int, str, tuple[int, ...]]] = []
        for role_id, node_ids in role_plans.items():
            for node_id in node_ids:
                node = lattice.get(node_id)
                if node is None:
                    continue
                pure_pattern_ids = tuple(
                    int(pattern_id)
                    for pattern_id in node.pattern_ids
                    if int(role_id) in self.patterns[int(pattern_id)].role_ids
                )
                pure_size = int(sum(self.patterns[pattern_id].vector_count for pattern_id in pure_pattern_ids))
                node_size = self._node_size(node)
                if 0 < pure_size < node_size:
                    impurity = float(node_size) / float(max(1, pure_size))
                    candidates.append((impurity, pure_size, int(role_id), node_id, pure_pattern_ids))
        candidates.sort(key=lambda item: (item[0], -item[1]), reverse=True)

        copied: dict[int, str] = {}
        refined = 0
        copied_pattern_count = 0
        deleted_nodes = 0
        for _impurity, _pure_size, role_id, node_id, pattern_ids in candidates:
            current_plan = set(role_plans.get(role_id, ()))
            if node_id not in current_plan or node_id not in lattice:
                continue
            copy_size = self._pure_pattern_copy_size(lattice, copied, pattern_ids)
            if copy_size > buffer:
                continue

            added_node_ids = []
            spent = 0
            copied_before = set(copied)
            for pattern_id in pattern_ids:
                standalone_id, delta = self._separate_pure_pattern(lattice, copied, int(pattern_id))
                added_node_ids.append(standalone_id)
                spent += int(delta)
            buffer -= int(spent)
            copied_pattern_count += len(set(copied) - copied_before)

            current_plan.discard(node_id)
            current_plan.update(added_node_ids)
            role_plans[role_id] = tuple(sorted(current_plan))
            ref[node_id] = max(0, int(ref.get(node_id, 0)) - 1)
            if ref[node_id] == 0 and node_id in lattice:
                buffer += self._node_size(lattice[node_id])
                del lattice[node_id]
                deleted_nodes += 1
            refined += 1
        return lattice, role_plans, {
            "enabled": True,
            "refined_route_count": int(refined),
            "copied_pattern_count": int(copied_pattern_count),
            "deleted_unref_node_count": int(deleted_nodes),
        }

    def _build_role_query_plans(self, lattice: dict[str, _LatticeNode], *, exact: bool = False) -> dict[int, tuple[str, ...]]:
        locations: dict[int, list[str]] = defaultdict(list)
        for node_id, node in lattice.items():
            for pattern_id in node.pattern_ids:
                locations[int(pattern_id)].append(node_id)

        role_plans: dict[int, tuple[str, ...]] = {}
        for role_id in self.roles:
            authorized_patterns = {
                int(pattern_id)
                for pattern_id, pattern in self.patterns.items()
                if int(role_id) in pattern.role_ids and int(pattern.vector_count) > 0
            }
            if not authorized_patterns:
                role_plans[int(role_id)] = tuple()
                continue

            selected: set[str] = set()
            pending: set[int] = set()
            for pattern_id in authorized_patterns:
                candidate_nodes = [node_id for node_id in locations.get(pattern_id, []) if node_id in lattice]
                if len(candidate_nodes) == 1:
                    selected.add(candidate_nodes[0])
                else:
                    pending.add(pattern_id)

            covered = self._covered_patterns(lattice, selected) & authorized_patterns
            pending -= covered
            if exact and pending:
                exact_nodes = self._solve_exact_coverage(lattice, pending, locations, int(role_id))
                if exact_nodes is not None:
                    selected.update(exact_nodes)
                    role_plans[int(role_id)] = tuple(sorted(selected))
                    continue

            self._greedy_extend_coverage(lattice, selected, pending, locations, int(role_id))
            role_plans[int(role_id)] = tuple(sorted(selected))
        return role_plans

    def _solve_exact_coverage(
        self,
        lattice: dict[str, _LatticeNode],
        pending: set[int],
        locations: dict[int, list[str]],
        role_id: int,
    ) -> set[str] | None:
        candidate_node_ids = sorted({
            node_id
            for pattern_id in pending
            for node_id in locations.get(int(pattern_id), [])
            if node_id in lattice
        })
        if not candidate_node_ids:
            return None
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import lil_matrix
        except Exception:
            return None

        pending_list = sorted(int(pattern_id) for pattern_id in pending)
        node_index = {node_id: index for index, node_id in enumerate(candidate_node_ids)}
        pattern_index = {pattern_id: index for index, pattern_id in enumerate(pending_list)}
        matrix = lil_matrix((len(pending_list), len(candidate_node_ids)), dtype=float)
        for node_id in candidate_node_ids:
            node = lattice[node_id]
            column = node_index[node_id]
            for pattern_id in set(node.pattern_ids) & set(pending_list):
                matrix[pattern_index[int(pattern_id)], column] = 1.0
        costs = np.array(
            [self._node_cost_for_role(lattice[node_id], int(role_id), final=True) for node_id in candidate_node_ids],
            dtype=float,
        )
        constraints = LinearConstraint(matrix.tocsr(), np.ones(len(pending_list)), np.full(len(pending_list), np.inf))
        try:
            result = milp(
                c=costs,
                integrality=np.ones(len(candidate_node_ids)),
                bounds=Bounds(np.zeros(len(candidate_node_ids)), np.ones(len(candidate_node_ids))),
                constraints=constraints,
                options={"disp": False},
            )
        except Exception:
            return None
        if not getattr(result, "success", False) or result.x is None:
            return None
        selected = {candidate_node_ids[index] for index, value in enumerate(result.x) if float(value) >= 0.5}
        covered = self._covered_patterns(lattice, selected) & set(pending)
        if set(pending) <= covered:
            return selected
        return None

    def _greedy_extend_coverage(
        self,
        lattice: dict[str, _LatticeNode],
        selected: set[str],
        pending: set[int],
        locations: dict[int, list[str]],
        role_id: int,
    ) -> None:
        while pending:
            best_node_id = None
            best_rank = None
            fallback_node_id = None
            fallback_rank = None
            for pattern_id in sorted(pending):
                for candidate_node_id in locations.get(pattern_id, []):
                    node = lattice.get(candidate_node_id)
                    if node is None:
                        continue
                    cover_count = len((set(node.pattern_ids) & pending))
                    rank = (
                        -max(1, cover_count),
                        self._node_cost_for_role(node, int(role_id), final=True),
                        self._node_size(node),
                        candidate_node_id,
                    )
                    if cover_count > 0 and (best_rank is None or rank < best_rank):
                        best_rank = rank
                        best_node_id = candidate_node_id
                    if fallback_rank is None or rank < fallback_rank:
                        fallback_rank = rank
                        fallback_node_id = candidate_node_id
            if best_node_id is None:
                best_node_id = fallback_node_id
            if best_node_id is None or best_node_id not in lattice:
                break
            selected.add(best_node_id)
            pending -= (set(lattice[best_node_id].pattern_ids) & pending)

    def _translate_role_plans(
        self,
        role_plans: dict[int, tuple[str, ...]],
        node_id_map: dict[str, str],
    ) -> dict[int, tuple[str, ...]]:
        translated: dict[int, tuple[str, ...]] = {}
        for role_id, node_ids in role_plans.items():
            mapped: list[str] = []
            for node_id in node_ids:
                mapped_id = node_id_map.get(str(node_id))
                if mapped_id is not None:
                    mapped.append(mapped_id)
            translated[int(role_id)] = tuple(sorted(dict.fromkeys(mapped)))
        return translated

    def _covered_patterns(self, lattice: dict[str, _LatticeNode], node_ids: Iterable[str]) -> set[int]:
        covered: set[int] = set()
        for node_id in node_ids:
            node = lattice.get(node_id)
            if node is not None:
                covered.update(int(pattern_id) for pattern_id in node.pattern_ids)
        return covered

    def _materializable_nodes(self, lattice: dict[str, _LatticeNode]) -> list[VedaNode]:
        nodes: list[VedaNode] = []
        for ordinal, node in enumerate(sorted(lattice.values(), key=lambda item: (item.node_id, item.role_ids)), start=1):
            pattern_ids = tuple(sorted(int(pattern_id) for pattern_id in node.pattern_ids))
            document_pattern_pairs = []
            document_ids = []
            for pattern_id in pattern_ids:
                pattern = self.patterns[int(pattern_id)]
                for document_id in pattern.document_ids:
                    document_pattern_pairs.append((int(document_id), int(pattern_id)))
                    document_ids.append(int(document_id))
            vector_count = self._node_size(node)
            node_kind = "index" if int(vector_count) >= self.indexing_threshold else "leftover"
            stable_node_id = f"{ordinal}_{node.node_id}"
            nodes.append(
                VedaNode(
                    node_id=stable_node_id,
                    role_ids=node.role_ids,
                    pattern_ids=pattern_ids,
                    document_ids=tuple(sorted(set(document_ids))),
                    document_pattern_pairs=tuple(sorted(document_pattern_pairs)),
                    vector_count=int(vector_count),
                    node_kind=node_kind,
                    table_name=get_node_table_name(stable_node_id),
                    metadata={
                        "source_node_id": node.node_id,
                        "virtual_components": list(node.virtual_components),
                    },
                )
            )
        return nodes

    def _build_user_routes(
        self,
        nodes: list[VedaNode],
        role_plans: dict[int, tuple[str, ...]],
        user_roles: dict[int, tuple[int, ...]],
    ) -> list[VedaRoute]:
        source_to_materialized = {
            str(node.metadata.get("source_node_id", node.node_id)): node
            for node in nodes
        }
        materialized_by_id = {node.node_id: node for node in nodes}
        node_lookup = {**source_to_materialized, **materialized_by_id}
        routes: list[VedaRoute] = []
        pattern_roles = {int(pattern_id): set(pattern.role_ids) for pattern_id, pattern in self.patterns.items()}
        pattern_vectors = {int(pattern_id): int(pattern.vector_count) for pattern_id, pattern in self.patterns.items()}
        for user_id, roles in sorted(user_roles.items()):
            user_role_set = set(int(role_id) for role_id in roles)
            selected_nodes: dict[str, VedaNode] = {}
            for role_id in user_role_set:
                for planned_node_id in role_plans.get(int(role_id), ()):
                    node = node_lookup.get(planned_node_id)
                    if node is not None:
                        selected_nodes[node.node_id] = node
            for node in selected_nodes.values():
                accessible_patterns = tuple(
                    int(pattern_id)
                    for pattern_id in node.pattern_ids
                    if pattern_roles[int(pattern_id)] & user_role_set
                )
                if not accessible_patterns:
                    continue
                accessible_vectors = int(sum(pattern_vectors[pattern_id] for pattern_id in accessible_patterns))
                impurity = float(node.vector_count) / float(max(1, accessible_vectors))
                route_kind = str(node.node_kind)
                if impurity > 1.000001 and route_kind == "index":
                    route_kind = "impure_index"
                routes.append(
                    VedaRoute(
                        user_id=int(user_id),
                        node_id=node.node_id,
                        table_name=node.table_name,
                        route_kind=route_kind,
                        pattern_ids=accessible_patterns,
                        node_vector_count=int(node.vector_count),
                        accessible_vector_count=int(accessible_vectors),
                        impurity_factor=float(impurity),
                    )
                )
        routes.sort(key=lambda route: (route.user_id, route.route_kind, route.node_id))
        return routes

    def _role_index(self, lattice: dict[str, _LatticeNode]) -> dict[tuple[int, ...], str]:
        return {node.role_ids: node_id for node_id, node in lattice.items()}

    def _new_node_id(self, prefix: str, role_ids: Iterable[int], *, suffix: str | None = None) -> str:
        self._generated_node_counter += 1
        base = role_key(role_ids)
        parts = [prefix, str(self._generated_node_counter), base]
        if suffix:
            parts.append(str(suffix))
        return "_".join(parts)

    def _hnsw_cost(self, size: int, ef_search: int) -> float:
        return float(self.cost_a * math.log2(max(1, int(size)) + 1) + self.cost_b * max(1, int(ef_search)) + self.cost_c)
