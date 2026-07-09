from __future__ import annotations

from collections import defaultdict
import heapq
from dataclasses import dataclass, field
import itertools
import math
import time
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
    pattern_bits: int = 0
    role_bits: int = 0
    vector_count: int = -1


@dataclass(slots=True)
class _RolePlanSeed:
    selected_counts: dict[str, int]
    pending_bits: int


class _LocationOverlay:
    __slots__ = ("base", "overrides")

    def __init__(self, base, overrides: dict[int, list[str]]) -> None:
        self.base = base
        self.overrides = overrides

    def get(self, key, default=None):
        key = int(key)
        if key in self.overrides:
            return self.overrides[key]
        return self.base.get(key, default)


class _LatticeOverlay:
    __slots__ = ("base", "overrides", "removed")

    def __init__(
        self,
        base: dict[str, _LatticeNode],
        overrides: dict[str, _LatticeNode],
        removed: Iterable[str] = tuple(),
    ) -> None:
        self.base = base
        self.overrides = {str(node_id): node for node_id, node in overrides.items()}
        self.removed = frozenset(str(node_id) for node_id in removed)

    def get(self, key, default=None):
        key = str(key)
        if key in self.removed:
            return default
        if key in self.overrides:
            return self.overrides[key]
        return self.base.get(key, default)

    def __contains__(self, key) -> bool:
        key = str(key)
        return key not in self.removed and (key in self.overrides or key in self.base)

    def materialize(self) -> dict[str, _LatticeNode]:
        lattice = dict(self.base)
        for node_id in self.removed:
            lattice.pop(str(node_id), None)
        lattice.update(self.overrides)
        return lattice


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
        leftover_fixed_cost: float | None = None,
        max_veda_rounds: int = 8,
    ) -> None:
        self.indexing_threshold = max(1, int(indexing_threshold))
        self.storage_amplification = max(1.0, float(storage_amplification))
        self.ef_search = max(1, int(ef_search))
        self.cost_a = float(cost_a)
        self.cost_b = float(cost_b)
        self.cost_c = float(cost_c)
        self.linear_scan_cost = max(0.0, float(linear_scan_cost))
        self.leftover_fixed_cost = max(0.0, float(self.cost_c if leftover_fixed_cost is None else leftover_fixed_cost))
        self.max_veda_rounds = max(1, int(max_veda_rounds))

        self.patterns: dict[int, VedaPattern] = {}
        self.roles: tuple[int, ...] = tuple()
        self.exclusive_lattice: dict[str, _LatticeNode] = {}
        self.post_copy_lattice: dict[str, _LatticeNode] = {}
        self.node_by_roles: dict[tuple[int, ...], str] = {}
        self._generated_node_counter = 0
        self._pattern_sizes: dict[int, int] = {}
        self._pattern_roles: dict[int, frozenset[int]] = {}
        self._role_authorized_patterns: dict[int, frozenset[int]] = {}
        self._node_size_cache: dict[frozenset[int], int] = {}
        self._authorized_size_cache: dict[tuple[frozenset[int], int], int] = {}
        self._node_cost_cache: dict[tuple[frozenset[int], int, bool], float] = {}
        self._pattern_bits: dict[int, int] = {}
        self._bit_to_pattern: dict[int, int] = {}
        self._role_authorized_bits: dict[int, int] = {}
        self._role_bits: dict[int, int] = {}
        self._pattern_bits_size_cache: dict[int, int] = {}
        self._pattern_bits_roles_cache: dict[int, tuple[int, ...]] = {}
        self._pattern_bits_role_bits_cache: dict[int, int] = {}
        self._milp_backend = None
        self._milp_backend_checked = False
        self._active_algorithm = "effveda"

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

        self._pattern_sizes = {
            int(pattern.pattern_id): max(0, int(pattern.vector_count))
            for pattern in patterns
        }
        self._pattern_roles = {
            int(pattern.pattern_id): frozenset(int(role_id) for role_id in pattern.role_ids)
            for pattern in patterns
        }
        self._role_authorized_patterns = {
            int(role_id): frozenset(
                int(pattern.pattern_id)
                for pattern in patterns
                if int(role_id) in pattern.role_ids and int(pattern.vector_count) > 0
            )
            for role_id in self.roles
        }
        ordered_role_ids = sorted({int(role_id) for role_id in self.roles} | {int(role_id) for pattern in patterns for role_id in pattern.role_ids})
        self._role_bits = {int(role_id): 1 << index for index, role_id in enumerate(ordered_role_ids)}
        ordered_pattern_ids = sorted(int(pattern.pattern_id) for pattern in patterns)
        self._pattern_bits = {pattern_id: 1 << index for index, pattern_id in enumerate(ordered_pattern_ids)}
        self._bit_to_pattern = {index: pattern_id for index, pattern_id in enumerate(ordered_pattern_ids)}
        self._role_authorized_bits = {
            int(role_id): self._patterns_to_bits(pattern_ids)
            for role_id, pattern_ids in self._role_authorized_patterns.items()
        }
        self._node_size_cache.clear()
        self._authorized_size_cache.clear()
        self._node_cost_cache.clear()
        self._pattern_bits_size_cache.clear()
        self._pattern_bits_roles_cache.clear()
        self._pattern_bits_role_bits_cache.clear()
        self._milp_backend = None
        self._milp_backend_checked = False

        self.exclusive_lattice = self._build_exclusive_lattice(patterns)
        self.node_by_roles = {
            node.role_ids: node_id
            for node_id, node in self.exclusive_lattice.items()
        }
        algorithm = normalize_algorithm(algorithm)
        self._active_algorithm = str(algorithm)
        lattice = self._clone_lattice(self.exclusive_lattice)

        if algorithm == "veda":
            lattice, operation_metadata = self._run_veda(lattice, show_progress=show_progress)
        else:
            lattice, operation_metadata = self._run_effveda(lattice, show_progress=show_progress)

        lattice, finalize_metadata = self._finalize_lattice(lattice)
        role_plans = self._build_role_query_plans(lattice, exact=True, final=True)
        if algorithm == "veda":
            lattice, role_plans, super_impure_metadata = self._handle_super_impure_nodes(lattice, role_plans)
        else:
            super_impure_metadata = {"enabled": False, "reason": "effveda_planner_path"}

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
            "linear_scan_cost": float(self.linear_scan_cost),
            "leftover_fixed_cost": float(self.leftover_fixed_cost),
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

    def _make_lattice_node(
        self,
        *,
        node_id: str,
        role_ids: Iterable[int],
        pattern_ids: Iterable[int] | int,
        virtual_components: Iterable[str] = tuple(),
    ) -> _LatticeNode:
        if isinstance(pattern_ids, int):
            normalized_patterns = frozenset({int(pattern_ids)})
        elif isinstance(pattern_ids, frozenset):
            normalized_patterns = pattern_ids
        else:
            normalized_patterns = frozenset(int(pattern_id) for pattern_id in pattern_ids)
        normalized_roles = normalize_int_tuple(role_ids)
        pattern_bits = self._patterns_to_bits(normalized_patterns)
        role_bits = self._roles_to_bits(normalized_roles)
        vector_count = self._pattern_bits_size(pattern_bits) if pattern_bits else 0
        return _LatticeNode(
            node_id=str(node_id),
            role_ids=normalized_roles,
            pattern_ids=normalized_patterns,
            virtual_components=tuple(virtual_components),
            pattern_bits=int(pattern_bits),
            role_bits=int(role_bits),
            vector_count=int(vector_count),
        )

    def _build_exclusive_lattice(self, patterns: list[VedaPattern]) -> dict[str, _LatticeNode]:
        lattice: dict[str, _LatticeNode] = {}
        for pattern in patterns:
            node_id = role_key(pattern.role_ids)
            lattice[node_id] = self._make_lattice_node(
                node_id=node_id,
                role_ids=pattern.role_ids,
                pattern_ids=int(pattern.pattern_id),
                virtual_components=(node_id,),
            )
        return lattice

    def _clone_lattice(self, lattice: dict[str, _LatticeNode]) -> dict[str, _LatticeNode]:
        return {
            node_id: self._make_lattice_node(
                node_id=node.node_id,
                role_ids=node.role_ids,
                pattern_ids=node.pattern_ids,
                virtual_components=node.virtual_components,
            )
            for node_id, node in lattice.items()
        }


    def _patterns_to_bits(self, pattern_ids: Iterable[int]) -> int:
        bits = 0
        for pattern_id in pattern_ids:
            bits |= int(self._pattern_bits.get(int(pattern_id), 0))
        return int(bits)

    def _timing_add(self, timing: Optional[dict[str, dict[str, float | int]]], key: str, elapsed: float) -> None:
        if timing is None:
            return
        seconds = timing.setdefault("seconds", {})
        seconds[key] = float(seconds.get(key, 0.0)) + float(elapsed)

    def _timing_count(self, timing: Optional[dict[str, dict[str, float | int]]], key: str, count: int = 1) -> None:
        if timing is None:
            return
        counts = timing.setdefault("counts", {})
        counts[key] = int(counts.get(key, 0)) + int(count)

    def _node_pattern_bits(self, node: _LatticeNode) -> int:
        if int(node.pattern_bits) != 0 or not node.pattern_ids:
            return int(node.pattern_bits)
        bits = self._patterns_to_bits(node.pattern_ids)
        return int(bits)

    def _roles_to_bits(self, role_ids: Iterable[int]) -> int:
        bits = 0
        for role_id in role_ids:
            bits |= int(self._role_bits.get(int(role_id), 0))
        return int(bits)

    def _role_bits_to_roles(self, bits: int) -> tuple[int, ...]:
        bits = int(bits)
        return tuple(role_id for role_id, role_bit in self._role_bits.items() if bits & int(role_bit))

    def _roles_for_pattern_bits_bits(self, bits: int) -> int:
        bits = int(bits)
        cached = self._pattern_bits_role_bits_cache.get(bits)
        if cached is not None:
            return int(cached)
        role_bits = 0
        value = bits
        while value:
            low = value & -value
            index = low.bit_length() - 1
            pattern_id = self._bit_to_pattern.get(index)
            if pattern_id is not None:
                role_bits |= int(self._roles_to_bits(self._pattern_roles.get(int(pattern_id), frozenset())))
            value ^= low
        self._pattern_bits_role_bits_cache[bits] = int(role_bits)
        return int(role_bits)

    def _node_role_bits(self, node: _LatticeNode) -> int:
        if int(node.role_bits) != 0 or not node.role_ids:
            return int(node.role_bits)
        return int(self._roles_to_bits(node.role_ids))

    def _pattern_bits_size(self, bits: int) -> int:
        bits = int(bits)
        cached = self._pattern_bits_size_cache.get(bits)
        if cached is not None:
            return int(cached)
        total = 0
        value = bits
        while value:
            low = value & -value
            index = low.bit_length() - 1
            pattern_id = self._bit_to_pattern.get(index)
            if pattern_id is not None:
                total += int(self._pattern_sizes.get(int(pattern_id), 0))
            value ^= low
        self._pattern_bits_size_cache[bits] = int(total)
        return int(total)

    def _roles_for_pattern_bits(self, bits: int) -> tuple[int, ...]:
        bits = int(bits)
        cached = self._pattern_bits_roles_cache.get(bits)
        if cached is not None:
            return cached
        roles: set[int] = set()
        value = bits
        while value:
            low = value & -value
            index = low.bit_length() - 1
            pattern_id = self._bit_to_pattern.get(index)
            if pattern_id is not None:
                roles.update(self._pattern_roles.get(int(pattern_id), frozenset()))
            value ^= low
        result = tuple(sorted(roles))
        self._pattern_bits_roles_cache[bits] = result
        return result

    def _bits_to_patterns(self, bits: int) -> set[int]:
        values: set[int] = set()
        value = int(bits)
        while value:
            low = value & -value
            index = low.bit_length() - 1
            pattern_id = self._bit_to_pattern.get(index)
            if pattern_id is not None:
                values.add(int(pattern_id))
            value ^= low
        return values

    def _exclusive_vector_count(self) -> int:
        if self._pattern_sizes:
            return int(sum(self._pattern_sizes.values()))
        return int(sum(max(0, int(pattern.vector_count)) for pattern in self.patterns.values()))

    def _node_size(self, node: _LatticeNode | Iterable[int]) -> int:
        if isinstance(node, _LatticeNode):
            if int(node.vector_count) >= 0:
                return int(node.vector_count)
            pattern_bits = self._node_pattern_bits(node)
            if pattern_bits:
                return int(self._pattern_bits_size(pattern_bits))
            pattern_ids = node.pattern_ids
        else:
            pattern_ids = frozenset(int(pattern_id) for pattern_id in node)
        cached = self._node_size_cache.get(pattern_ids)
        if cached is not None:
            return int(cached)
        total = int(
            sum(
                self._pattern_sizes.get(int(pattern_id), max(0, int(self.patterns[int(pattern_id)].vector_count)))
                for pattern_id in pattern_ids
            )
        )
        self._node_size_cache[pattern_ids] = int(total)
        return int(total)

    def _authorized_size(self, node: _LatticeNode, role_id: int) -> int:
        role_id = int(role_id)
        key = (node.pattern_ids, role_id)
        cached = self._authorized_size_cache.get(key)
        if cached is not None:
            return int(cached)
        authorized_bits = int(self._node_pattern_bits(node)) & int(self._role_authorized_bits.get(role_id, 0))
        if authorized_bits:
            total = self._pattern_bits_size(authorized_bits)
        else:
            authorized_patterns = self._role_authorized_patterns.get(role_id)
            if authorized_patterns is None:
                authorized_patterns = frozenset(
                    int(pattern_id)
                    for pattern_id, pattern in self.patterns.items()
                    if role_id in pattern.role_ids and int(pattern.vector_count) > 0
                )
            if len(node.pattern_ids) <= len(authorized_patterns):
                total = sum(
                    self._pattern_sizes.get(int(pattern_id), max(0, int(self.patterns[int(pattern_id)].vector_count)))
                    for pattern_id in node.pattern_ids
                    if int(pattern_id) in authorized_patterns
                )
            else:
                total = sum(
                    self._pattern_sizes.get(int(pattern_id), max(0, int(self.patterns[int(pattern_id)].vector_count)))
                    for pattern_id in authorized_patterns
                    if int(pattern_id) in node.pattern_ids
                )
        self._authorized_size_cache[key] = int(total)
        return int(total)

    def _leftover_cost_for_role(self, authorized_size: int, leftover_count: int = 1) -> float:
        if int(authorized_size) <= 0:
            return float("inf")
        return float(max(1, int(leftover_count)) * self.leftover_fixed_cost + self.linear_scan_cost * int(authorized_size))

    def _node_cost_for_role(self, node: _LatticeNode, role_id: int, *, final: bool = False) -> float:
        role_id = int(role_id)
        key = (node.pattern_ids, role_id, bool(final))
        cached = self._node_cost_cache.get(key)
        if cached is not None:
            return float(cached)
        node_size = max(1, int(self._node_size(node)))
        authorized_size = int(self._authorized_size(node, role_id))
        if authorized_size <= 0:
            return float("inf")
        if node_size < self.indexing_threshold:
            authorized_bits = int(self._node_pattern_bits(node)) & int(self._role_authorized_bits.get(role_id, 0))
            leftover_count = int(authorized_bits.bit_count()) if authorized_bits else 1
            cost = self._leftover_cost_for_role(authorized_size, leftover_count)
            self._node_cost_cache[key] = cost
            return cost
        impurity = float(node_size) / float(max(1, authorized_size))
        cost = float(self.cost_a * math.log2(node_size + 1) + self.cost_b * impurity * self.ef_search + self.cost_c)
        self._node_cost_cache[key] = cost
        return cost

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


    def _role_plan_cost(
        self,
        lattice: dict[str, _LatticeNode],
        role_id: int,
        node_ids: Iterable[str],
        *,
        final: bool = False,
    ) -> float:
        total = 0.0
        for node_id in node_ids:
            node = lattice.get(str(node_id))
            if node is None:
                continue
            total += self._node_cost_for_role(node, int(role_id), final=final)
        return float(total)

    def _role_costs(
        self,
        lattice: dict[str, _LatticeNode],
        role_plans: dict[int, tuple[str, ...]],
        *,
        final: bool = False,
    ) -> dict[int, float]:
        return {
            int(role_id): self._role_plan_cost(lattice, int(role_id), node_ids, final=final)
            for role_id, node_ids in role_plans.items()
        }

    def _lattice_locations(self, lattice: dict[str, _LatticeNode]) -> dict[int, list[str]]:
        locations: dict[int, list[str]] = defaultdict(list)
        for node_id, node in lattice.items():
            for pattern_id in node.pattern_ids:
                locations[int(pattern_id)].append(str(node_id))
        return locations

    def _authorized_patterns_for_role(self, role_id: int) -> set[int]:
        role_id = int(role_id)
        cached = self._role_authorized_patterns.get(role_id)
        if cached is not None:
            return set(cached)
        return {
            int(pattern_id)
            for pattern_id, pattern in self.patterns.items()
            if role_id in pattern.role_ids and int(pattern.vector_count) > 0
        }

    def _build_role_query_plan(
        self,
        lattice: dict[str, _LatticeNode],
        role_id: int,
        *,
        exact: bool = False,
        final: bool = False,
        locations: Optional[dict[int, list[str]]] = None,
        sort_output: bool = True,
    ) -> tuple[str, ...]:
        role_id = int(role_id)
        authorized_patterns = self._authorized_patterns_for_role(role_id)
        if not authorized_patterns:
            return tuple()

        if locations is None:
            locations = self._lattice_locations(lattice)

        selected: set[str] = set()
        pending_bits = 0
        for pattern_id in authorized_patterns:
            single_node_id = None
            candidate_count = 0
            for candidate_node_id in locations.get(int(pattern_id), []):
                if candidate_node_id in lattice:
                    candidate_count += 1
                    if candidate_count == 1:
                        single_node_id = str(candidate_node_id)
                    else:
                        break
            if candidate_count == 1 and single_node_id is not None:
                selected.add(single_node_id)
            else:
                pending_bits |= int(self._pattern_bits.get(int(pattern_id), 0))

        covered_bits = 0
        for node_id in selected:
            node = lattice.get(node_id)
            if node is not None:
                covered_bits |= self._node_pattern_bits(node)
        pending_bits &= int(self._role_authorized_bits.get(role_id, self._patterns_to_bits(authorized_patterns)))
        pending_bits &= ~covered_bits

        if exact and pending_bits:
            pending = self._bits_to_patterns(pending_bits)
            exact_nodes = self._solve_exact_coverage(lattice, pending, locations, role_id, final=final)
            if exact_nodes is not None:
                selected.update(exact_nodes)
                return tuple(sorted(selected))

        self._greedy_extend_coverage_bits(lattice, selected, pending_bits, locations, role_id, final=final)
        return tuple(sorted(selected)) if sort_output else tuple(selected)

    def _single_owner_for_pattern(self, lattice, locations, pattern_id: int) -> str | None:
        single_node_id = None
        candidate_count = 0
        for candidate_node_id in locations.get(int(pattern_id), []):
            if candidate_node_id in lattice:
                candidate_count += 1
                if candidate_count == 1:
                    single_node_id = str(candidate_node_id)
                else:
                    break
        return single_node_id if candidate_count == 1 else None

    def _build_role_query_seed(self, lattice, role_id: int, locations) -> _RolePlanSeed:
        role_id = int(role_id)
        authorized_patterns = self._authorized_patterns_for_role(role_id)
        selected_counts: dict[str, int] = {}
        pending_bits = 0
        for pattern_id in authorized_patterns:
            single_node_id = self._single_owner_for_pattern(lattice, locations, int(pattern_id))
            if single_node_id is None:
                pending_bits |= int(self._pattern_bits.get(int(pattern_id), 0))
            else:
                selected_counts[single_node_id] = int(selected_counts.get(single_node_id, 0)) + 1
        return _RolePlanSeed(selected_counts=selected_counts, pending_bits=int(pending_bits))

    def _build_role_query_plan_from_seed(
        self,
        candidate_lattice,
        role_id: int,
        locations,
        seed: _RolePlanSeed,
        touched_patterns: Iterable[int],
        *,
        owner_delta: Optional[dict[int, tuple[str | None, str | None]]] = None,
        exact: bool = False,
        final: bool = False,
        sort_output: bool = True,
    ) -> tuple[str, ...]:
        role_id = int(role_id)
        auth_bits = int(self._role_authorized_bits.get(role_id, 0))
        if auth_bits == 0:
            return tuple()
        base_locations = getattr(locations, "base", None)
        base_lattice = getattr(candidate_lattice, "base", None)
        if base_locations is None or base_lattice is None:
            return self._build_role_query_plan(candidate_lattice, role_id, exact=exact, final=final, locations=locations, sort_output=sort_output)

        selected_counts = dict(seed.selected_counts)
        pending_bits = int(seed.pending_bits)
        for pattern_id in touched_patterns:
            pattern_id = int(pattern_id)
            bit = int(self._pattern_bits.get(pattern_id, 0))
            if bit == 0 or not (auth_bits & bit):
                continue

            if owner_delta is not None and pattern_id in owner_delta:
                base_single, candidate_single = owner_delta[pattern_id]
            else:
                base_single = self._single_owner_for_pattern(base_lattice, base_locations, pattern_id)
                candidate_single = self._single_owner_for_pattern(candidate_lattice, locations, pattern_id)
            if base_single is None:
                pending_bits &= ~bit
            else:
                next_count = int(selected_counts.get(base_single, 0)) - 1
                if next_count > 0:
                    selected_counts[base_single] = next_count
                else:
                    selected_counts.pop(base_single, None)

            if candidate_single is None:
                pending_bits |= bit
            else:
                selected_counts[candidate_single] = int(selected_counts.get(candidate_single, 0)) + 1
                pending_bits &= ~bit

        selected = {
            str(node_id)
            for node_id, count in selected_counts.items()
            if int(count) > 0 and str(node_id) in candidate_lattice
        }
        covered_bits = 0
        for node_id in selected:
            node = candidate_lattice.get(node_id)
            if node is not None:
                covered_bits |= self._node_pattern_bits(node)
        pending_bits &= auth_bits
        pending_bits &= ~covered_bits

        if exact and pending_bits:
            pending = self._bits_to_patterns(pending_bits)
            exact_nodes = self._solve_exact_coverage(candidate_lattice, pending, locations, role_id, final=final)
            if exact_nodes is not None:
                selected.update(exact_nodes)
                return tuple(sorted(selected))

        self._greedy_extend_coverage_bits(candidate_lattice, selected, pending_bits, locations, role_id, final=final)
        return tuple(sorted(selected)) if sort_output else tuple(selected)

    def _storage_amplification(self, lattice: dict[str, _LatticeNode]) -> float:
        return float(sum(self._node_size(node) for node in lattice.values()) / max(1, self._exclusive_vector_count()))

    def _descendant_ancestor_pairs(self, lattice: dict[str, _LatticeNode]) -> list[tuple[str, str]]:
        nodes = list(lattice.values())
        role_bits_by_node = {node.node_id: self._node_role_bits(node) for node in nodes}
        role_len_by_node = {node.node_id: len(node.role_ids) for node in nodes}
        nodes_by_role_bits: dict[int, list[_LatticeNode]] = defaultdict(list)
        for node in nodes:
            bits = int(role_bits_by_node.get(node.node_id, 0))
            if bits:
                nodes_by_role_bits[bits].append(node)

        pairs: list[tuple[str, str]] = []
        for child in nodes:
            child_bits = int(role_bits_by_node.get(child.node_id, 0))
            if child_bits == 0:
                continue
            estimated_subsets = (1 << int(child_bits.bit_count())) - 2
            if estimated_subsets < len(nodes):
                ancestor_bits = (child_bits - 1) & child_bits
                while ancestor_bits:
                    for ancestor in nodes_by_role_bits.get(ancestor_bits, ()):
                        if child.node_id != ancestor.node_id:
                            pairs.append((child.node_id, ancestor.node_id))
                    ancestor_bits = (ancestor_bits - 1) & child_bits
            else:
                for ancestor in nodes:
                    if child.node_id == ancestor.node_id:
                        continue
                    ancestor_bits = int(role_bits_by_node.get(ancestor.node_id, 0))
                    if ancestor_bits and ancestor_bits != child_bits and (ancestor_bits & child_bits) == ancestor_bits:
                        pairs.append((child.node_id, ancestor.node_id))
        pairs.sort(key=lambda item: (role_len_by_node[item[0]], role_len_by_node[item[1]], item[0], item[1]), reverse=True)
        return pairs

    def _copy_delta_size(self, lattice: dict[str, _LatticeNode], child_id: str, ancestor_id: str) -> int:
        child_exclusive = self.exclusive_lattice.get(child_id)
        ancestor = lattice.get(ancestor_id)
        if child_exclusive is None or ancestor is None:
            return 0
        missing_bits = self._node_pattern_bits(child_exclusive) & ~self._node_pattern_bits(ancestor)
        return int(self._pattern_bits_size(missing_bits))

    def _simulate_copy(self, lattice: dict[str, _LatticeNode], child_id: str, ancestor_id: str) -> dict[str, _LatticeNode]:
        child_exclusive = self.exclusive_lattice[child_id]
        next_lattice = self._clone_lattice(lattice)
        ancestor = next_lattice[ancestor_id]
        next_lattice[ancestor_id] = self._make_lattice_node(
            node_id=ancestor.node_id,
            role_ids=ancestor.role_ids,
            pattern_ids=ancestor.pattern_ids | child_exclusive.pattern_ids,
            virtual_components=tuple(dict.fromkeys((*ancestor.virtual_components, *child_exclusive.virtual_components))),
        )
        return next_lattice

    def _simulate_merge(self, lattice: dict[str, _LatticeNode], child_id: str, ancestor_id: str) -> dict[str, _LatticeNode]:
        next_lattice = self._clone_lattice(lattice)
        child = next_lattice[child_id]
        ancestor = next_lattice[ancestor_id]
        next_lattice[ancestor_id] = self._make_lattice_node(
            node_id=ancestor.node_id,
            role_ids=ancestor.role_ids,
            pattern_ids=ancestor.pattern_ids | child.pattern_ids,
            virtual_components=tuple(dict.fromkeys((*ancestor.virtual_components, *child.virtual_components))),
        )
        del next_lattice[child_id]
        return next_lattice


    def _candidate_copy_lattice(
        self,
        lattice: dict[str, _LatticeNode],
        child_id: str,
        ancestor_id: str,
    ) -> _LatticeOverlay | None:
        child_exclusive = self.exclusive_lattice.get(child_id)
        ancestor = lattice.get(ancestor_id)
        if child_exclusive is None or ancestor is None:
            return None
        new_ancestor = self._make_lattice_node(
            node_id=ancestor.node_id,
            role_ids=ancestor.role_ids,
            pattern_ids=ancestor.pattern_ids | child_exclusive.pattern_ids,
            virtual_components=tuple(dict.fromkeys((*ancestor.virtual_components, *child_exclusive.virtual_components))),
        )
        return _LatticeOverlay(lattice, {str(ancestor_id): new_ancestor})

    def _candidate_merge_lattice(
        self,
        lattice: dict[str, _LatticeNode],
        child_id: str,
        ancestor_id: str,
    ) -> _LatticeOverlay | None:
        child = lattice.get(child_id)
        ancestor = lattice.get(ancestor_id)
        if child is None or ancestor is None:
            return None
        new_ancestor = self._make_lattice_node(
            node_id=ancestor.node_id,
            role_ids=ancestor.role_ids,
            pattern_ids=ancestor.pattern_ids | child.pattern_ids,
            virtual_components=tuple(dict.fromkeys((*ancestor.virtual_components, *child.virtual_components))),
        )
        return _LatticeOverlay(lattice, {str(ancestor_id): new_ancestor}, removed=(str(child_id),))

    def _roles_for_patterns(self, pattern_ids: Iterable[int]) -> set[int]:
        roles: set[int] = set()
        if self._pattern_roles:
            for pattern_id in pattern_ids:
                roles.update(self._pattern_roles.get(int(pattern_id), frozenset()))
            return roles
        for pattern_id in pattern_ids:
            pattern = self.patterns.get(int(pattern_id))
            if pattern is not None:
                roles.update(int(role_id) for role_id in pattern.role_ids)
        return roles

    def _copy_affected_roles(
        self,
        lattice: dict[str, _LatticeNode],
        role_plans: dict[int, tuple[str, ...]],
        child_id: str,
        ancestor_id: str,
        role_plan_index: Optional[dict[str, set[int]]] = None,
    ) -> tuple[int, ...]:
        if role_plan_index is None:
            affected = {
                int(role_id)
                for role_id, node_ids in role_plans.items()
                if str(ancestor_id) in node_ids
            }
        else:
            affected = {int(role_id) for role_id in role_plan_index.get(str(ancestor_id), set())}

        child = self.exclusive_lattice.get(str(child_id)) or lattice.get(str(child_id))
        ancestor = lattice.get(str(ancestor_id))
        if child is not None:
            copied_bits = self._node_pattern_bits(child)
            if ancestor is not None:
                copied_bits &= ~self._node_pattern_bits(ancestor)
            affected_bits = self._roles_to_bits(affected) | self._roles_for_pattern_bits_bits(copied_bits)
            return self._role_bits_to_roles(affected_bits)
        return tuple(sorted(affected))

    def _merge_affected_roles(
        self,
        lattice: dict[str, _LatticeNode],
        role_plans: dict[int, tuple[str, ...]],
        child_id: str,
        ancestor_id: str,
        role_plan_index: Optional[dict[str, set[int]]] = None,
    ) -> tuple[int, ...]:
        if role_plan_index is None:
            affected = {
                int(role_id)
                for role_id, node_ids in role_plans.items()
                if str(child_id) in node_ids or str(ancestor_id) in node_ids
            }
        else:
            affected = set()
            affected.update(int(role_id) for role_id in role_plan_index.get(str(child_id), set()))
            affected.update(int(role_id) for role_id in role_plan_index.get(str(ancestor_id), set()))

        touched_bits = 0
        child = lattice.get(str(child_id))
        ancestor = lattice.get(str(ancestor_id))
        if child is not None:
            touched_bits |= self._node_pattern_bits(child)
        if ancestor is not None:
            touched_bits |= self._node_pattern_bits(ancestor)
        affected_bits = self._roles_to_bits(affected) | self._roles_for_pattern_bits_bits(touched_bits)
        return self._role_bits_to_roles(affected_bits)

    def _candidate_locations_overlay_for_patterns(
        self,
        base_locations: dict[int, list[str]],
        candidate_lattice: dict[str, _LatticeNode],
        changed_node_ids: Iterable[str],
        touched_patterns: Iterable[int],
    ) -> _LocationOverlay:
        touched_tuple = tuple(dict.fromkeys(int(pattern_id) for pattern_id in touched_patterns))
        if not touched_tuple:
            return _LocationOverlay(base_locations, {})
        changed = tuple(dict.fromkeys(str(node_id) for node_id in changed_node_ids))
        changed_nodes = [(node_id, candidate_lattice.get(str(node_id))) for node_id in changed]
        overrides: dict[int, list[str]] = {}
        for pattern_id in touched_tuple:
            owners = [
                str(node_id)
                for node_id in base_locations.get(pattern_id, [])
                if str(node_id) in candidate_lattice
            ]
            owner_set = set(owners)
            for node_id, node in changed_nodes:
                if node is not None and pattern_id in node.pattern_ids and node_id not in owner_set:
                    owners.append(str(node_id))
                    owner_set.add(str(node_id))
            overrides[pattern_id] = owners
        return _LocationOverlay(base_locations, overrides)

    def _candidate_locations_overlay(
        self,
        base_locations: dict[int, list[str]],
        candidate_lattice: dict[str, _LatticeNode],
        changed_node_ids: Iterable[str],
    ) -> _LocationOverlay:
        changed = tuple(dict.fromkeys(str(node_id) for node_id in changed_node_ids))
        touched_patterns: set[int] = set()
        for node_id in changed:
            node = candidate_lattice.get(str(node_id))
            if node is not None:
                touched_patterns.update(int(pattern_id) for pattern_id in node.pattern_ids)
        return self._candidate_locations_overlay_for_patterns(base_locations, candidate_lattice, changed, touched_patterns)


    def _replace_locations_for_nodes(
        self,
        locations: dict[int, list[str]],
        old_nodes: Iterable[_LatticeNode | None],
        new_nodes: Iterable[_LatticeNode | None],
    ) -> None:
        old_nodes_tuple = tuple(node for node in old_nodes if node is not None)
        new_nodes_tuple = tuple(node for node in new_nodes if node is not None)
        old_node_ids = {str(node.node_id) for node in old_nodes_tuple}
        touched_patterns: set[int] = set()
        for node in old_nodes_tuple:
            touched_patterns.update(int(pattern_id) for pattern_id in node.pattern_ids)
        for node in new_nodes_tuple:
            touched_patterns.update(int(pattern_id) for pattern_id in node.pattern_ids)

        for pattern_id in touched_patterns:
            owners = [
                str(node_id)
                for node_id in locations.get(int(pattern_id), [])
                if node_id and str(node_id) not in old_node_ids
            ]
            owner_set = set(owners)
            for node in new_nodes_tuple:
                node_id = str(node.node_id)
                if int(pattern_id) in node.pattern_ids and node_id not in owner_set:
                    owners.append(node_id)
                    owner_set.add(node_id)
            if owners:
                locations[int(pattern_id)] = owners
            else:
                locations.pop(int(pattern_id), None)

    def _candidate_plan_delta(
        self,
        lattice: dict[str, _LatticeNode],
        candidate_lattice: dict[str, _LatticeNode],
        role_plans: dict[int, tuple[str, ...]],
        role_costs: dict[int, float],
        affected_roles: Iterable[int],
        locations,
        role_plan_seed_cache: Optional[dict[int, _RolePlanSeed]] = None,
        touched_patterns: Optional[Iterable[int]] = None,
        collect_updates: bool = True,
        sort_plans: bool = True,
        non_increasing: bool = False,
        full_recompute: bool = False,
        timing: Optional[dict[str, dict[str, float | int]]] = None,
    ) -> tuple[float, dict[int, tuple[tuple[str, ...], float]]]:
        affected = tuple(dict.fromkeys(int(role_id) for role_id in affected_roles))
        if not affected:
            return 0.0, {}

        updates: dict[int, tuple[tuple[str, ...], float]] = {}
        total_delta = 0.0
        owner_delta = None
        if (
            not full_recompute
            and role_plan_seed_cache is not None
            and isinstance(locations, _LocationOverlay)
        ):
            owner_delta = {
                int(pattern_id): (
                    self._single_owner_for_pattern(lattice, locations.base, int(pattern_id)),
                    self._single_owner_for_pattern(candidate_lattice, locations, int(pattern_id)),
                )
                for pattern_id in (locations.overrides.keys() if touched_patterns is None else touched_patterns)
            }
        self._timing_count(timing, "role_delta_evaluations", len(affected))
        started = time.perf_counter()
        for role_id in affected:
            old_cost = role_costs.get(int(role_id))
            if old_cost is None:
                old_cost = self._role_plan_cost(lattice, int(role_id), role_plans.get(int(role_id), tuple()))
            if (
                not full_recompute
                and role_plan_seed_cache is not None
                and isinstance(locations, _LocationOverlay)
            ):
                seed = role_plan_seed_cache.get(int(role_id))
                if seed is None:
                    seed = self._build_role_query_seed(lattice, int(role_id), locations.base)
                    role_plan_seed_cache[int(role_id)] = seed
                new_plan = self._build_role_query_plan_from_seed(
                    candidate_lattice,
                    int(role_id),
                    locations,
                    seed,
                    locations.overrides.keys() if touched_patterns is None else touched_patterns,
                    owner_delta=owner_delta,
                    sort_output=sort_plans,
                )
            else:
                new_plan = self._build_role_query_plan(candidate_lattice, int(role_id), locations=locations, sort_output=sort_plans)
            new_cost = self._role_plan_cost(candidate_lattice, int(role_id), new_plan)
            if non_increasing and float(new_cost) > float(old_cost):
                new_plan = tuple(role_plans.get(int(role_id), tuple()))
                new_cost = float(old_cost)
            scored_new_cost = float(new_cost)
            total_delta += float(old_cost) - float(scored_new_cost)
            if collect_updates:
                updates[int(role_id)] = (new_plan, float(new_cost))
        self._timing_add(timing, "candidate_plan_delta_seconds", time.perf_counter() - started)
        return float(total_delta / max(1, len(role_plans))), updates

    def _role_plan_node_index(self, role_plans: dict[int, tuple[str, ...]]) -> dict[str, set[int]]:
        index: dict[str, set[int]] = defaultdict(set)
        for role_id, node_ids in role_plans.items():
            for node_id in node_ids:
                index[str(node_id)].add(int(role_id))
        return index

    def _apply_role_plan_updates(
        self,
        role_plans: dict[int, tuple[str, ...]],
        role_costs: dict[int, float],
        updates: dict[int, tuple[tuple[str, ...], float]],
        role_plan_index: Optional[dict[str, set[int]]] = None,
    ) -> None:
        for role_id, (new_plan, new_cost) in updates.items():
            role_id = int(role_id)
            old_plan = tuple(role_plans.get(role_id, tuple()))
            new_plan = tuple(new_plan)
            if role_plan_index is not None:
                for node_id in old_plan:
                    roles = role_plan_index.get(str(node_id))
                    if roles is not None:
                        roles.discard(role_id)
                        if not roles:
                            role_plan_index.pop(str(node_id), None)
                for node_id in new_plan:
                    role_plan_index.setdefault(str(node_id), set()).add(role_id)
            role_plans[role_id] = new_plan
            role_costs[role_id] = float(new_cost)

    def _node_log_cost(self, node: _LatticeNode | None) -> float:
        if node is None:
            return 0.0
        return float(math.log2(max(1, self._node_size(node)) + 1))

    def _role_log_plan_cost(self, lattice: dict[str, _LatticeNode], node_ids: Iterable[str]) -> float:
        return float(sum(self._node_log_cost(lattice.get(str(node_id))) for node_id in node_ids))

    def _node_plan_cost_for_role(self, lattice: dict[str, _LatticeNode], node_id: str, role_id: int) -> float:
        node = lattice.get(str(node_id))
        if node is None:
            return float("inf")
        return self._node_cost_for_role(node, int(role_id))

    def _renew_role_plans_for_candidate(
        self,
        candidate_lattice: dict[str, _LatticeNode],
        role_plans: dict[int, tuple[str, ...]],
        affected_roles: Iterable[int],
    ) -> None:
        affected = tuple(dict.fromkeys(int(role_id) for role_id in affected_roles))
        if not affected:
            return
        locations = self._lattice_locations(candidate_lattice)
        for role_id in affected:
            new_plan = self._build_role_query_plan(candidate_lattice, int(role_id), locations=locations)
            role_plans[int(role_id)] = tuple(new_plan)

    def _renew_all_role_plans(
        self,
        lattice: dict[str, _LatticeNode],
        role_plans: dict[int, tuple[str, ...]],
    ) -> None:
        role_plans.clear()
        role_plans.update(self._build_role_query_plans(lattice))


    def _phase_pair_indexes(
        self,
        pairs: Iterable[tuple[str, str]],
    ) -> dict[str, set[tuple[str, str]]]:
        pairs_by_node: dict[str, set[tuple[str, str]]] = defaultdict(set)
        for child_id, ancestor_id in pairs:
            pair = (str(child_id), str(ancestor_id))
            pairs_by_node[pair[0]].add(pair)
            pairs_by_node[pair[1]].add(pair)
        return pairs_by_node

    def _rebuild_copy_heap(
        self,
        scores: dict[tuple[str, str], tuple[float, int]],
        versions: dict[tuple[str, str], int],
    ) -> list[tuple[float, int, tuple[str, str], int]]:
        heap = [
            (-float(score), int(delta), pair, int(versions.get(pair, 0)))
            for pair, (score, delta) in scores.items()
        ]
        heapq.heapify(heap)
        return heap

    def _push_copy_score(
        self,
        heap: list[tuple[float, int, tuple[str, str], int]],
        scores: dict[tuple[str, str], tuple[float, int]],
        versions: dict[tuple[str, str], int],
        pair: tuple[str, str],
        score: float,
        delta: int,
    ) -> None:
        versions[pair] = int(versions.get(pair, 0)) + 1
        scores[pair] = (float(score), int(delta))
        heapq.heappush(heap, (-float(score), int(delta), pair, versions[pair]))


    def _score_veda_copy_pair(
        self,
        lattice: dict[str, _LatticeNode],
        role_plans: dict[int, tuple[str, ...]],
        role_costs: dict[int, float],
        locations: dict[int, list[str]],
        child_id: str,
        ancestor_id: str,
        role_plan_index: Optional[dict[str, set[int]]] = None,
        role_plan_seed_cache: Optional[dict[int, _RolePlanSeed]] = None,
        affected_roles: Optional[Iterable[int]] = None,
        collect_updates: bool = True,
        timing: Optional[dict[str, dict[str, float | int]]] = None,
    ) -> tuple[float, int, _LatticeOverlay, dict[int, tuple[tuple[str, ...], float]]] | None:
        if child_id not in lattice or ancestor_id not in lattice:
            return None
        delta = self._copy_delta_size(lattice, child_id, ancestor_id)
        if delta <= 0:
            return None
        candidate_lattice = self._candidate_copy_lattice(lattice, child_id, ancestor_id)
        if candidate_lattice is None:
            return None

        ancestor = lattice.get(str(ancestor_id))
        child_exclusive = self.exclusive_lattice.get(str(child_id))
        touched_bits = 0
        if child_exclusive is not None:
            touched_bits = self._node_pattern_bits(child_exclusive)
            if ancestor is not None:
                touched_bits &= ~self._node_pattern_bits(ancestor)
        if affected_roles is None:
            affected_roles = self._copy_affected_roles(lattice, role_plans, child_id, ancestor_id, role_plan_index)
        touched_patterns = self._bits_to_patterns(touched_bits)
        candidate_locations = self._candidate_locations_overlay_for_patterns(
            locations,
            candidate_lattice,
            (ancestor_id,),
            touched_patterns,
        )
        avg_delta_cost, updates = self._candidate_plan_delta(
            lattice,
            candidate_lattice,
            role_plans,
            role_costs,
            affected_roles,
            candidate_locations,
            role_plan_seed_cache,
            touched_patterns,
            collect_updates=collect_updates,
            sort_plans=collect_updates,
            non_increasing=True,
            full_recompute=True,
            timing=timing,
        )
        return float(avg_delta_cost / float(delta + 1)), int(delta), candidate_lattice, updates

    def _score_veda_merge_pair(
        self,
        lattice: dict[str, _LatticeNode],
        role_plans: dict[int, tuple[str, ...]],
        role_costs: dict[int, float],
        locations: dict[int, list[str]],
        child_id: str,
        ancestor_id: str,
        role_plan_index: Optional[dict[str, set[int]]] = None,
        role_plan_seed_cache: Optional[dict[int, _RolePlanSeed]] = None,
        affected_roles: Optional[Iterable[int]] = None,
        collect_updates: bool = True,
        timing: Optional[dict[str, dict[str, float | int]]] = None,
    ) -> tuple[float, _LatticeOverlay, dict[int, tuple[tuple[str, ...], float]]] | None:
        if child_id not in lattice or ancestor_id not in lattice:
            return None
        candidate_lattice = self._candidate_merge_lattice(lattice, child_id, ancestor_id)
        if candidate_lattice is None:
            return None

        child = lattice.get(str(child_id))
        ancestor = lattice.get(str(ancestor_id))
        touched_bits = 0
        if child is not None:
            touched_bits |= self._node_pattern_bits(child)
        if ancestor is not None:
            touched_bits |= self._node_pattern_bits(ancestor)
        if affected_roles is None:
            affected_roles = self._merge_affected_roles(lattice, role_plans, child_id, ancestor_id, role_plan_index)
        touched_patterns = self._bits_to_patterns(touched_bits)
        candidate_locations = self._candidate_locations_overlay_for_patterns(
            locations,
            candidate_lattice,
            (child_id, ancestor_id),
            touched_patterns,
        )
        # Algorithm 13 scores VEDA merge with the query-plan log benefit, not
        # the final HNSW/leftover execution model. If both nodes are already in
        # QP(r), the paper charges the direct replacement a+c; only the one-side
        # case recomputes coverage.
        affected = tuple(dict.fromkeys(int(role_id) for role_id in affected_roles))
        updates: dict[int, tuple[tuple[str, ...], float]] = {}
        total_delta = 0.0
        self._timing_count(timing, "role_delta_evaluations", len(affected))
        started = time.perf_counter()
        for role_id in affected:
            old_plan = tuple(role_plans.get(int(role_id), tuple()))
            old_set = set(str(node_id) for node_id in old_plan)
            has_ancestor = str(ancestor_id) in old_set
            has_child = str(child_id) in old_set
            if has_ancestor and has_child:
                old_score = self._node_log_cost(lattice.get(str(ancestor_id))) + self._node_log_cost(lattice.get(str(child_id)))
                new_score = self._node_log_cost(candidate_lattice.get(str(ancestor_id)))
                new_plan = tuple(node_id for node_id in old_plan if str(node_id) != str(child_id))
                self._timing_count(timing, "merge_both_in_qp_roles")
            elif has_ancestor or has_child:
                old_score = self._role_log_plan_cost(lattice, old_plan)
                new_plan = self._build_role_query_plan(
                    candidate_lattice,
                    int(role_id),
                    locations=candidate_locations,
                    sort_output=collect_updates,
                )
                new_score = self._role_log_plan_cost(candidate_lattice, new_plan)
                self._timing_count(timing, "merge_recomputed_qp_roles")
            else:
                continue
            total_delta += float(old_score) - float(new_score)
            if collect_updates:
                new_cost = self._role_plan_cost(candidate_lattice, int(role_id), new_plan)
                updates[int(role_id)] = (tuple(new_plan), float(new_cost))
        self._timing_add(timing, "candidate_plan_delta_seconds", time.perf_counter() - started)
        return float(total_delta / max(1, len(role_plans))), candidate_lattice, updates

    def _run_veda(self, lattice: dict[str, _LatticeNode], *, show_progress: bool) -> tuple[dict[str, _LatticeNode], dict[str, object]]:
        timing: dict[str, dict[str, float | int]] = {"seconds": {}, "counts": {}}
        total_started = time.perf_counter()
        budget_vectors = int(math.floor(self.storage_amplification * self._exclusive_vector_count()))
        copy_count = 0
        merge_count = 0
        rounds = 0
        started = time.perf_counter()
        pairs = self._descendant_ancestor_pairs(self.exclusive_lattice)
        self._timing_add(timing, "pair_generation_seconds", time.perf_counter() - started)
        self._timing_count(timing, "candidate_pairs", len(pairs))
        started = time.perf_counter()
        pairs_by_node = self._phase_pair_indexes(pairs)
        self._timing_add(timing, "pair_index_seconds", time.perf_counter() - started)
        started = time.perf_counter()
        role_plans = self._build_role_query_plans(lattice)
        self._timing_add(timing, "initial_role_plans_seconds", time.perf_counter() - started)
        started = time.perf_counter()
        role_costs = self._role_costs(lattice, role_plans)
        self._timing_add(timing, "initial_role_costs_seconds", time.perf_counter() - started)
        progress = tqdm(desc="Veda greedy rounds", unit="round", disable=not show_progress)
        first_round = True
        try:
            while True:
                rounds += 1
                progress.update(1)
                started = time.perf_counter()
                copied = self._veda_copy_phase(
                    lattice,
                    pairs,
                    budget_vectors,
                    role_plans,
                    role_costs,
                    pairs_by_node,
                    show_progress=show_progress,
                    timing=timing,
                )
                self._timing_add(timing, "copy_phase_seconds", time.perf_counter() - started)
                copy_count += copied
                if not first_round and copied <= 0:
                    break
                started = time.perf_counter()
                merged = self._veda_merge_phase(
                    lattice,
                    pairs,
                    role_plans,
                    role_costs,
                    pairs_by_node,
                    show_progress=show_progress,
                    timing=timing,
                )
                self._timing_add(timing, "merge_phase_seconds", time.perf_counter() - started)
                merge_count += merged
                if merged <= 0:
                    break
                first_round = False
        finally:
            progress.close()
        self._timing_add(timing, "total_veda_greedy_seconds", time.perf_counter() - total_started)
        return lattice, {
            "copy_operations": int(copy_count),
            "merge_operations": int(merge_count),
            "rounds": int(rounds),
            "storage_budget_vectors": int(budget_vectors),
            "candidate_pair_count": int(len(pairs)),
            "timing": timing,
            "benefit_function": "incremental affected-role QP benefit with C_theta and impurity inflation; final QP rebuilt after greedy operations",
        }

    def _veda_copy_phase(
        self,
        lattice: dict[str, _LatticeNode],
        pairs: list[tuple[str, str]],
        budget_vectors: int,
        role_plans: dict[int, tuple[str, ...]],
        role_costs: dict[int, float],
        pairs_by_node: dict[str, set[tuple[str, str]]],
        *,
        show_progress: bool = False,
        timing: Optional[dict[str, dict[str, float | int]]] = None,
    ) -> int:
        applied = 0
        scores: dict[tuple[str, str], tuple[float, int]] = {}
        versions: dict[tuple[str, str], int] = {}
        heap: list[tuple[float, int, tuple[str, str], int]] = []
        locations = self._lattice_locations(lattice)
        role_plan_index = self._role_plan_node_index(role_plans)
        role_plan_seed_cache: dict[int, _RolePlanSeed] = {}
        affected_roles_cache: dict[tuple[str, str], tuple[int, ...]] = {}
        dead_pairs: set[tuple[str, str]] = set()
        current_storage = int(sum(self._node_size(node) for node in lattice.values()))

        def affected_roles_for_copy(child_id: str, ancestor_id: str) -> tuple[int, ...]:
            pair = (str(child_id), str(ancestor_id))
            cached = affected_roles_cache.get(pair)
            if cached is not None:
                return cached
            result = self._copy_affected_roles(lattice, role_plans, pair[0], pair[1], role_plan_index)
            affected_roles_cache[pair] = result
            return result

        def refresh_scores(target_pairs: Iterable[tuple[str, str]] | None = None) -> None:
            started = time.perf_counter()
            full_refresh = target_pairs is None
            if target_pairs is None:
                targets = pairs
            else:
                targets = sorted({(str(child_id), str(ancestor_id)) for child_id, ancestor_id in target_pairs})
            self._timing_count(timing, "copy_refresh_calls")
            self._timing_count(timing, "copy_refresh_edges", len(targets))
            iterator = targets
            progress = None
            if show_progress and full_refresh and len(targets) >= 1024:
                progress = tqdm(targets, desc="Veda copy score", unit="edge", leave=False)
                iterator = progress
            try:
                for child_id, ancestor_id in iterator:
                    pair = (str(child_id), str(ancestor_id))
                    if pair in dead_pairs:
                        self._timing_count(timing, "copy_dead_pair_skips")
                        continue
                    delta = self._copy_delta_size(lattice, pair[0], pair[1])
                    if delta <= 0:
                        dead_pairs.add(pair)
                        scores.pop(pair, None)
                        versions[pair] = int(versions.get(pair, 0)) + 1
                        self._timing_count(timing, "copy_zero_delta_pairs")
                        continue
                    if delta > int(budget_vectors - current_storage):
                        scores.pop(pair, None)
                        versions[pair] = int(versions.get(pair, 0)) + 1
                        self._timing_count(timing, "copy_over_budget_score_skips")
                        continue
                    scored = self._score_veda_copy_pair(
                        lattice,
                        role_plans,
                        role_costs,
                        locations,
                        pair[0],
                        pair[1],
                        role_plan_index,
                        role_plan_seed_cache,
                        affected_roles_for_copy(pair[0], pair[1]),
                        collect_updates=False,
                        timing=timing,
                    )
                    if scored is None:
                        dead_pairs.add(pair)
                        scores.pop(pair, None)
                        versions[pair] = int(versions.get(pair, 0)) + 1
                        self._timing_count(timing, "copy_unscorable_pairs")
                        continue
                    score, delta, _candidate_lattice, _updates = scored
                    self._push_copy_score(heap, scores, versions, pair, float(score), int(delta))
                    self._timing_count(timing, "copy_scored_pairs")
            finally:
                if progress is not None:
                    progress.close()
                self._timing_add(timing, "copy_refresh_seconds", time.perf_counter() - started)

        def related_pairs_for_ancestor(ancestor_id: str) -> set[tuple[str, str]]:
            return {
                pair
                for pair in pairs_by_node.get(str(ancestor_id), set())
                if str(pair[1]) == str(ancestor_id)
            }

        refresh_scores(None)
        stop_refresh_done = False
        while True:
            buffer = int(budget_vectors - current_storage)
            if buffer <= 0:
                return applied

            selected = None
            if len(heap) > max(4096, len(scores) * 4):
                heap = self._rebuild_copy_heap(scores, versions)
            while heap:
                neg_score, delta, pair, version = heapq.heappop(heap)
                current = scores.get(pair)
                if current is None or versions.get(pair) != version:
                    continue
                score = -float(neg_score)
                if (float(current[0]), int(current[1])) != (score, int(delta)):
                    continue
                if score < 0.0:
                    if not stop_refresh_done:
                        refresh_scores(None)
                        stop_refresh_done = True
                        selected = None
                        break
                    return applied
                if int(delta) > buffer:
                    continue
                selected = (pair, score, int(delta))
                break
            if selected is None:
                if not stop_refresh_done:
                    refresh_scores(None)
                    stop_refresh_done = True
                    continue
                return applied

            stop_refresh_done = False
            (child_id, ancestor_id), _score, _delta = selected
            pair = (str(child_id), str(ancestor_id))
            rescored = self._score_veda_copy_pair(lattice, role_plans, role_costs, locations, child_id, ancestor_id, role_plan_index, role_plan_seed_cache, affected_roles_for_copy(child_id, ancestor_id), collect_updates=True, timing=timing)
            if rescored is None:
                dead_pairs.add(pair)
                scores.pop(pair, None)
                versions[pair] = int(versions.get(pair, 0)) + 1
                continue
            current_score, current_delta, candidate_lattice, updates = rescored
            if float(current_score) != float(_score) or int(current_delta) != int(_delta):
                self._push_copy_score(heap, scores, versions, pair, float(current_score), int(current_delta))
                continue
            if current_score < 0.0 or current_delta > buffer:
                continue

            old_ancestor = lattice.get(str(ancestor_id))
            new_ancestor = candidate_lattice.get(str(ancestor_id))
            next_lattice = candidate_lattice.materialize()
            lattice.clear()
            lattice.update(next_lattice)
            self._replace_locations_for_nodes(locations, (old_ancestor,), (new_ancestor,))
            self._apply_role_plan_updates(role_plans, role_costs, updates, role_plan_index)
            current_storage += int(current_delta)
            applied += 1
            stop_refresh_done = False
            role_plan_seed_cache.clear()
            affected_roles_cache.clear()
            refresh_scores(related_pairs_for_ancestor(str(ancestor_id)))

    def _veda_merge_phase(
        self,
        lattice: dict[str, _LatticeNode],
        pairs: list[tuple[str, str]],
        role_plans: dict[int, tuple[str, ...]],
        role_costs: dict[int, float],
        pairs_by_node: dict[str, set[tuple[str, str]]],
        *,
        show_progress: bool = False,
        timing: Optional[dict[str, dict[str, float | int]]] = None,
    ) -> int:
        applied = 0
        scores: dict[tuple[str, str], float] = {}
        versions: dict[tuple[str, str], int] = {}
        heap: list[tuple[float, tuple[str, str], int]] = []
        locations = self._lattice_locations(lattice)
        role_plan_index = self._role_plan_node_index(role_plans)
        role_plan_seed_cache: dict[int, _RolePlanSeed] = {}
        affected_roles_cache: dict[tuple[str, str], tuple[int, ...]] = {}

        def affected_roles_for_merge(child_id: str, ancestor_id: str) -> tuple[int, ...]:
            pair = (str(child_id), str(ancestor_id))
            cached = affected_roles_cache.get(pair)
            if cached is not None:
                return cached
            result = self._merge_affected_roles(lattice, role_plans, pair[0], pair[1], role_plan_index)
            affected_roles_cache[pair] = result
            return result

        def push_score(pair: tuple[str, str], score: float) -> None:
            versions[pair] = int(versions.get(pair, 0)) + 1
            scores[pair] = float(score)
            heapq.heappush(heap, (-float(score), pair, versions[pair]))

        def rebuild_heap() -> list[tuple[float, tuple[str, str], int]]:
            rebuilt = [(-float(score), pair, int(versions.get(pair, 0))) for pair, score in scores.items()]
            heapq.heapify(rebuilt)
            return rebuilt

        def refresh_scores(target_pairs: Iterable[tuple[str, str]] | None = None) -> None:
            started = time.perf_counter()
            full_refresh = target_pairs is None
            if target_pairs is None:
                targets = pairs
            else:
                targets = sorted({(str(child_id), str(ancestor_id)) for child_id, ancestor_id in target_pairs})
            self._timing_count(timing, "merge_refresh_calls")
            self._timing_count(timing, "merge_refresh_edges", len(targets))
            iterator = targets
            progress = None
            if show_progress and full_refresh and len(targets) >= 1024:
                progress = tqdm(targets, desc="Veda merge score", unit="edge", leave=False)
                iterator = progress
            try:
                for child_id, ancestor_id in iterator:
                    pair = (str(child_id), str(ancestor_id))
                    scored = self._score_veda_merge_pair(
                        lattice,
                        role_plans,
                        role_costs,
                        locations,
                        pair[0],
                        pair[1],
                        role_plan_index,
                        role_plan_seed_cache,
                        affected_roles_for_merge(pair[0], pair[1]),
                        collect_updates=False,
                        timing=timing,
                    )
                    if scored is None:
                        scores.pop(pair, None)
                        versions[pair] = int(versions.get(pair, 0)) + 1
                        self._timing_count(timing, "merge_unscorable_pairs")
                        continue
                    score, _candidate_lattice, _updates = scored
                    push_score(pair, float(score))
                    if float(score) > 0.0:
                        self._timing_count(timing, "merge_positive_pairs")
                    else:
                        self._timing_count(timing, "merge_nonpositive_pairs")
                    self._timing_count(timing, "merge_scored_pairs")
            finally:
                if progress is not None:
                    progress.close()
                self._timing_add(timing, "merge_refresh_seconds", time.perf_counter() - started)

        def related_pairs_for_merge(child_id: str, ancestor_id: str) -> set[tuple[str, str]]:
            targets = {
                pair
                for pair in pairs_by_node.get(str(ancestor_id), set())
                if str(pair[1]) == str(ancestor_id)
            }
            targets.update(
                pair
                for pair in pairs_by_node.get(str(child_id), set())
                if str(pair[0]) == str(child_id)
            )
            return targets

        refresh_scores(None)
        stop_refresh_done = False
        while True:
            selected = None
            if len(heap) > max(4096, len(scores) * 4):
                heap = rebuild_heap()
            while heap:
                neg_score, pair, version = heapq.heappop(heap)
                current = scores.get(pair)
                if current is None or versions.get(pair) != version:
                    continue
                score = -float(neg_score)
                if float(current) != score:
                    continue
                if score <= 0.0:
                    if not stop_refresh_done:
                        refresh_scores(None)
                        stop_refresh_done = True
                        selected = None
                        break
                    return applied
                selected = (pair, score)
                break
            if selected is None:
                if not stop_refresh_done:
                    refresh_scores(None)
                    stop_refresh_done = True
                    continue
                return applied

            stop_refresh_done = False
            (child_id, ancestor_id), _score = selected
            pair = (str(child_id), str(ancestor_id))
            rescored = self._score_veda_merge_pair(
                lattice,
                role_plans,
                role_costs,
                locations,
                child_id,
                ancestor_id,
                role_plan_index,
                role_plan_seed_cache,
                affected_roles_for_merge(child_id, ancestor_id),
                collect_updates=True,
                timing=timing,
            )
            if rescored is None:
                scores.pop(pair, None)
                versions[pair] = int(versions.get(pair, 0)) + 1
                continue
            current_score, candidate_lattice, updates = rescored
            if float(current_score) != float(_score):
                push_score(pair, float(current_score))
                continue
            if float(current_score) <= 0.0:
                scores.pop(pair, None)
                versions[pair] = int(versions.get(pair, 0)) + 1
                continue

            old_child = lattice.get(str(child_id))
            old_ancestor = lattice.get(str(ancestor_id))
            new_ancestor = candidate_lattice.get(str(ancestor_id))
            next_lattice = candidate_lattice.materialize()
            lattice.clear()
            lattice.update(next_lattice)
            self._replace_locations_for_nodes(locations, (old_child, old_ancestor), (new_ancestor,))
            self._apply_role_plan_updates(role_plans, role_costs, updates, role_plan_index)
            applied += 1
            self._timing_count(timing, "merge_applied")
            stop_refresh_done = False
            role_plan_seed_cache.clear()
            affected_roles_cache.clear()
            refresh_scores(related_pairs_for_merge(str(child_id), str(ancestor_id)))

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
                        lattice[target_id] = self._make_lattice_node(
                            node_id=target_id,
                            role_ids=target_roles,
                            pattern_ids=node.pattern_ids,
                            virtual_components=node.virtual_components,
                        )
                    else:
                        target = lattice[target_id]
                        lattice[target_id] = self._make_lattice_node(
                            node_id=target.node_id,
                            role_ids=target.role_ids,
                            pattern_ids=target.pattern_ids | node.pattern_ids,
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
        deferred_unindexable: set[str] = set()
        iterator_total = len(node_order)
        progress = tqdm(total=iterator_total, desc="EffVeda merge nodes", unit="node", disable=not show_progress)

        def requeue_deferred(*, exclude: set[str] | None = None) -> None:
            if not deferred_unindexable:
                return
            excluded = exclude or set()
            requeued = 0
            for deferred_id in sorted(deferred_unindexable):
                if deferred_id in excluded:
                    continue
                if deferred_id in lattice and self._node_size(lattice[deferred_id]) < self.indexing_threshold:
                    node_order.append(deferred_id)
                    requeued += 1
            deferred_unindexable.clear()
            if requeued > 0:
                try:
                    progress.total = (progress.total or 0) + requeued
                    progress.refresh()
                except Exception:
                    pass

        try:
            while index < len(node_order):
                node_id = node_order[index]
                if node_id not in lattice or self._node_size(lattice[node_id]) >= self.indexing_threshold:
                    deferred_unindexable.discard(str(node_id))
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
                    deferred_unindexable.discard(str(node_id))
                    deferred_unindexable.discard(str(candidate_id))
                    requeue_deferred(exclude={str(node_id), str(candidate_id)})
                    if self._node_size(lattice[node_id]) >= self.indexing_threshold:
                        break
                if node_id in lattice and self._node_size(lattice[node_id]) < self.indexing_threshold:
                    if merged_this_node:
                        continue
                    deferred_unindexable.add(str(node_id))
                else:
                    deferred_unindexable.discard(str(node_id))
                index += 1
                progress.update(1)
        finally:
            progress.close()
        return applied

    def _refresh_virtual_components(self, lattice: dict[str, _LatticeNode]) -> None:
        self.post_copy_lattice = {
            node.node_id: self._make_lattice_node(
                node_id=node.node_id,
                role_ids=node.role_ids,
                pattern_ids=node.pattern_ids,
                virtual_components=(node.node_id,),
            )
            for node in lattice.values()
        }
        for node_id, node in list(lattice.items()):
            lattice[node_id] = self._make_lattice_node(
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
            if node_roles & candidate_roles:
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
        return self._make_lattice_node(
            node_id=left.node_id,
            role_ids=normalize_int_tuple(set(left.role_ids) | set(right.role_ids)),
            pattern_ids=left.pattern_ids | right.pattern_ids,
            virtual_components=tuple(dict.fromkeys((*left.virtual_components, *right.virtual_components))),
        )

    def _finalize_lattice(self, lattice: dict[str, _LatticeNode]) -> tuple[dict[str, _LatticeNode], dict[str, object]]:
        finalized: dict[str, _LatticeNode] = {}
        indexable_count = 0
        leftover_count = 0
        for node in lattice.values():
            finalized[node.node_id] = node
            if self._node_size(node) >= self.indexing_threshold:
                indexable_count += 1
            else:
                leftover_count += 1
        return finalized, {
            "split_small_nodes_into_leftovers": False,
            "small_nodes_kept_as_leftovers": int(leftover_count),
            "indexable_nodes_kept": int(indexable_count),
            "paper_semantics": "small lattice nodes are kept as leftover vectors without per-pattern splitting",
        }

    def _standalone_node_for_pattern(self, lattice: dict[str, _LatticeNode], pattern_id: int) -> str | None:
        target_bit = int(self._pattern_bits.get(int(pattern_id), 0))
        if target_bit:
            for node_id, node in lattice.items():
                if int(self._node_pattern_bits(node)) == target_bit:
                    return node_id
            return None
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
        lattice[node_id] = self._make_lattice_node(
            node_id=node_id,
            role_ids=pattern.role_ids,
            pattern_ids=pattern_id,
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

    def _build_role_query_plans(self, lattice: dict[str, _LatticeNode], *, exact: bool = False, final: bool = False) -> dict[int, tuple[str, ...]]:
        locations = self._lattice_locations(lattice)
        return {
            int(role_id): self._build_role_query_plan(lattice, int(role_id), exact=exact, final=final, locations=locations)
            for role_id in self.roles
        }

    def _get_milp_backend(self):
        if self._milp_backend_checked:
            return self._milp_backend
        self._milp_backend_checked = True
        try:
            import numpy as np
            from scipy.optimize import Bounds, LinearConstraint, milp
            from scipy.sparse import coo_matrix
        except Exception:
            self._milp_backend = None
        else:
            self._milp_backend = (np, Bounds, LinearConstraint, milp, coo_matrix)
        return self._milp_backend

    def _solve_exact_coverage(
        self,
        lattice: dict[str, _LatticeNode],
        pending: set[int],
        locations: dict[int, list[str]],
        role_id: int,
        *,
        final: bool = False,
    ) -> set[str] | None:
        candidate_node_ids = sorted({
            node_id
            for pattern_id in pending
            for node_id in locations.get(int(pattern_id), [])
            if node_id in lattice
        })
        if not candidate_node_ids:
            return None
        backend = self._get_milp_backend()
        if backend is None:
            return None
        np, Bounds, LinearConstraint, milp, coo_matrix = backend

        pending_list = sorted(int(pattern_id) for pattern_id in pending)
        pattern_index = {pattern_id: index for index, pattern_id in enumerate(pending_list)}
        pending_set = set(pending_list)
        row_indices: list[int] = []
        col_indices: list[int] = []
        for column, node_id in enumerate(candidate_node_ids):
            node = lattice[node_id]
            if len(node.pattern_ids) <= len(pending_set):
                for pattern_id in node.pattern_ids:
                    pattern_id = int(pattern_id)
                    row = pattern_index.get(pattern_id)
                    if row is not None:
                        row_indices.append(int(row))
                        col_indices.append(int(column))
            else:
                for pattern_id in pending_set:
                    if int(pattern_id) in node.pattern_ids:
                        row_indices.append(int(pattern_index[int(pattern_id)]))
                        col_indices.append(int(column))
        matrix = coo_matrix(
            (np.ones(len(row_indices), dtype=float), (row_indices, col_indices)),
            shape=(len(pending_list), len(candidate_node_ids)),
        )
        costs = np.array(
            [
                self._node_cost_for_role(lattice[node_id], int(role_id), final=final)
                if final
                else self._node_log_cost(lattice[node_id])
                for node_id in candidate_node_ids
            ],
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


    def _greedy_extend_coverage_bits(
        self,
        lattice: dict[str, _LatticeNode],
        selected: set[str],
        pending_bits: int,
        locations,
        role_id: int,
        *,
        final: bool = False,
    ) -> None:
        pending = int(pending_bits)
        node_info_cache: dict[str, tuple[int, int, float, float]] = {}
        while pending:
            best_node_id = None
            best_rank = None
            seen_nodes: set[str] = set()
            scan = pending
            while scan:
                low = scan & -scan
                index = low.bit_length() - 1
                pattern_id = self._bit_to_pattern.get(index)
                if pattern_id is not None:
                    for candidate_node_id in locations.get(int(pattern_id), []):
                        node_id = str(candidate_node_id)
                        if node_id in seen_nodes:
                            continue
                        seen_nodes.add(node_id)
                        cached_info = node_info_cache.get(node_id)
                        if cached_info is None:
                            node = lattice.get(node_id)
                            if node is None:
                                continue
                            node_bits = self._node_pattern_bits(node)
                            node_size = self._node_size(node)
                            node_log_cost = self._node_log_cost(node)
                            node_final_cost = self._node_cost_for_role(node, int(role_id), final=final) if final else float(node_log_cost)
                            cached_info = (int(node_bits), int(node_size), float(node_log_cost), float(node_final_cost))
                            node_info_cache[node_id] = cached_info
                        node_bits, node_size, node_log_cost, node_final_cost = cached_info
                        covered_bits = int(node_bits) & pending
                        cover_count = int(covered_bits.bit_count())
                        if cover_count <= 0:
                            continue
                        rank = (
                            -cover_count,
                            float(node_final_cost),
                            int(node_size),
                            node_id,
                        )
                        if best_rank is None or rank < best_rank:
                            best_rank = rank
                            best_node_id = node_id
                scan ^= low
            if best_node_id is None:
                break
            selected.add(best_node_id)
            cached_info = node_info_cache.get(best_node_id)
            if cached_info is None:
                node = lattice.get(best_node_id)
                if node is None:
                    break
                node_log_cost = self._node_log_cost(node)
                node_final_cost = self._node_cost_for_role(node, int(role_id), final=final) if final else float(node_log_cost)
                cached_info = (self._node_pattern_bits(node), self._node_size(node), float(node_log_cost), float(node_final_cost))
                node_info_cache[best_node_id] = cached_info
            pending &= ~int(cached_info[0])

    def _greedy_extend_coverage(
        self,
        lattice: dict[str, _LatticeNode],
        selected: set[str],
        pending: set[int],
        locations: dict[int, list[str]],
        role_id: int,
        *,
        final: bool = False,
    ) -> None:
        while pending:
            best_node_id = None
            best_rank = None
            for pattern_id in sorted(pending):
                for candidate_node_id in locations.get(int(pattern_id), []):
                    node_id = str(candidate_node_id)
                    node = lattice.get(node_id)
                    if node is None:
                        continue
                    covered = set(int(pid) for pid in node.pattern_ids) & pending
                    cover_count = len(covered)
                    if cover_count <= 0:
                        continue
                    rank = (
                        -cover_count,
                        self._node_log_cost(node),
                        self._node_size(node),
                        node_id,
                    )
                    if best_rank is None or rank < best_rank:
                        best_rank = rank
                        best_node_id = node_id
            if best_node_id is None:
                break
            selected.add(best_node_id)
            pending -= (set(int(pid) for pid in lattice[best_node_id].pattern_ids) & pending)

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
                    table_name=get_node_table_name(stable_node_id, algorithm=self._active_algorithm),
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
