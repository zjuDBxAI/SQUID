"""Targeted refinement utilities for oversized dynamic partitions.

The functions in this module inspect a single oversized partition produced by
AnonySys dynamic partitioning and recursively split it using role predicates
only. This is useful as a follow-up rebalancing step when the storage budget
prevents the main partitioner from evenly distributing the load up-front.
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

repo_root = Path(__file__).resolve().parents[4]
repo_root_str = str(repo_root)
if repo_root_str not in sys.path:
    sys.path.append(repo_root_str)

from services.config import get_db_connection  # pylint: disable=wrong-import-position


@dataclass
class RoleDocument:
    block_id: int
    doc_id: int
    accessible_roles: Set[str]


@dataclass
class RoleTreeNode:
    depth: int
    documents: List[RoleDocument]
    split_role: Optional[str] = None
    left_child: Optional["RoleTreeNode"] = None
    right_child: Optional["RoleTreeNode"] = None
    required_roles: Set[str] = field(default_factory=set)

    def is_leaf(self) -> bool:
        return self.split_role is None


def _fetch_partition_documents(table_name: str) -> List[RoleDocument]:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            f"""
            SELECT p.block_id,
                   p.document_id,
                   array_agg(DISTINCT pa.role_id) AS roles
            FROM {table_name} AS p
            JOIN PermissionAssignment pa ON p.document_id = pa.document_id
            GROUP BY p.block_id, p.document_id
            ORDER BY p.document_id, p.block_id;
            """
        )
        documents: List[RoleDocument] = []
        for block_id, document_id, roles in cur.fetchall():
            documents.append(
                RoleDocument(
                    block_id=int(block_id),
                    doc_id=int(document_id),
                    accessible_roles={str(role) for role in roles or []},
                )
            )
    finally:
        cur.close()
        conn.close()
    return documents


def _partition_by_role(
    documents: Sequence[RoleDocument],
    role_value: str,
) -> Tuple[List[RoleDocument], List[RoleDocument]]:
    left_docs: List[RoleDocument] = []
    right_docs: List[RoleDocument] = []
    for doc in documents:
        if role_value in doc.accessible_roles:
            left_docs.append(doc)
        else:
            right_docs.append(doc)
    return left_docs, right_docs


def _find_best_role_split(
    documents: Sequence[RoleDocument],
    candidate_roles: Iterable[str],
    min_leaf_size: Optional[int],
) -> Tuple[Optional[str], Optional[List[RoleDocument]], Optional[List[RoleDocument]]]:
    best_role: Optional[str] = None
    best_left: Optional[List[RoleDocument]] = None
    best_right: Optional[List[RoleDocument]] = None
    best_cost = None

    for role_value in candidate_roles:
        left_docs, right_docs = _partition_by_role(documents, role_value)
        if not left_docs or not right_docs:
            continue
        cost = math.log(len(left_docs)) + math.log(len(right_docs))
        if best_cost is None or cost < best_cost:
            best_cost = cost
            best_role = role_value
            best_left = left_docs
            best_right = right_docs

    return best_role, best_left, best_right


def _build_role_tree(
    documents: Sequence[RoleDocument],
    available_roles: Set[str],
    min_leaf_size: Optional[int],
    depth: int = 0,
    required_roles: Optional[Set[str]] = None,
) -> RoleTreeNode:
    req_roles = set(required_roles) if required_roles else set()

    if len(documents) <= 1 or not available_roles:
        return RoleTreeNode(depth=depth, documents=list(documents), required_roles=req_roles)

    best_role, left_docs, right_docs = _find_best_role_split(documents, available_roles, min_leaf_size)
    if best_role is None or left_docs is None or right_docs is None:
        return RoleTreeNode(depth=depth, documents=list(documents), required_roles=req_roles)

    remaining_roles = set(available_roles)
    remaining_roles.discard(best_role)

    left_required = set(req_roles)
    left_required.add(best_role)

    left_child = _build_role_tree(left_docs, remaining_roles, min_leaf_size, depth + 1, left_required)
    right_child = _build_role_tree(right_docs, remaining_roles, min_leaf_size, depth + 1, req_roles)

    return RoleTreeNode(
        depth=depth,
        documents=list(documents),
        split_role=best_role,
        left_child=left_child,
        right_child=right_child,
        required_roles=req_roles,
    )


def _collect_leaves(root: RoleTreeNode) -> List[RoleTreeNode]:
    leaves: List[RoleTreeNode] = []

    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            leaves.append(node)
        else:
            if node.right_child is not None:
                stack.append(node.right_child)
            if node.left_child is not None:
                stack.append(node.left_child)
    return leaves


def inspect_partition(table_name: str) -> None:
    """CLI helper to print the role-split tree for a materialized partition."""
    documents = _fetch_partition_documents(table_name)
    total_blocks = len(documents)
    logger = logging.getLogger(__name__)

    if total_blocks == 0:
        logger.info("Partition %s is empty; skipping", table_name)
        return

    role_values = {role for doc in documents for role in doc.accessible_roles}
    if not role_values:
        logger.info("Partition %s has no role information; skipping", table_name)
        return

    logger.info(
        "Building role-only refinement for %s (blocks=%s, unique_roles=%s)",
        table_name,
        total_blocks,
        len(role_values),
    )

    tree_root = _build_role_tree(documents, role_values, min_leaf_size=None)
    leaves = _collect_leaves(tree_root)

    logger.info("%s split into %s role-derived partitions", table_name, len(leaves))
    for idx, leaf in enumerate(sorted(leaves, key=lambda n: len(n.documents), reverse=True)):
        logger.info(
            "  leaf %s size=%s required_roles=%s",
            idx,
            len(leaf.documents),
            sorted(leaf.required_roles) or ["-"],
        )


def rebalance_heavy_partition(
    partition_assignment: Dict[int, Set[int]],
    comb_role_trackers: Dict[Tuple[int, ...], Dict[int, Set[int]]],
    document_index_to_roles: Dict[int, Set[int]],
    logger: Optional[logging.Logger] = None,
    target_partitions: Optional[Set[int]] = None,
    min_leaf_size: Optional[int] = None,
) -> Tuple[Dict[int, Set[int]], Dict[Tuple[int, ...], Dict[int, Set[int]]]]:
    """Split oversized partitions using role predicates and update trackers accordingly."""
    log = logger or logging.getLogger(__name__)
    if log.level == logging.NOTSET:
        log.setLevel(logging.INFO)

    if not partition_assignment:
        return {}, defaultdict(lambda: defaultdict(set))

    # Tuning knobs for the greedy refinement.
    max_partitions_per_role = 3
    min_docs_threshold = max(min_leaf_size or 1, 1)
    min_candidate_docs = min_docs_threshold
    min_remaining_docs = min_docs_threshold
    max_subset_size = 3
    top_role_limit = 8
    min_improvement = 1e-6
    allow_single_role_split = True

    next_pid = max(partition_assignment.keys()) + 1
    partition_docs: Dict[int, Set[int]] = {}
    partition_role_docs: Dict[int, Dict[int, Set[int]]] = {}
    partition_allowed_roles: Dict[int, Set[int]] = {}
    partition_leaf_info: Dict[int, List[Tuple[int, Set[int], Set[int], Dict[int, Set[int]]]]] = {}
    partition_required_docs: Dict[int, Dict[int, Set[int]]] = {}

    for pid, doc_indices in partition_assignment.items():
        doc_set = set(doc_indices)
        partition_docs[pid] = doc_set

        allowed_roles: Set[int] = set()
        for comb, partition_roles in comb_role_trackers.items():
            if pid in partition_roles:
                allowed_roles.update(partition_roles[pid])
        partition_allowed_roles[pid] = allowed_roles

        role_doc_map: Dict[int, Set[int]] = defaultdict(set)
        for doc_idx in doc_set:
            for role in document_index_to_roles.get(doc_idx, set()):
                if allowed_roles and role not in allowed_roles:
                    continue
                role_doc_map[role].add(doc_idx)
        partition_role_docs[pid] = {role: set(doc_ids) for role, doc_ids in role_doc_map.items()}
        partition_required_docs[pid] = {role: set(doc_ids) for role, doc_ids in role_doc_map.items()}

    role_to_partitions: Dict[int, Set[int]] = defaultdict(set)
    for pid, role_doc_map in partition_role_docs.items():
        for role, docs in role_doc_map.items():
            if docs:
                role_to_partitions[role].add(pid)

    def _role_cost(partition_size: int, docs_for_role: int) -> float:
        if partition_size <= 0 or docs_for_role <= 0:
            return 0.0
        sel = docs_for_role / partition_size
        sel = max(sel, 1e-9)
        return math.log(max(partition_size, 1)) / sel

    def _total_cost_for_role(role: int) -> float:
        total = 0.0
        for part_id in role_to_partitions.get(role, set()):
            docs = partition_role_docs.get(part_id, {}).get(role)
            if not docs:
                continue
            total += _role_cost(len(partition_docs.get(part_id, set())), len(docs))
        return total

    def _apply_candidate(pid: int, candidate: Dict[str, object]) -> int:
        nonlocal next_pid
        new_pid = candidate.get("new_pid")  # type: ignore[assignment]
        if new_pid is None:
            new_pid = next_pid
            candidate["new_pid"] = new_pid
        candidate_docs: Set[int] = candidate["docs"]  # type: ignore[assignment]
        role_doc_map: Dict[int, Set[int]] = candidate["role_doc_map"]  # type: ignore[assignment]
        affected_roles: Set[int] = candidate["affected_roles"]  # type: ignore[assignment]

        # Remove documents from the source partition.
        partition_docs[pid].difference_update(candidate_docs)

        # Initialise the new partition structures.
        partition_docs[new_pid] = set(candidate_docs)
        partition_allowed_roles[new_pid] = set(partition_allowed_roles.get(pid, set()))
        partition_role_docs[new_pid] = {role: set(docs) for role, docs in role_doc_map.items()}

        for role in affected_roles:
            # Update old partition mapping.
            source_role_docs = partition_role_docs.get(pid, {}).get(role)
            if source_role_docs is not None:
                source_role_docs.difference_update(candidate_docs)
                if source_role_docs:
                    partition_role_docs[pid][role] = source_role_docs
                else:
                    partition_role_docs[pid].pop(role, None)

            updated_source_docs = partition_role_docs.get(pid, {}).get(role)
            if updated_source_docs:
                role_to_partitions[role].add(pid)
            else:
                role_to_partitions[role].discard(pid)

            moved_docs = partition_role_docs.get(new_pid, {}).get(role)
            if moved_docs:
                role_to_partitions[role].add(new_pid)
            else:
                role_to_partitions[role].discard(new_pid)

        next_pid += 1
        return new_pid

    def _collect_partition_info(pid_list: List[int]) -> List[Tuple[int, Set[int], Set[int], Dict[int, Set[int]]]]:
        infos: List[Tuple[int, Set[int], Set[int], Dict[int, Set[int]]]] = []
        for part_id in pid_list:
            docs_here = partition_docs.get(part_id, set())
            role_doc_map = partition_role_docs.get(part_id, {})
            roles_present = set(role_doc_map.keys())
            infos.append(
                (
                    part_id,
                    set(docs_here),
                    set(roles_present),
                    {role: set(doc_ids) for role, doc_ids in role_doc_map.items()},
                )
            )
        return infos

    beam_width = 4
    max_beam_depth = 3
    max_candidates_per_state = 6

    @dataclass
    class BeamState:
        total_cost: float
        partition_cost: float
        depth: int
        remaining_docs: Set[int]
        remaining_role_docs: Dict[int, Set[int]]
        new_partitions: List[Tuple[Set[int], Dict[int, Set[int]]]]
        actions: List[Dict[str, object]]

    def _state_signature(
        remaining_docs: Set[int],
        new_partitions: List[Tuple[Set[int], Dict[int, Set[int]]]],
    ) -> Tuple[frozenset[int], Tuple[frozenset[int], ...]]:
        partition_sigs = tuple(sorted(frozenset(doc_set) for doc_set, _ in new_partitions))
        return frozenset(remaining_docs), partition_sigs

    def _compute_partition_cost(
        remaining_docs: Set[int],
        remaining_role_docs: Dict[int, Set[int]],
        new_partitions: List[Tuple[Set[int], Dict[int, Set[int]]]],
    ) -> float:
        partition_cost = 0.0
        if remaining_docs:
            part_size = len(remaining_docs)
            for docs in remaining_role_docs.values():
                if docs:
                    partition_cost += _role_cost(part_size, len(docs))
        for part_docs, role_map in new_partitions:
            if not part_docs:
                continue
            part_size = len(part_docs)
            for docs in role_map.values():
                if docs:
                    partition_cost += _role_cost(part_size, len(docs))
        return partition_cost

    def _beam_search_for_partition(
        pid: int,
        allowed_roles: Set[int],
        external_partition_counts: Dict[int, int],
        external_cost_base: float,
    ) -> List[Dict[str, object]]:
        initial_remaining_docs = set(partition_docs.get(pid, set()))
        initial_role_docs = {
            role: set(doc_set) for role, doc_set in partition_role_docs.get(pid, {}).items()
        }
        initial_partition_cost = _compute_partition_cost(initial_remaining_docs, initial_role_docs, [])
        initial_total_cost = external_cost_base + initial_partition_cost

        initial_state = BeamState(
            total_cost=initial_total_cost,
            partition_cost=initial_partition_cost,
            depth=0,
            remaining_docs=initial_remaining_docs,
            remaining_role_docs=initial_role_docs,
            new_partitions=[],
            actions=[],
        )

        best_state = initial_state
        beam: List[BeamState] = [initial_state]
        seen_signatures: Set[Tuple[frozenset[int], Tuple[frozenset[int], ...]]] = {
            _state_signature(initial_state.remaining_docs, initial_state.new_partitions)
        }

        def _violates_role_budget(
            remaining_role_docs: Dict[int, Set[int]],
            new_partitions: List[Tuple[Set[int], Dict[int, Set[int]]]],
        ) -> bool:
            all_roles: Set[int] = set(external_partition_counts.keys())
            all_roles.update(remaining_role_docs.keys())
            for _, role_map in new_partitions:
                all_roles.update(role_map.keys())

            for role in all_roles:
                count = external_partition_counts.get(role, 0)
                if remaining_role_docs.get(role):
                    count += 1
                for _, role_map in new_partitions:
                    if role_map.get(role):
                        count += 1
                if count > max_partitions_per_role:
                    return True
            return False

        def _generate_children(state: BeamState) -> List[BeamState]:
            if not state.remaining_docs:
                return []

            role_costs: List[Tuple[int, float]] = []
            part_size = len(state.remaining_docs)
            for role, docs in state.remaining_role_docs.items():
                if docs:
                    role_costs.append((role, _role_cost(part_size, len(docs))))
            if not role_costs:
                return []

            role_costs.sort(key=lambda item: item[1], reverse=True)
            top_roles_list = [role for role, _ in role_costs[:top_role_limit]]

            children: List[BeamState] = []
            seen_doc_keys: Set[frozenset[int]] = set()

            def _attempt_candidate(candidate_roles: Sequence[int]) -> None:
                nonlocal children
                if not candidate_roles:
                    return
                candidate_docs = state.remaining_role_docs.get(candidate_roles[0], set()).copy()
                for role in candidate_roles[1:]:
                    candidate_docs.intersection_update(state.remaining_role_docs.get(role, set()))
                    if len(candidate_docs) < min_candidate_docs:
                        break
                if len(candidate_docs) < min_candidate_docs:
                    return

                doc_key = frozenset(candidate_docs)
                if not doc_key or doc_key in seen_doc_keys:
                    return
                seen_doc_keys.add(doc_key)

                new_remaining_docs = state.remaining_docs - candidate_docs
                if len(new_remaining_docs) < min_remaining_docs:
                    return

                candidate_role_map: Dict[int, Set[int]] = defaultdict(set)
                for doc_idx in candidate_docs:
                    for role in document_index_to_roles.get(doc_idx, set()):
                        if allowed_roles and role not in allowed_roles:
                            continue
                        candidate_role_map[role].add(doc_idx)

                affected_roles = {role for role, docs in candidate_role_map.items() if docs}
                if not affected_roles:
                    return

                new_remaining_role_docs = {
                    role: set(docs) for role, docs in state.remaining_role_docs.items()
                }
                for role, docs in candidate_role_map.items():
                    if role in new_remaining_role_docs:
                        new_remaining_role_docs[role].difference_update(docs)
                        if not new_remaining_role_docs[role]:
                            new_remaining_role_docs.pop(role)

                new_partitions = [
                    (set(doc_set), {r: set(role_docs) for r, role_docs in role_map.items()})
                    for doc_set, role_map in state.new_partitions
                ]
                new_partitions.append(
                    (
                        set(candidate_docs),
                        {role: set(docs) for role, docs in candidate_role_map.items() if docs},
                    )
                )

                if _violates_role_budget(new_remaining_role_docs, new_partitions):
                    return

                partition_cost = _compute_partition_cost(new_remaining_docs, new_remaining_role_docs, new_partitions)
                total_cost = external_cost_base + partition_cost
                new_actions = list(state.actions)
                new_actions.append(
                    {
                        "docs": set(candidate_docs),
                        "role_doc_map": {
                            role: set(docs) for role, docs in candidate_role_map.items() if docs
                        },
                        "affected_roles": set(affected_roles),
                        "subset_roles": tuple(candidate_roles),
                        "delta_cost": state.total_cost - total_cost,
                        "cost_before": state.total_cost,
                        "cost_after": total_cost,
                        "pre_size": len(state.remaining_docs),
                        "post_size": len(new_remaining_docs),
                        "new_size": len(candidate_docs),
                    }
                )

                child_state = BeamState(
                    total_cost=total_cost,
                    partition_cost=partition_cost,
                    depth=state.depth + 1,
                    remaining_docs=new_remaining_docs,
                    remaining_role_docs=new_remaining_role_docs,
                    new_partitions=new_partitions,
                    actions=new_actions,
                )
                children.append(child_state)

            max_subset = min(max_subset_size, len(top_roles_list))
            for subset_size in range(max_subset, 1, -1):
                for subset in combinations(top_roles_list, subset_size):
                    _attempt_candidate(subset)
                    if len(children) >= max_candidates_per_state:
                        break
                if len(children) >= max_candidates_per_state:
                    break

            if len(children) < max_candidates_per_state and allow_single_role_split:
                for role in top_roles_list:
                    _attempt_candidate((role,))
                    if len(children) >= max_candidates_per_state:
                        break

            children.sort(key=lambda item: item.total_cost)
            return children[:max_candidates_per_state]

        for depth in range(max_beam_depth):
            next_states: List[BeamState] = []
            for state in beam:
                for child in _generate_children(state):
                    signature = _state_signature(child.remaining_docs, child.new_partitions)
                    if signature in seen_signatures:
                        continue
                    seen_signatures.add(signature)
                    next_states.append(child)
                    if child.total_cost + min_improvement < best_state.total_cost:
                        best_state = child
                if len(next_states) >= beam_width * max_candidates_per_state:
                    break

            if not next_states:
                break

            combined_states = beam + next_states
            combined_states.sort(key=lambda item: item.total_cost)

            pruned_states: List[BeamState] = []
            local_seen: Set[Tuple[frozenset[int], Tuple[frozenset[int], ...]]] = set()
            for state in combined_states:
                if len(pruned_states) >= beam_width:
                    break
                signature = _state_signature(state.remaining_docs, state.new_partitions)
                if signature in local_seen:
                    continue
                local_seen.add(signature)
                pruned_states.append(state)
            beam = pruned_states

        if best_state.total_cost + min_improvement >= initial_total_cost:
            return []
        return best_state.actions

    for pid in sorted(partition_assignment.keys()):
        if target_partitions is not None and pid not in target_partitions:
            partition_leaf_info[pid] = _collect_partition_info([pid])
            continue

        if len(partition_docs.get(pid, set())) <= min_candidate_docs:
            log.debug(
                "Partition %s skipped greedy refinement (docs=%s <= threshold=%s)",
                pid,
                len(partition_docs.get(pid, set())),
                min_candidate_docs,
            )
            partition_leaf_info[pid] = _collect_partition_info([pid])
            continue

        allowed_roles = partition_allowed_roles.get(pid, set())

        relevant_roles = set(partition_role_docs.get(pid, {}).keys())
        if allowed_roles:
            relevant_roles.update(role for role in allowed_roles if role in role_to_partitions)

        external_partition_counts: Dict[int, int] = {}
        external_cost_base = 0.0
        for role in relevant_roles:
            partitions = role_to_partitions.get(role, set())
            external_partition_counts[role] = len(partitions - {pid})
            for other_pid in partitions:
                if other_pid == pid:
                    continue
                other_docs = partition_role_docs.get(other_pid, {}).get(role, set())
                if other_docs:
                    external_cost_base += _role_cost(len(partition_docs.get(other_pid, set())), len(other_docs))

        log.info(
            "Starting beam refinement for partition %s (docs=%s, unique_roles=%s)",
            pid,
            len(partition_docs.get(pid, set())),
            len(partition_role_docs.get(pid, {})),
        )
        beam_actions = _beam_search_for_partition(pid, allowed_roles, external_partition_counts, external_cost_base)
        if not beam_actions:
            log.info(
                "Beam search found no improving plan for partition %s; terminating with %s docs remaining",
                pid,
                len(partition_docs.get(pid, set())),
            )
            partition_leaf_info[pid] = _collect_partition_info([pid])
            continue

        created_partitions: List[int] = []
        for iteration, candidate in enumerate(beam_actions):
            affected_roles = sorted(candidate.get("affected_roles", []))
            docs_moved = candidate.get("docs", set())
            delta_cost = candidate.get("delta_cost", 0.0)
            cost_before = candidate.get("cost_before", 0.0)
            cost_after = candidate.get("cost_after", 0.0)
            pre_size = candidate.get("pre_size", len(partition_docs.get(pid, set())))

            log.info(
                "Iteration %s on partition %s -> roles=%s docs=%s delta=%.4f (cost %.4f -> %.4f)",
                iteration,
                pid,
                affected_roles,
                len(docs_moved),
                delta_cost,
                cost_before,
                cost_after,
            )

            new_partition_id = _apply_candidate(pid, candidate)
            created_partitions.append(new_partition_id)

            remaining_docs = len(partition_docs.get(pid, set()))
            new_partition_docs = len(partition_docs.get(new_partition_id, set()))
            log.info(
                "  Created partition %s with %s docs; partition %s now retains %s docs (before=%s)",
                new_partition_id,
                new_partition_docs,
                pid,
                remaining_docs,
                pre_size,
            )

            if log.isEnabledFor(logging.DEBUG):
                role_debug = ", ".join(
                    f"{role}:{_total_cost_for_role(role):.4f}" for role in affected_roles
                )
                log.debug(
                    "After iteration %s on partition %s -> roles cost snapshot: %s",
                    iteration,
                    pid,
                    role_debug or "-",
                )

        final_summary = [
            (part_id, len(partition_docs.get(part_id, set()))) for part_id in [pid] + created_partitions
        ]
        log.info(
            "Finished refinement for partition %s after %s iteration(s); partitions=%s",
            pid,
            len(beam_actions),
            final_summary,
        )
        partition_leaf_info[pid] = _collect_partition_info([pid] + created_partitions)

    updated_assignment: Dict[int, Set[int]] = {}
    for info_list in partition_leaf_info.values():
        for part_id, doc_set, _roles_present, _role_map in info_list:
            updated_assignment[part_id] = set(doc_set)

    refined_comb_trackers: Dict[Tuple[int, ...], Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
    for comb, partition_roles in comb_role_trackers.items():
        for pid, roles in partition_roles.items():
            if pid not in partition_leaf_info:
                refined_comb_trackers[comb][pid].update(roles)
                continue
            leaf_info = partition_leaf_info[pid]
            sorted_leaf_info = sorted(
                leaf_info,
                key=lambda item: (len(item[1]), len(item[2])),
            )
            required_docs = partition_required_docs.get(pid, {})
            candidate_indices = list(range(len(sorted_leaf_info)))
            best_subset: Optional[List[int]] = None
            best_cost: Optional[Tuple[int, int]] = None

            for r in range(1, len(candidate_indices) + 1):
                for subset in combinations(candidate_indices, r):
                    covers_all = True
                    for role in roles:
                        needed = required_docs.get(role, set())
                        covered = set()
                        for idx in subset:
                            role_doc_map = sorted_leaf_info[idx][3]
                            covered.update(role_doc_map.get(role, set()))
                        if needed and not needed.issubset(covered):
                            covers_all = False
                            break
                    if not covers_all:
                        continue
                    total_docs = sum(len(sorted_leaf_info[idx][1]) for idx in subset)
                    cost = (len(subset), total_docs)
                    if best_cost is None or cost < best_cost:
                        best_cost = cost
                        best_subset = list(subset)
                if best_subset is not None:
                    break

            if best_subset is None:
                best_subset = [0] if sorted_leaf_info else []

            for idx in best_subset:
                new_pid, _doc_set, _roles_present, role_doc_map = sorted_leaf_info[idx]
                for role in roles:
                    if role_doc_map.get(role):
                        refined_comb_trackers[comb][new_pid].add(role)

            for role in roles:
                role_assigned = any(
                    role in refined_comb_trackers[comb][sorted_leaf_info[idx][0]]
                    for idx in best_subset
                )
                if not role_assigned and sorted_leaf_info:
                    fallback_pid = sorted_leaf_info[best_subset[0]][0]
                    refined_comb_trackers[comb][fallback_pid].add(role)

    used_partitions: Set[int] = set()
    for partition_roles in refined_comb_trackers.values():
        for pid, roles in list(partition_roles.items()):
            if not roles:
                partition_roles.pop(pid)
                continue
            used_partitions.add(pid)

    filtered_assignment = {
        pid: docs for pid, docs in updated_assignment.items() if pid in used_partitions or target_partitions is None
    }

    return filtered_assignment, refined_comb_trackers


def remap_comb_role_trackers(
    comb_role_trackers: Dict[Tuple[int, ...], Dict[int, Set[int]]],
    partition_mapping: Dict[int, int],
) -> Dict[Tuple[int, ...], Dict[int, Set[int]]]:
    remapped: Dict[Tuple[int, ...], Dict[int, Set[int]]] = defaultdict(lambda: defaultdict(set))
    for comb, partitions in comb_role_trackers.items():
        for pid, roles in partitions.items():
            new_pid = partition_mapping.get(pid, pid)
            remapped[comb][new_pid].update(roles)
    return remapped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recursively split AnonySys partitions using role predicates only and report the structure.",
    )
    parser.add_argument(
        "--partition-prefix",
        type=str,
        default="documentblocks_partition",
        help="Prefix of AnonySys materialized partition tables.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only process the first N partitions (after sorting by name).",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT tablename
            FROM pg_tables
            WHERE tablename LIKE %s
            ORDER BY tablename;
            """,
            (f"{args.partition_prefix}_%",),
        )
        tables = [row[0] for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()

    if args.limit is not None:
        tables = tables[: args.limit]

    logging.info("Discovered %s partition tables with prefix '%s'", len(tables), args.partition_prefix)

    for table_name in tables:
        inspect_partition(table_name)


if __name__ == "__main__":
    main()
