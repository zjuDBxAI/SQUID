import json
import os

import sys
from typing import Dict, Set, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)
from controller.dynamic_partition.load_result_to_database import load_result_to_database
from basic_benchmark.initialize_dynamic_partition_tables import initialize_dynamic_partition_tables_in_comb

from services.config import get_db_connection, get_document_vector_dimension
from controller.baseline.prefilter.initialize_partitions import create_indexes_for_all_role_tables
from controller.initialize_main_tables import create_indexes
from controller.dynamic_partition.acorn.helper import (
    fetch_initial_data,
    prepare_background_data,
    delete_faiss_files,
)
from services.logger import get_logger


from collections import defaultdict

import math


logger = get_logger(__name__)


def init_user_role_combination_data():
    """
    Retrieve unique role combinations and calculate their weights (percentage of users).

    Returns:
        tuple: (role_combinations_set, role_weights)
    """
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch role combinations and their weights
    cur.execute("""
        SELECT roles, COUNT(*)::float / SUM(COUNT(*)) OVER () AS weight
        FROM (
            SELECT user_id, array_agg(role_id ORDER BY role_id) AS roles
            FROM userroles
            GROUP BY user_id
        ) subquery
        GROUP BY roles
        ORDER BY roles;
    """)
    results = cur.fetchall()
    conn.close()

    # Convert results into a set of role_combinations and a dict of role_weights
    role_combinations_set = {tuple(row[0]) for row in results}  # Set of unique role combinations
    role_weights = {tuple(row[0]): row[1] for row in results}  # Mapping of role_combination -> weight

    return role_combinations_set, role_weights


def _normalize_comb_key(comb):
    if isinstance(comb, tuple):
        return comb
    if isinstance(comb, list):
        return tuple(comb)
    if isinstance(comb, (set, frozenset)):
        return tuple(sorted(comb))
    if comb is None:
        return tuple()
    return (comb,)


def _resolve_weight(comb, role_weights):
    if not role_weights:
        return 1.0
    key = _normalize_comb_key(comb)
    if key in role_weights:
        return role_weights[key]
    # Fallback: try individual roles
    if isinstance(key, tuple) and key:
        for role in key:
            if role in role_weights:
                return role_weights[role]
    # Default weight when nothing recorded
    return 1.0


def compute_query_time(
    comb_trackers,
    loads,
    selectivity_map,
    vector_dim,
    gamma,
    comb_to_update,
    role_weights=None,
):
    """
    Compute the weighted ACORN query complexity for the specified combinations.
    """
    total_query_time = 0.0
    for comb in comb_to_update:
        partitions = comb_trackers.get(comb, {})
        if isinstance(partitions, dict):
            partition_ids = partitions.keys()
        else:
            partition_ids = partitions

        total_docs = sum(loads.get(pid, 0) for pid in partition_ids)
        if total_docs <= 0:
            continue

        key = _normalize_comb_key(comb)
        selectivity = selectivity_map.get(key, 0.0)
        if selectivity <= 0:
            continue

        sel = max(selectivity, 1e-12)
        effective_sn = max(sel * total_docs, 1e-12)
        query_cost = (vector_dim + gamma) * math.log(effective_sn) + math.log(1.0 / sel)
        weight = _resolve_weight(comb, role_weights)
        total_query_time += weight * query_cost

    return total_query_time


def compute_sel_whole(
    comb_trackers,
    loads,
    role_to_documents,
    comb_to_update,
    role_weights=None,
    partition_assignment=None,
):
    """
    Compute overall selectivity and return per-combination selectivity values.
    """
    total_weighted_sel = 0.0
    total_weight = 0.0
    selectivity_map = {}

    for comb in comb_to_update:
        partitions = comb_trackers.get(comb, {})
        if isinstance(partitions, dict):
            partition_ids = partitions.keys()
        else:
            partition_ids = partitions

        role_docs = set()
        for role in _normalize_comb_key(comb):
            role_docs.update(role_to_documents.get(role, set()))

        partition_sels = []
        for pid in partition_ids:
            partition_docs = loads.get(pid, 0)
            if partition_docs > 0:
                assigned_docs = partition_assignment.get(pid, set()) if partition_assignment else set()
                overlap = len(role_docs & assigned_docs)
                if overlap > 0:
                    partition_sels.append(overlap / partition_docs)

        avg_sel = sum(partition_sels) / len(partition_sels) if partition_sels else 0.0
        key = _normalize_comb_key(comb)
        selectivity_map[key] = avg_sel

        weight = _resolve_weight(comb, role_weights)
        total_weighted_sel += avg_sel * weight
        total_weight += weight

    overall = total_weighted_sel / total_weight if total_weight > 0 else 0.0
    return overall, selectivity_map


def update_comb_role_tracker_stage1(comb, target_partition_id, temp_comb_trackers, max_partition_id):
    """
    Forcefully reassign all roles involved in `comb` to `target_partition_id` and remove the original partition.

    :param comb: tuple, the current comb being split
    :param target_partition_id: int, the new target partition ID
    :param temp_comb_trackers: dict, comb -> {partition_id: {roles}}, the tracker that needs to be updated
    """
    affected_combs = set()

    # Identify all affected combs that contain any role from `comb`
    for affected_comb in temp_comb_trackers.keys():
        if any(role in comb for role in affected_comb):
            affected_combs.add(affected_comb)

    # Force update all affected combs so that their roles are only provided by `target_partition_id`
    for affected_comb in affected_combs:
        new_partition_mapping = {}
        moved_roles = set()  # Track roles moved to `target_partition_id`

        for partition_id, roles in temp_comb_trackers[affected_comb].items():
            if partition_id != max_partition_id:
                new_partition_mapping[partition_id] = roles
                continue
            roles_to_move = roles & set(comb)  # Extract roles that overlap with `comb`

            if roles_to_move:
                moved_roles.update(roles_to_move)  # Record roles being moved
                updated_roles = roles - roles_to_move  # Remaining roles stay in the current partition

                if updated_roles:
                    new_partition_mapping[partition_id] = updated_roles  # Keep the partition if it still has roles
            else:
                new_partition_mapping[partition_id] = roles

        # Assign all moved roles to `target_partition_id`
        if moved_roles:
            new_partition_mapping[target_partition_id] = moved_roles

        temp_comb_trackers[affected_comb] = new_partition_mapping  # Update the tracker structure


def split_comb_roles(
    role_to_documents_index,
    alpha,
    vector_dim,
    gamma,
    role_combinations,
    combination_roles_to_documents,
    comb_role_weights,
    single_role_weights,
):
    partition_assignment = {0: set(doc for docs in role_to_documents_index.values() for doc in docs)}
    partition_loads = {0: len(partition_assignment[0])}
    documents_number = sum(partition_loads.values())
    comb_role_trackers = defaultdict(lambda: defaultdict(set))
    role_trackers = defaultdict(lambda: defaultdict(set))

    if documents_number == 0:
        return partition_assignment, comb_role_trackers

    for comb, docs in combination_roles_to_documents.items():
        for role in comb:  # Iterate through all roles in the current combination
            partition_id = 0  # Initially, all combinations are assigned to partition 0
            comb_role_trackers[comb][partition_id].add(role)

    while sum(partition_loads.values()) <= alpha * documents_number:
        # Select the largest partition
        sorted_partitions = sorted(partition_assignment.keys(), key=lambda pid: len(partition_assignment[pid]),
                                   reverse=True)
        for partition_id in sorted_partitions:
            max_partition_id = partition_id
            max_partition_combinations = set(
                comb for comb, partition_roles in comb_role_trackers.items()
                if max_partition_id in partition_roles and set(partition_roles[max_partition_id]) == set(comb)
            )
            if len(max_partition_combinations) > 1:
                break  # Stop when multiple combinations exist in the largest partition

        if len(max_partition_combinations) == 1 or len(max_partition_combinations) == 0:
            logger.info("Partition limit reached. No further split possible.")
            break  # Exit if no further splitting can be done

        query_time_in_comb_before = 0
        # Prepare the target partition list for splitting
        target_partition_list = [max(partition_assignment.keys()) + 1]

        # Use `comb_role_trackers` to determine involved combinations
        involved_combinations = set()

        # Iterate through `comb_role_trackers` to find combinations involving `max_partition_id` and `target_partition_list[0]`
        for comb, partition_roles in comb_role_trackers.items():  # `partition_roles` is {partition_id: set(roles)}
            if max_partition_id in partition_roles:  # Check if `partition_id` exists
                involved_combinations.add(comb)

        split_flag_count = 0
        for comb, partition_roles in comb_role_trackers.items():
            if len(comb) == 1:  # Only process single-role combinations
                role = next(iter(comb))  # Extract the only role
                for partition_id, roles in partition_roles.items():
                    role_trackers[frozenset({role})][partition_id] = roles.copy()  # Store role-partition mappings

        involved_roles = {role for role, partitions in role_trackers.items() if max_partition_id in partitions}
        while sum(partition_loads.values()) <= alpha * documents_number:
            sorted_partitions = sorted(partition_assignment.keys(), key=lambda pid: len(partition_assignment[pid]),
                                       reverse=True)
            for partition_id in sorted_partitions:
                current_max_partition_id = partition_id
                max_partition_combinations = set(
                    comb for comb, partition_roles in comb_role_trackers.items()
                    if current_max_partition_id in partition_roles and set(
                        partition_roles[current_max_partition_id]) == set(comb)
                )
                if len(max_partition_combinations) > 1:
                    break

            if current_max_partition_id != max_partition_id and len(
                    partition_assignment[current_max_partition_id]) != len(partition_assignment[max_partition_id]):
                break

            if query_time_in_comb_before == 0:
                _, sel_map_comb_before = compute_sel_whole(
                    comb_role_trackers,
                    partition_loads,
                    role_to_documents_index,
                    involved_combinations,
                    comb_role_weights,
                    partition_assignment,
                )
                query_time_in_comb_before = compute_query_time(
                    comb_role_trackers,
                    partition_loads,
                    sel_map_comb_before,
                    vector_dim,
                    gamma,
                    involved_combinations,
                    comb_role_weights,
                )
                _, sel_map_role_before = compute_sel_whole(
                    role_trackers,
                    partition_loads,
                    role_to_documents_index,
                    involved_roles,
                    single_role_weights,
                    partition_assignment,
                )
                query_time_in_role_before = compute_query_time(
                    role_trackers,
                    partition_loads,
                    sel_map_role_before,
                    vector_dim,
                    gamma,
                    involved_roles,
                    single_role_weights,
                )

            priority_queue = []

            for comb in max_partition_combinations:
                current_partitions = comb_role_trackers.get(comb)
                if not current_partitions or max_partition_id not in current_partitions:
                    continue
                if len(comb) > 1:
                    continue

                # Copy data to temporary structures
                temp_partition_assignment = {pid: docs.copy() for pid, docs in partition_assignment.items()}
                temp_partition_loads = partition_loads.copy()
                temp_comb_trackers = {comb: partitions.copy() for comb, partitions in comb_role_trackers.items()}

                # Compute the initial storage size before splitting
                prev_storage = sum(temp_partition_loads.values())

                # Assign the role to a new partition
                target_partition_id = target_partition_list[0]
                if target_partition_id not in temp_partition_assignment:
                    temp_partition_assignment[target_partition_id] = set()

                temp_partition_assignment[target_partition_id].update(combination_roles_to_documents[comb])

                update_comb_role_tracker_stage1(comb, target_partition_id, temp_comb_trackers, max_partition_id)

                temp_role_trackers = defaultdict(lambda: defaultdict(set))

                for comb_, partition_roles in temp_comb_trackers.items():
                    if len(comb_) == 1:  # Process only single-role combinations
                        role = next(iter(comb_))  # Extract the single role
                        for partition_id, roles in partition_roles.items():
                            temp_role_trackers[frozenset({role})][
                                partition_id] = roles.copy()  # Track roles for each partition

                # Identify roles currently assigned to `max_partition_id`
                remaining_roles_in_max_partition = set()
                for _, partition_roles in temp_comb_trackers.items():
                    if max_partition_id in partition_roles:
                        remaining_roles_in_max_partition.update(
                            partition_roles[max_partition_id])  # Collect roles in the max partition

                # Determine the documents still required in `max_partition_id`
                all_docs_in_remaining = set()
                for role in remaining_roles_in_max_partition:
                    all_docs_in_remaining.update(role_to_documents_index[role])

                # Keep only the documents related to the remaining roles in `max_partition_id`
                temp_partition_assignment[max_partition_id] &= all_docs_in_remaining

                temp_partition_loads[max_partition_id] = len(temp_partition_assignment[max_partition_id])
                temp_partition_loads[target_partition_id] = len(temp_partition_assignment[target_partition_id])

                # Compute the new storage size after splitting
                new_storage = sum(temp_partition_loads.values())
                storage_growth = (new_storage - prev_storage) / prev_storage if prev_storage > 0 else 0  # 避免除零错误

                # Compute the new query time
                _, sel_map_comb_after = compute_sel_whole(
                    temp_comb_trackers,
                    temp_partition_loads,
                    role_to_documents_index,
                    involved_combinations,
                    comb_role_weights,
                    temp_partition_assignment,
                )
                current_query_time_in_comb = compute_query_time(
                    temp_comb_trackers,
                    temp_partition_loads,
                    sel_map_comb_after,
                    vector_dim,
                    gamma,
                    involved_combinations,
                    comb_role_weights,
                )

                _, sel_map_role_after = compute_sel_whole(
                    temp_role_trackers,
                    temp_partition_loads,
                    role_to_documents_index,
                    involved_roles,
                    single_role_weights,
                    temp_partition_assignment,
                )
                current_query_time_in_role = compute_query_time(
                    temp_role_trackers,
                    temp_partition_loads,
                    sel_map_role_after,
                    vector_dim,
                    gamma,
                    involved_roles,
                    single_role_weights,
                )

                baseline_comb = max(query_time_in_comb_before, 1e-12)
                baseline_role = max(query_time_in_role_before, 1e-12)
                query_in_comb_delta = (current_query_time_in_comb - query_time_in_comb_before) / baseline_comb
                query_in_role_delta = (current_query_time_in_role - query_time_in_role_before) / baseline_role

                epsilon = 1e-10  # small number
                storage_flag = 1
                if storage_growth < 0:
                    storage_flag = -100

                combined_delta = storage_flag * (query_in_role_delta + query_in_comb_delta) / (
                    storage_growth + epsilon
                )
                if query_in_role_delta < 0 and query_in_comb_delta < 10:
                    heapq.heappush(
                        priority_queue,
                        (combined_delta, query_in_role_delta, query_in_comb_delta, comb, target_partition_id),
                    )

            if not priority_queue:
                break

            best_combined_delta, query_in_role_delta, query_in_comb_delta, best_comb, target_partition_id = heapq.heappop(
                priority_queue)
            logger.info(
                "Selected for splitting: best_role_delta=%.6f, best_comb_delta=%.6f, best_comb=%s, target_partition_id=%s",
                query_in_role_delta,
                query_in_comb_delta,
                best_comb,
                target_partition_id,
            )

            # Perform the split operation
            if target_partition_id not in partition_assignment:
                # If the target partition is new, initialize it
                partition_assignment[target_partition_id] = set()
                partition_loads[target_partition_id] = 0  # Initialize partition load

            partition_assignment[target_partition_id].update(combination_roles_to_documents[best_comb])

            update_comb_role_tracker_stage1(best_comb, target_partition_id, comb_role_trackers, max_partition_id)

            # Identify roles still assigned to `max_partition_id`
            remaining_roles_in_max_partition = set()
            for comb, partition_roles in comb_role_trackers.items():
                if max_partition_id in partition_roles:
                    remaining_roles_in_max_partition.update(
                        partition_roles[max_partition_id])  # Retrieve roles in the max partition

            # Determine the documents still required in `max_partition_id`
            all_docs_in_remaining = set()
            for role in remaining_roles_in_max_partition:
                all_docs_in_remaining.update(role_to_documents_index[role])

            # Retain only the documents relevant to `remaining_roles_in_max_partition` in `max_partition_id`
            partition_assignment[max_partition_id] &= all_docs_in_remaining
            partition_loads[max_partition_id] = len(partition_assignment[max_partition_id])  # Update partition size
            partition_loads[target_partition_id] = len(
                partition_assignment[target_partition_id])  # Update new partition size

            # Update previous query times with the computed deltas
            query_time_in_comb_before = max(current_query_time_in_comb, 1e-12)
            query_time_in_role_before = max(current_query_time_in_role, 1e-12)
            split_flag_count += 1  # Increment split operation counter

    return partition_assignment, comb_role_trackers


import heapq


def calculate_single_role_weights_from_queries(user_to_roles, role_combinations):
    """
    Calculate single role weights based on user queries in a JSON file.

    Args:
        user_to_roles (dict): Mapping of user_id to their roles.
        role_combinations (list of tuple): All unique role combinations.

    Returns:
        dict: A dictionary mapping single roles to their aggregated weights.
    """
    # Step 1: Parse the query JSON file
    user_weights = {}
    folder = os.path.join(project_root, "basic_benchmark")
    query_file_path = os.path.join(folder, "query_dataset.json")

    try:
        with open(query_file_path, 'r') as f:
            query_data = json.load(f)
            for entry in query_data:
                user_id = entry["user_id"]
                weight = entry.get("query_block_selectivity", 0)  # Default weight is 0 if not present
                user_weights[user_id] = weight
    except FileNotFoundError:
        logger.warning("⚠️ Query file %s not found.", query_file_path)
        return {}

    # Step 2: Aggregate weights by role combination
    combination_weights = defaultdict(float)

    for user_id, roles in user_to_roles.items():
        user_weight = user_weights.get(user_id, 0)
        role_combination = tuple(sorted(roles))  # Ensure consistent format
        combination_weights[role_combination] += user_weight

    # Step 3: Convert combination weights into single role weights
    single_role_weights = defaultdict(float)

    for role_combination, weight in combination_weights.items():
        for role in role_combination:
            single_role_weights[role] += weight  # Accumulate weights across all combinations

    # Step 4: Fill missing roles with a small default weight
    all_roles = {role for comb in role_combinations for role in comb}  # Extract all unique roles
    total_roles = len(all_roles)
    default_weight = 1 / (total_roles + 1e-6)  # Avoid division by zero

    role_weight_mapping = {role: default_weight for role in all_roles}

    # Merge computed weights with default values
    for role, weight in single_role_weights.items():
        role_weight_mapping[role] = weight

    return role_weight_mapping


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description="ACORN dynamic partition initializer.")
    parser.add_argument("--storage", type=float, default=1.5, help="Storage multiplier alpha.")
    parser.add_argument("--gamma", type=float, default=12.0, help="ACORN neighbour expansion factor γ.")
    parser.add_argument("--vector-dim", type=int, default=None, help="Vector dimensionality d (auto if omitted).")
    args = parser.parse_args()

    roles, documents, permissions, _, user_to_roles = fetch_initial_data()
    role_to_documents, document_to_index = prepare_background_data(roles, documents, permissions)
    role_to_documents_index = {
        role: {document_to_index[doc] for doc in docs if doc in document_to_index}
        for role, docs in role_to_documents.items()
    }

    role_combinations, comb_role_weights = init_user_role_combination_data()
    single_role_weights = calculate_single_role_weights_from_queries(user_to_roles, role_combinations)

    combination_roles_to_documents = {}
    for comb in role_combinations:
        indices = set()
        for role in comb:
            indices.update(role_to_documents_index.get(role, set()))
        combination_roles_to_documents[tuple(sorted(comb))] = indices

    all_roles = {role for comb in role_combinations for role in comb}
    for role in all_roles:
        combination_roles_to_documents.setdefault((role,), role_to_documents_index.get(role, set()))

    role_combinations = sorted(combination_roles_to_documents.keys())

    alpha = args.storage
    gamma = args.gamma
    vector_dim = args.vector_dim or get_document_vector_dimension()
    logger.info("Using alpha=%.3f, gamma=%.3f, vector_dim=%d", alpha, gamma, vector_dim)

    create_indexes(index_type="hnsw")
    create_indexes_for_all_role_tables(index_type="hnsw")

    partition_assignment, comb_role_trackers = split_comb_roles(
        role_to_documents_index,
        alpha,
        vector_dim,
        gamma,
        role_combinations,
        combination_roles_to_documents,
        comb_role_weights,
        single_role_weights,
    )

    delete_faiss_files(project_root)

    converted_comb_role_trackers = {
        comb: set(partition_roles.keys())
        for comb, partition_roles in comb_role_trackers.items()
    }

    load_result_to_database(partition_assignment, converted_comb_role_trackers, increment_update=False)
    initialize_dynamic_partition_tables_in_comb(index_type=None)
    logger.info("ACORN dynamic partition generation complete.")
