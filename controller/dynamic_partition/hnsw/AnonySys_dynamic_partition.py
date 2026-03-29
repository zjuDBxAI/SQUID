import json
import os

import sys
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)
from controller.dynamic_partition.load_result_to_database import load_result_to_database
from basic_benchmark.initialize_dynamic_partition_tables import initialize_dynamic_partition_tables_in_comb

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)
from services.config import get_db_connection
from controller.baseline.prefilter.initialize_partitions import create_indexes_for_all_role_tables
from controller.initialize_main_tables import create_indexes
from controller.dynamic_partition.hnsw.helper import (
    fetch_initial_data,
    prepare_background_data,
    delete_faiss_files,
    clean_empty_partitions,
    reorganize_partitions,
)
from controller.dynamic_partition.hnsw.heavy_partition_refine import (
    rebalance_heavy_partition,
    remap_comb_role_trackers,
)
from services.logger import get_logger

from controller.dynamic_partition.get_parameter import get_recall_parameters, get_QPS_parameters

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


def calculate_role_weights_from_queries(user_to_roles, role_combinations):
    """
    Calculate role weights based on user queries in a JSON file.

    Args:
        user_to_roles (dict): Mapping of user_id to their roles.
        role_combinations (list of tuple): All unique role combinations.
        query_file_path (str): Path to the JSON file containing user query data.

    Returns:
        dict: A dictionary mapping role combinations to their weights.
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
                weight = entry.get("query_block_selectivity", 0)
                user_weights[user_id] = weight
    except FileNotFoundError:
        logger.warning("Query file %s not found.", query_file_path)
        return {}

    # Step 2: Aggregate weights by role combination
    combination_weights = defaultdict(float)

    # Map user weights to role combinations
    for user_id, roles in user_to_roles.items():
        user_weight = user_weights.get(user_id, 0)
        role_combination = tuple(sorted(roles))
        combination_weights[role_combination] += user_weight

    # Step 3: Fill missing role combinations with zero weight
    role_weight_mapping = {tuple(sorted(comb)): 0 for comb in role_combinations}
    for role_combination, weight in combination_weights.items():
        # role_weight_mapping[role_combination] = 10 * weight + 1/len(role_combinations)
        role_weight_mapping[role_combination] = weight

    return role_weight_mapping


def compute_query_time(comb_trackers, loads, sel_whole, topk, k, beta, a, b, comb_to_update, role_weights=None,
                       recall=None):
    """
    Compute query time based on unique user role combinations and their weights.

    Args:
        comb_trackers (dict): Mapping of role combinations to the partitions they are assigned to.
        loads (dict): Mapping of partitions to their document counts.
        sel_whole (float): Overall selectivity.
        topk (int): Top K parameter for query optimization.
        k (float): Tuning parameter for query efficiency.
        beta (float): Tuning parameter for query efficiency.
        a (float): Weight for query cost model.
        b (float): Bias for query cost model.
        comb_to_update (set): Role combinations involved in the current update.
        role_weights (dict): Mapping of role combinations to their weights.

    Returns:
        float: Total query time considering weights and partitions.
    """
    # Calculate dynamic value for ef_search
    if recall is None:
        # by default, choosing the highest recall as possible.
        x = 3
        while (1 + x / 10) - k >= 1:
            x -= 1
        dynamic_value = 1 + x / 10
    else:
        # assume recall is very big, if recall is small, you should implement efs * su/k as described in paper eq.9
        dynamic_value = recall + 1 / 2

    safe_sel = max(sel_whole, 1e-6)
    delta = max(dynamic_value - k, 1e-6)
    inner = 1 / delta - 1
    if inner <= 0:
        inner = 1e-6
    safe_beta = beta if abs(beta) > 1e-6 else 1e-6

    ef_search = math.log(inner) / (-4 * safe_beta * safe_sel) * topk + k * topk / safe_sel

    total_query_time = 0
    for comb in comb_to_update:
        weight = role_weights.get(comb, 0) if role_weights else 1  # Get the weight of the combination
        if weight == 0:  # in this case , single role mode
            weight = role_weights.get(next(iter(comb)), 1) if comb else 0
        tracked_partitions = comb_trackers.get(comb, set())  # Get partitions for this combination

        for partition in tracked_partitions:
            if partition in loads:
                n = loads[partition]
                total_query_time += weight * math.log(n) * (a * ef_search + b)

    return total_query_time


def compute_sel_whole(comb_trackers, loads, role_to_documents, comb_to_update, role_weights=None,
                      partition_assignment=None):
    """
    Compute overall selectivity considering user role combinations and their weights.

    Args:
        comb_trackers (dict): Mapping of role combinations to the partitions they are assigned to.
        loads (dict): Mapping of partitions to their document counts.
        role_to_documents (dict): Mapping of roles to the documents they are associated with.
        comb_to_update (set): Role combinations involved in the current update.
        role_weights (dict): Mapping of role combinations to their weights.

    Returns:
        float: Weighted overall selectivity.
    """
    total_weighted_sel = 0
    total_weight = 0

    for comb in comb_to_update:
        tracked_partitions = comb_trackers.get(comb, set())  # Partitions tracked for this combination
        role_docs = set()
        for role in comb:
            role_docs.update(role_to_documents.get(role, set()))

        # Compute selectivity per partition and take the average
        partition_sels = []
        for pid in tracked_partitions:
            partition_docs = loads.get(pid, 0)
            if partition_docs > 0:
                sel = len(role_docs & partition_assignment.get(pid, set())) / partition_docs
                partition_sels.append(sel)

        avg_sel = sum(partition_sels) / len(partition_sels) if partition_sels else 0  # Average selectivity

        # Incorporate role weight into the calculation
        weight = role_weights.get(comb, 0) if role_weights else 1
        if weight == 0:  # in this case , single role mode
            weight = role_weights.get(next(iter(comb)), 1) if comb else 0
        total_weighted_sel += avg_sel * weight
        total_weight += weight

    # Return the weighted average selectivity
    return total_weighted_sel / total_weight if total_weight > 0 else 0


def update_partition_combinations(pid, role_trackers):
    """
    Update the combinations of roles within a partition.
    """
    # Get roles in the partition
    roles_in_partition = {
        role for role, partitions in role_trackers.items() if pid in partitions
    }

    # Ensure combinations are valid subsets of roles in the partition
    valid_combinations = {
        tuple(comb) for comb in role_combinations if set(comb).issubset(roles_in_partition)
    }

    return valid_combinations


import re


def parse_log_file(log_file):
    """
    Parse the log file to extract the history of split operations, ensuring correct parsing of `best_comb`.
    """
    split_steps = []
    try:
        with open(log_file, "r") as file:
            for line in file:
                # Use regex to extract `best_delta`, `best_comb`, and `target_partition_id`
                match = re.search(r"best_delta=([-0-9.]+), best_comb=\((.*?)\), target_partition_id=(\d+)", line)
                if match:
                    best_delta = float(match.group(1))  # Convert best_delta to a float
                    best_comb_str = match.group(2).strip()  # Extract and trim best_comb string

                    # Properly handle `best_comb`, including cases like `(9,)` and `(5, 9, 100)`
                    if best_comb_str.endswith(","):  # Single-element tuple case, e.g., `(9,)`
                        best_comb_str = best_comb_str[:-1]  # Remove the trailing comma
                    best_comb = tuple(
                        map(int, best_comb_str.split(","))) if best_comb_str else ()  # Convert to tuple of integers

                    target_partition_id = int(match.group(3))  # Convert target_partition_id to an integer
                    split_steps.append((best_delta, best_comb, target_partition_id))  # Store extracted values

        logger.debug("Loaded %d split steps from %s.", len(split_steps), log_file)
        return split_steps
    except FileNotFoundError:
        logger.warning("Log file %s not found. Recomputing...", log_file)
        return None
    except Exception as e:
        logger.error("Failed to parse log file %s: %s", log_file, e)
        return None


from itertools import combinations


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


def update_comb_role_tracker_stage2(comb, target_partition_id, comb_role_trackers, partition_assignment,
                                    role_to_documents_index, topk, k, beta, a, b):
    """
    Update `comb_role_trackers` to ensure `comb` and other affected combs select the optimal partition combination.

    :param comb: tuple, the comb that needs to be updated (already split into `target_partition_id`)
    :param target_partition_id: int, the new partition ID
    :param comb_role_trackers: dict, mapping comb -> {partition_id -> set(roles)}
    :param partition_assignment: dict, mapping partition_id -> set of documents
    :param role_to_documents_index: dict, mapping role -> set of documents
    """
    # Calculate the maximum recall allowed by the performance model curve
    x = 3
    while (1 + x / 10) - k >= 1:
        x -= 1
    dynamic_value = 1 + x / 10

    # Retrieve all documents required by `comb`
    comb_docs = set()
    for role in comb:
        comb_docs.update(role_to_documents_index[role])

    # Identify all other combs that contain roles from `comb` (affected combs)
    affected_combs = {other_comb for other_comb in comb_role_trackers if any(role in other_comb for role in comb)}

    # Ensure the affected combs are also considered for updates
    affected_combs.add(comb)

    for affected_comb in affected_combs:
        # Retrieve all documents required by `affected_comb`
        affected_comb_docs = set()
        for role in affected_comb:
            affected_comb_docs.update(role_to_documents_index[role])

        # Only consider partitions that `affected_comb` was previously assigned to + `target_partition_id`
        original_partitions = set(comb_role_trackers[affected_comb].keys())
        if {target_partition_id} == original_partitions:
            continue

        candidate_partitions = original_partitions | {target_partition_id}

        # Compute all possible partition combinations
        best_partitions = None
        best_query_time = float('inf')
        flag = False
        for r in range(1, len(candidate_partitions) + 1):  # Iterate through all possible partition combinations
            for partition_subset in combinations(candidate_partitions, r):
                # Calculate the document coverage of the partition combination
                covered_docs = set()
                for pid in partition_subset:
                    covered_docs.update(partition_assignment[pid])

                # Ensure the selected partition combination fully covers `affected_comb_docs`
                if not affected_comb_docs.issubset(covered_docs):
                    continue

                rows_product = 1
                total_sel = 0  # Compute the average selectivity and row product
                for pid in partition_subset:
                    partition_docs = partition_assignment[pid]
                    sel = len(affected_comb_docs & partition_docs) / len(partition_docs)  # Compute selectivity
                    total_sel += sel
                    rows_product *= len(partition_docs)
                avg_sel = total_sel / len(partition_subset)
                flag = True

                # Update the best partition combination
                ef_search = math.log(1 / (dynamic_value - k) - 1) / (
                        -4 * beta * avg_sel) * topk + k * topk / avg_sel
                query_time = math.log(rows_product) * (a * ef_search + b)

                if query_time < best_query_time:
                    best_query_time = query_time
                    best_partitions = partition_subset

        # Update `comb_role_trackers`
        if best_partitions:
            # Clear old partitions and reassign
            comb_role_trackers[affected_comb] = {pid: set() for pid in best_partitions}

            # Iterate through each role in `affected_comb` to determine the optimal partition
            for role in affected_comb:
                role_docs = role_to_documents_index[role]

                # Identify partitions that fully cover `role_docs`
                candidate_partitions = {pid for pid in best_partitions if role_docs.issubset(partition_assignment[pid])}

                if candidate_partitions:
                    # Select the partition with the least number of documents
                    best_partition = min(candidate_partitions, key=lambda pid: len(partition_assignment[pid]))

                    # Ensure `affected_comb` is initialized
                    if affected_comb not in comb_role_trackers:
                        comb_role_trackers[affected_comb] = {}

                    # Only update the partition assignment for `role`, without affecting others
                    if best_partition not in comb_role_trackers[affected_comb]:
                        comb_role_trackers[affected_comb][best_partition] = set()
                    comb_role_trackers[affected_comb][best_partition].add(role)

                else:
                    # If no single partition fully covers `role_docs`, assign `role` to all `best_partitions`
                    for pid in best_partitions:
                        if affected_comb not in comb_role_trackers:
                            comb_role_trackers[affected_comb] = {}

                        if pid not in comb_role_trackers[affected_comb]:
                            comb_role_trackers[affected_comb][pid] = set()
                        comb_role_trackers[affected_comb][pid].add(role)
        elif not flag:
            logger.warning("No valid partition set found for comb %s.", affected_comb)


def split_comb_roles(role_to_documents_index, alpha, topk, k, beta, a, b,
                     role_combinations, combination_roles_to_documents, comb_role_weights, single_role_weights,
                     combination_mode=False, recall = None):
    partition_assignment = {0: set(doc for docs in role_to_documents_index.values() for doc in docs)}
    partition_loads = {0: len(partition_assignment[0])}
    documents_number = sum(partition_loads.values())
    comb_role_trackers = defaultdict(lambda: defaultdict(set))
    role_trackers = defaultdict(lambda: defaultdict(set))

    for comb, docs in combination_roles_to_documents.items():
        for role in comb:  # Iterate through all roles in the current combination
            partition_id = 0  # Initially, all combinations are assigned to partition 0
            comb_role_trackers[comb][partition_id].add(role)

    combination_mode = combination_mode
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
                sel_whole_in_comb_before = compute_sel_whole(comb_role_trackers, partition_loads,
                                                             role_to_documents_index,
                                                             involved_combinations, comb_role_weights,
                                                             partition_assignment)
                query_time_in_comb_before = compute_query_time(comb_role_trackers, partition_loads,
                                                               sel_whole_in_comb_before, topk, k, beta, a, b,
                                                               involved_combinations, comb_role_weights, recall = recall)
                sel_whole_in_role_before = compute_sel_whole(role_trackers, partition_loads,
                                                             role_to_documents_index,
                                                             involved_roles, single_role_weights, partition_assignment)
                query_time_in_role_before = compute_query_time(role_trackers, partition_loads,
                                                               sel_whole_in_role_before, topk, k, beta, a, b,
                                                               involved_roles, single_role_weights, recall = recall)

            priority_queue = []

            for comb in max_partition_combinations:
                # In non-combination mode, prioritize splitting single roles
                if not combination_mode and len(comb) > 1:
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

                # Update `comb_role_trackers`
                if combination_mode:
                    update_comb_role_tracker_stage2(comb, target_partition_id, temp_comb_trackers,
                                                    temp_partition_assignment,
                                                    role_to_documents_index, topk, k, beta, a, b)
                else:
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
                sel_whole_in_comb_after = compute_sel_whole(temp_comb_trackers, temp_partition_loads,
                                                            role_to_documents_index,
                                                            involved_combinations, comb_role_weights,
                                                            temp_partition_assignment)
                current_query_time_in_comb = compute_query_time(temp_comb_trackers, temp_partition_loads,
                                                                sel_whole_in_comb_after, topk,
                                                                k, beta, a, b, involved_combinations, comb_role_weights, recall = recall)

                sel_whole_in_role_after = compute_sel_whole(temp_role_trackers, temp_partition_loads,
                                                            role_to_documents_index,
                                                            involved_roles, single_role_weights,
                                                            temp_partition_assignment)
                current_query_time_in_role = compute_query_time(temp_role_trackers, temp_partition_loads,
                                                                sel_whole_in_role_after, topk,
                                                                k, beta, a, b, involved_roles, single_role_weights, recall = recall)

                query_in_comb_delta = (
                                              current_query_time_in_comb - query_time_in_comb_before) / query_time_in_comb_before  # 查询时间的变化
                query_in_role_delta = (
                                              current_query_time_in_role - query_time_in_role_before) / query_time_in_role_before  # 查询时间的变化

                epsilon = 1e-10  # small number
                storage_flag = 1
                if storage_growth < 0:
                    storage_flag = -100

                if combination_mode == True:
                    combined_delta = storage_flag * query_in_comb_delta / (storage_growth + epsilon)  # 计算综合 delta
                    if query_in_comb_delta < 0:
                        heapq.heappush(priority_queue,
                                       (combined_delta, query_in_role_delta, query_in_comb_delta, comb,
                                        target_partition_id))

                else:
                    combined_delta = storage_flag * (query_in_role_delta + query_in_comb_delta) / (
                            storage_growth + epsilon)
                    if query_in_role_delta < 0 and query_in_comb_delta < 10:
                        heapq.heappush(priority_queue, (
                            combined_delta, query_in_role_delta, query_in_comb_delta, comb, target_partition_id))

            if not priority_queue and split_flag_count == 0:
                combination_mode = True
                logger.info("Switching to combination mode")
                continue
            elif not priority_queue:
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

            if combination_mode == True:
                logger.debug("Start stage2 combination-mode updates")
                update_comb_role_tracker_stage2(best_comb, target_partition_id, comb_role_trackers,
                                                partition_assignment,
                                                role_to_documents_index, topk, k, beta, a, b)
            else:
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
            query_time_in_comb_before += query_in_comb_delta * query_time_in_comb_before
            query_time_in_role_before += query_in_role_delta * query_time_in_role_before
            split_flag_count += 1  # Increment split operation counter

    return partition_assignment, comb_role_trackers


from copy import deepcopy
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

    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Initialize dynamic partition tables with configurable parameters.")
    parser.add_argument('--storage', type=float, default=1.5,
                        help="Storage parameter (alpha). Higher values increase storage. Default is 1.5")
    parser.add_argument('--recall', type=float, default=None,
                        help="Recall parameter. Default is None (will use parameters from file)")

    # Parse arguments
    args = parser.parse_args()


    roles, documents, permissions, avg_blocks_per_document, user_to_roles = fetch_initial_data()
    # Prepare background data
    role_to_documents, document_to_index = prepare_background_data(roles, documents, permissions)
    role_to_documents_index = {
        role: set(sorted({document_to_index[doc] for doc in docs if doc in document_to_index}))
        # Sort and convert to set
        for role, docs in role_to_documents.items()
    }

    document_index_to_roles: Dict[int, Set[int]] = defaultdict(set)
    for role, doc_indices in role_to_documents_index.items():
        for idx in doc_indices:
            document_index_to_roles[idx].add(role)

    role_combinations, comb_role_weights = init_user_role_combination_data()

    single_role_weights = calculate_single_role_weights_from_queries(user_to_roles, role_combinations)
    combination_roles_to_documents = {}

    for comb in role_combinations:
        all_documents = set()  # Collect all documents from roles in this combination

        for role in comb:
            if role in role_to_documents:
                all_documents.update(role_to_documents_index[role])  # Collect document IDs

        combination_roles_to_documents[tuple(comb)] = all_documents

    # **Extension: Add all individual roles as new combinations**
    all_roles = {role for comb in role_combinations for role in comb}  # Retrieve all roles that have appeared

    for role in all_roles:
        if (role,) not in combination_roles_to_documents:  # Prevent duplicate entries
            combination_roles_to_documents[(role,)] = role_to_documents_index.get(role,
                                                                                  set())  # Directly map the role to its documents

    # **Expand role_combinations**
    expanded_role_combinations = set(role_combinations)  # Use a set to remove duplicates
    for role in all_roles:
        expanded_role_combinations.add((role,))  # Add individual role combinations

    role_combinations = expanded_role_combinations

    m = len(roles)  # Number of roles
    n = len(documents)  # Total number of documents
    c = m  # Number of partitions
    alpha = args.storage
    recall = args.recall
    topk = 10
    
    # Define the JSON file path (fixed location in controller/dynamic_partition/hnsw/)
    json_file_path = os.path.join(project_root, "controller", "dynamic_partition", "hnsw", "parameter_hnsw.json")


    result_data = {}

    create_indexes(index_type="hnsw")
    create_indexes_for_all_role_tables(index_type="hnsw")
    # Check if the file already exists
    if os.path.exists(json_file_path):
        # If the file exists, read and return data
        with open(json_file_path, "r") as json_file:
            result_data = json.load(json_file)
            logger.info("Data loaded from parameter_hnsw.json")
    else:
        # If the file does not exist, generate data and save it
        params_recall = get_recall_parameters(index_type="hnsw")
        k = params_recall[0]
        beta = params_recall[1]
        params_qps, join_times = get_QPS_parameters(index_type="hnsw")
        a = params_qps[0]
        b = params_qps[1]
        logger.info("Parameters:")
        logger.info("  k: %s", k)
        logger.info("  beta: %s", beta)
        logger.info("  a: %s", a)
        logger.info("  b: %s", b)
        logger.info("  join_times: %s", join_times)
        result_data = {
            "k": k,
            "beta": beta,
            "a": a,
            "b": b,
            "join_times": join_times
        }
        # Write the results to the JSON file
        with open(json_file_path, "w") as json_file:
            json.dump(result_data, json_file, indent=4)
        logger.info("Data written to parameter_hnsw.json")

    default_ef_search = 5

    partition_assignment, comb_role_trackers = split_comb_roles(role_to_documents_index, alpha, topk, result_data["k"],
                                                                result_data["beta"], result_data["a"], result_data["b"],
                                                                role_combinations, combination_roles_to_documents,
                                                                comb_role_weights, single_role_weights,
                                                                combination_mode=False, recall = recall)

    logger.info(
        "Starting targeted greedy refinement (initial partitions: %s)",
        len(partition_assignment),
    )
    sorted_partitions = sorted(
        ((pid, len(docs)) for pid, docs in partition_assignment.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    largest_partition_id = sorted_partitions[0][0] if sorted_partitions else None
    largest_size = sorted_partitions[0][1] if sorted_partitions else 0
    second_size = sorted_partitions[1][1] if len(sorted_partitions) > 1 else 0
    if largest_partition_id is not None and largest_size > 0:
        ratio = float('inf') if second_size == 0 else largest_size / max(second_size, 1)
        logger.info(
            "Refining partition %s (size=%s, second=%s, ratio=%.2f)",
            largest_partition_id,
            largest_size,
            second_size,
            ratio,
        )
        partition_assignment, comb_role_trackers = rebalance_heavy_partition(
            partition_assignment,
            comb_role_trackers,
            document_index_to_roles,
            logger,
            target_partitions={largest_partition_id},
        )
        partition_assignment = clean_empty_partitions(partition_assignment)
        partition_assignment, partition_mapping = reorganize_partitions(partition_assignment)
        comb_role_trackers = remap_comb_role_trackers(comb_role_trackers, partition_mapping)
        logger.info(
            "Role refinement produced %s partitions after reindexing",
            len(partition_assignment),
        )
        logger.info("Sample comb_role_trackers mapping (up to 10 entries):")
        for idx, (comb, mapping) in enumerate(sorted(comb_role_trackers.items(), key=lambda item: item[0])):
            logger.info("  comb=%s -> partitions=%s", comb, sorted(mapping.keys()))
            if idx >= 9:
                break
        tracked_partitions = sorted({
            pid for partition_roles in comb_role_trackers.values() for pid in partition_roles.keys()
        })
        empty_mappings = [comb for comb, mapping in comb_role_trackers.items() if not mapping]
        logger.info(
            "comb_role_trackers covers %s combinations across %s partitions after refinement",
            len(comb_role_trackers),
            len(tracked_partitions),
        )
        if empty_mappings:
            logger.warning(
                "Combinations with no partitions after refinement (showing up to 5): %s",
                empty_mappings[:5],
            )
    else:
        logger.info("No non-empty partitions found for greedy refinement; skipping.")

    delete_faiss_files(project_root)

    converted_comb_role_trackers = {
        comb: set(partition_roles.keys())
        for comb, partition_roles in comb_role_trackers.items()
    }

    roles_in_partition_0: Set[int] = set()
    for partition_mapping in comb_role_trackers.values():
        if 0 in partition_mapping:
            roles_in_partition_0.update(partition_mapping[0])

    logger.info(
        "Number of unique roles assigned to partition_id 0 after refinement: %d",
        len(roles_in_partition_0),
    )

    load_result_to_database(partition_assignment, converted_comb_role_trackers, increment_update=False)
    initialize_dynamic_partition_tables_in_comb(index_type=None)
    initialize_dynamic_partition_tables_in_comb(index_type="hnsw")
    logger.info("done")
