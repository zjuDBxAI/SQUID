import json
import os
import sys
import shutil
import heapq
from collections import defaultdict
from itertools import combinations
from copy import deepcopy
import math

# File is in logical_partition_benchmark/dynamic_logical_partition/
# Project root is 2 levels up
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

# Import shared functions from physical partition implementation
from controller.dynamic_partition.hnsw.AnonySys_dynamic_partition import (
    init_user_role_combination_data,
    calculate_role_weights_from_queries,
    compute_query_time,
    compute_sel_whole,
    update_comb_role_tracker_stage1,
    update_comb_role_tracker_stage2,
    calculate_single_role_weights_from_queries
)

from controller.dynamic_partition.load_result_to_database import load_result_to_database
from basic_benchmark.initialize_dynamic_partition_tables import initialize_dynamic_partition_tables_in_comb
from services.config import get_db_connection
from controller.baseline.prefilter.initialize_partitions import create_indexes_for_all_role_tables
from controller.initialize_main_tables import create_indexes
from controller.dynamic_partition.hnsw.helper import fetch_initial_data, prepare_background_data, delete_faiss_files
from services.logger import get_logger
from controller.dynamic_partition.get_parameter import get_recall_parameters, get_QPS_parameters

logger = get_logger(__name__)


def load_hnsw_config():
    """
    Load HNSW configuration from hnsw_config.json.

    Returns:
        dict: Configuration containing M and ef_construction parameters
    """
    config_path = os.path.join(os.path.dirname(__file__), "hnsw_config.json")

    if not os.path.exists(config_path):
        logger.warning("HNSW config file not found at %s, using default values", config_path)
        return {"M": 16, "ef_construction": 64}

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
            logger.info("HNSW config loaded: M=%d, ef_construction=%d", config["M"], config["ef_construction"])
            return config
    except Exception as e:
        logger.error("Error loading HNSW config: %s. Using default values.", e)
        return {"M": 16, "ef_construction": 64}


def get_vector_dimension():
    """
    Query the database to get the vector dimension from documentblocks table.

    Returns:
        int: Vector dimension
    """
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        # Use format_type to get the vector dimension from column definition
        # For vector(300), this will return "vector(300)"
        cur.execute("""
            SELECT format_type(atttypid, atttypmod) AS column_type
            FROM pg_attribute
            WHERE attrelid = 'documentblocks'::regclass
            AND attname = 'vector'
        """)
        result = cur.fetchone()
        if result and result[0]:
            # Parse dimension from "vector(300)" -> 300
            import re
            match = re.search(r'vector\((\d+)\)', result[0])
            if match:
                return int(match.group(1))

        # Fallback: query actual vector data to get dimension
        cur.execute("""
            SELECT vector_dims(vector) AS dimension
            FROM documentblocks
            LIMIT 1
        """)
        result = cur.fetchone()
        if result and result[0]:
            return result[0]

        logger.warning("Could not determine vector dimension from database, using default 300")
        return 300
    except Exception as e:
        logger.error("Error querying vector dimension: %s", e)
        return 300
    finally:
        cur.close()
        conn.close()


def compute_logical_storage(partition_assignment, documents_number, vector_dimension, hnsw_config=None):
    """
    Compute storage for logical partition mode.

    Storage = Shared vector table + Sum of HNSW graph structures (pointers only)

    In logical partition mode:
    - Vectors are stored once in a shared table
    - Each partition has its own HNSW graph structure (only pointers, no vectors)
    - Baseline (1x) = shared vectors + single HNSW graph for all documents

    Args:
        partition_assignment (dict): Mapping of partition_id -> set of document indices
        documents_number (int): Total number of documents
        vector_dimension (int): Dimension of vectors
        hnsw_config (dict): HNSW configuration with M and ef_construction. If None, loads from config file.

    Returns:
        float: Storage ratio (current_storage / baseline_storage)
    """
    # Load HNSW config if not provided
    if hnsw_config is None:
        hnsw_config = load_hnsw_config()

    # Shared vector table storage (stored once, regardless of partitions)
    # Each vector: vector_dimension * 4 bytes (float32)
    shared_vector_storage = documents_number * vector_dimension * 4

    # HNSW graph structure storage for each partition
    # Based on HNSW implementation:
    # - Base layer: M * 2 * 4 bytes (each node has up to 2M edges)
    # - Upper layers: M * 1 * 4 bytes (average across layers)
    # - Total: M * 3 * 4 bytes per node
    # Reference: https://stackoverflow.com/questions/77401874
    M = hnsw_config["M"]  # HNSW M parameter from config
    bytes_per_node_in_graph = M * 3 * 4  # M * 3 * 4 bytes per node

    total_graph_storage = 0
    for pid, docs in partition_assignment.items():
        num_docs_in_partition = len(docs)
        partition_graph_storage = num_docs_in_partition * bytes_per_node_in_graph
        total_graph_storage += partition_graph_storage

    total_storage = shared_vector_storage + total_graph_storage

    # Baseline (1x): shared vectors + single HNSW graph for the entire dataset
    baseline_storage = shared_vector_storage + documents_number * bytes_per_node_in_graph

    return total_storage / baseline_storage


def split_comb_roles_logical(role_to_documents_index, alpha, topk, k, beta, a, b,
                             role_combinations, combination_roles_to_documents, comb_role_weights, single_role_weights,
                             vector_dimension, combination_mode=False, recall=None, hnsw_config=None):
    """
    Split roles into partitions using logical partition mode.

    In logical partition mode, storage is computed as:
    Storage = Shared vector table + Sum of HNSW graph structures (pointers only)

    Args:
        alpha (float): Storage budget multiplier
        vector_dimension (int): Dimension of embedding vectors (queried from database)
        hnsw_config (dict): HNSW configuration with M and ef_construction. If None, loads from config file.
    """
    # Load HNSW config if not provided
    if hnsw_config is None:
        hnsw_config = load_hnsw_config()

    partition_assignment = {0: set(doc for docs in role_to_documents_index.values() for doc in docs)}
    partition_loads = {0: len(partition_assignment[0])}
    documents_number = len(partition_assignment[0])
    comb_role_trackers = defaultdict(lambda: defaultdict(set))
    role_trackers = defaultdict(lambda: defaultdict(set))

    for comb, docs in combination_roles_to_documents.items():
        for role in comb:  # Iterate through all roles in the current combination
            partition_id = 0  # Initially, all combinations are assigned to partition 0
            comb_role_trackers[comb][partition_id].add(role)

    combination_mode = combination_mode

    # Use logical partition storage calculation instead of physical
    while compute_logical_storage(partition_assignment, documents_number, vector_dimension, hnsw_config) <= alpha:
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
        while compute_logical_storage(partition_assignment, documents_number, vector_dimension, hnsw_config) <= alpha:
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

                # Compute the initial storage size before splitting (logical partition mode)
                prev_storage = compute_logical_storage(temp_partition_assignment, documents_number, vector_dimension, hnsw_config)

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

                # Compute the new storage size after splitting (logical partition mode)
                new_storage = compute_logical_storage(temp_partition_assignment, documents_number, vector_dimension, hnsw_config)
                storage_growth = (new_storage - prev_storage) / prev_storage if prev_storage > 0 else 0

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


def delete_benchmark_dynamic_partition_indexes():
    """
    Remove dynamic partition index artifacts generated by the benchmark runs.
    """
    config_path = os.path.join(project_root, "logical_partition_benchmark", "benchmark", "config.json")
    index_root = None
    try:
        with open(config_path, "r") as cfg_file:
            cfg = json.load(cfg_file)
            index_root = cfg.get("index_storage_path")
    except FileNotFoundError:
        logger.warning("Benchmark config not found at %s; skip index cleanup.", config_path)
        return
    except Exception as exc:
        logger.warning("Failed to read benchmark config %s (%s); skip index cleanup.", config_path, exc)
        return

    if not index_root:
        logger.warning("`index_storage_path` not set in benchmark config (%s); skip index cleanup.", config_path)
        return

    paths_to_remove = [
        os.path.join(index_root, "dynamic_partition"),
        os.path.join(index_root, "physical_dynamic_partition"),
    ]

    for path in paths_to_remove:
        if os.path.isdir(path):
            try:
                shutil.rmtree(path)
                logger.info("Removed benchmark dynamic partition index directory: %s", path)
            except Exception as exc:
                logger.warning("Failed to remove directory %s: %s", path, exc)
        else:
            logger.debug("Benchmark dynamic partition index directory not found, skipping: %s", path)


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
            logger.info("Data loaded from %s", json_file_path)
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

    # Get vector dimension from database
    vector_dim = get_vector_dimension()
    logger.info("Vector dimension: %d", vector_dim)

    # Load HNSW configuration
    hnsw_config = load_hnsw_config()
    logger.info("Using HNSW config: M=%d, ef_construction=%d", hnsw_config["M"], hnsw_config["ef_construction"])

    # Use logical partition mode
    partition_assignment, comb_role_trackers = split_comb_roles_logical(
        role_to_documents_index, alpha, topk, result_data["k"],
        result_data["beta"], result_data["a"], result_data["b"],
        role_combinations, combination_roles_to_documents,
        comb_role_weights, single_role_weights, vector_dim,
        combination_mode=False, recall=recall, hnsw_config=hnsw_config
    )

    converted_comb_role_trackers = {
        comb: set(partition_roles.keys())  # Take the keys of partition_roles as the set of partition_ids
        for comb, partition_roles in comb_role_trackers.items()
    }

    roles_in_partition_0 = set()
    # Iterate through comb_role_trackers to find roles in partition_id=0
    for comb, partition_mapping in comb_role_trackers.items():
        if 0 in partition_mapping:  # Check if partition_id=0 exists
            roles_in_partition_0.update(partition_mapping[0])  # Collect roles

    # Count the number of unique roles
    count_roles_in_partition_0 = len(roles_in_partition_0)

    logger.info("Number of unique roles assigned to partition_id 0: %d", count_roles_in_partition_0)

    # delete partition index in acorn_benchmark
    delete_faiss_files(project_root)

    load_result_to_database(partition_assignment, converted_comb_role_trackers, increment_update=False)
    delete_benchmark_dynamic_partition_indexes()
    logger.info("done")
