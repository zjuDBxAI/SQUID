import sys

import matplotlib.pyplot as plt
import json
import os
import numpy as np
from psycopg2 import sql

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)
print(project_root)
from services.read_dataset_function import generate_query_dataset
from basic_benchmark.initialize_dynamic_partition_tables import initialize_dynamic_partition_tables
from controller.dynamic_partition.hnsw.analysis.analysis_hnsw_recall import calculate_actual_recall_batch
from controller.dynamic_partition.hnsw.helper import calculate_hnsw_recall, fetch_initial_data, prepare_background_data
from controller.dynamic_partition.search import merge_results

# Row-level security imports
from controller.baseline.pg_row_security.row_level_security import (
    disable_row_level_security, drop_database_users, create_database_users, enable_row_level_security,
    get_db_connection_for_many_users
)
from basic_benchmark.common_function import ground_truth_func

topk = None
sel = None


def piecewise_recall_model(x, k, beta):
    """
    Piecewise model combining a linear function and a shifted sigmoid function:
    - Linear for x <= k * topk
    - Sigmoid for x > k * topk
    """
    global sel, topk

    # Calculate x_c as proportional to topk
    x_c = k * topk / sel

    # Sigmoid growth rate
    b = beta * 4 * sel / topk

    # Shift for smooth transition
    shift = x_c * sel / topk - 0.5

    # Piecewise function
    return np.piecewise(
        x,
        [x <= x_c, x > x_c],
        [
            lambda x: x * sel / topk,  # Linear part
            lambda x: (1 / (1 + np.exp(-b * (x - x_c)))) + shift  # Sigmoid part
        ]
    )


def dynamic_partition_recall_analysis(user_id, query_vector, topk=5, ef_searchs=None):
    """
    Search using SQL query execution time for performance statistics with RLS.
    Supports multiple ef_search values in a single function call.

    :param user_id: User ID for which to perform the recall analysis.
    :param query_vector: Query vector for similarity search.
    :param topk: Number of top results to retrieve.
    :param ef_searchs: List of ef_search values to evaluate.
    :return: A dictionary mapping each ef_search to its recall results.
    """
    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()
    cur.execute(f"SET jit = off;")
    results = {}

    # Pre-compute ground truth outside the ef_search loop
    ground_truth = ground_truth_func(user_id=user_id, query_vector=query_vector, topk=topk)
    ground_truth_set = set((result[1], result[0]) for result in ground_truth)
    for ef_search in ef_searchs:
        cur.execute(f"SET hnsw.ef_search = {ef_search};")
        all_results = []

        # Fetch partition IDs
        cur.execute("""
            SELECT partition_id FROM RolePartitions rp
            JOIN UserRoles ur ON rp.role_id = ur.role_id
            WHERE ur.user_id = %s;
        """, [user_id])
        accessible_partitions = {row[0] for row in cur.fetchall()}

        # Search each partition and collect results
        for partition_id in accessible_partitions:
            partition_table = sql.Identifier(f"documentblocks_partition_{partition_id}")

            # Execute actual search
            query = sql.SQL(
                """
                SELECT block_id, document_id, block_content, 
                       vector <-> %s::vector AS distance
                FROM {}
                ORDER BY distance
                LIMIT %s;
                """
            ).format(partition_table)

            cur.execute(query, [query_vector, topk])
            all_results.extend(cur.fetchall())

        # Merge results and calculate recall
        merged_results = merge_results(all_results, topk)
        # Convert merged results into a set of (document_id, block_id)
        retrieved_set = set((result[1], result[0]) for result in merged_results)

        # Calculate recall as the ratio of correct matches to ground truth size
        correct_matches = len(retrieved_set & ground_truth_set)
        recall = correct_matches / len(ground_truth_set) if ground_truth_set else 0

        results[ef_search] = recall

    cur.close()
    conn.close()

    return results


def dynamic_partition_recall_analysis_with_groundtruth(user_id, query_vector, topk, ef_searchs, ground_truth_set):
    """
    Optimized version that accepts pre-computed ground truth set to avoid redundant computation.
    """
    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()
    cur.execute(f"SET jit = off;")
    results = {}

    for ef_search in ef_searchs:
        cur.execute(f"SET hnsw.ef_search = {ef_search};")
        all_results = []

        # Fetch partition IDs
        cur.execute("""
            SELECT partition_id FROM RolePartitions rp
            JOIN UserRoles ur ON rp.role_id = ur.role_id
            WHERE ur.user_id = %s;
        """, [user_id])
        accessible_partitions = {row[0] for row in cur.fetchall()}

        # Search each partition and collect results
        for partition_id in accessible_partitions:
            partition_table = sql.Identifier(f"documentblocks_partition_{partition_id}")

            query = sql.SQL(
                """
                SELECT block_id, document_id, block_content,
                       vector <-> %s::vector AS distance
                FROM {}
                ORDER BY distance
                LIMIT %s;
                """
            ).format(partition_table)

            cur.execute(query, [query_vector, topk])
            all_results.extend(cur.fetchall())

        # Merge results and calculate recall
        merged_results = merge_results(all_results, topk)
        retrieved_set = set((result[1], result[0]) for result in merged_results)

        # Calculate recall using pre-computed ground truth
        correct_matches = len(retrieved_set & ground_truth_set)
        recall = correct_matches / len(ground_truth_set) if ground_truth_set else 0
        results[ef_search] = recall

    cur.close()
    conn.close()
    return results


def calculate_hnsw_recall_global(user_id, ef_search_values, topk, p, x, roles, role_to_documents, document_to_index, c,
                                 n, m,
                                 k=1,
                                 beta=0.44240961):
    """
    Calculate recall based on the given formula for each role, and return the average recall.

    :param ef_search: Current ef_search value.
    :param topk: Number of top results needed.
    :param p: Dictionary with keys (j, k) and values for p[j,k] (whether document j is in partition k).
    :param x: Dictionary with keys (i, k) and values for x[i,k] (whether partition k is required by role i).
    :param role_to_documents: Dictionary mapping each role to its required documents.
    :param c: Number of partitions.
    :param n: Total number of documents.
    :param k: Coefficient k (default 1 as per the formula).
    :param beta: Coefficient beta (default 0.44240961).
    :return: Average recall across all roles.
    """
    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()

    # Fetch roles associated with the user
    cur.execute("SELECT role_id FROM UserRoles WHERE user_id = %s;", [user_id])
    user_roles = [row[0] for row in cur.fetchall()]
    role_recalls = []  # To store recall for each role
    results = {}

    # Step 1: Calculate recall for each role
    sum_partition_count = 0
    sum_required_count = 0
    for ef_search in ef_search_values:
        for role_id in user_roles:
            if role_id not in role_to_documents:
                continue  # Skip roles without document mappings
            documents = role_to_documents[role_id]
            role_sel_sum = 0  # Sel sum for the current role
            doc_indices = [document_to_index[doc] for doc in documents]
            # Iterate over all partitions for this role
            for k in range(c):
                if x.get((role_id - 1, k), 0):  # If partition k is required by this role
                    partition_total_document_count = sum(
                        p.get((j, k), 0) for j in range(n))  # Total documents in partition k

                    if partition_total_document_count == 0:
                        continue  # Skip partitions with no documents

                    # Calculate required documents in this partition for the current role
                    partition_required_document_count = sum(1 for doc_idx in doc_indices if p.get((doc_idx, k), 0))

                    # Calculate sel for this partition and add to the role's sel sum
                    role_sel_sum += partition_required_document_count / partition_total_document_count
                    sum_partition_count += partition_total_document_count
                    sum_required_count += partition_required_document_count

        role_sel_whole = sum_required_count / sum_partition_count

        threshold = k * topk / role_sel_whole
        if ef_search <= threshold:
            recall = ef_search * role_sel_whole / topk
        else:
            exponent = -4 * beta * role_sel_whole / topk * (ef_search - threshold)
            recall = 1 / (1 + np.exp(exponent)) + (k - 0.5)
        results[ef_search] = min(1, recall)

    cur.close()
    conn.close()
    return results


def validate_recall_model_with_avg(query_dataset, ef_search_values, topk, p, x, roles, role_to_documents,
                                   document_to_index,
                                   c, n, m, k=1, beta=0.44240961):
    """
    Validate the relationship between ef_search and recall using experiment results and formula predictions.

    Parameters:
    - query_dataset: List of query data with user_id, query_vector, etc.
    - ef_search_values: List of ef_search values for validation.
    - topk: Number of top results needed.
    - p, x, roles, role_to_documents: Parameters required for recall calculation.
    - document_to_index: Mapping of documents to their index in the data.
    - avg_blocks_per_document: Average blocks per document.
    - c: Number of partitions.
    - n: Total number of documents.
    - m: Total number of roles.
    - k, beta: Coefficients for the fitted recall model.
    """
    # Store actual and formula-predicted recall values
    actual_recalls = {ef_search: [] for ef_search in ef_search_values}
    formula_recalls = []

    # Batch compute all ground truths first (MUCH faster!)
    from basic_benchmark.common_function import ground_truth_func_batch, set_ground_truth_total_queries
    set_ground_truth_total_queries(len(query_dataset))
    print(f"Computing ground truth for {len(query_dataset)} queries in batch mode...")
    ground_truth_results_list = ground_truth_func_batch(query_dataset)
    print(f"Ground truth computation complete!")

    # Run experiments to get actual recall values
    for i, query in enumerate(query_dataset):
        user_id = query["user_id"]
        query_vector = query["query_vector"]

        # Use pre-computed ground truth
        ground_truth = ground_truth_results_list[i]
        ground_truth_set = set((result[1], result[0]) for result in ground_truth)

        # Calculate actual recall for all ef_search values (reuse ground truth)
        query_results = dynamic_partition_recall_analysis_with_groundtruth(
            user_id, query_vector, topk, ef_search_values, ground_truth_set
        )

        for ef_search, recall in query_results.items():
            actual_recalls[ef_search].append(recall)

    # Calculate formula-predicted recall for each ef_search
    for ef_search in ef_search_values:
        recall = calculate_hnsw_recall(ef_search, topk, p, x, roles, role_to_documents, document_to_index,
                                       c, n, m, k, beta)
        formula_recalls.append(recall)

    # Calculate the average actual recall for each ef_search
    avg_actual_recalls = [np.mean(actual_recalls[ef_search]) if actual_recalls[ef_search] else 0
                          for ef_search in ef_search_values]

    print(f"Actual Recalls: {avg_actual_recalls}")
    print(f"Formula Predicted Recalls: {formula_recalls}")


    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(ef_search_values, avg_actual_recalls, label="Actual Recalls", marker="o", linestyle="-", color="blue")
    plt.plot(ef_search_values, formula_recalls, label="Formula Predicted Recalls", marker="s", linestyle="--",
             color="red")
    plt.xlabel("Ef_Search")
    plt.ylabel("Recall")
    plt.title("Recall Validation: Actual vs Formula")
    plt.legend()
    plt.grid(True)

    # Adjust x-axis and y-axis ticks
    if max(ef_search_values) < 100:
        plt.xticks(ticks=range(0, max(ef_search_values) + 1, 10))  # Set x-axis ticks every 100
    else:
        plt.xticks(ticks=range(0, max(ef_search_values) + 1, 100))  # Set x-axis ticks every 100

    plt.yticks(ticks=np.arange(0, 1.01, 0.1))  # Set y-axis ticks from 0 to 1.0 every 0.1

    # Force y-axis to start from 0
    plt.ylim(0, 1.0)
    # Save the validation plot
    validation_plot_filename = "recall_validation_adjusted.png"
    plt.savefig(validation_plot_filename, dpi=300)
    plt.show()

    print(f"Validation plot saved to: {validation_plot_filename}")


def validate_recall_model_with_per(query_dataset, ef_search_values, topk, p, x, roles, role_to_documents,
                                   document_to_index,
                                   c, n, m, k=1, beta=0.44240961):
    """
    Validate the relationship between ef_search and recall using experiment results and formula predictions.

    Parameters:
    - query_dataset: List of query data with user_id, query_vector, etc.
    - ef_search_values: List of ef_search values for validation.
    - topk: Number of top results needed.
    - p, x, roles, role_to_documents: Parameters required for recall calculation.
    - document_to_index: Mapping of documents to their index in the data.
    - avg_blocks_per_document: Average blocks per document.
    - c: Number of partitions.
    - n: Total number of documents.
    - m: Total number of roles.
    - k, beta: Coefficients for the fitted recall model.
    """
    # Store actual and formula-predicted recall values
    actual_recalls = {ef_search: [] for ef_search in ef_search_values}
    formula_recalls = {ef_search: [] for ef_search in ef_search_values}

    # Run experiments to get actual recall values
    for query in query_dataset:
        user_id = query["user_id"]
        query_vector = query["query_vector"]

        # Calculate actual recall for all ef_search values
        query_results = dynamic_partition_recall_analysis(user_id, query_vector, topk=topk, ef_searchs=ef_search_values)
        formula_results = calculate_hnsw_recall_global(user_id, ef_search_values, topk, p, x, roles, role_to_documents,
                                                       document_to_index,
                                                       c, n, m, k, beta)
        for ef_search, recall in query_results.items():
            actual_recalls[ef_search].append(recall)

        for ef_search, recall in formula_results.items():
            formula_recalls[ef_search].append(recall)

    # Calculate the average actual recall for each ef_search
    avg_actual_recalls = [np.mean(actual_recalls[ef_search]) if actual_recalls[ef_search] else 0
                          for ef_search in ef_search_values]
    avg_formula_recalls = [np.mean(formula_recalls[ef_search]) if formula_recalls[ef_search] else 0
                           for ef_search in ef_search_values]
    print(f"Actual Recalls: {avg_actual_recalls}")
    print(f"Formula Predicted Recalls: {avg_formula_recalls}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(ef_search_values, avg_actual_recalls, label="Actual Recalls", marker="o", linestyle="-", color="blue")
    plt.plot(ef_search_values, avg_formula_recalls, label="Formula Predicted Recalls", marker="s", linestyle="--",
             color="red")
    plt.xlabel("Ef_Search")
    plt.ylabel("Recall")
    plt.title("Recall Validation: Actual vs Formula")
    plt.legend()
    plt.grid(True)

    # Adjust x-axis and y-axis ticks
    plt.xticks(ticks=range(0, max(ef_search_values) + 1, 100))  # Set x-axis ticks every 100
    plt.yticks(ticks=np.arange(0, 1.01, 0.1))  # Set y-axis ticks from 0 to 1.0 every 0.1

    # Force y-axis to start from 0
    plt.ylim(0, 1.0)
    # Save the validation plot
    validation_plot_filename = "recall_validation_adjusted.png"
    plt.savefig(validation_plot_filename, dpi=300)
    plt.show()

    print(f"Validation plot saved to: {validation_plot_filename}")


def main():
    generate_query_dataset(num_queries=300, topk=5, output_file="query_dataset.json", zipf_param=0)
    initialize_dynamic_partition_tables()
    p = {}
    x = {}
    delta = {}

    hnsw_folder = os.path.join(project_root, "controller/dynamic_partition/hnsw")
    solution_path = os.path.join(hnsw_folder, "solution_0.txt")

    with open(solution_path, "r") as file:
        for line in file:
            if line.startswith("p["):
                # Parse p[j,k] values
                parts = line.strip().split(" = ")
                key = parts[0][2:-1]  # Extract "j,k"
                j, k = map(int, key.split(","))
                value = float(parts[1])
                p[(j, k)] = value
            elif line.startswith("x["):
                # Parse x[i,k] values
                parts = line.strip().split(" = ")
                key = parts[0][2:-1]  # Extract "i,k"
                i, k = map(int, key.split(","))
                value = float(parts[1])
                x[(i, k)] = value
            elif line.startswith("delta["):
                # Parse delta[i,j,k] values
                parts = line.strip().split(" = ")
                key = parts[0][6:-1]  # Extract "i,j,k"
                i, j, k = map(int, key.split(","))
                value = float(parts[1])
                delta[(i, j, k)] = value

    roles, documents, permissions, avg_blocks_per_document = fetch_initial_data()

    # Prepare background data
    role_to_documents, document_to_index = prepare_background_data(roles, documents, permissions)
    m = len(roles)  # Number of roles
    n = len(documents)  # Total number of documents
    c = m  # Number of partitions
    alpha = 8
    topk = 5
    ef_search = 10

    # Define the JSON file path
    json_file_path = os.path.join(hnsw_folder, "parameter_hnsw.json")

    result_data = {}
    # Check if the file already exists
    if os.path.exists(json_file_path):
        # If the file exists, read and return data
        with open(json_file_path, "r") as json_file:
            result_data = json.load(json_file)
            print("Data loaded from parameter_hnsw.json")

    # Load query dataset
    benchmark_folder = os.path.join(project_root, "basic_benchmark")
    query_dataset_path = os.path.join(benchmark_folder, "query_dataset.json")
    with open(query_dataset_path, "r") as f:
        query_dataset = json.load(f)

    # Define Ef_Search values to test
    ef_search_values = [1, 2, 3, 4, 5, 10, 30, 70, 100, 150, 200, 250, 300, 400, 800]
    # ef_search_values = [1, 2, 3, 4, 5, 10, 30, 70]
    validate_recall_model_with_avg(query_dataset, ef_search_values, topk, p, x, roles, role_to_documents, document_to_index,
                                   c, n, m, result_data["k"], result_data["beta"])
    # validate_recall_model_with_per(query_dataset, ef_search_values, topk, p, x, roles, role_to_documents,
    #                                document_to_index,
    #                                c, n, m, result_data["k"], result_data["beta"])


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
