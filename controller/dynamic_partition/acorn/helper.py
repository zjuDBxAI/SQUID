import json
import os
import sys

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(project_root)
from services.config import get_db_connection
from collections import defaultdict


def calculate_partition_count(p_values):
    used_partitions = {k for _, k in p_values.keys()}
    return len(used_partitions)

    # Remove empty partitions before entering the loop


def clean_empty_partitions(partition_assignment):
    """
    Remove all partitions that are empty.

    :param partition_assignment: Dictionary of partition assignments.
    :return: Cleaned partition assignment with non-empty sets only.
    """
    return {k: v for k, v in partition_assignment.items() if v}  # Keep only non-empty partitions


# Step: Reorganize partition_assignment and re-index partitions from 0 to c-1
def reorganize_partitions(partition_assignment):
    """
    Reassign partition numbers sequentially from 0 to c-1.
    """
    new_partition_assignment = {}
    partition_mapping = {}  # Track the mapping from old to new partition numbers

    # Assign new sequential IDs to each partition
    for new_id, old_id in enumerate(sorted(partition_assignment.keys())):
        new_partition_assignment[new_id] = partition_assignment[old_id]
        partition_mapping[old_id] = new_id

    return new_partition_assignment, partition_mapping



# Step 1: Retrieve initial data from the database
def fetch_initial_data():
    conn = get_db_connection()
    cur = conn.cursor()

    # Fetch roles
    cur.execute("SELECT role_id FROM Roles;")
    roles = [row[0] for row in cur.fetchall()]

    # Fetch document blocks (for document_id ordering)
    cur.execute("SELECT DISTINCT document_id FROM PermissionAssignment ORDER BY document_id;")
    # cur.execute("SELECT distinct document_id FROM documentblocks ORDER BY document_id;")
    documents = [row[0] for row in cur.fetchall()]

    # Fetch permissions (role_id -> document_id)
    cur.execute("SELECT role_id, document_id FROM PermissionAssignment;")
    permissions = cur.fetchall()

    # Fetch user-role mapping (user_id -> role_id)
    cur.execute("SELECT user_id, role_id FROM UserRoles;")
    user_roles = cur.fetchall()
    user_to_roles = {}
    for user_id, role_id in user_roles:
        if user_id not in user_to_roles:
            user_to_roles[user_id] = []
        user_to_roles[user_id].append(role_id)

    # Calculate the average number of blocks per document
    cur.execute("SELECT document_id, COUNT(block_id) FROM documentblocks GROUP BY document_id")
    document_block_counts = [row[1] for row in cur.fetchall()]
    avg_blocks_per_document = sum(document_block_counts) / len(document_block_counts) if document_block_counts else 0

    conn.close()
    return roles, documents, permissions, avg_blocks_per_document, user_to_roles

# Prepare background data to prevent repeated calculations
def prepare_background_data(roles, documents, permissions):
    valid_document_ids = {doc_id for _, doc_id in permissions}
    # Construct a mapping of role_id to required document_ids
    role_to_documents = {role: set() for role in roles}
    for role_id, document_id in permissions:
        if role_id in role_to_documents:
            role_to_documents[role_id].add(document_id)

    # Construct a mapping of document_id to its index for fast lookup
    document_to_index = {doc_id: idx for idx, doc_id in enumerate(documents) if doc_id in valid_document_ids}

    return role_to_documents, document_to_index


# Function to compute x_{i,k} based on p_{j,k}, optimizing to minimize the number of partitions accessed
def compute_role_partition_access(roles, documents, role_to_documents, p_values, c):
    # Initialize x_{i,k} dictionary
    x_values = {}

    # If p_values are not yet available (e.g., initial phase), return the default x_values as zeros
    if not p_values or any(value is None for value in p_values.values()):
        for i in range(len(roles)):
            for k in range(c):
                x_values[(i, k)] = 0
        return x_values

    # Construct a mapping of partition to the list of documents it contains
    partition_to_docs = defaultdict(set)
    for (j, k), value in p_values.items():
        if value > 0.5:  # If document j is assigned to partition k
            doc_id = documents[j]
            partition_to_docs[k].add(doc_id)

    # Determine which partitions each role needs to access (minimize the number of partitions)
    for i, role in enumerate(roles):
        required_docs = role_to_documents[role]

        # Track partitions that can cover the required documents
        uncovered_docs = set(required_docs)
        selected_partitions = set()

        # Greedily choose partitions that cover the most uncovered documents
        while uncovered_docs:
            best_partition = None
            best_coverage = 0

            # Find the partition that covers the most uncovered documents
            for k in range(c):
                docs_in_partition = partition_to_docs.get(k, set())
                coverage = len(uncovered_docs.intersection(docs_in_partition))
                # Choose the partition with the most coverage; break ties by partition size
                if coverage > best_coverage or (coverage == best_coverage and len(docs_in_partition) < len(partition_to_docs.get(best_partition, set()))):
                    best_partition = k
                    best_coverage = coverage

            if best_partition is None:
                print(f"Uncovered Documents: {uncovered_docs}")
                raise ValueError("No partition found to cover the required documents, check input consistency.")
            # Add the selected partition to the result and remove covered documents
            selected_partitions.add(best_partition)
            covered_docs = uncovered_docs.intersection(partition_to_docs[best_partition])
            uncovered_docs -= covered_docs

        # Update the x_{i,k} dictionary with the selected partitions
        for k in selected_partitions:
            x_values[(i, k)] = 1

    # For all other x[i,k] not selected, set them to 0 explicitly
    for i in range(len(roles)):
        for k in range(c):
            if (i, k) not in x_values:
                x_values[(i, k)] = 0

    return x_values


def calculate_hnsw_recall(ef_search, topk, p, x, roles, role_to_documents, document_to_index, c, n, m, k=1,
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
    role_recalls = []  # To store recall for each role

    # Step 1: Calculate recall for each role
    for i in range(m):
        documents = role_to_documents[roles[i]]
        role_sel_sum = 0  # Sel sum for the current role
        doc_indices = [document_to_index[doc] for doc in documents]
        sum_partition_count = 0
        sum_required_count = 0
        # Iterate over all partitions for this role
        for k in range(c):
            if x.get((i, k), 0):  # If partition k is required by this role
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
        # Step 2: Use the role's sel sum to calculate recall for this role
        if role_sel_sum == 0:
            role_recalls.append(0)  # No sel means recall is 0 for this role
            continue

        role_sel_whole = sum_required_count / sum_partition_count

        threshold = k * topk / role_sel_whole
        if ef_search <= threshold:
            recall = ef_search * role_sel_whole / topk
        else:
            exponent = -4 * beta * role_sel_whole / topk * (ef_search - threshold)
            recall = 1 / (1 + np.exp(exponent)) + (k - 0.5)

        role_recalls.append(min(recall, 1))

    # Step 3: Calculate the average recall across all roles
    avg_recall = np.mean(role_recalls) if role_recalls else 0

    return avg_recall


def calculate_hnsw_role_avg_qps(p, x, roles, role_to_documents, avg_blocks_per_document, c, n, m, ef_search,
                                constant_join_time,
                                a=550.97, b=183157):
    """
    Calculate query time based on the given formula for each role, and return the average query time.

    :param p: Dictionary with keys (j, k) and values for p[j,k] (whether document j is in partition k).
    :param x: Dictionary with keys (i, k) and values for x[i,k] (whether partition k is required by role i).
    :param role_to_documents: Dictionary mapping each role to its required documents.
    :param c: Number of partitions.
    :param n: Total number of documents.
    :param ef_search: Current ef_search value.
    :param constant_join_time: Hash join constant.
    :param a: Coefficient a (default 550.97).
    :param b: Coefficient b (default 183157).
    :return: The average query time across all roles.
    """
    role_query_times = []  # To store query time for each role

    # Step 1: Calculate query time for each role
    for i in range(m):
        documents = role_to_documents[roles[i]]
        role_query_time = 0  # Query time for the current role
        role_partition_count = 0  # Number of partitions relevant to this role

        # Iterate over all partitions relevant to this role
        for k in range(c):
            if x.get((i, k), 0):  # If partition k is required by this role
                role_partition_count += 1  # Count the partition for this role
                partition_total_document_count = sum(
                    p.get((j, k), 0) for j in range(n))  # Total documents in partition k
                if partition_total_document_count > 0:
                    # Add log(n_i) * (a * ef_search + b) to the role's query time
                    role_query_time += np.log(partition_total_document_count * avg_blocks_per_document) * (
                            a * ef_search + b)

        # Add the constant join time based on this role's partition count
        role_query_time += constant_join_time * role_partition_count

        # Append the role's query time to the list
        role_query_times.append(role_query_time)

    # Step 2: Calculate the average query time across all roles
    avg_query_time = np.mean(role_query_times) if role_query_times else 0

    return avg_query_time


def calculate_hnsw_user_avg_qps(p, x, roles, role_to_documents, avg_blocks_per_document, c, n, m, ef_search,
                                constant_join_time, a=550.97, b=183157):
    """
    Calculate the average query time for a single ef_search value.

    Parameters:
        p (dict): Document-to-partition mapping.
        x (dict): Role-to-partition mapping.
        roles (list): List of roles.
        role_to_documents (dict): Mapping of roles to their required documents.
        avg_blocks_per_document (float): Average number of blocks per document.
        c (int): Number of partitions.
        n (int): Total number of documents.
        m (int): Number of roles.
        ef_search (int): The ef_search value to evaluate.
        constant_join_time (float): The constant time to join partitions.
        a (float): Query time coefficient.
        b (float): Query time coefficient.

    Returns:
        float: The average query time for the given ef_search.
    """
    import os
    import json
    import numpy as np
    from controller.dynamic_partition.hnsw.validate.modelqps_vs_realqps import \
        calculate_hnsw_qps_by_user_with_ef_searches

    # Load query dataset
    benchmark_folder = os.path.join(project_root, "basic_benchmark")
    query_dataset_path = os.path.join(benchmark_folder, "query_dataset.json")
    with open(query_dataset_path, "r") as f:
        query_dataset = json.load(f)

    # Store actual and formula-predicted query times
    formula_query_times = []

    # Run experiments to get query times
    for query in query_dataset:
        user_id = query["user_id"]
        query_vector = query["query_vector"]

        # Execute three repetitions and collect query times
        for repetition in range(1):  # You can adjust the repetition count
            formula_query_results = calculate_hnsw_qps_by_user_with_ef_searches(
                user_id, p, x, role_to_documents, avg_blocks_per_document, roles, c, n, [ef_search],
                constant_join_time, a, b
            )
            # Append the query time for this ef_search
            query_time = formula_query_results.get(ef_search, [0])[0]  # Use 0 as a fallback for missing values
            formula_query_times.append(query_time)

    # Calculate the average formula query time for the given ef_search
    avg_formula_query_time = np.mean(formula_query_times) if formula_query_times else 0

    print(f"Formula Query Time for ef_search={ef_search}: {avg_formula_query_time}")

    return avg_formula_query_time


def save_solution_to_file(p, x, delta, file_name="solution.txt"):
    """
    Saves p, x, and delta values into a text file in the required format.
    Prints the objective value to the console.
    """
    # Calculate the objective value
    # Save to file
    with open(file_name, "w") as file:
        # Save p values
        for (j, k), value in p.items():
            file.write(f"p[{j},{k}] = {value}\n")

        # Save x values
        for (i, k), value in x.items():
            file.write(f"x[{i},{k}] = {value}\n")

        # Save delta values (if any)
        for (i, j, k), value in delta.items():
            file.write(f"delta[{i},{j},{k}] = {value}\n")

    print(f"Solution saved to {file_name}.")


def delete_faiss_files(project_root):
    """
    Deletes all .faiss files in the 'index_file/dynamic_partition' folder under 'acorn_benchmark'.

    Args:
        project_root (str): The root directory of the project.
    """
    # Define the target folder path
    dynamic_partition_folder = os.path.join(project_root, "acorn_benchmark", "index_file", "dynamic_partition")

    # Check if the folder exists
    if not os.path.exists(dynamic_partition_folder):
        print(f"Folder does not exist: {dynamic_partition_folder}")
        return

    # Iterate through files and delete .faiss files
    for filename in os.listdir(dynamic_partition_folder):
        if filename.endswith(".faiss"):  # Only target .faiss files
            file_path = os.path.join(dynamic_partition_folder, filename)
            try:
                os.remove(file_path)
                print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

