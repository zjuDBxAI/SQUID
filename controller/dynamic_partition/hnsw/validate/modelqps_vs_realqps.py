import sys

import numpy as np
import matplotlib.pyplot as plt
import json
import os

from psycopg2 import sql

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)
print(project_root)
from controller.dynamic_partition.hnsw.helper import calculate_hnsw_role_avg_qps, fetch_initial_data, \
    prepare_background_data, calculate_hnsw_user_avg_qps
from controller.dynamic_partition.search import dynamic_partition_search_statistics_sql

from services.read_dataset_function import generate_query_dataset_with_roles_and_repetitions, \
    generate_query_dataset_for_cache

# Row-level security imports
from controller.baseline.pg_row_security.row_level_security import (
    disable_row_level_security, drop_database_users, create_database_users, enable_row_level_security,
    get_db_connection_for_many_users
)

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


def dynamic_partition_search_analysis(user_id, query_vector, topk=5, ef_searchs=None):
    """
    Search using SQL query execution time for performance statistics with RLS.
    Supports multiple ef_search values in a single function call.
    """
    # conn = get_db_connection()
    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()
    cur.execute(f"SET jit = off;")
    n_total_rows = 0
    results = {}

    for ef_search in ef_searchs:
        cur.execute(f"SET hnsw.ef_search = {ef_search};")
        total_query_time = 0

        # Step 1: Check if the user has access to role_id = ?
        cur.execute("""
            SELECT ur.role_id 
            FROM UserRoles ur 
            WHERE ur.user_id = %s AND ur.role_id = 11;
        """, [user_id])
        has_role = cur.fetchone() is not None

        # Step 1: Fetch accessible partitions using EXPLAIN ANALYZE
        explain_query = """
            EXPLAIN (ANALYZE, VERBOSE)
            SELECT partition_id FROM RolePartitions rp
            JOIN UserRoles ur ON rp.role_id = ur.role_id
            WHERE ur.user_id = %s;
        """
        cur.execute(explain_query, [user_id])
        explain_plan = cur.fetchall()

        # Parse and accumulate execution time
        for row in explain_plan:
            if "Execution Time" in row[0]:
                query_time = float(row[0].split()[-2]) * 1000 * 1000  # Convert ms to seconds
                # total_query_time += query_time

        # Fetch partition IDs
        cur.execute("""
            SELECT partition_id FROM RolePartitions rp
            JOIN UserRoles ur ON rp.role_id = ur.role_id
            WHERE ur.user_id = %s;
        """, [user_id])
        accessible_partitions = {row[0] for row in cur.fetchall()}

        # Step 2: Search each partition with EXPLAIN ANALYZE
        for partition_id in accessible_partitions:
            partition_table = sql.Identifier(f"documentblocks_partition_{partition_id}")
            cur.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(partition_table))
            n_total_rows += cur.fetchone()[0]

            explain_query = sql.SQL(
                """
                EXPLAIN (ANALYZE,VERBOSE,BUFFERS)
                SELECT block_id, document_id, block_content, 
                       vector <-> %s::vector AS distance
                FROM {}
                ORDER BY distance
                LIMIT %s;
                """
            ).format(partition_table)

            # Capture query plan and execution time
            cur.execute(explain_query, [query_vector, topk])
            explain_plan = cur.fetchall()

            for row in explain_plan:
                line = row[0].strip()  # Clean each row

                # Check for the overall execution time
                if "Execution Time" in line:
                    execution_time = float(line.split()[-2]) * 1000 * 1000  # Convert ms to nanoseconds
                    total_query_time += execution_time
        results[ef_search] = (total_query_time, n_total_rows)

    cur.close()
    conn.close()

    return results


def calculate_hnsw_qps_by_user_with_ef_searches(user_id, p, x, role_to_documents, avg_blocks_per_document, roles, c, n,
                                                ef_search_values,
                                                constant_join_time, a=550.97, b=183157):
    """
    Calculate query time for a specific user for multiple ef_search values.

    :param user_id: The user ID for which query time is calculated.
    :param p: Dictionary with keys (j, k) and values for p[j,k] (whether document j is in partition k).
    :param x: Dictionary with keys (i, k) and values for x[i,k] (whether partition k is required by role i).
    :param role_to_documents: Dictionary mapping each role to its required documents.
    :param c: Number of partitions.
    :param n: Total number of documents.
    :param ef_search_values: List of ef_search values.
    :param constant_join_time: Hash join constant.
    :param a: Coefficient a (default 550.97).
    :param b: Coefficient b (default 183157).
    :return: Dictionary with ef_search as key and (total_query_time, n_total_rows) as values.
    """
    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()

    # Fetch roles associated with the user
    cur.execute("SELECT role_id FROM UserRoles WHERE user_id = %s;", [user_id])
    user_roles = [row[0] for row in cur.fetchall()]

    results = {}

    # Loop over ef_search values
    for ef_search in ef_search_values:
        total_query_time = 0  # Accumulate query time for all roles
        n_total_rows = 0  # Total number of documents accessed

        for role_id in user_roles:
            if role_id not in role_to_documents:
                continue  # Skip roles without document mappings

            documents = role_to_documents[role_id]
            role_query_time = 0
            role_partition_count = 0

            # Iterate over all partitions relevant to this role
            for k in range(c):
                if x.get((role_id - 1, k), 0):  # If partition k is required by this role
                    role_partition_count += 1
                    partition_total_document_count = sum(
                        p.get((j, k), 0) for j in range(n))  # Total documents in partition k
                    n_total_rows += partition_total_document_count  # Add to total row count

                    if partition_total_document_count > 0:
                        # Add log(n_i) * (a * ef_search + b) to the role's query time
                        role_query_time += np.log(partition_total_document_count * avg_blocks_per_document) * (
                                a * ef_search + b)
            # Add the constant join time based on this role's partition count
            role_query_time += constant_join_time * role_partition_count
            total_query_time += role_query_time  # Accumulate for all roles
        # Store results for the current ef_search
        results[ef_search] = (total_query_time, n_total_rows)

    cur.close()
    conn.close()

    return results


def calculate_hnsw_qps_by_user_with_ef_searches_by_tables(user_id, role_to_documents, c, n, ef_search_values,
                                                          constant_join_time, a=550.97, b=183157):
    """
    Calculate query time for a specific user for multiple ef_search values using table-based operations.

    :param user_id: The user ID for which query time is calculated.
    :param role_to_documents: Dictionary mapping each role to its required documents.
    :param c: Number of partitions.
    :param n: Total number of documents.
    :param ef_search_values: List of ef_search values.
    :param constant_join_time: Hash join constant.
    :param a: Coefficient a (default 550.97).
    :param b: Coefficient b (default 183157).
    :return: Dictionary with ef_search as key and (total_query_time, n_total_rows) as values.
    """
    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()

    # Fetch roles associated with the user
    cur.execute("SELECT role_id FROM UserRoles WHERE user_id = %s;", [user_id])
    user_roles = [row[0] for row in cur.fetchall()]

    results = {}

    # Process each ef_search value
    for ef_search in ef_search_values:
        total_query_time = 0
        n_total_rows = 0

        cur.execute(f"SET hnsw.ef_search = {ef_search};")

        for role_id in user_roles:
            if role_id not in role_to_documents:
                continue  # Skip roles without document mappings

            # Query RolePartitions for accessible partitions
            cur.execute("""
                SELECT partition_id 
                FROM RolePartitions 
                WHERE role_id = %s;
            """, [role_id])
            accessible_partitions = {row[0] for row in cur.fetchall()}

            # Process each accessible partition
            for partition_id in accessible_partitions:
                partition_table = sql.Identifier(f"documentblocks_partition_{partition_id}")

                # Fetch total rows in this partition
                cur.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(partition_table))
                partition_total_document_count = cur.fetchone()[0]
                n_total_rows += partition_total_document_count

                if partition_total_document_count > 0:
                    # Calculate role query time for this partition
                    total_query_time += np.log(partition_total_document_count) * (a * ef_search + b)

                # Add join time for the partition
                total_query_time += constant_join_time * len(accessible_partitions)

                # Assume we only need one partition per role for simplicity
        # Store results for the current ef_search
        results[ef_search] = (total_query_time, n_total_rows)

    cur.close()
    conn.close()

    return results


def validate_query_time_model(query_dataset, ef_search_values, p, x, roles, role_to_documents, avg_blocks_per_document,
                              c, n, m, join_times,
                              a=550.97, b=183157):
    """
    Validate the relationship between ef_search and query time using experiment results and formula predictions.

    Parameters:
    - query_dataset: List of query data with user_id, query_vector, etc.
    - ef_search_values: List of ef_search values for validation.
    - p, x, roles, role_to_documents, c, n, m: Parameters required for calculate_hnsw_qps.
    - join_times: Join times constant used in the formula.
    - a, b: Coefficients for formula calculation.
    """
    # Store actual and formula-predicted query times
    actual_query_times = {ef_search: [] for ef_search in ef_search_values}
    formula_query_times = {ef_search: [] for ef_search in ef_search_values}

    # Run experiments to get actual query times
    for query in query_dataset:
        user_id = query["user_id"]
        query_vector = query["query_vector"]
        topk = query["topk"]
        # Execute three repetitions and collect query times
        for repetition in range(3):
            query_results = dynamic_partition_search_analysis(user_id, query_vector, topk=topk,
                                                              ef_searchs=ef_search_values)

            if repetition == 2:  # Only use the third repetition's time
                formula_query_results = calculate_hnsw_qps_by_user_with_ef_searches(user_id, p, x, role_to_documents,
                                                                                    avg_blocks_per_document,
                                                                                    roles, c,
                                                                                    n, ef_search_values, join_times,
                                                                                    a, b)
                # formula_query_results = calculate_hnsw_qps_by_user_with_ef_searches_by_tables(user_id,
                #                                                                               role_to_documents, c, n,
                #                                                                               ef_search_values,
                #                                                                               join_times, a, b)
                for ef_search, query_time in query_results.items():
                    actual_query_times[ef_search].append(query_time[0])

                for ef_search, query_time in formula_query_results.items():
                    formula_query_times[ef_search].append(query_time[0])

    # Calculate the average actual query time for each ef_search
    avg_actual_query_times = [np.mean(actual_query_times[ef_search]) if actual_query_times[ef_search] else 0
                              for ef_search in ef_search_values]
    avg_formula_query_times = [np.mean(formula_query_times[ef_search]) if formula_query_times[ef_search] else 0
                               for ef_search in ef_search_values]
    # Calculate query times using the formula for each ef_search
    # for ef_search in ef_search_values:
    #     formula_query_time = calculate_hnsw_qps(p, x, roles, role_to_documents, c, n, m, ef_search, join_times, a, b)
    #     formula_query_times.append(formula_query_time)

    print(f"Actual Query Times: {avg_actual_query_times}")
    print(f"Formula Query Times: {avg_formula_query_times}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(ef_search_values, avg_actual_query_times, label="Actual Query Times", marker="o", linestyle="-",
             color="blue")
    plt.plot(ef_search_values, avg_formula_query_times, label="Formula Predicted Query Times", marker="s",
             linestyle="--",
             color="red")
    plt.xlabel("Ef_Search")
    plt.ylabel("Query Time (ns)")
    plt.title("Query Time Validation: Actual vs Formula")
    plt.legend()
    plt.grid(True)

    # Save the validation plot
    validation_plot_filename = "query_time_validation.png"
    plt.savefig(validation_plot_filename, dpi=300)
    plt.show()

    print(f"Validation plot saved to: {validation_plot_filename}")


def validate_query_time_model_avg(query_dataset, ef_search_values, p, x, roles, role_to_documents,
                                  avg_blocks_per_document,
                                  c, n, m, join_times,
                                  a=550.97, b=183157):
    """
    Validate the relationship between ef_search and query time using experiment results and formula predictions.

    Parameters:
    - query_dataset: List of query data with user_id, query_vector, etc.
    - ef_search_values: List of ef_search values for validation.
    - p, x, roles, role_to_documents, c, n, m: Parameters required for calculate_hnsw_qps.
    - join_times: Join times constant used in the formula.
    - a, b: Coefficients for formula calculation.
    """
    # Store actual and formula-predicted query times
    actual_query_times = {ef_search: [] for ef_search in ef_search_values}
    formula_query_times = {ef_search: [] for ef_search in ef_search_values}

    # Run experiments to get actual query times
    for query in query_dataset:
        user_id = query["user_id"]
        query_vector = query["query_vector"]

        # Execute three repetitions and collect query times
        for repetition in range(3):
            query_results = dynamic_partition_search_analysis(user_id, query_vector, topk=5,
                                                              ef_searchs=ef_search_values)

            if repetition == 1:  # Only use the third repetition's time
                for ef_search, query_time in query_results.items():
                    actual_query_times[ef_search].append(query_time[0])
                break

    # Calculate the average actual query time for each ef_search
    avg_actual_query_times = [np.mean(actual_query_times[ef_search]) if actual_query_times[ef_search] else 0
                              for ef_search in ef_search_values]
    # Calculate query times using the formula for each ef_search
    for ef_search in ef_search_values:
        formula_query_time = calculate_hnsw_user_avg_qps(p, x, roles, role_to_documents, c, n, m, ef_search, join_times, a, b)
        formula_query_times.append(formula_query_time)

    print(f"Actual Query Times: {avg_actual_query_times}")
    print(f"Formula Query Times: {formula_query_times}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.plot(ef_search_values, avg_actual_query_times, label="Actual Query Times", marker="o", linestyle="-",
             color="blue")
    plt.plot(ef_search_values, formula_query_times, label="Formula Predicted Query Times", marker="s",
             linestyle="--",
             color="red")
    plt.xlabel("Ef_Search")
    plt.ylabel("Query Time (ns)")
    plt.title("Query Time Validation: Actual vs Formula")
    plt.legend()
    plt.grid(True)

    # Save the validation plot
    validation_plot_filename = "query_time_validation.png"
    plt.savefig(validation_plot_filename, dpi=300)
    plt.show()

    print(f"Validation plot saved to: {validation_plot_filename}")


def main():
    generate_query_dataset_for_cache(num_queries=100, topk=5, output_file="query_dataset.json", zipf_param=0)
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

    roles, documents, permissions, avg_blocks_per_document, user_to_roles = fetch_initial_data()

    # Prepare background data
    role_to_documents, document_to_index = prepare_background_data(roles, documents, permissions)
    m = len(roles)  # Number of roles
    n = len(documents)  # Total number of documents
    c = m  # Number of partitions
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
    ef_search_values = [5, 20, 40, 80, 120, 200, 300, 400]

    validate_query_time_model(query_dataset, ef_search_values, p, x, roles, role_to_documents, avg_blocks_per_document,
                              c, n, m,
                              result_data["join_times"], result_data["a"], result_data["b"])


# Run the main function when the script is executed
if __name__ == "__main__":
    main()
