import json
import math

import numpy as np
import os
import sys
import time

from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

plt.rcParams.update({"font.size": 22})

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)

from services.logger import get_logger

logger = get_logger(__name__)
logger.info("project_root set to %s", project_root)
from controller.dynamic_partition.hnsw.analysis.analysis_hnsw_recall import search_documents_rls_for_join_time_analysis, \
    search_documents_rls_for_analysis_with_execution_time

from psycopg2 import sql
from services.config import get_db_connection


def _safe_float(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def save_query_time_plot_data(ef_search_values, avg_query_times, avg_total_rows, fitted_query_times, filename):
    plot_data = []

    for ef_search, actual_time, total_rows, predicted_time in zip(
        ef_search_values, avg_query_times, avg_total_rows, fitted_query_times
    ):
        actual_ms = _safe_float(actual_time / 1_000_000.0) if actual_time else None
        predicted_ms = _safe_float(predicted_time / 1_000_000.0) if predicted_time else None
        plot_data.append(
            {
                "ef_search": int(ef_search),
                "avg_query_time": _safe_float(actual_time),
                "predicted_query_time": _safe_float(predicted_time),
                "avg_total_rows": _safe_float(total_rows),
                "avg_query_time_ms": actual_ms,
                "predicted_query_time_ms": predicted_ms,
            }
        )

    with open(filename, "w", encoding="utf-8") as outfile:
        json.dump(plot_data, outfile, indent=2)
    logger.info("Saved query time plot data to %s", filename)


def search_documents_role_partition_analysis(user_id, query_vector, topk=5, ef_searchs=None):
    """
    Search documents with role partition statistics for multiple ef_search values.
    Computes SQL run time for each ef_search value.
    :param user_id: User ID for which to execute the query.
    :param query_vector: Query vector for similarity search.
    :param topk: Number of top results to retrieve.
    :param ef_searchs: List of ef_search values to evaluate.
    :return: A dictionary mapping each ef_search to its (query time, total rows).
    """
    results = {}
    conn = get_db_connection()  # Get the database connection
    cur = conn.cursor()
    cur.execute(f"SET max_parallel_workers_per_gather = 0;")

    # Query 1: Get the role IDs for the user
    cur.execute("SELECT role_id FROM userroles WHERE user_id = %s", [user_id])
    role_ids = cur.fetchall()

    if not role_ids:
        logger.info("No roles found for user_id %s.", user_id)
        cur.close()
        conn.close()
        return {ef_search: (0, 0) for ef_search in ef_searchs}

    role_ids = [row[0] for row in role_ids]
    table_entries = []
    for role_id in role_ids:
        table_identifier = sql.Identifier(f"documentblocks_role_{role_id}")
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(table_identifier))
        row_count = cur.fetchone()[0]
        table_entries.append((table_identifier, row_count))

    # Process each ef_search value
    for ef_search in ef_searchs:
        cur.execute(f"SET hnsw.ef_search = {ef_search};")  # Dynamically set ef_search

        aggregate_query_time = 0
        aggregate_rows = 0

        # Execute EXPLAIN ANALYZE to time the query
        explain_query = sql.SQL(
            """
            EXPLAIN (ANALYZE, VERBOSE)
            SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
            FROM {}
            ORDER BY distance
            LIMIT %s
            """
        )

        for table_identifier, row_count in table_entries:
            if row_count <= 0:
                continue

            cur.execute(explain_query.format(table_identifier), [query_vector, topk])
            explain_plan = cur.fetchall()

            # Parse query time from EXPLAIN ANALYZE
            table_query_time = 0
            for row in explain_plan:
                if "Execution Time" in row[0]:
                    query_time = float(row[0].split()[-2]) * 1000 * 1000  # Convert to nanoseconds
                    table_query_time += query_time

            aggregate_query_time += table_query_time
            aggregate_rows += row_count

        results[ef_search] = (aggregate_query_time, aggregate_rows)

    cur.close()
    conn.close()

    return results


def search_documents_brute_force_for_analysis_with_execution_time(user_id, query_vector, topk, ef_search_values):
    """
    Search documents with role-level security for multiple ef_search values using EXPLAIN ANALYZE.
    Computes SQL execution time for each ef_search value.

    :param user_id: User ID for which to execute the query.
    :param query_vector: Query vector for similarity search.
    :param topk: Number of top results to retrieve.
    :param ef_search_values: List of ef_search values to evaluate.
    :return: A dictionary mapping each ef_search to its (query time, total rows).
    """
    results = {}
    conn = get_db_connection()  # Reuse the same connection for this user
    cur = conn.cursor()

    try:
        # Disable parallelism for consistent timing
        cur.execute(f"SET max_parallel_workers_per_gather = 0;")
        # Query for the role's document blocks
        table_name = sql.Identifier("documentblocks")

        # Count total rows in the table
        cur.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(table_name))
        n_total_rows = cur.fetchone()[0]  # Total rows for the first role

        # Process each ef_search value
        for ef_search in ef_search_values:
            cur.execute(f"SET LOCAL hnsw.ef_search = {ef_search};")  # Dynamically set ef_search

            # Execute EXPLAIN ANALYZE to time the query
            explain_query = sql.SQL(
                """
                EXPLAIN (ANALYZE, VERBOSE)
                SELECT block_id, document_id, block_content, vector <-> %s::vector AS distance
                FROM {}
                ORDER BY distance
                LIMIT %s
                """
            ).format(table_name)

            cur.execute(explain_query, [query_vector, topk])
            explain_plan = cur.fetchall()

            # Parse query time and join time from EXPLAIN ANALYZE
            total_adjusted_time = 0  # Initialize cumulative adjusted query time

            for row in explain_plan:
                line = row[0].strip()
                # Parse overall execution time
                if "Execution Time" in line:
                    query_time = float(line.split()[-2]) * 1000 * 1000  # Convert ms to ns
                    total_adjusted_time += query_time

            # Store results for the current ef_search
            results[ef_search] = (total_adjusted_time, n_total_rows)

    finally:
        cur.close()
        conn.close()

    return results


def run_experiment_on_ef_search(queries, ef_search_values=None, repetitions=2):
    """
    Run experiments for a given query dataset.
    Only include the final repetition's time in calculations.
    Calculate the average k value for all queries.
    """
    if ef_search_values is None:
        ef_search_values = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    logger.info(
        "Running ef_search experiment query_count=%d, values=%s",
        len(queries),
        ef_search_values,
    )
    start = time.perf_counter()
    ef_search_results = {ef: [] for ef in ef_search_values}  # Store k values grouped by ef_search
    actual_query_times = {ef: [] for ef in ef_search_values}
    total_rows_by_ef_search = {ef: [] for ef in ef_search_values}  # Store total rows per ef_search

    for ef_search in ef_search_values:
        logger.info("Processing ef_search=%d across all queries", ef_search)
        for query in queries:
            user_id = query["user_id"]
            query_vector = query["query_vector"]

            for repetition in range(repetitions):
                query_results = search_documents_role_partition_analysis(
                    user_id,
                    query_vector,
                    topk=5,
                    ef_searchs=[ef_search],
                )
                query_time, n_total_rows = query_results.get(ef_search, (0, 0))

                if repetition == repetitions - 1 and n_total_rows > 0:
                    actual_query_times[ef_search].append(query_time)
                    total_rows_by_ef_search[ef_search].append(n_total_rows)
                    k = query_time / np.log(n_total_rows)
                    ef_search_results[ef_search].append(k)

    # Step 2: Compute the average k value for each ef_search after all queries
    results = []
    for ef_search in ef_search_values:
        avg_k = np.mean(ef_search_results[ef_search]) if ef_search_results[ef_search] else 0
        avg_query_time = np.mean(actual_query_times[ef_search]) if actual_query_times[ef_search] else 0
        avg_total_rows = np.mean(total_rows_by_ef_search[ef_search]) if total_rows_by_ef_search[ef_search] else 0
        results.append({
            "ef_search": ef_search,
            "avg_k": avg_k,
            "k_values": ef_search_results[ef_search],
            "avg_query_time": avg_query_time,
            "query_times": actual_query_times[ef_search],
            "avg_total_rows": avg_total_rows,
            "total_rows": total_rows_by_ef_search[ef_search]
        })
    elapsed = time.perf_counter() - start
    logger.info("ef_search experiment completed in %.2f min", elapsed / 60)
    return results


def fit_query_time_function_with_log(results):
    """
    Fit f(ef_search) to the average query_time values using a linear function,
    incorporating log(n) into the calculation.

    Parameters:
    - results: List of dictionaries containing "ef_search" and "avg_query_time".
    - logn_values: List of log(n) values corresponding to the query dataset.

    Returns:
    - params: Fitted parameters for the linear model (a, b).
    """
    ef_search_values = np.array([result["ef_search"] for result in results])
    avg_query_times = np.array([result["avg_query_time"] for result in results])
    avg_total_rows = np.array([result["avg_total_rows"] for result in results])

    # Incorporate log(n) into the average query time
    logn_values = np.log(avg_total_rows)
    normalized_query_times = avg_query_times / logn_values

    # Define the linear function: f(x) = a * x + b
    def func(x, a, b):
        return a * x + b

    # Fit the curve
    initial_params = [1, 1]  # Initial guesses for a, b
    params, _ = curve_fit(func, ef_search_values, normalized_query_times, p0=initial_params)

    # Generate fitted values
    fitted_query_times = func(ef_search_values, *params) * logn_values

    # Plot the original data and fitted curve
    avg_query_times_ms = avg_query_times / 1_000_000.0
    fitted_query_times_ms = fitted_query_times / 1_000_000.0

    plt.figure(figsize=(8, 6), dpi=600)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)
    plt.plot(
        ef_search_values,
        fitted_query_times_ms,
        color="#d1495b",
        linewidth=2.5,
        label=f"Model Fit\n$a={params[0]:.2f}, b={params[1]:.2f}$",
        zorder=2,
    )
    plt.scatter(
        ef_search_values,
        avg_query_times_ms,
        color="#1d3557",
        marker="o",
        s=140,
        edgecolors="white",
        linewidths=1.2,
        label="Measured",
        zorder=3,
    )
    plt.xlabel("ef_search", fontsize=28, fontweight="normal")
    plt.ylabel("Query Time (ms)", fontsize=28, fontweight="normal")
    plt.xticks(ef_search_values, fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=22, loc="upper left")
    plt.tight_layout()
    output_dir = os.path.dirname(os.path.abspath(__file__))
    plot_filename = "query_time_analysis.pdf"
    plot_path = os.path.join(output_dir, plot_filename)
    plt.savefig(plot_path, dpi=600, bbox_inches="tight")
    plt.close()

    data_filename = "query_time_analysis_data.json"
    data_path = os.path.join(output_dir, data_filename)
    save_query_time_plot_data(ef_search_values, avg_query_times, avg_total_rows, fitted_query_times, data_path)

    logger.info("Fitted parameters: a=%.2f, b=%.2f", params[0], params[1])
    return params

def fit_ef_search_function_linear(results):
    """
    Fit f(ef_search) to the average k values using a linear function.
    """
    ef_search_values = np.array([result["ef_search"] for result in results])
    avg_k_values = np.array([result["avg_k"] for result in results])

    # Define the linear function: f(x) = a * x + b
    def func(x, a, b):
        return a * x + b

    # Fit the curve
    initial_params = [1, 1]  # Initial guesses for a, b
    params, _ = curve_fit(func, ef_search_values, avg_k_values, p0=initial_params)

    # Generate fitted values
    fitted_k_values = func(ef_search_values, *params)

    # Plot the original data and fitted curve
    plt.figure(figsize=(10, 6))
    plt.scatter(ef_search_values, avg_k_values, label="Data", color="blue")
    plt.plot(ef_search_values, fitted_k_values, label=f"Fitted Curve: a * x + b\na={params[0]:.2f}, b={params[1]:.2f}",
             color="red")
    plt.xlabel("ef_search")
    plt.ylabel("Average k Value")
    plt.title("Fitting Average k as a Function of ef_search (Linear Model)")
    plt.legend()
    plt.grid(True)
    plot_filename = f'qps_analysis.pdf'
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.show()

    logger.info("Fitted parameters: a=%.2f, b=%.2f", params[0], params[1])
    return params


def run_experiment_on_join_time(queries):
    """
    Run experiments across the entire query dataset to calculate the average join time.
    Use the third repetition's join time in calculations.
    """
    logger.info("Running join-time experiment for %d queries", len(queries))
    start = time.perf_counter()
    results = []  # To store results for each query's join time

    for query in queries:
        user_id = query["user_id"]
        query_vector = query["query_vector"]

        join_times = []  # Store join times for three repetitions

        # Execute three repetitions
        for repetition in range(3):
            join_time = search_documents_rls_for_join_time_analysis(
                user_id, query_vector
            )
            join_times.append(join_time)

        # Only use the third repetition's join time
        if len(join_times) >= 3:
            third_join_time = join_times[2]  # Take the third repetition
            results.append(third_join_time)

    # Calculate the average join time across all queries
    avg_join_time = np.mean(results) if results else 0
    elapsed = time.perf_counter() - start
    logger.info(
        "Join-time experiment completed in %.2f min (avg=%.2f)",
        elapsed / 60,
        avg_join_time,
    )

    return avg_join_time


def get_hnsw_qps_parameters():
    import json

    # Load generated queries
    benchmark_folder = os.path.join(project_root, "basic_benchmark")
    query_dataset_path = os.path.join(benchmark_folder, "query_dataset.json")
    logger.info("Loading QPS query dataset from %s", query_dataset_path)
    with open(query_dataset_path, "r") as infile:
        queries = json.load(infile)

    ef_search_values = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    # Run experiments
    logger.info("Starting QPS ef_search experiment")
    results = run_experiment_on_ef_search(queries, ef_search_values)

    # Fit the function to the results using a linear model
    # fitted_params = fit_ef_search_function_linear(results)
    logger.info("Fitting QPS model")
    fit_start = time.perf_counter()
    fitted_params = fit_query_time_function_with_log(results)
    fit_elapsed = time.perf_counter() - fit_start
    logger.info("QPS model fit completed in %.2fs", fit_elapsed)
    join_times = run_experiment_on_join_time(queries)
    # Print fitted parameters
    logger.info(
        "Fitted Function Parameters: a=%.2f, b=%.2f",
        fitted_params[0],
        fitted_params[1],
    )
    return fitted_params, join_times


if __name__ == "__main__":
    get_hnsw_qps_parameters()
