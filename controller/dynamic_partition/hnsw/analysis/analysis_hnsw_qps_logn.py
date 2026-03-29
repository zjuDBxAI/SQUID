import os
import sys
import time

import numpy as np
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)

from services.logger import get_logger
from controller.dynamic_partition.hnsw.analysis.analysis_hnsw_qps import (
    run_experiment_on_ef_search,
    run_experiment_on_join_time,
)

logger = get_logger(__name__)
logger.info("project_root set to %s", project_root)


def fit_query_time_function_with_efs_and_logn(results):
    """
    Fit query_time = a * ef_search + b * log(n_total_rows).
    """
    ef_search_values = np.array([result["ef_search"] for result in results], dtype=float)
    avg_query_times = np.array([result["avg_query_time"] for result in results], dtype=float)
    avg_total_rows = np.array([result["avg_total_rows"] for result in results], dtype=float)

    valid_mask = avg_total_rows > 0
    if not np.any(valid_mask):
        logger.warning("No valid avg_total_rows values for fitting ef_search/log(n) model.")
        return np.array([0.0, 0.0])

    ef_search_values = ef_search_values[valid_mask]
    avg_query_times = avg_query_times[valid_mask]
    logn_values = np.log(avg_total_rows[valid_mask])

    def func(data, a, b):
        ef, logn = data
        return a * ef + b * logn

    initial_params = [1.0, 1.0]
    try:
        params, _ = curve_fit(func, (ef_search_values, logn_values), avg_query_times, p0=initial_params)
    except RuntimeError as exc:
        logger.error("Failed to fit ef_search/log(n) model: %s", exc)
        return np.array([0.0, 0.0])

    fitted_query_times = func((ef_search_values, logn_values), *params)
    sort_idx = np.argsort(ef_search_values)

    plt.figure(figsize=(10, 6))
    plt.scatter(ef_search_values, avg_query_times, label="Data (Average Query Times)", color="blue")
    plt.plot(
        ef_search_values[sort_idx],
        fitted_query_times[sort_idx],
        label=(
            "Fitted Curve: a * ef_search + b * log(n)\n"
            f"a={params[0]:.2f}, b={params[1]:.2f}"
        ),
        color="green",
    )
    plt.xlabel("ef_search")
    plt.ylabel("Query Time")
    plt.title("Fitting Query Time with ef_search and log(n) (Linear Model)")
    plt.legend()
    plt.grid(True)
    plot_filename = "query_time_analysis_efs_logn.pdf"
    plt.savefig(plot_filename, bbox_inches="tight")
    plt.show()

    logger.info(
        "Fitted parameters for query_time = a * ef_search + b * log(n): a=%.2f, b=%.2f",
        params[0],
        params[1],
    )
    return params


def get_hnsw_qps_parameters_with_logn():
    import json

    benchmark_folder = os.path.join(project_root, "basic_benchmark")
    query_dataset_path = os.path.join(benchmark_folder, "query_dataset.json")
    logger.info("Loading QPS query dataset from %s", query_dataset_path)
    with open(query_dataset_path, "r") as infile:
        queries = json.load(infile)

    ef_search_values = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]

    logger.info("Starting QPS ef_search experiment (logn model)")
    start = time.perf_counter()
    results = run_experiment_on_ef_search(queries, ef_search_values)
    elapsed = time.perf_counter() - start
    logger.info("ef_search experiment finished in %.2f min", elapsed / 60)

    logger.info("Fitting QPS model (query_time = a * ef_search + b * log(n))")
    fit_start = time.perf_counter()
    combined_params = fit_query_time_function_with_efs_and_logn(results)
    fit_elapsed = time.perf_counter() - fit_start
    logger.info("QPS model fit completed in %.2fs", fit_elapsed)

    join_times = run_experiment_on_join_time(queries)
    logger.info(
        "Combined ef_search/log(n) model parameters: a=%.2f, b=%.2f",
        combined_params[0],
        combined_params[1],
    )
    return combined_params, join_times


if __name__ == "__main__":
    get_hnsw_qps_parameters_with_logn()
