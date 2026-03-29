import json
import os
import random
import sys
import time
from operator import index

import psycopg2
from psycopg2 import sql
from scipy.stats import alpha



# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)
from controller.dynamic_partition.load_result_to_database import drop_indexes_for_all_partitions
from controller.dynamic_partition.search import dynamic_partition_search_stats_parameter

from controller.dynamic_partition.hnsw.analysis.analysis_hnsw_qps import get_hnsw_qps_parameters
from controller.dynamic_partition.hnsw.analysis.analysis_hnsw_recall import get_hnsw_recall_parameters
from services.read_dataset_function import generate_query_dataset_for_cache, generate_query_dataset

from controller.baseline.prefilter.initialize_partitions import drop_indexes_for_all_role_tables

from controller.baseline.prefilter.prefilter_role import search_documents_role_partition_get_parameter

from controller.baseline.pg_row_security.row_level_security import drop_database_users, create_database_users
from controller.initialize_main_tables import drop_indexes, create_indexes
from basic_benchmark.common_function import get_index_type, prepare_query_dataset, run_test
from services.config import get_db_connection
from services.logger import get_logger


logger = get_logger(__name__)


def get_alpha_beta_gamma_hashjointime(k_queries=1):
    drop_indexes()
    drop_indexes_for_all_partitions()

    # Generate and use the first k_queries
    queries = prepare_query_dataset(regenerate=True, num_queries=10)[:k_queries]

    # Initialize accumulators for the total times across all queries
    total_query_time_accum = 0
    total_hashjoin_time_accum = 0
    total_fetch_qual_time_accum = 0
    total_proj_time_accum = 0
    num_queries = len(queries)  # Total number of queries

    # Loop twice to ensure caching
    for i in range(2):
        for query in queries:
            user_id = query["user_id"]
            query_vector = query["query_vector"]
            topk = query.get("topk", 5)

            # Call the function and get the times for each query
            _, avg_fetch_qual_time, avg_hashjoin_time, avg_proj_time = (
                dynamic_partition_search_stats_parameter(user_id, query_vector, topk=topk)
            )

            # Accumulate times for averaging in the second loop only
            if i == 1:  # Second pass for caching
                total_hashjoin_time_accum += avg_hashjoin_time
                total_fetch_qual_time_accum += avg_fetch_qual_time
                total_proj_time_accum += avg_proj_time

    # Calculate average times across all queries (second pass)
    average_total_query_time = total_query_time_accum / num_queries
    average_hashjoin_time = total_hashjoin_time_accum / num_queries
    alpha_beta = total_fetch_qual_time_accum / num_queries
    gama = total_proj_time_accum / num_queries

    return alpha_beta, gama, average_hashjoin_time


def get_alpha2(k_queries=1):
    drop_indexes()
    drop_indexes_for_all_role_tables()

    # Generate and use the first k_queries
    queries = prepare_query_dataset(regenerate=True, num_queries=10)[:k_queries]

    # Initialize accumulator for total alpha2 across all queries
    total_alpha2 = 0
    num_queries = len(queries)  # Total number of queries

    # Loop twice to ensure caching
    for i in range(2):
        for query in queries:
            user_id = query["user_id"]
            query_vector = query["query_vector"]
            topk = query.get("topk", 5)

            # Call the function and get the alpha2 for each query
            _, alpha2 = search_documents_role_partition_get_parameter(user_id, query_vector, topk=topk)

            # Accumulate alpha2 in the second pass only
            if i == 1:  # Second pass for caching
                total_alpha2 += alpha2

    # Calculate average alpha2 across all queries (second pass)
    avg_alpha2 = total_alpha2 / num_queries

    return avg_alpha2

def _ensure_query_dataset(generator_fn, *, output_file, num_queries, **generator_kwargs):
    """Create the heavy query dataset only when we really need to."""
    benchmark_folder = os.path.join(project_root, "basic_benchmark")
    dataset_path = os.path.join(benchmark_folder, output_file)

    regenerate = generator_kwargs.pop("regenerate", False)

    if regenerate or not os.path.exists(dataset_path):
        logger.info(
            "Generating %s with %d queries (regenerate=%s)...",
            output_file,
            num_queries,
            regenerate,
        )
        start = time.perf_counter()
        generator_fn(
            num_queries=num_queries,
            output_file=dataset_path,
            **generator_kwargs,
        )
        elapsed = time.perf_counter() - start
        logger.info("Saved dataset to %s in %.2fs", dataset_path, elapsed)
    else:
        logger.info("Reusing cached dataset at %s", dataset_path)


def get_recall_parameters(index_type="hnsw", *, num_queries=None, regenerate=False):
    if index_type == "hnsw":
        query_count = num_queries or int(os.getenv("HNSW_RECALL_PARAM_QUERIES", "1000"))
        logger.info(
            "Preparing recall parameters (queries=%d, regenerate=%s)",
            query_count,
            regenerate,
        )
        _ensure_query_dataset(
            generate_query_dataset,
            output_file="query_dataset.json",
            num_queries=query_count,
            topk=5,
            zipf_param=0,
            num_threads=4,
            regenerate=regenerate,
        )
        start = time.perf_counter()
        params = get_hnsw_recall_parameters()
        elapsed = time.perf_counter() - start
        logger.info("Finished recall parameter fit in %.2f min", elapsed / 60)
        return params
    elif index_type == "ivfflat":
        return None



def get_QPS_parameters(index_type="hnsw", *, num_queries=None, regenerate=False):
    if index_type == "hnsw":
        query_count = num_queries or int(os.getenv("HNSW_QPS_PARAM_QUERIES", "1000"))
        logger.info(
            "Preparing QPS parameters (queries=%d, regenerate=%s)",
            query_count,
            regenerate,
        )
        _ensure_query_dataset(
            generate_query_dataset_for_cache,
            output_file="query_dataset.json",
            num_queries=query_count,
            topk=5,
            zipf_param=0,
            num_threads=4,
            regenerate=regenerate,
        )
        start = time.perf_counter()
        params = get_hnsw_qps_parameters()
        elapsed = time.perf_counter() - start
        logger.info("Finished QPS parameter fit in %.2f min", elapsed / 60)
        return params
    elif index_type == "ivfflat":
        return None


def save_parameter_to_json(index_type=None):
    # Define the JSON file path
    json_file_path = f"parameter_{index_type}.json"
    result_data = {}
    if index_type is None:
        k = 10
        alpha_beta, gama, average_hashjoin_time = get_alpha_beta_gamma_hashjointime(k_queries=k)
        avg_alpha2 = get_alpha2(k_queries=k)
        alpha_beta *= 10 ** 6
        gama *= 10 ** 6
        average_hashjoin_time *= 10 ** 6
        avg_alpha2 *= 10 ** 6

        print("Parameters:")
        print(f"  Alpha_Beta: {alpha_beta}")
        print(f"  Gama: {gama}")
        print(f"  Average Hash Join Time: {average_hashjoin_time}")
        print(f"  Average Alpha2: {avg_alpha2}")

        result_data = {
            "alpha_beta": alpha_beta,
            "gama": gama,
            "average_hashjoin_time": average_hashjoin_time,
            "alpha2": avg_alpha2
        }
    elif index_type == "hnsw":
        params_recall = get_recall_parameters(index_type=index_type)
        k = params_recall[0]
        beta = params_recall[1]
        params_qps, join_times = get_QPS_parameters(index_type=index_type)
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
    elif index_type == "ivfflat":
        return None
        # Write the results to the JSON file
    with open(json_file_path, "w") as json_file:
        json.dump(result_data, json_file, indent=4)
    logger.info("Data written to parameter_%s.json", index_type)


if __name__ == '__main__':
    save_parameter_to_json(index_type="hnsw")
