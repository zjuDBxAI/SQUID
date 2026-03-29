import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
from psycopg2 import sql

PROJECT_ROOT = Path(__file__).resolve().parents[4]
import sys

sys.path.append(str(PROJECT_ROOT))

from services.logger import get_logger  # noqa: E402
from controller.baseline.pg_row_security.row_level_security import (  # noqa: E402
    get_db_connection_for_many_users,
)
from controller.dynamic_partition.search import merge_results  # noqa: E402
from basic_benchmark.common_function import ground_truth_func  # noqa: E402

logger = get_logger(__name__)


def load_parameters() -> Dict[str, float]:
    param_path = PROJECT_ROOT / "controller" / "dynamic_partition" / "hnsw" / "parameter_hnsw.json"
    logger.info("Loading HNSW parameters from %s", param_path)
    with param_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)
    required = {"k", "beta"}
    missing = required - data.keys()
    if missing:
        raise KeyError(f"Missing required keys {missing} in parameter file {param_path}")
    return data


def load_query_dataset() -> List[Dict]:
    dataset_path = PROJECT_ROOT / "basic_benchmark" / "query_dataset.json"
    logger.info("Loading query dataset from %s", dataset_path)
    with dataset_path.open("r", encoding="utf-8") as infile:
        return json.load(infile)


def fetch_dynamic_partitions(cur, user_id: int) -> List[int]:
    cur.execute(
        """
        SELECT role_id
        FROM UserRoles
        WHERE user_id = %s;
        """,
        [user_id],
    )
    roles = [row[0] for row in cur.fetchall()]
    if not roles:
        return []

    sorted_roles = sorted(roles)
    cur.execute(
        """
        SELECT partition_id
        FROM CombRolePartitions
        WHERE comb_role = %s::integer[];
        """,
        [sorted_roles],
    )
    partitions = [row[0] for row in cur.fetchall()]
    return partitions


def collect_dynamic_results(
    user_id: int,
    query_vector: List[float],
    topk: int,
    ef_search_values: List[int],
) -> Dict[int, List[tuple]]:
    conn = get_db_connection_for_many_users(user_id)
    cur = conn.cursor()
    try:
        cur.execute("SET max_parallel_workers_per_gather = 0;")
        cur.execute("SET jit = off;")

        partitions = fetch_dynamic_partitions(cur, user_id)
        if not partitions:
            logger.warning("No dynamic partitions found for user %s; skipping query.", user_id)
            return {ef: [] for ef in ef_search_values}

        results_by_ef = {}
        for ef_search in ef_search_values:
            cur.execute(f"SET hnsw.ef_search = {ef_search};")
            all_results = []
            for partition_id in partitions:
                partition_table = sql.Identifier(f"documentblocks_partition_{partition_id}")
                search_query = sql.SQL(
                    """
                    SELECT block_id, document_id, block_content,
                           vector <-> %s::vector AS distance
                    FROM {}
                    ORDER BY distance
                    LIMIT %s;
                    """
                ).format(partition_table)

                cur.execute(search_query, [query_vector, topk])
                all_results.extend(cur.fetchall())

            merged_results = merge_results(all_results, topk)
            results_by_ef[ef_search] = merged_results
        return results_by_ef
    finally:
        cur.close()
        conn.close()


def compute_actual_dynamic_recalls(queries: List[Dict], ef_search_values: List[int]) -> List[float]:
    actual_recalls = {ef: [] for ef in ef_search_values}

    for query in queries:
        user_id = query["user_id"]
        query_vector = query["query_vector"]
        topk = query["topk"]

        ground_truth_results = ground_truth_func(user_id=user_id, query_vector=query_vector, topk=topk)
        ground_truth_set = set((row[1], row[0]) for row in ground_truth_results)
        if not ground_truth_set:
            continue

        partition_results = collect_dynamic_results(user_id, query_vector, topk, ef_search_values)
        for ef_search in ef_search_values:
            retrieved = partition_results.get(ef_search, [])
            retrieved_set = set((row[1], row[0]) for row in retrieved)
            correct_matches = len(retrieved_set & ground_truth_set)
            recall = correct_matches / len(ground_truth_set)
            actual_recalls[ef_search].append(recall)

    averaged = [
        float(np.mean(actual_recalls[ef_search])) if actual_recalls[ef_search] else 0.0
        for ef_search in ef_search_values
    ]
    return averaged


def compute_predicted_recalls(
    ef_search_values: List[int],
    gamma: float,
    beta: float,
    topk: int,
    avg_selectivity: float,
) -> List[float]:
    if topk <= 0:
        raise ValueError("topk must be positive to compute predicted recalls.")
    if avg_selectivity <= 0:
        raise ValueError("Average selectivity must be positive to compute predicted recalls.")

    x_c = gamma * topk / avg_selectivity
    sigmoid_rate = beta * 4 * avg_selectivity / topk
    shift = x_c * avg_selectivity / topk - 0.5

    predictions = []
    for ef_search in ef_search_values:
        if ef_search <= x_c:
            recall = ef_search * avg_selectivity / topk
        else:
            exponent = -sigmoid_rate * (ef_search - x_c)
            recall = 1.0 / (1.0 + np.exp(exponent)) + shift
        predictions.append(float(min(max(recall, 0.0), 1.0)))
    return predictions


def save_results(
    ef_search_values: List[int],
    actual_recalls: List[float],
    predicted_recalls: List[float],
) -> Path:
    output_path = Path(__file__).with_name("recall_model_dynamic_partition_validation_data.json")
    records = []
    for ef_search, actual, predicted in zip(ef_search_values, actual_recalls, predicted_recalls):
        records.append(
            {
                "ef_search": int(ef_search),
                "actual_recall": float(actual),
                "predicted_recall": float(predicted),
            }
        )

    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(records, outfile, indent=2)
    logger.info("Saved dynamic-partition recall validation data to %s", output_path)
    return output_path


def main():
    params = load_parameters()
    queries = load_query_dataset()
    if not queries:
        raise RuntimeError("Query dataset is empty; cannot perform validation.")

    ef_search_values = [1, 3, 5, 7, 10, 15, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    topk = queries[0]["topk"]
    avg_selectivity = float(np.mean([query["query_block_selectivity"] for query in queries]))

    logger.info("Computing actual recalls using dynamic partitions for %d queries.", len(queries))
    actual_recalls = compute_actual_dynamic_recalls(queries, ef_search_values)

    logger.info(
        "Computing predicted recalls with gamma=%.4f, beta=%.4f (topk=%d, avg_selectivity=%.6f)",
        params["k"],
        params["beta"],
        topk,
        avg_selectivity,
    )
    predicted_recalls = compute_predicted_recalls(
        ef_search_values,
        params["k"],
        params["beta"],
        topk,
        avg_selectivity,
    )

    save_results(ef_search_values, actual_recalls, predicted_recalls)


if __name__ == "__main__":
    main()
