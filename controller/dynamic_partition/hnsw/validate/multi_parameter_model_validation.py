import argparse
import json
import os
import sys
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np

project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
sys.path.append(project_root)

from services.logger import get_logger  # noqa: E402
from controller.dynamic_partition.hnsw.analysis.analysis_hnsw_qps import (  # noqa: E402
    run_experiment_on_ef_search,
)
from controller.dynamic_partition.hnsw.analysis.analysis_hnsw_recall import (  # noqa: E402
    calculate_actual_recall_batch,
)
from basic_benchmark.common_function import ground_truth_func  # noqa: E402

logger = get_logger(__name__)

ParameterVariant = Dict[str, float]

PARAMETER_VARIANTS: Sequence[ParameterVariant] = (
    {
        "label": "Tree",
        "k": 0.4739,
        "beta": 0.5488,
        "a": 1830.0,
        "b": -308.33,
    },
    {
        "label": "ERBAC",
        "k": 0.47956901,
        "beta": 0.37764396,
        "a": 2203.28,
        "b": 36450.57,
    },
)


def _load_queries(dataset_path: str) -> List[dict]:
    with open(dataset_path, "r", encoding="utf-8") as src:
        queries = json.load(src)
    if not queries:
        raise RuntimeError(f"Query dataset {dataset_path} is empty.")
    return queries


def _compute_qps_actual(
    queries: Sequence[dict],
    ef_search_values: Sequence[int],
    repetitions: int,
) -> Tuple[List[int], List[float], List[float]]:
    results = run_experiment_on_ef_search(list(queries), list(ef_search_values), repetitions=repetitions)
    ef_values = [int(row["ef_search"]) for row in results]
    actual_times = [float(row["avg_query_time"]) for row in results]
    avg_total_rows = [float(row["avg_total_rows"]) for row in results]
    return ef_values, actual_times, avg_total_rows


def _compute_qps_predictions(
    ef_search_values: Sequence[int],
    avg_total_rows: Sequence[float],
    variant: ParameterVariant,
) -> List[float]:
    predictions: List[float] = []
    for ef_search, total_rows in zip(ef_search_values, avg_total_rows):
        if total_rows <= 0:
            predictions.append(0.0)
            continue
        log_n = np.log(total_rows)
        predictions.append(float((variant["a"] * ef_search + variant["b"]) * log_n))
    return predictions


def _compute_recall_actual(
    queries: Sequence[dict],
    ef_search_values: Sequence[int],
) -> Tuple[int, float, List[float]]:
    topk = int(queries[0]["topk"])
    selectivities = [query["query_block_selectivity"] for query in queries]
    avg_selectivity = float(np.mean(selectivities))
    recall_map = {ef: [] for ef in ef_search_values}

    for query in queries:
        user_id = query["user_id"]
        query_vector = query["query_vector"]
        recalls = calculate_actual_recall_batch(
            user_id,
            query_vector,
            topk,
            ground_truth_func,
            list(ef_search_values),
        )
        for ef_search, recall in zip(ef_search_values, recalls):
            recall_map[ef_search].append(recall)

    averaged = [
        float(np.mean(recall_map[ef_search])) if recall_map[ef_search] else 0.0
        for ef_search in ef_search_values
    ]
    return topk, avg_selectivity, averaged


def _compute_recall_predictions(
    ef_search_values: Sequence[int],
    topk: int,
    avg_selectivity: float,
    variant: ParameterVariant,
) -> List[float]:
    if topk <= 0:
        raise ValueError("topk must be positive.")
    if avg_selectivity <= 0:
        raise ValueError("Average selectivity must be positive.")

    k_param = variant["k"]
    beta_param = variant["beta"]
    x_c = k_param * topk / avg_selectivity
    sigmoid_rate = beta_param * 4 * avg_selectivity / topk
    shift = x_c * avg_selectivity / topk - 0.5

    predictions: List[float] = []
    for ef_search in ef_search_values:
        if ef_search <= x_c:
            recall = ef_search * avg_selectivity / topk
        else:
            exponent = -sigmoid_rate * (ef_search - x_c)
            recall = 1.0 / (1.0 + np.exp(exponent)) + shift
        predictions.append(float(min(max(recall, 0.0), 1.0)))
    return predictions


def _formatted_qps_label(variant: ParameterVariant) -> str:
    return variant["label"]


def _formatted_recall_label(variant: ParameterVariant) -> str:
    return variant["label"]


def _plot_qps(
    ef_search_values: Sequence[int],
    actual_times: Sequence[float],
    avg_total_rows: Sequence[float],
    prediction_series: List[Tuple[ParameterVariant, List[float]]],
    output_dir: str,
) -> None:
    actual_ms = [time_val / 1_000_000.0 for time_val in actual_times]

    plt.figure(figsize=(8, 6), dpi=600)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, (variant, predicted_times) in enumerate(prediction_series):
        predicted_ms = [time_val / 1_000_000.0 for time_val in predicted_times]
        color = color_cycle[idx % len(color_cycle)] if color_cycle else "#d1495b"
        plt.plot(
            ef_search_values,
            predicted_ms,
            color=color,
            linewidth=2.5,
            label=_formatted_qps_label(variant),
            zorder=2 + idx,
        )

    plt.scatter(
        ef_search_values,
        actual_ms,
        color="#1d3557",
        marker="o",
        s=140,
        edgecolors="white",
        linewidths=1.2,
        label="Measured",
        zorder=2 + len(prediction_series),
    )
    plt.xlabel("ef_search", fontsize=28, fontweight="normal")
    plt.ylabel("Average Query Time (ms)", fontsize=28, fontweight="normal")
    xticks = list(ef_search_values)
    tick_labels = [str(value) if idx % 2 == 0 else "" for idx, value in enumerate(xticks)]
    plt.xticks(xticks, tick_labels, fontsize=26)
    plt.yticks(fontsize=26)
    plt.legend(fontsize=22, loc="upper left")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "qps_model_validation_multi.pdf")
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    logger.info("Saved multi-parameter QPS validation plot to %s", output_path)

    records: List[Dict[str, float]] = []
    for idx, ef_search in enumerate(ef_search_values):
        record: Dict[str, float] = {
            "ef_search": int(ef_search),
            "avg_total_rows": float(avg_total_rows[idx]),
            "actual_query_time": float(actual_times[idx]),
            "actual_query_time_ms": float(actual_ms[idx]),
        }
        prediction_payload = []
        for variant, predicted_times in prediction_series:
            predicted_value = float(predicted_times[idx])
            prediction_payload.append(
                {
                    "label": variant["label"],
                    "display_label": _formatted_qps_label(variant),
                    "a": float(variant["a"]),
                    "b": float(variant["b"]),
                    "k": float(variant["k"]),
                    "beta": float(variant["beta"]),
                    "query_time": predicted_value,
                    "query_time_ms": float(predicted_value / 1_000_000.0),
                }
            )
        record["predictions"] = prediction_payload
        records.append(record)

    data_path = os.path.join(output_dir, "qps_model_validation_multi_data.json")
    with open(data_path, "w", encoding="utf-8") as dst:
        json.dump(records, dst, indent=2)
    logger.info("Saved multi-parameter QPS validation data to %s", data_path)


def _select_xticks(values: Sequence[int], threshold: int = 100, step: int = 200) -> np.ndarray:
    unique = np.array(sorted(set(values)))
    if unique.size == 0:
        return unique
    max_val = unique[-1]
    if max_val <= threshold:
        return np.array([max_val], dtype=int)
    ticks = np.arange(threshold, max_val, step, dtype=float)
    if ticks.size == 0:
        ticks = np.array([threshold], dtype=float)
    ticks = sorted({int(round(tick)) for tick in ticks})
    return np.array(ticks, dtype=int)


def _plot_recall(
    ef_search_values: Sequence[int],
    actual_recalls: Sequence[float],
    topk: int,
    avg_selectivity: float,
    prediction_series: List[Tuple[ParameterVariant, List[float]]],
    output_dir: str,
) -> None:
    plt.figure(figsize=(8, 6), dpi=600)
    plt.grid(True, linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    for idx, (variant, predicted_recalls) in enumerate(prediction_series):
        color = color_cycle[idx % len(color_cycle)] if color_cycle else "#d1495b"
        plt.plot(
            ef_search_values,
            predicted_recalls,
            color=color,
            linewidth=2.5,
            label=_formatted_recall_label(variant),
            zorder=2 + idx,
        )
        x_c = variant["k"] * topk / avg_selectivity
        plt.axvline(
            x_c,
            color=color,
            linestyle="--",
            linewidth=1.8,
            alpha=0.6,
            zorder=1,
        )

    plt.scatter(
        ef_search_values,
        actual_recalls,
        color="#1d3557",
        marker="o",
        s=140,
        edgecolors="white",
        linewidths=1.2,
        label="Measured",
        zorder=2 + len(prediction_series),
    )
    plt.xlabel("ef_search", fontsize=28, fontweight="normal")
    plt.ylabel("Recall", fontsize=28, fontweight="normal")
    xticks = _select_xticks(ef_search_values, threshold=100, step=200)
    plt.xticks(xticks, fontsize=26)
    plt.yticks(fontsize=26)
    plt.ylim(0, 1.05)
    plt.legend(fontsize=22, loc="lower right")
    plt.tight_layout()

    output_path = os.path.join(output_dir, "recall_model_validation_multi.pdf")
    plt.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close()
    logger.info("Saved multi-parameter recall validation plot to %s", output_path)

    records: List[Dict[str, float]] = []
    for idx, ef_search in enumerate(ef_search_values):
        record: Dict[str, float] = {
            "ef_search": int(ef_search),
            "actual_recall": float(actual_recalls[idx]),
        }
        prediction_payload = []
        for variant, predicted_recalls in prediction_series:
            prediction_payload.append(
                {
                    "label": variant["label"],
                    "display_label": _formatted_recall_label(variant),
                    "a": float(variant["a"]),
                    "b": float(variant["b"]),
                    "k": float(variant["k"]),
                    "beta": float(variant["beta"]),
                    "recall": float(predicted_recalls[idx]),
                }
            )
        record["predictions"] = prediction_payload
        records.append(record)

    data_path = os.path.join(output_dir, "recall_model_validation_multi_data.json")
    with open(data_path, "w", encoding="utf-8") as dst:
        json.dump(records, dst, indent=2)
    logger.info("Saved multi-parameter recall validation data to %s", data_path)


def _ensure_output_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate QPS and recall validation plots for multiple parameter variants.",
    )
    default_dataset = os.path.join(project_root, "basic_benchmark", "query_dataset.json")
    default_output = os.path.dirname(os.path.abspath(__file__))

    parser.add_argument("--query-dataset", default=default_dataset, help="Path to query_dataset.json.")
    parser.add_argument("--output-dir", default=default_output, help="Directory for generated artifacts.")
    parser.add_argument(
        "--ef-search-qps",
        nargs="+",
        type=int,
        default=[20, 40, 60, 80, 100, 120, 140, 160, 180, 200],
        help="ef_search values for the QPS validation curve.",
    )
    parser.add_argument(
        "--ef-search-recall",
        nargs="+",
        type=int,
        default=[1, 3, 5, 7, 10, 20, 30, 40, 50, 75, 100, 150, 200, 300, 400, 800, 1000],
        help="ef_search values for the recall validation curve.",
    )
    parser.add_argument("--repetitions", type=int, default=2, help="Number of repetitions for QPS experiments.")
    parser.add_argument("--skip-qps", action="store_true", help="Skip QPS validation plot generation.")
    parser.add_argument("--skip-recall", action="store_true", help="Skip recall validation plot generation.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    queries = _load_queries(args.query_dataset)
    output_dir = _ensure_output_dir(args.output_dir)

    if not args.skip_qps:
        logger.info("Starting QPS validation for %d parameter variants.", len(PARAMETER_VARIANTS))
        ef_qps, actual_times, avg_total_rows = _compute_qps_actual(queries, args.ef_search_qps, args.repetitions)
        qps_prediction_series = [
            (variant, _compute_qps_predictions(ef_qps, avg_total_rows, variant))
            for variant in PARAMETER_VARIANTS
        ]
        _plot_qps(ef_qps, actual_times, avg_total_rows, qps_prediction_series, output_dir)

    if not args.skip_recall:
        logger.info("Starting recall validation for %d parameter variants.", len(PARAMETER_VARIANTS))
        topk, avg_sel, actual_recalls = _compute_recall_actual(queries, args.ef_search_recall)
        recall_prediction_series = [
            (variant, _compute_recall_predictions(args.ef_search_recall, topk, avg_sel, variant))
            for variant in PARAMETER_VARIANTS
        ]
        _plot_recall(args.ef_search_recall, actual_recalls, topk, avg_sel, recall_prediction_series, output_dir)


if __name__ == "__main__":
    main()
