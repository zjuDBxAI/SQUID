import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def _load_parameters(project_root: str) -> dict:
    param_path = os.path.join(project_root, "controller", "dynamic_partition", "hnsw", "parameter_hnsw.json")
    with open(param_path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def _load_data(data_path: str) -> Sequence[dict]:
    with open(data_path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def compute_piecewise_prediction(
    ef_search_values: np.ndarray,
    k: float,
    beta: float,
    topk: int,
    avg_selectivity: float,
) -> np.ndarray:
    if topk <= 0:
        raise ValueError("topk must be positive.")
    if avg_selectivity <= 0:
        raise ValueError("Average selectivity must be positive.")

    x_c = k * topk / avg_selectivity
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
    return np.array(predictions)


def plot_recall_from_json(
    data_path: str,
    query_dataset_path: str,
    output_path: str,
    font_size: int = 18,
) -> None:
    data_path = Path(data_path).resolve()
    data = _load_data(str(data_path))
    try:
        project_root = data_path.parents[4]
    except IndexError:
        raise ValueError(f"Unable to determine project root from data path {data_path}") from None
    params = _load_parameters(str(project_root))

    with open(query_dataset_path, "r", encoding="utf-8") as infile:
        queries = json.load(infile)
    if not queries:
        raise RuntimeError("Query dataset is empty.")
    topk = queries[0]["topk"]
    avg_selectivity = float(np.mean([query["query_block_selectivity"] for query in queries]))

    ef_search = np.array([row["ef_search"] for row in data])
    actual = np.array([row["average_recall"] for row in data])
    predicted = compute_piecewise_prediction(
        ef_search,
        params["k"],
        params["beta"],
        topk,
        avg_selectivity,
    )

    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=600)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)
    ax.plot(
        ef_search,
        predicted,
        color="#d1495b",
        linewidth=2.3,
        label="Model Prediction",
        zorder=2,
    )
    ax.scatter(
        ef_search,
        actual,
        color="#1d3557",
        marker="o",
        s=110,
        edgecolors="white",
        linewidths=1.1,
        label="Measured",
        zorder=3,
    )

    ax.set_xlabel("ef_search", fontsize=font_size + 4)
    ax.set_ylabel("Recall", fontsize=font_size + 4)
    # Sparse ticks: at most 6 overall, and below 100 only keep one marker (formed by the last value <=100)
    if len(ef_search) > 6:
        indices = np.linspace(0, len(ef_search) - 1, 6, dtype=int)
        xtick_positions = ef_search[indices]
    else:
        xtick_positions = ef_search
    small_mask = xtick_positions <= 100
    if small_mask.any():
        last_small = xtick_positions[small_mask][-1]
        xtick_positions = [p for p in xtick_positions if p > 100]
        xtick_positions.insert(0, last_small)
    ax.set_xticks(xtick_positions)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=font_size, loc="lower right", framealpha=0.9)
    # vertical dashed line at transition
    x_c = params["k"] * topk / avg_selectivity
    ax.axvline(x_c, color="#2a9d8f", linestyle="--", linewidth=1.8, label=None)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot HNSW recall curve from JSON data.")
    default_data = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "recall_analysis_data.json",
    )
    parser.add_argument("--data", default=default_data, help="Path to recall_analysis_data.json.")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "recall_analysis_plot.pdf",
        ),
        help="Output figure path.",
    )
    parser.add_argument("--font-size", type=int, default=18, help="Base font size for the plot.")
    args = parser.parse_args()

    data_path = Path(args.data).resolve()
    try:
        project_root = data_path.parents[4]
    except IndexError:
        raise ValueError(f"Unable to determine project root from data path {data_path}") from None
    query_dataset_path = project_root / "basic_benchmark" / "query_dataset.json"

    plot_recall_from_json(args.data, str(query_dataset_path), args.output, args.font_size)


if __name__ == "__main__":
    main()
