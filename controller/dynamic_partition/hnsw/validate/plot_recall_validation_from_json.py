import argparse
import json
import os
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np


def load_validation_data(path: str) -> Sequence[dict]:
    with open(path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def _load_parameters(project_root: str) -> dict:
    param_path = os.path.join(project_root, "controller", "dynamic_partition", "hnsw", "parameter_hnsw.json")
    with open(param_path, "r", encoding="utf-8") as infile:
        return json.load(infile)


def _select_xticks(values: Sequence[float], threshold: int = 100, step: int = 200) -> np.ndarray:
    unique = np.array(sorted(set(values)))
    if unique.size == 0:
        return unique

    max_val = unique[-1]
    if max_val <= threshold:
        return np.array([max_val], dtype=int)

    ticks = np.arange(threshold, max_val, step, dtype=float)
    if ticks.size == 0:
        ticks = np.array([threshold], dtype=float)
    ticks = sorted({int(round(t)) for t in ticks})
    return np.array(ticks, dtype=int)


def plot_recall_validation(data_path: str, output_path: str, font_size: int = 14) -> None:
    data_path = Path(data_path).resolve()
    records = load_validation_data(str(data_path))
    try:
        project_root = data_path.parents[4]
    except IndexError:
        raise ValueError(f"Unable to determine project root from data path {data_path}") from None

    params = _load_parameters(str(project_root))
    query_dataset_path = project_root / "basic_benchmark" / "query_dataset.json"
    with open(query_dataset_path, "r", encoding="utf-8") as infile:
        queries = json.load(infile)
    if not queries:
        raise RuntimeError("Query dataset is empty; cannot compute recall predictions.")
    topk = queries[0]["topk"]
    avg_selectivity = float(np.mean([query["query_block_selectivity"] for query in queries]))

    ef_search = np.array([row["ef_search"] for row in records])
    actual = np.array([row["actual_recall"] for row in records])
    predicted = np.array([row["predicted_recall"] for row in records])

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
        edgecolors="white",
        linewidths=1.1,
        s=90,
        label="Measured",
        zorder=3,
    )

    ax.set_xlabel("ef_search", fontsize=font_size + 4)
    ax.set_ylabel("Recall", fontsize=font_size + 4)
    xtick_positions = _select_xticks(ef_search, threshold=100, step=200)
    ax.set_xticks(xtick_positions)
    ax.tick_params(axis="x", labelsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.set_ylim(0, 1.05)
    gamma = params["k"]
    x_c = gamma * topk / avg_selectivity
    ax.axvline(x_c, color="#2a9d8f", linestyle="--", linewidth=1.8, zorder=1)
    ax.legend(fontsize=font_size, loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot recall validation curve from JSON data.")
    default_data = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "recall_model_validation_data.json",
    )
    parser.add_argument("--data", default=default_data, help="Path to recall_model_validation_data.json.")
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "recall_model_validation_plot.pdf",
        ),
        help="Output figure path.",
    )
    parser.add_argument("--font-size", type=int, default=20, help="Base font size for the plot.")
    args = parser.parse_args()
    plot_recall_validation(args.data, args.output, args.font_size)


if __name__ == "__main__":
    main()
