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


def plot_query_time_from_json(
    data_path: str,
    output_path: str,
    font_size: int = 14,
) -> None:
    data_path = Path(data_path).resolve()
    data = _load_data(str(data_path))
    # data_path.parents: [analysis, hnsw, dynamic_partition, controller, project_root]
    try:
        project_root = data_path.parents[4]
    except IndexError:
        raise ValueError(f"Unable to determine project root from data path {data_path}") from None
    params = _load_parameters(str(project_root))

    ef_search = np.array([row["ef_search"] for row in data])
    actual_ms = np.array(
        [
            row.get("avg_query_time_ms")
            if row.get("avg_query_time_ms") is not None
            else row["avg_query_time"] / 1_000_000.0
            for row in data
        ]
    )
    predicted_ms = np.array(
        [
            row.get("predicted_query_time_ms")
            if row.get("predicted_query_time_ms") is not None
            else row["predicted_query_time"] / 1_000_000.0
            for row in data
        ]
    )

    plt.rcParams.update({"font.size": font_size})
    fig, ax = plt.subplots(figsize=(7.5, 5.5), dpi=600)
    ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.6, zorder=0)
    ax.plot(
        ef_search,
        predicted_ms,
        color="#d1495b",
        linewidth=2.3,
        label="Model Fit",
        zorder=2,
    )
    ax.scatter(
        ef_search,
        actual_ms,
        color="#1d3557",
        edgecolors="white",
        linewidths=1.1,
        s=90,
        label="Measured",
        zorder=3,
    )

    ax.set_xlabel("ef_search", fontsize=font_size + 4)
    ax.set_ylabel("Query Time (ms)", fontsize=font_size + 4)
    ax.set_xticks(ef_search)
    ax.set_xticklabels([str(v) for v in ef_search], fontsize=font_size)
    ax.tick_params(axis="y", labelsize=font_size)
    ax.set_ylim(0, max(predicted_ms.max(), actual_ms.max()) * 1.1)
    ax.legend(fontsize=font_size - 1, loc="upper left", framealpha=0.9)

    fig.tight_layout()
    fig.savefig(output_path, dpi=600, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot HNSW QPS curve from JSON data.")
    parser.add_argument(
        "--data",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "query_time_analysis_data.json",
        ),
        help="Path to query_time_analysis_data.json.",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "query_time_analysis_plot.pdf",
        ),
        help="Output figure path.",
    )
    parser.add_argument("--font-size", type=int, default=20, help="Base font size for the plot.")
    args = parser.parse_args()

    plot_query_time_from_json(args.data, args.output, args.font_size)


if __name__ == "__main__":
    main()
