from __future__ import annotations

import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
OUTPUT_DIR = PROJECT_ROOT / "basic_benchmark" / "result" / "direct_pg_qps"

METHODS = (
    ("SQUID", "#4C78A8", (1173.34, 1100.32, 1283.0)),
    ("EFFVEDA", "#F58518", (1009.0, 984.0, 1134.0)),
)

SEARCH_MODES = (
    ("HNSW", ""),
    ("PG compensation", "///"),
    ("Native", "..."),
)


def write_csv(path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "search_mode", "qps"])
        for method, _color, values in METHODS:
            for (mode, _hatch), qps in zip(SEARCH_MODES, values):
                writer.writerow([method, mode, f"{float(qps):.6f}"])


def plot(output_base: Path) -> None:
    fig, ax = plt.subplots(figsize=(4.2, 2.75))

    group_centers = list(range(len(METHODS)))
    bar_width = 0.19
    offsets = (-bar_width, 0.0, bar_width)

    for group_index, (method, color, values) in enumerate(METHODS):
        for mode_index, ((mode, hatch), qps) in enumerate(zip(SEARCH_MODES, values)):
            x = group_centers[group_index] + offsets[mode_index]
            bar = ax.bar(
                x,
                float(qps),
                width=bar_width,
                color=color,
                edgecolor="#2f2f2f",
                linewidth=0.85,
                hatch=hatch,
                alpha=0.94,
                zorder=3,
            )[0]
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 22,
                f"{float(qps):.0f}",
                ha="center",
                va="bottom",
                fontsize=9.2,
                color="#222222",
            )

    legend_handles = [
        Patch(facecolor="#d9d9d9", edgecolor="#2f2f2f", linewidth=0.8, hatch=hatch, label=mode)
        for mode, hatch in SEARCH_MODES
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.16),
        ncol=3,
        frameon=False,
        fontsize=9.8,
        handlelength=1.5,
        columnspacing=1.2,
    )

    ax.set_xticks(group_centers)
    ax.set_xticklabels([method for method, _color, _values in METHODS], fontsize=12.5)
    ax.set_ylabel("QPS", fontsize=13.5)
    ax.tick_params(axis="y", labelsize=10.8)
    ax.set_ylim(0, 1450)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.65, alpha=0.65)
    ax.grid(False, axis="x")

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#5f5f5f")
        ax.spines[spine].set_linewidth(0.85)

    fig.tight_layout(pad=0.35)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_base = OUTPUT_DIR / "qps_search_mode_comparison"
    write_csv(output_base.with_suffix(".csv"))
    plot(output_base)


if __name__ == "__main__":
    main()
