from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from ex1 import (
    DEFAULT_RECALL_ANCHORS,
    DISPLAY_METHOD_NAME,
    METHOD_STYLES,
    _format_recall_tick,
    _format_time_tick,
    _method_order,
    _style_for_method,
    load_points,
    write_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
DEFAULT_LOG_DIR = BENCHMARK_DIR / "efs_logs" / "YFCC"
DEFAULT_OUTPUT = DEFAULT_LOG_DIR / "yfcc_recall_time.png"
DEFAULT_MIN_RECALL = 0.8


def _draw_break_marks(upper_ax, lower_ax) -> None:
    kwargs = dict(marker=[(-1, -0.7), (1, 0.7)], markersize=8, linestyle="none", color="#5f5f5f", mec="#5f5f5f", mew=1.0, clip_on=False)
    upper_ax.plot([0, 1], [0, 0], transform=upper_ax.transAxes, **kwargs)
    lower_ax.plot([0, 1], [1, 1], transform=lower_ax.transAxes, **kwargs)


def plot_broken_y(points_by_method: dict, output: Path, *, min_recall: float) -> None:
    y_ranges = ((18.0, 36.5), (4.35, 8.2), (0.0, 3.0))
    fig, axes = plt.subplots(
        3,
        1,
        sharex=True,
        figsize=(5.35, 4.15),
        gridspec_kw={"height_ratios": [1.28, 0.82, 0.82], "hspace": 0.055},
    )

    all_recalls: list[float] = []
    plotted = 0
    ordered_methods = _method_order(set(points_by_method))
    for index, method in enumerate(ordered_methods):
        points = points_by_method.get(method, [])
        if not points:
            continue
        plotted += 1
        xs = [point.recall for point in points]
        ys = [point.query_time_ms for point in points]
        all_recalls.extend(xs)
        style = _style_for_method(method, index)
        style["zorder"] = 10 if method == "OURS" else 3 + index
        for ax in axes:
            ax.plot(xs, ys, label=DISPLAY_METHOD_NAME.get(method, method), **style)

    if plotted == 0:
        raise RuntimeError(f"No valid YFCC benchmark log points were found under {DEFAULT_LOG_DIR}")

    for ax, ylim in zip(axes, y_ranges):
        ax.set_ylim(*ylim)
        ax.yaxis.set_major_formatter(FuncFormatter(_format_time_tick))
        ax.grid(True, which="major", color="#d8d8d8", linewidth=0.65, alpha=0.55)
        ax.tick_params(axis="both", labelsize=11, direction="in")
        for spine in ax.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#5f5f5f")

    axes[0].set_yticks([20, 25, 30, 35])
    axes[1].set_yticks([5, 6, 7, 8])
    axes[2].set_yticks([0, 1, 2, 3])

    axes[0].spines["bottom"].set_visible(False)
    axes[1].spines["top"].set_visible(False)
    axes[1].spines["bottom"].set_visible(False)
    axes[2].spines["top"].set_visible(False)
    axes[0].tick_params(labelbottom=False, bottom=False)
    axes[1].tick_params(labelbottom=False, bottom=False)

    _draw_break_marks(axes[0], axes[1])
    _draw_break_marks(axes[1], axes[2])

    max_recall = max(all_recalls) if all_recalls else 1.0
    axes[-1].set_xlim(float(min_recall), min(1.0, max_recall + 0.004))
    axes[-1].xaxis.set_major_formatter(FuncFormatter(_format_recall_tick))
    axes[-1].set_xlabel("Recall@10", fontsize=13)
    fig.text(0.025, 0.5, "Query Time (ms)", va="center", rotation="vertical", fontsize=13)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.54, 1.02),
        ncol=4,
        frameon=False,
        fontsize=9.7,
        handlelength=2.0,
        columnspacing=1.25,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.14, right=0.985, bottom=0.13, top=0.86)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot YFCC recall-latency curves with a broken y-axis.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--min-recall", type=float, default=DEFAULT_MIN_RECALL)
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    output = Path(args.output).resolve()
    anchors = tuple(anchor for anchor in DEFAULT_RECALL_ANCHORS if anchor >= float(args.min_recall))
    points_by_method = load_points(
        log_dir,
        min_recall=float(args.min_recall),
        anchors=anchors,
        keep_raw_points=True,
    )
    plot_broken_y(points_by_method, output, min_recall=float(args.min_recall))
    csv_path = write_csv(points_by_method, output)
    print(f"Saved figure to {output}")
    if output.suffix.lower() != ".pdf":
        print(f"Saved figure to {output.with_suffix('.pdf')}")
    if csv_path is not None:
        print(f"Saved parsed data to {csv_path}")


if __name__ == "__main__":
    main()
