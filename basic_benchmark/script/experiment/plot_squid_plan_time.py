from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
DEFAULT_INPUT = PROJECT_ROOT / "basic_benchmark" / "result" / "plan_time" / "erbac_sift_plan_time_summary.csv"
DEFAULT_OUTPUT = PROJECT_ROOT / "basic_benchmark" / "result" / "plan_time" / "squid_plan_time_memory.png"

SQUID_COLOR = "#4C78A8"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def load_squid_points(path: Path) -> list[tuple[float, float, float]]:
    points: list[tuple[float, float, float]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if str(row.get("method", "")).strip().upper() != "SQUID":
                continue
            points.append(
                (
                    float(row["memory_budget"]),
                    float(row["plan_seconds_mean"]),
                    float(row["partition_count_mean"]),
                )
            )
    return sorted(points, key=lambda item: item[0])


def write_plot_csv(points: list[tuple[float, float, float]], output: Path) -> Path:
    csv_path = output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset_tag", "method", "memory_budget", "plan_seconds", "plan_minutes", "partition_count"],
        )
        writer.writeheader()
        for memory, seconds, partitions in points:
            writer.writerow(
                {
                    "dataset_tag": "erbac+sift",
                    "method": "SQUID",
                    "memory_budget": memory,
                    "plan_seconds": seconds,
                    "plan_minutes": seconds / 60.0,
                    "partition_count": partitions,
                }
            )
    return csv_path


def plot(points: list[tuple[float, float, float]], output: Path) -> None:
    if not points:
        raise RuntimeError("No SQUID plan-time points found")
    xs = [item[0] for item in points]
    ys = [item[1] for item in points]

    fig, ax = plt.subplots(figsize=(4.8, 3.15))
    ax.bar(
        xs,
        ys,
        width=0.58,
        color=SQUID_COLOR,
        edgecolor="#222222",
        linewidth=0.55,
        alpha=0.96,
    )
    ax.set_xlabel("Memory Budget", fontsize=14)
    ax.set_ylabel("Planning Time (s)", fontsize=14)
    ax.set_xticks([1, 2, 3, 4, 5, 6])
    ax.set_xlim(0.75, 6.25)
    ax.set_ylim(0, max(ys) * 1.14)
    ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.65, alpha=0.65)
    ax.tick_params(axis="both", labelsize=12, direction="in", width=1.05)
    for spine in ax.spines.values():
        spine.set_linewidth(1.05)
        spine.set_color("#555555")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot SQUID planning time across memory budgets.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    points = load_squid_points(args.input.resolve())
    plot(points, args.output.resolve())
    csv_path = write_plot_csv(points, args.output.resolve())
    print(f"Saved figure to {args.output.resolve()}")
    print(f"Saved figure to {args.output.resolve().with_suffix('.pdf')}")
    print(f"Saved plot data to {csv_path}")


if __name__ == "__main__":
    main()
