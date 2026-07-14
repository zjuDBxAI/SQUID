from __future__ import annotations

import argparse
import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
    "mathtext.fontset": "dejavusans",
    "axes.unicode_minus": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT = BENCHMARK_DIR / "result" / "memory" / "erbac_sift_partition_storage.csv"
DEFAULT_OUTPUT = BENCHMARK_DIR / "result" / "memory" / "erbac_sift_memory_usage.png"
BUDGETS = (1, 2, 3, 4, 5, 6)

PREFERRED_METHOD_ORDER = ("effveda", "veda", "QDTree", "AnonySys", "RLS", "ROLE", "OURS")
DISPLAY_METHOD_NAME = {
    "AnonySys": "HONEYBEE",
    "OURS": "SQUID",
    "effveda": "EFFVEDA",
    "veda": "VEDA",
    "QDTree": "HQI",
    "RLS": "RLS",
    "ROLE": "ROLE",
}
METHOD_STYLES = {
    "OURS": {"color": "#4C78A8"},
    "effveda": {"color": "#F58518"},
    "veda": {"color": "#7F3C8D"},
    "QDTree": {"color": "#B279A2"},
    "AnonySys": {"color": "#E45756"},
    "RLS": {"color": "#11A579"},
    "ROLE": {"color": "#79706E"},
}
METHOD_ALIASES = {
    "ours": "OURS",
    "squid": "OURS",
    "honeybee": "AnonySys",
    "anonysys": "AnonySys",
    "hqi": "QDTree",
    "qdtree": "QDTree",
    "rls": "RLS",
    "role": "ROLE",
    "effveda": "effveda",
    "veda": "veda",
}


@dataclass(frozen=True)
class MemoryPoint:
    method: str
    source_memory_ratio: float
    budget: int
    total_gb: float
    total_mb: float
    relation_count: int
    notes: str


def normalize_method(value: str) -> str:
    return METHOD_ALIASES.get(str(value).strip().lower(), str(value).strip())


def method_order(methods: set[str]) -> list[str]:
    ordered = [method for method in PREFERRED_METHOD_ORDER if method in methods]
    ordered.extend(sorted(method for method in methods if method not in set(ordered)))
    return ordered


def nearest_budget(memory_ratio: float) -> int:
    return min(BUDGETS, key=lambda budget: (abs(float(memory_ratio) - budget), budget))


def parse_float(value: str) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isnan(parsed):
        return None
    return parsed


def parse_int(value: str) -> int:
    parsed = parse_float(value)
    return int(parsed) if parsed is not None else 0


def load_points(input_path: Path) -> dict[str, dict[int, MemoryPoint]]:
    selected: dict[tuple[str, int], MemoryPoint] = {}
    with input_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            total_gb = parse_float(row.get("total_gb", ""))
            total_mb = parse_float(row.get("total_mb", ""))
            memory_ratio = parse_float(row.get("memory_ratio", ""))
            if total_gb is None or total_mb is None or memory_ratio is None:
                continue
            method = normalize_method(str(row.get("method", "")))
            budget = nearest_budget(memory_ratio)
            if method == "veda" and budget >= 3:
                continue
            if method == "effveda" and budget >= 6:
                continue
            point = MemoryPoint(
                method=method,
                source_memory_ratio=memory_ratio,
                budget=budget,
                total_gb=total_gb,
                total_mb=total_mb,
                relation_count=parse_int(row.get("relation_count", "")),
                notes=str(row.get("notes", "")),
            )
            key = (method, budget)
            old = selected.get(key)
            if old is None:
                selected[key] = point
                continue
            old_distance = abs(old.source_memory_ratio - budget)
            new_distance = abs(point.source_memory_ratio - budget)
            if (new_distance, -point.total_gb) < (old_distance, -old.total_gb):
                selected[key] = point

    grouped: dict[str, dict[int, MemoryPoint]] = {}
    for (method, budget), point in selected.items():
        grouped.setdefault(method, {})[budget] = point
    return {method: grouped[method] for method in method_order(set(grouped))}


def write_plot_csv(points_by_method: dict[str, dict[int, MemoryPoint]], output: Path) -> Path:
    csv_path = output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, by_budget in points_by_method.items():
        for budget in BUDGETS:
            point = by_budget.get(budget)
            if point is None:
                continue
            rows.append({
                "dataset_tag": "erbac+sift",
                "method": DISPLAY_METHOD_NAME.get(method, method),
                "raw_method": method,
                "memory_budget": budget,
                "source_memory_ratio": point.source_memory_ratio,
                "actual_memory_gb": point.total_gb,
                "actual_memory_mb": point.total_mb,
                "relation_count": point.relation_count,
                "notes": point.notes,
            })
    fieldnames = [
        "dataset_tag",
        "method",
        "raw_method",
        "memory_budget",
        "source_memory_ratio",
        "actual_memory_gb",
        "actual_memory_mb",
        "relation_count",
        "notes",
    ]
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def format_gb(value: float, _pos) -> str:
    return f"{value:.0f}" if value >= 10 else f"{value:.1f}"


def plot(points_by_method: dict[str, dict[int, MemoryPoint]], output: Path) -> None:
    methods = list(points_by_method)
    if not methods:
        raise RuntimeError("No memory usage points found")
    fig, ax = plt.subplots(figsize=(7.8, 4.15))
    group_width = 0.92
    max_y = 0.0
    legend_seen: set[str] = set()
    for budget in BUDGETS:
        present_methods = [method for method in methods if budget in points_by_method[method]]
        if not present_methods:
            continue
        bar_width = group_width / len(present_methods)
        offsets = [
            -group_width / 2 + bar_width / 2 + index * bar_width
            for index in range(len(present_methods))
        ]
        for index, method in enumerate(present_methods):
            point = points_by_method[method][budget]
            max_y = max(max_y, point.total_gb)
            label = DISPLAY_METHOD_NAME.get(method, method)
            legend_label = label if method not in legend_seen else "_nolegend_"
            legend_seen.add(method)
            color = METHOD_STYLES.get(method, {}).get("color", "#4C78A8")
            ax.bar(
                budget + offsets[index],
                point.total_gb,
                width=bar_width * 0.98,
                label=legend_label,
                color=color,
                edgecolor="#222222",
                linewidth=0.45,
                alpha=0.96,
            )

    ax.set_xlabel("Memory Budget", fontsize=18)
    ax.set_ylabel("Actual Memory Usage (GB)", fontsize=18)
    ax.set_xticks(list(BUDGETS))
    ax.set_xlim(0.48, 6.52)
    ax.set_ylim(0, max_y * 1.12)
    ax.yaxis.set_major_formatter(FuncFormatter(format_gb))
    ax.grid(True, axis="y", color="#d8d8d8", linewidth=0.65, alpha=0.55)
    ax.tick_params(axis="both", labelsize=14, direction="in", width=1.15)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.23),
        ncol=7,
        frameon=False,
        fontsize=10.8,
        handlelength=1.2,
        columnspacing=0.9,
        handletextpad=0.35,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
        spine.set_color("#5f5f5f")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot erbac+sift actual memory usage by memory budget.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    points = load_points(args.input.resolve())
    plot(points, args.output.resolve())
    csv_path = write_plot_csv(points, args.output.resolve())
    for method, by_budget in points.items():
        print(DISPLAY_METHOD_NAME.get(method, method))
        for budget in BUDGETS:
            point = by_budget.get(budget)
            if point is None:
                print(f"  budget={budget}: missing")
            else:
                print(
                    f"  budget={budget}: {point.total_gb:.3f} GB "
                    f"(source memory={point.source_memory_ratio:g})"
                )
    print(f"Saved figure to {args.output.resolve()}")
    if args.output.suffix.lower() != ".pdf":
        print(f"Saved figure to {args.output.resolve().with_suffix('.pdf')}")
    print(f"Saved plot data to {csv_path}")


if __name__ == "__main__":
    main()
