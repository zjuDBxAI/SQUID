from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = SCRIPT_DIR / "memory_query_time.png"

DISPLAY_METHOD_NAME = {
    "AnonySys": "HONEYBEE",
    "OURS": "SQUID",
    "effveda": "EFFVEDA",
    "veda": "VEDA",
    "RLS": "RLS",
    "ROLE": "ROLE",
    "QDTree": "HQI",
}

METHOD_ORDER = ("OURS", "AnonySys", "ROLE", "effveda", "veda", "RLS", "QDTree")
EMPHASIS_SINGLE_POINT_METHODS = {"RLS", "ROLE", "QDTree"}
METHOD_STYLES = {
    "OURS": {"color": "#2F6B9A", "marker": "o", "linewidth": 2.05, "markersize": 6.3, "zorder": 8},
    "AnonySys": {"color": "#C44E52", "marker": "v", "linewidth": 1.65, "markersize": 5.8, "zorder": 5},
    "ROLE": {"color": "#4C4C4C", "marker": "X", "linewidth": 1.55, "markersize": 8.4, "zorder": 9},
    "effveda": {"color": "#E69F00", "marker": "s", "linewidth": 1.55, "markersize": 5.8, "zorder": 4},
    "veda": {"color": "#7B3294", "marker": "D", "linewidth": 1.55, "markersize": 5.6, "zorder": 4},
    "RLS": {"color": "#009E73", "marker": "P", "linewidth": 1.45, "markersize": 8.2, "zorder": 9},
    "QDTree": {"color": "#8C6D31", "marker": "^", "linewidth": 1.45, "markersize": 8.2, "zorder": 9},
}

# Query time is in milliseconds. Missing entries mean the method was not run at that memory ratio.
DATA = {
    "AnonySys": {
        1.0: 8.2,
        1.5: 2.7,
        2.0: 1.6,
        2.5: 1.4,
        3.0: 1.1,
        3.5: 0.9,
    },
    "OURS": {
        1.0: 2.4,
        1.5: 1.5,
        2.0: 1.3,
        2.5: 0.9,
        3.0: 0.9,
        3.5: 1.0,
    },
    "effveda": {
        1.0: 2.7,
        1.5: 1.7,
        2.0: 1.5,
        2.5: 1.2,
        3.0: 1.0,
        3.5: 1.0,
    },
    "veda": {
        1.0: 2.6,
        1.5: 2.0,
        2.0: 1.7,
        2.5: 1.3,
        3.0: 1.2,
        3.5: 0.9,
    },
    "RLS": {
        1.0: 8.4,
    },
    "QDTree": {
        1.0: 3.1,
    },
    "ROLE": {
        3.5: 0.9,
    },
}


def _format_memory(value: float, _pos) -> str:
    if abs(value - round(value)) < 1e-9:
        return f"{int(round(value))}x"
    return f"{value:g}x"


def _format_time(value: float, _pos) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def _style(method: str) -> dict[str, object]:
    style = dict(METHOD_STYLES[method])
    style.setdefault("alpha", 0.84)
    style.setdefault("markeredgecolor", "white")
    style.setdefault("markeredgewidth", 0.55)
    if method in EMPHASIS_SINGLE_POINT_METHODS:
        style["alpha"] = 0.98
        style["markeredgecolor"] = "#1f1f1f"
        style["markeredgewidth"] = 0.95
    style["linestyle"] = "-"
    return style


def write_csv(output: Path) -> Path:
    csv_path = output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["method", "display_method", "memory_ratio", "query_time_ms"])
        writer.writeheader()
        for method in METHOD_ORDER:
            for memory, query_time in sorted(DATA.get(method, {}).items()):
                writer.writerow({
                    "method": method,
                    "display_method": DISPLAY_METHOD_NAME.get(method, method),
                    "memory_ratio": memory,
                    "query_time_ms": query_time,
                })
    return csv_path


def plot(output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.6, 3.55))

    for method in METHOD_ORDER:
        series = DATA.get(method, {})
        if not series:
            continue
        xs = [memory for memory, _query_time in sorted(series.items())]
        ys = [query_time for _memory, query_time in sorted(series.items())]
        label = DISPLAY_METHOD_NAME.get(method, method)
        style = _style(method)
        ax.plot(xs, ys, label=label, **style)

    ax.set_xlim(0.9, 3.6)
    ax.set_ylim(0.55, 8.9)
    ax.xaxis.set_major_locator(MultipleLocator(0.5))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_memory))
    ax.yaxis.set_major_formatter(FuncFormatter(_format_time))
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.65, alpha=0.55)
    ax.tick_params(axis="both", labelsize=10.5, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#5f5f5f")

    ax.set_xlabel("Memory Budget", fontsize=13)
    ax.set_ylabel("Query Time (ms)", fontsize=13)

    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.52, 1.035),
        ncol=4,
        frameon=False,
        fontsize=9.4,
        handlelength=1.8,
        columnspacing=1.05,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.9))
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot query time under different memory budgets.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output image path.")
    args = parser.parse_args()

    output = Path(args.output).resolve()
    plot(output)
    csv_path = write_csv(output)
    print(f"Saved figure to {output}")
    if output.suffix.lower() != ".pdf":
        print(f"Saved figure to {output.with_suffix('.pdf')}")
    print(f"Saved data to {csv_path}")


if __name__ == "__main__":
    main()
