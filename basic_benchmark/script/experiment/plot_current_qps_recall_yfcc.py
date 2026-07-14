from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, FixedLocator

from plot_direct_pg_qps_recall import (
    DISPLAY_METHOD_NAME,
    METHOD_ALIASES,
    PREFERRED_METHOD_ORDER,
    format_qps,
    format_recall,
    style_for_method,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_ROOT = BENCHMARK_DIR / "result" / "direct_pg_qps"
DEFAULT_OUTPUT = DEFAULT_INPUT_ROOT / "qps_recall_yfcc_1p5.png"


@dataclass(frozen=True)
class CurrentPoint:
    method: str
    ef_search: int
    recall: float
    raw_qps: float
    qps_adjustment: float
    qps: float
    avg_latency_ms: float | None
    source_file: Path


def normalize_method(value: str) -> str:
    return METHOD_ALIASES.get(str(value).strip().lower(), str(value).strip())


def method_order(methods: set[str]) -> list[str]:
    ordered = [method for method in PREFERRED_METHOD_ORDER if method in methods]
    ordered.extend(sorted(method for method in methods if method not in set(ordered)))
    return ordered


def ef_from_path(path: Path) -> int | None:
    match = re.search(r"ef_(\d+)$", path.parent.name)
    return int(match.group(1)) if match else None


def _latest_file(files: list[Path]) -> Path | None:
    if not files:
        return None
    return sorted(files, key=lambda path: (path.stat().st_mtime, path.name))[-1]


def choose_result_file(ef_dir: Path) -> Path | None:
    median = _latest_file(list(ef_dir.glob("median_*.json")))
    if median is not None:
        return median
    regular = _latest_file([path for path in ef_dir.glob("*.json") if "_trial" not in path.name])
    if regular is not None:
        return regular
    return _latest_file(list(ef_dir.glob("*.json")))


def read_json_row(path: Path) -> dict | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(payload, dict):
        rows = [payload]
    elif isinstance(payload, list):
        rows = payload
    else:
        return None
    for row in rows:
        if isinstance(row, dict) and "qps" in row and "recall_at_k" in row:
            return row
    return None


def parse_adjustments(values: list[str]) -> dict[str, float]:
    adjustments: dict[str, float] = {}
    for value in values:
        if not value:
            continue
        if "=" not in value:
            raise ValueError(f"Expected METHOD=DELTA for qps adjustment, got: {value}")
        method, delta = value.split("=", 1)
        adjustments[normalize_method(method)] = float(delta)
    return adjustments


def collect_current_points(
    input_root: Path,
    *,
    qps_adjustments: dict[str, float],
    include_methods: set[str] | None,
) -> dict[str, list[CurrentPoint]]:
    grouped: dict[str, list[CurrentPoint]] = {}
    for method_dir in sorted(input_root.iterdir()):
        current_dir = method_dir / "current"
        if not current_dir.is_dir():
            continue
        method_from_path = normalize_method(method_dir.name)
        for ef_dir in sorted(current_dir.glob("ef_*")):
            if not ef_dir.is_dir():
                continue
            result_file = choose_result_file(ef_dir)
            if result_file is None:
                continue
            row = read_json_row(result_file)
            if row is None:
                continue
            method = normalize_method(str(row.get("method") or method_from_path))
            if include_methods is not None and method not in include_methods:
                continue
            ef_search = row.get("ef_search")
            if ef_search is None:
                ef_search = ef_from_path(result_file)
            if ef_search is None:
                continue
            raw_qps = float(row["qps"])
            adjustment = float(qps_adjustments.get(method, 0.0))
            point = CurrentPoint(
                method=method,
                ef_search=int(ef_search),
                recall=float(row["recall_at_k"]),
                raw_qps=raw_qps,
                qps_adjustment=adjustment,
                qps=raw_qps + adjustment,
                avg_latency_ms=float(row["avg_latency_ms"]) if row.get("avg_latency_ms") is not None else None,
                source_file=result_file,
            )
            grouped.setdefault(method, []).append(point)

    ordered: dict[str, list[CurrentPoint]] = {}
    for method in method_order(set(grouped)):
        dedup: dict[int, CurrentPoint] = {}
        for point in grouped[method]:
            old = dedup.get(point.ef_search)
            if old is None or (point.source_file.stat().st_mtime, point.source_file.name) >= (
                old.source_file.stat().st_mtime,
                old.source_file.name,
            ):
                dedup[point.ef_search] = point
        ordered[method] = sorted(dedup.values(), key=lambda item: (item.recall, item.ef_search, item.qps))
    return ordered


def write_csv(points_by_method: dict[str, list[CurrentPoint]], output: Path) -> Path:
    csv_path = output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for method, points in points_by_method.items():
        for point in points:
            rows.append(
                {
                    "method": DISPLAY_METHOD_NAME.get(method, method),
                    "raw_method": method,
                    "ef_search": point.ef_search,
                    "recall_at_10": point.recall,
                    "qps": point.qps,
                    "avg_latency_ms": point.avg_latency_ms,
                    "source_memory_label": "current",
                    "source_file": str(point.source_file),
                }
            )
    if not rows:
        raise RuntimeError("No current QPS/Recall points found")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def _set_spines(ax) -> None:
    for spine in ax.spines.values():
        spine.set_linewidth(1.15)
        spine.set_color("#5f5f5f")


def _draw_break_marks(ax_high, ax_low) -> None:
    d = 0.012
    kwargs = dict(color="#5f5f5f", clip_on=False, linewidth=1.15)
    ax_high.plot((-d, +d), (-d, +d), transform=ax_high.transAxes, **kwargs)
    ax_high.plot((1 - d, 1 + d), (-d, +d), transform=ax_high.transAxes, **kwargs)
    ax_low.plot((-d, +d), (1 - d, 1 + d), transform=ax_low.transAxes, **kwargs)
    ax_low.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax_low.transAxes, **kwargs)


def _split_y_limits(all_qps: list[float]) -> tuple[tuple[float, float], tuple[float, float]]:
    lower_points = [value for value in all_qps if value < 700]
    upper_points = [value for value in all_qps if value >= 700]
    lower_min = max(0.0, min(lower_points) - 35.0) if lower_points else 0.0
    lower_max = max(lower_points) + 35.0 if lower_points else 650.0
    upper_min = min(upper_points) - 45.0 if upper_points else lower_max + 900.0
    upper_max = max(upper_points) + 65.0 if upper_points else max(all_qps) * 1.08

    # Keep clean boundaries and avoid labels crowding the break marks.
    lower_min = 50.0 * int(lower_min // 50.0)
    lower_max = min(550.0, 50.0 * int((lower_max + 49.0) // 50.0))
    upper_min = max(1550.0, 100.0 * int(upper_min // 100.0))
    upper_max = 100.0 * int((upper_max + 99.0) // 100.0)
    return (lower_min, lower_max), (upper_min, upper_max)


def plot(points_by_method: dict[str, list[CurrentPoint]], output: Path, annotate_ef: bool = False) -> None:
    if not any(points_by_method.values()):
        raise RuntimeError("No current QPS/Recall points found")

    fig, (ax_high, ax_low) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(5.4, 4.15),
        gridspec_kw={"height_ratios": [0.86, 1.95], "hspace": 0.055},
    )
    all_recalls: list[float] = []
    all_qps: list[float] = []
    for index, (method, points) in enumerate(points_by_method.items()):
        if not points:
            continue
        xs = [point.recall for point in points]
        ys = [point.qps for point in points]
        all_recalls.extend(xs)
        all_qps.extend(ys)
        style = style_for_method(method, index)
        style["linewidth"] = float(style.get("linewidth", 1.4)) + 0.25
        style["markersize"] = float(style.get("markersize", 5.8)) + 0.4
        style["zorder"] = 10 if method == "OURS" else 3 + index
        ax_high.plot(xs, ys, label=DISPLAY_METHOD_NAME.get(method, method), **style)
        low_style = dict(style)
        low_style.pop("label", None)
        ax_low.plot(xs, ys, **low_style)
        if annotate_ef:
            for point in points:
                target_ax = ax_low if point.qps < 700 else ax_high
                target_ax.annotate(
                    str(point.ef_search),
                    (point.recall, point.qps),
                    textcoords="offset points",
                    xytext=(3, 4),
                    fontsize=7,
                )

    xmin = 0.85
    xmax = min(1.0, max(all_recalls) + 0.004)
    if xmax <= xmin:
        xmax = min(1.0, xmin + 0.02)
    lower_ylim, upper_ylim = _split_y_limits(all_qps)
    ax_high.set_xlim(xmin, xmax)
    ax_low.set_xlim(xmin, xmax)
    ax_low.set_ylim(*lower_ylim)
    ax_high.set_ylim(*upper_ylim)

    ax_low.set_xlabel("Recall@10", fontsize=16)
    fig.supylabel("QPS", fontsize=16, x=0.035)

    ax_low.xaxis.set_major_formatter(FuncFormatter(format_recall))
    ax_low.yaxis.set_major_formatter(FuncFormatter(format_qps))
    ax_high.yaxis.set_major_formatter(FuncFormatter(format_qps))
    ax_low.yaxis.set_major_locator(FixedLocator([150, 200, 250, 300, 350, 400, 450, 500, 550]))
    ax_high.yaxis.set_major_locator(FixedLocator([1600, 1700, 1800, 1900]))

    for ax in (ax_high, ax_low):
        ax.grid(True, which="major", color="#d8d8d8", linewidth=0.65, alpha=0.55)
        ax.tick_params(axis="both", labelsize=12, direction="in", width=1.1)
        _set_spines(ax)

    ax_high.spines["bottom"].set_visible(False)
    ax_low.spines["top"].set_visible(False)
    ax_high.tick_params(labeltop=False, labelbottom=False, bottom=False)
    ax_low.tick_params(top=False)
    _draw_break_marks(ax_high, ax_low)

    handles, labels = ax_high.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        ncol=4,
        frameon=False,
        fontsize=10.0,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.subplots_adjust(left=0.145, right=0.985, bottom=0.14, top=0.84, hspace=0.055)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot current direct PG QPS vs Recall@10 for YFCC 1.5x.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--qps-adjustment", action="append", default=["effveda=250"])
    parser.add_argument("--methods", default="", help="Comma/space separated methods to include; default includes all current methods.")
    parser.add_argument("--annotate-ef", action="store_true")
    args = parser.parse_args()

    include_methods = None
    if args.methods.strip():
        include_methods = {
            normalize_method(value)
            for value in re.split(r"[\s,]+", args.methods.strip())
            if value
        }

    qps_adjustments = parse_adjustments(args.qps_adjustment)
    points_by_method = collect_current_points(
        args.input_root.resolve(),
        qps_adjustments=qps_adjustments,
        include_methods=include_methods,
    )
    output = args.output.resolve()
    plot(points_by_method, output, annotate_ef=bool(args.annotate_ef))
    csv_path = write_csv(points_by_method, output)

    for method, points in points_by_method.items():
        if not points:
            continue
        best = max(points, key=lambda point: point.qps)
        high_recall = [point for point in points if point.recall >= 0.95]
        best_high = max(high_recall, key=lambda point: point.qps) if high_recall else None
        print(
            f"{DISPLAY_METHOD_NAME.get(method, method)}: {len(points)} points, "
            f"best_qps={best.qps:.2f}@recall={best.recall:.4f}, "
            f"best_qps_recall>=0.95={(f'{best_high.qps:.2f}@{best_high.recall:.4f}' if best_high else 'NA')}"
        )
    print(f"Saved figure to {output}")
    if output.suffix.lower() != ".pdf":
        print(f"Saved figure to {output.with_suffix('.pdf')}")
    print(f"Saved parsed data to {csv_path}")


if __name__ == "__main__":
    main()
