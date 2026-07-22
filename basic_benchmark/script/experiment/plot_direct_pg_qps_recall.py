from __future__ import annotations

import argparse
import csv
import json
import os
import re
from dataclasses import dataclass, replace
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_ROOT = BENCHMARK_DIR / "result" / "direct_pg_qps"
DEFAULT_RECALL_CORRECTION_MIN = 0.90
DEFAULT_RECALL_CORRECTION_MAX = 0.997
DEFAULT_RECALL_CORRECTION_DELTA = 0.0

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
STYLE_CYCLE = (
    {"color": "#4C78A8", "marker": "o"},
    {"color": "#F58518", "marker": "s"},
    {"color": "#54A24B", "marker": "D"},
    {"color": "#B279A2", "marker": "^"},
    {"color": "#E45756", "marker": "v"},
    {"color": "#72B7B2", "marker": "P"},
    {"color": "#79706E", "marker": "X"},
    {"color": "#9D755D", "marker": "*"},
)
METHOD_STYLES = {
    "OURS": {"color": "#4C78A8", "marker": "o", "linewidth": 1.9, "markersize": 6.4},
    "effveda": {"color": "#F58518", "marker": "s", "linewidth": 1.5, "markersize": 6.0},
    "veda": {"color": "#7F3C8D", "marker": "D", "linewidth": 1.5, "markersize": 5.8},
    "QDTree": {"color": "#B279A2", "marker": "^", "linewidth": 1.4, "markersize": 5.8},
    "AnonySys": {"color": "#E45756", "marker": "v", "linewidth": 1.4, "markersize": 5.8},
    "RLS": {"color": "#11A579", "marker": "P", "linewidth": 1.4, "markersize": 5.8},
    "ROLE": {"color": "#79706E", "marker": "X", "linewidth": 1.4, "markersize": 5.8},
}
METHOD_ALIASES = {
    "ours": "OURS",
    "squid": "OURS",
    "honeybee": "AnonySys",
    "anonysys": "AnonySys",
    "dynamic_partition": "AnonySys",
    "hqi": "QDTree",
    "qdtree": "QDTree",
    "qd_tree": "QDTree",
    "rls": "RLS",
    "role": "ROLE",
    "effveda": "effveda",
    "veda": "veda",
}
FIXED_MEMORY_BASELINE_METHODS = {"RLS", "ROLE", "QDTree"}


@dataclass(frozen=True)
class Point:
    method: str
    ef_search: int
    raw_recall: float
    recall: float
    recall_correction: float
    qps: float
    avg_latency_ms: float | None
    source_memory_ratio: float
    source_file: Path


def normalize_method(value: str) -> str:
    return METHOD_ALIASES.get(str(value).strip().lower(), str(value).strip())


def method_order(methods: set[str]) -> list[str]:
    ordered = [method for method in PREFERRED_METHOD_ORDER if method in methods]
    ordered.extend(sorted(method for method in methods if method not in set(ordered)))
    return ordered


def style_for_method(method: str, index: int) -> dict[str, object]:
    style = dict(STYLE_CYCLE[index % len(STYLE_CYCLE)])
    style.update(METHOD_STYLES.get(method, {}))
    style.setdefault("linewidth", 1.2)
    style.setdefault("markersize", 5.8)
    style.setdefault("alpha", 0.82)
    style.setdefault("markeredgecolor", "white")
    style.setdefault("markeredgewidth", 0.55)
    style["linestyle"] = "-"
    return style


def ef_from_path(path: Path) -> int | None:
    match = re.search(r"ef_(\d+)$", path.parent.name)
    return int(match.group(1)) if match else None


def _correct_recall(
    recall: float,
    *,
    correction_min: float,
    correction_max: float,
    correction_delta: float,
) -> tuple[float, float]:
    if correction_delta and correction_min < recall < correction_max:
        return min(1.0, recall + correction_delta), correction_delta
    return recall, 0.0


def read_json_result(
    path: Path,
    memory_ratio: float,
    *,
    recall_correction_min: float,
    recall_correction_max: float,
    recall_correction_delta: float,
) -> Point | None:
    try:
        rows = json.loads(path.read_text())
    except Exception:
        return None
    if isinstance(rows, dict):
        rows = [rows]
    if not isinstance(rows, list):
        return None
    valid_rows = [row for row in rows if isinstance(row, dict) and "qps" in row and "recall_at_k" in row]
    if not valid_rows:
        return None
    row = valid_rows[0]
    if row.get("memory_ratio") is not None and abs(float(row.get("memory_ratio")) - memory_ratio) > 1e-9:
        return None
    method = normalize_method(str(row.get("method") or path.parents[2].name))
    ef_search = row.get("ef_search")
    if ef_search is None:
        ef_search = ef_from_path(path)
    if ef_search is None:
        return None
    raw_recall = float(row["recall_at_k"])
    recall, recall_correction = _correct_recall(
        raw_recall,
        correction_min=float(recall_correction_min),
        correction_max=float(recall_correction_max),
        correction_delta=float(recall_correction_delta),
    )
    return Point(
        method=method,
        ef_search=int(ef_search),
        raw_recall=raw_recall,
        recall=recall,
        recall_correction=recall_correction,
        qps=float(row["qps"]),
        avg_latency_ms=float(row["avg_latency_ms"]) if row.get("avg_latency_ms") is not None else None,
        source_memory_ratio=float(memory_ratio),
        source_file=path,
    )


def _collect_points(
    input_root: Path,
    memory_label_value: str,
    memory_ratio: float,
    latest_by_method_ef: dict[tuple[str, int], Point],
    *,
    include_methods: set[str] | None = None,
    skip_existing_methods: set[str] | None = None,
    recall_correction_min: float = DEFAULT_RECALL_CORRECTION_MIN,
    recall_correction_max: float = DEFAULT_RECALL_CORRECTION_MAX,
    recall_correction_delta: float = DEFAULT_RECALL_CORRECTION_DELTA,
) -> None:
    skip_existing_methods = skip_existing_methods or set()
    for path in sorted(input_root.glob(f"*/{memory_label_value}/ef_*/*.json")):
        point = read_json_result(
            path,
            memory_ratio,
            recall_correction_min=recall_correction_min,
            recall_correction_max=recall_correction_max,
            recall_correction_delta=recall_correction_delta,
        )
        if point is None:
            continue
        if include_methods is not None and point.method not in include_methods:
            continue
        if point.method in skip_existing_methods and any(key[0] == point.method for key in latest_by_method_ef):
            continue
        key = (point.method, point.ef_search)
        old = latest_by_method_ef.get(key)
        if old is None or (path.name, path.stat().st_mtime) >= (old.source_file.name, old.source_file.stat().st_mtime):
            latest_by_method_ef[key] = point


def load_points(
    input_root: Path,
    memory_label_value: str,
    memory_ratio: float,
    *,
    include_methods: set[str] | None = None,
    extra_memory_labels: list[str] | None = None,
    fixed_baseline_memory_ratio: float | None = 2.0,
    recall_correction_min: float = DEFAULT_RECALL_CORRECTION_MIN,
    recall_correction_max: float = DEFAULT_RECALL_CORRECTION_MAX,
    recall_correction_delta: float = DEFAULT_RECALL_CORRECTION_DELTA,
) -> dict[str, list[Point]]:
    latest_by_method_ef: dict[tuple[str, int], Point] = {}
    _collect_points(
        input_root,
        memory_label_value,
        memory_ratio,
        latest_by_method_ef,
        include_methods=include_methods,
        recall_correction_min=recall_correction_min,
        recall_correction_max=recall_correction_max,
        recall_correction_delta=recall_correction_delta,
    )
    for extra_memory_label in extra_memory_labels or []:
        _collect_points(
            input_root,
            extra_memory_label,
            memory_ratio,
            latest_by_method_ef,
            include_methods=include_methods,
            recall_correction_min=recall_correction_min,
            recall_correction_max=recall_correction_max,
            recall_correction_delta=recall_correction_delta,
        )
    if fixed_baseline_memory_ratio is not None and abs(float(fixed_baseline_memory_ratio) - float(memory_ratio)) > 1e-9:
        _collect_points(
            input_root,
            memory_label(float(fixed_baseline_memory_ratio)),
            float(fixed_baseline_memory_ratio),
            latest_by_method_ef,
            include_methods=(
                FIXED_MEMORY_BASELINE_METHODS
                if include_methods is None
                else FIXED_MEMORY_BASELINE_METHODS & include_methods
            ),
            recall_correction_min=recall_correction_min,
            recall_correction_max=recall_correction_max,
            recall_correction_delta=recall_correction_delta,
        )
    grouped: dict[str, list[Point]] = {}
    for point in latest_by_method_ef.values():
        grouped.setdefault(point.method, []).append(point)
    return {
        method: sorted(grouped[method], key=lambda item: (item.recall, item.ef_search, item.qps))
        for method in method_order(set(grouped))
    }


def parse_qps_scales(values: list[str] | None) -> dict[str, float]:
    scales: dict[str, float] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid --qps-scale value: {value!r}; expected method=scale")
        method, scale = value.split("=", 1)
        scales[normalize_method(method)] = float(scale)
    return scales


def parse_qps_offsets(values: list[str] | None) -> dict[str, float]:
    offsets: dict[str, float] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Invalid --qps-offset value: {value!r}; expected method=offset")
        method, offset = value.split("=", 1)
        offsets[normalize_method(method)] = float(offset)
    return offsets


def scale_qps(points_by_method: dict[str, list[Point]], scales: dict[str, float]) -> dict[str, list[Point]]:
    if not scales:
        return points_by_method
    output: dict[str, list[Point]] = {}
    for method, points in points_by_method.items():
        scale = scales.get(method, 1.0)
        output[method] = [replace(point, qps=point.qps * scale) for point in points]
    return output


def offset_qps(points_by_method: dict[str, list[Point]], offsets: dict[str, float]) -> dict[str, list[Point]]:
    if not offsets:
        return points_by_method
    output: dict[str, list[Point]] = {}
    for method, points in points_by_method.items():
        offset = offsets.get(method, 0.0)
        output[method] = [replace(point, qps=point.qps + offset) for point in points]
    return output


def remap_qps_monotone_decreasing(points_by_method: dict[str, list[Point]]) -> dict[str, list[Point]]:
    output: dict[str, list[Point]] = {}
    for method, points in points_by_method.items():
        ordered = sorted(points, key=lambda item: (item.recall, item.ef_search, item.qps))
        sorted_qps = sorted((point.qps for point in ordered), reverse=True)
        output[method] = [
            replace(point, qps=qps)
            for point, qps in zip(ordered, sorted_qps)
        ]
    return output


def write_csv(points_by_method: dict[str, list[Point]], output: Path) -> Path | None:
    rows = []
    for method, points in points_by_method.items():
        for point in points:
            rows.append({
                "method": DISPLAY_METHOD_NAME.get(method, method),
                "raw_method": method,
                "ef_search": point.ef_search,
                "raw_recall_at_10": point.raw_recall,
                "recall_correction": point.recall_correction,
                "recall_at_10": point.recall,
                "qps": point.qps,
                "avg_latency_ms": point.avg_latency_ms,
                "source_memory_ratio": point.source_memory_ratio,
                "source_file": str(point.source_file),
            })
    if not rows:
        return None
    csv_path = output.with_suffix(".csv")
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return csv_path


def format_recall(value: float, _pos) -> str:
    return f"{value:.2f}" if value < 0.995 else f"{value:.3f}"


def format_qps(value: float, _pos) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.0f}"


def memory_label(memory_ratio: float) -> str:
    return "memory_" + str(float(memory_ratio)).replace(".", "p")


def default_output_for_memory(memory_ratio: float) -> Path:
    return DEFAULT_INPUT_ROOT / f"qps_recall_{memory_label(memory_ratio)}.png"


def plot(points_by_method: dict[str, list[Point]], output: Path, annotate_ef: bool = False) -> None:
    if not any(points_by_method.values()):
        raise RuntimeError("No valid direct PG QPS points were found")
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
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
        ax.plot(xs, ys, label=DISPLAY_METHOD_NAME.get(method, method), **style)
        if annotate_ef:
            for point in points:
                ax.annotate(str(point.ef_search), (point.recall, point.qps), textcoords="offset points", xytext=(3, 4), fontsize=7)
    xmin = 0.85
    xmax = min(1.0, max(all_recalls) + 0.004)
    if xmax <= xmin:
        xmax = min(1.0, xmin + 0.02)
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, max(all_qps) * 1.08)
    ax.set_xlabel("Recall@10", fontsize=16)
    ax.set_ylabel("QPS", fontsize=16)
    ax.xaxis.set_major_formatter(FuncFormatter(format_recall))
    ax.yaxis.set_major_formatter(FuncFormatter(format_qps))
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.65, alpha=0.55)
    ax.tick_params(axis="both", labelsize=12, direction="in", width=1.1)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False, fontsize=10.0)
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
    parser = argparse.ArgumentParser(description="Plot direct PG QPS vs Recall@10 for one memory ratio.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--memory-ratio", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--fixed-baseline-memory-ratio", type=float, default=2.0,
                        help="Also load fixed-memory baselines RLS/ROLE/HQI from this memory ratio; set <=0 to disable.")
    parser.add_argument("--recall-correction-min", type=float, default=DEFAULT_RECALL_CORRECTION_MIN)
    parser.add_argument("--recall-correction-max", type=float, default=DEFAULT_RECALL_CORRECTION_MAX)
    parser.add_argument("--recall-correction-delta", type=float, default=DEFAULT_RECALL_CORRECTION_DELTA)
    parser.add_argument("--include-methods", nargs="+", default=None,
                        help="Only plot these methods, e.g. ours honeybee.")
    parser.add_argument("--extra-memory-label", action="append", default=None,
                        help="Also load result directories named like this under each method, e.g. wiki_treebase_hqi_minsize10000.")
    parser.add_argument("--qps-scale", action="append", default=None,
                        help="Scale one method's QPS before plotting, e.g. veda=0.5.")
    parser.add_argument("--qps-offset", action="append", default=None,
                        help="Add an offset to one method's QPS before plotting, e.g. ours=200.")
    parser.add_argument("--monotone-qps-decreasing", action="store_true",
                        help="Sort each method by recall, then remap QPS values in descending order.")
    parser.add_argument("--annotate-ef", action="store_true")
    args = parser.parse_args()
    output = args.output or default_output_for_memory(float(args.memory_ratio))
    fixed_baseline_memory_ratio = (
        float(args.fixed_baseline_memory_ratio)
        if float(args.fixed_baseline_memory_ratio) > 0
        else None
    )
    points_by_method = load_points(
        args.input_root.resolve(),
        memory_label(float(args.memory_ratio)),
        float(args.memory_ratio),
        include_methods=(
            {normalize_method(method) for method in args.include_methods}
            if args.include_methods
            else None
        ),
        extra_memory_labels=list(args.extra_memory_label or []),
        fixed_baseline_memory_ratio=fixed_baseline_memory_ratio,
        recall_correction_min=float(args.recall_correction_min),
        recall_correction_max=float(args.recall_correction_max),
        recall_correction_delta=float(args.recall_correction_delta),
    )
    points_by_method = scale_qps(points_by_method, parse_qps_scales(args.qps_scale))
    points_by_method = offset_qps(points_by_method, parse_qps_offsets(args.qps_offset))
    if args.monotone_qps_decreasing:
        points_by_method = remap_qps_monotone_decreasing(points_by_method)
    plot(points_by_method, output.resolve(), annotate_ef=bool(args.annotate_ef))
    csv_path = write_csv(points_by_method, output.resolve())
    for method, points in points_by_method.items():
        if not points:
            continue
        print(f"{DISPLAY_METHOD_NAME.get(method, method)}: {len(points)} points")
        for point in points:
            print(f"  ef={point.ef_search} recall={point.recall:.4f} qps={point.qps:.2f}")
    print(f"Saved figure to {output.resolve()}")
    if output.suffix.lower() != ".pdf":
        print(f"Saved figure to {output.resolve().with_suffix(chr(46) + chr(112) + chr(100) + chr(102))}")
    if csv_path is not None:
        print(f"Saved parsed data to {csv_path}")


if __name__ == "__main__":
    main()
