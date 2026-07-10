from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MultipleLocator


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
DEFAULT_LOG_DIR = BENCHMARK_DIR / "efs_logs" / "treebase_sift"
DEFAULT_OUTPUT = DEFAULT_LOG_DIR / "ex1_recall_time.png"
DEFAULT_MIN_RECALL = 0.8
DEFAULT_RECALL_ANCHORS = (0.80, 0.85, 0.90, 0.92, 0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995, 0.999)

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


@dataclass(frozen=True)
class Point:
    method: str
    ef_search: int
    recall: float
    query_time_ms: float
    log_file: Path


@dataclass(frozen=True)
class CurvePoint:
    method: str
    recall: float
    query_time_ms: float
    ef_search: str
    source_file: str
    interpolated: bool


def _method_from_path(path: Path) -> str | None:
    name = path.name
    parent = path.parent.name
    if parent and parent != DEFAULT_LOG_DIR.name:
        return parent
    if name.startswith("AnonySys_"):
        return "AnonySys"
    if name.startswith("QDTree_"):
        return "QDTree"
    if name.startswith("RLS_"):
        return "RLS"
    if name.startswith("ROLE_"):
        return "ROLE"
    if name.startswith("OURS_") or "OURS" in name:
        return "OURS"
    if "effveda" in name.lower():
        return "effveda"
    if "veda" in name.lower():
        return "veda"
    return None


def _ef_from_name(name: str) -> int | None:
    match = re.search(r"(?:^|_)(?:efs|ef)(\d+)(?=_|$)", name)
    return int(match.group(1)) if match else None


def _timestamp_from_name(name: str) -> int | None:
    match = re.search(r"_(\d{8}_\d{6})(?=\.log$)", name)
    return int(match.group(1).replace("_", "")) if match else None


def _last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text)
    return float(values[-1]) if values else None


def _log_sort_key(path: Path) -> tuple[int, float]:
    timestamp = _timestamp_from_name(path.name)
    return (timestamp if timestamp is not None else 0, path.stat().st_mtime)


def _parse_log(path: Path) -> Point | None:
    method = _method_from_path(path)
    ef_search = _ef_from_name(path.name)
    if method is None or ef_search is None:
        return None

    text = path.read_text(encoding="utf-8", errors="ignore")
    recall = _last_float(r"Average Recall:\s*([0-9]*\.?[0-9]+)", text)
    query_time_seconds = _last_float(r"Average Query Time:\s*([0-9]*\.?[0-9]+)\s*seconds", text)
    if recall is None or query_time_seconds is None:
        return None

    return Point(
        method=method,
        ef_search=int(ef_search),
        recall=float(recall),
        query_time_ms=float(query_time_seconds) * 1000.0,
        log_file=path,
    )


def _method_order(methods: set[str]) -> list[str]:
    ordered = [method for method in PREFERRED_METHOD_ORDER if method in methods]
    ordered.extend(sorted(method for method in methods if method not in set(ordered)))
    return ordered


def _dedupe_and_sort(points: list[Point]) -> list[Point]:
    best_by_recall: dict[float, Point] = {}
    for point in points:
        key = round(float(point.recall), 6)
        old = best_by_recall.get(key)
        if old is None or (point.query_time_ms, point.ef_search) < (old.query_time_ms, old.ef_search):
            best_by_recall[key] = point
    return sorted(best_by_recall.values(), key=lambda item: (item.recall, item.query_time_ms, item.ef_search))


def _monotone_frontier(points: list[Point], *, min_recall: float) -> list[Point]:
    filtered = [point for point in points if point.recall >= min_recall]
    candidates = _dedupe_and_sort(filtered)
    frontier_reversed: list[Point] = []
    best_time = float("inf")
    for point in reversed(candidates):
        if point.query_time_ms < best_time:
            frontier_reversed.append(point)
            best_time = point.query_time_ms
    frontier = list(reversed(frontier_reversed))

    monotone: list[Point] = []
    max_time = -float("inf")
    for point in frontier:
        if point.query_time_ms > max_time:
            monotone.append(point)
            max_time = point.query_time_ms
    return monotone


def _as_curve_point(point: Point) -> CurvePoint:
    return CurvePoint(
        method=point.method,
        recall=point.recall,
        query_time_ms=point.query_time_ms,
        ef_search=str(point.ef_search),
        source_file=str(point.log_file),
        interpolated=False,
    )


def _interpolate_between(left: Point, right: Point, recall: float) -> CurvePoint:
    if abs(right.recall - left.recall) < 1e-12:
        return _as_curve_point(left if left.query_time_ms <= right.query_time_ms else right)
    ratio = (float(recall) - left.recall) / (right.recall - left.recall)
    query_time = left.query_time_ms + ratio * (right.query_time_ms - left.query_time_ms)
    return CurvePoint(
        method=left.method,
        recall=float(recall),
        query_time_ms=float(query_time),
        ef_search=f"{left.ef_search}-{right.ef_search}",
        source_file=f"{left.log_file}|{right.log_file}",
        interpolated=True,
    )


def _sample_at_anchors(points: list[Point], anchors: tuple[float, ...]) -> list[CurvePoint]:
    if not points:
        return []
    selected: list[CurvePoint] = []

    def add(point: CurvePoint) -> None:
        key = (round(point.recall, 9), round(point.query_time_ms, 9), point.ef_search)
        existing = {
            (round(item.recall, 9), round(item.query_time_ms, 9), item.ef_search)
            for item in selected
        }
        if key not in existing:
            selected.append(point)

    # Keep real endpoints even when they do not land on the shared anchors.
    # This preserves methods whose first valid recall is already above 0.8.
    add(_as_curve_point(points[0]))

    for anchor in anchors:
        if anchor < points[0].recall - 1e-12 or anchor > points[-1].recall + 1e-12:
            continue
        exact = next((point for point in points if abs(point.recall - anchor) <= 1e-9), None)
        if exact is not None:
            add(_as_curve_point(exact))
            continue
        for left, right in zip(points, points[1:]):
            if left.recall <= anchor <= right.recall:
                add(_interpolate_between(left, right, anchor))
                break

    add(_as_curve_point(points[-1]))
    return sorted(selected, key=lambda item: (item.recall, item.query_time_ms, item.ef_search))


def load_points(
    log_dir: Path,
    *,
    min_recall: float,
    anchors: tuple[float, ...],
    keep_raw_points: bool = False,
) -> dict[str, list[CurvePoint]]:
    latest_by_method_ef: dict[tuple[str, int], Point] = {}
    for path in sorted(log_dir.rglob("*.log")):
        point = _parse_log(path)
        if point is None:
            continue
        key = (point.method, point.ef_search)
        old = latest_by_method_ef.get(key)
        if old is None or _log_sort_key(point.log_file) >= _log_sort_key(old.log_file):
            latest_by_method_ef[key] = point

    raw_grouped: dict[str, list[Point]] = {}
    for point in latest_by_method_ef.values():
        raw_grouped.setdefault(point.method, []).append(point)

    sampled: dict[str, list[CurvePoint]] = {}
    for method, points in raw_grouped.items():
        if keep_raw_points:
            sampled[method] = [
                _as_curve_point(point)
                for point in _dedupe_and_sort(
                    [point for point in points if point.recall >= min_recall]
                )
            ]
            continue
        frontier = _monotone_frontier(points, min_recall=min_recall)
        sampled[method] = _sample_at_anchors(frontier, anchors)
    return {method: sampled.get(method, []) for method in _method_order(set(sampled))}


def write_csv(points_by_method: dict[str, list[CurvePoint]], output: Path) -> Path | None:
    rows = []
    for method, points in points_by_method.items():
        for point in points:
            rows.append({
                "method": DISPLAY_METHOD_NAME.get(point.method, point.method),
                "raw_method": point.method,
                "ef_search": point.ef_search,
                "recall": point.recall,
                "query_time_ms": point.query_time_ms,
                "interpolated": point.interpolated,
                "source_file": point.source_file,
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


def _style_for_method(method: str, index: int) -> dict[str, object]:
    style = dict(STYLE_CYCLE[index % len(STYLE_CYCLE)])
    style.update(METHOD_STYLES.get(method, {}))
    style.setdefault("linewidth", 1.0)
    style.setdefault("markersize", 5.8)
    style.setdefault("alpha", 0.78)
    style.setdefault("markeredgecolor", "white")
    style.setdefault("markeredgewidth", 0.55)
    style["linestyle"] = "-"
    return style


def _format_recall_tick(value: float, _pos) -> str:
    if value >= 0.995 - 1e-9:
        return f"{value:.3f}"
    return f"{value:.2f}"


def _format_time_tick(value: float, _pos) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def plot(points_by_method: dict[str, list[CurvePoint]], output: Path, *, annotate_ef: bool = False, min_recall: float = DEFAULT_MIN_RECALL, anchors: tuple[float, ...] = DEFAULT_RECALL_ANCHORS) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 3.8))
    plotted = 0
    all_recalls: list[float] = []
    for index, (method, points) in enumerate(points_by_method.items()):
        if not points:
            continue
        plotted += 1
        xs = [point.recall for point in points]
        ys = [point.query_time_ms for point in points]
        all_recalls.extend(xs)
        style = _style_for_method(method, index)
        style["zorder"] = 10 if method == "OURS" else 3 + index
        ax.plot(xs, ys, label=DISPLAY_METHOD_NAME.get(method, method), **style)
        if annotate_ef:
            for point in points:
                ax.annotate(
                    str(point.ef_search),
                    (point.recall, point.query_time_ms),
                    textcoords="offset points",
                    xytext=(3, 4),
                    fontsize=7,
                )

    if plotted == 0:
        raise RuntimeError(f"No valid benchmark log points were found under {DEFAULT_LOG_DIR}")

    ax.set_xlabel("Recall@10", fontsize=13)
    ax.set_ylabel("Query Time (ms)", fontsize=13)
    ax.set_xlim(float(min_recall), min(1.0, max(all_recalls) + 0.004 if all_recalls else 1.0))
    ax.xaxis.set_major_formatter(FuncFormatter(_format_recall_tick))
    max_time = max((point.query_time_ms for points in points_by_method.values() for point in points), default=1.0)
    ax.set_ylim(0, max_time * 1.08)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_time_tick))
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.65, alpha=0.55)
    ax.tick_params(axis="both", labelsize=11, direction="in")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        ncol=4,
        frameon=False,
        fontsize=9.5,
    )
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color("#5f5f5f")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _parse_anchors(value: str) -> tuple[float, ...]:
    anchors = tuple(float(part.strip()) for part in value.split(",") if part.strip())
    if not anchors:
        raise argparse.ArgumentTypeError("at least one recall anchor is required")
    return tuple(sorted(set(anchors)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Ex1 recall-latency curves from treebase_sift ef-search logs.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory containing method subdirectories and .log files.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output image path.")
    parser.add_argument("--min-recall", type=float, default=DEFAULT_MIN_RECALL, help="Drop points below this recall.")
    parser.add_argument("--anchors", type=_parse_anchors, default=DEFAULT_RECALL_ANCHORS, help="Comma-separated recall anchors for aligned interpolation.")
    parser.add_argument("--keep-raw-points", action="store_true", help="Plot every valid raw EF point instead of the sampled Pareto frontier.")
    parser.add_argument("--annotate-ef", action="store_true", help="Annotate each point with its ef_search value or interpolation bracket.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    output = Path(args.output).resolve()
    anchors = tuple(anchor for anchor in tuple(args.anchors) if anchor >= float(args.min_recall))
    points_by_method = load_points(
        log_dir,
        min_recall=float(args.min_recall),
        anchors=anchors,
        keep_raw_points=bool(args.keep_raw_points),
    )
    plot(points_by_method, output, annotate_ef=bool(args.annotate_ef), min_recall=float(args.min_recall), anchors=anchors)
    csv_path = write_csv(points_by_method, output)

    for method, points in points_by_method.items():
        print(f"{DISPLAY_METHOD_NAME.get(method, method)}: {len(points)} points")
        for point in points:
            tag = "interp" if point.interpolated else f"ef={point.ef_search}"
            print(f"  recall={point.recall:.4f} query_time={point.query_time_ms:.3f} ms {tag}")
    print(f"Saved figure to {output}")
    if output.suffix.lower() != ".pdf":
        print(f"Saved figure to {output.with_suffix(chr(46) + chr(112) + chr(100) + chr(102))}")
    if csv_path is not None:
        print(f"Saved parsed data to {csv_path}")


if __name__ == "__main__":
    main()
