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
from matplotlib.ticker import FuncFormatter


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
DEFAULT_INPUT_ROOT = BENCHMARK_DIR / "result" / "direct_pg_qps"
DEFAULT_OUTPUT = DEFAULT_INPUT_ROOT / "qps_recall_memory_2p0.png"

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


@dataclass(frozen=True)
class Point:
    method: str
    ef_search: int
    recall: float
    qps: float
    avg_latency_ms: float | None
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


def read_json_result(path: Path, memory_ratio: float) -> Point | None:
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
    return Point(
        method=method,
        ef_search=int(ef_search),
        recall=float(row["recall_at_k"]),
        qps=float(row["qps"]),
        avg_latency_ms=float(row["avg_latency_ms"]) if row.get("avg_latency_ms") is not None else None,
        source_file=path,
    )


def load_points(input_root: Path, memory_label: str, memory_ratio: float) -> dict[str, list[Point]]:
    latest_by_method_ef: dict[tuple[str, int], Point] = {}
    for path in sorted(input_root.glob(f"*/{memory_label}/ef_*/*.json")):
        point = read_json_result(path, memory_ratio)
        if point is None:
            continue
        key = (point.method, point.ef_search)
        old = latest_by_method_ef.get(key)
        if old is None or (path.name, path.stat().st_mtime) >= (old.source_file.name, old.source_file.stat().st_mtime):
            latest_by_method_ef[key] = point
    grouped: dict[str, list[Point]] = {}
    for point in latest_by_method_ef.values():
        grouped.setdefault(point.method, []).append(point)
    return {
        method: sorted(grouped[method], key=lambda item: (item.recall, item.ef_search, item.qps))
        for method in method_order(set(grouped))
    }


def write_csv(points_by_method: dict[str, list[Point]], output: Path) -> Path | None:
    rows = []
    for method, points in points_by_method.items():
        for point in points:
            rows.append({
                "method": DISPLAY_METHOD_NAME.get(method, method),
                "raw_method": method,
                "ef_search": point.ef_search,
                "recall_at_10": point.recall,
                "qps": point.qps,
                "avg_latency_ms": point.avg_latency_ms,
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
    ax.set_xlabel("Recall@10", fontsize=13)
    ax.set_ylabel("QPS", fontsize=13)
    ax.xaxis.set_major_formatter(FuncFormatter(format_recall))
    ax.yaxis.set_major_formatter(FuncFormatter(format_qps))
    ax.grid(True, which="major", color="#d8d8d8", linewidth=0.65, alpha=0.55)
    ax.tick_params(axis="both", labelsize=11, direction="in")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.22), ncol=4, frameon=False, fontsize=9.5)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--annotate-ef", action="store_true")
    args = parser.parse_args()
    memory_label = "memory_" + str(float(args.memory_ratio)).replace(".", "p")
    points_by_method = load_points(args.input_root.resolve(), memory_label, float(args.memory_ratio))
    plot(points_by_method, args.output.resolve(), annotate_ef=bool(args.annotate_ef))
    csv_path = write_csv(points_by_method, args.output.resolve())
    for method, points in points_by_method.items():
        if not points:
            continue
        print(f"{DISPLAY_METHOD_NAME.get(method, method)}: {len(points)} points")
        for point in points:
            print(f"  ef={point.ef_search} recall={point.recall:.4f} qps={point.qps:.2f}")
    print(f"Saved figure to {args.output.resolve()}")
    if args.output.suffix.lower() != ".pdf":
        print(f"Saved figure to {args.output.resolve().with_suffix(chr(46) + chr(112) + chr(100) + chr(102))}")
    if csv_path is not None:
        print(f"Saved parsed data to {csv_path}")


if __name__ == "__main__":
    main()
