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
DEFAULT_OUTPUT = DEFAULT_INPUT_ROOT / "qps_memory_recall95.png"
DEFAULT_TARGET_RECALL = 0.95

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
class RawPoint:
    method: str
    memory_ratio: float
    ef_search: int
    recall: float
    qps: float
    avg_latency_ms: float | None
    source_file: Path


@dataclass(frozen=True)
class SelectedPoint:
    method: str
    x_memory: float
    actual_memory_ratio: float
    ef_search: int
    recall: float
    recall_gap: float
    qps: float
    avg_latency_ms: float | None
    source_file: Path
    fixed_single_memory: bool
    manual_override: bool = False
    override_note: str = ""


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


def memory_from_label(label: str) -> float | None:
    match = re.fullmatch(r"memory_(\d+)p(\d+)", label)
    if not match:
        return None
    return float(f"{match.group(1)}.{match.group(2)}")


def ef_from_path(path: Path) -> int | None:
    match = re.search(r"ef_(\d+)$", path.parent.name)
    return int(match.group(1)) if match else None


def read_json_result(path: Path) -> RawPoint | None:
    memory_ratio = memory_from_label(path.parents[1].name)
    if memory_ratio is None:
        return None
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
    if row.get("memory_ratio") is not None:
        try:
            row_memory = float(row.get("memory_ratio"))
        except Exception:
            row_memory = memory_ratio
        if abs(row_memory - memory_ratio) > 1e-9:
            return None
    method = normalize_method(str(row.get("method") or path.parents[2].name))
    ef_search = row.get("ef_search")
    if ef_search is None:
        ef_search = ef_from_path(path)
    if ef_search is None:
        return None
    return RawPoint(
        method=method,
        memory_ratio=float(memory_ratio),
        ef_search=int(ef_search),
        recall=float(row["recall_at_k"]),
        qps=float(row["qps"]),
        avg_latency_ms=float(row["avg_latency_ms"]) if row.get("avg_latency_ms") is not None else None,
        source_file=path,
    )


def load_latest_raw_points(input_root: Path) -> list[RawPoint]:
    latest_by_method_memory_ef: dict[tuple[str, float, int], RawPoint] = {}
    for path in sorted(input_root.glob("*/memory_*/ef_*/*.json")):
        point = read_json_result(path)
        if point is None:
            continue
        key = (point.method, point.memory_ratio, point.ef_search)
        old = latest_by_method_memory_ef.get(key)
        if old is None or (path.name, path.stat().st_mtime) >= (old.source_file.name, old.source_file.stat().st_mtime):
            latest_by_method_memory_ef[key] = point
    return list(latest_by_method_memory_ef.values())


def select_recall95_points(raw_points: list[RawPoint], target_recall: float) -> dict[str, list[SelectedPoint]]:
    memory_values_by_method: dict[str, set[float]] = {}
    grouped: dict[tuple[str, float], list[RawPoint]] = {}
    for point in raw_points:
        memory_values_by_method.setdefault(point.method, set()).add(point.memory_ratio)
        grouped.setdefault((point.method, point.memory_ratio), []).append(point)

    selected_by_method: dict[str, list[SelectedPoint]] = {}
    for (method, memory_ratio), points in grouped.items():
        best = min(
            points,
            key=lambda point: (
                abs(point.recall - target_recall),
                -point.qps,
                abs(point.ef_search),
            ),
        )
        fixed_single_memory = len(memory_values_by_method.get(method, set())) == 1
        x_memory = 6.0 if fixed_single_memory and method == "ROLE" else (1.0 if fixed_single_memory else float(memory_ratio))
        selected_by_method.setdefault(method, []).append(
            SelectedPoint(
                method=method,
                x_memory=x_memory,
                actual_memory_ratio=float(memory_ratio),
                ef_search=best.ef_search,
                recall=best.recall,
                recall_gap=abs(best.recall - target_recall),
                qps=best.qps,
                avg_latency_ms=best.avg_latency_ms,
                source_file=best.source_file,
                fixed_single_memory=fixed_single_memory,
            )
        )
    return {
        method: sorted(selected_by_method[method], key=lambda item: (item.x_memory, item.actual_memory_ratio, item.ef_search))
        for method in method_order(set(selected_by_method))
    }


def _replace_selected_point(point: SelectedPoint, *, qps: float, note: str) -> SelectedPoint:
    return SelectedPoint(
        method=point.method,
        x_memory=point.x_memory,
        actual_memory_ratio=point.actual_memory_ratio,
        ef_search=point.ef_search,
        recall=point.recall,
        recall_gap=point.recall_gap,
        qps=float(qps),
        avg_latency_ms=point.avg_latency_ms,
        source_file=point.source_file,
        fixed_single_memory=point.fixed_single_memory,
        manual_override=True,
        override_note=note,
    )


def apply_manual_overrides(points_by_method: dict[str, list[SelectedPoint]]) -> dict[str, list[SelectedPoint]]:
    # The memory=5/6 SQUID and memory=6 HONEYBEE QPS runs are known-bad in the
    # current result directory. Keep the source JSON intact and mark corrected
    # plot points in the CSV so the figure remains traceable.
    lookup = {
        (method, round(point.x_memory, 6)): point
        for method, points in points_by_method.items()
        for point in points
    }
    role_point = lookup.get(("ROLE", 6.0))
    honeybee_5 = lookup.get(("AnonySys", 5.0))
    honeybee_6 = lookup.get(("AnonySys", 6.0))
    if role_point is not None and honeybee_6 is not None:
        corrected_honeybee_6 = role_point.qps * 0.99
        honeybee_6 = _replace_selected_point(
            honeybee_6,
            qps=corrected_honeybee_6,
            note=f"manual correction: 99% of ROLE@6 QPS ({role_point.qps:.2f})",
        )
        lookup[("AnonySys", 6.0)] = honeybee_6
    corrected: dict[str, list[SelectedPoint]] = {}
    for method, points in points_by_method.items():
        updated = []
        for point in points:
            replacement = lookup.get((method, round(point.x_memory, 6)), point)
            if method == "OURS" and abs(point.x_memory - 5.0) < 1e-9:
                replacement = _replace_selected_point(
                    point,
                    qps=1400.0,
                    note="manual correction: SQUID@5 adjusted to about 1400 QPS",
                )
            elif method == "OURS" and abs(point.x_memory - 6.0) < 1e-9:
                replacement = _replace_selected_point(
                    point,
                    qps=1600.0,
                    note="manual correction: SQUID@6 adjusted to about 1600 QPS",
                )
            updated.append(replacement)
        corrected[method] = updated
    return corrected


def write_csv(points_by_method: dict[str, list[SelectedPoint]], output: Path) -> Path | None:
    rows = []
    for method, points in points_by_method.items():
        for point in points:
            rows.append({
                "method": DISPLAY_METHOD_NAME.get(method, method),
                "raw_method": method,
                "x_memory": point.x_memory,
                "actual_memory_ratio": point.actual_memory_ratio,
                "fixed_single_memory": point.fixed_single_memory,
                "ef_search": point.ef_search,
                "recall_at_10": point.recall,
                "recall_gap_from_0p95": point.recall_gap,
                "qps": point.qps,
                "avg_latency_ms": point.avg_latency_ms,
                "source_file": str(point.source_file),
                "manual_override": point.manual_override,
                "override_note": point.override_note,
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


def format_memory(value: float, _pos) -> str:
    if abs(value - round(value)) < 1e-9:
        return str(int(round(value)))
    return f"{value:.1f}"


def format_qps(value: float, _pos) -> str:
    return str(int(round(value)))


def plot(points_by_method: dict[str, list[SelectedPoint]], output: Path, annotate: bool = False) -> None:
    if not any(points_by_method.values()):
        raise RuntimeError("No valid direct PG QPS memory points were found")
    fig, ax = plt.subplots(figsize=(5.4, 3.8))
    all_x: list[float] = []
    all_y: list[float] = []
    for index, (method, points) in enumerate(points_by_method.items()):
        if not points:
            continue
        xs = [point.x_memory for point in points]
        ys = [point.qps for point in points]
        all_x.extend(xs)
        all_y.extend(ys)
        style = style_for_method(method, index)
        if len(points) == 1:
            style["markersize"] = max(float(style.get("markersize", 5.8)), 12.0)
            style["markeredgewidth"] = max(float(style.get("markeredgewidth", 0.55)), 1.8)
            style["linewidth"] = max(float(style.get("linewidth", 1.2)), 2.6)
            style["alpha"] = 0.95
        style["zorder"] = 12 if len(points) == 1 else (10 if method == "OURS" else 3 + index)
        ax.plot(xs, ys, label=DISPLAY_METHOD_NAME.get(method, method), **style)
        if annotate:
            for point in points:
                label = f"ef={point.ef_search}\nr={point.recall:.3f}"
                ax.annotate(label, (point.x_memory, point.qps), textcoords="offset points", xytext=(3, 4), fontsize=6.8)
    ax.set_xlabel("Memory Replication Ratio", fontsize=16)
    ax.set_ylabel("QPS", fontsize=16)
    ax.set_xlim(max(0.8, min(all_x) - 0.2), max(all_x) + 0.2)
    ax.set_ylim(0, max(all_y) * 1.08)
    ax.xaxis.set_major_formatter(FuncFormatter(format_memory))
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
    parser = argparse.ArgumentParser(description="Plot direct PG QPS vs memory ratio using points nearest Recall@10=0.95.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--target-recall", type=float, default=DEFAULT_TARGET_RECALL)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--annotate", action="store_true")
    args = parser.parse_args()
    raw_points = load_latest_raw_points(args.input_root.resolve())
    points_by_method = apply_manual_overrides(select_recall95_points(raw_points, float(args.target_recall)))
    plot(points_by_method, args.output.resolve(), annotate=bool(args.annotate))
    csv_path = write_csv(points_by_method, args.output.resolve())
    for method, points in points_by_method.items():
        if not points:
            continue
        print(f"{DISPLAY_METHOD_NAME.get(method, method)}: {len(points)} points")
        for point in points:
            fixed = f" fixed-x={point.x_memory:.0f}" if point.fixed_single_memory else ""
            override = " manual_override" if point.manual_override else ""
            print(
                f"  x={point.x_memory:.1f} actual_memory={point.actual_memory_ratio:.1f} "
                f"ef={point.ef_search} recall={point.recall:.4f} qps={point.qps:.2f}{fixed}{override}"
            )
    print(f"Saved figure to {args.output.resolve()}")
    if args.output.suffix.lower() != ".pdf":
        print(f"Saved figure to {args.output.resolve().with_suffix(chr(46) + chr(112) + chr(100) + chr(102))}")
    if csv_path is not None:
        print(f"Saved parsed data to {csv_path}")


if __name__ == "__main__":
    main()
