from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent
DEFAULT_LOG_DIR = BENCHMARK_DIR / "efs_logs"
DEFAULT_OUTPUT = DEFAULT_LOG_DIR / "honeybee_vs_ours_recall_time.png"

METHOD_ORDER = ("AnonySys", "kmeans", "QDTree", "RLS")
DISPLAY_METHOD_NAME = {
    "AnonySys": "HONEYBEE",
    "kmeans": "ours",
    "QDTree": "HQI",
    "RLS": "RLS",
}


@dataclass(frozen=True)
class Point:
    method: str
    ef_search: int
    recall: float
    query_time_ms: float
    log_file: Path


def _method_from_name(name: str) -> str | None:
    if name.startswith("AnonySys_"):
        return "AnonySys"
    if name.startswith("kmeans_"):
        return "kmeans"
    if name.startswith("QDTree_"):
        return "QDTree"
    if name.startswith("RLS_"):
        return "RLS"
    return None


def _ef_from_name(name: str) -> int | None:
    match = re.search(r"(?:^|_)efs(\d+)(?=_)", name)
    if match is None:
        return None
    return int(match.group(1))


def _last_float(pattern: str, text: str) -> float | None:
    values = re.findall(pattern, text)
    if not values:
        return None
    return float(values[-1])


def _parse_log(path: Path) -> Point | None:
    method = _method_from_name(path.name)
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


def load_points(log_dir: Path) -> dict[str, list[Point]]:
    latest_by_method_ef: dict[tuple[str, int], Point] = {}
    for path in sorted(log_dir.glob("*.log")):
        point = _parse_log(path)
        if point is None:
            continue
        key = (point.method, point.ef_search)
        old = latest_by_method_ef.get(key)
        if old is None or path.stat().st_mtime >= old.log_file.stat().st_mtime:
            latest_by_method_ef[key] = point

    grouped: dict[str, list[Point]] = {method: [] for method in METHOD_ORDER}
    for point in latest_by_method_ef.values():
        grouped.setdefault(point.method, []).append(point)

    for method in grouped:
        grouped[method].sort(key=lambda item: (item.recall, item.ef_search))
    return grouped


def write_csv(points_by_method: dict[str, list[Point]], output: Path) -> None:
    rows = [
        {
            "method": DISPLAY_METHOD_NAME.get(point.method, point.method),
            "ef_search": point.ef_search,
            "recall": point.recall,
            "query_time_ms": point.query_time_ms,
            "log_file": str(point.log_file),
        }
        for method in METHOD_ORDER
        for point in points_by_method.get(method, [])
    ]
    if not rows:
        return
    csv_path = output.with_suffix(".csv")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(points_by_method: dict[str, list[Point]], output: Path, *, annotate_ef: bool = False) -> None:
    styles = {
        "AnonySys": {
            "color": "#e69f32",
            "marker": "s",
            "linestyle": ":",
            "linewidth": 1.0,
            "markersize": 6.0,
        },
        "kmeans": {
            "color": "#b12a2f",
            "marker": "X",
            "linestyle": ":",
            "linewidth": 1.0,
            "markersize": 7.0,
        },
        "QDTree": {
            "color": "#2f6fbb",
            "marker": "o",
            "linestyle": ":",
            "linewidth": 1.0,
            "markersize": 6.0,
        },
        "RLS": {
            "color": "#4d4d4d",
            "marker": "^",
            "linestyle": ":",
            "linewidth": 1.0,
            "markersize": 6.0,
        },
    }

    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    plotted = 0
    for method in METHOD_ORDER:
        points = points_by_method.get(method, [])
        if not points:
            continue
        plotted += 1
        xs = [point.recall for point in points]
        ys = [point.query_time_ms for point in points]
        ax.plot(xs, ys, label=DISPLAY_METHOD_NAME.get(method, method), **styles[method])
        if annotate_ef:
            for point in points:
                ax.annotate(
                    str(point.ef_search),
                    (point.recall, point.query_time_ms),
                    textcoords="offset points",
                    xytext=(3, 4),
                    fontsize=8,
                )

    if plotted == 0:
        raise RuntimeError("No valid benchmark log points were found.")

    ax.set_xlabel("Recall@10", fontsize=13)
    ax.set_ylabel("Query Time (ms)", fontsize=13)
    ax.set_xlim(0.7, 1.0)
    ax.grid(True, color="#d9d9d9", linewidth=0.7, alpha=0.65)
    ax.tick_params(axis="both", labelsize=11, direction="in")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.24),
        ncol=2,
        frameon=True,
        framealpha=1.0,
        edgecolor="black",
        fontsize=12,
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=300, bbox_inches="tight")
    if output.suffix.lower() != ".pdf":
        fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot Recall@10 vs Query Time from ef-search benchmark logs.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory containing benchmark .log files.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output image path.")
    parser.add_argument("--annotate-ef", action="store_true", help="Annotate each point with its ef_search value.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    output = Path(args.output).resolve()
    points_by_method = load_points(log_dir)
    plot(points_by_method, output, annotate_ef=bool(args.annotate_ef))
    write_csv(points_by_method, output)

    for method in METHOD_ORDER:
        points = points_by_method.get(method, [])
        print(f"{DISPLAY_METHOD_NAME.get(method, method)}: {len(points)} points")
        for point in points:
            print(
                f"  ef={point.ef_search:<4d} "
                f"recall={point.recall:.4f} "
                f"query_time={point.query_time_ms:.3f} ms"
            )
    print(f"Saved figure to {output}")
    if output.suffix.lower() != ".pdf":
        print(f"Saved figure to {output.with_suffix(chr(46) + chr(112) + chr(100) + chr(102))}")
    print(f"Saved parsed data to {output.with_suffix(chr(46) + chr(99) + chr(115) + chr(118))}")


if __name__ == "__main__":
    main()
