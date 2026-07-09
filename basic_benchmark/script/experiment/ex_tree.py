from __future__ import annotations

import argparse
from pathlib import Path

from ex1 import (
    DEFAULT_MIN_RECALL,
    DEFAULT_RECALL_ANCHORS,
    DISPLAY_METHOD_NAME,
    _parse_anchors,
    load_points,
    plot,
    write_csv,
)


SCRIPT_DIR = Path(__file__).resolve().parent
BENCHMARK_DIR = SCRIPT_DIR.parent.parent
DEFAULT_LOG_DIR = BENCHMARK_DIR / "efs_logs" / "treebase_sift"
DEFAULT_OUTPUT = DEFAULT_LOG_DIR / "ex_tree_recall_time.png"


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot treebase_sift recall-latency curves from ef-search logs.")
    parser.add_argument("--log-dir", default=str(DEFAULT_LOG_DIR), help="Directory containing method subdirectories and .log files.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output image path.")
    parser.add_argument("--min-recall", type=float, default=DEFAULT_MIN_RECALL, help="Drop points below this recall.")
    parser.add_argument("--anchors", type=_parse_anchors, default=DEFAULT_RECALL_ANCHORS, help="Comma-separated recall anchors for aligned interpolation.")
    parser.add_argument("--annotate-ef", action="store_true", help="Annotate each point with its ef_search value or interpolation bracket.")
    args = parser.parse_args()

    log_dir = Path(args.log_dir).resolve()
    output = Path(args.output).resolve()
    anchors = tuple(anchor for anchor in tuple(args.anchors) if anchor >= float(args.min_recall))

    points_by_method = load_points(log_dir, min_recall=float(args.min_recall), anchors=anchors)
    plot(points_by_method, output, annotate_ef=bool(args.annotate_ef), min_recall=float(args.min_recall), anchors=anchors)
    csv_path = write_csv(points_by_method, output)

    for method, points in points_by_method.items():
        print(f"{DISPLAY_METHOD_NAME.get(method, method)}: {len(points)} points")
        for point in points:
            tag = "interp" if point.interpolated else f"ef={point.ef_search}"
            print(f"  recall={point.recall:.4f} query_time={point.query_time_ms:.3f} ms {tag}")
    print(f"Saved figure to {output}")
    if output.suffix.lower() != ".pdf":
        print(f"Saved figure to {output.with_suffix('.pdf')}")
    if csv_path is not None:
        print(f"Saved parsed data to {csv_path}")


if __name__ == "__main__":
    main()
