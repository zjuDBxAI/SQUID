from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from argparse import Namespace
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
BENCHMARK_DIR = PROJECT_ROOT / "basic_benchmark"
DEFAULT_OUTPUT_DIR = BENCHMARK_DIR / "result" / "plan_time"

sys.path.insert(0, str(PROJECT_ROOT))

from basic_benchmark.script.measure_plan_time import (  # noqa: E402
    measure_honeybee,
    measure_kmeans,
    measure_veda,
)


MEMORY_BUDGETS = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0)
METHODS = ("ours", "honeybee", "veda", "effveda")
DISPLAY_NAMES = {
    "ours": "SQUID",
    "honeybee": "HONEYBEE",
    "veda": "VEDA",
    "effveda": "EFFVEDA",
}


def base_measure_args() -> Namespace:
    return Namespace(
        iterations=1,
        show_progress=False,
        document_limit=None,
        topk=10,
        ef_search=100,
        query_dataset_path=None,
        embedding_dim=None,
        honeybee_storage=1.0,
        honeybee_recall=0.99,
        honeybee_refine=True,
        honeybee_parameter_path=str(PROJECT_ROOT / "controller" / "dynamic_partition" / "hnsw" / "parameter_hnsw.json"),
        veda_algorithm="effveda",
        veda_indexing_threshold=2900,
        veda_storage_amplification=1.0,
        cluster_count=30,
        private_cluster_count=None,
        shared_cluster_count=5,
        shared_score_ratio=0.10,
        shared_route_limit=3,
        private_replication_budget_ratio=0.0,
        enable_split=False,
        private_edge_top_d=32,
        output_json=None,
        output_csv=None,
    )


def measure_one(method: str, memory_budget: float, iteration: int) -> dict[str, object]:
    args = base_measure_args()
    if method == "ours":
        args.private_replication_budget_ratio = max(0.0, float(memory_budget) - 1.0)
        row = measure_kmeans(args)
        raw_method = "KMEANS"
    elif method == "honeybee":
        args.honeybee_storage = float(memory_budget)
        row = measure_honeybee(args)
        raw_method = "HONEYBEE"
    elif method in {"veda", "effveda"}:
        args.veda_algorithm = method
        args.veda_storage_amplification = float(memory_budget)
        row = measure_veda(args)
        raw_method = str(row.get("method", method)).upper()
    else:
        raise ValueError(f"unsupported method: {method}")

    metadata = dict(row.get("metadata") or {})
    metadata.update(
        {
            "dataset_tag": "erbac+sift",
            "measurement": "pure planning time; no partition materialization or index build",
            "memory_budget": float(memory_budget),
            "raw_method": raw_method,
        }
    )
    return {
        "dataset_tag": "erbac+sift",
        "method": DISPLAY_NAMES.get(method, method.upper()),
        "raw_method": raw_method,
        "memory_budget": float(memory_budget),
        "iteration": int(iteration),
        "plan_seconds": float(row["plan_seconds"]),
        "plan_minutes": float(row["plan_seconds"]) / 60.0,
        "partition_count": int(row["partition_count"]),
        "metadata": metadata,
    }


def write_raw_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_tag",
        "method",
        "raw_method",
        "memory_budget",
        "iteration",
        "plan_seconds",
        "plan_minutes",
        "partition_count",
        "metadata",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    **{key: row[key] for key in fieldnames if key != "metadata"},
                    "metadata": json.dumps(row["metadata"], ensure_ascii=False, sort_keys=True),
                }
            )


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, float], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((str(row["method"]), float(row["memory_budget"])), []).append(row)

    summary: list[dict[str, object]] = []
    for (method, memory_budget), values in sorted(grouped.items(), key=lambda item: (item[0][0], item[0][1])):
        seconds = [float(row["plan_seconds"]) for row in values]
        partitions = [int(row["partition_count"]) for row in values]
        summary.append(
            {
                "dataset_tag": "erbac+sift",
                "method": method,
                "memory_budget": memory_budget,
                "iterations": len(values),
                "plan_seconds_mean": statistics.mean(seconds),
                "plan_seconds_min": min(seconds),
                "plan_seconds_max": max(seconds),
                "plan_seconds_stdev": statistics.stdev(seconds) if len(seconds) > 1 else 0.0,
                "plan_minutes_mean": statistics.mean(seconds) / 60.0,
                "partition_count_mean": statistics.mean(partitions),
                "partition_count_min": min(partitions),
                "partition_count_max": max(partitions),
            }
        )
    return summary


def write_summary_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset_tag",
        "method",
        "memory_budget",
        "iterations",
        "plan_seconds_mean",
        "plan_seconds_min",
        "plan_seconds_max",
        "plan_seconds_stdev",
        "plan_minutes_mean",
        "partition_count_mean",
        "partition_count_min",
        "partition_count_max",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Collect ERBAC+SIFT planning time across memory budgets.")
    parser.add_argument("--methods", nargs="+", default=list(METHODS), choices=list(METHODS))
    parser.add_argument("--memory-budgets", nargs="+", type=float, default=list(MEMORY_BUDGETS))
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    for iteration in range(1, max(1, int(args.iterations)) + 1):
        for memory_budget in args.memory_budgets:
            for method in args.methods:
                print(
                    f"[plan-time] method={method} memory={float(memory_budget):.1f} iteration={iteration}",
                    flush=True,
                )
                row = measure_one(method, float(memory_budget), iteration)
                rows.append(row)
                print(
                    f"[plan-time] done method={row['method']} memory={float(memory_budget):.1f} "
                    f"seconds={float(row['plan_seconds']):.3f} partitions={int(row['partition_count'])}",
                    flush=True,
                )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_json = output_dir / "erbac_sift_plan_time_raw.json"
    raw_csv = output_dir / "erbac_sift_plan_time_raw.csv"
    summary_json = output_dir / "erbac_sift_plan_time_summary.json"
    summary_csv = output_dir / "erbac_sift_plan_time_summary.csv"

    summary_rows = summarize(rows)
    raw_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    summary_json.write_text(json.dumps(summary_rows, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    write_raw_csv(raw_csv, rows)
    write_summary_csv(summary_csv, summary_rows)

    print(f"Saved raw JSON to {raw_json}")
    print(f"Saved raw CSV to {raw_csv}")
    print(f"Saved summary JSON to {summary_json}")
    print(f"Saved summary CSV to {summary_csv}")


if __name__ == "__main__":
    main()
