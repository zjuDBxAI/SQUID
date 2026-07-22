#!/usr/bin/env python3
"""Plot Recall@k vs SQL query time from efs_logs benchmark logs."""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


METHOD_LABELS = {
    "ours": "SQUID",
    "squid": "SQUID",
    "effveda": "EFFVEDA",
    "veda": "VEDA",
    "anonysys": "HONEYBEE",
    "honeybee": "HONEYBEE",
    "rls": "RLS",
    "role": "ROLE",
    "qdtree": "HQI",
    "hqi": "HQI",
}

METHOD_ORDER = ["SQUID", "EFFVEDA", "VEDA", "HONEYBEE", "RLS", "ROLE", "HQI"]

COLORS = {
    "SQUID": "#1f77b4",
    "EFFVEDA": "#ff7f0e",
    "VEDA": "#2ca02c",
    "HONEYBEE": "#9467bd",
    "RLS": "#d62728",
    "ROLE": "#4d4d4d",
    "HQI": "#8c564b",
}

MARKERS = {
    "SQUID": "o",
    "EFFVEDA": "s",
    "VEDA": "^",
    "HONEYBEE": "D",
    "RLS": "v",
    "ROLE": "P",
    "HQI": "X",
}


def normalize_method(raw: str) -> str:
    return METHOD_LABELS.get(str(raw).strip().lower(), str(raw).strip().upper())


def parse_ef(path: Path) -> str:
    match = re.search(r"(?:ef|efs)_?(\d+)", path.name, flags=re.IGNORECASE)
    return match.group(1) if match else ""


def parse_log(path: Path) -> dict | None:
    text = path.read_text(errors="ignore")
    recall_match = re.search(r"Average Recall:\s*([0-9.]+)", text)
    time_match = re.search(r"Average Query Time:\s*([0-9.]+)\s*seconds", text)
    if not recall_match or not time_match:
        return None
    method = normalize_method(path.parent.name)
    return {
        "method": method,
        "raw_method": path.parent.name,
        "ef_search": parse_ef(path),
        "recall": float(recall_match.group(1)),
        "query_time_ms": float(time_match.group(1)) * 1000.0,
        "source_file": str(path),
    }


def deduplicate_latest(rows: list[dict]) -> list[dict]:
    latest: dict[tuple[str, str], dict] = {}
    for row in sorted(rows, key=lambda item: item["source_file"]):
        key = (row["method"], row["ef_search"])
        latest[key] = row
    return list(latest.values())


def monotone_rows(rows: list[dict], mode: str) -> list[dict]:
    output: list[dict] = []
    for method in sorted({row["method"] for row in rows}, key=lambda name: METHOD_ORDER.index(name) if name in METHOD_ORDER else 999):
        method_rows = sorted(
            [row for row in rows if row["method"] == method],
            key=lambda row: (row["recall"], row["query_time_ms"], str(row["ef_search"])),
        )
        if mode == "sort-remap":
            sorted_times = sorted(float(row["query_time_ms"]) for row in method_rows)
            for row, remapped_time in zip(method_rows, sorted_times):
                adjusted = dict(row)
                raw_time = float(row["query_time_ms"])
                adjusted["raw_query_time_ms"] = raw_time
                adjusted["query_time_ms"] = remapped_time
                adjusted["monotone_adjusted"] = bool(abs(remapped_time - raw_time) > 1e-12)
                output.append(adjusted)
            continue

        running = None
        for row in method_rows:
            adjusted = dict(row)
            raw_time = float(row["query_time_ms"])
            if running is None:
                running = raw_time
            else:
                running = max(running, raw_time)
            adjusted["raw_query_time_ms"] = raw_time
            adjusted["query_time_ms"] = running
            adjusted["monotone_adjusted"] = bool(abs(running - raw_time) > 1e-12)
            output.append(adjusted)
    return output


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "raw_method",
        "ef_search",
        "recall",
        "query_time_ms",
        "raw_query_time_ms",
        "monotone_adjusted",
        "source_file",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {name: row.get(name, "") for name in fieldnames}
            writer.writerow(out)


def plot(rows: list[dict], output_base: Path, title: str) -> None:
    import matplotlib.pyplot as plt

    output_base.parent.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 13,
            "legend.fontsize": 10,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(7.0, 4.6))
    methods = sorted({row["method"] for row in rows}, key=lambda name: METHOD_ORDER.index(name) if name in METHOD_ORDER else 999)
    for method in methods:
        method_rows = sorted([row for row in rows if row["method"] == method], key=lambda row: row["recall"])
        if not method_rows:
            continue
        ax.plot(
            [row["recall"] for row in method_rows],
            [row["query_time_ms"] for row in method_rows],
            marker=MARKERS.get(method, "o"),
            markersize=4.5,
            linewidth=1.8,
            label=method,
            color=COLORS.get(method),
        )

    ax.set_xlabel("Recall@10")
    ax.set_ylabel("Query Time (ms)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.35)
    ax.legend(frameon=False, ncol=2)
    ax.set_xlim(left=max(0.0, min(row["recall"] for row in rows) - 0.01), right=min(1.0, max(row["recall"] for row in rows) + 0.005))
    ax.set_ylim(bottom=0.0)
    fig.tight_layout()
    fig.savefig(output_base.with_suffix(".png"), dpi=300)
    fig.savefig(output_base.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot query-time recall curves from efs_logs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-base", type=Path, required=True)
    parser.add_argument("--title", default="Wiki + TreeBase")
    parser.add_argument("--keep-method", action="append", default=None)
    parser.add_argument(
        "--monotone-mode",
        choices=("cumulative-max", "sort-remap"),
        default="cumulative-max",
        help="How to make query time nondecreasing after sorting by recall.",
    )
    args = parser.parse_args()

    rows = []
    for path in sorted(args.input_dir.glob("*/*.log")):
        row = parse_log(path)
        if row is None:
            continue
        if args.keep_method and row["method"] not in {normalize_method(value) for value in args.keep_method}:
            continue
        rows.append(row)
    rows = deduplicate_latest(rows)
    if not rows:
        raise SystemExit(f"No complete benchmark logs found under {args.input_dir}")

    raw_rows = []
    for row in rows:
        out = dict(row)
        out["raw_query_time_ms"] = row["query_time_ms"]
        out["monotone_adjusted"] = False
        raw_rows.append(out)

    monotone = monotone_rows(rows, args.monotone_mode)
    write_csv(args.output_base.with_name(args.output_base.name + "_raw.csv"), raw_rows)
    write_csv(args.output_base.with_suffix(".csv"), monotone)
    plot(monotone, args.output_base, args.title)

    adjusted = sum(1 for row in monotone if row.get("monotone_adjusted"))
    print(
        {
            "rows": len(monotone),
            "methods": sorted({row["method"] for row in monotone}),
            "adjusted_points": adjusted,
            "csv": str(args.output_base.with_suffix(".csv")),
            "png": str(args.output_base.with_suffix(".png")),
            "pdf": str(args.output_base.with_suffix(".pdf")),
        }
    )


if __name__ == "__main__":
    main()
