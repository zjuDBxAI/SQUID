from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import statistics
import sys
import time
from typing import Any, Iterable, Sequence

import numpy as np
from psycopg2 import sql


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from services.config import get_db_connection, get_maintenance_settings  # noqa: E402


RESULT_DIR = Path(__file__).resolve().parent / "result"
TARGET_RECALL = 0.95


@dataclass(frozen=True)
class RecallProbeRow:
    source_table: str
    table_vectors: int
    topk: int
    ef_search: int
    recall: float
    query_count: int
    avg_hnsw_time_ms: float
    avg_exact_time_ms: float
    target_recall: float
    target_met: bool


@dataclass(frozen=True)
class ThresholdRow:
    source_table: str
    table_vectors: int
    topk: int
    target_recall: float
    required_ef_search: int
    recall_at_required: float
    interpolated_ef_search: float
    ef_per_k: float
    status: str


def parse_int_list(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [int(value) for value in raw]
    return sorted(dict.fromkeys(value for value in values if value > 0))


def _write_csv(path: Path, rows: Sequence[Any], fallback_fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(fallback_fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _safe_index_name(table_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", table_name).strip("_").lower()
    return f"{normalized[:45]}_vector_hnsw_dbg_idx"


def _table_count(cur, table_name: str) -> int:
    cur.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(sql.Identifier(table_name)))
    return int(cur.fetchone()[0])


def _hnsw_index_exists(cur, table_name: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = 'public'
          AND tablename = %s
          AND indexdef ILIKE '%%USING hnsw%%'
          AND indexdef ILIKE '%%vector%%'
        LIMIT 1;
        """,
        [str(table_name)],
    )
    return cur.fetchone() is not None


def _configure_index_build(cur) -> None:
    settings = get_maintenance_settings()
    commands = [
        sql.SQL("SET maintenance_work_mem = {};").format(sql.Literal(f"{settings['maintenance_work_mem_gb']}GB")),
        sql.SQL("SET max_parallel_maintenance_workers = {};").format(
            sql.Literal(int(settings["max_parallel_maintenance_workers"]))
        ),
    ]
    for command in commands:
        try:
            cur.execute(command)
        except Exception:
            cur.connection.rollback()


def _ensure_hnsw_index(
    cur,
    table_name: str,
    *,
    create_index: bool,
    hnsw_m: int,
    hnsw_ef_construction: int,
) -> None:
    if _hnsw_index_exists(cur, table_name):
        return
    if not create_index:
        raise RuntimeError(
            f"Table {table_name} has no HNSW index on vector. "
            "Pass --create-index if you want this debug script to build one."
        )
    index_name = _safe_index_name(table_name)
    print(f"[efs-vs-k] building HNSW index {index_name} on {table_name}")
    _configure_index_build(cur)
    cur.execute(
        sql.SQL(
            """
            CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (vector vector_l2_ops)
            WITH (m = {}, ef_construction = {});
            """
        ).format(
            sql.Identifier(index_name),
            sql.Identifier(table_name),
            sql.Literal(int(hnsw_m)),
            sql.Literal(int(hnsw_ef_construction)),
        )
    )
    cur.execute(sql.SQL("ANALYZE {};").format(sql.Identifier(table_name)))


def _load_query_vectors(cur, table_name: str, query_limit: int, query_offset: int) -> list[str]:
    cur.execute(
        sql.SQL(
            """
            SELECT vector::text
            FROM {}
            WHERE vector IS NOT NULL
            ORDER BY md5(block_id::text)
            OFFSET %s
            LIMIT %s;
            """
        ).format(sql.Identifier(table_name)),
        [int(query_offset), int(query_limit)],
    )
    vectors = [str(row[0]) for row in cur.fetchall()]
    if not vectors:
        raise RuntimeError(f"No query vectors found in {table_name}")
    return vectors


def _query_block_ids(
    cur,
    table_name: str,
    query_vector: str,
    topk: int,
    *,
    exact: bool,
    ef_search: int | None = None,
) -> list[int]:
    if exact:
        cur.execute("SET enable_indexscan = off;")
        cur.execute("SET enable_bitmapscan = off;")
        cur.execute("SET enable_seqscan = on;")
    else:
        cur.execute("SET enable_indexscan = on;")
        cur.execute("SET enable_bitmapscan = on;")
        cur.execute("SET enable_seqscan = off;")
        cur.execute(sql.SQL("SET hnsw.ef_search = {};").format(sql.Literal(int(ef_search or 40))))

    cur.execute(
        sql.SQL(
            """
            SELECT block_id
            FROM {}
            ORDER BY vector <-> %s
            LIMIT %s;
            """
        ).format(sql.Identifier(table_name)),
        [query_vector, int(topk)],
    )
    return [int(row[0]) for row in cur.fetchall()]


def probe_efs_vs_k(
    *,
    source_table: str,
    k_values: Sequence[int],
    ef_values: Sequence[int],
    query_limit: int,
    query_offset: int,
    target_recall: float,
    create_index: bool,
    hnsw_m: int,
    hnsw_ef_construction: int,
) -> tuple[list[RecallProbeRow], list[ThresholdRow]]:
    rows: list[RecallProbeRow] = []
    max_k = max(k_values)

    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    try:
        table_vectors = _table_count(cur, source_table)
        _ensure_hnsw_index(
            cur,
            source_table,
            create_index=bool(create_index),
            hnsw_m=int(hnsw_m),
            hnsw_ef_construction=int(hnsw_ef_construction),
        )
        queries = _load_query_vectors(cur, source_table, int(query_limit), int(query_offset))
        print(
            f"[efs-vs-k] source={source_table}, N={table_vectors}, "
            f"queries={len(queries)}, k={list(k_values)}, efs={list(ef_values)}"
        )

        exact_results: list[list[int]] = []
        exact_elapsed = 0.0
        for idx, query_vector in enumerate(queries, start=1):
            started = time.perf_counter()
            exact_results.append(_query_block_ids(cur, source_table, query_vector, max_k, exact=True))
            exact_elapsed += time.perf_counter() - started
            if idx % 10 == 0 or idx == len(queries):
                print(f"[efs-vs-k] exact queries {idx}/{len(queries)}")

        avg_exact_ms = 1000.0 * exact_elapsed / max(1, len(queries))
        exact_sets_by_k: dict[int, list[set[int]]] = {
            int(k): [set(result[: int(k)]) for result in exact_results]
            for k in k_values
        }

        for topk in k_values:
            exact_sets = exact_sets_by_k[int(topk)]
            for ef_search in ef_values:
                hnsw_elapsed = 0.0
                total_recall = 0.0
                for idx, query_vector in enumerate(queries):
                    started = time.perf_counter()
                    approx = _query_block_ids(
                        cur,
                        source_table,
                        query_vector,
                        int(topk),
                        exact=False,
                        ef_search=int(ef_search),
                    )
                    hnsw_elapsed += time.perf_counter() - started
                    exact = exact_sets[idx]
                    total_recall += len(set(approx) & exact) / max(1, len(exact))

                recall = total_recall / max(1, len(queries))
                row = RecallProbeRow(
                    source_table=str(source_table),
                    table_vectors=int(table_vectors),
                    topk=int(topk),
                    ef_search=int(ef_search),
                    recall=float(recall),
                    query_count=int(len(queries)),
                    avg_hnsw_time_ms=1000.0 * hnsw_elapsed / max(1, len(queries)),
                    avg_exact_time_ms=float(avg_exact_ms),
                    target_recall=float(target_recall),
                    target_met=bool(recall >= target_recall),
                )
                rows.append(row)
                print(
                    "[efs-vs-k] "
                    f"k={row.topk}, ef={row.ef_search}, recall={row.recall:.4f}, "
                    f"hnsw_ms={row.avg_hnsw_time_ms:.3f}"
                )

        return rows, derive_thresholds(rows, target_recall=target_recall)
    finally:
        try:
            cur.execute("SET enable_seqscan = on;")
            cur.execute("SET enable_indexscan = on;")
            cur.execute("SET enable_bitmapscan = on;")
        except Exception:
            pass
        cur.close()
        conn.close()


def derive_thresholds(rows: Sequence[RecallProbeRow], *, target_recall: float) -> list[ThresholdRow]:
    grouped: dict[int, list[RecallProbeRow]] = {}
    for row in rows:
        grouped.setdefault(int(row.topk), []).append(row)

    thresholds: list[ThresholdRow] = []
    for topk, group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: int(item.ef_search))
        previous: RecallProbeRow | None = None
        selected: RecallProbeRow | None = None
        interpolated = float(ordered[-1].ef_search) if ordered else 0.0
        status = "lower_bound"

        for row in ordered:
            if row.recall >= target_recall:
                selected = row
                status = "met"
                if previous is None:
                    interpolated = float(row.ef_search)
                elif row.recall == previous.recall:
                    interpolated = float(row.ef_search)
                else:
                    ratio = (float(target_recall) - previous.recall) / (row.recall - previous.recall)
                    interpolated = float(previous.ef_search) + ratio * float(row.ef_search - previous.ef_search)
                break
            previous = row

        if selected is None:
            selected = ordered[-1]

        thresholds.append(
            ThresholdRow(
                source_table=str(selected.source_table),
                table_vectors=int(selected.table_vectors),
                topk=int(topk),
                target_recall=float(target_recall),
                required_ef_search=int(selected.ef_search),
                recall_at_required=float(selected.recall),
                interpolated_ef_search=float(interpolated),
                ef_per_k=float(selected.ef_search) / max(1.0, float(topk)),
                status=status,
            )
        )
    return thresholds


def _render_threshold_table(rows: Sequence[ThresholdRow]) -> str:
    lines = [
        "| k | required_efs | recall | efs/k | interpolated_efs | status |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row.topk} | {row.required_ef_search} | {row.recall_at_required:.4f} | "
            f"{row.ef_per_k:.3f} | {row.interpolated_ef_search:.2f} | {row.status} |"
        )
    return "\n".join(lines)


def _render_recall_matrix(rows: Sequence[RecallProbeRow]) -> str:
    k_values = sorted({int(row.topk) for row in rows})
    ef_values = sorted({int(row.ef_search) for row in rows})
    by_key = {(int(row.topk), int(row.ef_search)): float(row.recall) for row in rows}
    lines = [
        "| ef_search | " + " | ".join(f"k={k}" for k in k_values) + " |",
        "|---:|" + "|".join("---:" for _ in k_values) + "|",
    ]
    for ef_search in ef_values:
        values = [f"{by_key.get((topk, ef_search), 0.0):.4f}" for topk in k_values]
        lines.append(f"| {ef_search} | " + " | ".join(values) + " |")
    return "\n".join(lines)


def _write_plot(output_path: Path, thresholds: Sequence[ThresholdRow], rows: Sequence[RecallProbeRow]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written: list[Path] = []
    output_path.mkdir(parents=True, exist_ok=True)

    ordered_thresholds = sorted(thresholds, key=lambda row: int(row.topk))
    if ordered_thresholds:
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(
            [row.topk for row in ordered_thresholds],
            [row.required_ef_search for row in ordered_thresholds],
            marker="o",
            linewidth=2,
        )
        ax.set_xlabel("k")
        ax.set_ylabel("minimum ef_search for Recall@k >= 0.95")
        ax.set_title("Required ef_search vs k")
        ax.grid(True, linestyle="--", alpha=0.35)
        path = output_path / "efs_vs_k_required.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ax.plot(
            [row.topk for row in ordered_thresholds],
            [row.ef_per_k for row in ordered_thresholds],
            marker="o",
            linewidth=2,
        )
        ax.set_xlabel("k")
        ax.set_ylabel("required ef_search / k")
        ax.set_title("Required ef_search normalized by k")
        ax.grid(True, linestyle="--", alpha=0.35)
        path = output_path / "efs_vs_k_ratio.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    k_values = sorted({int(row.topk) for row in rows})
    ef_values = sorted({int(row.ef_search) for row in rows})
    if k_values and ef_values:
        matrix = []
        by_key = {(int(row.topk), int(row.ef_search)): float(row.recall) for row in rows}
        for ef_search in ef_values:
            matrix.append([by_key.get((topk, ef_search), 0.0) for topk in k_values])
        fig, ax = plt.subplots(figsize=(max(6.5, 0.6 * len(k_values)), max(4.0, 0.28 * len(ef_values))))
        image = ax.imshow(matrix, aspect="auto", origin="lower", vmin=0.0, vmax=1.0)
        ax.set_xticks(range(len(k_values)), [str(value) for value in k_values])
        ax.set_yticks(range(len(ef_values)), [str(value) for value in ef_values])
        ax.set_xlabel("k")
        ax.set_ylabel("ef_search")
        ax.set_title("Recall@k matrix")
        fig.colorbar(image, ax=ax, label="recall")
        path = output_path / "efs_vs_k_recall_heatmap.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    return written


def _write_report(
    path: Path,
    *,
    raw_csv: Path,
    threshold_csv: Path,
    summary_json: Path,
    plot_paths: Sequence[Path],
    rows: Sequence[RecallProbeRow],
    thresholds: Sequence[ThresholdRow],
) -> None:
    source_table = rows[0].source_table if rows else ""
    table_vectors = rows[0].table_vectors if rows else 0
    query_count = rows[0].query_count if rows else 0
    target_recall = rows[0].target_recall if rows else TARGET_RECALL
    plot_lines = [f"- {plot_path}" for plot_path in plot_paths] or ["- matplotlib is not available; only CSV/JSON/Markdown were written."]

    path.write_text(
        "\n".join(
            [
                "# ef_search vs k debug report",
                "",
                "This measurement uses the current PostgreSQL vectors only. It does not join or filter by RBAC/ABAC permissions.",
                "",
                "## Configuration",
                "",
                f"- source_table: `{source_table}`",
                f"- table_vectors: {table_vectors}",
                f"- query_count: {query_count}",
                f"- target: `Recall@k >= {target_recall:.2f}`",
                "",
                "## Required ef_search",
                "",
                _render_threshold_table(thresholds),
                "",
                "## Recall Matrix",
                "",
                _render_recall_matrix(rows),
                "",
                "## Recommended Views",
                "",
                "1. Plot `k` on the x-axis and `required_ef_search` on the y-axis. This directly shows how much larger efs must be when k grows at fixed Recall@k >= 0.95.",
                "2. Plot `k` on the x-axis and `required_ef_search / k` on the y-axis. If this curve is almost flat, the relation is close to linear in k; if it rises, larger k needs more than linear efs growth.",
                "3. Use the recall matrix as a heatmap with x=`k`, y=`ef_search`, color=`recall`. The 0.95 contour is the boundary you care about.",
                "",
                "## Artifacts",
                "",
                f"- raw_csv: {raw_csv}",
                f"- threshold_csv: {threshold_csv}",
                f"- summary_json: {summary_json}",
                *plot_lines,
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Explore the relation between ef_search and k for pure PostgreSQL HNSW recall without permission filters."
    )
    parser.add_argument("--source-table", default="documentblocks", help="Vector table to probe. Must contain block_id and vector.")
    parser.add_argument("--k-values", default="1,5,10,20,50,100", help="Comma-separated k values.")
    parser.add_argument(
        "--efs-values",
        default="5,10,20,40,80,120,160,240,320,480,640,960,1280",
        help="Comma-separated ef_search values to sweep.",
    )
    parser.add_argument("--query-limit", type=int, default=50)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--tag", default="efs_vs_k")
    parser.add_argument("--create-index", action="store_true", help="Build a HNSW index if the source table has none.")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    k_values = parse_int_list(args.k_values)
    ef_values = parse_int_list(args.efs_values)
    if not k_values:
        raise ValueError("--k-values cannot be empty")
    if not ef_values:
        raise ValueError("--efs-values cannot be empty")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, thresholds = probe_efs_vs_k(
        source_table=str(args.source_table),
        k_values=k_values,
        ef_values=ef_values,
        query_limit=int(args.query_limit),
        query_offset=int(args.query_offset),
        target_recall=TARGET_RECALL,
        create_index=bool(args.create_index),
        hnsw_m=int(args.hnsw_m),
        hnsw_ef_construction=int(args.hnsw_ef_construction),
    )

    tag = str(args.tag)
    raw_csv = output_dir / f"{tag}_raw.csv"
    threshold_csv = output_dir / f"{tag}_threshold.csv"
    summary_json = output_dir / f"{tag}_summary.json"
    report_md = output_dir / f"{tag}_report.md"

    _write_csv(
        raw_csv,
        rows,
        [
            "source_table",
            "table_vectors",
            "topk",
            "ef_search",
            "recall",
            "query_count",
            "avg_hnsw_time_ms",
            "avg_exact_time_ms",
            "target_recall",
            "target_met",
        ],
    )
    _write_csv(
        threshold_csv,
        thresholds,
        [
            "source_table",
            "table_vectors",
            "topk",
            "target_recall",
            "required_ef_search",
            "recall_at_required",
            "interpolated_ef_search",
            "ef_per_k",
            "status",
        ],
    )
    _write_json(
        summary_json,
        {
            "target_recall": TARGET_RECALL,
            "source_table": str(args.source_table),
            "k_values": k_values,
            "ef_values": ef_values,
            "query_limit": int(args.query_limit),
            "query_offset": int(args.query_offset),
            "thresholds": [asdict(row) for row in thresholds],
            "raw_csv": str(raw_csv),
            "threshold_csv": str(threshold_csv),
        },
    )
    plot_paths = _write_plot(output_dir, thresholds, rows)
    _write_report(
        report_md,
        raw_csv=raw_csv,
        threshold_csv=threshold_csv,
        summary_json=summary_json,
        plot_paths=plot_paths,
        rows=rows,
        thresholds=thresholds,
    )

    print("[efs-vs-k] required ef_search for Recall@k >= 0.95")
    for row in thresholds:
        print(
            f"  k={row.topk}: ef_search={row.required_ef_search}, "
            f"recall={row.recall_at_required:.4f}, ef/k={row.ef_per_k:.3f}, status={row.status}"
        )
    print(f"[efs-vs-k] wrote {raw_csv}")
    print(f"[efs-vs-k] wrote {threshold_csv}")
    print(f"[efs-vs-k] wrote {summary_json}")
    print(f"[efs-vs-k] wrote {report_md}")
    for plot_path in plot_paths:
        print(f"[efs-vs-k] wrote {plot_path}")


if __name__ == "__main__":
    main()
