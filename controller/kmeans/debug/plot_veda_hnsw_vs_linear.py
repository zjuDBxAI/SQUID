from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
from psycopg2 import sql

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from services.config import get_db_connection, get_maintenance_settings  # noqa: E402

RESULT_DIR = Path(__file__).resolve().parent / "result"
DEFAULT_SOURCE_TABLE = "documentblocks"
DEFAULT_QUERY_FILE = PROJECT_ROOT / "basic_benchmark" / "query_dataset.json"


@dataclass(frozen=True)
class SampleRow:
    source_table: str
    table_vectors: int
    topk: int
    ef_search: int
    hnsw_time_ms: float
    exact_time_ms: float
    recall: float


@dataclass(frozen=True)
class FitRow:
    model_name: str
    a: float
    b: float
    c: float
    rmse_ms: float
    mae_ms: float
    r2: float
    sample_count: int
    formula: str


@dataclass(frozen=True)
class IntersectionRow:
    topk: int
    table_vectors: int
    ef_search: int
    hnsw_time_ms: float
    linear_time_ms: float
    delta_ms: float
    status: str


def _parse_int_list(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [int(value) for value in raw]
    return sorted(dict.fromkeys(value for value in values if value > 0))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if fieldnames:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _safe_name(name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(name)).strip("_").lower()
    return normalized[:63] or "documentblocks"


def _table_exists(cur, table_name: str) -> bool:
    cur.execute("SELECT to_regclass(%s);", (table_name,))
    return cur.fetchone()[0] is not None


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
        (table_name,),
    )
    return cur.fetchone() is not None


def _configure_index_build(cur) -> None:
    settings = get_maintenance_settings()
    commands = [
        f"SET maintenance_work_mem = '{settings['maintenance_work_mem_gb']}GB';",
        f"SET max_parallel_maintenance_workers = {int(settings['max_parallel_maintenance_workers'])};",
    ]
    for command in commands:
        try:
            cur.execute(command)
        except Exception:
            cur.connection.rollback()


def _ensure_hnsw_index(cur, table_name: str, *, create_index: bool, hnsw_m: int, hnsw_ef_construction: int) -> None:
    if _hnsw_index_exists(cur, table_name):
        return
    if not create_index:
        raise RuntimeError(f"Table {table_name} has no HNSW index on vector. Pass --create-index to build it.")
    index_name = f"{_safe_name(table_name)[:45]}_vector_hnsw_dbg_idx"
    print(f"[veda-vs-hnsw-linear] building HNSW index {index_name} on {table_name}")
    _configure_index_build(cur)
    cur.execute(
        sql.SQL(
            "CREATE INDEX IF NOT EXISTS {} ON {} USING hnsw (vector vector_l2_ops) WITH (m = {}, ef_construction = {});"
        ).format(
            sql.Identifier(index_name),
            sql.Identifier(table_name),
            sql.Literal(int(hnsw_m)),
            sql.Literal(int(hnsw_ef_construction)),
        )
    )
    cur.execute(sql.SQL("ANALYZE {};").format(sql.Identifier(table_name)))


def _load_query_vectors(path: Path, limit: int | None = None, topk: int | None = None) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of query objects in {path}")

    queries: list[dict[str, object]] = []
    for item in data:
        if not isinstance(item, dict) or "query_vector" not in item:
            continue
        query_topk = int(topk if topk is not None else item.get("topk", 10))
        queries.append({"query_vector": item["query_vector"], "topk": query_topk})
        if limit is not None and len(queries) >= limit:
            break
    if not queries:
        raise ValueError(f"No query_vector entries found in {path}")
    return queries


def _exact_query(cur, table_name: str, query_vector: str, topk: int) -> list[int]:
    cur.execute("SET enable_indexscan = off;")
    cur.execute("SET enable_bitmapscan = off;")
    cur.execute("SET enable_seqscan = on;")
    cur.execute(
        sql.SQL("SELECT block_id FROM {} ORDER BY vector <-> %s LIMIT %s;").format(sql.Identifier(table_name)),
        [query_vector, int(topk)],
    )
    return [int(row[0]) for row in cur.fetchall()]


def _hnsw_query(cur, table_name: str, query_vector: str, topk: int, ef_search: int) -> list[int]:
    cur.execute("SET enable_indexscan = on;")
    cur.execute("SET enable_bitmapscan = on;")
    cur.execute("SET enable_seqscan = off;")
    cur.execute(sql.SQL("SET hnsw.ef_search = {};").format(sql.Literal(int(ef_search))))
    cur.execute(
        sql.SQL("SELECT block_id FROM {} ORDER BY vector <-> %s LIMIT %s;").format(sql.Identifier(table_name)),
        [query_vector, int(topk)],
    )
    return [int(row[0]) for row in cur.fetchall()]


def _fit_line(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float, float]:
    try:
        from scipy.optimize import lsq_linear

        result = lsq_linear(x, y, bounds=(0.0, np.inf), max_iter=10000)
        coef = result.x
    except Exception:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)
        coef = np.maximum(coef, 0.0)
    pred = x @ coef
    residuals = pred - y
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    mae = float(np.mean(np.abs(residuals)))
    denom = float(np.sum(np.square(y - np.mean(y))))
    r2 = 1.0 - float(np.sum(np.square(residuals))) / denom if denom > 0 else 1.0
    return coef, pred, rmse, mae, r2


def _prepare_sample_tables(
    cur,
    *,
    source_table: str,
    sizes: Sequence[int],
    table_prefix: str,
    rebuild: bool,
    hnsw_m: int,
    hnsw_ef_construction: int,
) -> dict[int, str]:
    sizes = sorted(dict.fromkeys(int(size) for size in sizes if int(size) > 0))
    if not sizes:
        raise ValueError("sizes cannot be empty")
    max_size = max(sizes)
    base_table = f"{table_prefix}_base_{max_size}"
    if rebuild and _table_exists(cur, base_table):
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(sql.Identifier(base_table)))
    if not _table_exists(cur, base_table) or _table_count(cur, base_table) < max_size:
        print(f"[veda-vs-hnsw-linear] building base sample table {base_table} with {max_size} rows")
        cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(sql.Identifier(base_table)))
        cur.execute(
            sql.SQL(
                """
                CREATE UNLOGGED TABLE {} AS
                SELECT row_number() OVER ()::integer AS sample_rank,
                       block_id,
                       document_id,
                       block_content,
                       vector
                FROM (
                    SELECT block_id, document_id, block_content, vector
                    FROM {}
                    WHERE vector IS NOT NULL
                    ORDER BY md5(block_id::text)
                    LIMIT %s
                ) AS sampled;
                """
            ).format(sql.Identifier(base_table), sql.Identifier(source_table)),
            (max_size,),
        )
        cur.execute(sql.SQL("ANALYZE {};").format(sql.Identifier(base_table)))
    _configure_index_build(cur)
    table_by_size: dict[int, str] = {}
    for size in sizes:
        table_name = f"{table_prefix}_n{size}"
        index_name = f"{table_name}_vector_hnsw_idx"
        if rebuild and _table_exists(cur, table_name):
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(sql.Identifier(table_name)))
        if not _table_exists(cur, table_name) or _table_count(cur, table_name) != size:
            print(f"[veda-vs-hnsw-linear] building calibration table {table_name} with {size} rows")
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {};").format(sql.Identifier(table_name)))
            cur.execute(
                sql.SQL(
                    """
                    CREATE UNLOGGED TABLE {} AS
                    SELECT block_id, document_id, block_content, vector
                    FROM {}
                    WHERE sample_rank <= %s;
                    """
                ).format(sql.Identifier(table_name), sql.Identifier(base_table)),
                (size,),
            )
            cur.execute(sql.SQL("ANALYZE {};").format(sql.Identifier(table_name)))
        if not _hnsw_index_exists(cur, table_name):
            print(f"[veda-vs-hnsw-linear] building HNSW index for {table_name}")
            cur.execute(
                sql.SQL(
                    "CREATE INDEX {} ON {} USING hnsw (vector vector_l2_ops) WITH (m = {}, ef_construction = {});"
                ).format(
                    sql.Identifier(index_name),
                    sql.Identifier(table_name),
                    sql.Literal(int(hnsw_m)),
                    sql.Literal(int(hnsw_ef_construction)),
                )
            )
            cur.execute(sql.SQL("ANALYZE {};").format(sql.Identifier(table_name)))
        table_by_size[int(size)] = table_name
    return table_by_size


def collect_samples(
    *,
    source_table: str,
    sizes: Sequence[int],
    query_limit: int,
    query_offset: int,
    topk: int,
    ef_values: Sequence[int],
    create_index: bool,
    rebuild: bool,
    table_prefix: str,
    hnsw_m: int,
    hnsw_ef_construction: int,
    query_file: Path | None = None,
) -> tuple[list[SampleRow], list[dict[str, object]]]:
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    rows: list[SampleRow] = []
    payload: list[dict[str, object]] = []
    try:
        table_by_size = _prepare_sample_tables(
            cur,
            source_table=source_table,
            sizes=sizes,
            table_prefix=table_prefix,
            rebuild=rebuild,
            hnsw_m=int(hnsw_m),
            hnsw_ef_construction=int(hnsw_ef_construction),
        )
        for table_name in table_by_size.values():
            _ensure_hnsw_index(
                cur,
                table_name,
                create_index=bool(create_index),
                hnsw_m=int(hnsw_m),
                hnsw_ef_construction=int(hnsw_ef_construction),
            )

        queries = _load_query_vectors(query_file or DEFAULT_QUERY_FILE, limit=int(query_limit), topk=int(topk))
        if int(query_offset) > 0:
            queries = queries[int(query_offset):]
        if not queries:
            raise RuntimeError("No queries selected after applying offset")

        for n_vectors, table_name in sorted(table_by_size.items()):
            exact_sets: list[set[int]] = []
            exact_elapsed = 0.0
            for query in queries:
                qvec = str(query["query_vector"])
                started = time.perf_counter()
                exact = _exact_query(cur, table_name, qvec, int(topk))
                exact_elapsed += time.perf_counter() - started
                exact_sets.append(set(exact[: int(topk)]))
            avg_exact_ms = 1000.0 * exact_elapsed / max(1, len(queries))

            for ef_search in ef_values:
                hnsw_elapsed = 0.0
                recall_total = 0.0
                for idx, query in enumerate(queries):
                    qvec = str(query["query_vector"])
                    started = time.perf_counter()
                    approx = _hnsw_query(cur, table_name, qvec, int(topk), int(ef_search))
                    hnsw_elapsed += time.perf_counter() - started
                    recall_total += len(set(approx[: int(topk)]) & exact_sets[idx]) / max(1, int(topk))
                recall = recall_total / max(1, len(queries))
                row = SampleRow(
                    source_table=str(source_table),
                    table_vectors=int(n_vectors),
                    topk=int(topk),
                    ef_search=int(ef_search),
                    hnsw_time_ms=1000.0 * hnsw_elapsed / max(1, len(queries)),
                    exact_time_ms=float(avg_exact_ms),
                    recall=float(recall),
                )
                rows.append(row)
                payload.append(asdict(row))
        return rows, payload
    finally:
        try:
            cur.execute("SET enable_seqscan = on;")
            cur.execute("SET enable_indexscan = on;")
            cur.execute("SET enable_bitmapscan = on;")
        except Exception:
            pass
        cur.close()
        conn.close()


def fit_linear_search(rows: Sequence[SampleRow]) -> FitRow:
    if not rows:
        raise ValueError("No rows to fit")
    exact_by_size: dict[int, list[float]] = {}
    for row in rows:
        exact_by_size.setdefault(int(row.table_vectors), []).append(float(row.exact_time_ms))
    points = [(size, float(np.mean(times))) for size, times in sorted(exact_by_size.items())]
    x = np.asarray([[float(size), 1.0] for size, _time_ms in points], dtype=np.float64)
    y = np.asarray([float(time_ms) for _size, time_ms in points], dtype=np.float64)
    coef, pred, rmse, mae, r2 = _fit_line(x, y)
    return FitRow(
        model_name="linear_scan_fit",
        a=float(coef[0]),
        b=float(coef[1]),
        c=0.0,
        rmse_ms=float(rmse),
        mae_ms=float(mae),
        r2=float(r2),
        sample_count=int(len(points)),
        formula="linear_time_ms(N)=a*N+b",
    )


def find_crossings(rows: Sequence[SampleRow], fit: FitRow) -> list[IntersectionRow]:
    grouped: dict[tuple[int, int], list[SampleRow]] = {}
    for row in rows:
        grouped.setdefault((int(row.topk), int(row.ef_search)), []).append(row)
    crossings: list[IntersectionRow] = []
    for (topk, ef_search), group in sorted(grouped.items()):
        ordered = sorted(group, key=lambda item: int(item.table_vectors))
        previous: tuple[SampleRow, float, float] | None = None
        found = False
        for row in ordered:
            linear_pred = fit.a * float(row.table_vectors) + fit.b
            delta = float(row.hnsw_time_ms - linear_pred)
            if previous is not None:
                prev_row, prev_delta, prev_linear = previous
                if prev_delta == 0.0:
                    crossings.append(
                        IntersectionRow(
                            topk=int(topk),
                            table_vectors=int(prev_row.table_vectors),
                            ef_search=int(ef_search),
                            hnsw_time_ms=float(prev_row.hnsw_time_ms),
                            linear_time_ms=float(prev_linear),
                            delta_ms=0.0,
                            status="cross_on_previous_point",
                        )
                    )
                    found = True
                    break
                if prev_delta * delta <= 0.0:
                    x0 = float(prev_row.table_vectors)
                    x1 = float(row.table_vectors)
                    denom = float(delta - prev_delta)
                    if denom == 0.0 or x1 == x0:
                        ratio = 0.5
                        cross_vectors = x1
                    else:
                        ratio = float((0.0 - prev_delta) / denom)
                        ratio = max(0.0, min(1.0, ratio))
                        cross_vectors = x0 + ratio * (x1 - x0)
                    cross_hnsw = float(prev_row.hnsw_time_ms) + ratio * float(row.hnsw_time_ms - prev_row.hnsw_time_ms)
                    cross_linear = fit.a * cross_vectors + fit.b
                    crossings.append(
                        IntersectionRow(
                            topk=int(topk),
                            table_vectors=int(round(cross_vectors)),
                            ef_search=int(ef_search),
                            hnsw_time_ms=float(cross_hnsw),
                            linear_time_ms=float(cross_linear),
                            delta_ms=float(cross_hnsw - cross_linear),
                            status=f"cross_near_N={cross_vectors:.2f}",
                        )
                    )
                    found = True
                    break
            previous = (row, delta, linear_pred)
        if not found:
            nearest = min(ordered, key=lambda item: abs(float(item.hnsw_time_ms) - (fit.a * float(item.table_vectors) + fit.b)))
            linear_pred = fit.a * float(nearest.table_vectors) + fit.b
            crossings.append(
                IntersectionRow(
                    topk=int(topk),
                    table_vectors=int(nearest.table_vectors),
                    ef_search=int(ef_search),
                    hnsw_time_ms=float(nearest.hnsw_time_ms),
                    linear_time_ms=float(linear_pred),
                    delta_ms=float(nearest.hnsw_time_ms - linear_pred),
                    status="no_crossing_in_range",
                )
            )
    return crossings


def _write_plot(output_dir: Path, rows: Sequence[SampleRow], fit: FitRow, crossings: Sequence[IntersectionRow]) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    if not rows:
        return written

    grouped: dict[tuple[int, int], list[SampleRow]] = {}
    for row in rows:
        grouped.setdefault((int(row.topk), int(row.ef_search)), []).append(row)

    exact_by_size: dict[int, list[float]] = {}
    for row in rows:
        exact_by_size.setdefault(int(row.table_vectors), []).append(float(row.exact_time_ms))
    exact_points = [(size, float(np.mean(times))) for size, times in sorted(exact_by_size.items())]

    fig, ax = plt.subplots(figsize=(7.0, 4.8))
    color_map = plt.cm.tab10
    for idx, ((topk, ef_search), group) in enumerate(sorted(grouped.items())):
        ordered = sorted(group, key=lambda item: item.table_vectors)
        xs = [item.table_vectors for item in ordered]
        hnsw = [item.hnsw_time_ms for item in ordered]
        color = color_map(idx % 10)
        ax.plot(xs, hnsw, marker="o", linewidth=1.8, color=color, label=f"HNSW k={topk}, ef={ef_search}")
    ax.plot(
        [size for size, _time_ms in exact_points],
        [time_ms for _size, time_ms in exact_points],
        marker="s",
        linewidth=2,
        linestyle="--",
        color="dimgray",
        label="No-index measured",
    )
    fit_x = np.linspace(float(min(row.table_vectors for row in rows)), float(max(row.table_vectors for row in rows)), 200)
    fit_y = fit.a * fit_x + fit.b
    ax.plot(fit_x, fit_y, color="black", linewidth=2.5, label=f"No-index fit: {fit.formula}")
    for row in crossings:
        ax.scatter([row.table_vectors], [row.linear_time_ms], color="red", zorder=5)
    ax.set_xlabel("N (table vectors)")
    ax.set_ylabel("Latency (ms)")
    ax.set_title("Veda HNSW vs no-index linear scan")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    path = output_dir / "veda_hnsw_vs_linear.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    residuals = [time_ms - (fit.a * size + fit.b) for size, time_ms in exact_points]
    ax.plot([size for size, _time_ms in exact_points], residuals, marker="o")
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xlabel("N (table vectors)")
    ax.set_ylabel("Residual (ms)")
    ax.set_title("No-index measured residual vs a*N+b")
    ax.grid(True, linestyle="--", alpha=0.35)
    fig.tight_layout()
    path = output_dir / "veda_hnsw_vs_linear_residual.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)
    return written


def _write_report(path: Path, *, fit: FitRow, crossings: Sequence[IntersectionRow], rows: Sequence[SampleRow], plot_paths: Sequence[Path], raw_csv: Path) -> None:
    plot_lines = [f"- {p}" for p in plot_paths] or ["- matplotlib unavailable"]
    path.write_text(
        "\n".join(
            [
                "# Veda HNSW vs Linear Scan",
                "",
                f"- linear fit: `{fit.formula}`",
                f"- a: {fit.a:.8f}",
                f"- b: {fit.b:.8f}",
                f"- r2: {fit.r2:.6f}",
                f"- sample_count: {fit.sample_count}",
                f"- topk values: {sorted({int(r.topk) for r in rows})}",
                "",
                "## Crossings",
                "",
                "| topk | N | ef_search | hnsw_ms | linear_ms | delta_ms | status |",
                "|---:|---:|---:|---:|---:|---:|---|",
                *[
                    f"| {row.topk} | {row.table_vectors} | {row.ef_search} | {row.hnsw_time_ms:.4f} | {row.linear_time_ms:.4f} | {row.delta_ms:.4f} | {row.status} |"
                    for row in crossings
                ],
                "",
                "## Artifacts",
                "",
                *plot_lines,
                f"- raw_csv: {raw_csv}",
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Veda HNSW against no-index scan and fit linear scan time as a*N+b.")
    parser.add_argument("--source-table", default=DEFAULT_SOURCE_TABLE)
    parser.add_argument("--sizes", default="5000,20000,80000")
    parser.add_argument("--ef-values", default="1,5,10,20,40,80,120,200")
    parser.add_argument("--topk", default="10")
    parser.add_argument("--query-limit", type=int, default=50)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--query-file", type=Path, default=DEFAULT_QUERY_FILE)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--tag", default="veda_hnsw_vs_linear")
    parser.add_argument("--create-index", action="store_true")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--table-prefix", default="veda_linear_dbg")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    sizes = _parse_int_list(args.sizes)
    ef_values = _parse_int_list(args.ef_values)
    topk_values = _parse_int_list(args.topk)
    if len(topk_values) != 1:
        raise ValueError("--topk must contain exactly one value")
    topk = int(topk_values[0])

    rows, payload = collect_samples(
        source_table=str(args.source_table),
        sizes=sizes,
        query_limit=int(args.query_limit),
        query_offset=int(args.query_offset),
        topk=topk,
        ef_values=ef_values,
        create_index=bool(args.create_index),
        rebuild=bool(args.rebuild),
        table_prefix=str(args.table_prefix),
        hnsw_m=int(args.hnsw_m),
        hnsw_ef_construction=int(args.hnsw_ef_construction),
        query_file=args.query_file,
    )

    fit = fit_linear_search(rows)
    crossings = find_crossings(rows, fit)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_csv = output_dir / f"{args.tag}_raw.csv"
    fit_json = output_dir / f"{args.tag}_fit.json"
    crossings_csv = output_dir / f"{args.tag}_crossings.csv"
    report_md = output_dir / f"{args.tag}_report.md"

    _write_csv(raw_csv, [asdict(row) for row in rows])
    _write_csv(crossings_csv, [asdict(row) for row in crossings])
    _write_json(fit_json, asdict(fit))
    _write_json(output_dir / f"{args.tag}_samples.json", payload)
    plots = _write_plot(output_dir, rows, fit, crossings)
    _write_report(report_md, fit=fit, crossings=crossings, rows=rows, plot_paths=plots, raw_csv=raw_csv)

    print(f"[veda-vs-hnsw-linear] fit a={fit.a:.8f}, b={fit.b:.8f}, r2={fit.r2:.4f}")
    for row in crossings:
        print(
            f"[veda-vs-hnsw-linear] topk={row.topk}, N={row.table_vectors}, ef={row.ef_search}, hnsw_ms={row.hnsw_time_ms:.4f}, linear_ms={row.linear_time_ms:.4f}, status={row.status}"
        )
    print(f"[veda-vs-hnsw-linear] wrote {raw_csv}")
    print(f"[veda-vs-hnsw-linear] wrote {crossings_csv}")
    print(f"[veda-vs-hnsw-linear] wrote {fit_json}")
    print(f"[veda-vs-hnsw-linear] wrote {report_md}")
    for path in plots:
        print(f"[veda-vs-hnsw-linear] wrote {path}")


if __name__ == "__main__":
    main()
