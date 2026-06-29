from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Callable, Iterable, Sequence

import numpy as np
from psycopg2 import sql
from scipy.optimize import curve_fit


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from services.config import get_db_connection, get_maintenance_settings  # noqa: E402


RESULT_DIR = Path(__file__).resolve().parent / "result"


@dataclass(frozen=True)
class RecallRow:
    table_name: str
    table_vectors: int
    filter_selectivity: float
    filter_threshold: int
    topk: int
    ef_search: int
    normalized_effort: float
    recall: float
    query_count: int
    avg_hnsw_time_ms: float
    avg_exact_time_ms: float
    avg_returned_rows: float


@dataclass(frozen=True)
class FitRow:
    model: str
    scope: str
    n_points: int
    r2: float | None
    rmse: float | None
    mae: float | None
    params_json: str


def parse_int_list(raw: str | Iterable[int]) -> list[int]:
    if isinstance(raw, str):
        values = [int(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [int(value) for value in raw]
    return sorted(dict.fromkeys(value for value in values if value > 0))


def parse_float_list(raw: str | Iterable[float]) -> list[float]:
    if isinstance(raw, str):
        values = [float(part.strip()) for part in raw.split(",") if part.strip()]
    else:
        values = [float(value) for value in raw]
    cleaned = []
    for value in values:
        if 0.0 < value <= 1.0:
            cleaned.append(value)
    return sorted(dict.fromkeys(cleaned), reverse=True)


def _safe_table_suffix(value: int) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_")


def _safe_index_name(table_name: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", table_name).strip("_").lower()
    return f"{normalized[:45]}_hnsw_idx"


def _write_csv(path: Path, rows: Sequence[object], fallback_fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(asdict(rows[0]).keys()) if rows else list(fallback_fields)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _table_exists(cur, table_name: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM information_schema.tables
        WHERE table_schema = current_schema()
          AND table_name = %s
        LIMIT 1;
        """,
        [str(table_name)],
    )
    return cur.fetchone() is not None


def _table_count(cur, table_name: str) -> int:
    cur.execute(sql.SQL("SELECT COUNT(*) FROM {};").format(sql.Identifier(table_name)))
    return int(cur.fetchone()[0])


def _hnsw_index_exists(cur, table_name: str) -> bool:
    cur.execute(
        """
        SELECT 1
        FROM pg_indexes
        WHERE schemaname = current_schema()
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


def _ensure_sample_table(
    cur,
    *,
    source_table: str,
    table_name: str,
    table_vectors: int,
    rebuild: bool,
    hnsw_m: int,
    hnsw_ef_construction: int,
) -> None:
    if rebuild and _table_exists(cur, table_name):
        cur.execute(sql.SQL("DROP TABLE {};").format(sql.Identifier(table_name)))

    if not _table_exists(cur, table_name):
        print(f"[filter-recall] create {table_name} with N={table_vectors}")
        cur.execute(
            sql.SQL(
                """
                CREATE UNLOGGED TABLE {} AS
                SELECT
                    block_id,
                    document_id,
                    vector,
                    ((hashtext(block_id::text)::bigint + 2147483648) % 1000000)::integer AS filter_bucket
                FROM {}
                WHERE vector IS NOT NULL
                ORDER BY md5(block_id::text)
                LIMIT {};
                """
            ).format(
                sql.Identifier(table_name),
                sql.Identifier(source_table),
                sql.Literal(int(table_vectors)),
            )
        )
        cur.execute(sql.SQL("ALTER TABLE {} ADD PRIMARY KEY (block_id);").format(sql.Identifier(table_name)))
        cur.execute(sql.SQL("CREATE INDEX ON {} (filter_bucket);").format(sql.Identifier(table_name)))

    actual = _table_count(cur, table_name)
    if actual < int(table_vectors):
        raise RuntimeError(f"{table_name} has only {actual} rows, expected {table_vectors}")

    if not _hnsw_index_exists(cur, table_name):
        index_name = _safe_index_name(table_name)
        print(f"[filter-recall] build HNSW index {index_name} on {table_name}")
        _configure_index_build(cur)
        cur.execute(
            sql.SQL(
                """
                CREATE INDEX {} ON {} USING hnsw (vector vector_l2_ops)
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
            ORDER BY md5((block_id + 7919)::text)
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
    filter_threshold: int,
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
        cur.execute("SET enable_bitmapscan = off;")
        cur.execute("SET enable_seqscan = off;")
        cur.execute(sql.SQL("SET hnsw.ef_search = {};").format(sql.Literal(int(ef_search or 40))))

    cur.execute(
        sql.SQL(
            """
            SELECT block_id
            FROM {}
            WHERE filter_bucket < %s
            ORDER BY vector <-> %s
            LIMIT %s;
            """
        ).format(sql.Identifier(table_name)),
        [int(filter_threshold), query_vector, int(topk)],
    )
    return [int(row[0]) for row in cur.fetchall()]


def probe_curves(
    *,
    source_table: str,
    n_values: Sequence[int],
    rho_values: Sequence[float],
    ef_values: Sequence[int],
    topk: int,
    query_limit: int,
    query_offset: int,
    rebuild: bool,
    table_prefix: str,
    hnsw_m: int,
    hnsw_ef_construction: int,
) -> list[RecallRow]:
    rows: list[RecallRow] = []
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    try:
        for n_value in n_values:
            table_name = f"{table_prefix}_{_safe_table_suffix(int(n_value))}"
            _ensure_sample_table(
                cur,
                source_table=str(source_table),
                table_name=table_name,
                table_vectors=int(n_value),
                rebuild=bool(rebuild),
                hnsw_m=int(hnsw_m),
                hnsw_ef_construction=int(hnsw_ef_construction),
            )
            actual_n = _table_count(cur, table_name)
            queries = _load_query_vectors(cur, table_name, int(query_limit), int(query_offset))
            print(
                f"[filter-recall] table={table_name}, N={actual_n}, "
                f"queries={len(queries)}, rho={list(rho_values)}, ef={list(ef_values)}"
            )

            for rho in rho_values:
                threshold = max(1, min(1000000, int(round(float(rho) * 1000000.0))))
                exact_results: list[set[int]] = []
                exact_elapsed = 0.0
                for idx, query_vector in enumerate(queries, start=1):
                    started = time.perf_counter()
                    exact = _query_block_ids(
                        cur,
                        table_name,
                        query_vector,
                        int(topk),
                        threshold,
                        exact=True,
                    )
                    exact_elapsed += time.perf_counter() - started
                    exact_results.append(set(exact))
                    if idx % 10 == 0 or idx == len(queries):
                        print(f"[filter-recall] exact N={actual_n}, rho={rho:g}, {idx}/{len(queries)}")
                avg_exact_ms = 1000.0 * exact_elapsed / max(1, len(queries))

                for ef_search in ef_values:
                    hnsw_elapsed = 0.0
                    total_recall = 0.0
                    total_returned = 0
                    for idx, query_vector in enumerate(queries):
                        started = time.perf_counter()
                        approx = _query_block_ids(
                            cur,
                            table_name,
                            query_vector,
                            int(topk),
                            threshold,
                            exact=False,
                            ef_search=int(ef_search),
                        )
                        hnsw_elapsed += time.perf_counter() - started
                        total_returned += len(approx)
                        exact = exact_results[idx]
                        total_recall += len(set(approx) & exact) / max(1, len(exact))
                    recall = total_recall / max(1, len(queries))
                    row = RecallRow(
                        table_name=str(table_name),
                        table_vectors=int(actual_n),
                        filter_selectivity=float(rho),
                        filter_threshold=int(threshold),
                        topk=int(topk),
                        ef_search=int(ef_search),
                        normalized_effort=float(ef_search) * float(rho) / max(1.0, float(topk)),
                        recall=float(recall),
                        query_count=int(len(queries)),
                        avg_hnsw_time_ms=1000.0 * hnsw_elapsed / max(1, len(queries)),
                        avg_exact_time_ms=float(avg_exact_ms),
                        avg_returned_rows=float(total_returned) / max(1, len(queries)),
                    )
                    rows.append(row)
                    print(
                        "[filter-recall] "
                        f"N={row.table_vectors}, rho={row.filter_selectivity:g}, "
                        f"ef={row.ef_search}, x={row.normalized_effort:.3f}, "
                        f"recall={row.recall:.4f}, rows={row.avg_returned_rows:.2f}, "
                        f"hnsw_ms={row.avg_hnsw_time_ms:.3f}"
                    )
        return rows
    finally:
        try:
            cur.execute("SET enable_seqscan = on;")
            cur.execute("SET enable_indexscan = on;")
            cur.execute("SET enable_bitmapscan = on;")
        except Exception:
            pass
        cur.close()
        conn.close()


def _fit_quality(y: np.ndarray, pred: np.ndarray) -> tuple[float | None, float, float]:
    residual = y - pred
    rmse = float(math.sqrt(np.mean(residual * residual)))
    mae = float(np.mean(np.abs(residual)))
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y - float(np.mean(y))) ** 2))
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None
    return r2, rmse, mae


def _fit_model(
    name: str,
    scope: str,
    x: np.ndarray,
    y: np.ndarray,
    fn: Callable[..., np.ndarray],
    p0: Sequence[float],
    bounds: tuple[Sequence[float], Sequence[float]],
    param_names: Sequence[str],
) -> FitRow | None:
    try:
        params, _ = curve_fit(fn, x, y, p0=np.asarray(p0, dtype=np.float64), bounds=bounds, maxfev=50000)
        pred = np.clip(fn(x, *params), 0.0, 1.0)
        r2, rmse, mae = _fit_quality(y, pred)
        payload = {name_: float(value) for name_, value in zip(param_names, params)}
        return FitRow(
            model=str(name),
            scope=str(scope),
            n_points=int(len(x)),
            r2=r2,
            rmse=float(rmse),
            mae=float(mae),
            params_json=json.dumps(payload, sort_keys=True),
        )
    except Exception as exc:
        print(f"[filter-recall] fit failed: model={name}, scope={scope}, error={exc}")
        return None


def fit_recall_models(rows: Sequence[RecallRow]) -> list[FitRow]:
    def exp_fixed(x: np.ndarray, lam: float) -> np.ndarray:
        return 1.0 - np.exp(-lam * x)

    def exp_rmax(x: np.ndarray, rmax: float, lam: float) -> np.ndarray:
        return rmax * (1.0 - np.exp(-lam * x))

    def michaelis(x: np.ndarray, rmax: float, km: float) -> np.ndarray:
        return rmax * x / (km + x)

    def hill(x: np.ndarray, rmax: float, km: float, h: float) -> np.ndarray:
        xp = np.power(np.maximum(x, 1e-12), h)
        kp = float(km) ** float(h)
        return rmax * xp / (kp + xp)

    scopes: dict[str, list[RecallRow]] = {"all": list(rows)}
    for n_value in sorted({int(row.table_vectors) for row in rows}):
        scopes[f"N={n_value}"] = [row for row in rows if int(row.table_vectors) == n_value]

    fit_rows: list[FitRow] = []
    for scope, selected in scopes.items():
        selected = [row for row in selected if row.normalized_effort > 0.0]
        if len(selected) < 5:
            continue
        x = np.asarray([row.normalized_effort for row in selected], dtype=np.float64)
        y = np.asarray([row.recall for row in selected], dtype=np.float64)
        candidates = [
            ("exp_fixed_rmax1", exp_fixed, [1.0], ([1e-9], [1000.0]), ["lambda"]),
            ("exp_rmax", exp_rmax, [1.0, 1.0], ([0.01, 1e-9], [1.0, 1000.0]), ["rmax", "lambda"]),
            ("michaelis_menten", michaelis, [1.0, 1.0], ([0.01, 1e-9], [1.0, 1000.0]), ["rmax", "K"]),
            ("hill", hill, [1.0, 1.0, 1.0], ([0.01, 1e-9, 0.05], [1.0, 1000.0, 10.0]), ["rmax", "K", "h"]),
        ]
        for name, fn, p0, bounds, param_names in candidates:
            row = _fit_model(name, scope, x, y, fn, p0, bounds, param_names)
            if row is not None:
                fit_rows.append(row)
    return fit_rows


def _plot_curves(output_dir: Path, rows: Sequence[RecallRow], fit_rows: Sequence[FitRow], tag: str) -> list[Path]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    colors = plt.cm.viridis(np.linspace(0.05, 0.95, max(1, len({row.filter_selectivity for row in rows}))))

    for n_value in sorted({int(row.table_vectors) for row in rows}):
        fig, ax = plt.subplots(figsize=(7.4, 4.8))
        selected_n = [row for row in rows if int(row.table_vectors) == n_value]
        rho_values = sorted({row.filter_selectivity for row in selected_n}, reverse=True)
        color_by_rho = {rho: colors[idx % len(colors)] for idx, rho in enumerate(rho_values)}
        for rho in rho_values:
            group = sorted([row for row in selected_n if row.filter_selectivity == rho], key=lambda row: row.ef_search)
            ax.plot(
                [row.ef_search for row in group],
                [row.recall for row in group],
                marker="o",
                linewidth=1.8,
                color=color_by_rho[rho],
                label=f"rho={rho:g}",
            )
        ax.set_xscale("log", base=2)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel("ef_search")
        ax.set_ylabel("Recall@10")
        ax.set_title(f"Recall@10 vs ef_search, N={n_value}")
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(ncol=2, fontsize=8)
        path = output_dir / f"{tag}_N{n_value}_recall_vs_efs.png"
        fig.tight_layout()
        fig.savefig(path, dpi=180)
        plt.close(fig)
        written.append(path)

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for n_value in sorted({int(row.table_vectors) for row in rows}):
        group = sorted([row for row in rows if int(row.table_vectors) == n_value], key=lambda row: row.normalized_effort)
        ax.scatter(
            [row.normalized_effort for row in group],
            [row.recall for row in group],
            s=24,
            alpha=0.75,
            label=f"N={n_value}",
        )
    best = None
    all_fit = [row for row in fit_rows if row.scope == "all" and row.r2 is not None]
    if all_fit:
        best = sorted(all_fit, key=lambda row: (row.rmse if row.rmse is not None else float("inf")))[0]
    if best is not None:
        xs = np.linspace(0.0, max(row.normalized_effort for row in rows) * 1.05, 300)
        params = json.loads(best.params_json)
        if best.model == "exp_fixed_rmax1":
            ys = 1.0 - np.exp(-float(params["lambda"]) * xs)
        elif best.model == "exp_rmax":
            ys = float(params["rmax"]) * (1.0 - np.exp(-float(params["lambda"]) * xs))
        elif best.model == "michaelis_menten":
            ys = float(params["rmax"]) * xs / (float(params["K"]) + xs)
        elif best.model == "hill":
            h = float(params["h"])
            xp = np.power(np.maximum(xs, 1e-12), h)
            kp = float(params["K"]) ** h
            ys = float(params["rmax"]) * xp / (kp + xp)
        else:
            ys = None
        if ys is not None:
            ax.plot(xs, np.clip(ys, 0.0, 1.0), color="black", linewidth=2.2, label=f"best fit: {best.model}")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("normalized effort = ef_search * rho / k")
    ax.set_ylabel("Recall@10")
    ax.set_title("Recall curve collapse by ef_search * rho / k")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(fontsize=8)
    path = output_dir / f"{tag}_normalized_collapse.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    written.append(path)

    return written


def _render_fit_table(fits: Sequence[FitRow]) -> str:
    lines = [
        "| scope | model | R2 | RMSE | MAE | params |",
        "|---|---|---:|---:|---:|---|",
    ]
    for row in sorted(fits, key=lambda item: (item.scope, item.rmse if item.rmse is not None else float("inf"))):
        r2 = "" if row.r2 is None else f"{row.r2:.6f}"
        lines.append(
            f"| {row.scope} | {row.model} | {r2} | {row.rmse:.6f} | {row.mae:.6f} | `{row.params_json}` |"
        )
    return "\n".join(lines)


def _render_recall_matrix(rows: Sequence[RecallRow]) -> str:
    lines: list[str] = []
    for n_value in sorted({int(row.table_vectors) for row in rows}):
        lines.append(f"### N={n_value}")
        selected_n = [row for row in rows if int(row.table_vectors) == n_value]
        ef_values = sorted({int(row.ef_search) for row in selected_n})
        rho_values = sorted({float(row.filter_selectivity) for row in selected_n}, reverse=True)
        by_key = {
            (float(row.filter_selectivity), int(row.ef_search)): float(row.recall)
            for row in selected_n
        }
        lines.append("| rho / ef | " + " | ".join(str(ef) for ef in ef_values) + " |")
        lines.append("|---:|" + "|".join("---:" for _ in ef_values) + "|")
        for rho in rho_values:
            values = [f"{by_key.get((rho, ef), 0.0):.4f}" for ef in ef_values]
            lines.append(f"| {rho:g} | " + " | ".join(values) + " |")
        lines.append("")
    return "\n".join(lines)


def _write_report(
    path: Path,
    *,
    raw_csv: Path,
    fit_csv: Path,
    summary_json: Path,
    plots: Sequence[Path],
    rows: Sequence[RecallRow],
    fits: Sequence[FitRow],
) -> None:
    best = None
    all_fit = [row for row in fits if row.scope == "all" and row.r2 is not None]
    if all_fit:
        best = sorted(all_fit, key=lambda row: row.rmse)[0]
    if best is None:
        recommendation = "No fit was produced."
    elif best.model in {"exp_fixed_rmax1", "exp_rmax"}:
        recommendation = (
            "The best global fit is a saturating exponential. A simple usable form is "
            "`Recall = Rmax * (1 - exp(-lambda * ef_search * rho / k))`."
        )
    elif best.model == "michaelis_menten":
        recommendation = (
            "The best global fit is Michaelis-Menten. A simple usable form is "
            "`Recall = Rmax * x / (K + x)`, where `x = ef_search * rho / k`."
        )
    else:
        recommendation = (
            "The best global fit is Hill. A simple usable form is "
            "`Recall = Rmax * x^h / (K^h + x^h)`, where `x = ef_search * rho / k`."
        )

    path.write_text(
        "\n".join(
            [
                "# Filtered ef_search vs Recall@10",
                "",
                "This debug run simulates permission filtering with a deterministic hash predicate.",
                "For each sampled table size `N`, it builds a separate HNSW index so that `N` changes the graph size.",
                "",
                "## Formula Check",
                "",
                "The fitted variable is:",
                "",
                "\\[",
                "x = \\frac{ef_s \\cdot \\rho}{k}",
                "\\]",
                "",
                recommendation,
                "",
                "## Fit Results",
                "",
                _render_fit_table(fits),
                "",
                "## Recall Matrix",
                "",
                _render_recall_matrix(rows),
                "",
                "## Artifacts",
                "",
                f"- raw_csv: `{raw_csv}`",
                f"- fit_csv: `{fit_csv}`",
                f"- summary_json: `{summary_json}`",
                *[f"- plot: `{plot}`" for plot in plots],
                "",
            ]
        ),
        encoding="utf-8",
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe Recall@10 as a function of table size N, filter selectivity rho, and HNSW ef_search."
    )
    parser.add_argument("--source-table", default="documentblocks")
    parser.add_argument("--n-values", default="10000,50000,100000")
    parser.add_argument("--rho-values", default="1.0,0.5,0.2,0.1,0.05,0.02")
    parser.add_argument("--efs-values", default="10,20,40,80,160,320,640,1000")
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--query-limit", type=int, default=20)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--table-prefix", default="kmeans_debug_filter_recall")
    parser.add_argument("--rebuild", action="store_true")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    parser.add_argument("--output-dir", type=Path, default=RESULT_DIR)
    parser.add_argument("--tag", default="filter_efs_recall")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    n_values = parse_int_list(args.n_values)
    rho_values = parse_float_list(args.rho_values)
    ef_values = parse_int_list(args.efs_values)
    if not n_values:
        raise ValueError("--n-values cannot be empty")
    if not rho_values:
        raise ValueError("--rho-values cannot be empty")
    if not ef_values:
        raise ValueError("--efs-values cannot be empty")

    rows = probe_curves(
        source_table=str(args.source_table),
        n_values=n_values,
        rho_values=rho_values,
        ef_values=ef_values,
        topk=int(args.topk),
        query_limit=int(args.query_limit),
        query_offset=int(args.query_offset),
        rebuild=bool(args.rebuild),
        table_prefix=str(args.table_prefix),
        hnsw_m=int(args.hnsw_m),
        hnsw_ef_construction=int(args.hnsw_ef_construction),
    )
    fits = fit_recall_models(rows)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tag = str(args.tag)
    raw_csv = output_dir / f"{tag}_raw.csv"
    fit_csv = output_dir / f"{tag}_fits.csv"
    summary_json = output_dir / f"{tag}_summary.json"
    report_md = output_dir / f"{tag}_report.md"

    _write_csv(
        raw_csv,
        rows,
        [
            "table_name",
            "table_vectors",
            "filter_selectivity",
            "filter_threshold",
            "topk",
            "ef_search",
            "normalized_effort",
            "recall",
            "query_count",
            "avg_hnsw_time_ms",
            "avg_exact_time_ms",
            "avg_returned_rows",
        ],
    )
    _write_csv(
        fit_csv,
        fits,
        ["model", "scope", "n_points", "r2", "rmse", "mae", "params_json"],
    )
    plots = _plot_curves(output_dir, rows, fits, tag)
    _write_json(
        summary_json,
        {
            "source_table": str(args.source_table),
            "n_values": n_values,
            "rho_values": rho_values,
            "ef_values": ef_values,
            "topk": int(args.topk),
            "query_limit": int(args.query_limit),
            "raw_csv": str(raw_csv),
            "fit_csv": str(fit_csv),
            "plots": [str(path) for path in plots],
            "fits": [asdict(row) for row in fits],
        },
    )
    _write_report(
        report_md,
        raw_csv=raw_csv,
        fit_csv=fit_csv,
        summary_json=summary_json,
        plots=plots,
        rows=rows,
        fits=fits,
    )

    print("[filter-recall] best all-scope fits:")
    for row in sorted([fit for fit in fits if fit.scope == "all"], key=lambda fit: fit.rmse)[:4]:
        r2 = "NA" if row.r2 is None else f"{row.r2:.6f}"
        print(f"  {row.model}: R2={r2}, RMSE={row.rmse:.6f}, params={row.params_json}")
    print(f"[filter-recall] wrote {raw_csv}")
    print(f"[filter-recall] wrote {fit_csv}")
    print(f"[filter-recall] wrote {summary_json}")
    print(f"[filter-recall] wrote {report_md}")
    for path in plots:
        print(f"[filter-recall] wrote {path}")


if __name__ == "__main__":
    main()
