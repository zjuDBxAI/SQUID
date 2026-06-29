from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Iterable

import numpy as np
from psycopg2 import sql

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from controller.kmeans.train.pg_calibration import (  # noqa: E402
    load_query_vectors,
    parse_int_list,
    prepare_calibration_tables,
)
from services.config import get_db_connection  # noqa: E402


DEFAULT_QUERY_FILE = PROJECT_ROOT / "basic_benchmark" / "query_dataset.json"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "result"


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
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
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _parse_execution_time_ms(explain_rows: Iterable[tuple[object, ...]]) -> float:
    for row in explain_rows:
        line = str(row[0])
        match = re.search(r"Execution Time:\s*([0-9.]+)\s*ms", line)
        if match:
            return float(match.group(1))
    raise RuntimeError("Could not parse Execution Time from EXPLAIN ANALYZE output")


def _explain_hnsw_latency_ms(cur, table_name: str, query_vector: str, *, topk: int, ef_search: int) -> float:
    cur.execute("SET enable_indexscan = on;")
    cur.execute("SET enable_bitmapscan = on;")
    cur.execute("SET enable_seqscan = off;")
    cur.execute(sql.SQL("SET hnsw.ef_search = {};").format(sql.Literal(max(1, int(ef_search)))))
    cur.execute(
        sql.SQL(
            """
            EXPLAIN ANALYZE
            SELECT block_id
            FROM {}
            ORDER BY vector <-> %s
            LIMIT %s;
            """
        ).format(sql.Identifier(table_name)),
        (query_vector, int(topk)),
    )
    return _parse_execution_time_ms(cur.fetchall())


def _summary(latencies_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(latencies_ms, dtype=np.float64)
    return {
        "mean_latency_ms": float(np.mean(arr)),
        "median_latency_ms": float(np.median(arr)),
        "p95_latency_ms": float(np.percentile(arr, 95)),
    }


def _target_latency(row: dict[str, object], stat: str) -> float:
    key = f"{stat}_latency_ms"
    return float(row[key])


def _fit_line(
    x: np.ndarray,
    y: np.ndarray,
    *,
    nonnegative: bool = True,
) -> tuple[np.ndarray, float, float, float]:
    if nonnegative:
        try:
            from scipy.optimize import lsq_linear

            result = lsq_linear(x, y, bounds=(0.0, np.inf), max_iter=10000)
            coef = result.x
        except Exception:
            coef, *_ = np.linalg.lstsq(x, y, rcond=None)
            coef = np.maximum(coef, 0.0)
    else:
        coef, *_ = np.linalg.lstsq(x, y, rcond=None)

    pred = x @ coef
    residuals = pred - y
    rmse = float(np.sqrt(np.mean(np.square(residuals))))
    mae = float(np.mean(np.abs(residuals)))
    denom = float(np.sum(np.square(y - np.mean(y))))
    r2 = 1.0 - float(np.sum(np.square(residuals))) / denom if denom > 0 else 1.0
    return coef, rmse, mae, r2


def _measure_table(
    table_name: str,
    queries: list[dict[str, object]],
    *,
    topk: int,
    ef_search: int,
    warmup_queries: int,
) -> list[float]:
    conn = get_db_connection()
    conn.autocommit = True
    cur = conn.cursor()
    try:
        warmups = min(max(0, int(warmup_queries)), len(queries))
        for query in queries[:warmups]:
            _explain_hnsw_latency_ms(
                cur,
                table_name,
                str(query["query_vector"]),
                topk=int(topk),
                ef_search=int(ef_search),
            )
        return [
            _explain_hnsw_latency_ms(
                cur,
                table_name,
                str(query["query_vector"]),
                topk=int(topk),
                ef_search=int(ef_search),
            )
            for query in queries
        ]
    finally:
        try:
            cur.execute("SET enable_seqscan = on;")
        except Exception:
            pass
        cur.close()
        conn.close()


def run_calibration(args: argparse.Namespace) -> dict[str, object]:
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    sizes = parse_int_list(args.sizes)
    efs_values = parse_int_list(args.efs_values)
    if 1 not in efs_values:
        efs_values = [1] + efs_values
    queries = load_query_vectors(args.query_file, limit=int(args.query_limit), topk=int(args.query_topk))

    table_by_size = prepare_calibration_tables(
        sizes,
        source_table=str(args.source_table),
        table_prefix=str(args.table_prefix),
        rebuild=bool(args.rebuild),
        hnsw_m=int(args.hnsw_m),
        hnsw_ef_construction=int(args.hnsw_ef_construction),
    )

    size_rows: list[dict[str, object]] = []
    print("[veda-cost-train] size sweep: fixed efs=1")
    for n_vectors, table_name in sorted(table_by_size.items()):
        latencies = _measure_table(
            table_name,
            queries,
            topk=int(args.query_topk),
            ef_search=1,
            warmup_queries=int(args.warmup_queries),
        )
        row = {
            "table_name": table_name,
            "n_vectors": int(n_vectors),
            "log2_1p_n": math.log2(1.0 + float(n_vectors)),
            "ef_search": 1,
            "topk": int(args.query_topk),
            "query_count": len(queries),
            **_summary(latencies),
        }
        size_rows.append(row)
        print(
            "[veda-cost-train] "
            f"N={n_vectors}, ef=1, median_ms={row['median_latency_ms']:.4f}, mean_ms={row['mean_latency_ms']:.4f}"
        )

    y_size = np.asarray([_target_latency(row, args.latency_stat) for row in size_rows], dtype=np.float64)
    x_size = np.asarray([[float(row["log2_1p_n"]), 1.0] for row in size_rows], dtype=np.float64)
    size_coef, size_rmse, size_mae, size_r2 = _fit_line(x_size, y_size, nonnegative=not args.allow_negative)
    a = float(size_coef[0])
    c1 = float(size_coef[1])

    fixed_size = int(args.fixed_size) if args.fixed_size else max(sizes)
    if fixed_size not in table_by_size:
        raise ValueError(f"--fixed-size {fixed_size} must be one of --sizes {sizes}")
    fixed_table = table_by_size[fixed_size]

    efs_rows: list[dict[str, object]] = []
    print(f"[veda-cost-train] efs sweep: fixed N={fixed_size}")
    for ef_search in efs_values:
        latencies = _measure_table(
            fixed_table,
            queries,
            topk=int(args.query_topk),
            ef_search=int(ef_search),
            warmup_queries=int(args.warmup_queries),
        )
        row = {
            "table_name": fixed_table,
            "n_vectors": int(fixed_size),
            "log2_1p_n": math.log2(1.0 + float(fixed_size)),
            "ef_search": int(ef_search),
            "efs_log2_efs": float(ef_search) * math.log2(max(2.0, float(ef_search))),
            "topk": int(args.query_topk),
            "query_count": len(queries),
            **_summary(latencies),
        }
        efs_rows.append(row)
        print(
            "[veda-cost-train] "
            f"N={fixed_size}, ef={ef_search}, median_ms={row['median_latency_ms']:.4f}, "
            f"mean_ms={row['mean_latency_ms']:.4f}"
        )

    y_efs = np.asarray([_target_latency(row, args.latency_stat) for row in efs_rows], dtype=np.float64)
    x_efs_linear = np.asarray([[float(row["ef_search"]), 1.0] for row in efs_rows], dtype=np.float64)
    linear_coef, linear_rmse, linear_mae, linear_r2 = _fit_line(
        x_efs_linear,
        y_efs,
        nonnegative=not args.allow_negative,
    )
    b = float(linear_coef[0])
    c2 = float(linear_coef[1])

    x_efs_log = np.asarray([[float(row["efs_log2_efs"]), 1.0] for row in efs_rows], dtype=np.float64)
    log_coef, log_rmse, log_mae, log_r2 = _fit_line(
        x_efs_log,
        y_efs,
        nonnegative=not args.allow_negative,
    )

    c_from_size = c1 - b * 1.0
    c_from_efs = c2 - a * math.log2(1.0 + float(fixed_size))
    c = 0.5 * (c_from_size + c_from_efs)

    selected_efs_term = "linear_efs" if linear_r2 >= log_r2 else "efs_log_efs"
    payload: dict[str, object] = {
        "model_name": "veda_appendix_b_hnsw_cost",
        "formula": "C_theta(N,efs)=a*log2(1+N)+b*efs+c",
        "latency_unit": "ms",
        "latency_stat": str(args.latency_stat),
        "a": a,
        "b": b,
        "c": c,
        "c1_size_intercept": c1,
        "c2_efs_intercept": c2,
        "c_from_size": c_from_size,
        "c_from_efs": c_from_efs,
        "selected_efs_term_by_r2": selected_efs_term,
        "size_sweep": {
            "formula": "T_size(N)=a*log2(1+N)+c1",
            "rmse_ms": size_rmse,
            "mae_ms": size_mae,
            "r2": size_r2,
        },
        "efs_sweep_linear": {
            "formula": "T_efs(efs)=b*efs+c2",
            "b": b,
            "c2": c2,
            "rmse_ms": linear_rmse,
            "mae_ms": linear_mae,
            "r2": linear_r2,
        },
        "efs_sweep_log": {
            "formula": "T_efs(efs)=b_log*efs*log2(efs)+c2_log",
            "b_log": float(log_coef[0]),
            "c2_log": float(log_coef[1]),
            "rmse_ms": log_rmse,
            "mae_ms": log_mae,
            "r2": log_r2,
        },
        "sizes": sizes,
        "efs_values": efs_values,
        "fixed_size": fixed_size,
        "query_file": str(args.query_file),
        "query_count": len(queries),
        "query_topk": int(args.query_topk),
        "source_table": str(args.source_table),
        "table_prefix": str(args.table_prefix),
        "hnsw_m": int(args.hnsw_m),
        "hnsw_ef_construction": int(args.hnsw_ef_construction),
        "size_sweep_csv": str(output_dir / "veda_cost_size_sweep.csv"),
        "efs_sweep_csv": str(output_dir / "veda_cost_efs_sweep.csv"),
    }

    _write_csv(output_dir / "veda_cost_size_sweep.csv", size_rows)
    _write_csv(output_dir / "veda_cost_efs_sweep.csv", efs_rows)
    _write_json(output_dir / "veda_cost_fit.json", payload)
    (output_dir / "veda_cost_fit.md").write_text(
        "\n".join(
            [
                "# Veda Appendix B Cost Model Fit",
                "",
                "Formula:",
                "",
                "`C_theta(N,efs)=a*log2(1+N)+b*efs+c`",
                "",
                f"latency_stat = `{args.latency_stat}`",
                f"a = {a:.10f}",
                f"b = {b:.10f}",
                f"c = {c:.10f}",
                "",
                f"size_sweep_r2 = {size_r2:.6f}",
                f"efs_linear_r2 = {linear_r2:.6f}",
                f"efs_log_r2 = {log_r2:.6f}",
                f"selected_efs_term_by_r2 = `{selected_efs_term}`",
                "",
                f"fixed_size = {fixed_size}",
                f"sizes = {','.join(str(v) for v in sizes)}",
                f"efs_values = {','.join(str(v) for v in efs_values)}",
                f"query_count = {len(queries)}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return payload


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce Veda Appendix B HNSW cost-model calibration on current PostgreSQL documentblocks data."
    )
    parser.add_argument("--sizes", default="5000,20000,80000", help="Comma-separated sampled table sizes.")
    parser.add_argument("--efs-values", default="1,5,10,20,40,80,120,200", help="Comma-separated ef_search values.")
    parser.add_argument("--fixed-size", type=int, default=0, help="N used for the efs sweep; defaults to max --sizes.")
    parser.add_argument("--query-file", type=Path, default=DEFAULT_QUERY_FILE)
    parser.add_argument("--query-limit", type=int, default=50)
    parser.add_argument("--query-topk", type=int, default=1, help="Use 1 to match Veda Appendix B size-sweep setup.")
    parser.add_argument("--warmup-queries", type=int, default=3)
    parser.add_argument("--source-table", default="documentblocks")
    parser.add_argument("--table-prefix", default="veda_cost_train")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--rebuild", action="store_true", help="Rebuild sampled calibration tables and HNSW indexes.")
    parser.add_argument("--latency-stat", choices=["mean", "median", "p95"], default="median")
    parser.add_argument("--allow-negative", action="store_true", help="Use unconstrained least squares.")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    payload = run_calibration(args)
    print("[veda-cost-train] fitted model")
    print(f"  C(N,efs) = {payload['a']:.10f} * log2(1+N) + {payload['b']:.10f} * efs + {payload['c']:.10f}")
    print(f"  size_r2 = {payload['size_sweep']['r2']:.6f}")
    print(f"  efs_linear_r2 = {payload['efs_sweep_linear']['r2']:.6f}")
    print(f"  efs_log_r2 = {payload['efs_sweep_log']['r2']:.6f}")
    print(f"  selected = {payload['selected_efs_term_by_r2']}")
    print(f"  wrote {args.output_dir / 'veda_cost_fit.json'}")


if __name__ == "__main__":
    main()
