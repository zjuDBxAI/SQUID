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
from typing import Iterable, Sequence

import numpy as np
from psycopg2 import sql


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from controller.kmeans.common import DEFAULT_QUERY_DATASET_PATH, TenantRoute  # noqa: E402
from controller.kmeans.cost_model import DEFAULT_COST_A, DEFAULT_COST_B, DEFAULT_COST_C  # noqa: E402
from controller.kmeans.search import (  # noqa: E402
    _allowed_pattern_ids,
    _base_ef,
    _build_candidate_query,
    _clamp_ef,
    _configure_search_session,
    _execute_candidate_search,
    _extract_execution_time_seconds,
    _probe_adaptive_ef,
)
from controller.kmeans.storage import load_tenant_routes  # noqa: E402
from services.config import get_db_connection  # noqa: E402


RESULT_DIR = Path(__file__).resolve().parent / "result"


@dataclass(frozen=True)
class RouteLatencyRow:
    query_index: int
    tenant_id: int
    partition_id: str
    table_name: str
    route_kind: str
    partition_vectors: int
    accessible_vectors: int
    selectivity: float
    inverse_selectivity: float
    pattern_count: int
    topk: int
    base_ef: int
    probe_rows: int
    probe_hits: int
    final_ef: int
    model_cost: float
    model_fixed_cost: float
    model_filter_cost: float
    sql_time_ms: float
    authorized_rows: int
    best_distance: float | None
    query_total_routes: int
    query_model_total_cost: float
    query_sql_total_ms: float


@dataclass(frozen=True)
class QueryLatencyRow:
    query_index: int
    tenant_id: int
    topk: int
    route_count: int
    model_total_cost: float
    sql_total_ms: float
    min_selectivity: float
    mean_selectivity: float
    max_inverse_selectivity: float
    total_partition_vectors: int
    total_accessible_vectors: int


def _parse_execution_time_ms(explain_rows: Iterable[tuple[object, ...]]) -> float:
    seconds = float(_extract_execution_time_seconds(explain_rows))
    return 1000.0 * seconds


def _explain_candidate_search_ms(cur, route: TenantRoute, *, query_vector: str, ef_search: int) -> float:
    ef_search = _clamp_ef(int(ef_search), route)
    cur.execute(sql.SQL("SET hnsw.ef_search = {};").format(sql.Literal(int(ef_search))))
    query, params = _build_candidate_query(route, query_vector=query_vector, candidate_limit=int(ef_search))
    cur.execute(sql.SQL("EXPLAIN ANALYZE {}").format(query), params)
    return _parse_execution_time_ms(cur.fetchall())


def _has_hnsw_index(cur, table_name: str) -> bool:
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


def _load_queries(path: Path, *, limit: int, offset: int) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise RuntimeError(f"Expected query dataset list, got {type(payload)!r}: {path}")
    sliced = payload[int(offset) : int(offset) + int(limit)]
    queries: list[dict[str, object]] = []
    for item in sliced:
        if not isinstance(item, dict):
            continue
        if "user_id" not in item or "query_vector" not in item:
            continue
        queries.append(item)
    if not queries:
        raise RuntimeError(f"No valid queries loaded from {path}")
    return queries


def _route_model_cost(route: TenantRoute, *, ef_search: int) -> tuple[float, float, float]:
    partition_vectors = max(1, int(route.partition_vector_count or 0))
    accessible_vectors = int(route.accessible_vector_count or 0)
    if accessible_vectors <= 0:
        return 0.0, 0.0, 0.0
    selectivity = min(1.0, max(1e-12, float(accessible_vectors) / float(partition_vectors)))
    inverse_selectivity = 1.0 / float(selectivity)
    log_vectors = math.log2(float(partition_vectors) + 1.0)
    fixed = float(DEFAULT_COST_A) * float(log_vectors) + float(DEFAULT_COST_C)
    effective_ef = min(float(partition_vectors), float(max(1, int(ef_search))) * float(inverse_selectivity))
    filtering = float(DEFAULT_COST_B) * float(effective_ef) * float(log_vectors)
    return float(fixed + filtering), float(fixed), float(filtering)


def _candidate_probe_stats(candidate_rows: Sequence[tuple[object, ...]], route: TenantRoute) -> tuple[int, int, float | None]:
    allowed = _allowed_pattern_ids(route)
    hit_count = 0
    best_distance: float | None = None
    for row in candidate_rows:
        if len(row) >= 5 and int(row[3]) in allowed:
            hit_count += 1
            distance = float(row[4])
            if best_distance is None or distance < best_distance:
                best_distance = distance
    return int(len(candidate_rows)), int(hit_count), best_distance


def _write_csv(path: Path, rows: Sequence[object], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _safe_corr(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    if float(np.std(xa)) <= 0.0 or float(np.std(ya)) <= 0.0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def _rankdata(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    order = np.argsort(arr, kind="mergesort")
    ranks = np.empty(len(arr), dtype=np.float64)
    i = 0
    while i < len(arr):
        j = i + 1
        while j < len(arr) and arr[order[j]] == arr[order[i]]:
            j += 1
        rank = 0.5 * (i + j - 1) + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def _safe_spearman(x: Sequence[float], y: Sequence[float]) -> float | None:
    if len(x) < 2 or len(y) < 2:
        return None
    return _safe_corr(_rankdata(x), _rankdata(y))


def _linear_fit_summary(x: Sequence[float], y: Sequence[float]) -> dict[str, float | None]:
    if len(x) < 2:
        return {"slope": None, "intercept": None, "r2": None, "mae_ms": None, "rmse_ms": None}
    xa = np.asarray(x, dtype=np.float64)
    ya = np.asarray(y, dtype=np.float64)
    design = np.column_stack([xa, np.ones_like(xa)])
    coef, *_ = np.linalg.lstsq(design, ya, rcond=None)
    pred = design @ coef
    residual = ya - pred
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((ya - float(np.mean(ya))) ** 2))
    return {
        "slope": float(coef[0]),
        "intercept": float(coef[1]),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else None,
        "mae_ms": float(np.mean(np.abs(residual))),
        "rmse_ms": float(math.sqrt(np.mean(residual * residual))),
    }


def _quantile(values: Sequence[float], q: float) -> float | None:
    if not values:
        return None
    return float(np.percentile(np.asarray(values, dtype=np.float64), float(q)))


def _bucket_summary(rows: Sequence[RouteLatencyRow]) -> list[dict[str, object]]:
    buckets = [
        ("rho_le_0.02", lambda r: r.selectivity <= 0.02),
        ("rho_0.02_0.05", lambda r: 0.02 < r.selectivity <= 0.05),
        ("rho_0.05_0.10", lambda r: 0.05 < r.selectivity <= 0.10),
        ("rho_0.10_0.25", lambda r: 0.10 < r.selectivity <= 0.25),
        ("rho_0.25_0.50", lambda r: 0.25 < r.selectivity <= 0.50),
        ("rho_gt_0.50", lambda r: r.selectivity > 0.50),
    ]
    output: list[dict[str, object]] = []
    for name, pred in buckets:
        selected = [row for row in rows if pred(row)]
        if not selected:
            output.append({"bucket": name, "count": 0})
            continue
        output.append(
            {
                "bucket": name,
                "count": len(selected),
                "model_cost_p50": _quantile([row.model_cost for row in selected], 50),
                "model_cost_p90": _quantile([row.model_cost for row in selected], 90),
                "sql_ms_p50": _quantile([row.sql_time_ms for row in selected], 50),
                "sql_ms_p90": _quantile([row.sql_time_ms for row in selected], 90),
                "final_ef_p50": _quantile([row.final_ef for row in selected], 50),
                "final_ef_p90": _quantile([row.final_ef for row in selected], 90),
                "partition_vectors_p50": _quantile([row.partition_vectors for row in selected], 50),
                "partition_vectors_p90": _quantile([row.partition_vectors for row in selected], 90),
            }
        )
    return output


def _plot_scatter(path: Path, rows: Sequence[RouteLatencyRow], query_rows: Sequence[QueryLatencyRow]) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        return []

    written: list[str] = []
    path.mkdir(parents=True, exist_ok=True)

    if rows:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        sc = ax.scatter(
            [row.model_cost for row in rows],
            [row.sql_time_ms for row in rows],
            c=[row.selectivity for row in rows],
            s=14,
            alpha=0.65,
            cmap="viridis",
        )
        ax.set_xlabel("Planner route cost")
        ax.set_ylabel("Real route SQL time (ms)")
        ax.set_title("KMeans Cost Model vs Real Route Latency")
        fig.colorbar(sc, ax=ax, label="selectivity A(t,P)/|P|")
        fig.tight_layout()
        out = path / "kmeans_cost_vs_route_latency.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        written.append(str(out))

        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        ax.scatter(
            [row.inverse_selectivity for row in rows],
            [row.sql_time_ms for row in rows],
            s=14,
            alpha=0.65,
        )
        ax.set_xlabel("Inverse selectivity |P|/A(t,P)")
        ax.set_ylabel("Real route SQL time (ms)")
        ax.set_title("Filtering Difficulty vs Real Route Latency")
        ax.set_xscale("log")
        fig.tight_layout()
        out = path / "kmeans_inverse_selectivity_vs_latency.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        written.append(str(out))

    if query_rows:
        fig, ax = plt.subplots(figsize=(7.0, 5.0))
        sc = ax.scatter(
            [row.model_total_cost for row in query_rows],
            [row.sql_total_ms for row in query_rows],
            c=[row.route_count for row in query_rows],
            s=22,
            alpha=0.75,
            cmap="plasma",
        )
        ax.set_xlabel("Planner query cost sum")
        ax.set_ylabel("Real query SQL time sum (ms)")
        ax.set_title("KMeans Cost Model vs Real Query Latency")
        fig.colorbar(sc, ax=ax, label="route count")
        fig.tight_layout()
        out = path / "kmeans_cost_vs_query_latency.png"
        fig.savefig(out, dpi=180)
        plt.close(fig)
        written.append(str(out))

    return written


def validate_cost_model_latency(
    *,
    query_dataset_path: Path,
    query_limit: int,
    query_offset: int,
    ef_search: int,
    topk_override: int | None,
    warmup: int,
    require_hnsw: bool,
    output_prefix: str,
) -> dict[str, object]:
    queries = _load_queries(query_dataset_path, limit=int(query_limit), offset=int(query_offset))
    base_ef = _base_ef(topk=int(topk_override or queries[0].get("topk", 10)), ef_min=int(ef_search))
    print(
        "[cost-validate] "
        f"queries={len(queries)}, ef_search={ef_search}, base_ef_example={base_ef}, "
        f"cost=(a={DEFAULT_COST_A:.6g}, b={DEFAULT_COST_B:.6g}, c={DEFAULT_COST_C:.6g})"
    )

    route_rows: list[RouteLatencyRow] = []
    query_rows: list[QueryLatencyRow] = []

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        _configure_search_session(cur)
        seen_tables: set[str] = set()
        for query_index, query in enumerate(queries):
            tenant_id = int(query["user_id"])
            query_vector = str(query["query_vector"])
            topk = int(topk_override or query.get("topk", 10))
            routes = [route for route in load_tenant_routes(tenant_id, refresh=False) if route.pattern_ids]
            if not routes:
                continue
            for route in routes:
                if route.table_name in seen_tables:
                    continue
                seen_tables.add(route.table_name)
                if require_hnsw and not _has_hnsw_index(cur, route.table_name):
                    raise RuntimeError(
                        f"Partition table {route.table_name} has no HNSW index. "
                        "Build kmeans indexes first or pass --require-hnsw false."
                    )

            # Warm up each route with the base search path; warmup does not enter output.
            if query_index < int(warmup):
                for route in routes:
                    _execute_candidate_search(cur, route, query_vector=query_vector, ef_search=int(ef_search))

            per_query_route_rows: list[RouteLatencyRow] = []
            for route in routes:
                route_base_ef = _base_ef(topk=int(topk), ef_min=int(ef_search))
                probe_rows = _execute_candidate_search(cur, route, query_vector=query_vector, ef_search=int(route_base_ef))
                probe_count, probe_hits, best_distance = _candidate_probe_stats(probe_rows, route)
                final_ef = _probe_adaptive_ef(route, probe_rows=probe_rows, base_ef=int(route_base_ef))
                sql_time_ms = _explain_candidate_search_ms(
                    cur,
                    route,
                    query_vector=query_vector,
                    ef_search=int(final_ef),
                )
                final_rows = _execute_candidate_search(cur, route, query_vector=query_vector, ef_search=int(final_ef))
                _final_count, final_hits, final_best_distance = _candidate_probe_stats(final_rows, route)
                model_cost, fixed_cost, filter_cost = _route_model_cost(route, ef_search=int(ef_search))
                partition_vectors = max(1, int(route.partition_vector_count or 0))
                accessible_vectors = max(0, int(route.accessible_vector_count or 0))
                selectivity = float(accessible_vectors) / float(partition_vectors) if accessible_vectors > 0 else 0.0
                inverse_selectivity = 1.0 / max(1e-12, float(selectivity))
                row = RouteLatencyRow(
                    query_index=int(query_index),
                    tenant_id=int(tenant_id),
                    partition_id=str(route.partition_id),
                    table_name=str(route.table_name),
                    route_kind=str(route.route_kind),
                    partition_vectors=int(partition_vectors),
                    accessible_vectors=int(accessible_vectors),
                    selectivity=float(selectivity),
                    inverse_selectivity=float(inverse_selectivity),
                    pattern_count=int(len(route.pattern_ids)),
                    topk=int(topk),
                    base_ef=int(route_base_ef),
                    probe_rows=int(probe_count),
                    probe_hits=int(probe_hits),
                    final_ef=int(final_ef),
                    model_cost=float(model_cost),
                    model_fixed_cost=float(fixed_cost),
                    model_filter_cost=float(filter_cost),
                    sql_time_ms=float(sql_time_ms),
                    authorized_rows=int(final_hits),
                    best_distance=final_best_distance if final_best_distance is not None else best_distance,
                    query_total_routes=int(len(routes)),
                    query_model_total_cost=0.0,
                    query_sql_total_ms=0.0,
                )
                per_query_route_rows.append(row)

            query_model_total = float(sum(row.model_cost for row in per_query_route_rows))
            query_sql_total = float(sum(row.sql_time_ms for row in per_query_route_rows))
            selectivities = [row.selectivity for row in per_query_route_rows]
            query_row = QueryLatencyRow(
                query_index=int(query_index),
                tenant_id=int(tenant_id),
                topk=int(topk),
                route_count=int(len(per_query_route_rows)),
                model_total_cost=float(query_model_total),
                sql_total_ms=float(query_sql_total),
                min_selectivity=float(min(selectivities)) if selectivities else 0.0,
                mean_selectivity=float(statistics.fmean(selectivities)) if selectivities else 0.0,
                max_inverse_selectivity=float(max(row.inverse_selectivity for row in per_query_route_rows))
                if per_query_route_rows
                else 0.0,
                total_partition_vectors=int(sum(row.partition_vectors for row in per_query_route_rows)),
                total_accessible_vectors=int(sum(row.accessible_vectors for row in per_query_route_rows)),
            )
            query_rows.append(query_row)
            for row in per_query_route_rows:
                route_rows.append(
                    RouteLatencyRow(
                        **{
                            **asdict(row),
                            "query_model_total_cost": float(query_model_total),
                            "query_sql_total_ms": float(query_sql_total),
                        }
                    )
                )
            print(
                "[cost-validate] "
                f"query {query_index + 1}/{len(queries)} user={tenant_id} "
                f"routes={len(per_query_route_rows)} model={query_model_total:.4f} sql_ms={query_sql_total:.4f}"
            )
    finally:
        cur.close()
        conn.close()

    route_model = [row.model_cost for row in route_rows]
    route_sql = [row.sql_time_ms for row in route_rows]
    query_model = [row.model_total_cost for row in query_rows]
    query_sql = [row.sql_total_ms for row in query_rows]
    inverse_selectivity = [row.inverse_selectivity for row in route_rows]
    final_ef = [row.final_ef for row in route_rows]

    summary: dict[str, object] = {
        "query_dataset_path": str(query_dataset_path),
        "query_limit": int(query_limit),
        "query_offset": int(query_offset),
        "ef_search": int(ef_search),
        "topk_override": None if topk_override is None else int(topk_override),
        "cost_a": float(DEFAULT_COST_A),
        "cost_b": float(DEFAULT_COST_B),
        "cost_c": float(DEFAULT_COST_C),
        "route_sample_count": len(route_rows),
        "query_sample_count": len(query_rows),
        "route_pearson_model_vs_sql": _safe_corr(route_model, route_sql),
        "route_spearman_model_vs_sql": _safe_spearman(route_model, route_sql),
        "query_pearson_model_vs_sql": _safe_corr(query_model, query_sql),
        "query_spearman_model_vs_sql": _safe_spearman(query_model, query_sql),
        "route_fit_model_to_sql": _linear_fit_summary(route_model, route_sql),
        "query_fit_model_to_sql": _linear_fit_summary(query_model, query_sql),
        "route_pearson_inverse_selectivity_vs_sql": _safe_corr(inverse_selectivity, route_sql),
        "route_spearman_inverse_selectivity_vs_sql": _safe_spearman(inverse_selectivity, route_sql),
        "route_pearson_final_ef_vs_sql": _safe_corr(final_ef, route_sql),
        "route_spearman_final_ef_vs_sql": _safe_spearman(final_ef, route_sql),
        "route_sql_ms_p50": _quantile(route_sql, 50),
        "route_sql_ms_p90": _quantile(route_sql, 90),
        "route_model_cost_p50": _quantile(route_model, 50),
        "route_model_cost_p90": _quantile(route_model, 90),
        "query_sql_ms_p50": _quantile(query_sql, 50),
        "query_sql_ms_p90": _quantile(query_sql, 90),
        "query_model_cost_p50": _quantile(query_model, 50),
        "query_model_cost_p90": _quantile(query_model, 90),
        "selectivity_buckets": _bucket_summary(route_rows),
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    route_csv = RESULT_DIR / f"{output_prefix}_route_samples.csv"
    query_csv = RESULT_DIR / f"{output_prefix}_query_samples.csv"
    summary_json = RESULT_DIR / f"{output_prefix}_summary.json"
    report_md = RESULT_DIR / f"{output_prefix}_report.md"
    _write_csv(route_csv, route_rows, RouteLatencyRow.__dataclass_fields__.keys())
    _write_csv(query_csv, query_rows, QueryLatencyRow.__dataclass_fields__.keys())
    with summary_json.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2, sort_keys=True)

    plot_paths = _plot_scatter(RESULT_DIR, route_rows, query_rows)
    report = [
        "# KMeans Cost Model Latency Validation",
        "",
        f"- route samples: {len(route_rows)}",
        f"- query samples: {len(query_rows)}",
        f"- ef_search: {int(ef_search)}",
        f"- cost model: `a log2(1+N) + c + b * ef * |P|/A(t,P)`",
        f"- route Pearson: {summary['route_pearson_model_vs_sql']}",
        f"- route Spearman: {summary['route_spearman_model_vs_sql']}",
        f"- query Pearson: {summary['query_pearson_model_vs_sql']}",
        f"- query Spearman: {summary['query_spearman_model_vs_sql']}",
        f"- route linear fit R2: {summary['route_fit_model_to_sql']['r2']}",
        f"- query linear fit R2: {summary['query_fit_model_to_sql']['r2']}",
        "",
        "## Outputs",
        "",
        f"- {route_csv}",
        f"- {query_csv}",
        f"- {summary_json}",
    ]
    report.extend(f"- {path}" for path in plot_paths)
    report_md.write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"[cost-validate] wrote {summary_json}")
    print(f"[cost-validate] wrote {report_md}")
    return summary


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate kmeans planner cost model against real route/query latency.")
    parser.add_argument("--query-dataset-path", default=DEFAULT_QUERY_DATASET_PATH)
    parser.add_argument("--query-limit", type=int, default=40)
    parser.add_argument("--query-offset", type=int, default=0)
    parser.add_argument("--ef-search", type=int, default=40)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--require-hnsw", type=_str_to_bool, default=True)
    parser.add_argument("--output-prefix", default="kmeans_cost_latency_validation")
    args = parser.parse_args()
    validate_cost_model_latency(
        query_dataset_path=Path(args.query_dataset_path),
        query_limit=int(args.query_limit),
        query_offset=int(args.query_offset),
        ef_search=int(args.ef_search),
        topk_override=args.topk,
        warmup=int(args.warmup),
        require_hnsw=bool(args.require_hnsw),
        output_prefix=str(args.output_prefix),
    )


if __name__ == "__main__":
    main()
