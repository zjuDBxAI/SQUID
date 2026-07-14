from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from psycopg2 import sql
from scipy.optimize import least_squares


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent
OUTPUT_DIR = PROJECT_ROOT / "basic_benchmark" / "result" / "recallmodel"

sys.path.insert(0, str(PROJECT_ROOT))

from basic_benchmark.direct_pg_qps import (  # noqa: E402
    PROJECT_ROOT as DIRECT_PROJECT_ROOT,
    Route,
    _normalize_method_name,
    _tables_with_index_am,
    load_ours_routes,
    load_queries,
    resolve_versioned_plan,
)
from services.config import get_db_connection  # noqa: E402


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "Nimbus Sans", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


DEFAULT_EF_VALUES = (5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 65, 80, 120, 160, 240, 320, 500)
DEFAULT_K_VALUES = (1, 10, 100)
DEFAULT_TARGETS = (0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98)
COLORS = {
    "easy": "#4C78A8",
    "medium": "#F58518",
    "hard": "#E45756",
    1: "#4C78A8",
    10: "#F58518",
    100: "#7F3C8D",
}
MARKERS = {1: "o", 10: "s", 100: "D"}


@dataclass(frozen=True)
class QueryVector:
    query_index: int
    vector: str


@dataclass(frozen=True)
class RouteSample:
    route_key: str
    user_id: int
    route: Route
    route_class: str

    @property
    def n(self) -> int:
        return max(1, int(self.route.partition_vectors))

    @property
    def accessible(self) -> int:
        if self.route.pure:
            return self.n
        return max(0, int(self.route.accessible_vectors))

    @property
    def rho(self) -> float:
        if self.route.pure:
            return 1.0
        if self.n <= 0 or self.accessible <= 0:
            return 0.0
        return min(1.0, max(1e-12, float(self.accessible) / float(self.n)))


@dataclass(frozen=True)
class Observation:
    split: str
    route_key: str
    route_class: str
    user_id: int
    table_name: str
    partition_id: str
    query_index: int
    n: int
    accessible: int
    rho: float
    k: int
    ef_search: int
    exact_count: int
    approx_count: int
    measured_recall: float


def parse_int_list(values: list[str] | None, default: tuple[int, ...]) -> list[int]:
    if not values:
        return list(default)
    parsed: list[int] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            parsed.append(int(part))
    return sorted(set(parsed))


def parse_float_list(values: list[str] | None, default: tuple[float, ...]) -> list[float]:
    if not values:
        return list(default)
    parsed: list[float] = []
    for value in values:
        for part in str(value).replace(",", " ").split():
            parsed.append(float(part))
    return sorted(set(parsed))


def result_key(row: tuple) -> tuple[int, int]:
    return int(row[0]), int(row[1])


def configure_common(cur) -> None:
    cur.execute("SET jit = off")
    cur.execute("SET max_parallel_workers_per_gather = 0")
    cur.execute("SET enable_bitmapscan = on")
    try:
        cur.execute("SET hnsw.iterative_scan = off")
    except Exception:
        cur.connection.rollback()


def set_exact_mode(cur) -> None:
    cur.execute("SET enable_seqscan = on")
    cur.execute("SET enable_indexscan = off")
    cur.execute("SET enable_bitmapscan = off")


def set_hnsw_mode(cur, ef_search: int) -> None:
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET enable_indexscan = on")
    cur.execute("SET enable_bitmapscan = on")
    cur.execute(f"SET hnsw.ef_search = {max(1, int(ef_search))}")


def execute_route(cur, route: Route, query_vector: str, limit: int) -> list[tuple]:
    if route.pure:
        statement = sql.SQL(
            """
            SELECT block_id, document_id, vector <-> %s::vector AS distance
            FROM {}
            ORDER BY vector <-> %s::vector
            LIMIT %s
            """
        ).format(sql.Identifier(route.table_name))
        params = [query_vector, query_vector, int(limit)]
    else:
        statement = sql.SQL(
            """
            SELECT block_id, document_id, vector <-> %s::vector AS distance
            FROM {}
            WHERE pattern_id = ANY(%s::bigint[])
            ORDER BY vector <-> %s::vector
            LIMIT %s
            """
        ).format(sql.Identifier(route.table_name))
        params = [query_vector, list(route.pattern_ids), query_vector, int(limit)]
    cur.execute(statement, params)
    return list(cur.fetchall())


def classify_route(route: Route) -> str:
    n = max(1, int(route.partition_vectors))
    accessible = n if route.pure else max(0, int(route.accessible_vectors))
    rho = min(1.0, max(0.0, float(accessible) / float(n))) if n > 0 else 0.0
    difficulty = float(n) / max(rho, 1e-9)
    if rho >= 0.995 and n <= 5000:
        return "easy"
    if difficulty <= 45_000:
        return "easy"
    if difficulty <= 90_000:
        return "medium"
    return "hard"


def pick_evenly(values: list[RouteSample], count: int) -> list[RouteSample]:
    if count <= 0 or not values:
        return []
    if len(values) <= count:
        return list(values)
    indexes = np.linspace(0, len(values) - 1, num=count)
    picked: list[RouteSample] = []
    seen: set[str] = set()
    for raw_index in indexes:
        item = values[int(round(float(raw_index)))]
        if item.route_key not in seen:
            picked.append(item)
            seen.add(item.route_key)
    for item in values:
        if len(picked) >= count:
            break
        if item.route_key not in seen:
            picked.append(item)
            seen.add(item.route_key)
    return picked


def sample_routes(memory_ratio: float, route_count: int, max_k: int) -> tuple[object, list[RouteSample]]:
    selection = resolve_versioned_plan("ours", memory_ratio)
    routes_by_user = load_ours_routes(selection)
    route_tables = {route.table_name for routes in routes_by_user.values() for route in routes}
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            indexed_tables = _tables_with_index_am(cur, route_tables, "hnsw")
    finally:
        conn.close()

    candidates: list[RouteSample] = []
    for user_id, routes in routes_by_user.items():
        for route in routes:
            if route.table_name not in indexed_tables:
                continue
            n = max(1, int(route.partition_vectors))
            accessible = n if route.pure else max(0, int(route.accessible_vectors))
            if n < max_k or accessible < max_k:
                continue
            rho = min(1.0, max(0.0, float(accessible) / float(n)))
            if rho <= 0.0:
                continue
            route_key = f"u{int(user_id)}:{route.table_name}:{route.partition_id}:{','.join(map(str, route.pattern_ids[:8]))}"
            candidates.append(RouteSample(route_key, int(user_id), route, classify_route(route)))

    if not candidates:
        raise RuntimeError(f"No HNSW-indexed SQUID routes found for memory_ratio={memory_ratio}")

    by_class: dict[str, list[RouteSample]] = {"easy": [], "medium": [], "hard": []}
    for sample in candidates:
        by_class.setdefault(sample.route_class, []).append(sample)
    for route_class, values in by_class.items():
        values.sort(key=lambda item: (float(item.n) / max(item.rho, 1e-9), item.n, item.route_key))

    base = max(1, route_count // 3)
    selected: list[RouteSample] = []
    selected.extend(pick_evenly(by_class.get("easy", []), base))
    selected.extend(pick_evenly(by_class.get("medium", []), base))
    selected.extend(pick_evenly(by_class.get("hard", []), route_count - len(selected)))

    if len(selected) < route_count:
        selected_keys = {item.route_key for item in selected}
        leftovers = sorted(
            (item for item in candidates if item.route_key not in selected_keys),
            key=lambda item: (item.route_class, float(item.n) / max(item.rho, 1e-9), item.route_key),
        )
        selected.extend(leftovers[: route_count - len(selected)])

    selected.sort(key=lambda item: (item.route_class, float(item.n) / max(item.rho, 1e-9), item.route_key))
    return selection, selected[:route_count]


def split_routes(samples: list[RouteSample]) -> dict[str, str]:
    split: dict[str, str] = {}
    for index, sample in enumerate(samples):
        split[sample.route_key] = "test" if index % 3 == 2 else "train"
    if all(value == "train" for value in split.values()) and samples:
        split[samples[-1].route_key] = "test"
    return split


def collect_observations(
    *,
    memory_ratio: float,
    route_count: int,
    query_vectors: int,
    ef_values: list[int],
    k_values: list[int],
    query_file: Path,
    ground_truth_file: Path,
) -> tuple[object, list[Observation]]:
    max_k = max(k_values)
    selection, samples = sample_routes(memory_ratio, route_count, max_k)
    split_by_route = split_routes(samples)
    queries = load_queries(query_file, ground_truth_file, max(1, query_vectors))
    query_samples = [QueryVector(index, query.vector) for index, query in enumerate(queries[:query_vectors])]

    observations: list[Observation] = []
    conn = get_db_connection()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            configure_common(cur)
            total = len(samples) * len(query_samples)
            completed = 0
            for sample in samples:
                for query in query_samples:
                    completed += 1
                    print(
                        f"[{completed}/{total}] route={sample.route_class} "
                        f"N={sample.n} rho={sample.rho:.4f} table={sample.route.table_name}",
                        flush=True,
                    )
                    set_exact_mode(cur)
                    exact_rows = execute_route(cur, sample.route, query.vector, max_k)
                    exact_keys = [result_key(row) for row in exact_rows]
                    for ef_search in ef_values:
                        set_hnsw_mode(cur, ef_search)
                        approx_by_k: dict[int, list[tuple[int, int]]] = {}
                        for k in k_values:
                            approx_rows = execute_route(cur, sample.route, query.vector, int(k))
                            approx_by_k[int(k)] = [result_key(row) for row in approx_rows]
                        for k in k_values:
                            exact_topk = exact_keys[: int(k)]
                            if len(exact_topk) < int(k):
                                continue
                            approx_keys = approx_by_k[int(k)]
                            recall = len(set(exact_topk) & set(approx_keys)) / float(len(exact_topk))
                            observations.append(
                                Observation(
                                    split=split_by_route[sample.route_key],
                                    route_key=sample.route_key,
                                    route_class=sample.route_class,
                                    user_id=sample.user_id,
                                    table_name=sample.route.table_name,
                                    partition_id=str(sample.route.partition_id),
                                    query_index=query.query_index,
                                    n=sample.n,
                                    accessible=sample.accessible,
                                    rho=sample.rho,
                                    k=int(k),
                                    ef_search=int(ef_search),
                                    exact_count=len(exact_topk),
                                    approx_count=len(approx_keys),
                                    measured_recall=float(recall),
                                )
                            )
    finally:
        conn.close()
    return selection, observations


def observation_to_row(item: Observation) -> dict[str, object]:
    return {
        "split": item.split,
        "route_key": item.route_key,
        "route_class": item.route_class,
        "user_id": item.user_id,
        "table_name": item.table_name,
        "partition_id": item.partition_id,
        "query_index": item.query_index,
        "partition_vectors": item.n,
        "accessible_vectors": item.accessible,
        "selectivity": item.rho,
        "k": item.k,
        "ef_search": item.ef_search,
        "exact_count": item.exact_count,
        "approx_count": item.approx_count,
        "measured_recall": item.measured_recall,
    }


def write_observations(path: Path, observations: list[Observation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(observation_to_row(observations[0]).keys()) if observations else [
        "split",
        "route_key",
        "route_class",
        "user_id",
        "table_name",
        "partition_id",
        "query_index",
        "partition_vectors",
        "accessible_vectors",
        "selectivity",
        "k",
        "ef_search",
        "exact_count",
        "approx_count",
        "measured_recall",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in observations:
            writer.writerow(observation_to_row(item))


def load_observations(path: Path) -> list[Observation]:
    observations: list[Observation] = []
    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            observations.append(
                Observation(
                    split=str(row["split"]),
                    route_key=str(row["route_key"]),
                    route_class=str(row["route_class"]),
                    user_id=int(row["user_id"]),
                    table_name=str(row["table_name"]),
                    partition_id=str(row["partition_id"]),
                    query_index=int(row["query_index"]),
                    n=int(float(row["partition_vectors"])),
                    accessible=int(float(row["accessible_vectors"])),
                    rho=float(row["selectivity"]),
                    k=int(float(row["k"])),
                    ef_search=int(float(row["ef_search"])),
                    exact_count=int(float(row["exact_count"])),
                    approx_count=int(float(row["approx_count"])),
                    measured_recall=float(row["measured_recall"]),
                )
            )
    return observations


def recall_model(params: np.ndarray, n_values: np.ndarray, k_values: np.ndarray, ef_values: np.ndarray, rho_values: np.ndarray) -> np.ndarray:
    h = max(float(params[0]), 1e-9)
    lambda0 = max(float(params[1]), 1e-12)
    lambda1 = max(float(params[2]), 0.0)
    lam = np.maximum(lambda0 + lambda1 * np.log1p(np.maximum(n_values, 1.0)), 1e-12)
    x = np.maximum(ef_values * rho_values / np.maximum(k_values, 1.0), 1e-12)
    z = np.clip(h * (np.log(lam) - np.log(x)), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))


def fit_model(observations: list[Observation]) -> dict[str, object]:
    train = [row for row in observations if row.split == "train"]
    if len(train) < 12:
        train = list(observations)
    n_values = np.array([row.n for row in train], dtype=float)
    k_values = np.array([row.k for row in train], dtype=float)
    ef_values = np.array([row.ef_search for row in train], dtype=float)
    rho_values = np.array([row.rho for row in train], dtype=float)
    y = np.array([row.measured_recall for row in train], dtype=float)

    def residual(params: np.ndarray) -> np.ndarray:
        pred = recall_model(params, n_values, k_values, ef_values, rho_values)
        weights = 1.0 + 0.7 * y
        return (pred - y) * weights

    result = least_squares(
        residual,
        x0=np.array([1.25, 0.15, 0.035], dtype=float),
        bounds=(np.array([0.25, 1e-6, 0.0]), np.array([8.0, 50.0, 10.0])),
        max_nfev=20_000,
    )
    params = result.x

    def metrics(rows: list[Observation]) -> dict[str, float]:
        if not rows:
            return {"rmse": float("nan"), "mae": float("nan"), "count": 0}
        pred = recall_model(
            params,
            np.array([row.n for row in rows], dtype=float),
            np.array([row.k for row in rows], dtype=float),
            np.array([row.ef_search for row in rows], dtype=float),
            np.array([row.rho for row in rows], dtype=float),
        )
        actual = np.array([row.measured_recall for row in rows], dtype=float)
        errors = pred - actual
        return {
            "rmse": float(np.sqrt(np.mean(errors * errors))),
            "mae": float(np.mean(np.abs(errors))),
            "count": int(len(rows)),
        }

    return {
        "h": float(params[0]),
        "lambda0": float(params[1]),
        "lambda1": float(params[2]),
        "train": metrics([row for row in observations if row.split == "train"]),
        "test": metrics([row for row in observations if row.split == "test"]),
        "all": metrics(observations),
        "optimizer_cost": float(result.cost),
        "optimizer_success": bool(result.success),
        "optimizer_message": str(result.message),
    }


def predict_required_ef(params: dict[str, object], n: int, k: int, rho: float, target: float, max_ef: int) -> tuple[int, bool]:
    h = float(params["h"])
    lam = float(params["lambda0"]) + float(params["lambda1"]) * math.log1p(max(1, int(n)))
    lam = max(lam, 1e-12)
    rho = max(float(rho), 1e-12)
    target = min(0.999999, max(1e-9, float(target)))
    required = (float(k) * lam / rho) * ((target / (1.0 - target)) ** (1.0 / h))
    ef_search = int(math.ceil(required))
    clipped = ef_search > int(max_ef)
    return max(1, min(int(max_ef), ef_search)), clipped


def rows_by_route_query(observations: list[Observation]) -> dict[tuple[str, int], Observation]:
    selected: dict[tuple[str, int], Observation] = {}
    for row in observations:
        key = (row.route_key, row.query_index)
        selected.setdefault(key, row)
    return selected


def route_from_observation(row: Observation, route_lookup: dict[str, RouteSample]) -> RouteSample:
    sample = route_lookup.get(row.route_key)
    if sample is None:
        raise KeyError(row.route_key)
    return sample


def rebuild_route_lookup(memory_ratio: float, route_count: int, max_k: int) -> dict[str, RouteSample]:
    _, samples = sample_routes(memory_ratio, route_count, max_k)
    return {sample.route_key: sample for sample in samples}


def evaluate_targets(
    *,
    params: dict[str, object],
    observations: list[Observation],
    memory_ratio: float,
    route_count: int,
    max_k: int,
    targets: list[float],
    max_ef: int,
    query_file: Path,
    ground_truth_file: Path,
) -> list[dict[str, object]]:
    query_count = max(row.query_index for row in observations) + 1
    queries = load_queries(query_file, ground_truth_file, query_count)
    query_vectors = {index: QueryVector(index, query.vector) for index, query in enumerate(queries[:query_count])}
    route_lookup = rebuild_route_lookup(memory_ratio, route_count, max_k)
    test_groups = [
        row
        for row in rows_by_route_query([item for item in observations if item.split == "test"]).values()
        if row.route_key in route_lookup
    ]
    if not test_groups:
        test_groups = [
            row
            for row in rows_by_route_query(observations).values()
            if row.route_key in route_lookup
        ][:6]

    rows: list[dict[str, object]] = []
    conn = get_db_connection()
    conn.autocommit = True
    try:
        with conn.cursor() as cur:
            configure_common(cur)
            for group in test_groups:
                sample = route_from_observation(group, route_lookup)
                query = query_vectors[group.query_index]
                set_exact_mode(cur)
                exact_rows = execute_route(cur, sample.route, query.vector, max_k)
                exact_keys = [result_key(row) for row in exact_rows]
                for k in sorted({row.k for row in observations}):
                    exact_topk = exact_keys[: int(k)]
                    if len(exact_topk) < int(k):
                        continue
                    for target in targets:
                        predicted_ef, clipped = predict_required_ef(params, sample.n, int(k), sample.rho, target, max_ef)
                        set_hnsw_mode(cur, predicted_ef)
                        approx_rows = execute_route(cur, sample.route, query.vector, int(k))
                        approx_keys = [result_key(row) for row in approx_rows]
                        recall = len(set(exact_topk) & set(approx_keys)) / float(len(exact_topk))
                        rows.append(
                            {
                                "route_key": sample.route_key,
                                "route_class": sample.route_class,
                                "query_index": query.query_index,
                                "partition_vectors": sample.n,
                                "accessible_vectors": sample.accessible,
                                "selectivity": sample.rho,
                                "k": int(k),
                                "target_recall": float(target),
                                "predicted_ef_search": int(predicted_ef),
                                "ef_clipped": bool(clipped),
                                "achieved_recall": float(recall),
                                "approx_count": len(approx_keys),
                            }
                        )
    finally:
        conn.close()
    return rows


def write_dict_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if rows:
        fieldnames = list(rows[0].keys())
    else:
        fieldnames = []
    with path.open("w", newline="") as handle:
        if not fieldnames:
            return
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_by_key(rows: list[Observation], key_fields: tuple[str, ...]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Observation]] = {}
    for row in rows:
        key = tuple(getattr(row, field) for field in key_fields)
        grouped.setdefault(key, []).append(row)
    result: list[dict[str, object]] = []
    for key, values in grouped.items():
        first = values[0]
        item = {field: value for field, value in zip(key_fields, key)}
        item.update(
            {
                "n": first.n,
                "rho": first.rho,
                "route_class": first.route_class,
                "mean_recall": statistics.mean(row.measured_recall for row in values),
                "count": len(values),
            }
        )
        result.append(item)
    return result


def choose_plot_routes(observations: list[Observation], k: int = 10) -> list[str]:
    candidates: dict[str, Observation] = {}
    for row in observations:
        if row.k == k:
            candidates.setdefault(row.route_key, row)
    by_class: dict[str, list[Observation]] = {"easy": [], "medium": [], "hard": []}
    for row in candidates.values():
        by_class.setdefault(row.route_class, []).append(row)
    chosen: list[str] = []
    for route_class in ("easy", "medium", "hard"):
        values = sorted(by_class.get(route_class, []), key=lambda row: (float(row.n) / max(row.rho, 1e-9), row.route_key))
        if values:
            chosen.append(values[len(values) // 2].route_key)
    if len(chosen) < 3:
        fallback = sorted(candidates.values(), key=lambda row: (float(row.n) / max(row.rho, 1e-9), row.route_key))
        for row in pick_evenly([RouteSample(row.route_key, row.user_id, Route(row.table_name), row.route_class) for row in fallback], 3):
            if row.route_key not in chosen:
                chosen.append(row.route_key)
    return chosen[:3]


def plot_figure(
    *,
    observations: list[Observation],
    fit_params: dict[str, object],
    target_rows: list[dict[str, object]],
    output: Path,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.15), constrained_layout=True)
    ax = axes[0]

    plot_k = 10 if any(row.k == 10 for row in observations) else sorted({row.k for row in observations})[0]
    route_keys = choose_plot_routes(observations, plot_k)
    averaged = mean_by_key([row for row in observations if row.k == plot_k and row.route_key in route_keys], ("route_key", "ef_search"))
    ef_min = min(row.ef_search for row in observations)
    ef_max = max(row.ef_search for row in observations)
    ef_grid = np.geomspace(max(1, ef_min), max(ef_min + 1, ef_max), 160)

    route_meta = {row.route_key: row for row in observations if row.route_key in route_keys and row.k == plot_k}
    class_rank = {"easy": 0, "medium": 1, "hard": 2}
    for route_key in sorted(route_keys, key=lambda key: class_rank.get(route_meta[key].route_class, 9)):
        meta = route_meta[route_key]
        route_class = meta.route_class
        color = COLORS.get(route_class, "#4C78A8")
        points = sorted((row for row in averaged if row["route_key"] == route_key), key=lambda row: int(row["ef_search"]))
        ax.scatter(
            [int(row["ef_search"]) for row in points],
            [float(row["mean_recall"]) for row in points],
            s=30,
            color=color,
            alpha=0.9,
            edgecolor="white",
            linewidth=0.45,
        )
        pred = recall_model(
            np.array([fit_params["h"], fit_params["lambda0"], fit_params["lambda1"]], dtype=float),
            np.full_like(ef_grid, float(meta.n)),
            np.full_like(ef_grid, float(plot_k)),
            ef_grid,
            np.full_like(ef_grid, float(meta.rho)),
        )
        label = f"{route_class}: N={meta.n/1000:.1f}K, rho={meta.rho:.2f}"
        ax.plot(ef_grid, pred, color=color, linewidth=2.0, label=label)

    ax.set_xscale("log")
    ax.set_xlabel(r"Search effort $ef_s$", fontsize=12.5)
    ax.set_ylabel(f"Recall@{plot_k}", fontsize=12.5)
    ax.set_ylim(-0.03, 1.03)
    ax.grid(True, axis="both", color="#dddddd", linewidth=0.65, alpha=0.65)
    ax.tick_params(axis="both", labelsize=10.8, direction="in")
    ax.legend(frameon=False, fontsize=8.2, loc="lower right", handlelength=1.6)

    ax = axes[1]
    ax.plot([0.88, 1.0], [0.88, 1.0], color="#666666", linestyle="--", linewidth=1.1, label="ideal")
    available_ks = sorted({int(row["k"]) for row in target_rows})
    plot_ks = [10] if 10 in available_ks else ([k for k in available_ks if k != 1] or available_ks)
    for k_index, k in enumerate(plot_ks):
        subset = [row for row in target_rows if int(row["k"]) == k]
        grouped: dict[float, list[float]] = {}
        for row in subset:
            grouped.setdefault(float(row["target_recall"]), []).append(float(row["achieved_recall"]))
        xs = sorted(grouped)
        ys = [statistics.mean(grouped[x]) for x in xs]
        ax.plot(
            xs,
            ys,
            color=COLORS.get(k, "#4C78A8"),
            marker=MARKERS.get(k, "o"),
            linewidth=1.8,
            markersize=5.8,
            label=f"k={k}",
        )

    ax.set_xlabel("Target recall", fontsize=12.5)
    ax.set_ylabel("Achieved Recall@k", fontsize=12.5)
    ax.set_xlim(0.885, 0.99)
    ax.set_ylim(0.885, 1.012)
    ax.grid(True, axis="both", color="#dddddd", linewidth=0.65, alpha=0.65)
    ax.tick_params(axis="both", labelsize=10.8, direction="in")
    ax.legend(frameon=False, fontsize=8.5, loc="lower right", handlelength=1.6)

    for ax_item in axes:
        for spine in ax_item.spines.values():
            spine.set_linewidth(1.0)
            spine.set_color("#555555")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def summarize_observations(observations: list[Observation], fit_params: dict[str, object], selection: object) -> dict[str, object]:
    route_keys = sorted({row.route_key for row in observations})
    return {
        "dataset_tag": "erbac+sift",
        "method": "SQUID",
        "memory_ratio": getattr(selection, "memory_ratio", None),
        "registry_id": getattr(selection, "registry_id", None),
        "plan_id": getattr(selection, "plan_id", None),
        "table_prefix": getattr(selection, "table_prefix", None),
        "route_count": len(route_keys),
        "observation_count": len(observations),
        "k_values": sorted({row.k for row in observations}),
        "ef_values": sorted({row.ef_search for row in observations}),
        "partition_vectors_min": min(row.n for row in observations),
        "partition_vectors_max": max(row.n for row in observations),
        "selectivity_min": min(row.rho for row in observations),
        "selectivity_max": max(row.rho for row in observations),
        "fit": fit_params,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit and plot the SQUID route-level recall model.")
    parser.add_argument("--memory-ratio", type=float, default=1.5)
    parser.add_argument("--route-count", type=int, default=18)
    parser.add_argument("--query-vectors", type=int, default=3)
    parser.add_argument("--ef-values", nargs="*", default=None)
    parser.add_argument("--k-values", nargs="*", default=None)
    parser.add_argument("--targets", nargs="*", default=None)
    parser.add_argument("--max-predicted-ef", type=int, default=1000)
    parser.add_argument("--query-file", type=Path, default=DIRECT_PROJECT_ROOT / "basic_benchmark" / "query_dataset.json")
    parser.add_argument("--ground-truth-file", type=Path, default=DIRECT_PROJECT_ROOT / "basic_benchmark" / "ground_truth_cache.json")
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--force", action="store_true", help="Re-collect route-level observations even if CSV exists.")
    args = parser.parse_args()

    if _normalize_method_name("ours") != "ours":
        raise RuntimeError("Unexpected method normalization failure")

    ef_values = parse_int_list(args.ef_values, DEFAULT_EF_VALUES)
    k_values = parse_int_list(args.k_values, DEFAULT_K_VALUES)
    targets = parse_float_list(args.targets, DEFAULT_TARGETS)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    observations_path = output_dir / "recall_model_observations.csv"
    target_path = output_dir / "recall_model_target_eval.csv"
    params_path = output_dir / "recall_model_fit.json"
    figure_path = output_dir / "recall_model_analysis.png"

    if observations_path.exists() and not args.force:
        observations = load_observations(observations_path)
        selection = resolve_versioned_plan("ours", args.memory_ratio)
        print(f"Loaded {len(observations)} cached observations from {observations_path}")
    else:
        selection, observations = collect_observations(
            memory_ratio=float(args.memory_ratio),
            route_count=max(3, int(args.route_count)),
            query_vectors=max(1, int(args.query_vectors)),
            ef_values=ef_values,
            k_values=k_values,
            query_file=args.query_file.resolve(),
            ground_truth_file=args.ground_truth_file.resolve(),
        )
        write_observations(observations_path, observations)
        print(f"Wrote {len(observations)} observations to {observations_path}")

    fit_params = fit_model(observations)
    target_rows = evaluate_targets(
        params=fit_params,
        observations=observations,
        memory_ratio=float(args.memory_ratio),
        route_count=max(3, int(args.route_count)),
        max_k=max(k_values),
        targets=targets,
        max_ef=max(ef_values + [int(args.max_predicted_ef)]),
        query_file=args.query_file.resolve(),
        ground_truth_file=args.ground_truth_file.resolve(),
    )
    write_dict_rows(target_path, target_rows)
    summary = summarize_observations(observations, fit_params, selection)
    summary["target_eval"] = {
        "count": len(target_rows),
        "target_recall_values": targets,
        "mean_abs_target_error": (
            statistics.mean(abs(float(row["achieved_recall"]) - float(row["target_recall"])) for row in target_rows)
            if target_rows
            else None
        ),
    }
    params_path.write_text(json.dumps(summary, indent=2, sort_keys=True))
    plot_figure(observations=observations, fit_params=fit_params, target_rows=target_rows, output=figure_path)

    print(json.dumps(summary["fit"], indent=2, sort_keys=True))
    print(f"Saved figure to {figure_path}")
    print(f"Saved figure to {figure_path.with_suffix('.pdf')}")
    print(f"Saved fit summary to {params_path}")
    print(f"Saved target evaluation to {target_path}")


if __name__ == "__main__":
    main()
