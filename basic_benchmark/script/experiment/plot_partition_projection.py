#!/usr/bin/env python3
"""Project sampled rows from real partition tables into a shared 2D space."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import psycopg2
from matplotlib.patches import Polygon
from psycopg2 import sql
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = PROJECT_ROOT / "basic_benchmark/result/case_study/treebase_1p5_partition_projection.pdf"

METHOD_LABELS = {
    "ours": "SQUID",
    "honeybee": "HONEYBEE",
    "veda": "VEDA",
    "effveda": "EFFVEDA",
}
METHOD_ORDER = ("ours", "honeybee", "veda", "effveda")


@dataclass(frozen=True)
class PartitionTable:
    registry_id: int
    method: str
    memory_ratio: float
    table_prefix: str
    relation_name: str
    partition_index: int


@dataclass(frozen=True)
class SampleRow:
    method: str
    registry_id: int
    partition_name: str
    partition_index: int
    block_id: int
    document_id: int
    vector: np.ndarray


@dataclass(frozen=True)
class RouteSummary:
    method: str
    user_id: int
    route_tables: tuple[str, ...]
    route_count: int
    route_vector_count: int
    accessible_vector_count: int
    selectivity: float


@dataclass(frozen=True)
class ProjectionResult:
    row_coords: np.ndarray
    extra_coords: np.ndarray


def load_config() -> dict[str, object]:
    with (PROJECT_ROOT / "config.json").open("r", encoding="utf-8") as handle:
        return json.load(handle)


def connect(args: argparse.Namespace):
    config = load_config()
    return psycopg2.connect(
        dbname=args.dbname or os.environ.get("DBNAME") or config["dbname"],
        user=args.db_user or os.environ.get("DB_USER") or config["user"],
        password=args.db_password or os.environ.get("DB_PASSWORD") or config["password"],
        host=args.db_host or os.environ.get("DB_HOST") or config["host"],
        port=args.db_port or os.environ.get("DB_PORT") or config["port"],
    )


def parse_vector(value: object) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value.astype(np.float32, copy=False)
    if isinstance(value, memoryview):
        return np.frombuffer(value, dtype=np.float32)
    if isinstance(value, (bytes, bytearray)):
        text = value.decode("utf-8")
    else:
        text = str(value)
    return np.fromstring(text.strip().strip("[]"), sep=",", dtype=np.float32)


def resolve_partition_tables(args: argparse.Namespace) -> list[PartitionTable]:
    conn = connect(args)
    try:
        with conn.cursor() as cur:
            if args.registries:
                cur.execute(
                    """
                    SELECT r.registry_id,
                           r.method,
                           r.memory_ratio::FLOAT8,
                           r.table_prefix,
                           br.relation_name
                    FROM benchmark_plan_registry r
                    JOIN benchmark_plan_relations br USING (registry_id)
                    WHERE r.registry_id = ANY(%s)
                      AND br.relation_kind = 'partition'
                    ORDER BY r.registry_id, br.relation_name
                    """,
                    [list(args.registries)],
                )
            else:
                cur.execute(
                    """
                    SELECT r.registry_id,
                           r.method,
                           r.memory_ratio::FLOAT8,
                           r.table_prefix,
                           br.relation_name
                    FROM benchmark_plan_registry r
                    JOIN benchmark_plan_relations br USING (registry_id)
                    WHERE r.method = ANY(%s)
                      AND ABS(r.memory_ratio::FLOAT8 - %s) < 1e-9
                      AND r.state = 'ready'
                      AND br.relation_kind = 'partition'
                    ORDER BY r.method, br.relation_name
                    """,
                    [list(args.methods), float(args.memory_ratio)],
                )
            rows = cur.fetchall()
    finally:
        conn.close()

    tables: list[PartitionTable] = []
    counters: Counter[tuple[int, str]] = Counter()
    for registry_id, method, memory_ratio, table_prefix, relation_name in rows:
        key = (int(registry_id), str(method))
        partition_index = counters[key]
        counters[key] += 1
        tables.append(
            PartitionTable(
                registry_id=int(registry_id),
                method=str(method),
                memory_ratio=float(memory_ratio),
                table_prefix=str(table_prefix),
                relation_name=str(relation_name),
                partition_index=int(partition_index),
            )
        )
    return tables


def sample_partition_rows(args: argparse.Namespace, tables: Iterable[PartitionTable]) -> list[SampleRow]:
    conn = connect(args)
    rows: list[SampleRow] = []
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT setseed(%s)", [float(args.seed % 1000000) / 1000000.0])
            for table in tables:
                cur.execute(
                    sql.SQL(
                        """
                        SELECT block_id, document_id, vector::TEXT
                        FROM {}
                        ORDER BY random()
                        LIMIT %s
                        """
                    ).format(sql.Identifier(table.relation_name)),
                    [int(args.samples_per_partition)],
                )
                for block_id, document_id, vector in cur.fetchall():
                    parsed = parse_vector(vector)
                    if parsed.size == 0:
                        continue
                    rows.append(
                        SampleRow(
                            method=table.method,
                            registry_id=table.registry_id,
                            partition_name=table.relation_name,
                            partition_index=table.partition_index,
                            block_id=int(block_id),
                            document_id=int(document_id),
                            vector=parsed,
                        )
                    )
    finally:
        conn.close()
    return rows


def table_row_counts(args: argparse.Namespace, table_names: Iterable[str]) -> dict[str, int]:
    names = sorted({name for name in table_names if name})
    if not names:
        return {}
    conn = connect(args)
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.relname, COALESCE(s.n_live_tup, 0)::BIGINT
                FROM pg_class c
                JOIN pg_namespace n ON n.oid = c.relnamespace
                LEFT JOIN pg_stat_user_tables s ON s.relid = c.oid
                WHERE n.nspname = 'public'
                  AND c.relname = ANY(%s)
                """,
                [names],
            )
            return {str(name): int(count or 0) for name, count in cur.fetchall()}
    finally:
        conn.close()


def visible_vector_counts(args: argparse.Namespace) -> dict[int, int]:
    conn = connect(args)
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('public.user_accessible_documents') IS NOT NULL")
            has_access_table = bool(cur.fetchone()[0])
            if has_access_table:
                cur.execute(
                    """
                    SELECT user_id, COUNT(*)::BIGINT * 100 AS visible_vectors
                    FROM user_accessible_documents
                    GROUP BY user_id
                    """
                )
            else:
                cur.execute(
                    """
                    SELECT ur.user_id, COUNT(DISTINCT pa.document_id)::BIGINT * 100 AS visible_vectors
                    FROM userroles ur
                    JOIN permissionassignment pa ON pa.role_id = ur.role_id
                    GROUP BY ur.user_id
                    """
                )
            return {int(user_id): int(count or 0) for user_id, count in cur.fetchall()}
    finally:
        conn.close()


def query_vectors_by_user(args: argparse.Namespace) -> dict[int, np.ndarray]:
    if not args.query_file or not args.query_file.exists():
        return {}
    try:
        rows = json.loads(args.query_file.read_text())
    except Exception:
        return {}
    result: dict[int, np.ndarray] = {}
    for row in rows:
        if not isinstance(row, dict) or "user_id" not in row or "query_vector" not in row:
            continue
        user_id = int(row["user_id"])
        result.setdefault(user_id, parse_vector(row["query_vector"]))
    return result


def load_route_summaries(args: argparse.Namespace) -> dict[str, dict[int, RouteSummary]]:
    sys.path.insert(0, str(PROJECT_ROOT))
    from basic_benchmark.direct_pg_qps import (  # pylint: disable=import-outside-toplevel
        load_honeybee_routes,
        load_ours_routes,
        load_veda_routes,
        resolve_versioned_plan,
    )

    loaders = {
        "ours": lambda: load_ours_routes(resolve_versioned_plan("ours", float(args.memory_ratio))),
        "honeybee": lambda: load_honeybee_routes(resolve_versioned_plan("honeybee", float(args.memory_ratio))),
        "veda": lambda: load_veda_routes("veda", resolve_versioned_plan("veda", float(args.memory_ratio))),
        "effveda": lambda: load_veda_routes("effveda", resolve_versioned_plan("effveda", float(args.memory_ratio))),
    }
    all_routes = {method: loaders[method]() for method in args.methods if method in loaders}
    counts = table_row_counts(
        args,
        (
            route.table_name
            for routes_by_user in all_routes.values()
            for routes in routes_by_user.values()
            for route in routes
        ),
    )
    visible_counts = visible_vector_counts(args)

    summaries: dict[str, dict[int, RouteSummary]] = {}
    for method, routes_by_user in all_routes.items():
        method_summaries: dict[int, RouteSummary] = {}
        for user_id, routes in routes_by_user.items():
            route_tables = tuple(str(route.table_name) for route in routes)
            route_vector_count = 0
            accessible_vector_count = 0
            pure_route_count = 0
            for route in routes:
                partition_vectors = int(getattr(route, "partition_vectors", 0) or 0)
                if partition_vectors <= 0:
                    partition_vectors = int(counts.get(str(route.table_name), 0))
                route_vector_count += partition_vectors
                accessible_vector_count += int(getattr(route, "accessible_vectors", 0) or 0)
                if bool(getattr(route, "pure", False)):
                    pure_route_count += 1
            if accessible_vector_count <= 0:
                if routes and pure_route_count == len(routes):
                    accessible_vector_count = route_vector_count
                else:
                    accessible_vector_count = min(int(visible_counts.get(int(user_id), 0)), route_vector_count)
            else:
                accessible_vector_count = min(accessible_vector_count, route_vector_count)
            selectivity = accessible_vector_count / route_vector_count if route_vector_count > 0 else 0.0
            method_summaries[int(user_id)] = RouteSummary(
                method=method,
                user_id=int(user_id),
                route_tables=route_tables,
                route_count=len(route_tables),
                route_vector_count=route_vector_count,
                accessible_vector_count=accessible_vector_count,
                selectivity=selectivity,
            )
        summaries[method] = method_summaries
    return summaries


def choose_route_user(args: argparse.Namespace, summaries: dict[str, dict[int, RouteSummary]]) -> int | None:
    if args.route_user_id is not None:
        return int(args.route_user_id)
    required = [method for method in ("ours", "honeybee", "veda", "effveda") if method in summaries]
    if not required or "ours" not in summaries:
        return None
    common_users = set(summaries["ours"])
    for method in required:
        common_users &= set(summaries[method])
    if not common_users:
        return None

    def score(user_id: int) -> tuple[float, float, float]:
        ours = summaries["ours"][user_id]
        baselines = [summaries[method][user_id] for method in required if method != "ours"]
        route_gap = sum(max(0, item.route_count - ours.route_count) for item in baselines)
        vector_gap = sum(max(0, item.route_vector_count - ours.route_vector_count) for item in baselines)
        compactness = -ours.route_count
        return (float(route_gap), float(vector_gap), float(compactness))

    return max(common_users, key=score)


def fit_projection(
    args: argparse.Namespace,
    rows: list[SampleRow],
    extra_vectors: list[np.ndarray] | None = None,
) -> ProjectionResult:
    extra_vectors = extra_vectors or []
    row_vectors = np.stack([row.vector for row in rows]).astype(np.float32, copy=False)
    if extra_vectors:
        vectors = np.vstack([row_vectors, np.stack(extra_vectors).astype(np.float32, copy=False)])
    else:
        vectors = row_vectors

    fit_indices = np.arange(len(rows))
    if args.fit_unique_vectors:
        first_seen: dict[tuple[int, int], int] = {}
        for index, row in enumerate(rows):
            first_seen.setdefault((row.block_id, row.document_id), index)
        fit_indices = np.array(sorted(first_seen.values()), dtype=np.int64)

    fit_vectors = vectors[fit_indices]
    scaler = StandardScaler(with_mean=True, with_std=True)
    fit_scaled = scaler.fit_transform(fit_vectors)
    all_scaled = scaler.transform(vectors)

    pca_components = min(int(args.pca_components), fit_scaled.shape[1], fit_scaled.shape[0] - 1)
    pca_components = max(2, pca_components)
    pca = PCA(n_components=pca_components, random_state=int(args.seed))
    fit_pca = pca.fit_transform(fit_scaled)
    all_pca = pca.transform(all_scaled)

    if args.projection == "pca":
        all_2d = all_pca[:, :2].astype(np.float32, copy=False)
        return ProjectionResult(all_2d[: len(rows)], all_2d[len(rows) :])

    try:
        import umap
    except Exception as exc:  # pragma: no cover - fallback for environments without UMAP
        print(f"[projection] UMAP unavailable ({exc}); falling back to PCA-2D.", file=sys.stderr)
        all_2d = all_pca[:, :2].astype(np.float32, copy=False)
        return ProjectionResult(all_2d[: len(rows)], all_2d[len(rows) :])

    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=int(args.umap_neighbors),
        min_dist=float(args.umap_min_dist),
        metric="euclidean",
        random_state=int(args.seed),
        transform_seed=int(args.seed),
    )
    fit_2d = reducer.fit_transform(fit_pca)
    if len(fit_indices) == len(rows):
        row_coords = fit_2d.astype(np.float32, copy=False)
        extra_coords = reducer.transform(all_pca[len(rows) :]).astype(np.float32, copy=False) if extra_vectors else np.empty((0, 2), dtype=np.float32)
        return ProjectionResult(row_coords, extra_coords)
    all_2d = reducer.transform(all_pca).astype(np.float32, copy=False)
    return ProjectionResult(all_2d[: len(rows)], all_2d[len(rows) :])


def partition_rank(rows: list[SampleRow]) -> dict[tuple[str, str], int]:
    counts: Counter[tuple[str, str]] = Counter((row.method, row.partition_name) for row in rows)
    ranks: dict[tuple[str, str], int] = {}
    grouped: defaultdict[str, list[tuple[str, int]]] = defaultdict(list)
    for (method, partition_name), count in counts.items():
        grouped[method].append((partition_name, count))
    for method, entries in grouped.items():
        for rank, (partition_name, _) in enumerate(sorted(entries, key=lambda item: (-item[1], item[0])), start=1):
            ranks[(method, partition_name)] = rank
    return ranks


def color_for_partition(index: int, total: int) -> tuple[float, float, float, float]:
    cmap = plt.get_cmap("tab20" if total <= 20 else "hsv")
    if total <= 1:
        return cmap(0.0)
    return cmap((index % total) / max(total - 1, 1))


def add_hull(ax, points: np.ndarray, color: tuple[float, float, float, float]) -> None:
    if len(points) < 3:
        return
    try:
        from scipy.spatial import ConvexHull
    except Exception:
        return
    try:
        hull = ConvexHull(points)
    except Exception:
        return
    polygon = Polygon(
        points[hull.vertices],
        closed=True,
        facecolor=color,
        edgecolor=color,
        linewidth=0.8,
        alpha=0.10,
        zorder=1,
    )
    ax.add_patch(polygon)


def plot_projection(args: argparse.Namespace, rows: list[SampleRow], coords: np.ndarray) -> None:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    ranks = partition_rank(rows)
    methods = [method for method in METHOD_ORDER if any(row.method == method for row in rows)]
    methods.extend(sorted({row.method for row in rows} - set(methods)))

    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4), sharex=True, sharey=True)
    axes_flat = list(axes.ravel())

    x_pad = (float(coords[:, 0].max()) - float(coords[:, 0].min())) * 0.05 or 1.0
    y_pad = (float(coords[:, 1].max()) - float(coords[:, 1].min())) * 0.05 or 1.0
    xlim = (float(coords[:, 0].min()) - x_pad, float(coords[:, 0].max()) + x_pad)
    ylim = (float(coords[:, 1].min()) - y_pad, float(coords[:, 1].max()) + y_pad)

    for ax, method in zip(axes_flat, methods):
        indices = [index for index, row in enumerate(rows) if row.method == method]
        method_rows = [rows[index] for index in indices]
        method_coords = coords[indices]
        partitions = sorted({row.partition_name for row in method_rows})
        partition_count = len(partitions)
        colored = {
            partition
            for partition in partitions
            if ranks.get((method, partition), 10**9) <= int(args.top_colored_partitions)
        }

        ax.scatter(
            method_coords[:, 0],
            method_coords[:, 1],
            s=float(args.point_size),
            c="#D0D0D0",
            alpha=float(args.grey_alpha),
            linewidths=0,
            rasterized=True,
            zorder=2,
        )

        for rank, partition in enumerate(partitions):
            if partition not in colored and not args.color_all_partitions:
                continue
            part_indices = [i for i, row in enumerate(method_rows) if row.partition_name == partition]
            if not part_indices:
                continue
            part_coords = method_coords[part_indices]
            color = color_for_partition(rank, max(partition_count, 1))
            if args.draw_hulls:
                add_hull(ax, part_coords, color)
            ax.scatter(
                part_coords[:, 0],
                part_coords[:, 1],
                s=float(args.point_size),
                color=color,
                alpha=float(args.point_alpha),
                linewidths=0,
                rasterized=True,
                zorder=3,
            )

        ax.set_title(
            f"{METHOD_LABELS.get(method, method.upper())}: {partition_count} partitions, {len(indices)} samples",
            fontsize=10,
            pad=6,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#B0B0B0")

    for ax in axes_flat[len(methods) :]:
        ax.axis("off")

    fig.suptitle(
        f"Shared {args.projection.upper()} projection of sampled partition vectors",
        fontsize=12,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output, bbox_inches="tight")
    if args.write_png:
        fig.savefig(output.with_suffix(".png"), dpi=int(args.png_dpi), bbox_inches="tight")
    plt.close(fig)


def _format_count(value: int) -> str:
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return str(value)


def plot_route_projection(
    args: argparse.Namespace,
    rows: list[SampleRow],
    coords: np.ndarray,
    route_summaries: dict[str, dict[int, RouteSummary]],
    user_id: int,
    query_coord: np.ndarray | None = None,
) -> None:
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    methods = [method for method in METHOD_ORDER if any(row.method == method for row in rows)]
    methods.extend(sorted({row.method for row in rows} - set(methods)))
    fig, axes = plt.subplots(2, 2, figsize=(8.0, 6.4), sharex=True, sharey=True)
    axes_flat = list(axes.ravel())

    x_pad = (float(coords[:, 0].max()) - float(coords[:, 0].min())) * 0.05 or 1.0
    y_pad = (float(coords[:, 1].max()) - float(coords[:, 1].min())) * 0.05 or 1.0
    xlim = (float(coords[:, 0].min()) - x_pad, float(coords[:, 0].max()) + x_pad)
    ylim = (float(coords[:, 1].min()) - y_pad, float(coords[:, 1].max()) + y_pad)

    for ax, method in zip(axes_flat, methods):
        indices = [index for index, row in enumerate(rows) if row.method == method]
        method_rows = [rows[index] for index in indices]
        method_coords = coords[indices]
        summary = route_summaries.get(method, {}).get(int(user_id))
        route_tables = set(summary.route_tables if summary else ())

        ax.scatter(
            method_coords[:, 0],
            method_coords[:, 1],
            s=float(args.point_size),
            c="#CFCFCF",
            alpha=float(args.grey_alpha),
            linewidths=0,
            rasterized=True,
            zorder=1,
        )

        highlighted_indices = [
            local_index
            for local_index, row in enumerate(method_rows)
            if row.partition_name in route_tables
        ]
        if highlighted_indices:
            highlighted_coords = method_coords[highlighted_indices]
            ax.scatter(
                highlighted_coords[:, 0],
                highlighted_coords[:, 1],
                s=float(args.route_point_size),
                c="#D62728",
                alpha=float(args.route_alpha),
                linewidths=0,
                rasterized=True,
                zorder=3,
            )
            if args.draw_route_hulls:
                for table_name in sorted(route_tables):
                    part_indices = [
                        local_index
                        for local_index, row in enumerate(method_rows)
                        if row.partition_name == table_name
                    ]
                    if part_indices:
                        add_hull(ax, method_coords[part_indices], (0.84, 0.15, 0.16, 1.0))

        if query_coord is not None and query_coord.size == 2:
            ax.scatter(
                [float(query_coord[0])],
                [float(query_coord[1])],
                marker="*",
                s=120,
                color="#111111",
                edgecolors="white",
                linewidths=0.7,
                zorder=5,
            )

        partition_count = len({row.partition_name for row in method_rows})
        if summary:
            subtitle = (
                f"routes={summary.route_count}, "
                f"route vec={_format_count(summary.route_vector_count)}, "
                f"sel={summary.selectivity:.3f}"
            )
        else:
            subtitle = "no route metadata"
        ax.set_title(
            f"{METHOD_LABELS.get(method, method.upper())}: {partition_count} partitions\n{subtitle}",
            fontsize=9.5,
            pad=6,
        )
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(0.7)
            spine.set_color("#B0B0B0")

    for ax in axes_flat[len(methods) :]:
        ax.axis("off")

    fig.suptitle(
        f"Route-level view for user {user_id}: grey=sampled partitions, red=visited partitions",
        fontsize=12,
        y=0.985,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    fig.savefig(output, bbox_inches="tight")
    if args.write_png:
        fig.savefig(output.with_suffix(".png"), dpi=int(args.png_dpi), bbox_inches="tight")
    plt.close(fig)


def write_samples_csv(path: Path, rows: list[SampleRow], coords: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "method",
        "registry_id",
        "partition_name",
        "partition_index",
        "block_id",
        "document_id",
        "x",
        "y",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row, xy in zip(rows, coords):
            writer.writerow(
                {
                    "method": row.method,
                    "registry_id": row.registry_id,
                    "partition_name": row.partition_name,
                    "partition_index": row.partition_index,
                    "block_id": row.block_id,
                    "document_id": row.document_id,
                    "x": float(xy[0]),
                    "y": float(xy[1]),
                }
            )


def write_partition_summary(path: Path, tables: list[PartitionTable], rows: list[SampleRow]) -> None:
    counts: Counter[tuple[str, int, str]] = Counter(
        (row.method, row.registry_id, row.partition_name) for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "method",
                "registry_id",
                "memory_ratio",
                "partition_index",
                "partition_name",
                "sampled_rows",
            ],
        )
        writer.writeheader()
        for table in tables:
            writer.writerow(
                {
                    "method": table.method,
                    "registry_id": table.registry_id,
                    "memory_ratio": table.memory_ratio,
                    "partition_index": table.partition_index,
                    "partition_name": table.relation_name,
                    "sampled_rows": counts[(table.method, table.registry_id, table.relation_name)],
                }
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot a shared 2D projection of sampled partition rows.")
    parser.add_argument("--dbname", default=None, help="Database name. Defaults to config.json or DBNAME.")
    parser.add_argument("--db-user", default=None)
    parser.add_argument("--db-password", default=None)
    parser.add_argument("--db-host", default=None)
    parser.add_argument("--db-port", default=None)
    parser.add_argument("--methods", nargs="+", default=list(METHOD_ORDER))
    parser.add_argument("--memory-ratio", type=float, default=1.5)
    parser.add_argument("--registries", nargs="*", type=int, default=None)
    parser.add_argument("--samples-per-partition", type=int, default=80)
    parser.add_argument("--plot-mode", choices=["overview", "route"], default="route")
    parser.add_argument("--route-user-id", type=int, default=None)
    parser.add_argument("--query-file", type=Path, default=PROJECT_ROOT / "basic_benchmark/query_dataset.json")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--projection", choices=["umap", "pca"], default="umap")
    parser.add_argument("--pca-components", type=int, default=50)
    parser.add_argument("--umap-neighbors", type=int, default=25)
    parser.add_argument("--umap-min-dist", type=float, default=0.08)
    parser.add_argument("--fit-unique-vectors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--top-colored-partitions", type=int, default=16)
    parser.add_argument("--color-all-partitions", action="store_true")
    parser.add_argument("--draw-hulls", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--point-size", type=float, default=4.5)
    parser.add_argument("--point-alpha", type=float, default=0.72)
    parser.add_argument("--grey-alpha", type=float, default=0.22)
    parser.add_argument("--route-point-size", type=float, default=7.5)
    parser.add_argument("--route-alpha", type=float, default=0.88)
    parser.add_argument("--draw-route-hulls", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--write-png", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--png-dpi", type=int, default=240)
    parser.add_argument("--samples-csv", type=Path, default=None)
    parser.add_argument("--partition-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    tables = resolve_partition_tables(args)
    if not tables:
        raise SystemExit("No partition tables matched the requested methods/registries.")

    by_method = Counter(table.method for table in tables)
    print("[projection] matched partition tables: " + ", ".join(f"{m}={c}" for m, c in sorted(by_method.items())))

    rows = sample_partition_rows(args, tables)
    if not rows:
        raise SystemExit("No sample rows were fetched from the matched partition tables.")
    print(f"[projection] sampled rows: {len(rows)}")

    route_summaries: dict[str, dict[int, RouteSummary]] = {}
    route_user_id: int | None = None
    query_vectors: dict[int, np.ndarray] = {}
    extra_vectors: list[np.ndarray] = []
    if args.plot_mode == "route":
        route_summaries = load_route_summaries(args)
        route_user_id = choose_route_user(args, route_summaries)
        if route_user_id is None:
            raise SystemExit("Unable to choose a route user. Pass --route-user-id explicitly.")
        query_vectors = query_vectors_by_user(args)
        if route_user_id in query_vectors:
            extra_vectors.append(query_vectors[route_user_id])
        print(f"[projection] route user: {route_user_id}")

    projection = fit_projection(args, rows, extra_vectors=extra_vectors)
    coords = projection.row_coords
    if args.plot_mode == "route":
        query_coord = projection.extra_coords[0] if len(projection.extra_coords) else None
        plot_route_projection(args, rows, coords, route_summaries, int(route_user_id), query_coord=query_coord)
    else:
        plot_projection(args, rows, coords)

    samples_csv = args.samples_csv or args.output.with_suffix(".samples.csv")
    partition_csv = args.partition_csv or args.output.with_suffix(".partitions.csv")
    write_samples_csv(samples_csv, rows, coords)
    write_partition_summary(partition_csv, tables, rows)

    print(f"[projection] figure={args.output}")
    if args.write_png:
        print(f"[projection] png={args.output.with_suffix('.png')}")
    print(f"[projection] samples={samples_csv}")
    print(f"[projection] partitions={partition_csv}")


if __name__ == "__main__":
    main()
