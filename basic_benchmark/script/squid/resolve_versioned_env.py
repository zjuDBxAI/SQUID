#!/usr/bin/env python3
"""Emit shell exports that point SQL-time benchmarks at a versioned plan.

The legacy query-time scripts measure SQL EXPLAIN time through method search
functions that normally read "current" metadata tables.  This helper maps a
ready registry entry to environment variables, so those search functions can
read an already materialized versioned layout without rebuilding or copying it.
"""

from __future__ import annotations

import argparse
import shlex
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from services.config import get_db_connection  # noqa: E402


def normalize_method(value: str) -> str:
    normalized = str(value).strip().lower().replace("-", "_")
    if normalized in {"squid", "kmeans"}:
        return "ours"
    if normalized in {"anonysys", "dynamic_partition"}:
        return "honeybee"
    if normalized in {"qdtree", "qd_tree"}:
        return "hqi"
    return normalized


def shell_export(name: str, value: object) -> str:
    return f"export {name}={shlex.quote(str(value))}"


def resolve_registry(method: str, memory_ratio: float) -> tuple[int, str, dict]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT to_regclass('benchmark_plan_registry') IS NOT NULL")
            if not cur.fetchone()[0]:
                raise RuntimeError("benchmark_plan_registry does not exist")
            cur.execute(
                """
                SELECT registry_id, table_prefix, metadata
                FROM benchmark_plan_registry
                WHERE method = %s
                  AND memory_ratio = %s
                  AND state = 'ready'
                ORDER BY registry_id DESC
                LIMIT 1
                """,
                [method, float(memory_ratio)],
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"No ready versioned plan for method={method} memory_ratio={memory_ratio}")
            return int(row[0]), str(row[1]), dict(row[2] or {})
    finally:
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Resolve versioned-plan env vars for SQL query-time scripts.")
    parser.add_argument("--method", required=True)
    parser.add_argument("--memory-ratio", type=float, required=True)
    args = parser.parse_args()

    method = normalize_method(args.method)
    registry_method = "honeybee" if method == "honeybee" else method
    registry_id, table_prefix, metadata = resolve_registry(registry_method, float(args.memory_ratio))

    exports: list[tuple[str, object]] = [
        ("BENCHMARK_VERSIONED_REGISTRY_ID", registry_id),
        ("BENCHMARK_VERSIONED_METHOD", method),
        ("BENCHMARK_VERSIONED_TABLE_PREFIX", table_prefix),
    ]

    metadata_tables = dict(metadata.get("metadata_tables") or {})
    if method == "ours":
        if not metadata_tables:
            raise RuntimeError(f"Versioned OURS plan {registry_id} has no metadata_tables")
        exports.extend(
            [
                ("KMEANS_PLAN_TABLE", metadata_tables["plan"]),
                ("KMEANS_PARTITION_TABLE", metadata_tables["partition"]),
                ("KMEANS_PATTERN_TABLE", metadata_tables["pattern"]),
                ("KMEANS_ROUTE_TABLE", metadata_tables["route"]),
            ]
        )
    elif method in {"veda", "effveda"}:
        if not metadata_tables:
            raise RuntimeError(f"Versioned {method} plan {registry_id} has no metadata_tables")
        exports.extend(
            [
                ("VEDA_PLAN_TABLE", metadata_tables["plan"]),
                ("VEDA_NODE_TABLE", metadata_tables["partition"]),
                ("VEDA_PATTERN_TABLE", metadata_tables["pattern"]),
                ("VEDA_ROUTE_TABLE", metadata_tables["route"]),
            ]
        )
    elif method == "honeybee":
        mapping_table = metadata.get("mapping_table") or f"{table_prefix}_comb_role_partitions"
        exports.extend(
            [
                ("HONEYBEE_MAPPING_TABLE", mapping_table),
                ("HONEYBEE_TABLE_PREFIX", table_prefix),
                ("DYNAMIC_PARTITION_MAPPING_TABLE", mapping_table),
                ("DYNAMIC_PARTITION_TABLE_PREFIX", table_prefix),
            ]
        )
    else:
        raise RuntimeError(f"Versioned SQL query-time env is not implemented for method={method}")

    for name, value in exports:
        print(shell_export(name, value))


if __name__ == "__main__":
    main()
