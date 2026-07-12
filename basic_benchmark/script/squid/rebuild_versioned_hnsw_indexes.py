#!/usr/bin/env python3
"""Rebuild ordinary HNSW indexes for versioned partition tables.

This tool does not rebuild plans or repartition data. It resolves ready plans from
benchmark_plan_registry by method and memory ratio, then rebuilds only vector ANN
indexes on relations registered as partition tables.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from psycopg2 import sql


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from services.config import get_db_connection, get_maintenance_settings  # noqa: E402


METHODS = ("ours", "veda", "effveda")
ANN_INDEX_AMS = ("hnsw", "squidhnsw", "vedahnsw", "ivfflat")


def parse_values(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        result.extend(item for item in str(value).replace(",", " " ).split() if item)
    return result


def resolve_plan(cur, method: str, memory_ratio: float) -> tuple[int, str]:
    cur.execute(
        """
        SELECT registry_id, table_prefix
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
    return int(row[0]), str(row[1])


def partition_tables(cur, registry_id: int) -> list[str]:
    cur.execute(
        """
        SELECT relation_name
        FROM benchmark_plan_relations
        WHERE registry_id = %s
          AND relation_kind = 'partition'
        ORDER BY relation_name
        """,
        [int(registry_id)],
    )
    return [str(row[0]) for row in cur.fetchall()]


def vector_indexes(cur, table_name: str) -> list[tuple[str, str]]:
    cur.execute(
        """
        SELECT index_class.relname, am.amname
        FROM pg_index ix
        JOIN pg_class table_class ON table_class.oid = ix.indrelid
        JOIN pg_class index_class ON index_class.oid = ix.indexrelid
        JOIN pg_am am ON am.oid = index_class.relam
        JOIN pg_attribute attr ON attr.attrelid = table_class.oid
                              AND attr.attnum = ANY(ix.indkey)
        WHERE table_class.oid = to_regclass(%s)
          AND attr.attname = 'vector'
          AND am.amname = ANY(%s)
        ORDER BY index_class.relname
        """,
        [table_name, list(ANN_INDEX_AMS)],
    )
    return [(str(row[0]), str(row[1])) for row in cur.fetchall()]


def safe_index_name(table_name: str) -> str:
    candidate = f"{table_name}_vector_hnsw_idx"
    if len(candidate) <= 63:
        return candidate
    import hashlib

    digest = hashlib.blake2b(candidate.encode("utf-8"), digest_size=6).hexdigest()
    return f"idx_{digest}_vector_hnsw"[:63]


def configure_build_session(cur) -> None:
    settings = get_maintenance_settings()
    cur.execute(sql.SQL("SET maintenance_work_mem = {}").format(sql.Literal(f"{settings['maintenance_work_mem_gb']}GB")))
    cur.execute(sql.SQL("SET max_parallel_maintenance_workers = {}").format(sql.Literal(int(settings["max_parallel_maintenance_workers"]))))
    cur.execute("SET synchronous_commit = OFF")


def rebuild_table(cur, table_name: str, *, dry_run: bool, force: bool, progress_label: str | None = None) -> dict:
    existing_indexes = vector_indexes(cur, table_name)
    has_plain_hnsw = any(am == "hnsw" for _name, am in existing_indexes)
    has_non_hnsw_ann = any(am != "hnsw" for _name, am in existing_indexes)
    new_index = safe_index_name(table_name)
    actions = {
        "table": table_name,
        "drop_indexes": [{"name": name, "am": am} for name, am in existing_indexes],
        "create_index": new_index,
    }
    if has_plain_hnsw and not has_non_hnsw_ann and not force:
        actions["skipped"] = "already_hnsw"
        if progress_label:
            print(f"{progress_label} skip table={table_name} already_hnsw=true", flush=True)
        return actions
    if dry_run:
        if progress_label:
            print(f"{progress_label} dry-run table={table_name} drop={len(existing_indexes)} create={new_index}", flush=True)
        return actions

    if progress_label:
        print(f"{progress_label} rebuilding table={table_name} drop={len(existing_indexes)} create={new_index}", flush=True)
    for index_name, _am in existing_indexes:
        cur.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(index_name)))
    cur.execute(
        sql.SQL(
            "CREATE INDEX {} ON {} USING hnsw (vector vector_l2_ops) WITH (m = 16, ef_construction = 64)"
        ).format(sql.Identifier(new_index), sql.Identifier(table_name))
    )
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild HNSW indexes for versioned SQUID/VEDA/EffVeda partitions")
    parser.add_argument("--method", nargs="+", required=True, help="Methods: ours veda effveda; comma-separated is accepted")
    parser.add_argument("--memory-ratio", nargs="+", required=True, help="Memory ratios such as 1.0 2.0; comma-separated is accepted")
    parser.add_argument("--execute", action="store_true", help="Actually rebuild indexes. Omit for dry-run.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max partition tables per method/memory, useful for smoke tests")
    parser.add_argument("--force", action="store_true", help="Rebuild even if a partition table already has an ordinary HNSW index.")
    args = parser.parse_args()

    methods = parse_values(args.method)
    memory_ratios = [float(value) for value in parse_values(args.memory_ratio)]
    invalid = sorted(set(methods) - set(METHODS))
    if invalid:
        raise SystemExit(f"Unsupported methods for this tool: {invalid}; expected {METHODS}")

    dry_run = not bool(args.execute)
    conn = get_db_connection()
    summaries: list[dict] = []
    try:
        with conn.cursor() as cur:
            configure_build_session(cur)
            for method in methods:
                for memory_ratio in memory_ratios:
                    registry_id, table_prefix = resolve_plan(cur, method, memory_ratio)
                    tables = partition_tables(cur, registry_id)
                    if args.limit is not None:
                        tables = tables[: max(0, int(args.limit))]
                    if not tables:
                        raise RuntimeError(f"No registered partition tables for method={method} memory_ratio={memory_ratio}")
                    print(
                        f"[start] method={method} memory={memory_ratio:g} registry_id={registry_id} "
                        f"tables={len(tables)} dry_run={dry_run} plan_rebuild=false",
                        flush=True,
                    )
                    table_actions = []
                    total_tables = len(tables)
                    for table_index, table_name in enumerate(tables, start=1):
                        progress_label = f"[{method} memory={memory_ratio:g} {table_index}/{total_tables}]"
                        try:
                            action = rebuild_table(
                                cur,
                                table_name,
                                dry_run=dry_run,
                                force=bool(args.force),
                                progress_label=progress_label,
                            )
                            table_actions.append(action)
                            if not dry_run:
                                conn.commit()
                                if action.get("skipped") != "already_hnsw":
                                    print(f"{progress_label} committed table={table_name}", flush=True)
                        except BaseException:
                            conn.rollback()
                            print(f"{progress_label} failed table={table_name}", flush=True)
                            raise
                    print(
                        f"[done] method={method} memory={memory_ratio:g} tables={len(table_actions)} dry_run={dry_run}",
                        flush=True,
                    )
                    summaries.append({
                        "method": method,
                        "memory_ratio": memory_ratio,
                        "registry_id": registry_id,
                        "table_prefix": table_prefix,
                        "tables": len(table_actions),
                        "dry_run": dry_run,
                        "actions": table_actions[:5],
                        "truncated_actions": max(0, len(table_actions) - 5),
                    })
            if dry_run:
                conn.rollback()
    except BaseException:
        conn.rollback()
        raise
    finally:
        conn.close()

    print(json.dumps(summaries, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
