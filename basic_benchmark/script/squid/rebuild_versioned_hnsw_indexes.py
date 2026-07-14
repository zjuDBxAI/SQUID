#!/usr/bin/env python3
"""Rebuild ANN indexes for versioned partition tables.

This tool does not rebuild plans or repartition data. It resolves ready plans from
benchmark_plan_registry by method and memory ratio, then rebuilds only vector ANN
indexes on relations registered as partition tables.
"""

from __future__ import annotations

import argparse
import hashlib
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
INDEX_TYPES = ("hnsw", "native", "squidhnsw", "vedahnsw", "ivfflat")
POSTGRES_IDENTIFIER_LIMIT = 63


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


def relation_names(cur, registry_id: int, relation_kind: str) -> list[str]:
    cur.execute(
        """
        SELECT relation_name
        FROM benchmark_plan_relations
        WHERE registry_id = %s
          AND relation_kind = %s
        ORDER BY relation_name
        """,
        [int(registry_id), str(relation_kind)],
    )
    return [str(row[0]) for row in cur.fetchall()]


def veda_node_kinds(cur, registry_id: int, plan_id: int) -> dict[str, str]:
    metadata_tables = relation_names(cur, registry_id, "partition_metadata")
    if not metadata_tables:
        return {}
    node_kinds: dict[str, str] = {}
    for metadata_table in metadata_tables:
        cur.execute(
            sql.SQL(
                """
                SELECT table_name, node_kind
                FROM {}
                WHERE plan_id = %s
                """
            ).format(sql.Identifier(metadata_table)),
            [int(plan_id)],
        )
        for table_name, node_kind in cur.fetchall():
            node_kinds[str(table_name)] = str(node_kind)
    return node_kinds


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


def safe_index_name(table_name: str, suffix: str) -> str:
    candidate = f"{table_name}_{suffix}"
    if len(candidate) <= POSTGRES_IDENTIFIER_LIMIT:
        return candidate

    digest = hashlib.blake2b(candidate.encode("utf-8"), digest_size=8).hexdigest()
    compact_suffix = suffix.replace("_", "")
    return f"idx_{digest}_{compact_suffix}"[:POSTGRES_IDENTIFIER_LIMIT]


def target_index_type(method: str, requested_index_type: str) -> str:
    requested = str(requested_index_type).strip().lower()
    if requested == "native":
        return "squidhnsw" if method == "ours" else "vedahnsw"
    if requested == "squidhnsw" and method != "ours":
        raise ValueError("--index-type squidhnsw is only valid for method=ours")
    if requested == "vedahnsw" and method not in {"veda", "effveda"}:
        raise ValueError("--index-type vedahnsw is only valid for method=veda/effveda")
    return requested


def configure_build_session(cur) -> None:
    settings = get_maintenance_settings()
    cur.execute(sql.SQL("SET maintenance_work_mem = {}").format(sql.Literal(f"{settings['maintenance_work_mem_gb']}GB")))
    cur.execute(sql.SQL("SET max_parallel_maintenance_workers = {}").format(sql.Literal(int(settings["max_parallel_maintenance_workers"]))))
    cur.execute("SET synchronous_commit = OFF")


def ensure_auxiliary_indexes(cur, method: str, table_name: str) -> list[str]:
    created_indexes: list[str] = []
    if method == "ours":
        pattern_idx = safe_index_name(table_name, "pattern_idx")
        pattern_document_idx = safe_index_name(table_name, "pattern_document_idx")
        cur.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (pattern_id)").format(
                sql.Identifier(pattern_idx),
                sql.Identifier(table_name),
            )
        )
        cur.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (pattern_id, document_id)").format(
                sql.Identifier(pattern_document_idx),
                sql.Identifier(table_name),
            )
        )
        created_indexes.extend([pattern_idx, pattern_document_idx])
    elif method in {"veda", "effveda"}:
        pattern_idx = safe_index_name(table_name, "pattern_idx")
        role_ids_idx = safe_index_name(table_name, "role_ids_idx")
        cur.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (pattern_id)").format(
                sql.Identifier(pattern_idx),
                sql.Identifier(table_name),
            )
        )
        cur.execute(
            sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} USING GIN (role_ids)").format(
                sql.Identifier(role_ids_idx),
                sql.Identifier(table_name),
            )
        )
        created_indexes.extend([pattern_idx, role_ids_idx])
    return created_indexes


def create_vector_index_sql(table_name: str, index_name: str, index_type: str, *, hnsw_m: int, hnsw_ef_construction: int) -> sql.SQL:
    if index_type in {"hnsw", "squidhnsw", "vedahnsw"}:
        include_clause = sql.SQL(" INCLUDE (pattern_id)") if index_type in {"squidhnsw", "vedahnsw"} else sql.SQL("")
        return sql.SQL(
            """
            CREATE INDEX {} ON {} USING {} (vector vector_l2_ops){}
            WITH (m = {m}, ef_construction = {ef})
            """
        ).format(
            sql.Identifier(index_name),
            sql.Identifier(table_name),
            sql.SQL(index_type),
            include_clause,
            m=sql.Literal(int(hnsw_m)),
            ef=sql.Literal(int(hnsw_ef_construction)),
        )
    if index_type == "ivfflat":
        return sql.SQL("CREATE INDEX {} ON {} USING ivfflat (vector vector_l2_ops)").format(
            sql.Identifier(index_name),
            sql.Identifier(table_name),
        )
    raise ValueError(f"Unsupported index type: {index_type}")


def rebuild_table(
    cur,
    method: str,
    table_name: str,
    *,
    index_type: str,
    create_ann: bool,
    dry_run: bool,
    force: bool,
    hnsw_m: int,
    hnsw_ef_construction: int,
    progress_label: str | None = None,
) -> dict:
    existing_indexes = vector_indexes(cur, table_name)
    has_target_index = any(am == index_type for _name, am in existing_indexes)
    has_other_ann = any(am != index_type for _name, am in existing_indexes)
    new_index = safe_index_name(table_name, "vector_idx") if create_ann else None
    auxiliary_indexes = [
        safe_index_name(table_name, "pattern_idx"),
        safe_index_name(table_name, "pattern_document_idx") if method == "ours" else safe_index_name(table_name, "role_ids_idx"),
    ]
    actions = {
        "table": table_name,
        "index_type": index_type,
        "create_ann": bool(create_ann),
        "drop_indexes": [{"name": name, "am": am} for name, am in existing_indexes],
        "create_index": new_index,
        "auxiliary_indexes": auxiliary_indexes,
    }
    if dry_run:
        if progress_label:
            print(
                f"{progress_label} dry-run table={table_name} index_type={index_type} "
                f"create_ann={create_ann} drop={len(existing_indexes)} create={new_index}",
                flush=True,
            )
        return actions

    ensure_auxiliary_indexes(cur, method, table_name)

    if not create_ann:
        for index_name, _am in existing_indexes:
            cur.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(index_name)))
        actions["skipped"] = "ann_disabled_for_node_kind"
        if progress_label:
            print(f"{progress_label} skip ANN table={table_name} create_ann=false", flush=True)
        return actions

    if has_target_index and not has_other_ann and not force:
        actions["skipped"] = f"already_{index_type}"
        if progress_label:
            print(f"{progress_label} skip table={table_name} already_{index_type}=true", flush=True)
        return actions

    if progress_label:
        print(
            f"{progress_label} rebuilding table={table_name} index_type={index_type} "
            f"drop={len(existing_indexes)} create={new_index}",
            flush=True,
        )
    for index_name, _am in existing_indexes:
        cur.execute(sql.SQL("DROP INDEX IF EXISTS {}").format(sql.Identifier(index_name)))
    cur.execute(create_vector_index_sql(
        table_name,
        str(new_index),
        index_type,
        hnsw_m=hnsw_m,
        hnsw_ef_construction=hnsw_ef_construction,
    ))
    return actions


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild ANN indexes for versioned SQUID/VEDA/EffVeda partitions")
    parser.add_argument("--method", nargs="+", required=True, help="Methods: ours veda effveda; comma-separated is accepted")
    parser.add_argument("--memory-ratio", nargs="+", required=True, help="Memory ratios such as 1.0 2.0; comma-separated is accepted")
    parser.add_argument("--index-type", choices=INDEX_TYPES, default="hnsw",
                        help="ANN index to rebuild. native maps ours->squidhnsw and veda/effveda->vedahnsw.")
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    parser.add_argument("--execute", action="store_true", help="Actually rebuild indexes. Omit for dry-run.")
    parser.add_argument("--limit", type=int, default=None, help="Optional max partition tables per method/memory, useful for smoke tests")
    parser.add_argument("--force", action="store_true", help="Rebuild even if a partition table already has the requested ANN index.")
    parser.add_argument(
        "--veda-all-routes-ann",
        action="store_true",
        help="For VEDA/EffVeda vedahnsw rebuilds, also create ANN indexes on leftover nodes.",
    )
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
                    cur.execute(
                        "SELECT plan_id FROM benchmark_plan_registry WHERE registry_id = %s",
                        [int(registry_id)],
                    )
                    plan_id = int(cur.fetchone()[0])
                    method_index_type = target_index_type(method, args.index_type)
                    node_kinds = (
                        veda_node_kinds(cur, registry_id, plan_id)
                        if method in {"veda", "effveda"} and method_index_type == "vedahnsw"
                        else {}
                    )
                    tables = partition_tables(cur, registry_id)
                    if args.limit is not None:
                        tables = tables[: max(0, int(args.limit))]
                    if not tables:
                        raise RuntimeError(f"No registered partition tables for method={method} memory_ratio={memory_ratio}")
                    print(
                        f"[start] method={method} memory={memory_ratio:g} registry_id={registry_id} "
                        f"index_type={method_index_type} tables={len(tables)} dry_run={dry_run} plan_rebuild=false",
                        flush=True,
                    )
                    table_actions = []
                    total_tables = len(tables)
                    for table_index, table_name in enumerate(tables, start=1):
                        progress_label = f"[{method} memory={memory_ratio:g} {table_index}/{total_tables}]"
                        try:
                            action = rebuild_table(
                                cur,
                                method,
                                table_name,
                                index_type=method_index_type,
                                create_ann=not (
                                    method in {"veda", "effveda"}
                                    and method_index_type == "vedahnsw"
                                    and not bool(args.veda_all_routes_ann)
                                    and node_kinds.get(table_name, "index") != "index"
                                ),
                                dry_run=dry_run,
                                force=bool(args.force),
                                hnsw_m=int(args.hnsw_m),
                                hnsw_ef_construction=int(args.hnsw_ef_construction),
                                progress_label=progress_label,
                            )
                            table_actions.append(action)
                            if not dry_run:
                                conn.commit()
                                if not str(action.get("skipped", "")).startswith("already_"):
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
                        "index_type": method_index_type,
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
