#!/usr/bin/env python3
"""Build one versioned materialized plan and register it for direct QPS."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from psycopg2 import sql


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from plan_manifest import METHODS, create_manifest  # noqa: E402
from services.config import get_db_connection  # noqa: E402
from controller.dynamic_partition.load_result_to_database import (  # noqa: E402
    comb_role_mapping_table_name,
)


def _python() -> str:
    candidate = PROJECT_ROOT / "venv" / "bin" / "python"
    return str(candidate) if candidate.exists() else sys.executable


def _run(cmd: list[str], *, cwd: Path) -> None:
    print("[CMD]", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), check=True)


def _registry_init() -> None:
    schema = (SCRIPT_DIR / "schema.sql").read_text()
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(schema)
        conn.commit()
    finally:
        conn.close()


def _drop_relation(cur, relation_name: str) -> None:
    cur.execute("SELECT to_regclass(%s)", [f"public.{relation_name}"])
    if cur.fetchone()[0] is None:
        return
    cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(relation_name)))


def _delete_if_table_exists(cur, table_name: str, where_sql: str, params: list[object]) -> None:
    cur.execute("SELECT to_regclass(%s)", [f"public.{table_name}"])
    if cur.fetchone()[0] is None:
        return
    cur.execute(sql.SQL("DELETE FROM {} ").format(sql.Identifier(table_name)) + sql.SQL(where_sql), params)


def _relations_with_prefix(cur, table_prefix: str) -> list[str]:
    cur.execute(
        """
        SELECT tablename
        FROM pg_tables
        WHERE schemaname = current_schema()
          AND (tablename = %s OR tablename LIKE %s)
        ORDER BY tablename
        """,
        [str(table_prefix), f"{table_prefix}_%"],
    )
    return [str(row[0]) for row in cur.fetchall()]


def _cleanup_kmeans_current_metadata(cur, table_prefix: str) -> None:
    cur.execute("SELECT to_regclass('public.kmeans_current_partitions')")
    if cur.fetchone()[0] is None:
        return
    cur.execute(
        """
        SELECT DISTINCT plan_id
        FROM kmeans_current_partitions
        WHERE partition_id LIKE %s
           OR table_name LIKE %s
        """,
        [f"{table_prefix}_%", f"{table_prefix}_%"],
    )
    plan_ids = [int(row[0]) for row in cur.fetchall()]
    if not plan_ids:
        return
    cur.execute("DELETE FROM kmeans_current_routes WHERE plan_id = ANY(%s)", [plan_ids])
    cur.execute("DELETE FROM kmeans_current_partitions WHERE plan_id = ANY(%s)", [plan_ids])
    cur.execute("DELETE FROM kmeans_current_patterns WHERE plan_id = ANY(%s)", [plan_ids])
    cur.execute("DELETE FROM kmeans_current_plan WHERE plan_id = ANY(%s)", [plan_ids])


def _cleanup_veda_current_metadata(cur, method: str, table_prefix: str) -> None:
    cur.execute("SELECT to_regclass('public.veda_current_nodes')")
    if cur.fetchone()[0] is None:
        return
    cur.execute(
        """
        SELECT DISTINCT plan_id
        FROM veda_current_nodes
        WHERE table_name LIKE %s
        """,
        [f"{table_prefix}_%"],
    )
    node_plan_ids = {int(row[0]) for row in cur.fetchall()}
    cur.execute(
        """
        SELECT plan_id
        FROM veda_current_plan
        WHERE algorithm = %s
          AND metadata->>'versioned_table_prefix' = %s
        """,
        [str(method), str(table_prefix)],
    )
    plan_ids = sorted(node_plan_ids | {int(row[0]) for row in cur.fetchall()})
    if not plan_ids:
        return
    _delete_if_table_exists(cur, "veda_current_user_routes", "WHERE plan_id = ANY(%s)", [plan_ids])
    _delete_if_table_exists(cur, "veda_current_role_plan", "WHERE plan_id = ANY(%s)", [plan_ids])
    _delete_if_table_exists(cur, "veda_current_nodes", "WHERE plan_id = ANY(%s)", [plan_ids])
    _delete_if_table_exists(cur, "veda_current_patterns", "WHERE plan_id = ANY(%s)", [plan_ids])
    _delete_if_table_exists(cur, "veda_current_plan", "WHERE plan_id = ANY(%s)", [plan_ids])


def _cleanup_existing_version(manifest: dict) -> None:
    method = str(manifest["method"])
    table_prefix = str(manifest["table_prefix"])
    metadata_relations = {
        str(manifest["plan_relation"]),
        str(manifest["partition_relation"]),
        str(manifest["pattern_relation"]),
        str(manifest["route_relation"]),
    }
    if method == "honeybee":
        metadata_relations.add(comb_role_mapping_table_name(table_prefix))

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT registry_id
                FROM benchmark_plan_registry
                WHERE (method = %s AND memory_ratio = %s)
                   OR table_prefix = %s
                """,
                [method, float(manifest["memory_ratio"]), table_prefix],
            )
            registry_ids = [int(row[0]) for row in cur.fetchall()]
            if registry_ids:
                cur.execute(
                    "SELECT relation_name FROM benchmark_plan_relations WHERE registry_id = ANY(%s)",
                    [registry_ids],
                )
                metadata_relations.update(str(row[0]) for row in cur.fetchall())
                cur.execute("DELETE FROM benchmark_plan_registry WHERE registry_id = ANY(%s)", [registry_ids])

            if method == "ours":
                _cleanup_kmeans_current_metadata(cur, table_prefix)
            elif method in {"veda", "effveda"}:
                _cleanup_veda_current_metadata(cur, method, table_prefix)

            for relation_name in sorted(set(_relations_with_prefix(cur, table_prefix)) | metadata_relations):
                _drop_relation(cur, relation_name)
        conn.commit()
    finally:
        conn.close()


def _register_plan(manifest: dict, plan_id: int, metadata: dict) -> int:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO benchmark_plan_registry (
                    method, memory_ratio, plan_id, table_prefix, state,
                    measured_space_bytes, metadata
                )
                VALUES (%s, %s, %s, %s, 'ready', NULL, %s::jsonb)
                ON CONFLICT (method, memory_ratio) DO UPDATE
                SET plan_id = EXCLUDED.plan_id,
                    table_prefix = EXCLUDED.table_prefix,
                    state = 'ready',
                    measured_space_bytes = EXCLUDED.measured_space_bytes,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                RETURNING registry_id
                """,
                [
                    manifest["method"],
                    float(manifest["memory_ratio"]),
                    int(plan_id),
                    manifest["table_prefix"],
                    json.dumps(metadata),
                ],
            )
            registry_id = int(cur.fetchone()[0])
        conn.commit()
        return registry_id
    finally:
        conn.close()


def _register_relations(registry_id: int, relations: dict[str, list[str]]) -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM benchmark_plan_relations WHERE registry_id = %s", [int(registry_id)])
            for relation_kind, names in relations.items():
                for name in sorted(set(names)):
                    cur.execute(
                        """
                        INSERT INTO benchmark_plan_relations (registry_id, relation_name, relation_kind)
                        VALUES (%s, %s, %s)
                        ON CONFLICT (registry_id, relation_name) DO UPDATE
                        SET relation_kind = EXCLUDED.relation_kind
                        """,
                        [int(registry_id), str(name), str(relation_kind)],
                    )
        conn.commit()
    finally:
        conn.close()


def _copy_table_for_plan(source: str, target: str, plan_id: int) -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql.SQL("DROP TABLE IF EXISTS {} CASCADE").format(sql.Identifier(target)))
            cur.execute(
                sql.SQL("CREATE TABLE {} AS SELECT * FROM {} WHERE plan_id = %s").format(
                    sql.Identifier(target),
                    sql.Identifier(source),
                ),
                [int(plan_id)],
            )
            cur.execute(
                sql.SQL("CREATE INDEX {} ON {} (plan_id)").format(
                    sql.Identifier(_safe_index_name(target, "plan_idx")),
                    sql.Identifier(target),
                )
            )
        conn.commit()
    finally:
        conn.close()


def _safe_index_name(table_name: str, suffix: str) -> str:
    candidate = f"{table_name}_{suffix}"
    if len(candidate) <= 63:
        return candidate
    digest = hashlib.blake2b(candidate.encode("utf-8"), digest_size=6).hexdigest()
    return f"idx_{digest}_{suffix}"[:63]


def _copy_metadata_tables(manifest: dict, plan_id: int, *, kind: str) -> dict[str, str]:
    plan_relation = str(manifest["plan_relation"])
    partition_relation = str(manifest["partition_relation"])
    pattern_relation = str(manifest["pattern_relation"])
    route_relation = str(manifest["route_relation"])
    if kind == "ours":
        sources = {
            plan_relation: "kmeans_current_plan",
            partition_relation: "kmeans_current_partitions",
            pattern_relation: "kmeans_current_patterns",
            route_relation: "kmeans_current_routes",
        }
    else:
        sources = {
            plan_relation: "veda_current_plan",
            partition_relation: "veda_current_nodes",
            pattern_relation: "veda_current_patterns",
            route_relation: "veda_current_user_routes",
        }
    for target, source in sources.items():
        _copy_table_for_plan(source, target, plan_id)
    return {
        "plan": plan_relation,
        "partition": partition_relation,
        "pattern": pattern_relation,
        "route": route_relation,
    }


def _latest_kmeans_plan_id() -> int:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT plan_id FROM kmeans_current_plan ORDER BY plan_id DESC LIMIT 1")
            row = cur.fetchone()
            if row is None:
                raise RuntimeError("No kmeans plan was materialized")
            return int(row[0])
    finally:
        conn.close()


def _latest_veda_plan_id(algorithm: str) -> int:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT plan_id FROM veda_current_plan WHERE algorithm = %s ORDER BY plan_id DESC LIMIT 1",
                [algorithm],
            )
            row = cur.fetchone()
            if row is None:
                raise RuntimeError(f"No {algorithm} plan was materialized")
            return int(row[0])
    finally:
        conn.close()


def _kmeans_relations(plan_id: int, metadata_tables: dict[str, str] | None = None) -> dict[str, list[str]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT table_name FROM kmeans_current_partitions WHERE plan_id = %s", [int(plan_id)])
            partitions = [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()
    return {
        "plan_metadata": [metadata_tables["plan"] if metadata_tables else "kmeans_current_plan"],
        "partition_metadata": [metadata_tables["partition"] if metadata_tables else "kmeans_current_partitions"],
        "pattern_metadata": [metadata_tables["pattern"] if metadata_tables else "kmeans_current_patterns"],
        "route_metadata": [metadata_tables["route"] if metadata_tables else "kmeans_current_routes"],
        "partition": partitions,
    }


def _veda_relations(plan_id: int, metadata_tables: dict[str, str] | None = None) -> dict[str, list[str]]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT table_name FROM veda_current_nodes WHERE plan_id = %s", [int(plan_id)])
            partitions = [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()
    return {
        "plan_metadata": [metadata_tables["plan"] if metadata_tables else "veda_current_plan"],
        "partition_metadata": [metadata_tables["partition"] if metadata_tables else "veda_current_nodes"],
        "pattern_metadata": [metadata_tables["pattern"] if metadata_tables else "veda_current_patterns"],
        "route_metadata": [metadata_tables["route"] if metadata_tables else "veda_current_user_routes"],
        "partition": partitions,
    }


def _honeybee_relations(table_prefix: str) -> dict[str, list[str]]:
    mapping_table = comb_role_mapping_table_name(table_prefix)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tablename
                FROM pg_tables
                WHERE schemaname = current_schema()
                  AND tablename LIKE %s
                ORDER BY tablename
                """,
                [f"{table_prefix}_partition_%"],
            )
            partitions = [str(row[0]) for row in cur.fetchall()]
    finally:
        conn.close()
    return {
        "mapping_metadata": [mapping_table],
        "partition": partitions,
    }




def _grant_select_public(relations: dict[str, list[str]]) -> None:
    table_names = sorted({name for names in relations.values() for name in names})
    if not table_names:
        return
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for table_name in table_names:
                cur.execute(sql.SQL("GRANT SELECT ON {} TO PUBLIC").format(sql.Identifier(table_name)))
        conn.commit()
    finally:
        conn.close()


def _build_ours(args: argparse.Namespace, manifest: dict) -> tuple[int, dict[str, list[str]], dict]:
    table_prefix = str(manifest["table_prefix"])
    budget_ratio = max(0.0, float(args.memory_ratio) - 1.0)
    cmd = [
        _python(), "test_kmeans_partition.py",
        "--prepare", "true",
        "--private-replication-budget-ratio", str(budget_ratio),
        "--private-edge-top-d", str(args.private_edge_top_d),
        "--enable-split", str(args.enable_split).lower(),
        "--enable-index", "true",
        "--index-type", str(args.ours_index_type),
        "--statistics-type", "sql",
        "--query-num", str(args.query_num),
        "--iterations", "1",
        "--record-recall", "false",
        "--warm-up", "false",
        "--show-progress", str(args.show_progress).lower(),
        "--use-ground-truth-cache", "true",
        "--ef-search", str(args.ef_search),
        "--table-prefix", table_prefix,
        "--versioned-plan", "true",
        "--result-tag", f"versioned_{table_prefix}",
    ]
    if args.document_limit is not None:
        cmd.extend(["--document-limit", str(args.document_limit)])
    _run(cmd, cwd=PROJECT_ROOT / "basic_benchmark")
    plan_id = _latest_kmeans_plan_id()
    metadata_tables = _copy_metadata_tables(manifest, plan_id, kind="ours")
    return plan_id, _kmeans_relations(plan_id, metadata_tables), {
        "private_replication_budget_ratio": budget_ratio,
        "metadata_tables": metadata_tables,
    }


def _build_veda(args: argparse.Namespace, method: str, manifest: dict) -> tuple[int, dict[str, list[str]], dict]:
    table_prefix = str(manifest["table_prefix"])
    cmd = [
        _python(), "test_veda.py",
        "--prepare", "true",
        "--algorithm", method,
        "--enable-index", "true",
        "--index-type", str(args.veda_index_type),
        "--statistics-type", "sql",
        "--query-num", str(args.query_num),
        "--iterations", "1",
        "--record-recall", "false",
        "--warm-up", "false",
        "--show-progress", str(args.show_progress).lower(),
        "--use-ground-truth-cache", "true",
        "--storage-amplification", str(args.memory_ratio),
        "--indexing-threshold", str(args.veda_indexing_threshold),
        "--ef-search", str(args.veda_plan_ef),
        "--table-prefix", table_prefix,
        "--versioned-plan", "true",
        "--result-tag", f"versioned_{table_prefix}",
    ]
    if args.document_limit is not None:
        cmd.extend(["--document-limit", str(args.document_limit)])
    _run(cmd, cwd=PROJECT_ROOT / "basic_benchmark")
    plan_id = _latest_veda_plan_id(method)
    metadata_tables = _copy_metadata_tables(manifest, plan_id, kind="veda")
    return plan_id, _veda_relations(plan_id, metadata_tables), {
        "veda_plan_ef": int(args.veda_plan_ef),
        "metadata_tables": metadata_tables,
    }


def _build_honeybee(args: argparse.Namespace, table_prefix: str) -> tuple[int, dict[str, list[str]], dict]:
    cmd = [
        _python(),
        str(PROJECT_ROOT / "controller" / "dynamic_partition" / "hnsw" / "AnonySys_dynamic_partition.py"),
        "--storage", str(args.memory_ratio),
        "--recall", str(args.honeybee_recall),
        "--table-prefix", table_prefix,
    ]
    _run(cmd, cwd=PROJECT_ROOT)
    relations = _honeybee_relations(table_prefix)
    return 0, relations, {
        "mapping_table": comb_role_mapping_table_name(table_prefix),
        "honeybee_recall": float(args.honeybee_recall),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build and register one versioned materialized plan")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--memory-ratio", type=float, required=True)
    parser.add_argument("--version", default=None)
    parser.add_argument("--query-num", type=int, default=1)
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ef-search", type=int, default=100)
    parser.add_argument("--ours-index-type", choices=["squidhnsw", "hnsw", "ivfflat"], default="squidhnsw")
    parser.add_argument("--enable-split", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--private-edge-top-d", type=int, default=32)
    parser.add_argument("--veda-index-type", choices=["hnsw", "vedahnsw", "ivfflat"], default="vedahnsw")
    parser.add_argument("--veda-indexing-threshold", type=int, default=2900)
    parser.add_argument("--veda-plan-ef", type=int, default=100)
    parser.add_argument("--honeybee-recall", type=float, default=0.99)
    args = parser.parse_args()

    version = args.version or f"mem_{str(float(args.memory_ratio)).replace('.', 'p')}"
    manifest = create_manifest(args.method, args.memory_ratio, version).as_dict()
    table_prefix = str(manifest["table_prefix"])

    _registry_init()
    _cleanup_existing_version(manifest)
    if args.method == "ours":
        plan_id, relations, metadata = _build_ours(args, manifest)
    elif args.method in {"veda", "effveda"}:
        plan_id, relations, metadata = _build_veda(args, args.method, manifest)
    elif args.method == "honeybee":
        plan_id, relations, metadata = _build_honeybee(args, table_prefix)
    else:
        raise SystemExit(f"Unsupported method: {args.method}")

    _grant_select_public(relations)
    metadata = {
        **metadata,
        "version": manifest["version"],
        "table_prefix": table_prefix,
        "memory_ratio": float(args.memory_ratio),
    }
    registry_id = _register_plan(manifest, plan_id, metadata)
    _register_relations(registry_id, relations)
    print(json.dumps({
        "registry_id": registry_id,
        "method": args.method,
        "memory_ratio": float(args.memory_ratio),
        "plan_id": int(plan_id),
        "table_prefix": table_prefix,
        "relations": {key: len(value) for key, value in relations.items()},
    }, sort_keys=True))


if __name__ == "__main__":
    main()
