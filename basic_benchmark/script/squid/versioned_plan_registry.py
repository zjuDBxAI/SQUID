#!/usr/bin/env python3
"""Manage metadata for versioned benchmark materializations.

This tool never builds, drops, or alters vector/partition tables. Builders own
those operations after they are made namespace-aware. The registry maps a
method and storage amplification to one immutable materialization version.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from plan_manifest import METHODS, create_manifest
from services.config import get_db_connection  # noqa: E402


STATES = ("building", "ready", "failed", "retired")


def schema_path() -> Path:
    return SCRIPT_DIR / "schema.sql"


def initialize_registry() -> None:
    statements = schema_path().read_text()
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(statements)
        conn.commit()
    finally:
        conn.close()


def registry_row(method: str, memory_ratio: float) -> dict | None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT registry_id, method, memory_ratio, plan_id, table_prefix,
                       state, measured_space_bytes, metadata, created_at, updated_at
                FROM benchmark_plan_registry
                WHERE method = %s AND memory_ratio = %s
                """,
                [method, memory_ratio],
            )
            row = cur.fetchone()
    finally:
        conn.close()
    if row is None:
        return None
    keys = (
        "registry_id", "method", "memory_ratio", "plan_id", "table_prefix",
        "state", "measured_space_bytes", "metadata", "created_at", "updated_at",
    )
    return {key: value for key, value in zip(keys, row)}


def register(args: argparse.Namespace) -> None:
    metadata = json.loads(args.metadata)
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO benchmark_plan_registry (
                    method, memory_ratio, plan_id, table_prefix, state,
                    measured_space_bytes, metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb)
                ON CONFLICT (method, memory_ratio) DO UPDATE
                SET plan_id = EXCLUDED.plan_id,
                    table_prefix = EXCLUDED.table_prefix,
                    state = EXCLUDED.state,
                    measured_space_bytes = EXCLUDED.measured_space_bytes,
                    metadata = EXCLUDED.metadata,
                    updated_at = now()
                RETURNING registry_id
                """,
                [
                    args.method,
                    args.memory_ratio,
                    args.plan_id,
                    args.table_prefix,
                    args.state,
                    args.measured_space_bytes,
                    json.dumps(metadata),
                ],
            )
            registry_id = int(cur.fetchone()[0])
        conn.commit()
    finally:
        conn.close()
    print(json.dumps({"registry_id": registry_id, "method": args.method, "memory_ratio": args.memory_ratio}))


def list_plans(args: argparse.Namespace) -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            clauses, params = [], []
            if args.method:
                clauses.append("method = %s")
                params.append(args.method)
            if args.state:
                clauses.append("state = %s")
                params.append(args.state)
            where = " WHERE " + " AND ".join(clauses) if clauses else ""
            cur.execute(
                "SELECT registry_id, method, memory_ratio, plan_id, table_prefix, state, measured_space_bytes "
                "FROM benchmark_plan_registry" + where + " ORDER BY method, memory_ratio",
                params,
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    for row in rows:
        print(json.dumps({
            "registry_id": int(row[0]), "method": row[1], "memory_ratio": float(row[2]),
            "plan_id": int(row[3]), "table_prefix": row[4], "state": row[5],
            "measured_space_bytes": row[6],
        }))


def resolve(args: argparse.Namespace) -> None:
    row = registry_row(args.method, args.memory_ratio)
    if row is None or row["state"] != "ready":
        raise SystemExit(f"No ready plan for method={args.method} memory_ratio={args.memory_ratio}")
    row["memory_ratio"] = float(row["memory_ratio"])
    row["created_at"] = row["created_at"].isoformat()
    row["updated_at"] = row["updated_at"].isoformat()
    print(json.dumps(row, default=str))


def register_relations(args: argparse.Namespace) -> None:
    relations = [str(item) for item in args.relation]
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            for relation in relations:
                cur.execute(
                    """
                    INSERT INTO benchmark_plan_relations (registry_id, relation_name, relation_kind)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (registry_id, relation_name) DO UPDATE
                    SET relation_kind = EXCLUDED.relation_kind
                    """,
                    [args.registry_id, relation, args.relation_kind],
                )
        conn.commit()
    finally:
        conn.close()
    print(json.dumps({"registry_id": args.registry_id, "registered_relations": len(relations)}))


def list_relations(args: argparse.Namespace) -> None:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT relation_name, relation_kind
                FROM benchmark_plan_relations
                WHERE registry_id = %s
                ORDER BY relation_kind, relation_name
                """,
                [args.registry_id],
            )
            rows = cur.fetchall()
    finally:
        conn.close()
    for relation_name, relation_kind in rows:
        print(json.dumps({"registry_id": args.registry_id, "relation_name": relation_name, "relation_kind": relation_kind}))


def manifest(args: argparse.Namespace) -> None:
    print(json.dumps(create_manifest(args.method, args.memory_ratio, args.version).as_dict(), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description="Registry for versioned benchmark materializations")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("init", help="Create registry tables from schema.sql")

    manifest_parser = subparsers.add_parser("manifest", help="Print stable names without connecting to PostgreSQL")
    manifest_parser.add_argument("--method", choices=METHODS, required=True)
    manifest_parser.add_argument("--memory-ratio", type=float, required=True)
    manifest_parser.add_argument("--version", required=True)

    register_parser = subparsers.add_parser("register", help="Insert or update one versioned plan")
    register_parser.add_argument("--method", choices=METHODS, required=True)
    register_parser.add_argument("--memory-ratio", type=float, required=True)
    register_parser.add_argument("--plan-id", type=int, required=True)
    register_parser.add_argument("--table-prefix", required=True)
    register_parser.add_argument("--state", choices=STATES, default="building")
    register_parser.add_argument("--measured-space-bytes", type=int, default=None)
    register_parser.add_argument("--metadata", default="{}", help="JSON metadata")

    list_parser = subparsers.add_parser("list", help="List registered plans")
    list_parser.add_argument("--method", choices=METHODS)
    list_parser.add_argument("--state", choices=STATES)

    resolve_parser = subparsers.add_parser("resolve", help="Resolve one ready plan")
    resolve_parser.add_argument("--method", choices=METHODS, required=True)
    resolve_parser.add_argument("--memory-ratio", type=float, required=True)

    relation_parser = subparsers.add_parser("register-relations", help="Register relations owned by a version")
    relation_parser.add_argument("--registry-id", type=int, required=True)
    relation_parser.add_argument("--relation-kind", choices=("partition", "index", "route_metadata", "pattern_metadata"), required=True)
    relation_parser.add_argument("--relation", action="append", required=True)

    list_relations_parser = subparsers.add_parser("relations", help="List relations owned by a version")
    list_relations_parser.add_argument("--registry-id", type=int, required=True)

    args = parser.parse_args()
    if args.command == "init":
        initialize_registry()
    elif args.command == "manifest":
        manifest(args)
    elif args.command == "register":
        register(args)
    elif args.command == "list":
        list_plans(args)
    elif args.command == "register-relations":
        register_relations(args)
    elif args.command == "relations":
        list_relations(args)
    else:
        resolve(args)


if __name__ == "__main__":
    main()
