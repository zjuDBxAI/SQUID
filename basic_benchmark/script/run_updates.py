#!/usr/bin/env python3
"""Run incremental SQUID updates from a JSON or JSONL workload."""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROLE_INSERT_OPERATIONS = {"role_insertion", "role_insert"}
ROLE_DELETE_OPERATIONS = {"role_deletion", "role_delete"}


def operation_name(record: dict[str, Any]) -> str:
    return str(record.get("operation") or record.get("op") or record.get("type") or "upsert").strip().lower()


def load_workload(path: Path) -> list[dict[str, Any]]:
    text = path.expanduser().read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"Update workload is empty: {path}")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        payload = []
        for line_number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                payload.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSONL record at line {line_number}: {error.msg}") from error

    if isinstance(payload, dict):
        payload = payload.get("updates", [payload])
    if not isinstance(payload, list):
        raise ValueError("Update workload must be a JSON object, JSON array, or JSONL file")

    records: list[dict[str, Any]] = []
    for index, record in enumerate(payload, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"Update record {index} must be a JSON object")
        records.append(record)
    return records


def integer_list(record: dict[str, Any], field_name: str, *, required: bool) -> list[int]:
    values = record.get(field_name)
    if values is None:
        if required:
            raise ValueError(f"{operation_name(record)} requires {field_name}")
        return []
    if not isinstance(values, list):
        raise ValueError(f"{field_name} must be a JSON array")
    return [int(value) for value in values]


def json_default(value: object) -> object:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, set | tuple):
        return list(value)
    return str(value)


def result_payload(result: object) -> object:
    return asdict(result) if is_dataclass(result) else result


def update_options(args: argparse.Namespace) -> dict[str, object]:
    return {
        "tau_del": args.tau_del,
        "max_operations": args.max_operations,
        "max_new_pattern_partitions": args.max_new_pattern_partitions,
        "create_indexes": args.create_indexes,
        "index_type": args.index_type,
        "vector_index_min_vectors": args.vector_index_min_vectors,
    }


def run_document_batch(records: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    operations = [operation_name(record) for record in records]
    if args.dry_run:
        return {
            "kind": "document_batch",
            "dry_run": True,
            "record_count": len(records),
            "operations": operations,
        }

    from controller.kmeans import apply_kmeans_update_batch

    started_at = time.perf_counter()
    result = apply_kmeans_update_batch(
        records,
        enable_maintenance=args.enable_maintenance,
        **update_options(args),
    )
    return {
        "kind": "document_batch",
        "record_count": len(records),
        "operations": operations,
        "wall_seconds": time.perf_counter() - started_at,
        "result": result_payload(result),
    }


def run_role_operation(record: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    operation = operation_name(record)
    if "role_id" not in record:
        raise ValueError(f"{operation} requires role_id")

    role_id = int(record["role_id"])
    is_insert = operation in ROLE_INSERT_OPERATIONS
    document_ids = integer_list(record, "document_ids", required=is_insert)
    user_ids = integer_list(record, "user_ids", required=True)

    if args.dry_run:
        return {
            "kind": operation,
            "dry_run": True,
            "role_id": role_id,
            "document_count": len(document_ids),
            "user_count": len(user_ids),
        }

    from controller.kmeans import delete_role_incrementally, insert_role_incrementally

    started_at = time.perf_counter()
    if is_insert:
        result = insert_role_incrementally(
            role_id=role_id,
            document_ids=document_ids,
            user_ids=user_ids,
            **update_options(args),
        )
    else:
        result = delete_role_incrementally(
            role_id=role_id,
            document_ids=document_ids or None,
            user_ids=user_ids,
            **update_options(args),
        )
    return {
        "kind": operation,
        "role_id": role_id,
        "document_count": len(document_ids),
        "user_count": len(user_ids),
        "wall_seconds": time.perf_counter() - started_at,
        "result": result_payload(result),
    }


def default_output_path() -> Path:
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    return PROJECT_ROOT / "basic_benchmark" / "result" / "updates" / f"{timestamp}.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run incremental SQUID updates from a JSON or JSONL workload.")
    parser.add_argument("--workload", type=Path, required=True, help="JSON object, JSON array, or JSONL workload file.")
    parser.add_argument("--batch-size", type=int, default=1, help="Document updates applied per maintenance batch.")
    parser.add_argument("--index-type", choices=["hnsw", "squidhnsw", "ivfflat"], default="hnsw")
    parser.add_argument("--tau-del", type=float, default=0.2)
    parser.add_argument("--max-operations", type=int, default=8)
    parser.add_argument("--max-new-pattern-partitions", type=int, default=2)
    parser.add_argument("--vector-index-min-vectors", type=int, default=1)
    parser.add_argument("--enable-maintenance", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--create-indexes", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Validate and group the workload without changing the database.")
    parser.add_argument("--output", type=Path, default=None, help="Result JSON path. Defaults to basic_benchmark/result/updates/." )
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    records = load_workload(args.workload)
    output_path = args.output.expanduser() if args.output is not None else default_output_path()
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path

    started_at = time.perf_counter()
    results: list[dict[str, Any]] = []
    pending_documents: list[dict[str, Any]] = []

    def record_result(result: dict[str, Any]) -> None:
        results.append(result)
        print(json.dumps(result, default=json_default, sort_keys=True), flush=True)

    def record_failure(kind: str, error: Exception) -> None:
        failure = {"kind": kind, "error_type": type(error).__name__, "error": str(error)}
        record_result(failure)
        if not args.continue_on_error:
            raise error

    def flush_document_batch() -> None:
        if not pending_documents:
            return
        batch = list(pending_documents)
        pending_documents.clear()
        try:
            record_result(run_document_batch(batch, args))
        except Exception as error:
            record_failure("document_batch", error)

    for record in records:
        operation = operation_name(record)
        if operation in ROLE_INSERT_OPERATIONS | ROLE_DELETE_OPERATIONS:
            flush_document_batch()
            try:
                record_result(run_role_operation(record, args))
            except Exception as error:
                record_failure(operation, error)
            continue

        pending_documents.append(record)
        if len(pending_documents) >= args.batch_size:
            flush_document_batch()

    flush_document_batch()
    summary = {
        "workload": str(args.workload),
        "dry_run": bool(args.dry_run),
        "index_type": args.index_type,
        "batch_size": args.batch_size,
        "record_count": len(records),
        "result_count": len(results),
        "total_wall_seconds": time.perf_counter() - started_at,
        "results": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, default=json_default, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[run_updates] wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
