import argparse
import hashlib
import json
import os
import re
import sys
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from datasets import load_dataset
from psycopg2 import Binary
from tqdm import tqdm

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from basic_benchmark.common_function import clear_ground_truth_cache
from controller.initialize_main_tables import initialize_database_deduplication
from controller.prepare_database import clear_db
from services.config import get_db_connection
from services.embedding_service import generate_embedding


DEFAULT_OUTPUT = Path(project_root) / "basic_benchmark" / "query_dataset_orgaccess_medium.json"
ORGACCESS_VECTOR_DIMENSION = 300


def _normalize_permission_cell(value) -> list[str]:
    if value is None:
        return ["orgaccess_public"]
    if isinstance(value, (list, tuple, set)):
        raw_parts = []
        for item in value:
            raw_parts.extend(_normalize_permission_cell(item))
        return raw_parts or ["orgaccess_public"]
    text = str(value).strip()
    if not text:
        return ["orgaccess_public"]
    parts = re.split(r"[,;|\n]+", text)
    normalized = []
    for part in parts:
        token = re.sub(r"\s+", "_", part.strip().lower())
        token = re.sub(r"[^a-z0-9_:+./-]+", "", token)
        if token:
            normalized.append(token)
    return normalized or ["orgaccess_public"]


def _text_or_empty(value) -> str:
    return "" if value is None else str(value).strip()


def _document_text(row: dict) -> str:
    parts = [
        f"Role: {_text_or_empty(row.get('user_role'))}",
        f"Permissions: {_text_or_empty(row.get('permissions'))}",
        f"Query: {_text_or_empty(row.get('query'))}",
        f"Expected response: {_text_or_empty(row.get('expected_response'))}",
        f"Rationale: {_text_or_empty(row.get('rationale'))}",
    ]
    return "\n".join(part for part in parts if part.split(": ", 1)[-1])


def _query_text(row: dict) -> str:
    query = _text_or_empty(row.get("query"))
    if query:
        return query
    return _document_text(row)


def _unique_user_key(row: dict, permissions: list[str]) -> tuple[str, tuple[str, ...]]:
    role = _text_or_empty(row.get("user_role")).lower() or "orgaccess_user"
    return role, tuple(sorted(set(permissions)))


def _load_medium_rows(limit: int | None, start_row: int) -> list[dict]:
    dataset = load_dataset("respai-lab/orgaccess", split="medium")
    end = len(dataset) if limit is None or limit <= 0 else min(len(dataset), start_row + int(limit))
    if start_row >= end:
        return []
    return [dict(row) for row in dataset.select(range(start_row, end))]


def _insert_orgaccess(rows: list[dict], *, topk: int, output_file: Path) -> None:
    permission_to_role_id: OrderedDict[str, int] = OrderedDict()
    user_key_to_id: OrderedDict[tuple[str, tuple[str, ...]], int] = OrderedDict()
    documents = []
    document_blocks = []
    permission_assignments: set[tuple[int, int]] = set()
    user_roles: set[tuple[int, int]] = set()
    queries = []
    created_at = datetime.now()

    for offset, row in enumerate(tqdm(rows, desc="Embedding orgaccess medium", unit="row")):
        document_id = offset + 1
        block_id = offset + 1
        permissions = sorted(set(_normalize_permission_cell(row.get("permissions"))))
        user_key = _unique_user_key(row, permissions)
        user_id = user_key_to_id.setdefault(user_key, len(user_key_to_id) + 1)

        role_ids = []
        for permission in permissions:
            role_id = permission_to_role_id.setdefault(permission, len(permission_to_role_id) + 1)
            role_ids.append(role_id)
            permission_assignments.add((role_id, document_id))
            user_roles.add((user_id, role_id))

        doc_text = _document_text(row)
        query_text = _query_text(row)
        doc_vector = generate_embedding(doc_text)
        query_vector = generate_embedding(query_text)
        hash_value = Binary(hashlib.sha1(doc_text.encode("utf-8")).digest())

        documents.append((document_id, f"orgaccess_medium_{document_id}", created_at, created_at))
        document_blocks.append((block_id, document_id, Binary(doc_text.encode("utf-8")), hash_value, doc_vector))
        queries.append(
            {
                "query_id": offset + 1,
                "user_id": user_id,
                "query_vector": query_vector,
                "topk": int(topk),
                "query_block_selectivity": 0.0,
                "source_dataset": "respai-lab/orgaccess",
                "source_split": "medium",
                "source_row": offset,
                "user_role": _text_or_empty(row.get("user_role")),
                "permissions": permissions,
            }
        )

    users = [
        (user_id, f"orgaccess_user_{user_id}")
        for user_id in user_key_to_id.values()
    ]
    roles = [
        (role_id, permission)
        for permission, role_id in permission_to_role_id.items()
    ]

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.executemany(
            "INSERT INTO Users (user_id, user_name) VALUES (%s, %s) ON CONFLICT (user_id) DO NOTHING",
            users,
        )
        cur.executemany(
            "INSERT INTO Roles (role_id, role_name) VALUES (%s, %s) ON CONFLICT (role_id) DO NOTHING",
            roles,
        )
        cur.executemany(
            """
            INSERT INTO Documents (document_id, document_name, created_at, updated_at)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (document_id) DO NOTHING
            """,
            documents,
        )
        cur.executemany(
            """
            INSERT INTO documentblocks (block_id, document_id, block_content, hash_value, vector)
            VALUES (%s, %s, %s, %s, %s)
            """,
            document_blocks,
        )
        cur.executemany(
            "INSERT INTO UserRoles (user_id, role_id) VALUES (%s, %s) ON CONFLICT DO NOTHING",
            sorted(user_roles),
        )
        cur.executemany(
            "INSERT INTO PermissionAssignment (role_id, document_id) VALUES (%s, %s)",
            sorted(permission_assignments),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()

    selectivity_by_user = _load_selectivity_by_user()
    for query in queries:
        query["query_block_selectivity"] = selectivity_by_user.get(int(query["user_id"]), 0.0)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as handle:
        json.dump(queries, handle, indent=2)

    print(
        "Loaded orgaccess medium: "
        f"{len(documents)} documents, {len(document_blocks)} vectors, "
        f"{len(users)} users, {len(roles)} roles, "
        f"{len(permission_assignments)} permission assignments."
    )
    print(f"Wrote query dataset to {output_file}")


def _load_selectivity_by_user() -> dict[int, float]:
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT COUNT(*) FROM documentblocks;")
        total_blocks = int(cur.fetchone()[0] or 0)
        if total_blocks <= 0:
            return {}
        cur.execute(
            """
            SELECT ur.user_id, COUNT(DISTINCT db.block_id)
            FROM UserRoles ur
            JOIN PermissionAssignment pa ON pa.role_id = ur.role_id
            JOIN documentblocks db ON db.document_id = pa.document_id
            GROUP BY ur.user_id
            """
        )
        return {int(user_id): float(count) / float(total_blocks) for user_id, count in cur.fetchall()}
    finally:
        cur.close()
        conn.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare respai-lab/orgaccess medium as a Multitenanthakes benchmark.")
    parser.add_argument("--limit", type=int, default=0, help="Rows to load from the medium split; 0 loads all rows.")
    parser.add_argument("--start-row", type=int, default=0, help="Starting row in the medium split.")
    parser.add_argument("--topk", type=int, default=10, help="Top-k stored in generated query JSON.")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Output query JSON path.")
    parser.add_argument("--clear", type=lambda value: str(value).lower() in {"1", "true", "yes", "y", "on"}, default=True)
    args = parser.parse_args()

    if args.clear:
        clear_db()
        initialize_database_deduplication(enable_index=False, vector_dimension=ORGACCESS_VECTOR_DIMENSION)

    rows = _load_medium_rows(limit=args.limit, start_row=max(0, int(args.start_row)))
    if not rows:
        raise RuntimeError("No orgaccess medium rows selected.")

    _insert_orgaccess(rows, topk=max(1, int(args.topk)), output_file=Path(args.output).resolve())
    clear_ground_truth_cache()


if __name__ == "__main__":
    main()
