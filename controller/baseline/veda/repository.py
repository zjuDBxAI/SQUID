from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from psycopg2 import sql

from services.config import get_db_connection

from .common import VedaPattern, normalize_int_tuple


def _default_db_connection_factory():
    return get_db_connection()


class VedaRepository:
    def __init__(self, *, db_connection_factory=_default_db_connection_factory) -> None:
        self.db_connection_factory = db_connection_factory

    def fetch_exclusive_patterns(self, *, document_limit: Optional[int] = None) -> list[VedaPattern]:
        query = sql.SQL(
            """
            WITH document_roles AS (
                SELECT
                    pa.document_id,
                    array_agg(DISTINCT pa.role_id ORDER BY pa.role_id)::BIGINT[] AS role_ids
                FROM PermissionAssignment pa
                GROUP BY pa.document_id
            ),
            document_block_counts AS (
                SELECT document_id, COUNT(*)::BIGINT AS vector_count
                FROM documentblocks
                GROUP BY document_id
            )
            SELECT
                dr.document_id,
                dr.role_ids,
                COALESCE(dbc.vector_count, 0)::BIGINT AS vector_count
            FROM document_roles dr
            JOIN document_block_counts dbc ON dbc.document_id = dr.document_id
            ORDER BY dr.document_id
            """
        )
        params: list[object] = []
        if document_limit is not None:
            query += sql.SQL(" LIMIT %s")
            params.append(int(document_limit))

        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(query, params)
                rows = cur.fetchall()
        finally:
            conn.close()

        grouped_documents: dict[tuple[int, ...], list[int]] = defaultdict(list)
        grouped_vector_counts: dict[tuple[int, ...], int] = defaultdict(int)
        for document_id, role_ids, vector_count in rows:
            normalized_roles = normalize_int_tuple(role_ids or ())
            if not normalized_roles:
                continue
            grouped_documents[normalized_roles].append(int(document_id))
            grouped_vector_counts[normalized_roles] += int(vector_count)

        patterns: list[VedaPattern] = []
        for pattern_id, role_ids in enumerate(sorted(grouped_documents), start=1):
            document_ids = tuple(sorted(int(document_id) for document_id in grouped_documents[role_ids]))
            patterns.append(
                VedaPattern(
                    pattern_id=int(pattern_id),
                    role_ids=role_ids,
                    document_ids=document_ids,
                    vector_count=int(grouped_vector_counts[role_ids]),
                )
            )
        return patterns

    def fetch_role_universe(self) -> tuple[int, ...]:
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT role_id FROM Roles ORDER BY role_id;")
                return tuple(int(row[0]) for row in cur.fetchall())
        finally:
            conn.close()

    def fetch_user_roles(self, user_ids: Iterable[int] | None = None) -> dict[int, tuple[int, ...]]:
        params: list[object] = []
        where_clause = ""
        if user_ids is not None:
            normalized = sorted({int(user_id) for user_id in user_ids})
            if not normalized:
                return {}
            where_clause = "WHERE user_id = ANY(%s::BIGINT[])"
            params.append(normalized)

        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    f"""
                    SELECT user_id, array_agg(DISTINCT role_id ORDER BY role_id)::BIGINT[] AS role_ids
                    FROM UserRoles
                    {where_clause}
                    GROUP BY user_id
                    ORDER BY user_id;
                    """,
                    params,
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return {
            int(user_id): normalize_int_tuple(role_ids or ())
            for user_id, role_ids in rows
        }

    def fetch_document_vector_count(self) -> int:
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documentblocks;")
                row = cur.fetchone()
        finally:
            conn.close()
        return int(row[0] if row else 0)
