from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

from psycopg2 import sql

from services.config import get_db_connection

from .common import DocumentRoleRecord, normalize_int_tuple


def _default_db_connection_factory():
    return get_db_connection()


@dataclass(slots=True)
class SieveRepositoryConfig:
    document_table: str = "documentblocks"
    userroles_table: str = "userroles"
    permission_table: str = "permissionassignment"


class SieveRepository:
    def __init__(self, *, db_connection_factory=_default_db_connection_factory, config: SieveRepositoryConfig | None = None) -> None:
        self.db_connection_factory = db_connection_factory
        self.config = config or SieveRepositoryConfig()

    def fetch_document_acl_rows(self, *, document_limit: Optional[int] = None) -> list[DocumentRoleRecord]:
        """Backward-compatible alias for the role-based document mapping."""
        return self.fetch_document_role_rows(document_limit=document_limit)

    def fetch_document_role_rows(self, *, document_limit: Optional[int] = None) -> list[DocumentRoleRecord]:
        query = sql.SQL(
            """
            WITH document_role_sets AS (
                SELECT
                    pa.document_id,
                    array_agg(DISTINCT pa.role_id ORDER BY pa.role_id) AS role_ids
                FROM {} AS pa
                GROUP BY pa.document_id
            ),
            document_block_counts AS (
                SELECT db.document_id, COUNT(*)::BIGINT AS block_count
                FROM {} AS db
                GROUP BY db.document_id
            )
            SELECT
                drs.document_id,
                drs.role_ids,
                COALESCE(dbc.block_count, 0)::BIGINT AS block_count
            FROM document_role_sets drs
            LEFT JOIN document_block_counts dbc
              ON dbc.document_id = drs.document_id
            ORDER BY drs.document_id
            """
        ).format(
            sql.Identifier(self.config.permission_table),
            sql.Identifier(self.config.document_table),
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

        records: list[DocumentRoleRecord] = []
        for document_id, role_ids, block_count in rows:
            normalized_roles = normalize_int_tuple(role_ids or ())
            records.append(
                DocumentRoleRecord(
                    document_id=int(document_id),
                    role_ids=normalized_roles,
                    block_count=int(block_count),
                )
            )
        return records

    def fetch_document_roles(self, *, document_limit: Optional[int] = None) -> list[DocumentRoleRecord]:
        """Backward-compatible alias for the role-based SIEVE variant."""
        return self.fetch_document_role_rows(document_limit=document_limit)


    def fetch_user_roles_for_users(self, user_ids: Iterable[int]) -> dict[int, tuple[int, ...]]:
        normalized_user_ids = sorted({int(user_id) for user_id in user_ids})
        if not normalized_user_ids:
            return {}

        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, array_agg(DISTINCT role_id ORDER BY role_id) AS role_ids
                    FROM userroles
                    WHERE user_id = ANY(%s::INT[])
                    GROUP BY user_id
                    ORDER BY user_id;
                    """,
                    [normalized_user_ids],
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        return {
            int(user_id): normalize_int_tuple(role_ids or ())
            for user_id, role_ids in rows
        }

    def fetch_user_roles(self) -> dict[int, tuple[int, ...]]:
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT user_id, array_agg(DISTINCT role_id ORDER BY role_id) AS role_ids
                    FROM userroles
                    GROUP BY user_id
                    ORDER BY user_id;
                    """
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        return {
            int(user_id): normalize_int_tuple(role_ids or ())
            for user_id, role_ids in rows
        }

    def fetch_role_universe(self) -> tuple[int, ...]:
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT role_id FROM roles ORDER BY role_id;")
                return tuple(int(row[0]) for row in cur.fetchall())
        finally:
            conn.close()

    def fetch_document_block_counts(self) -> dict[int, int]:
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT document_id, COUNT(*)::BIGINT
                    FROM documentblocks
                    GROUP BY document_id;
                    """
                )
                rows = cur.fetchall()
        finally:
            conn.close()
        return {int(document_id): int(block_count) for document_id, block_count in rows}

    def fetch_document_vector_count(self) -> int:
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documentblocks;")
                row = cur.fetchone()
        finally:
            conn.close()
        return int(row[0] if row else 0)

    def fetch_total_document_count(self) -> int:
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute("SELECT COUNT(*) FROM documents;")
                row = cur.fetchone()
        finally:
            conn.close()
        return int(row[0] if row else 0)

