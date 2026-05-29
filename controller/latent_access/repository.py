"""Data access helpers for latent access partitioning.

This module extracts document-level semantic representatives together with their
tenant visibility sets from the existing RBAC benchmark schema.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import numpy as np

from services.config import get_db_connection


def _default_db_connection_factory():
    return get_db_connection()


def _parse_vector(raw_value) -> np.ndarray:
    if raw_value is None:
        raise ValueError("Vector value is missing")
    if isinstance(raw_value, np.ndarray):
        vector = raw_value.astype(np.float32, copy=False)
    elif isinstance(raw_value, memoryview):
        vector = np.frombuffer(raw_value, dtype=np.float32)
    elif isinstance(raw_value, (bytes, bytearray)):
        vector = np.frombuffer(raw_value, dtype=np.float32)
    elif isinstance(raw_value, str):
        payload = raw_value.strip().strip("[]")
        if not payload:
            return np.zeros(0, dtype=np.float32)
        vector = np.asarray([float(item) for item in payload.split(",") if item], dtype=np.float32)
    elif hasattr(raw_value, "tolist"):
        vector = np.asarray(raw_value.tolist(), dtype=np.float32)
    else:
        vector = np.asarray(raw_value, dtype=np.float32)

    if vector.ndim != 1:
        vector = vector.ravel()
    return vector.astype(np.float32, copy=False)


@dataclass(slots=True)
class DocumentAccessRecord:
    document_id: int
    representative_block_id: int
    vector: np.ndarray
    tenant_ids: tuple[int, ...]


class LatentAccessRepository:
    """Read document-level semantic and access snapshots from PostgreSQL."""

    def __init__(self, *, db_connection_factory=_default_db_connection_factory) -> None:
        self.db_connection_factory = db_connection_factory

    def fetch_document_access_records(
        self,
        *,
        limit: Optional[int] = None,
    ) -> list[DocumentAccessRecord]:
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                query = """
                    WITH document_tenants AS (
                        SELECT
                            pa.document_id,
                            array_agg(DISTINCT ur.user_id ORDER BY ur.user_id) AS tenant_ids
                        FROM PermissionAssignment pa
                        JOIN UserRoles ur ON ur.role_id = pa.role_id
                        GROUP BY pa.document_id
                    ),
                    representative_blocks AS (
                        SELECT DISTINCT ON (db.document_id)
                            db.document_id,
                            db.block_id,
                            db.vector
                        FROM documentblocks db
                        WHERE db.vector IS NOT NULL
                        ORDER BY db.document_id, db.block_id
                    )
                    SELECT
                        rb.document_id,
                        rb.block_id,
                        rb.vector,
                        dt.tenant_ids
                    FROM representative_blocks rb
                    JOIN document_tenants dt ON dt.document_id = rb.document_id
                    ORDER BY rb.document_id
                """
                params = []
                if limit is not None:
                    query += " LIMIT %s"
                    params.append(int(limit))
                cur.execute(query, params)
                rows = cur.fetchall()
        finally:
            conn.close()

        records: list[DocumentAccessRecord] = []
        for document_id, block_id, vector_value, tenant_ids in rows:
            normalized_tenants = tuple(int(tenant_id) for tenant_id in (tenant_ids or ()))
            if not normalized_tenants:
                continue
            records.append(
                DocumentAccessRecord(
                    document_id=int(document_id),
                    representative_block_id=int(block_id),
                    vector=_parse_vector(vector_value),
                    tenant_ids=normalized_tenants,
                )
            )
        return records

    def fetch_document_block_counts(
        self,
        document_ids: Optional[Iterable[int]] = None,
    ) -> dict[int, int]:
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                if document_ids is None:
                    cur.execute(
                        """
                        SELECT document_id, COUNT(*)
                        FROM documentblocks
                        GROUP BY document_id;
                        """
                    )
                else:
                    normalized = sorted({int(document_id) for document_id in document_ids})
                    if not normalized:
                        return {}
                    cur.execute(
                        """
                        SELECT document_id, COUNT(*)
                        FROM documentblocks
                        WHERE document_id = ANY(%s)
                        GROUP BY document_id;
                        """,
                        [normalized],
                    )
                return {int(document_id): int(count) for document_id, count in cur.fetchall()}
        finally:
            conn.close()

    def fetch_block_vectors_for_documents(
        self,
        document_ids: Iterable[int],
    ) -> list[np.ndarray]:
        normalized = sorted({int(document_id) for document_id in document_ids})
        if not normalized:
            return []

        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT vector
                    FROM documentblocks
                    WHERE document_id = ANY(%s)
                      AND vector IS NOT NULL
                    ORDER BY document_id, block_id;
                    """,
                    [normalized],
                )
                return [_parse_vector(vector_value) for (vector_value,) in cur.fetchall()]
        finally:
            conn.close()

    def fetch_accessible_document_ids(self, user_id: int) -> set[int]:
        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT DISTINCT pa.document_id
                    FROM PermissionAssignment pa
                    JOIN UserRoles ur ON ur.role_id = pa.role_id
                    WHERE ur.user_id = %s;
                    """,
                    [int(user_id)],
                )
                return {int(document_id) for (document_id,) in cur.fetchall()}
        finally:
            conn.close()
