from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional

from services.config import get_db_connection


def _default_db_connection_factory():
    return get_db_connection()


class KMeansRepository:
    def __init__(self, *, db_connection_factory=_default_db_connection_factory) -> None:
        self.db_connection_factory = db_connection_factory

    def fetch_acl_rows(self, *, document_limit: Optional[int] = None) -> list[tuple[int, tuple[int, ...], tuple[int, ...], int]]:
        query = """
            WITH document_tenants AS (
                SELECT
                    pa.document_id,
                    array_agg(DISTINCT ur.user_id ORDER BY ur.user_id) AS tenant_ids
                FROM PermissionAssignment pa
                JOIN UserRoles ur ON ur.role_id = pa.role_id
                GROUP BY pa.document_id
            ),
            document_block_counts AS (
                SELECT document_id, COUNT(*)::BIGINT AS vector_count
                FROM documentblocks
                GROUP BY document_id
            )
            SELECT
                dt.document_id,
                dt.tenant_ids,
                COALESCE(dbc.vector_count, 0)::BIGINT AS vector_count
            FROM document_tenants dt
            JOIN document_block_counts dbc ON dbc.document_id = dt.document_id
            ORDER BY dt.document_id
        """
        params: list[object] = []
        if document_limit is not None:
            query += " LIMIT %s"
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
        for document_id, tenant_ids, vector_count in rows:
            normalized_tenants = tuple(sorted(int(tenant_id) for tenant_id in (tenant_ids or ())))
            if not normalized_tenants:
                continue
            grouped_documents[normalized_tenants].append(int(document_id))
            grouped_vector_counts[normalized_tenants] += int(vector_count)

        result: list[tuple[int, tuple[int, ...], tuple[int, ...], int]] = []
        for pattern_id, tenant_ids in enumerate(sorted(grouped_documents), start=1):
            document_ids = tuple(sorted(int(document_id) for document_id in grouped_documents[tenant_ids]))
            result.append((int(pattern_id), tenant_ids, document_ids, int(grouped_vector_counts[tenant_ids])))
        return result

    def fetch_tenant_pattern_ids(self, tenant_ids: Iterable[int]) -> dict[int, tuple[int, ...]]:
        normalized_tenants = sorted({int(tenant_id) for tenant_id in tenant_ids})
        if not normalized_tenants:
            return {}

        conn = self.db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    WITH latest_plan AS (
                        SELECT plan_id
                        FROM kmeans_current_plan
                        ORDER BY plan_id DESC
                        LIMIT 1
                    )
                    SELECT requested.tenant_id, kp.pattern_id
                    FROM latest_plan lp
                    JOIN kmeans_current_patterns kp ON kp.plan_id = lp.plan_id
                    JOIN unnest(%s::BIGINT[]) AS requested(tenant_id)
                      ON kp.tenant_ids @> ARRAY[requested.tenant_id]::BIGINT[]
                    ORDER BY requested.tenant_id, kp.pattern_id;
                    """,
                    [normalized_tenants],
                )
                rows = cur.fetchall()
        finally:
            conn.close()

        patterns_by_tenant: dict[int, set[int]] = {tenant_id: set() for tenant_id in normalized_tenants}
        for tenant_id, pattern_id in rows:
            tenant_id = int(tenant_id)
            if tenant_id in patterns_by_tenant:
                patterns_by_tenant[tenant_id].add(int(pattern_id))
        return {
            int(tenant_id): tuple(sorted(int(pattern_id) for pattern_id in pattern_ids))
            for tenant_id, pattern_ids in patterns_by_tenant.items()
        }
