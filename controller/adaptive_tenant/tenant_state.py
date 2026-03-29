"""Online tenant state reader for adaptive tenant control.

This module provides a DB-backed interface for reading the current tenant
status needed by the adaptive control plane:

- windowed operation frequencies with EMA smoothing
- tenant recall targets
- tenant size metrics
- dedicated and shared storage estimates

The default implementation treats ``tenant_id`` as the current project's
``user_id`` so it can work immediately on top of VectorSearch-RBAC. The scope
adapter interface is intentionally kept small so later work can replace this
mapping with a true tenant abstraction without rewriting the planner.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
import json
from typing import Any, Iterable, Optional

DEFAULT_RECALL_TARGET = 0.95
DEFAULT_EMA_DECAY = 0.8
DEFAULT_ROW_SAMPLE_LIMIT = 2048
DEFAULT_HNSW_GRAPH_BYTES_PER_VECTOR = 64


def _default_db_connection_factory():
    from services.config import get_db_connection

    return get_db_connection()


def _get_vector_dimension(default: int = 128) -> int:
    from services.config import get_document_vector_dimension

    return get_document_vector_dimension(default=default)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _json_dumps(value: Optional[dict[str, Any]]) -> str:
    return json.dumps(value or {})


def _ema(values: Iterable[float], decay: float) -> float:
    iterator = iter(values)
    try:
        result = float(next(iterator))
    except StopIteration:
        return 0.0

    for value in iterator:
        result = decay * result + (1.0 - decay) * float(value)
    return result


@dataclass(slots=True)
class TenantWindowStat:
    window_id: int
    tenant_id: int
    window_start: datetime
    window_end: datetime
    window_seconds: float
    read_count: int
    write_count: int
    query_rate: float
    write_rate: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class TenantSizeMetrics:
    tenant_id: int
    role_count: int
    document_count: int
    vector_count: int
    average_row_bytes: float
    estimated_table_bytes: float
    estimated_hnsw_index_bytes: float
    estimated_total_bytes: float


@dataclass(slots=True)
class TenantGroupMetrics:
    tenant_ids: tuple[int, ...]
    role_count: int
    document_count: int
    vector_count: int
    average_row_bytes: float
    estimated_table_bytes: float
    estimated_hnsw_index_bytes: float
    estimated_total_bytes: float


@dataclass(slots=True)
class TenantStateSnapshot:
    tenant_id: int
    tenant_name: str
    recall_target: float
    query_rate_ema: float
    write_rate_ema: float
    last_window_end: Optional[datetime]
    windows: list[TenantWindowStat]
    size: TenantSizeMetrics
    metadata: dict[str, Any] = field(default_factory=dict)


class TenantScopeAdapter(ABC):
    """Project-facing hook for mapping adaptive tenants to project tables."""

    @abstractmethod
    def list_tenant_ids(self, conn) -> list[int]:
        """Return all currently visible tenant ids."""

    @abstractmethod
    def fetch_size_metrics(
        self,
        conn,
        tenant_id: int,
        average_row_bytes: float,
        vector_dimension: int,
        hnsw_graph_bytes_per_vector: int,
    ) -> TenantSizeMetrics:
        """Compute current tenant size and storage metrics."""

    @abstractmethod
    def fetch_accessible_document_ids(self, conn, tenant_id: int) -> set[int]:
        """Return the accessible document ids for one tenant."""

    @abstractmethod
    def fetch_group_metrics(
        self,
        conn,
        tenant_ids: Iterable[int],
        average_row_bytes: float,
        vector_dimension: int,
        hnsw_graph_bytes_per_vector: int,
    ) -> TenantGroupMetrics:
        """Compute size/storage metrics for a shared partition of multiple tenants."""

    def fetch_many_size_metrics(
        self,
        conn,
        tenant_ids: Iterable[int],
        average_row_bytes: float,
        vector_dimension: int,
        hnsw_graph_bytes_per_vector: int,
    ) -> dict[int, TenantSizeMetrics]:
        return {
            int(tenant_id): self.fetch_size_metrics(
                conn,
                int(tenant_id),
                average_row_bytes=average_row_bytes,
                vector_dimension=vector_dimension,
                hnsw_graph_bytes_per_vector=hnsw_graph_bytes_per_vector,
            )
            for tenant_id in tenant_ids
        }

    def fetch_many_accessible_document_ids(
        self,
        conn,
        tenant_ids: Iterable[int],
    ) -> dict[int, set[int]]:
        return {
            int(tenant_id): self.fetch_accessible_document_ids(conn, int(tenant_id))
            for tenant_id in tenant_ids
        }

    def fetch_document_block_counts(
        self,
        conn,
        document_ids: Optional[Iterable[int]] = None,
    ) -> dict[int, int]:
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
                normalized_ids = sorted({int(document_id) for document_id in document_ids})
                if not normalized_ids:
                    return {}
                cur.execute(
                    """
                    SELECT document_id, COUNT(*)
                    FROM documentblocks
                    WHERE document_id = ANY(%s)
                    GROUP BY document_id;
                    """,
                    [normalized_ids],
                )
            return {int(row[0]): int(row[1]) for row in cur.fetchall()}


class UserTenantScopeAdapter(TenantScopeAdapter):
    """Default adapter: treat project ``user_id`` as ``tenant_id``."""

    def list_tenant_ids(self, conn) -> list[int]:
        with conn.cursor() as cur:
            cur.execute("SELECT DISTINCT user_id FROM UserRoles ORDER BY user_id;")
            return [int(row[0]) for row in cur.fetchall()]

    def fetch_size_metrics(
        self,
        conn,
        tenant_id: int,
        average_row_bytes: float,
        vector_dimension: int,
        hnsw_graph_bytes_per_vector: int,
    ) -> TenantSizeMetrics:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(DISTINCT ur.role_id) AS role_count,
                    COUNT(DISTINCT pa.document_id) AS document_count,
                    COUNT(DISTINCT (db.document_id, db.block_id)) AS vector_count
                FROM UserRoles ur
                JOIN PermissionAssignment pa ON pa.role_id = ur.role_id
                JOIN documentblocks db ON db.document_id = pa.document_id
                WHERE ur.user_id = %s;
                """,
                [tenant_id],
            )
            row = cur.fetchone()

        role_count = int(row[0] or 0)
        document_count = int(row[1] or 0)
        vector_count = int(row[2] or 0)

        estimated_table_bytes = vector_count * average_row_bytes
        estimated_hnsw_index_bytes = vector_count * (
            vector_dimension * 4 + hnsw_graph_bytes_per_vector
        )
        estimated_total_bytes = estimated_table_bytes + estimated_hnsw_index_bytes

        return TenantSizeMetrics(
            tenant_id=tenant_id,
            role_count=role_count,
            document_count=document_count,
            vector_count=vector_count,
            average_row_bytes=average_row_bytes,
            estimated_table_bytes=estimated_table_bytes,
            estimated_hnsw_index_bytes=estimated_hnsw_index_bytes,
            estimated_total_bytes=estimated_total_bytes,
        )

    def fetch_accessible_document_ids(self, conn, tenant_id: int) -> set[int]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT DISTINCT pa.document_id
                FROM UserRoles ur
                JOIN PermissionAssignment pa ON pa.role_id = ur.role_id
                WHERE ur.user_id = %s
                ORDER BY pa.document_id;
                """,
                [tenant_id],
            )
            return {int(row[0]) for row in cur.fetchall()}

    def fetch_group_metrics(
        self,
        conn,
        tenant_ids: Iterable[int],
        average_row_bytes: float,
        vector_dimension: int,
        hnsw_graph_bytes_per_vector: int,
    ) -> TenantGroupMetrics:
        sorted_ids = tuple(sorted({int(tenant_id) for tenant_id in tenant_ids}))
        if not sorted_ids:
            return TenantGroupMetrics(
                tenant_ids=(),
                role_count=0,
                document_count=0,
                vector_count=0,
                average_row_bytes=average_row_bytes,
                estimated_table_bytes=0.0,
                estimated_hnsw_index_bytes=0.0,
                estimated_total_bytes=0.0,
            )

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    COUNT(DISTINCT ur.role_id) AS role_count,
                    COUNT(DISTINCT pa.document_id) AS document_count,
                    COUNT(DISTINCT (db.document_id, db.block_id)) AS vector_count
                FROM UserRoles ur
                JOIN PermissionAssignment pa ON pa.role_id = ur.role_id
                JOIN documentblocks db ON db.document_id = pa.document_id
                WHERE ur.user_id = ANY(%s);
                """,
                [list(sorted_ids)],
            )
            row = cur.fetchone()

        role_count = int(row[0] or 0)
        document_count = int(row[1] or 0)
        vector_count = int(row[2] or 0)
        estimated_table_bytes = vector_count * average_row_bytes
        estimated_hnsw_index_bytes = vector_count * (
            vector_dimension * 4 + hnsw_graph_bytes_per_vector
        )
        estimated_total_bytes = estimated_table_bytes + estimated_hnsw_index_bytes

        return TenantGroupMetrics(
            tenant_ids=sorted_ids,
            role_count=role_count,
            document_count=document_count,
            vector_count=vector_count,
            average_row_bytes=average_row_bytes,
            estimated_table_bytes=estimated_table_bytes,
            estimated_hnsw_index_bytes=estimated_hnsw_index_bytes,
            estimated_total_bytes=estimated_total_bytes,
        )

    def fetch_many_size_metrics(
        self,
        conn,
        tenant_ids: Iterable[int],
        average_row_bytes: float,
        vector_dimension: int,
        hnsw_graph_bytes_per_vector: int,
    ) -> dict[int, TenantSizeMetrics]:
        normalized_ids = sorted({int(tenant_id) for tenant_id in tenant_ids})
        if not normalized_ids:
            return {}

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    ur.user_id AS tenant_id,
                    COUNT(DISTINCT ur.role_id) AS role_count,
                    COUNT(DISTINCT pa.document_id) AS document_count,
                    COUNT(DISTINCT (db.document_id, db.block_id)) AS vector_count
                FROM UserRoles ur
                LEFT JOIN PermissionAssignment pa ON pa.role_id = ur.role_id
                LEFT JOIN documentblocks db ON db.document_id = pa.document_id
                WHERE ur.user_id = ANY(%s)
                GROUP BY ur.user_id
                ORDER BY ur.user_id;
                """,
                [normalized_ids],
            )
            rows = cur.fetchall()

        result: dict[int, TenantSizeMetrics] = {}
        for row in rows:
            tenant_id = int(row[0])
            role_count = int(row[1] or 0)
            document_count = int(row[2] or 0)
            vector_count = int(row[3] or 0)
            estimated_table_bytes = vector_count * average_row_bytes
            estimated_hnsw_index_bytes = vector_count * (
                vector_dimension * 4 + hnsw_graph_bytes_per_vector
            )
            estimated_total_bytes = estimated_table_bytes + estimated_hnsw_index_bytes
            result[tenant_id] = TenantSizeMetrics(
                tenant_id=tenant_id,
                role_count=role_count,
                document_count=document_count,
                vector_count=vector_count,
                average_row_bytes=average_row_bytes,
                estimated_table_bytes=estimated_table_bytes,
                estimated_hnsw_index_bytes=estimated_hnsw_index_bytes,
                estimated_total_bytes=estimated_total_bytes,
            )
        return result

    def fetch_many_accessible_document_ids(
        self,
        conn,
        tenant_ids: Iterable[int],
    ) -> dict[int, set[int]]:
        normalized_ids = sorted({int(tenant_id) for tenant_id in tenant_ids})
        result = {tenant_id: set() for tenant_id in normalized_ids}
        if not normalized_ids:
            return result

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT ur.user_id, pa.document_id
                FROM UserRoles ur
                JOIN PermissionAssignment pa ON pa.role_id = ur.role_id
                WHERE ur.user_id = ANY(%s)
                GROUP BY ur.user_id, pa.document_id
                ORDER BY ur.user_id, pa.document_id;
                """,
                [normalized_ids],
            )
            for tenant_id, document_id in cur.fetchall():
                result[int(tenant_id)].add(int(document_id))
        return result


class TenantStateRepository:
    """DB-backed online interface for tenant runtime state."""

    def __init__(
        self,
        *,
        db_connection_factory=_default_db_connection_factory,
        scope_adapter: Optional[TenantScopeAdapter] = None,
        default_recall_target: float = DEFAULT_RECALL_TARGET,
        ema_decay: float = DEFAULT_EMA_DECAY,
        row_sample_limit: int = DEFAULT_ROW_SAMPLE_LIMIT,
        hnsw_graph_bytes_per_vector: int = DEFAULT_HNSW_GRAPH_BYTES_PER_VECTOR,
    ) -> None:
        self._db_connection_factory = db_connection_factory
        self._scope_adapter = scope_adapter or UserTenantScopeAdapter()
        self._default_recall_target = float(default_recall_target)
        self._ema_decay = float(ema_decay)
        self._row_sample_limit = int(row_sample_limit)
        self._hnsw_graph_bytes_per_vector = int(hnsw_graph_bytes_per_vector)

    def initialize_schema(self) -> None:
        conn = self._db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS adaptive_tenant_profiles (
                        tenant_id BIGINT PRIMARY KEY,
                        tenant_name TEXT NOT NULL,
                        recall_target DOUBLE PRECISION NOT NULL DEFAULT 0.95,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
                        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE TABLE IF NOT EXISTS adaptive_tenant_windows (
                        window_id BIGSERIAL PRIMARY KEY,
                        tenant_id BIGINT NOT NULL,
                        window_start TIMESTAMPTZ NOT NULL,
                        window_end TIMESTAMPTZ NOT NULL,
                        window_seconds DOUBLE PRECISION NOT NULL CHECK (window_seconds > 0),
                        read_count BIGINT NOT NULL DEFAULT 0,
                        write_count BIGINT NOT NULL DEFAULT 0,
                        metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
                        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
                    );
                    """
                )
                cur.execute(
                    """
                    CREATE INDEX IF NOT EXISTS idx_adaptive_tenant_windows_tenant_end
                    ON adaptive_tenant_windows (tenant_id, window_end DESC);
                    """
                )
            conn.commit()
        finally:
            conn.close()

    def ensure_tenant_profile(
        self,
        tenant_id: int,
        *,
        tenant_name: Optional[str] = None,
        recall_target: Optional[float] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        conn = self._db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO adaptive_tenant_profiles (
                        tenant_id,
                        tenant_name,
                        recall_target,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s::jsonb)
                    ON CONFLICT (tenant_id) DO UPDATE
                    SET tenant_name = EXCLUDED.tenant_name,
                        recall_target = EXCLUDED.recall_target,
                        metadata = adaptive_tenant_profiles.metadata || EXCLUDED.metadata,
                        updated_at = NOW();
                    """,
                    [
                        tenant_id,
                        tenant_name or f"tenant_{tenant_id}",
                        float(recall_target if recall_target is not None else self._default_recall_target),
                        _json_dumps(metadata),
                    ],
                )
            conn.commit()
        finally:
            conn.close()

    def set_recall_target(
        self,
        tenant_id: int,
        recall_target: float,
        *,
        tenant_name: Optional[str] = None,
    ) -> None:
        self.ensure_tenant_profile(
            tenant_id,
            tenant_name=tenant_name,
            recall_target=recall_target,
        )

    def record_window(
        self,
        tenant_id: int,
        *,
        read_count: int,
        write_count: int = 0,
        window_seconds: float = 60.0,
        window_end: Optional[datetime] = None,
        window_start: Optional[datetime] = None,
        metadata: Optional[dict[str, Any]] = None,
        recall_target: Optional[float] = None,
        tenant_name: Optional[str] = None,
    ) -> None:
        if window_seconds <= 0:
            raise ValueError("window_seconds must be positive")

        end_time = window_end or _utcnow()
        start_time = window_start or (end_time - timedelta(seconds=window_seconds))

        self.ensure_tenant_profile(
            tenant_id,
            tenant_name=tenant_name,
            recall_target=recall_target,
        )

        conn = self._db_connection_factory()
        try:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO adaptive_tenant_windows (
                        tenant_id,
                        window_start,
                        window_end,
                        window_seconds,
                        read_count,
                        write_count,
                        metadata
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb);
                    """,
                    [
                        tenant_id,
                        start_time,
                        end_time,
                        float(window_seconds),
                        int(read_count),
                        int(write_count),
                        _json_dumps(metadata),
                    ],
                )
            conn.commit()
        finally:
            conn.close()

    def list_tenant_ids(self, *, include_profiles: bool = True) -> list[int]:
        conn = self._db_connection_factory()
        try:
            tenant_ids = set(self._scope_adapter.list_tenant_ids(conn))
            if include_profiles:
                with conn.cursor() as cur:
                    cur.execute("SELECT tenant_id FROM adaptive_tenant_profiles ORDER BY tenant_id;")
                    tenant_ids.update(int(row[0]) for row in cur.fetchall())
            return sorted(tenant_ids)
        finally:
            conn.close()

    def get_tenant_state(
        self,
        tenant_id: int,
        *,
        window_limit: int = 10,
        ema_decay: Optional[float] = None,
    ) -> TenantStateSnapshot:
        conn = self._db_connection_factory()
        try:
            average_row_bytes = self._estimate_average_row_bytes(conn)
            vector_dimension = _get_vector_dimension(default=128)
            profile = self._fetch_profile(conn, tenant_id)
            windows = self._fetch_windows(conn, tenant_id, window_limit=window_limit)
            document_ids = self._scope_adapter.fetch_accessible_document_ids(conn, tenant_id)
            document_block_counts = self._scope_adapter.fetch_document_block_counts(conn, document_ids)
            size = self._build_size_metrics(
                tenant_id=tenant_id,
                role_count=self._fetch_role_count(conn, tenant_id),
                document_ids=document_ids,
                document_block_counts=document_block_counts,
                average_row_bytes=average_row_bytes,
                vector_dimension=vector_dimension,
            )
            decay = float(ema_decay if ema_decay is not None else self._ema_decay)
            query_rate_ema = _ema((window.query_rate for window in windows), decay)
            write_rate_ema = _ema((window.write_rate for window in windows), decay)
            last_window_end = windows[-1].window_end if windows else None

            return TenantStateSnapshot(
                tenant_id=tenant_id,
                tenant_name=profile["tenant_name"],
                recall_target=float(profile["recall_target"]),
                query_rate_ema=query_rate_ema,
                write_rate_ema=write_rate_ema,
                last_window_end=last_window_end,
                windows=windows,
                size=size,
                metadata=profile["metadata"],
            )
        finally:
            conn.close()

    def get_all_tenant_states(
        self,
        *,
        tenant_ids: Optional[Iterable[int]] = None,
        window_limit: int = 10,
        ema_decay: Optional[float] = None,
    ) -> list[TenantStateSnapshot]:
        ids = sorted({int(tenant_id) for tenant_id in (tenant_ids if tenant_ids is not None else self.list_tenant_ids())})
        if not ids:
            return []

        decay = float(ema_decay if ema_decay is not None else self._ema_decay)
        conn = self._db_connection_factory()
        try:
            average_row_bytes = self._estimate_average_row_bytes(conn)
            vector_dimension = _get_vector_dimension(default=128)
            profiles = self._fetch_profiles(conn, ids)
            windows_by_tenant = self._fetch_windows_for_many(conn, ids, window_limit=window_limit)
            role_counts = self._fetch_role_counts(conn, ids)
            document_cache = self._scope_adapter.fetch_many_accessible_document_ids(conn, ids)
            all_document_ids = set().union(*(document_cache.get(tenant_id, set()) for tenant_id in ids))
            document_block_counts = self._scope_adapter.fetch_document_block_counts(conn, all_document_ids)
            snapshots: list[TenantStateSnapshot] = []
            for tenant_id in ids:
                profile = profiles.get(int(tenant_id)) or {
                    "tenant_name": f"tenant_{tenant_id}",
                    "recall_target": self._default_recall_target,
                    "metadata": {},
                }
                windows = windows_by_tenant.get(int(tenant_id), [])
                size = self._build_size_metrics(
                    tenant_id=int(tenant_id),
                    role_count=role_counts.get(int(tenant_id), 0),
                    document_ids=document_cache.get(int(tenant_id), set()),
                    document_block_counts=document_block_counts,
                    average_row_bytes=average_row_bytes,
                    vector_dimension=vector_dimension,
                )
                query_rate_ema = _ema((window.query_rate for window in windows), decay)
                write_rate_ema = _ema((window.write_rate for window in windows), decay)
                last_window_end = windows[-1].window_end if windows else None
                snapshots.append(
                    TenantStateSnapshot(
                        tenant_id=int(tenant_id),
                        tenant_name=profile["tenant_name"],
                        recall_target=float(profile["recall_target"]),
                        query_rate_ema=query_rate_ema,
                        write_rate_ema=write_rate_ema,
                        last_window_end=last_window_end,
                        windows=windows,
                        size=size,
                        metadata=profile["metadata"],
                    )
                )
            return snapshots
        finally:
            conn.close()

    def get_recall_target(self, tenant_id: int) -> float:
        conn = self._db_connection_factory()
        try:
            profile = self._fetch_profile(conn, tenant_id)
            return float(profile["recall_target"])
        finally:
            conn.close()

    def estimate_storage_bytes(self, tenant_id: int) -> float:
        return self.get_tenant_state(tenant_id, window_limit=1).size.estimated_total_bytes

    def get_accessible_document_ids(self, tenant_id: int) -> set[int]:
        conn = self._db_connection_factory()
        try:
            return self._scope_adapter.fetch_accessible_document_ids(conn, tenant_id)
        finally:
            conn.close()

    def get_many_accessible_document_ids(self, tenant_ids: Iterable[int]) -> dict[int, set[int]]:
        normalized_ids = sorted({int(tenant_id) for tenant_id in tenant_ids})
        if not normalized_ids:
            return {}
        conn = self._db_connection_factory()
        try:
            return self._scope_adapter.fetch_many_accessible_document_ids(conn, normalized_ids)
        finally:
            conn.close()

    def get_document_block_counts(
        self,
        document_ids: Optional[Iterable[int]] = None,
    ) -> dict[int, int]:
        conn = self._db_connection_factory()
        try:
            return self._scope_adapter.fetch_document_block_counts(conn, document_ids=document_ids)
        finally:
            conn.close()

    def get_group_metrics(self, tenant_ids: Iterable[int]) -> TenantGroupMetrics:
        conn = self._db_connection_factory()
        try:
            return self._scope_adapter.fetch_group_metrics(
                conn,
                tenant_ids,
                average_row_bytes=self._estimate_average_row_bytes(conn),
                vector_dimension=_get_vector_dimension(default=128),
                hnsw_graph_bytes_per_vector=self._hnsw_graph_bytes_per_vector,
            )
        finally:
            conn.close()

    def _fetch_role_count(self, conn, tenant_id: int) -> int:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COUNT(DISTINCT role_id)
                FROM UserRoles
                WHERE user_id = %s;
                """,
                [tenant_id],
            )
            row = cur.fetchone()
        return int((row[0] if row else 0) or 0)

    def _fetch_role_counts(self, conn, tenant_ids: Iterable[int]) -> dict[int, int]:
        normalized_ids = sorted({int(tenant_id) for tenant_id in tenant_ids})
        if not normalized_ids:
            return {}

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT user_id, COUNT(DISTINCT role_id)
                FROM UserRoles
                WHERE user_id = ANY(%s)
                GROUP BY user_id;
                """,
                [normalized_ids],
            )
            rows = cur.fetchall()
        return {int(row[0]): int(row[1] or 0) for row in rows}

    def _build_size_metrics(
        self,
        *,
        tenant_id: int,
        role_count: int,
        document_ids: Iterable[int],
        document_block_counts: dict[int, int],
        average_row_bytes: float,
        vector_dimension: int,
    ) -> TenantSizeMetrics:
        normalized_document_ids = {int(document_id) for document_id in document_ids}
        document_count = len(normalized_document_ids)
        vector_count = sum(int(document_block_counts.get(document_id, 0)) for document_id in normalized_document_ids)
        estimated_table_bytes = vector_count * average_row_bytes
        estimated_hnsw_index_bytes = vector_count * (
            vector_dimension * 4 + self._hnsw_graph_bytes_per_vector
        )
        estimated_total_bytes = estimated_table_bytes + estimated_hnsw_index_bytes
        return TenantSizeMetrics(
            tenant_id=int(tenant_id),
            role_count=int(role_count),
            document_count=document_count,
            vector_count=vector_count,
            average_row_bytes=float(average_row_bytes),
            estimated_table_bytes=estimated_table_bytes,
            estimated_hnsw_index_bytes=estimated_hnsw_index_bytes,
            estimated_total_bytes=estimated_total_bytes,
        )

    def _fetch_profile(self, conn, tenant_id: int) -> dict[str, Any]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tenant_name, recall_target, metadata
                FROM adaptive_tenant_profiles
                WHERE tenant_id = %s;
                """,
                [tenant_id],
            )
            row = cur.fetchone()

        if row is None:
            return {
                "tenant_name": f"tenant_{tenant_id}",
                "recall_target": self._default_recall_target,
                "metadata": {},
            }

        return {
            "tenant_name": row[0],
            "recall_target": row[1],
            "metadata": row[2] or {},
        }

    def _fetch_profiles(self, conn, tenant_ids: Iterable[int]) -> dict[int, dict[str, Any]]:
        normalized_ids = sorted({int(tenant_id) for tenant_id in tenant_ids})
        if not normalized_ids:
            return {}

        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT tenant_id, tenant_name, recall_target, metadata
                FROM adaptive_tenant_profiles
                WHERE tenant_id = ANY(%s);
                """,
                [normalized_ids],
            )
            rows = cur.fetchall()

        return {
            int(row[0]): {
                "tenant_name": row[1],
                "recall_target": row[2],
                "metadata": row[3] or {},
            }
            for row in rows
        }

    def _fetch_windows(self, conn, tenant_id: int, *, window_limit: int) -> list[TenantWindowStat]:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT
                    window_id,
                    tenant_id,
                    window_start,
                    window_end,
                    window_seconds,
                    read_count,
                    write_count,
                    metadata
                FROM adaptive_tenant_windows
                WHERE tenant_id = %s
                ORDER BY window_end DESC
                LIMIT %s;
                """,
                [tenant_id, max(1, int(window_limit))],
            )
            rows = cur.fetchall()

        stats = []
        for row in reversed(rows):
            seconds = float(row[4])
            read_count = int(row[5])
            write_count = int(row[6])
            stats.append(
                TenantWindowStat(
                    window_id=int(row[0]),
                    tenant_id=int(row[1]),
                    window_start=row[2],
                    window_end=row[3],
                    window_seconds=seconds,
                    read_count=read_count,
                    write_count=write_count,
                    query_rate=read_count / seconds,
                    write_rate=write_count / seconds,
                    metadata=row[7] or {},
                )
            )
        return stats

    def _fetch_windows_for_many(
        self,
        conn,
        tenant_ids: Iterable[int],
        *,
        window_limit: int,
    ) -> dict[int, list[TenantWindowStat]]:
        normalized_ids = sorted({int(tenant_id) for tenant_id in tenant_ids})
        result = {tenant_id: [] for tenant_id in normalized_ids}
        if not normalized_ids:
            return result

        with conn.cursor() as cur:
            cur.execute(
                """
                WITH ranked_windows AS (
                    SELECT
                        window_id,
                        tenant_id,
                        window_start,
                        window_end,
                        window_seconds,
                        read_count,
                        write_count,
                        metadata,
                        ROW_NUMBER() OVER (
                            PARTITION BY tenant_id
                            ORDER BY window_end DESC, window_id DESC
                        ) AS rn
                    FROM adaptive_tenant_windows
                    WHERE tenant_id = ANY(%s)
                )
                SELECT
                    window_id,
                    tenant_id,
                    window_start,
                    window_end,
                    window_seconds,
                    read_count,
                    write_count,
                    metadata
                FROM ranked_windows
                WHERE rn <= %s
                ORDER BY tenant_id, window_end ASC, window_id ASC;
                """,
                [normalized_ids, max(1, int(window_limit))],
            )
            rows = cur.fetchall()

        for row in rows:
            seconds = float(row[4])
            read_count = int(row[5])
            write_count = int(row[6])
            result[int(row[1])].append(
                TenantWindowStat(
                    window_id=int(row[0]),
                    tenant_id=int(row[1]),
                    window_start=row[2],
                    window_end=row[3],
                    window_seconds=seconds,
                    read_count=read_count,
                    write_count=write_count,
                    query_rate=read_count / seconds,
                    write_rate=write_count / seconds,
                    metadata=row[7] or {},
                )
            )
        return result

    def _empty_size_metrics(
        self,
        *,
        tenant_id: int,
        average_row_bytes: float,
        vector_dimension: int,
    ) -> TenantSizeMetrics:
        _ = vector_dimension
        return TenantSizeMetrics(
            tenant_id=int(tenant_id),
            role_count=0,
            document_count=0,
            vector_count=0,
            average_row_bytes=float(average_row_bytes),
            estimated_table_bytes=0.0,
            estimated_hnsw_index_bytes=0.0,
            estimated_total_bytes=0.0,
        )

    def _estimate_average_row_bytes(self, conn) -> float:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT COALESCE(AVG(pg_column_size(sample_row)), 0)
                FROM (
                    SELECT db AS sample_row
                    FROM documentblocks AS db
                    LIMIT %s
                ) sampled;
                """,
                [max(1, self._row_sample_limit)],
            )
            row = cur.fetchone()
        return float(row[0] or 0.0)
