"""Role-level incremental maintenance for the frozen Tree-alpha traces.

The regular SQUID updater operates on document ACL deltas.  A Honeybee-style
role insertion/deletion is represented as one batched ACL delta per affected
document, with the role and its users created (or removed) under the exact
RBAC semantics frozen by ``role-evolution-v2``.
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Iterable, Mapping

from psycopg2.extras import execute_values

from .update import KMeansUpdateItem, KMeansUpdateResult, apply_kmeans_update_batch
from services.config import get_db_connection


def _normalise(values: Iterable[int]) -> tuple[int, ...]:
    return tuple(sorted({int(value) for value in values}))


@dataclass(frozen=True)
class KMeansRoleUpdateResult:
    """Role-trace maintenance result with separate logical and physical costs."""

    role_id: int
    operation: str
    logical_rbac_seconds: float
    maintenance: KMeansUpdateResult | None
    metadata: Mapping[str, object]


def _insert_role_membership(*, role_id: int, user_ids: tuple[int, ...]) -> float:
    if not user_ids:
        raise ValueError("role insertion requires at least one user")
    started = time.perf_counter()
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT 1 FROM roles WHERE role_id = %s", [role_id])
            if cur.fetchone() is not None:
                raise RuntimeError(f"Role {role_id} already exists")
            cur.execute("SELECT user_id FROM users WHERE user_id = ANY(%s)", [list(user_ids)])
            conflicts = sorted(int(row[0]) for row in cur.fetchall())
            if conflicts:
                raise RuntimeError(
                    "role insertion user ids already exist: " + ", ".join(map(str, conflicts[:10]))
                )
            cur.execute("INSERT INTO roles (role_id, role_name) VALUES (%s, %s)", [role_id, f"role_{role_id}"])
            execute_values(
                cur,
                "INSERT INTO users (user_id, user_name) VALUES %s",
                [(user_id, f"user_{user_id}") for user_id in user_ids],
            )
            execute_values(
                cur,
                "INSERT INTO userroles (user_id, role_id) VALUES %s",
                [(user_id, role_id) for user_id in user_ids],
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return time.perf_counter() - started


def _remove_empty_role(*, role_id: int, user_ids: tuple[int, ...]) -> float:
    """Remove a role after its permission edges have been incrementally revoked."""
    started = time.perf_counter()
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT user_id FROM userroles WHERE role_id = %s ORDER BY user_id", [role_id])
            actual_users = _normalise(row[0] for row in cur.fetchall())
            if actual_users != user_ids:
                raise RuntimeError(
                    f"role deletion users differ from trace for role {role_id}: "
                    f"expected {list(user_ids)}, found {list(actual_users)}"
                )
            cur.execute("SELECT COUNT(*) FROM permissionassignment WHERE role_id = %s", [role_id])
            if int(cur.fetchone()[0]) != 0:
                raise RuntimeError(f"role {role_id} still has permissions after revoke maintenance")
            cur.execute("DELETE FROM userroles WHERE role_id = %s", [role_id])
            if user_ids:
                cur.execute(
                    """
                    DELETE FROM users u
                    WHERE u.user_id = ANY(%s)
                      AND NOT EXISTS (SELECT 1 FROM userroles ur WHERE ur.user_id = u.user_id)
                    """,
                    [list(user_ids)],
                )
            cur.execute("DELETE FROM roles WHERE role_id = %s", [role_id])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return time.perf_counter() - started


def _role_documents(role_id: int) -> tuple[int, ...]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT document_id FROM permissionassignment WHERE role_id = %s ORDER BY document_id",
                [role_id],
            )
            return _normalise(row[0] for row in cur.fetchall())
    finally:
        conn.close()


def insert_role_incrementally(
    *,
    role_id: int,
    document_ids: Iterable[int],
    user_ids: Iterable[int],
    tau_del: float = 0.2,
    max_operations: int = 8,
    max_new_pattern_partitions: int = 2,
    create_indexes: bool = True,
    index_type: str = "squidhnsw",
    vector_index_min_vectors: int = 1,
) -> KMeansRoleUpdateResult:
    """Insert a new trace role and locally maintain SQUID partitions."""
    role_id = int(role_id)
    docs = _normalise(document_ids)
    users = _normalise(user_ids)
    if not docs:
        raise ValueError("role insertion requires at least one document")
    logical_seconds = _insert_role_membership(role_id=role_id, user_ids=users)
    try:
        maintenance = apply_kmeans_update_batch(
            [KMeansUpdateItem(operation="acl_grant", document_id=document_id, role_ids=(role_id,)) for document_id in docs],
            tau_del=tau_del,
            max_operations=max_operations,
            max_new_pattern_partitions=max_new_pattern_partitions,
            create_indexes=create_indexes,
            index_type=index_type,
            vector_index_min_vectors=vector_index_min_vectors,
        )
    except Exception:
        # The permission grants are transactional inside the updater.  Remove
        # the membership-only residue so a failed operation is not reusable as
        # a different trace state.
        _remove_empty_role(role_id=role_id, user_ids=users)
        raise
    return KMeansRoleUpdateResult(
        role_id=role_id,
        operation="role_insertion",
        logical_rbac_seconds=logical_seconds,
        maintenance=maintenance,
        metadata={
            "maintenance_mode": "incremental_role_insertion",
            "global_repartition_invoked": False,
            "document_count": len(docs),
            "user_count": len(users),
        },
    )


def delete_role_incrementally(
    *,
    role_id: int,
    user_ids: Iterable[int],
    document_ids: Iterable[int] | None = None,
    tau_del: float = 0.2,
    max_operations: int = 8,
    max_new_pattern_partitions: int = 2,
    create_indexes: bool = True,
    index_type: str = "squidhnsw",
    vector_index_min_vectors: int = 1,
) -> KMeansRoleUpdateResult:
    """Revoke a role's documents, maintain partitions, then delete the role/users."""
    role_id = int(role_id)
    users = _normalise(user_ids)
    docs = _normalise(document_ids if document_ids is not None else _role_documents(role_id))
    if not docs:
        raise ValueError(f"role deletion requires permissions for role {role_id}")
    maintenance = apply_kmeans_update_batch(
        [KMeansUpdateItem(operation="acl_revoke", document_id=document_id, role_ids=(role_id,)) for document_id in docs],
        tau_del=tau_del,
        max_operations=max_operations,
        max_new_pattern_partitions=max_new_pattern_partitions,
        create_indexes=create_indexes,
        index_type=index_type,
        vector_index_min_vectors=vector_index_min_vectors,
    )
    logical_seconds = _remove_empty_role(role_id=role_id, user_ids=users)
    return KMeansRoleUpdateResult(
        role_id=role_id,
        operation="role_deletion",
        logical_rbac_seconds=logical_seconds,
        maintenance=maintenance,
        metadata={
            "maintenance_mode": "incremental_role_deletion",
            "global_repartition_invoked": False,
            "document_count": len(docs),
            "user_count": len(users),
        },
    )


def apply_role_logical_operation(
    *,
    operation: str,
    role_id: int,
    document_ids: Iterable[int] = (),
    user_ids: Iterable[int] = (),
) -> dict[str, object]:
    """Apply one trace operation for SQUID global-rebuild mode only."""
    normalized_operation = str(operation).strip().lower()
    role_id = int(role_id)
    docs = _normalise(document_ids)
    users = _normalise(user_ids)
    if normalized_operation == "role_insertion":
        started = time.perf_counter()
        _insert_role_membership(role_id=role_id, user_ids=users)
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                execute_values(
                    cur,
                    "INSERT INTO permissionassignment (role_id, document_id) VALUES %s",
                    [(role_id, document_id) for document_id in docs],
                )
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        return {"operation": normalized_operation, "role_id": role_id, "logical_rbac_seconds": time.perf_counter() - started}
    if normalized_operation == "role_deletion":
        started = time.perf_counter()
        conn = get_db_connection()
        try:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM permissionassignment WHERE role_id = %s", [role_id])
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        _remove_empty_role(role_id=role_id, user_ids=users)
        return {"operation": normalized_operation, "role_id": role_id, "logical_rbac_seconds": time.perf_counter() - started}
    raise ValueError(f"unsupported role operation: {operation!r}")


__all__ = [
    "KMeansRoleUpdateResult",
    "apply_role_logical_operation",
    "delete_role_incrementally",
    "insert_role_incrementally",
]
