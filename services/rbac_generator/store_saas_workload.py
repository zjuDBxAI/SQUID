import argparse
import ast
import os
import sys
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from services.rbac_generator.saas_workload import SAASDataGenerator


@dataclass
class CompatibilityRole:
    role_id: int
    role_name: str


def _parse_vector(raw_vector):
    if raw_vector is None:
        return None
    if isinstance(raw_vector, np.ndarray):
        return raw_vector.astype(np.float32)
    if isinstance(raw_vector, (list, tuple)):
        return np.asarray(raw_vector, dtype=np.float32)
    if isinstance(raw_vector, str):
        return np.asarray(ast.literal_eval(raw_vector), dtype=np.float32)
    try:
        return np.asarray(list(raw_vector), dtype=np.float32)
    except TypeError as exc:
        raise ValueError(f"Unsupported vector type: {type(raw_vector)}") from exc


def _fetch_document_vectors():
    from services.config import get_db_connection

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "SELECT document_id, vector FROM documentblocks WHERE vector IS NOT NULL ORDER BY document_id, block_id;"
        )
        vectors_by_document = defaultdict(list)
        for document_id, raw_vector in cur.fetchall():
            vector = _parse_vector(raw_vector)
            if vector is not None and vector.size > 0:
                vectors_by_document[document_id].append(vector)

        if not vectors_by_document:
            return [], np.empty((0, 0), dtype=np.float32)

        document_ids = sorted(vectors_by_document.keys())
        document_vectors = []
        for document_id in document_ids:
            stacked = np.vstack(vectors_by_document[document_id])
            document_vectors.append(stacked.mean(axis=0))

        return document_ids, np.asarray(document_vectors, dtype=np.float32)
    finally:
        cur.close()
        conn.close()


def _reset_rbac_tables():
    from services.config import get_db_connection

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "TRUNCATE TABLE userroles, permissionassignment, users, roles RESTART IDENTITY CASCADE;"
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()


def _print_assignment_summary(permission_assignments, total_documents, owner_assignments, creator_assignments):
    from services.rbac_generator.common import (
        compute_average_selectivity,
        convert_to_role_assignments,
    )

    role_assignments = convert_to_role_assignments(permission_assignments)
    avg_selectivity = compute_average_selectivity(role_assignments, total_documents)

    role_sizes = sorted((len(docs) for docs in role_assignments.values()), reverse=True)
    top_n = max(1, len(role_sizes) // 5) if role_sizes else 1
    top_share = (sum(role_sizes[:top_n]) / sum(role_sizes)) if role_sizes else 0.0

    print(f"Average role selectivity: {avg_selectivity:.6f}")
    print(f"Top 20% roles hold {top_share:.6f} of all role-document assignments")
    print(f"Top 10 role sizes: {role_sizes[:10]}")
    print(f"Owner assignments: {len(owner_assignments)}")
    print(f"Creator assignments: {len(creator_assignments)}")


def _build_compatibility_rbac(tenants, permission_assignments):
    users = []
    roles = []
    user_roles = []

    for tenant in tenants:
        tenant_id = tenant.tenant_id
        users.append({"user_id": tenant_id, "user_name": tenant.tenant_name})
        roles.append(CompatibilityRole(role_id=tenant_id, role_name=f"tenant_{tenant_id}"))
        user_roles.append((tenant_id, tenant_id))

    return users, roles, user_roles, permission_assignments


def main():
    parser = argparse.ArgumentParser(
        description="Generate and store a tenant-only SaaS workload using the currently loaded document vectors."
    )
    parser.add_argument("--num-tenants", type=int, default=100)
    parser.add_argument("--num-roles", dest="num_tenants", type=int)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-doc", type=int, default=5000)
    parser.add_argument("--alpha", type=float, default=1.4)
    parser.add_argument("--outlier-ratio", type=float, default=0.08)
    parser.add_argument("--num-interest-centers", type=int, default=2)
    parser.add_argument("--max-shared-tenants", type=int, default=4)
    parser.add_argument("--min-doc-per-tenant", type=int, default=1)
    args = parser.parse_args()

    document_ids, document_vectors = _fetch_document_vectors()
    if not document_ids:
        raise RuntimeError(
            "No document vectors found. Load the dataset first so documentblocks contains vectors."
        )

    _reset_rbac_tables()

    generator = SAASDataGenerator(
        num_tenants=args.num_tenants,
        document_ids=document_ids,
        document_vectors=document_vectors,
        seed=args.seed,
        max_doc=args.max_doc,
        alpha=args.alpha,
        outlier_ratio=args.outlier_ratio,
        num_interest_centers=args.num_interest_centers,
        max_shared_tenants=args.max_shared_tenants,
        min_doc_per_tenant=args.min_doc_per_tenant,
    )
    workload = generator.generate_workload()

    tenant_document_assignments = workload["tenant_document_assignments"]
    permission_assignments = workload["permission_assignments"]
    owner_assignments = workload["owner_assignments"]
    creator_assignments = workload["creator_assignments"]
    tenant_profiles = workload["tenant_profiles"]
    tenants = workload["tenants"]

    assigned_document_ids = set()
    for docs in tenant_document_assignments.values():
        assigned_document_ids.update(docs)

    assert set(document_ids) == assigned_document_ids, "Not all document_ids are assigned"

    _print_assignment_summary(
        permission_assignments,
        len(document_ids),
        owner_assignments,
        creator_assignments,
    )

    users, roles, user_roles, permission_assignments = _build_compatibility_rbac(
        tenants,
        permission_assignments,
    )

    from services.read_dataset_function import store_rbac_data

    store_rbac_data(users, roles, user_roles, permission_assignments)

    print(
        f"Stored SaaS workload via RBAC compatibility layer: tenants={len(tenants)}, "
        f"synthetic_users={len(users)}, synthetic_roles={len(roles)}, "
        f"user_roles={len(user_roles)}, permission_assignments={len(permission_assignments)}"
    )
    print(f"Tenant profile count: {len(tenant_profiles)}")


if __name__ == "__main__":
    main()
