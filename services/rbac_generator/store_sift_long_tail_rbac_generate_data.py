import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from services.rbac_generator.long_tail_tenant_rbac_data_generator import LongTailTenantRBACDataGenerator


def _fetch_document_ids():
    from services.config import get_db_connection

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT document_id FROM documents ORDER BY document_id;")
        return [row[0] for row in cur.fetchall()]
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


def _print_assignment_summary(permission_assignments, total_documents):
    from services.rbac_generator.common import (
        compute_average_selectivity,
        convert_to_role_assignments,
    )

    role_assignments = convert_to_role_assignments(permission_assignments)
    avg_selectivity = compute_average_selectivity(role_assignments, total_documents)
    role_sizes = sorted((len(docs) for docs in role_assignments.values()), reverse=True)
    top_n = max(1, len(role_sizes) // 5)
    top_share = sum(role_sizes[:top_n]) / sum(role_sizes)

    print(f"Average role selectivity: {avg_selectivity:.6f}")
    print(f"Top 20% roles hold {top_share:.6f} of all role-document assignments")
    print(f"Top 10 role sizes: {role_sizes[:10]}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate a Curator-style long-tail tenant assignment scheme for the "
            "currently loaded SIFT-128 documents without changing existing logic."
        )
    )
    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--num-roles", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=1.5)
    parser.add_argument("--share-prob", type=float, default=0.05)
    parser.add_argument("--max-extra-roles", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    document_ids = _fetch_document_ids()
    if not document_ids:
        raise RuntimeError(
            "No documents found. Load the SIFT-128 dataset first with basic_benchmark/common_prepare_pipeline.py"
        )

    _reset_rbac_tables()

    generator = LongTailTenantRBACDataGenerator(
        num_users=args.num_users,
        num_roles=args.num_roles,
        document_ids=document_ids,
        alpha=args.alpha,
        share_prob=args.share_prob,
        max_extra_roles=args.max_extra_roles,
        seed=args.seed,
    )

    users, user_roles, document_assignments, permission_assignments = generator.generate_rbac_data()

    assigned_document_ids = set()
    for docs in document_assignments.values():
        assigned_document_ids.update(docs)
    assert set(document_ids) == assigned_document_ids, "Not all document_ids are assigned"

    _print_assignment_summary(permission_assignments, len(document_ids))

    from services.read_dataset_function import store_rbac_data

    store_rbac_data(users, generator.original_roles, user_roles, permission_assignments)
    print(
        f"Stored long-tail SIFT RBAC data: users={len(users)}, roles={len(generator.original_roles)}, "
        f"user_roles={len(user_roles)}, permission_assignments={len(permission_assignments)}"
    )


if __name__ == '__main__':
    main()
