import argparse
import os
import sys
from itertools import islice

from psycopg2.extras import execute_values

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from services.rbac_generator.flat_overlap_rbac_data_generator import FlatOverlapRBACDataGenerator
from services.config import get_db_connection


MU_SIMA_REFERENCE = (
    "MuSimA-style synthetic ABAC generation: user-specified attribute/value "
    "distributions control access cohorts; stored through the RBAC compatibility schema."
)


def _fetch_document_ids():
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute("SELECT document_id FROM documents ORDER BY document_id;")
        return [int(row[0]) for row in cur.fetchall()]
    finally:
        cur.close()
        conn.close()


def _reset_rbac_tables():
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


def _batched(iterator, batch_size):
    iterator = iter(iterator)
    while True:
        batch = list(islice(iterator, batch_size))
        if not batch:
            break
        yield batch


def _bulk_store(generator, user_roles, acl_cohorts, acl_documents, batch_size):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        execute_values(
            cur,
            "INSERT INTO users (user_id, user_name) VALUES %s",
            [(user["user_id"], user["user_name"]) for user in generator.users],
            page_size=batch_size,
        )
        execute_values(
            cur,
            "INSERT INTO roles (role_id, role_name) VALUES %s",
            [(role.role_id, role.role_name) for role in generator.roles],
            page_size=batch_size,
        )
        execute_values(
            cur,
            "INSERT INTO userroles (user_id, role_id) VALUES %s",
            user_roles,
            page_size=batch_size,
        )

        inserted_permissions = 0
        permission_iter = generator.iter_permission_assignments(acl_cohorts, acl_documents)
        for batch in _batched(permission_iter, batch_size):
            execute_values(
                cur,
                "INSERT INTO permissionassignment (role_id, document_id) VALUES %s",
                batch,
                page_size=batch_size,
            )
            inserted_permissions += len(batch)
            if inserted_permissions % max(batch_size * 20, 1) == 0:
                print(f"Inserted permission assignments: {inserted_permissions}")
        conn.commit()
        return inserted_permissions
    except Exception:
        conn.rollback()
        raise
    finally:
        cur.close()
        conn.close()


def main():
    parser = argparse.ArgumentParser(
        description="Generate a flat-overlap RBAC-compatible workload from MuSimA-style controlled ABAC distributions."
    )
    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--num-roles", type=int, default=1000)
    parser.add_argument("--num-acls", type=int, default=300)
    parser.add_argument("--roles-per-acl", type=int, default=60)
    parser.add_argument("--role-distribution", choices=("uniform", "zipf"), default="uniform")
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true", help="Generate and summarize without writing RBAC tables.")
    args = parser.parse_args()

    document_ids = _fetch_document_ids()
    if not document_ids:
        raise RuntimeError("No documents found. Load the vector dataset before generating permissions.")

    generator = FlatOverlapRBACDataGenerator(
        num_users=args.num_users,
        num_roles=args.num_roles,
        document_ids=document_ids,
        num_acl_patterns=args.num_acls,
        roles_per_acl=args.roles_per_acl,
        role_distribution=args.role_distribution,
        zipf_alpha=args.zipf_alpha,
        seed=args.seed,
    )
    user_roles = generator.assign_users_to_roles()
    acl_cohorts = generator.generate_acl_cohorts()
    acl_documents = generator.assign_documents_to_acls(acl_cohorts)
    summary = generator.summarize_structure(acl_cohorts, acl_documents)

    print(MU_SIMA_REFERENCE)
    print("Flat-overlap workload summary:")
    for key in sorted(summary):
        value = summary[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    if args.dry_run:
        print("Dry run complete; RBAC tables were not modified.")
        return

    _reset_rbac_tables()
    inserted_permissions = _bulk_store(
        generator,
        user_roles,
        acl_cohorts,
        acl_documents,
        max(1, int(args.batch_size)),
    )
    print(
        "Stored flat-overlap RBAC workload: "
        f"users={len(generator.users)}, roles={len(generator.roles)}, "
        f"user_roles={len(user_roles)}, acl_patterns={len(acl_cohorts)}, "
        f"permission_assignments={inserted_permissions}"
    )


if __name__ == "__main__":
    main()
