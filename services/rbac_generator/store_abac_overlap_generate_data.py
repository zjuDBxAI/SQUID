import argparse
import os
import sys
from itertools import islice

from psycopg2.extras import execute_values

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from services.config import get_db_connection
from services.rbac_generator.abac_overlap_data_generator import ABACOverlapDataGenerator


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


def _bulk_store(generator, user_roles, document_acls, batch_size):
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
        permission_iter = generator.iter_permission_assignments(document_acls)
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


def _parse_rule_templates(value: str):
    templates = tuple(item.strip() for item in str(value).split(",") if item.strip())
    if not templates:
        raise argparse.ArgumentTypeError("rule template list cannot be empty")
    return templates


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate ABAC-style overlapping ACLs and store them in the existing "
            "RBAC-compatible schema with one role per user."
        )
    )
    parser.add_argument("--num-users", type=int, default=1000)
    parser.add_argument("--departments", type=int, default=20)
    parser.add_argument("--projects", type=int, default=200)
    parser.add_argument("--teams", type=int, default=120)
    parser.add_argument("--regions", type=int, default=8)
    parser.add_argument("--clearance-levels", type=int, default=4)
    parser.add_argument("--projects-per-user", type=int, default=6)
    parser.add_argument("--teams-per-user", type=int, default=3)
    parser.add_argument("--projects-per-document", type=int, default=2)
    parser.add_argument("--teams-per-document", type=int, default=2)
    parser.add_argument("--rules-per-document", type=int, default=1)
    parser.add_argument("--rule-templates", type=_parse_rule_templates, default=("project", "team", "department"))
    parser.add_argument("--min-acl-size", type=int, default=30)
    parser.add_argument("--max-acl-size", type=int, default=90)
    parser.add_argument("--attribute-distribution", choices=("uniform", "zipf"), default="uniform")
    parser.add_argument("--zipf-alpha", type=float, default=1.2)
    parser.add_argument("--max-resample-attempts", type=int, default=100)
    parser.add_argument("--allow-duplicate-acls", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument("--dry-run", action="store_true", help="Generate and summarize without writing RBAC tables.")
    args = parser.parse_args()

    document_ids = _fetch_document_ids()
    if not document_ids:
        raise RuntimeError("No documents found. Load the vector dataset before generating permissions.")

    generator = ABACOverlapDataGenerator(
        num_users=args.num_users,
        document_ids=document_ids,
        departments=args.departments,
        projects=args.projects,
        teams=args.teams,
        regions=args.regions,
        clearance_levels=args.clearance_levels,
        projects_per_user=args.projects_per_user,
        teams_per_user=args.teams_per_user,
        projects_per_document=args.projects_per_document,
        teams_per_document=args.teams_per_document,
        rules_per_document=args.rules_per_document,
        rule_templates=args.rule_templates,
        min_acl_size=args.min_acl_size,
        max_acl_size=args.max_acl_size,
        attribute_distribution=args.attribute_distribution,
        zipf_alpha=args.zipf_alpha,
        max_resample_attempts=args.max_resample_attempts,
        reject_duplicate_acls=not args.allow_duplicate_acls,
        seed=args.seed,
    )
    generator.generate_user_attributes()
    _, document_acls = generator.generate_document_acls()
    summary = generator.summarize_structure(document_acls)

    print("ABAC-overlap workload summary:")
    for key in sorted(summary):
        value = summary[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    if args.dry_run:
        print("Dry run complete; RBAC tables were not modified.")
        return

    user_roles = generator.assign_users_to_roles()
    _reset_rbac_tables()
    inserted_permissions = _bulk_store(
        generator,
        user_roles,
        document_acls,
        max(1, int(args.batch_size)),
    )
    print(
        "Stored ABAC-overlap workload: "
        f"users={len(generator.users)}, roles={len(generator.roles)}, "
        f"user_roles={len(user_roles)}, documents={len(document_acls)}, "
        f"permission_assignments={inserted_permissions}"
    )


if __name__ == "__main__":
    main()
