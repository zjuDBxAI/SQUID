import argparse
import os
import sys
from itertools import islice
from pathlib import Path

from psycopg2.extras import execute_values

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from services.config import get_db_connection
from services.rbac_generator.yfcc_abac_data_generator import (
    DEFAULT_YFCC_METADATA_PATH,
    XU_STOLLER_STYLE_DEFAULTS,
    YFCCABACDataGenerator,
    load_yfcc_metadata_rows,
)


YFCC_ABAC_REFERENCE = (
    "YFCC multi-attribute ABAC workload: resource attributes are Curator/YFCC "
    "metadata labels; rules are multi-label conjunctive clauses following the "
    "ABAC policy-rule structure used by Xu-Stoller-style synthetic ABAC policies."
)


def _fetch_document_ids(limit=None):
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        if limit is None:
            cur.execute("SELECT document_id FROM documents ORDER BY document_id;")
            return [int(row[0]) for row in cur.fetchall()]
        cur.execute("SELECT document_id FROM documents ORDER BY document_id LIMIT %s;", (int(limit),))
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


def _bulk_store(generator, user_roles, batch_size):
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
        for batch in _batched(generator.iter_permission_assignments(), batch_size):
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
        description=(
            "Generate a multi-attribute ABAC workload from YFCC100M metadata labels "
            "and store the resolved user-document relation in the existing "
            "RBAC-compatible schema."
        )
    )
    parser.add_argument("--metadata-path", default=DEFAULT_YFCC_METADATA_PATH)
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--num-users", type=int, default=XU_STOLLER_STYLE_DEFAULTS["num_users"], help="|U|: number of users")
    parser.add_argument("--num-rules", type=int, default=XU_STOLLER_STYLE_DEFAULTS["num_rules"], help="Nrule: number of ABAC policy rules")
    parser.add_argument("--rules-per-user", type=int, default=XU_STOLLER_STYLE_DEFAULTS["rules_per_user"], help="Number of ABAC rules assigned to each user")
    parser.add_argument(
        "--conjunct-count",
        type=int,
        default=XU_STOLLER_STYLE_DEFAULTS["conjunct_count"],
        help="c: number of metadata labels required by each ABAC rule; must be >= 2.",
    )
    parser.add_argument(
        "--target-avg-sharing-degree",
        type=float,
        default=XU_STOLLER_STYLE_DEFAULTS["target_avg_sharing_degree"],
        help="sbar: target average number of users that can access each document.",
    )
    parser.add_argument("--overlap-probability", type=float, default=XU_STOLLER_STYLE_DEFAULTS["overlap_probability"], help="p_o: probability of sampling a rule overlapping an existing rule")
    parser.add_argument("--min-user-resource-pairs", type=int, default=XU_STOLLER_STYLE_DEFAULTS["min_user_resource_pairs"], help="N_urp: minimum user-resource pairs / rule support accepted")
    parser.add_argument("--candidate-pool-size", type=int, default=XU_STOLLER_STYLE_DEFAULTS["candidate_pool_size"], help="N_cand: number of candidate clauses sampled per rule")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=10000)
    parser.add_argument(
        "--sidecar-path",
        default="/data/Multitenanthakes/dataset/yfcc100m/yfcc_abac_policy.json",
        help="Where to write the generated ABAC rules and user attributes.",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    document_ids = _fetch_document_ids(limit=args.max_documents)
    if not document_ids:
        raise RuntimeError("No documents found. Load the vector dataset before generating permissions.")

    document_labels = load_yfcc_metadata_rows(
        metadata_path=args.metadata_path,
        document_ids=document_ids,
        row_mapping="document_id_minus_one",
    )

    generator = YFCCABACDataGenerator(
        document_ids=document_ids,
        document_labels=document_labels,
        num_users=args.num_users,
        num_rules=args.num_rules,
        rules_per_user=args.rules_per_user,
        conjunct_count=args.conjunct_count,
        target_avg_sharing_degree=args.target_avg_sharing_degree,
        overlap_probability=args.overlap_probability,
        min_user_resource_pairs=args.min_user_resource_pairs,
        candidate_pool_size=args.candidate_pool_size,
        seed=args.seed,
    )

    summary = generator.summarize_structure()
    print(YFCC_ABAC_REFERENCE)
    print("YFCC ABAC workload summary:")
    for key in sorted(summary):
        value = summary[key]
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")

    if args.sidecar_path:
        generator.write_policy_sidecar(Path(args.sidecar_path), summary=summary)
        print(f"Wrote ABAC policy sidecar: {args.sidecar_path}")

    if args.dry_run:
        print("Dry run complete; RBAC tables were not modified.")
        return

    user_roles = generator.assign_users_to_roles()
    _reset_rbac_tables()
    inserted_permissions = _bulk_store(
        generator,
        user_roles,
        max(1, int(args.batch_size)),
    )
    print(
        "Stored YFCC ABAC workload: "
        f"users={len(generator.users)}, roles={len(generator.roles)}, "
        f"user_roles={len(user_roles)}, documents={len(document_ids)}, "
        f"permission_assignments={inserted_permissions}"
    )


if __name__ == "__main__":
    main()
