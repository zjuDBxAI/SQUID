import argparse
import ast
import os
import sys
from collections import Counter, defaultdict

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from services.rbac_generator.saas_workload import SAASDataGenerator


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


def _format_pairs(pairs, limit=10):
    return ", ".join(f"{key}:{value}" for key, value in pairs[:limit])


def _summarize_tenant_sizes(tenant_document_assignments):
    tenant_sizes = sorted(
        ((tenant_id, len(doc_ids)) for tenant_id, doc_ids in tenant_document_assignments.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    total_assignments = sum(size for _, size in tenant_sizes)
    top_n = max(1, len(tenant_sizes) // 5) if tenant_sizes else 1
    top_share = (
        sum(size for _, size in tenant_sizes[:top_n]) / total_assignments if total_assignments else 0.0
    )

    print("== Tenant Size Distribution ==")
    print(f"tenant_count={len(tenant_sizes)} total_assignments={total_assignments} top20_share={top_share:.4f}")
    print(f"largest: {_format_pairs(tenant_sizes)}")
    print(f"smallest: {_format_pairs(list(reversed(tenant_sizes)), limit=min(10, len(tenant_sizes)))}")
    print()


def _summarize_sharing(permission_assignments, total_documents):
    doc_to_tenants = defaultdict(set)
    for tenant_id, document_id in permission_assignments:
        doc_to_tenants[document_id].add(tenant_id)

    sharing_counts = Counter(len(tenant_ids) for tenant_ids in doc_to_tenants.values())
    mean_sharing = (
        sum(shared * count for shared, count in sharing_counts.items()) / total_documents
        if total_documents else 0.0
    )

    print("== Sharing Distribution ==")
    print(f"covered_documents={len(doc_to_tenants)} total_documents={total_documents} mean_shared_tenants={mean_sharing:.4f}")
    for shared in sorted(sharing_counts):
        count = sharing_counts[shared]
        ratio = count / total_documents if total_documents else 0.0
        print(f"shared_by_{shared}_tenants={count} ({ratio:.4%})")
    print()


def _summarize_owner_creator(owner_assignments, creator_assignments):
    creator_hist = Counter(creator_assignments.values())
    owner_hist = Counter(owner_assignments.values())

    print("== Owner / Creator Summary ==")
    print(f"owner_documents={len(owner_assignments)} creator_documents={len(creator_assignments)} unique_creators={len(creator_hist)}")
    print(f"top_creators: {_format_pairs(creator_hist.most_common(10))}")
    print(f"top_owner_tenants: {_format_pairs(owner_hist.most_common(10))}")
    print()


def _summarize_semantics(generator, tenant_profiles, tenant_document_assignments, sample_tenants=8):
    tenant_sizes = sorted(
        ((tenant_id, len(doc_ids)) for tenant_id, doc_ids in tenant_document_assignments.items()),
        key=lambda item: item[1],
        reverse=True,
    )
    if not tenant_sizes:
        print("== Semantic Concentration ==")
        print("no tenant assignments")
        print()
        return

    selected_tenant_ids = []
    selected_tenant_ids.extend([tenant_id for tenant_id, _ in tenant_sizes[: min(3, len(tenant_sizes))]])
    middle_idx = len(tenant_sizes) // 2
    selected_tenant_ids.append(tenant_sizes[middle_idx][0])
    selected_tenant_ids.extend([tenant_id for tenant_id, _ in tenant_sizes[-min(3, len(tenant_sizes)) :]])
    selected_tenant_ids = list(dict.fromkeys(selected_tenant_ids))[:sample_tenants]

    print("== Semantic Concentration ==")
    print("tenant_id size target center_count selected_mean global_mean concentration outlier_mean local_mean")

    for tenant_id in selected_tenant_ids:
        profile = tenant_profiles[tenant_id]
        doc_ids = tenant_document_assignments.get(tenant_id, [])
        if not doc_ids:
            continue

        scores = generator._score_documents_for_tenant(profile.center_doc_ids)
        selected_indices = [generator.doc_index[doc_id] for doc_id in doc_ids]
        selected_scores = scores[selected_indices]
        global_mean = float(np.mean(scores))
        selected_mean = float(np.mean(selected_scores))
        concentration = selected_mean / global_mean if global_mean > 0 else 0.0

        outlier_count = min(int(round(len(doc_ids) * profile.outlier_ratio)), len(doc_ids))
        sorted_selected = np.sort(selected_scores)
        outlier_mean = float(np.mean(sorted_selected[:outlier_count])) if outlier_count > 0 else 0.0
        local_mean = float(np.mean(sorted_selected[outlier_count:])) if outlier_count < len(doc_ids) else 0.0

        print(
            f"{tenant_id} {len(doc_ids)} {profile.target_doc_count} {len(profile.center_doc_ids)} "
            f"{selected_mean:.4f} {global_mean:.4f} {concentration:.4f} {outlier_mean:.4f} {local_mean:.4f}"
        )
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Debug tenant-only SaaS workload generation without writing compatibility RBAC data."
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
    parser.add_argument("--sample-tenants", type=int, default=8)
    args = parser.parse_args()

    document_ids, document_vectors = _fetch_document_vectors()
    if not document_ids:
        raise RuntimeError(
            "No document vectors found. Load the dataset first so documentblocks contains vectors."
        )

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

    covered_documents = {doc_id for docs in tenant_document_assignments.values() for doc_id in docs}

    print("== Run Summary ==")
    print(f"documents={len(document_ids)} covered_documents={len(covered_documents)}")
    print(f"tenants={len(tenants)}")
    print(f"permission_assignments={len(permission_assignments)}")
    print()

    _summarize_tenant_sizes(tenant_document_assignments)
    _summarize_sharing(permission_assignments, len(document_ids))
    _summarize_owner_creator(owner_assignments, creator_assignments)
    _summarize_semantics(generator, tenant_profiles, tenant_document_assignments, sample_tenants=args.sample_tenants)


if __name__ == "__main__":
    main()
