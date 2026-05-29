import random
from dataclasses import dataclass
from collections import defaultdict

import numpy as np


@dataclass
class Tenant:
    tenant_id: int
    tenant_name: str


@dataclass
class TenantProfile:
    tenant_id: int
    target_doc_count: int
    outlier_ratio: float
    center_doc_ids: list


class SAASDataGenerator:
    """
    Pure tenant-based SaaS workload generator.

    There is only one principal concept: tenant.
    - creator_assignments maps document -> creator tenant
    - owner_assignments maps document -> owner tenant
    - permission_assignments stores (tenant_id, document_id)
    """

    def __init__(
        self,
        document_ids,
        document_vectors,
        seed,
        num_tenants=None,
        num_role=None,
        max_doc=5000,
        alpha=1.4,
        outlier_ratio=0.08,
        num_interest_centers=2,
        max_shared_tenants=4,
        min_doc_per_tenant=1,
    ):
        if num_tenants is None:
            num_tenants = num_role
        if num_tenants is None:
            raise ValueError("num_tenants must be provided")
        if num_tenants <= 0:
            raise ValueError("num_tenants must be positive")
        if max_doc <= 0:
            raise ValueError("max_doc must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0.0 <= outlier_ratio < 1.0:
            raise ValueError("outlier_ratio must be in [0, 1)")
        if num_interest_centers <= 0:
            raise ValueError("num_interest_centers must be positive")
        if max_shared_tenants <= 0:
            raise ValueError("max_shared_tenants must be positive")
        if min_doc_per_tenant <= 0:
            raise ValueError("min_doc_per_tenant must be positive")
        if not document_ids:
            raise ValueError("document_ids must not be empty")

        self.document_ids = list(document_ids)
        self.document_vectors = np.asarray(document_vectors, dtype=np.float32)
        if len(self.document_ids) != len(self.document_vectors):
            raise ValueError("document_ids and document_vectors must have the same length")

        self.seed = seed
        self.num_tenants = num_tenants
        self.max_doc = max_doc
        self.alpha = alpha
        self.outlier_ratio = outlier_ratio
        self.num_interest_centers = num_interest_centers
        self.max_shared_tenants = max_shared_tenants
        self.min_doc_per_tenant = min_doc_per_tenant

        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.tenants = [
            Tenant(tenant_id=tenant_id, tenant_name=f"tenant_{tenant_id}")
            for tenant_id in range(1, num_tenants + 1)
        ]
        self.tenant_ids = [tenant.tenant_id for tenant in self.tenants]

        self.document_vectors = self._normalize_vectors(self.document_vectors)
        self.doc_index = {doc_id: idx for idx, doc_id in enumerate(self.document_ids)}

        self.owner_assignments = {}
        self.creator_assignments = {}
        self.tenant_profiles = {}
        self.tenant_document_assignments = {}

    def _normalize_vectors(self, vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return vectors / norms

    def _build_long_tail_sizes(self, n_items, max_size, min_size=1, alpha=None, shuffle_ids=True):
        if n_items <= 0:
            raise ValueError("n_items must be positive")
        if max_size <= 0:
            raise ValueError("max_size must be positive")
        if min_size <= 0:
            raise ValueError("min_size must be positive")

        if alpha is None:
            alpha = self.alpha

        ranks = np.arange(1, n_items + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, alpha)
        weights = weights / weights[0]

        sizes = np.round(weights * max_size).astype(int)
        sizes = np.maximum(sizes, min_size)
        sizes = np.minimum(sizes, max_size)

        item_ids = list(range(1, n_items + 1))
        if shuffle_ids:
            self.rng.shuffle(item_ids)

        return {item_id: int(size) for item_id, size in zip(item_ids, sizes.tolist())}

    def build_long_tail_tenant_profiles(self):
        tenant_doc_sizes = self._build_long_tail_sizes(
            n_items=self.num_tenants,
            max_size=min(self.max_doc, len(self.document_ids)),
            min_size=self.min_doc_per_tenant,
            alpha=self.alpha,
            shuffle_ids=True,
        )

        profiles = {}
        for tenant_id, target_doc_count in tenant_doc_sizes.items():
            center_count = min(self.num_interest_centers, len(self.document_ids))
            center_doc_ids = self.rng.sample(self.document_ids, center_count)
            profiles[tenant_id] = TenantProfile(
                tenant_id=tenant_id,
                target_doc_count=target_doc_count,
                outlier_ratio=self.outlier_ratio,
                center_doc_ids=center_doc_ids,
            )

        self.tenant_profiles = profiles
        return profiles

    def _score_documents_for_tenant(self, center_doc_ids):
        center_indices = [self.doc_index[doc_id] for doc_id in center_doc_ids]
        center_vectors = self.document_vectors[center_indices]
        scores = self.document_vectors @ center_vectors.T
        if scores.ndim == 1:
            return scores
        return scores.max(axis=1)

    def _select_documents_for_tenant(self, profile):
        scores = self._score_documents_for_tenant(profile.center_doc_ids)
        ranked_indices = np.argsort(-scores)

        target_doc_count = min(profile.target_doc_count, len(self.document_ids))
        if target_doc_count <= 0:
            return []

        outlier_count = int(round(target_doc_count * profile.outlier_ratio))
        outlier_count = min(outlier_count, max(0, target_doc_count - 1))
        local_count = max(1, target_doc_count - outlier_count)

        # Keep local picks tightly concentrated around the most similar region.
        local_pool_size = min(len(ranked_indices), max(local_count, local_count * 2))
        local_pool = ranked_indices[:local_pool_size]

        if local_count >= len(local_pool):
            local_indices = np.asarray(local_pool)
        else:
            local_weights = scores[local_pool].astype(np.float64)
            local_weights = np.maximum(local_weights, 1e-8)
            local_weights = local_weights / local_weights.sum()
            local_indices = self.np_rng.choice(
                local_pool,
                size=local_count,
                replace=False,
                p=local_weights,
            )

        chosen_indices = list(np.atleast_1d(local_indices).tolist())

        if outlier_count > 0:
            tail_start = max(local_pool_size, int(len(ranked_indices) * 0.85))
            tail_pool = ranked_indices[tail_start:]
            if len(tail_pool) == 0:
                tail_pool = ranked_indices[local_pool_size:]
            if len(tail_pool) == 0:
                tail_pool = ranked_indices[-outlier_count:]

            tail_candidates = [idx for idx in tail_pool.tolist() if idx not in chosen_indices]
            if tail_candidates:
                sampled_outliers = self.np_rng.choice(
                    tail_candidates,
                    size=min(outlier_count, len(tail_candidates)),
                    replace=False,
                )
                chosen_indices.extend(np.atleast_1d(sampled_outliers).tolist())

        # Always include the semantic centers so the tenant keeps a strong core.
        for center_doc_id in profile.center_doc_ids:
            center_idx = self.doc_index[center_doc_id]
            if center_idx not in chosen_indices:
                chosen_indices.append(center_idx)

        # Trim back to target size by removing the weakest non-center docs first.
        chosen_indices = list(dict.fromkeys(chosen_indices))
        center_indices = {self.doc_index[doc_id] for doc_id in profile.center_doc_ids}
        if len(chosen_indices) > target_doc_count:
            chosen_indices.sort(key=lambda idx: (idx in center_indices, scores[idx]), reverse=True)
            chosen_indices = chosen_indices[:target_doc_count]

        chosen_doc_ids = [self.document_ids[int(i)] for i in chosen_indices]
        return sorted(set(chosen_doc_ids))

    def build_tenant_document_assignments(self, tenant_profiles):
        tenant_to_documents = {}
        for tenant_id, profile in tenant_profiles.items():
            tenant_to_documents[tenant_id] = self._select_documents_for_tenant(profile)
        self.tenant_document_assignments = tenant_to_documents
        return tenant_to_documents

    def invert_tenant_assignments(self, tenant_to_documents):
        document_to_tenants = defaultdict(set)
        for tenant_id, doc_ids in tenant_to_documents.items():
            for doc_id in doc_ids:
                document_to_tenants[doc_id].add(tenant_id)
        return document_to_tenants

    def cap_shared_tenants(self, document_to_tenants, tenant_profiles, max_shared_tenants=None):
        if max_shared_tenants is None:
            max_shared_tenants = self.max_shared_tenants

        for doc_id, tenant_ids in list(document_to_tenants.items()):
            if len(tenant_ids) <= max_shared_tenants:
                continue

            doc_idx = self.doc_index[doc_id]
            doc_vec = self.document_vectors[doc_idx]
            scored_tenants = []

            for tenant_id in tenant_ids:
                center_indices = [self.doc_index[c] for c in tenant_profiles[tenant_id].center_doc_ids]
                center_vecs = self.document_vectors[center_indices]
                sim = float((center_vecs @ doc_vec).max())
                scored_tenants.append((sim, tenant_id))

            scored_tenants.sort(reverse=True)
            kept_tenants = {tenant_id for _, tenant_id in scored_tenants[:max_shared_tenants]}
            document_to_tenants[doc_id] = kept_tenants

        return document_to_tenants

    def _ensure_full_document_coverage(self, document_to_tenants, tenant_profiles):
        for doc_id in self.document_ids:
            if doc_id in document_to_tenants and document_to_tenants[doc_id]:
                continue

            doc_idx = self.doc_index[doc_id]
            doc_vec = self.document_vectors[doc_idx]
            best_tenant_id = None
            best_score = -1.0

            for tenant_id, profile in tenant_profiles.items():
                center_indices = [self.doc_index[c] for c in profile.center_doc_ids]
                center_vecs = self.document_vectors[center_indices]
                score = float((center_vecs @ doc_vec).max())
                if score > best_score:
                    best_score = score
                    best_tenant_id = tenant_id

            document_to_tenants[doc_id].add(best_tenant_id)

        return document_to_tenants

    def _rebalance_tenant_assignments(self, document_to_tenants, tenant_profiles):
        tenant_to_documents = defaultdict(set)
        for doc_id, tenant_ids in document_to_tenants.items():
            for tenant_id in tenant_ids:
                tenant_to_documents[tenant_id].add(doc_id)

        scores_cache = {
            tenant_id: self._score_documents_for_tenant(profile.center_doc_ids)
            for tenant_id, profile in tenant_profiles.items()
        }

        # First prune oversized tenants, preferring to remove weakly matching docs that are shared elsewhere.
        for tenant_id, profile in tenant_profiles.items():
            docs = tenant_to_documents[tenant_id]
            excess = len(docs) - profile.target_doc_count
            if excess <= 0:
                continue

            removable = []
            for doc_id in docs:
                shared_count = len(document_to_tenants[doc_id])
                if shared_count <= 1:
                    continue
                score = float(scores_cache[tenant_id][self.doc_index[doc_id]])
                removable.append((score, -shared_count, doc_id))

            removable.sort()
            removed = 0
            for _, _, doc_id in removable:
                if removed >= excess:
                    break
                if tenant_id in document_to_tenants[doc_id] and len(document_to_tenants[doc_id]) > 1:
                    document_to_tenants[doc_id].remove(tenant_id)
                    tenant_to_documents[tenant_id].remove(doc_id)
                    removed += 1

        # Then fill undersized tenants with high-scoring docs that still have sharing capacity.
        ranked_cache = {
            tenant_id: np.argsort(-scores_cache[tenant_id])
            for tenant_id in tenant_profiles.keys()
        }
        for tenant_id, profile in tenant_profiles.items():
            docs = tenant_to_documents[tenant_id]
            deficit = profile.target_doc_count - len(docs)
            if deficit <= 0:
                continue

            for idx in ranked_cache[tenant_id]:
                if deficit <= 0:
                    break
                doc_id = self.document_ids[int(idx)]
                if doc_id in docs:
                    continue
                if len(document_to_tenants[doc_id]) >= self.max_shared_tenants:
                    continue
                document_to_tenants[doc_id].add(tenant_id)
                docs.add(doc_id)
                deficit -= 1

        return document_to_tenants

    def _choose_owner_tenant(self, doc_id, tenant_ids, tenant_profiles):
        doc_idx = self.doc_index[doc_id]
        doc_vec = self.document_vectors[doc_idx]

        best_tenant_id = None
        best_score = -1.0
        for tenant_id in tenant_ids:
            center_indices = [self.doc_index[c] for c in tenant_profiles[tenant_id].center_doc_ids]
            center_vecs = self.document_vectors[center_indices]
            score = float((center_vecs @ doc_vec).max())
            if score > best_score:
                best_score = score
                best_tenant_id = tenant_id

        return best_tenant_id

    def assign_owners_and_creators(self, document_to_tenants, tenant_profiles):
        self.owner_assignments = {}
        self.creator_assignments = {}

        for doc_id, tenant_ids in document_to_tenants.items():
            if not tenant_ids:
                raise ValueError(f"Document {doc_id} has no accessible tenants")

            owner_tenant_id = self._choose_owner_tenant(doc_id, tenant_ids, tenant_profiles)
            self.owner_assignments[doc_id] = owner_tenant_id

            # In the pure tenant model, creator is also a tenant.
            self.creator_assignments[doc_id] = owner_tenant_id

        return self.owner_assignments, self.creator_assignments

    def build_permission_assignments(self, document_to_tenants):
        tenant_to_documents = defaultdict(list)
        permission_assignments = []

        for doc_id, tenant_ids in document_to_tenants.items():
            for tenant_id in sorted(tenant_ids):
                tenant_to_documents[tenant_id].append(doc_id)
                permission_assignments.append((tenant_id, doc_id))

        tenant_document_assignments = {
            tenant_id: sorted(set(doc_ids))
            for tenant_id, doc_ids in tenant_to_documents.items()
        }
        self.tenant_document_assignments = tenant_document_assignments
        return tenant_document_assignments, permission_assignments

    def generate_workload(self):
        tenant_profiles = self.build_long_tail_tenant_profiles()
        tenant_to_documents = self.build_tenant_document_assignments(tenant_profiles)
        document_to_tenants = self.invert_tenant_assignments(tenant_to_documents)
        document_to_tenants = self.cap_shared_tenants(document_to_tenants, tenant_profiles)
        document_to_tenants = self._ensure_full_document_coverage(document_to_tenants, tenant_profiles)
        document_to_tenants = self._rebalance_tenant_assignments(document_to_tenants, tenant_profiles)
        self.assign_owners_and_creators(document_to_tenants, tenant_profiles)
        tenant_document_assignments, permission_assignments = self.build_permission_assignments(
            document_to_tenants
        )

        return {
            "tenants": self.tenants,
            "tenant_profiles": tenant_profiles,
            "tenant_document_assignments": tenant_document_assignments,
            "permission_assignments": permission_assignments,
            "owner_assignments": self.owner_assignments,
            "creator_assignments": self.creator_assignments,
        }
