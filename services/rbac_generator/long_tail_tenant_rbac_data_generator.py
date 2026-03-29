import os
import sys
from collections import defaultdict

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


class Role:
    def __init__(self, role_id, role_name, hierarchy_level=1):
        self.role_id = role_id
        self.role_name = role_name
        self.hierarchy_level = hierarchy_level


class LongTailTenantRBACDataGenerator:
    """Generate a Curator-style long-tail tenant workload for document permissions.

    Design goals:
    - each document has exactly one owner role
    - a small fraction of documents are shared with a few extra roles
    - role sizes follow a power-law distribution
    - every role gets at least one document when possible
    """

    def __init__(
        self,
        num_users=1000,
        num_roles=100,
        document_ids=None,
        alpha=1.5,
        share_prob=0.05,
        max_extra_roles=2,
        seed=42,
    ):
        if document_ids is None:
            raise ValueError("document_ids must be provided")
        if num_roles <= 0:
            raise ValueError("num_roles must be positive")
        if num_users <= 0:
            raise ValueError("num_users must be positive")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if not 0 <= share_prob <= 1:
            raise ValueError("share_prob must be in [0, 1]")
        if max_extra_roles < 0:
            raise ValueError("max_extra_roles must be non-negative")

        self.num_users = num_users
        self.num_roles = num_roles
        self.document_ids = list(document_ids)
        self.alpha = alpha
        self.share_prob = share_prob
        self.max_extra_roles = max_extra_roles
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.original_roles = [
            Role(i, f"role_{i}", hierarchy_level=1) for i in range(1, num_roles + 1)
        ]
        self.users = [
            {"user_id": i, "user_name": f"user_{i}"} for i in range(1, num_users + 1)
        ]

        ranks = np.arange(1, num_roles + 1, dtype=np.float64)
        weights = 1.0 / np.power(ranks, alpha)
        weights /= weights.sum()

        self.role_ids = np.arange(1, num_roles + 1, dtype=np.int64)
        self.rng.shuffle(self.role_ids)
        self.role_probs = weights

    def assign_users_to_roles_evenly(self):
        user_roles = []
        user_subsets = np.array_split(self.users, self.num_roles)
        for role, subset in zip(self.original_roles, user_subsets):
            for user in subset:
                user_roles.append((user["user_id"], role.role_id))
        return user_roles

    def _sample_additional_roles(self, owner_role_id, n_extra):
        candidate_mask = self.role_ids != owner_role_id
        candidate_ids = self.role_ids[candidate_mask]
        if len(candidate_ids) == 0 or n_extra <= 0:
            return set()

        candidate_probs = self.role_probs[candidate_mask]
        candidate_probs = candidate_probs / candidate_probs.sum()
        n_extra = min(n_extra, len(candidate_ids))
        extra_roles = self.rng.choice(
            candidate_ids,
            size=n_extra,
            replace=False,
            p=candidate_probs,
        )
        return {int(role_id) for role_id in np.atleast_1d(extra_roles)}

    def assign_long_tail_permissions(self):
        role_assignments = defaultdict(set)
        remaining_document_ids = self.document_ids.copy()

        # Ensure every role gets at least one document when possible.
        if len(remaining_document_ids) >= self.num_roles:
            self.rng.shuffle(remaining_document_ids)
            bootstrap_docs = remaining_document_ids[: self.num_roles]
            remaining_document_ids = remaining_document_ids[self.num_roles :]
            for role_id, doc_id in zip(self.role_ids.tolist(), bootstrap_docs):
                role_assignments[int(role_id)].add(doc_id)

        for doc_id in remaining_document_ids:
            owner_idx = int(self.rng.choice(self.num_roles, p=self.role_probs))
            owner_role_id = int(self.role_ids[owner_idx])
            assigned_roles = {owner_role_id}

            if self.max_extra_roles > 0 and self.share_prob > 0 and self.rng.random() < self.share_prob:
                n_extra = int(self.rng.integers(1, self.max_extra_roles + 1))
                assigned_roles.update(self._sample_additional_roles(owner_role_id, n_extra))

            for role_id in assigned_roles:
                role_assignments[role_id].add(doc_id)

        assigned_docs = {doc_id for docs in role_assignments.values() for doc_id in docs}
        missing_docs = set(self.document_ids) - assigned_docs
        if missing_docs:
            raise ValueError(f"Unassigned documents: {sorted(list(missing_docs))[:10]}")

        document_assignments = {
            role_id: sorted(doc_ids) for role_id, doc_ids in sorted(role_assignments.items())
        }
        return document_assignments

    def generate_rbac_data(self):
        user_roles = self.assign_users_to_roles_evenly()
        document_assignments = self.assign_long_tail_permissions()

        permission_assignments = []
        for role_id, documents in document_assignments.items():
            for document_id in documents:
                permission_assignments.append((role_id, document_id))

        return self.users, user_roles, document_assignments, permission_assignments
