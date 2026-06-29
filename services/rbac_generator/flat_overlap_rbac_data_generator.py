import math
import os
import sys
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


@dataclass(slots=True)
class Role:
    role_id: int
    role_name: str
    hierarchy_level: int = 1


class FlatOverlapRBACDataGenerator:
    """Generate a flat-overlap permission workload through an RBAC-compatible schema.

    The generator follows the controllable synthetic-ABAC idea used by MuSimA-style
    workloads: policy structure is produced from user-specified attribute/value
    distributions. In this compatibility layer, each role is an agent/user attribute
    value and each ACL pattern is a fixed-size cohort sampled from the role domain.

    Fixed-size cohorts make strict containment impossible unless two ACLs are equal;
    duplicate cohorts are rejected. Increasing roles_per_acl or using a skewed role
    distribution increases overlap while preserving near-zero containment.
    """

    def __init__(
        self,
        *,
        num_users: int = 1000,
        num_roles: int = 1000,
        document_ids: Iterable[int] | None = None,
        num_acl_patterns: int = 300,
        roles_per_acl: int = 60,
        role_distribution: str = "uniform",
        zipf_alpha: float = 1.2,
        seed: int = 42,
    ) -> None:
        if document_ids is None:
            raise ValueError("document_ids must be provided")
        if num_users <= 0:
            raise ValueError("num_users must be positive")
        if num_roles <= 0:
            raise ValueError("num_roles must be positive")
        if num_acl_patterns <= 0:
            raise ValueError("num_acl_patterns must be positive")
        if roles_per_acl <= 0:
            raise ValueError("roles_per_acl must be positive")
        if roles_per_acl > num_roles:
            raise ValueError("roles_per_acl cannot exceed num_roles")
        if zipf_alpha <= 0:
            raise ValueError("zipf_alpha must be positive")

        self.num_users = int(num_users)
        self.num_roles = int(num_roles)
        self.document_ids = list(document_ids)
        self.num_acl_patterns = int(num_acl_patterns)
        self.roles_per_acl = int(roles_per_acl)
        self.role_distribution = str(role_distribution).lower()
        self.zipf_alpha = float(zipf_alpha)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        if not self.document_ids:
            raise ValueError("document_ids cannot be empty")
        if self.num_acl_patterns > len(self.document_ids):
            raise ValueError("num_acl_patterns cannot exceed number of documents")

        self.roles = [Role(i, f"role_{i}") for i in range(1, self.num_roles + 1)]
        self.users = [
            {"user_id": i, "user_name": f"user_{i}"}
            for i in range(1, self.num_users + 1)
        ]
        self.role_ids = np.arange(1, self.num_roles + 1, dtype=np.int64)
        self.role_probs = self._build_role_distribution()

    def _build_role_distribution(self) -> np.ndarray:
        if self.role_distribution == "uniform":
            weights = np.ones(self.num_roles, dtype=np.float64)
        elif self.role_distribution == "zipf":
            ranks = np.arange(1, self.num_roles + 1, dtype=np.float64)
            weights = 1.0 / np.power(ranks, self.zipf_alpha)
            shuffled = self.rng.permutation(self.num_roles)
            weights = weights[shuffled]
        else:
            raise ValueError("role_distribution must be one of: uniform, zipf")
        weights /= weights.sum()
        return weights

    def assign_users_to_roles(self) -> list[tuple[int, int]]:
        """Assign each synthetic user to exactly one role, evenly by role id."""
        user_roles: list[tuple[int, int]] = []
        for index, user in enumerate(self.users):
            role_id = (index % self.num_roles) + 1
            user_roles.append((int(user["user_id"]), int(role_id)))
        return user_roles

    def generate_acl_cohorts(self) -> list[tuple[int, ...]]:
        """Sample unique fixed-size ACL role cohorts.

        Equal-size unique cohorts have zero strict containment by construction. The
        rejection loop prevents duplicate ACLs, which would otherwise be equal sets.
        """
        cohorts: list[tuple[int, ...]] = []
        seen: set[tuple[int, ...]] = set()
        max_attempts = max(10000, self.num_acl_patterns * 500)
        attempts = 0
        while len(cohorts) < self.num_acl_patterns:
            attempts += 1
            if attempts > max_attempts:
                raise RuntimeError(
                    "Failed to sample enough unique ACL cohorts. Reduce roles_per_acl, "
                    "reduce num_acl_patterns, or use a less skewed role distribution."
                )
            sampled = self.rng.choice(
                self.role_ids,
                size=self.roles_per_acl,
                replace=False,
                p=self.role_probs,
            )
            cohort = tuple(sorted(int(role_id) for role_id in sampled.tolist()))
            if cohort in seen:
                continue
            seen.add(cohort)
            cohorts.append(cohort)
        return cohorts

    def assign_documents_to_acls(
        self,
        acl_cohorts: list[tuple[int, ...]] | None = None,
    ) -> dict[int, list[int]]:
        if acl_cohorts is None:
            acl_cohorts = self.generate_acl_cohorts()
        if len(acl_cohorts) != self.num_acl_patterns:
            raise ValueError("acl_cohorts length must equal num_acl_patterns")

        document_ids = np.array(self.document_ids, dtype=object)
        self.rng.shuffle(document_ids)
        chunks = np.array_split(document_ids, self.num_acl_patterns)
        return {
            acl_id: [int(document_id) for document_id in chunks[acl_id].tolist()]
            for acl_id in range(self.num_acl_patterns)
        }

    def iter_permission_assignments(
        self,
        acl_cohorts: list[tuple[int, ...]],
        acl_documents: dict[int, list[int]],
    ):
        for acl_id, cohort in enumerate(acl_cohorts):
            for document_id in acl_documents.get(int(acl_id), []):
                for role_id in cohort:
                    yield (int(role_id), int(document_id))

    def generate_rbac_data(self):
        user_roles = self.assign_users_to_roles()
        acl_cohorts = self.generate_acl_cohorts()
        acl_documents = self.assign_documents_to_acls(acl_cohorts)
        permission_assignments = list(self.iter_permission_assignments(acl_cohorts, acl_documents))
        return self.users, self.roles, user_roles, permission_assignments

    def summarize_structure(
        self,
        acl_cohorts: list[tuple[int, ...]],
        acl_documents: dict[int, list[int]] | None = None,
        *,
        max_pair_samples: int = 1_000_000,
    ) -> dict[str, float | int]:
        n = len(acl_cohorts)
        total_pairs = n * (n - 1) // 2
        if total_pairs <= 0:
            return {
                "acl_patterns": n,
                "roles_per_acl": self.roles_per_acl,
                "containment_ratio": 0.0,
                "overlap_ratio": 0.0,
                "avg_intersection": 0.0,
                "avg_jaccard": 0.0,
            }

        if total_pairs <= max_pair_samples:
            pair_iter = combinations(range(n), 2)
            pair_count = total_pairs
        else:
            pair_count = max_pair_samples
            pair_iter = (
                tuple(self.rng.choice(n, size=2, replace=False).tolist())
                for _ in range(max_pair_samples)
            )

        sets = [set(cohort) for cohort in acl_cohorts]
        containment = 0
        overlap = 0
        intersection_sum = 0
        jaccard_sum = 0.0
        for left_id, right_id in pair_iter:
            left = sets[int(left_id)]
            right = sets[int(right_id)]
            inter = len(left & right)
            if inter > 0:
                overlap += 1
            if left < right or right < left:
                containment += 1
            intersection_sum += inter
            union_size = len(left | right)
            if union_size > 0:
                jaccard_sum += inter / union_size

        summary: dict[str, float | int] = {
            "acl_patterns": int(n),
            "roles": int(self.num_roles),
            "roles_per_acl": int(self.roles_per_acl),
            "sampled_pairs": int(pair_count),
            "containment_ratio": float(containment / max(1, pair_count)),
            "overlap_ratio": float(overlap / max(1, pair_count)),
            "avg_intersection": float(intersection_sum / max(1, pair_count)),
            "avg_jaccard": float(jaccard_sum / max(1, pair_count)),
            "expected_uniform_intersection": float((self.roles_per_acl ** 2) / max(1, self.num_roles)),
            "expected_uniform_overlap_probability": float(
                1.0 - math.exp(-float(self.roles_per_acl ** 2) / max(1, self.num_roles))
            ),
        }
        if acl_documents is not None:
            sizes = [len(acl_documents.get(acl_id, [])) for acl_id in range(n)]
            summary.update(
                {
                    "documents": int(sum(sizes)),
                    "min_documents_per_acl": int(min(sizes) if sizes else 0),
                    "max_documents_per_acl": int(max(sizes) if sizes else 0),
                    "avg_documents_per_acl": float(sum(sizes) / max(1, len(sizes))),
                    "permission_assignments": int(sum(sizes) * self.roles_per_acl),
                }
            )
        return summary
