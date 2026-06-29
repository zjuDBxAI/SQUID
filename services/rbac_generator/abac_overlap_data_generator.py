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


@dataclass(slots=True)
class DocumentPolicy:
    document_id: int
    department: int
    region: int
    sensitivity: int
    project_ids: tuple[int, ...]
    team_ids: tuple[int, ...]
    rule_templates: tuple[str, ...]
    acl_size: int
    fallback: bool = False


class ABACOverlapDataGenerator:
    """Generate ABAC-style overlapping document ACLs in the existing RBAC schema.

    The native policy model is:

        allow(user, document) = OR(rule(user_attrs, document_attrs))

    The generated ACLs are then projected into the repository's RBAC-compatible
    tables by creating one role per user:

        UserRoles(user_id, role_id=user_id)
        PermissionAssignment(role_id=user_id, document_id)

    This preserves document ACLs exactly while still using the existing schema.
    """

    DEFAULT_RULE_TEMPLATES = ("project", "team", "department")

    def __init__(
        self,
        *,
        num_users: int = 1000,
        document_ids: Iterable[int] | None = None,
        departments: int = 20,
        projects: int = 200,
        teams: int = 120,
        regions: int = 8,
        clearance_levels: int = 4,
        projects_per_user: int = 6,
        teams_per_user: int = 3,
        projects_per_document: int = 2,
        teams_per_document: int = 2,
        rules_per_document: int = 1,
        min_acl_size: int = 30,
        max_acl_size: int = 90,
        attribute_distribution: str = "uniform",
        zipf_alpha: float = 1.2,
        seed: int = 42,
        max_resample_attempts: int = 100,
        reject_duplicate_acls: bool = True,
        rule_templates: Iterable[str] | None = None,
    ) -> None:
        if document_ids is None:
            raise ValueError("document_ids must be provided")
        if num_users <= 0:
            raise ValueError("num_users must be positive")
        if min_acl_size <= 0 or max_acl_size <= 0:
            raise ValueError("ACL size bounds must be positive")
        if min_acl_size > max_acl_size:
            raise ValueError("min_acl_size cannot exceed max_acl_size")
        if max_acl_size > num_users:
            raise ValueError("max_acl_size cannot exceed num_users")
        if max_resample_attempts <= 0:
            raise ValueError("max_resample_attempts must be positive")

        self.num_users = int(num_users)
        self.document_ids = [int(document_id) for document_id in document_ids]
        if not self.document_ids:
            raise ValueError("document_ids cannot be empty")

        self.departments = self._positive_int("departments", departments)
        self.projects = self._positive_int("projects", projects)
        self.teams = self._positive_int("teams", teams)
        self.regions = self._positive_int("regions", regions)
        self.clearance_levels = self._positive_int("clearance_levels", clearance_levels)
        self.projects_per_user = self._bounded_count("projects_per_user", projects_per_user, self.projects)
        self.teams_per_user = self._bounded_count("teams_per_user", teams_per_user, self.teams)
        self.projects_per_document = self._bounded_count("projects_per_document", projects_per_document, self.projects)
        self.teams_per_document = self._bounded_count("teams_per_document", teams_per_document, self.teams)
        self.rules_per_document = self._positive_int("rules_per_document", rules_per_document)
        self.min_acl_size = int(min_acl_size)
        self.max_acl_size = int(max_acl_size)
        self.attribute_distribution = str(attribute_distribution).lower()
        self.zipf_alpha = float(zipf_alpha)
        if self.zipf_alpha <= 0:
            raise ValueError("zipf_alpha must be positive")
        self.seed = int(seed)
        self.max_resample_attempts = int(max_resample_attempts)
        self.reject_duplicate_acls = bool(reject_duplicate_acls)
        self.rng = np.random.default_rng(self.seed)

        templates = tuple(rule_templates or self.DEFAULT_RULE_TEMPLATES)
        invalid = set(templates) - set(self.DEFAULT_RULE_TEMPLATES)
        if invalid:
            raise ValueError(f"Unsupported rule templates: {sorted(invalid)}")
        if not templates:
            raise ValueError("At least one rule template is required")
        if self.rules_per_document > len(templates):
            raise ValueError("rules_per_document cannot exceed available rule_templates")
        self.rule_templates = templates

        self.users = [
            {"user_id": i, "user_name": f"user_{i}"}
            for i in range(1, self.num_users + 1)
        ]
        self.roles = [Role(i, f"user_role_{i}") for i in range(1, self.num_users + 1)]
        self.role_ids = np.arange(1, self.num_users + 1, dtype=np.int64)

        self.user_departments: np.ndarray | None = None
        self.user_regions: np.ndarray | None = None
        self.user_clearance: np.ndarray | None = None
        self.user_project_matrix: np.ndarray | None = None
        self.user_team_matrix: np.ndarray | None = None
        self.document_policies: dict[int, DocumentPolicy] = {}
        self.document_acls: dict[int, tuple[int, ...]] = {}

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    @staticmethod
    def _bounded_count(name: str, value: int, upper: int) -> int:
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        if value > int(upper):
            raise ValueError(f"{name} cannot exceed {upper}")
        return value

    def _distribution_probs(self, cardinality: int) -> np.ndarray:
        if self.attribute_distribution == "uniform":
            weights = np.ones(int(cardinality), dtype=np.float64)
        elif self.attribute_distribution == "zipf":
            ranks = np.arange(1, int(cardinality) + 1, dtype=np.float64)
            weights = 1.0 / np.power(ranks, self.zipf_alpha)
            weights = weights[self.rng.permutation(int(cardinality))]
        else:
            raise ValueError("attribute_distribution must be one of: uniform, zipf")
        weights /= weights.sum()
        return weights

    def _sample_values(self, cardinality: int, count: int) -> np.ndarray:
        values = np.arange(1, int(cardinality) + 1, dtype=np.int64)
        probs = self._distribution_probs(int(cardinality))
        return self.rng.choice(values, size=int(count), replace=False, p=probs)

    def assign_users_to_roles(self) -> list[tuple[int, int]]:
        return [(int(user["user_id"]), int(user["user_id"])) for user in self.users]

    def generate_user_attributes(self) -> None:
        department_probs = self._distribution_probs(self.departments)
        region_probs = self._distribution_probs(self.regions)
        clearance_probs = self._distribution_probs(self.clearance_levels)

        self.user_departments = self.rng.choice(
            np.arange(1, self.departments + 1, dtype=np.int64),
            size=self.num_users,
            replace=True,
            p=department_probs,
        )
        self.user_regions = self.rng.choice(
            np.arange(1, self.regions + 1, dtype=np.int64),
            size=self.num_users,
            replace=True,
            p=region_probs,
        )
        self.user_clearance = self.rng.choice(
            np.arange(1, self.clearance_levels + 1, dtype=np.int64),
            size=self.num_users,
            replace=True,
            p=clearance_probs,
        )

        self.user_project_matrix = np.zeros((self.num_users, self.projects + 1), dtype=bool)
        self.user_team_matrix = np.zeros((self.num_users, self.teams + 1), dtype=bool)
        for user_idx in range(self.num_users):
            projects = self._sample_values(self.projects, self.projects_per_user)
            teams = self._sample_values(self.teams, self.teams_per_user)
            self.user_project_matrix[user_idx, projects] = True
            self.user_team_matrix[user_idx, teams] = True

    def _sample_document_policy_inputs(self) -> tuple[int, int, int, tuple[int, ...], tuple[int, ...], tuple[str, ...]]:
        department = int(self._sample_values(self.departments, 1)[0])
        region = int(self._sample_values(self.regions, 1)[0])
        sensitivity = int(self._sample_values(self.clearance_levels, 1)[0])
        project_ids = tuple(sorted(int(value) for value in self._sample_values(self.projects, self.projects_per_document)))
        team_ids = tuple(sorted(int(value) for value in self._sample_values(self.teams, self.teams_per_document)))
        templates = tuple(
            sorted(
                str(value)
                for value in self.rng.choice(
                    np.asarray(self.rule_templates, dtype=object),
                    size=self.rules_per_document,
                    replace=False,
                ).tolist()
            )
        )
        return department, region, sensitivity, project_ids, team_ids, templates

    def _evaluate_policy(
        self,
        *,
        department: int,
        region: int,
        sensitivity: int,
        project_ids: tuple[int, ...],
        team_ids: tuple[int, ...],
        rule_templates: tuple[str, ...],
    ) -> tuple[int, ...]:
        if (
            self.user_departments is None
            or self.user_regions is None
            or self.user_clearance is None
            or self.user_project_matrix is None
            or self.user_team_matrix is None
        ):
            raise RuntimeError("generate_user_attributes must be called before evaluating policies")

        clearance_ok = self.user_clearance >= int(sensitivity)
        allowed = np.zeros(self.num_users, dtype=bool)
        template_set = set(rule_templates)

        if "project" in template_set:
            project_match = self.user_project_matrix[:, list(project_ids)].any(axis=1)
            allowed |= project_match & clearance_ok

        if "team" in template_set:
            team_match = self.user_team_matrix[:, list(team_ids)].any(axis=1)
            allowed |= team_match & clearance_ok

        if "department" in template_set:
            department_match = self.user_departments == int(department)
            region_match = self.user_regions == int(region)
            allowed |= department_match & region_match & clearance_ok

        return tuple(int(user_id) for user_id in self.role_ids[allowed].tolist())

    def _candidate_score(self, acl_size: int, duplicate: bool) -> tuple[int, int, int]:
        target = (self.min_acl_size + self.max_acl_size) // 2
        empty_penalty = 1 if acl_size == 0 else 0
        duplicate_penalty = 1 if duplicate else 0
        return empty_penalty, duplicate_penalty, abs(int(acl_size) - int(target))

    def generate_document_acls(self) -> tuple[dict[int, DocumentPolicy], dict[int, tuple[int, ...]]]:
        if self.user_departments is None:
            self.generate_user_attributes()

        seen_acls: set[tuple[int, ...]] = set()
        policies: dict[int, DocumentPolicy] = {}
        acls: dict[int, tuple[int, ...]] = {}
        fallback_count = 0

        for document_id in self.document_ids:
            best = None
            best_score = None
            for _ in range(self.max_resample_attempts):
                department, region, sensitivity, project_ids, team_ids, templates = self._sample_document_policy_inputs()
                acl = self._evaluate_policy(
                    department=department,
                    region=region,
                    sensitivity=sensitivity,
                    project_ids=project_ids,
                    team_ids=team_ids,
                    rule_templates=templates,
                )
                duplicate = bool(self.reject_duplicate_acls and acl in seen_acls)
                acl_size = len(acl)
                score = self._candidate_score(acl_size, duplicate)
                candidate = (department, region, sensitivity, project_ids, team_ids, templates, acl)
                if best is None or score < best_score:
                    best = candidate
                    best_score = score
                if (
                    self.min_acl_size <= acl_size <= self.max_acl_size
                    and acl_size > 0
                    and not duplicate
                ):
                    best = candidate
                    best_score = score
                    break

            if best is None:
                raise RuntimeError("Failed to sample a document ACL candidate")

            department, region, sensitivity, project_ids, team_ids, templates, acl = best
            fallback = not (
                self.min_acl_size <= len(acl) <= self.max_acl_size
                and len(acl) > 0
                and not (self.reject_duplicate_acls and acl in seen_acls)
            )
            if fallback:
                fallback_count += 1
            seen_acls.add(acl)
            policies[int(document_id)] = DocumentPolicy(
                document_id=int(document_id),
                department=int(department),
                region=int(region),
                sensitivity=int(sensitivity),
                project_ids=tuple(project_ids),
                team_ids=tuple(team_ids),
                rule_templates=tuple(templates),
                acl_size=int(len(acl)),
                fallback=bool(fallback),
            )
            acls[int(document_id)] = tuple(acl)

        self.document_policies = policies
        self.document_acls = acls
        if fallback_count:
            print(f"ABAC-overlap generator accepted {fallback_count} fallback ACLs outside the exact target window.")
        return policies, acls

    def iter_permission_assignments(self, document_acls: dict[int, tuple[int, ...]] | None = None):
        if document_acls is None:
            document_acls = self.document_acls or self.generate_document_acls()[1]
        for document_id in sorted(document_acls):
            for user_id in document_acls[int(document_id)]:
                yield (int(user_id), int(document_id))

    def generate_rbac_data(self):
        self.generate_user_attributes()
        _, document_acls = self.generate_document_acls()
        user_roles = self.assign_users_to_roles()
        permission_assignments = list(self.iter_permission_assignments(document_acls))
        return self.users, self.roles, user_roles, permission_assignments

    def summarize_structure(
        self,
        document_acls: dict[int, tuple[int, ...]] | None = None,
        *,
        max_pair_samples: int = 1_000_000,
    ) -> dict[str, float | int]:
        if document_acls is None:
            document_acls = self.document_acls or self.generate_document_acls()[1]

        acl_values = [tuple(acl) for _, acl in sorted(document_acls.items())]
        acl_sets = [set(acl) for acl in acl_values]
        acl_sizes = [len(acl) for acl in acl_values]
        distinct_acls = len(set(acl_values))
        n = len(acl_sets)
        total_pairs = n * (n - 1) // 2

        containment = 0
        overlap = 0
        intersection_sum = 0
        jaccard_sum = 0.0
        pair_count = 0

        if total_pairs > 0:
            if total_pairs <= max_pair_samples:
                pair_iter = combinations(range(n), 2)
                pair_count = total_pairs
            else:
                pair_count = int(max_pair_samples)
                pair_iter = (
                    tuple(self.rng.choice(n, size=2, replace=False).tolist())
                    for _ in range(pair_count)
                )
            for left_id, right_id in pair_iter:
                left = acl_sets[int(left_id)]
                right = acl_sets[int(right_id)]
                inter = len(left & right)
                if inter:
                    overlap += 1
                if left < right or right < left:
                    containment += 1
                intersection_sum += inter
                union_size = len(left | right)
                if union_size:
                    jaccard_sum += inter / union_size

        template_counts: dict[str, int] = {}
        fallback_count = 0
        for policy in self.document_policies.values():
            fallback_count += int(bool(policy.fallback))
            for template in policy.rule_templates:
                template_counts[template] = template_counts.get(template, 0) + 1

        acl_array = np.asarray(acl_sizes, dtype=np.float64)
        return {
            "users": int(self.num_users),
            "documents": int(len(document_acls)),
            "distinct_acl_patterns": int(distinct_acls),
            "duplicate_acl_patterns": int(len(document_acls) - distinct_acls),
            "min_acl_size": int(min(acl_sizes) if acl_sizes else 0),
            "max_acl_size": int(max(acl_sizes) if acl_sizes else 0),
            "avg_acl_size": float(acl_array.mean() if acl_sizes else 0.0),
            "std_acl_size": float(acl_array.std() if acl_sizes else 0.0),
            "target_min_acl_size": int(self.min_acl_size),
            "target_max_acl_size": int(self.max_acl_size),
            "fallback_documents": int(fallback_count),
            "permission_assignments": int(sum(acl_sizes)),
            "sampled_pairs": int(pair_count),
            "containment_ratio": float(containment / max(1, pair_count)),
            "overlap_ratio": float(overlap / max(1, pair_count)),
            "avg_intersection": float(intersection_sum / max(1, pair_count)),
            "avg_jaccard": float(jaccard_sum / max(1, pair_count)),
            "project_rule_uses": int(template_counts.get("project", 0)),
            "team_rule_uses": int(template_counts.get("team", 0)),
            "department_rule_uses": int(template_counts.get("department", 0)),
            "expected_project_match_users": float(
                self.num_users
                * (1.0 - math.exp(-float(self.projects_per_user * self.projects_per_document) / max(1, self.projects)))
            ),
            "expected_team_match_users": float(
                self.num_users
                * (1.0 - math.exp(-float(self.teams_per_user * self.teams_per_document) / max(1, self.teams)))
            ),
        }
