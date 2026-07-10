import json
import os
import pickle
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import numpy as np

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)


DEFAULT_YFCC_METADATA_PATH = (
    "/data/Multitenanthakes/dataset/yfcc100m/"
    "yfcc_subsampled_nvec_1000000_nlabel_1000_mds.pkl"
)

# Xu-Stoller-style synthetic ABAC workload defaults used by the YFCC
# materialization script. The mapping is:
# |U| -> num_users, Nrule -> num_rules, c -> conjunct_count,
# sbar -> target_avg_sharing_degree, p_o -> overlap_probability,
# N_urp -> min_user_resource_pairs, N_cand -> candidate_pool_size.
XU_STOLLER_STYLE_DEFAULTS = {
    "num_users": 1000,
    "num_rules": 50,
    "rules_per_user": 2,
    "conjunct_count": 2,
    "target_avg_sharing_degree": 13.37,
    "overlap_probability": 0.5,
    "min_user_resource_pairs": 16,
    "candidate_pool_size": 256,
}


@dataclass(slots=True)
class Role:
    role_id: int
    role_name: str
    hierarchy_level: int = 1


@dataclass(slots=True)
class YFCCABACRule:
    rule_id: int
    labels: tuple[int, ...]
    support_size: int
    source_doc_pos: int
    parent_rule_id: int | None = None


def load_yfcc_metadata_rows(
    *,
    metadata_path: str | Path = DEFAULT_YFCC_METADATA_PATH,
    document_ids: Iterable[int],
    row_mapping: str = "order",
) -> list[tuple[int, ...]]:
    """Load YFCC metadata labels and align them to current document ids.

    The Curator/YFCC file stores one list of integer metadata labels per vector.
    This helper does not modify the vector dataset; it only returns label rows to
    be used as resource attributes for ABAC workload generation.
    """

    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        raise FileNotFoundError(f"YFCC metadata file does not exist: {metadata_path}")

    document_ids = [int(document_id) for document_id in document_ids]
    with metadata_path.open("rb") as f:
        all_rows = pickle.load(f)

    if row_mapping == "order":
        if len(document_ids) > len(all_rows):
            raise ValueError(
                f"Need {len(document_ids)} metadata rows, but {metadata_path} has {len(all_rows)} rows"
            )
        selected_rows = all_rows[: len(document_ids)]
    elif row_mapping == "document_id_minus_one":
        max_index = max(document_ids) - 1 if document_ids else -1
        if max_index >= len(all_rows):
            raise ValueError(
                f"document_id_minus_one mapping needs row {max_index}, "
                f"but {metadata_path} has only {len(all_rows)} rows"
            )
        selected_rows = [all_rows[document_id - 1] for document_id in document_ids]
    else:
        raise ValueError("row_mapping must be one of: order, document_id_minus_one")

    return [tuple(sorted(set(int(label) for label in row))) for row in selected_rows]


class YFCCABACDataGenerator:
    """Generate multi-attribute ABAC permissions from YFCC metadata labels.

    The native policy model follows the ABAC rule shape used in policy-mining
    work: allow(user, document, read) if a user attribute expression and a
    resource attribute expression satisfy a rule. Here:

    - resource attribute: metadata label set M(d) from YFCC;
    - user attribute: interest label set I(u);
    - rule clause: C, a multi-label conjunction sampled from real YFCC label
      co-occurrences;
    - access condition: C subset I(u) and C subset M(d).

    The generated user-document relation is projected into the repository's
    existing RBAC-compatible schema by creating one role per user.
    """

    def __init__(
        self,
        *,
        document_ids: Iterable[int],
        document_labels: Iterable[Iterable[int]],
        num_users: int = XU_STOLLER_STYLE_DEFAULTS["num_users"],
        num_rules: int = XU_STOLLER_STYLE_DEFAULTS["num_rules"],
        rules_per_user: int = XU_STOLLER_STYLE_DEFAULTS["rules_per_user"],
        conjunct_count: int = XU_STOLLER_STYLE_DEFAULTS["conjunct_count"],
        target_avg_sharing_degree: float = XU_STOLLER_STYLE_DEFAULTS["target_avg_sharing_degree"],
        overlap_probability: float = XU_STOLLER_STYLE_DEFAULTS["overlap_probability"],
        min_user_resource_pairs: int = XU_STOLLER_STYLE_DEFAULTS["min_user_resource_pairs"],
        candidate_pool_size: int = XU_STOLLER_STYLE_DEFAULTS["candidate_pool_size"],
        seed: int = 42,
    ) -> None:
        self.document_ids = [int(document_id) for document_id in document_ids]
        self.document_labels = [
            tuple(sorted(set(int(label) for label in labels)))
            for labels in document_labels
        ]
        if len(self.document_ids) != len(self.document_labels):
            raise ValueError("document_ids and document_labels must have the same length")
        if not self.document_ids:
            raise ValueError("document_ids cannot be empty")

        self.num_users = self._positive_int("num_users", num_users)
        self.num_rules = self._positive_int("num_rules", num_rules)
        self.rules_per_user = self._positive_int("rules_per_user", rules_per_user)
        self.conjunct_count = self._positive_int("conjunct_count", conjunct_count)
        if self.conjunct_count < 2:
            raise ValueError("conjunct_count must be at least 2 for multi-attribute ABAC")

        self.target_avg_sharing_degree = float(target_avg_sharing_degree)
        if self.target_avg_sharing_degree <= 0:
            raise ValueError("target_avg_sharing_degree must be positive")

        self.overlap_probability = float(overlap_probability)
        if not 0.0 <= self.overlap_probability <= 1.0:
            raise ValueError("overlap_probability must be in [0, 1]")

        self.min_user_resource_pairs = self._positive_int(
            "min_user_resource_pairs", min_user_resource_pairs
        )
        self.candidate_pool_size = self._positive_int("candidate_pool_size", candidate_pool_size)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)

        self.users = [
            {"user_id": i, "user_name": f"user_{i}"}
            for i in range(1, self.num_users + 1)
        ]
        self.roles = [Role(i, f"user_role_{i}") for i in range(1, self.num_users + 1)]
        self.rules: list[YFCCABACRule] = []
        self.user_rule_assignments: dict[int, tuple[int, ...]] = {}
        self.user_interest_labels: dict[int, tuple[int, ...]] = {}
        self._support_cache: dict[tuple[int, ...], np.ndarray] = {}

        self._build_inverted_metadata()

    @staticmethod
    def _positive_int(name: str, value: int) -> int:
        value = int(value)
        if value <= 0:
            raise ValueError(f"{name} must be positive")
        return value

    def _build_inverted_metadata(self) -> None:
        label_to_positions: dict[int, list[int]] = {}
        candidate_positions = []
        for pos, labels in enumerate(self.document_labels):
            if len(labels) >= self.conjunct_count:
                candidate_positions.append(pos)
            for label in labels:
                label_to_positions.setdefault(int(label), []).append(pos)

        if not candidate_positions:
            raise ValueError(
                "No metadata row contains enough labels for the requested conjunct_count"
            )

        self.candidate_doc_positions = np.asarray(candidate_positions, dtype=np.int64)
        self.label_to_positions = {
            label: np.asarray(positions, dtype=np.int64)
            for label, positions in label_to_positions.items()
        }

    def assign_users_to_roles(self) -> list[tuple[int, int]]:
        return [(int(user["user_id"]), int(user["user_id"])) for user in self.users]

    def _support_positions(self, labels: Iterable[int]) -> np.ndarray:
        key = tuple(sorted(set(int(label) for label in labels)))
        cached = self._support_cache.get(key)
        if cached is not None:
            return cached

        postings = []
        for label in key:
            posting = self.label_to_positions.get(label)
            if posting is None:
                empty = np.empty(0, dtype=np.int64)
                self._support_cache[key] = empty
                return empty
            postings.append(posting)

        postings.sort(key=len)
        support = postings[0]
        for posting in postings[1:]:
            support = np.intersect1d(support, posting, assume_unique=True)
            if support.size == 0:
                break

        self._support_cache[key] = support
        return support

    def _sample_clause_from_document(self) -> tuple[tuple[int, ...], int]:
        source_pos = int(self.rng.choice(self.candidate_doc_positions))
        labels = self.document_labels[source_pos]
        clause = tuple(
            sorted(
                int(label)
                for label in self.rng.choice(
                    np.asarray(labels, dtype=np.int64),
                    size=self.conjunct_count,
                    replace=False,
                ).tolist()
            )
        )
        return clause, source_pos

    def _sample_overlapping_clause(self, parent: YFCCABACRule) -> tuple[tuple[int, ...], int]:
        parent_labels = tuple(parent.labels)
        keep_count = max(1, self.conjunct_count - 1)
        kept = tuple(
            sorted(
                int(label)
                for label in self.rng.choice(
                    np.asarray(parent_labels, dtype=np.int64),
                    size=min(keep_count, len(parent_labels)),
                    replace=False,
                ).tolist()
            )
        )
        parent_support = self._support_positions(parent_labels)
        if parent_support.size == 0:
            return self._sample_clause_from_document()

        for _ in range(20):
            source_pos = int(self.rng.choice(parent_support))
            label_pool = [label for label in self.document_labels[source_pos] if label not in kept]
            if len(label_pool) >= self.conjunct_count - len(kept):
                extra = self.rng.choice(
                    np.asarray(label_pool, dtype=np.int64),
                    size=self.conjunct_count - len(kept),
                    replace=False,
                ).tolist()
                return tuple(sorted(int(label) for label in kept + tuple(extra))), source_pos

        return self._sample_clause_from_document()

    def _candidate_score(
        self,
        labels: tuple[int, ...],
        support_size: int,
        target_support: float,
        duplicate: bool,
    ) -> tuple[int, float, int]:
        duplicate_penalty = 1 if duplicate else 0
        return duplicate_penalty, abs(float(support_size) - float(target_support)), int(labels[0])

    def generate_rules(self) -> list[YFCCABACRule]:
        if self.rules:
            return self.rules

        target_support = (
            self.target_avg_sharing_degree * float(len(self.document_ids)) / float(self.num_users)
        )
        min_support = max(1, int(np.ceil(self.min_user_resource_pairs)))
        seen_clauses: set[tuple[int, ...]] = set()

        for rule_id in range(1, self.num_rules + 1):
            best = None
            best_score = None

            for _ in range(self.candidate_pool_size):
                parent_rule = None
                if self.rules and self.rng.random() < self.overlap_probability:
                    parent_rule = self.rules[int(self.rng.integers(0, len(self.rules)))]
                    clause, source_pos = self._sample_overlapping_clause(parent_rule)
                else:
                    clause, source_pos = self._sample_clause_from_document()

                if len(clause) != self.conjunct_count:
                    continue
                support = self._support_positions(clause)
                if support.size < min_support:
                    continue

                duplicate = clause in seen_clauses
                score = self._candidate_score(clause, int(support.size), target_support, duplicate)
                if best is None or score < best_score:
                    best = (clause, source_pos, parent_rule.rule_id if parent_rule else None, support)
                    best_score = score

            if best is None:
                raise RuntimeError(
                    "Failed to generate an ABAC rule. Try reducing conjunct_count "
                    "or increasing candidate_pool_size."
                )

            clause, source_pos, parent_rule_id, support = best
            seen_clauses.add(clause)
            self.rules.append(
                YFCCABACRule(
                    rule_id=rule_id,
                    labels=tuple(clause),
                    support_size=int(support.size),
                    source_doc_pos=int(source_pos),
                    parent_rule_id=parent_rule_id,
                )
            )

        return self.rules

    def assign_users_to_rules(self) -> dict[int, tuple[int, ...]]:
        if self.user_rule_assignments:
            return self.user_rule_assignments
        rules = self.generate_rules()

        assignments: dict[int, tuple[int, ...]] = {}
        interest_labels: dict[int, tuple[int, ...]] = {}
        rule_count = len(rules)

        # Every rule should induce a non-empty cohort when possible. Each user
        # receives rules_per_user clauses, and authorization is the union of the
        # documents covered by those clauses.
        for index, user_id in enumerate(range(1, self.num_users + 1)):
            base_rule_index = index % rule_count
            selected_indices = {base_rule_index}
            while len(selected_indices) < min(self.rules_per_user, rule_count):
                selected_indices.add(int(self.rng.integers(0, rule_count)))
            selected_rules = tuple(rules[i] for i in sorted(selected_indices))
            assignments[int(user_id)] = tuple(int(rule.rule_id) for rule in selected_rules)
            labels = sorted({label for rule in selected_rules for label in rule.labels})
            interest_labels[int(user_id)] = tuple(int(label) for label in labels)

        self.user_rule_assignments = assignments
        self.user_interest_labels = interest_labels
        return assignments

    def _document_positions_for_user_rules(self, rule_ids: tuple[int, ...], rule_by_id: dict[int, YFCCABACRule]) -> np.ndarray:
        positions = []
        for rule_id in rule_ids:
            positions.append(self._support_positions(rule_by_id[int(rule_id)].labels))
        if not positions:
            return np.empty(0, dtype=np.int64)
        if len(positions) == 1:
            return positions[0]
        return np.unique(np.concatenate(positions))

    def iter_permission_assignments(self):
        assignments = self.assign_users_to_rules()
        rule_by_id = {rule.rule_id: rule for rule in self.generate_rules()}
        for user_id in sorted(assignments):
            for doc_pos in self._document_positions_for_user_rules(assignments[user_id], rule_by_id):
                yield (int(user_id), int(self.document_ids[int(doc_pos)]))

    def summarize_structure(self) -> dict[str, float | int]:
        rules = self.generate_rules()
        assignments = self.assign_users_to_rules()
        rule_by_id = {rule.rule_id: rule for rule in rules}
        user_access_positions = [
            self._document_positions_for_user_rules(assignments[user_id], rule_by_id)
            for user_id in sorted(assignments)
        ]
        user_access_sizes = [int(len(positions)) for positions in user_access_positions]
        permission_assignments = int(sum(user_access_sizes))

        doc_sharing = np.zeros(len(self.document_ids), dtype=np.int32)
        for positions in user_access_positions:
            if len(positions):
                doc_sharing[positions] += 1

        rule_supports = np.asarray([rule.support_size for rule in rules], dtype=np.float64)
        user_access = np.asarray(user_access_sizes, dtype=np.float64)
        nonzero_doc_sharing = doc_sharing[doc_sharing > 0]
        parent_count = sum(1 for rule in rules if rule.parent_rule_id is not None)
        rule_pair_jaccard = self._sample_rule_support_jaccard(rules)

        sharing_hist = Counter(int(value) for value in nonzero_doc_sharing.tolist())
        max_shared_by = max(sharing_hist.keys(), default=0)

        return {
            "users": int(self.num_users),
            "documents": int(len(self.document_ids)),
            "rules": int(len(rules)),
            "rules_per_user": int(self.rules_per_user),
            "conjunct_count": int(self.conjunct_count),
            "metadata_labels": int(len(self.label_to_positions)),
            "target_avg_sharing_degree": float(self.target_avg_sharing_degree),
            "avg_sharing_degree": float(permission_assignments / max(1, len(self.document_ids))),
            "covered_documents": int(np.count_nonzero(doc_sharing)),
            "covered_document_ratio": float(np.count_nonzero(doc_sharing) / max(1, len(self.document_ids))),
            "permission_assignments": int(permission_assignments),
            "avg_user_selectivity": float(user_access.mean() / max(1, len(self.document_ids))),
            "min_user_access": int(user_access.min() if user_access.size else 0),
            "max_user_access": int(user_access.max() if user_access.size else 0),
            "avg_rule_support": float(rule_supports.mean() if rule_supports.size else 0.0),
            "min_rule_support": int(rule_supports.min() if rule_supports.size else 0),
            "max_rule_support": int(rule_supports.max() if rule_supports.size else 0),
            "overlap_generated_rules": int(parent_count),
            "avg_rule_support_jaccard": float(rule_pair_jaccard),
            "max_document_shared_by": int(max_shared_by),
            "single_user_documents": int(sharing_hist.get(1, 0)),
        }

    def _sample_rule_support_jaccard(self, rules: list[YFCCABACRule], max_pairs: int = 1000) -> float:
        if len(rules) < 2:
            return 0.0
        pairs = list(combinations(range(len(rules)), 2))
        if len(pairs) > max_pairs:
            indices = self.rng.choice(len(pairs), size=max_pairs, replace=False)
            pairs = [pairs[int(index)] for index in indices]
        total = 0.0
        for left_idx, right_idx in pairs:
            left = self._support_positions(rules[left_idx].labels)
            right = self._support_positions(rules[right_idx].labels)
            intersection = np.intersect1d(left, right, assume_unique=True).size
            union = len(left) + len(right) - intersection
            total += float(intersection / union) if union else 0.0
        return total / max(1, len(pairs))

    def write_policy_sidecar(self, output_path: str | Path, summary: dict[str, float | int] | None = None) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": "yfcc_multi_attribute_abac",
            "references": [
                "NIST SP 800-162 ABAC definition",
                "Xu and Stoller ABAC policy mining synthetic policy methodology",
                "Curator YFCC100M workload metadata-label setup",
            ],
            "config": {
                "num_users": self.num_users,
                "num_rules": self.num_rules,
                "rules_per_user": self.rules_per_user,
                "conjunct_count": self.conjunct_count,
                "target_avg_sharing_degree": self.target_avg_sharing_degree,
                "overlap_probability": self.overlap_probability,
                "min_user_resource_pairs": self.min_user_resource_pairs,
                "candidate_pool_size": self.candidate_pool_size,
                "seed": self.seed,
            },
            "rules": [asdict(rule) for rule in self.generate_rules()],
            "user_rule_assignments": self.assign_users_to_rules(),
            "user_interest_labels": self.user_interest_labels,
            "summary": summary if summary is not None else self.summarize_structure(),
        }
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, sort_keys=True)
