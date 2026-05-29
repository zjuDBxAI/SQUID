from __future__ import annotations

from collections import Counter
import json
import os
from typing import Iterable, Optional

from .common import HistoricalPredicateRecord, normalize_int_tuple


def build_role_position_map(role_ids: Iterable[int]) -> dict[int, int]:
    return {int(role_id): idx for idx, role_id in enumerate(sorted({int(value) for value in role_ids}))}


def roles_to_mask(role_ids: Iterable[int], role_positions: dict[int, int]) -> int:
    mask = 0
    for role_id in normalize_int_tuple(role_ids):
        position = role_positions.get(int(role_id))
        if position is None:
            continue
        mask |= 1 << int(position)
    return mask


def mask_is_subset(lhs_mask: int, rhs_mask: int) -> bool:
    return (int(lhs_mask) & ~int(rhs_mask)) == 0


def masks_intersect(lhs_mask: int, rhs_mask: int) -> bool:
    return (int(lhs_mask) & int(rhs_mask)) != 0


def load_historical_user_ids(
    query_dataset_path: str,
    *,
    historical_filters_percentage: float = 1.0,
    workload_window_size: Optional[int] = None,
) -> list[int]:
    if not os.path.exists(query_dataset_path):
        raise FileNotFoundError(f"Historical query dataset not found: {query_dataset_path}")
    with open(query_dataset_path, "r", encoding="utf-8") as query_file:
        records = json.load(query_file)
    if not isinstance(records, list):
        raise ValueError(f"Expected a JSON list in {query_dataset_path}")

    percentage = max(0.0, min(1.0, float(historical_filters_percentage)))
    limit = int(len(records) * percentage)
    if percentage > 0.0 and limit == 0 and records:
        limit = 1
    selected = records[:limit]
    if workload_window_size is not None and int(workload_window_size) > 0:
        selected = selected[-int(workload_window_size):]

    user_ids: list[int] = []
    for record in selected:
        if not isinstance(record, dict) or "user_id" not in record:
            continue
        user_ids.append(int(record["user_id"]))
    return user_ids


def count_historical_user_queries(user_ids: Iterable[int]) -> Counter[int]:
    counts: Counter[int] = Counter()
    for user_id in user_ids:
        counts[int(user_id)] += 1
    return counts


def build_acl_candidate_records(
    document_records,
    historical_user_queries: Counter[int],
    role_positions: dict[int, int],
    *,
    bitvector_cutoff: int,
) -> list[HistoricalPredicateRecord]:
    grouped_block_counts: dict[tuple[int, ...], int] = {}
    acl_keys_by_user: dict[int, set[tuple[int, ...]]] = {}
    for record in document_records:
        acl_ids = normalize_int_tuple(record.role_ids)
        if not acl_ids:
            continue
        grouped_block_counts[acl_ids] = int(grouped_block_counts.get(acl_ids, 0)) + int(record.block_count)
        for user_id in acl_ids:
            acl_keys_by_user.setdefault(int(user_id), set()).add(acl_ids)

    candidate_keys: set[tuple[int, ...]] = set(grouped_block_counts)
    for user_id, query_count in historical_user_queries.items():
        if int(query_count) > 0:
            candidate_keys.add((int(user_id),))

    def acl_intersection_cardinality(candidate_acl: tuple[int, ...]) -> int:
        matched_acl_keys: set[tuple[int, ...]] = set()
        for user_id in candidate_acl:
            matched_acl_keys.update(acl_keys_by_user.get(int(user_id), set()))
        return int(sum(int(grouped_block_counts[acl_ids]) for acl_ids in matched_acl_keys))

    records: list[HistoricalPredicateRecord] = []
    for acl_ids in sorted(candidate_keys, key=lambda value: (len(value), value)):
        cardinality = acl_intersection_cardinality(acl_ids)
        if cardinality < int(bitvector_cutoff):
            continue
        query_count = 0
        if len(acl_ids) == 1:
            query_count = int(historical_user_queries.get(int(acl_ids[0]), 0))
        records.append(
            HistoricalPredicateRecord(
                user_id=-1,
                role_ids=acl_ids,
                role_mask=roles_to_mask(acl_ids, role_positions),
                query_count=int(query_count),
                cardinality=int(cardinality),
            )
        )

    records.sort(key=lambda record: (int(record.cardinality), int(record.role_mask), record.role_ids))
    return records


def build_historical_role_candidate_records(
    document_records,
    historical_user_queries: Counter[int],
    user_roles: dict[int, tuple[int, ...]],
    role_positions: dict[int, int],
    *,
    bitvector_cutoff: int,
) -> list[HistoricalPredicateRecord]:
    grouped_block_counts: dict[tuple[int, ...], int] = {}
    doc_role_sets_by_role: dict[int, set[tuple[int, ...]]] = {}
    for record in document_records:
        role_ids = normalize_int_tuple(record.role_ids)
        if not role_ids:
            continue
        grouped_block_counts[role_ids] = int(grouped_block_counts.get(role_ids, 0)) + int(record.block_count)
        for role_id in role_ids:
            doc_role_sets_by_role.setdefault(int(role_id), set()).add(role_ids)

    candidate_query_counts: Counter[tuple[int, ...]] = Counter()
    for user_id, query_count in historical_user_queries.items():
        role_ids = normalize_int_tuple(user_roles.get(int(user_id), ()))
        if not role_ids:
            continue
        candidate_query_counts[role_ids] += int(query_count)

    def role_set_cardinality(candidate_role_ids: tuple[int, ...]) -> int:
        matched_role_sets: set[tuple[int, ...]] = set()
        for role_id in candidate_role_ids:
            matched_role_sets.update(doc_role_sets_by_role.get(int(role_id), set()))
        return int(sum(int(grouped_block_counts[doc_role_ids]) for doc_role_ids in matched_role_sets))

    records: list[HistoricalPredicateRecord] = []
    for role_ids, query_count in sorted(candidate_query_counts.items(), key=lambda item: (len(item[0]), item[0])):
        cardinality = role_set_cardinality(role_ids)
        if cardinality < int(bitvector_cutoff):
            continue
        records.append(
            HistoricalPredicateRecord(
                user_id=-1,
                role_ids=role_ids,
                role_mask=roles_to_mask(role_ids, role_positions),
                query_count=int(query_count),
                cardinality=int(cardinality),
            )
        )

    records.sort(key=lambda record: (int(record.cardinality), int(record.role_mask), record.role_ids))
    return records


def tally_historical_predicates(
    user_ids: Iterable[int],
    user_roles: dict[int, tuple[int, ...]],
    role_positions: dict[int, int],
    cardinalities: dict[tuple[int, ...], int],
    *,
    bitvector_cutoff: int,
) -> list[HistoricalPredicateRecord]:
    counts: Counter[tuple[int, ...]] = Counter()
    for user_id in user_ids:
        role_ids = normalize_int_tuple(user_roles.get(int(user_id), ()))
        if not role_ids:
            continue
        counts[role_ids] += 1

    records: list[HistoricalPredicateRecord] = []
    for role_ids, query_count in counts.items():
        cardinality = int(cardinalities.get(role_ids, 0))
        if cardinality < int(bitvector_cutoff):
            continue
        records.append(
            HistoricalPredicateRecord(
                user_id=-1,
                role_ids=role_ids,
                role_mask=roles_to_mask(role_ids, role_positions),
                query_count=int(query_count),
                cardinality=cardinality,
            )
        )

    records.sort(key=lambda record: (int(record.cardinality), int(record.role_mask), record.role_ids))
    return records

