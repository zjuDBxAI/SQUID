from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


SIEVE_PARTITION_TABLE_PREFIX = "sieve_documentblocks_partition_"
SIEVE_ROOT_TABLE = "sieve_documentblocks_root"
SIEVE_PLAN_TABLE = "sieve_current_plan"
SIEVE_DOCUMENT_TABLE = "sieve_current_documents"
SIEVE_CANDIDATE_TABLE = "sieve_current_candidates"
SIEVE_PARTITION_TABLE = "sieve_current_partitions"
SIEVE_EDGE_TABLE = "sieve_current_edges"
SIEVE_HASSE_EDGE_TABLE = "sieve_current_hasse_edges"


def normalize_int_tuple(values: Iterable[int] | None) -> tuple[int, ...]:
    if not values:
        return tuple()
    return tuple(sorted({int(value) for value in values}))


def round_to_multiple_of_four(value: float) -> int:
    rounded = int(round(float(value) / 4.0) * 4)
    return max(4, rounded)


@dataclass(slots=True)
class DocumentRoleRecord:
    document_id: int
    role_ids: tuple[int, ...]
    block_count: int


@dataclass(slots=True)
class HistoricalPredicateRecord:
    user_id: int
    role_ids: tuple[int, ...]
    role_mask: int
    query_count: int
    cardinality: int


@dataclass(slots=True)
class SieveCandidate:
    candidate_id: int
    role_ids: tuple[int, ...]
    role_mask: int
    query_count: int
    cardinality: int
    scaled_m: int
    scaled_size: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SievePartition:
    partition_id: str
    candidate_id: int
    partition_kind: str
    table_name: str
    role_ids: tuple[int, ...]
    role_mask: int
    cardinality: int
    vector_count: int
    m: int
    ef_construction: int
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class SievePlan:
    partitions: list[SievePartition]
    candidates: list[SieveCandidate]
    dag_edges: list[tuple[int, int]]
    hasse_edges: list[tuple[str, str]]
    metadata: dict[str, object] = field(default_factory=dict)
