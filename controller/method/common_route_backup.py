from __future__ import annotations

from dataclasses import dataclass, field
import os
import re

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_QUERY_DATASET_PATH = os.path.join(PROJECT_ROOT, "basic_benchmark", "query_dataset.json")

WORKLOAD_AWARE_PARTITION_TABLE_PREFIX = "workload_documentblocks_partition_"
PLAN_TABLE = "dynamic_partition_current_plan"
PARTITION_TABLE = "dynamic_partition_current_partitions"
PARTITION_DOCUMENT_TABLE = "dynamic_partition_current_partition_documents"


def _parse_vector(raw_value) -> np.ndarray:
    if raw_value is None:
        raise ValueError("Vector value is missing")
    if isinstance(raw_value, np.ndarray):
        vector = raw_value.astype(np.float32, copy=False)
    elif isinstance(raw_value, memoryview):
        vector = np.frombuffer(raw_value, dtype=np.float32)
    elif isinstance(raw_value, (bytes, bytearray)):
        vector = np.frombuffer(raw_value, dtype=np.float32)
    elif isinstance(raw_value, str):
        payload = raw_value.strip().strip("[]")
        if not payload:
            return np.zeros(0, dtype=np.float32)
        vector = np.asarray([float(item) for item in payload.split(",") if item], dtype=np.float32)
    elif hasattr(raw_value, "tolist"):
        vector = np.asarray(raw_value.tolist(), dtype=np.float32)
    else:
        vector = np.asarray(raw_value, dtype=np.float32)
    if vector.ndim != 1:
        vector = vector.ravel()
    return vector.astype(np.float32, copy=False)


def _normalize_vector(vector: np.ndarray) -> np.ndarray:
    array = np.asarray(vector, dtype=np.float32).ravel()
    if array.size == 0:
        return array.astype(np.float32, copy=False)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return array.astype(np.float32, copy=False)
    return (array / norm).astype(np.float32, copy=False)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0:
        return matrix.astype(np.float32, copy=False)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0.0] = 1.0
    return (matrix / norms).astype(np.float32, copy=False)


def _sanitize_partition_id(partition_id: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(partition_id))
    return sanitized.strip("_") or "default"


def get_partition_table_name(partition_id: str) -> str:
    return f"{WORKLOAD_AWARE_PARTITION_TABLE_PREFIX}{_sanitize_partition_id(partition_id)}"


def _weighted_jaccard_from_sets(left, right, *, tenant_weights: dict[int, float]) -> float:
    left_set = set(int(value) for value in left)
    right_set = set(int(value) for value in right)
    union = left_set | right_set
    if not union:
        return 1.0
    intersection = left_set & right_set
    numerator = sum(float(tenant_weights.get(tenant_id, 1.0)) for tenant_id in intersection)
    denominator = sum(float(tenant_weights.get(tenant_id, 1.0)) for tenant_id in union)
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


def _weighted_jaccard_from_dicts(left: dict[int, float], right: dict[int, float]) -> float:
    keys = set(left) | set(right)
    if not keys:
        return 1.0
    numerator = 0.0
    denominator = 0.0
    for key in keys:
        left_value = float(left.get(key, 0.0))
        right_value = float(right.get(key, 0.0))
        numerator += min(left_value, right_value)
        denominator += max(left_value, right_value)
    if denominator <= 0.0:
        return 0.0
    return float(numerator / denominator)


@dataclass(slots=True)
class DocumentAccessRecord:
    document_id: int
    representative_block_id: int
    vector: np.ndarray
    tenant_ids: tuple[int, ...]


@dataclass(slots=True)
class WorkloadQuery:
    tenant_id: int
    query_vector: np.ndarray
    topk: int
    weight: float


@dataclass(slots=True)
class ACLLogicalPattern:
    pattern_id: int
    tenant_ids: tuple[int, ...]
    ordered_tenant_ids: tuple[int, ...]
    document_ids: tuple[int, ...]
    vector_count: int
    document_count: int
    entry_tenant_ids: tuple[int, ...] = ()
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class PrefixDagNode:
    node_id: int
    prefix_tenants: tuple[int, ...]
    children: dict[int, int] = field(default_factory=dict)
    terminal_pattern_ids: set[int] = field(default_factory=set)
    supplemental_pattern_ids: set[int] = field(default_factory=set)
    document_count: int = 0
    terminal_document_count: int = 0


@dataclass(slots=True)
class WorkloadAwarePartition:
    partition_id: str
    table_name: str
    document_ids: tuple[int, ...]
    tenant_ids: tuple[int, ...]
    vector_count: int
    logical_pattern_ids: tuple[int, ...]
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def document_count(self) -> int:
        return len(self.document_ids)


@dataclass(slots=True)
class WorkloadAwarePlan:
    partitions: list[WorkloadAwarePartition]
    logical_patterns: list[ACLLogicalPattern]
    dag_nodes: list[PrefixDagNode]
    tenant_order: tuple[int, ...]
    metadata: dict[str, object] = field(default_factory=dict)
