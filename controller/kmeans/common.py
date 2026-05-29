from __future__ import annotations

from dataclasses import dataclass, field
import os
import re

import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DEFAULT_QUERY_DATASET_PATH = os.path.join(PROJECT_ROOT, "basic_benchmark", "query_dataset.json")

KMEANS_PARTITION_TABLE_PREFIX = "kmeans_documentblocks_partition_"
PLAN_TABLE = "kmeans_current_plan"
PARTITION_TABLE = "kmeans_current_partitions"
ROUTE_TABLE = "kmeans_current_routes"
PATTERN_TABLE = "kmeans_current_patterns"


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


def _sanitize_partition_id(partition_id: str) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(partition_id))
    return sanitized.strip("_") or "default"


def get_partition_table_name(cluster_id: int | str) -> str:
    return f"{KMEANS_PARTITION_TABLE_PREFIX}{_sanitize_partition_id(str(cluster_id))}"


@dataclass(slots=True)
class ACLPattern:
    pattern_id: int
    tenant_ids: tuple[int, ...]
    document_ids: tuple[int, ...]
    vector_count: int
    document_count: int
    weight: float
    score: float = 0.0
    zone: str = "private"


@dataclass(slots=True)
class KMeansPartition:
    partition_id: str
    cluster_id: int
    partition_kind: str
    table_name: str
    tenant_ids: tuple[int, ...]
    pattern_ids: tuple[int, ...]
    document_ids: tuple[int, ...]
    document_pattern_pairs: tuple[tuple[int, int], ...]
    vector_count: int
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def document_count(self) -> int:
        return len(self.document_ids)


@dataclass(slots=True)
class TenantRoute:
    tenant_id: int
    partition_id: str
    table_name: str
    route_kind: str
    cluster_id: int
    pattern_ids: tuple[int, ...]
    partition_vector_count: int = 0
    accessible_vector_count: int = 0


@dataclass(slots=True)
class KMeansPlan:
    partitions: list[KMeansPartition]
    tenant_routes: list[TenantRoute]
    tenant_to_cluster: dict[int, int]
    patterns: list[ACLPattern]
    metadata: dict[str, object] = field(default_factory=dict)
