from __future__ import annotations

from dataclasses import dataclass, field
import os
import re
from typing import Iterable


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DEFAULT_QUERY_DATASET_PATH = os.path.join(PROJECT_ROOT, "basic_benchmark", "query_dataset.json")

VEDA_NODE_TABLE_PREFIX = os.environ.get("VEDA_NODE_TABLE_PREFIX", "veda_documentblocks_node_")
VEDA_PLAN_TABLE = os.environ.get("VEDA_PLAN_TABLE", "veda_current_plan")
VEDA_PATTERN_TABLE = os.environ.get("VEDA_PATTERN_TABLE", "veda_current_patterns")
VEDA_NODE_TABLE = os.environ.get("VEDA_NODE_TABLE", "veda_current_nodes")
VEDA_ROLE_PLAN_TABLE = os.environ.get("VEDA_ROLE_PLAN_TABLE", "veda_current_role_plans")
VEDA_ROUTE_TABLE = os.environ.get("VEDA_ROUTE_TABLE", "veda_current_user_routes")

_PAPER_DEFAULT_COST_A = 0.0821
_PAPER_DEFAULT_COST_B = 0.1159
_PAPER_DEFAULT_COST_C = 2.3110


def _load_shared_cost_defaults() -> tuple[float, float, float, str, str, str]:
    use_shared = str(os.environ.get("VEDA_USE_SHARED_COST_MODEL", "")).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    }
    if not use_shared:
        return (
            _PAPER_DEFAULT_COST_A,
            _PAPER_DEFAULT_COST_B,
            _PAPER_DEFAULT_COST_C,
            "veda_appendix_b_hnsw_cost: a*log2(N+1)+b*ef+c",
            "paper_default",
            "paper",
        )
    try:
        from controller.kmeans.cost_model import DEFAULT_COST_MODEL, cost_model_metadata

        metadata = cost_model_metadata(DEFAULT_COST_MODEL)
        return (
            float(DEFAULT_COST_MODEL.cost_a),
            float(DEFAULT_COST_MODEL.cost_b_graph),
            float(DEFAULT_COST_MODEL.cost_c),
            "veda_hnsw_cost_with_shared_parameters: a*log2(N+1)+b*ef+c",
            str(metadata.get("cost_model_source", DEFAULT_COST_MODEL.source)),
            "shared_parameters",
        )
    except Exception:
        return (
            _PAPER_DEFAULT_COST_A,
            _PAPER_DEFAULT_COST_B,
            _PAPER_DEFAULT_COST_C,
            "veda_appendix_b_hnsw_cost: a*log2(N+1)+b*ef+c",
            "paper_default_after_shared_cost_load_failure",
            "paper",
        )


(
    DEFAULT_COST_A,
    DEFAULT_COST_B,
    DEFAULT_COST_C,
    DEFAULT_COST_FORMULA,
    DEFAULT_COST_SOURCE,
    DEFAULT_COST_STYLE,
) = _load_shared_cost_defaults()


def normalize_int_tuple(values: Iterable[int] | None) -> tuple[int, ...]:
    if not values:
        return tuple()
    return tuple(sorted({int(value) for value in values}))


def normalize_algorithm(value: str | None) -> str:
    normalized = str(value or "effveda").strip().lower().replace("-", "_")
    if normalized in {"eff", "efficient", "eff_veda"}:
        return "effveda"
    if normalized not in {"veda", "effveda"}:
        raise ValueError(f"Unsupported Veda algorithm: {value}")
    return normalized


def _sanitize_identifier_part(value: object) -> str:
    sanitized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value))
    return sanitized.strip("_") or "default"


def get_node_table_name(node_id: str | int, algorithm: str | None = None) -> str:
    algorithm_part = ""
    if algorithm:
        algorithm_part = f"{_sanitize_identifier_part(normalize_algorithm(algorithm))}_"
    return f"{VEDA_NODE_TABLE_PREFIX}{algorithm_part}{_sanitize_identifier_part(node_id)}"


def get_node_table_prefix(algorithm: str | None = None) -> str:
    if algorithm is None:
        return VEDA_NODE_TABLE_PREFIX
    return f"{VEDA_NODE_TABLE_PREFIX}{_sanitize_identifier_part(normalize_algorithm(algorithm))}_"


def role_key(role_ids: Iterable[int]) -> str:
    roles = normalize_int_tuple(role_ids)
    if not roles:
        return "empty"
    return "r_" + "_".join(str(role_id) for role_id in roles)


@dataclass(slots=True)
class VedaPattern:
    pattern_id: int
    role_ids: tuple[int, ...]
    document_ids: tuple[int, ...]
    vector_count: int

    @property
    def document_count(self) -> int:
        return len(self.document_ids)


@dataclass(slots=True)
class VedaNode:
    node_id: str
    role_ids: tuple[int, ...]
    pattern_ids: tuple[int, ...]
    document_ids: tuple[int, ...]
    document_pattern_pairs: tuple[tuple[int, int], ...]
    vector_count: int
    node_kind: str
    table_name: str
    metadata: dict[str, object] = field(default_factory=dict)

    @property
    def document_count(self) -> int:
        return len(self.document_ids)


@dataclass(slots=True)
class VedaRoute:
    user_id: int
    node_id: str
    table_name: str
    route_kind: str
    pattern_ids: tuple[int, ...]
    node_vector_count: int
    accessible_vector_count: int
    impurity_factor: float


@dataclass(slots=True)
class VedaPlan:
    algorithm: str
    patterns: list[VedaPattern]
    nodes: list[VedaNode]
    role_plans: dict[int, tuple[str, ...]]
    user_routes: list[VedaRoute]
    metadata: dict[str, object] = field(default_factory=dict)
