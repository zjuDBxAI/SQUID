from __future__ import annotations

"""KMeans/SQUID planner cost model.

The planner uses the KMeans/SQUID latency formula with calibrated wiki latency
coefficients.
The ef term remains adaptive: first use the learned recall model to estimate the
clean base ef required for target recall, then expand it by route selectivity as
base_ef / rho.
"""

from dataclasses import dataclass
from functools import lru_cache
import json
import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_COST_MODEL_PATH = Path(
    os.environ.get(
        "KMEANS_COST_MODEL_JSON",
        PROJECT_ROOT
        / "controller"
        / "kmeans"
        / "train"
        / "result"
        / "logefrho_cost_dense_nnls_model.json",
    )
)
DEFAULT_EFS_MODEL_PATH = Path(
    os.environ.get(
        "KMEANS_EFS_MODEL_JSON",
        PROJECT_ROOT / "controller" / "kmeans" / "train" / "result" / "hnsw_only_main_model.json",
    )
)

# pgvector HNSW defaults from pgvector/src/hnsw.h and hnsw.c.
_PGVECTOR_HNSW_M = 16
_PGVECTOR_HNSW_SCAN_SCALING_FACTOR = 0.55

# PostgreSQL default cost parameters.  These are not trained; they mirror the
# current server defaults used by the optimizer unless the deployment changes
# GUCs such as random_page_cost or cpu_tuple_cost.
_PG_SEQ_PAGE_COST = 1.0
_PG_RANDOM_PAGE_COST = 4.0
_PG_CPU_TUPLE_COST = 0.01
_PG_CPU_INDEX_TUPLE_COST = 0.005
_PG_CPU_OPERATOR_COST = 0.0025
_PG_TUPLES_PER_PAGE_APPROX = 100.0
_PG_VECTOR_DISTANCE_OPS_PER_TUPLE = 128.0

_FALLBACK_EFS_H = 2.22087246219851
_FALLBACK_EFS_LAMBDA0 = 0.07626788754243294
_FALLBACK_EFS_LAMBDA1 = 0.031809305747412114
_FALLBACK_TARGET_RECALL = 0.99
_FALLBACK_TOPK = 10
_FALLBACK_COST_A = 0.05134035703436479
_FALLBACK_COST_B_GRAPH = 0.04059425277225399
_FALLBACK_COST_C = 0.0


@dataclass(frozen=True)
class KMeansCostModel:
    hnsw_m: int
    hnsw_scan_scaling_factor: float
    pg_seq_page_cost: float
    pg_random_page_cost: float
    pg_cpu_tuple_cost: float
    pg_cpu_index_tuple_cost: float
    pg_cpu_operator_cost: float
    pg_tuples_per_page_approx: float
    pg_vector_distance_ops_per_tuple: float
    hnsw_max_ef_search: int
    cost_a: float
    cost_b_graph: float
    cost_c: float
    efs_h: float
    efs_lambda0: float
    efs_lambda1: float
    target_recall: float
    topk: int
    source: str
    efs_source: str


@dataclass(frozen=True)
class PgHnswCostBreakdown:
    cost: float
    ratio: float
    entry_level: int
    layer0_tuples_max: float
    layer0_selectivity: float
    estimated_index_tuples: float
    route_ef: float


def _read_json(path: Path) -> dict[str, object]:
    try:
        if path.is_file():
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
    except Exception:
        return {}
    return {}


@lru_cache(maxsize=1)
def load_latency_cost_model(
    path: str | os.PathLike[str] | None = None,
    efs_path: str | os.PathLike[str] | None = None,
) -> KMeansCostModel:
    candidate = Path(path) if path is not None else DEFAULT_COST_MODEL_PATH
    efs_candidate = Path(efs_path) if efs_path is not None else DEFAULT_EFS_MODEL_PATH

    payload = _read_json(candidate)
    settings = payload.get("settings", {}) if isinstance(payload.get("settings", {}), dict) else {}
    efs_payload = payload.get("efs_model", {}) if isinstance(payload.get("efs_model", {}), dict) else {}

    external_efs = _read_json(efs_candidate)
    if external_efs:
        efs_payload = {**efs_payload, **external_efs}

    return KMeansCostModel(
        hnsw_m=int(settings.get("hnsw_m", _PGVECTOR_HNSW_M)),
        hnsw_scan_scaling_factor=float(settings.get("hnsw_scan_scaling_factor", _PGVECTOR_HNSW_SCAN_SCALING_FACTOR)),
        pg_seq_page_cost=float(settings.get("pg_seq_page_cost", _PG_SEQ_PAGE_COST)),
        pg_random_page_cost=float(settings.get("pg_random_page_cost", _PG_RANDOM_PAGE_COST)),
        pg_cpu_tuple_cost=float(settings.get("pg_cpu_tuple_cost", _PG_CPU_TUPLE_COST)),
        pg_cpu_index_tuple_cost=float(settings.get("pg_cpu_index_tuple_cost", _PG_CPU_INDEX_TUPLE_COST)),
        pg_cpu_operator_cost=float(settings.get("pg_cpu_operator_cost", _PG_CPU_OPERATOR_COST)),
        pg_tuples_per_page_approx=float(settings.get("pg_tuples_per_page_approx", _PG_TUPLES_PER_PAGE_APPROX)),
        pg_vector_distance_ops_per_tuple=float(settings.get("pg_vector_distance_ops_per_tuple", _PG_VECTOR_DISTANCE_OPS_PER_TUPLE)),
        hnsw_max_ef_search=max(1, int(settings.get("max_ef_search", 5000))),
        cost_a=_FALLBACK_COST_A,
        cost_b_graph=_FALLBACK_COST_B_GRAPH,
        cost_c=_FALLBACK_COST_C,
        efs_h=float(efs_payload.get("h", _FALLBACK_EFS_H)),
        efs_lambda0=float(efs_payload.get("lambda0", _FALLBACK_EFS_LAMBDA0)),
        efs_lambda1=float(efs_payload.get("lambda1", _FALLBACK_EFS_LAMBDA1)),
        target_recall=float(os.environ.get("KMEANS_TARGET_RECALL", efs_payload.get("target_recall", settings.get("target_recall", _FALLBACK_TARGET_RECALL)))),
        topk=max(1, int(settings.get("topk", _FALLBACK_TOPK))),
        source="Veda.pdf Appendix B coefficients: a=0.0821,b=0.1159,c=2.3110",
        efs_source=str(efs_candidate),
    )


def load_cost_model(path: str | os.PathLike[str] | None = None) -> tuple[float, float, float]:
    """Backward-compatible tuple API used by old debug scripts."""

    active = load_latency_cost_model(path)
    return float(active.cost_a), float(active.cost_b_graph), float(active.cost_c)


def required_base_ef_star(
    *,
    partition_vectors: int,
    topk: int | None = None,
    target_recall: float | None = None,
    model: KMeansCostModel | None = None,
) -> float:
    active = model or DEFAULT_COST_MODEL
    n_value = max(1, int(partition_vectors))
    k_value = max(1, int(topk if topk is not None else active.topk))
    target = max(1e-6, min(0.999999, float(target_recall if target_recall is not None else active.target_recall)))
    lam_n = max(1e-12, float(active.efs_lambda0) + float(active.efs_lambda1) * math.log1p(float(n_value)))
    x_value = lam_n * ((target / (1.0 - target)) ** (1.0 / max(1e-12, float(active.efs_h))))
    return float(min(float(n_value), max(float(k_value), float(k_value) * float(x_value))))


def _route_ef_for_cost(
    *,
    partition_vectors: int,
    accessible_vectors: int,
    ef_search: int | None,
    topk: int,
    target_recall: float | None,
    use_adaptive_ef: bool,
    model: KMeansCostModel,
) -> float:
    n_value = max(1, int(partition_vectors))
    accessible = max(1, int(accessible_vectors))
    rho = max(1e-12, min(1.0, float(accessible) / float(n_value)))
    if use_adaptive_ef:
        base_ef = required_base_ef_star(
            partition_vectors=n_value,
            topk=max(1, int(topk)),
            target_recall=target_recall,
            model=model,
        )
        return float(min(float(n_value), max(1.0, float(base_ef) / float(rho))))
    return float(min(float(n_value), max(1, int(ef_search if ef_search is not None else topk))))


def pgvector_hnsw_cost_breakdown(
    *,
    partition_vectors: int,
    accessible_vectors: int,
    ef_search: int | None = None,
    topk: int | None = None,
    target_recall: float | None = None,
    use_adaptive_ef: bool = True,
    model: KMeansCostModel | None = None,
) -> PgHnswCostBreakdown:
    active = model or DEFAULT_COST_MODEL
    n_value = max(1, int(partition_vectors))
    accessible = max(1, int(accessible_vectors))
    k_value = max(1, int(topk if topk is not None else active.topk))
    m = max(2, int(active.hnsw_m))
    ef_value = _route_ef_for_cost(
        partition_vectors=n_value,
        accessible_vectors=accessible,
        ef_search=ef_search,
        topk=k_value,
        target_recall=target_recall,
        use_adaptive_ef=bool(use_adaptive_ef),
        model=active,
    )

    log_n = math.log(float(n_value)) if n_value > 1 else 0.0
    log_m = max(1e-12, math.log(float(m)))
    log_ef = math.log(max(1.0, float(ef_value)))
    entry_level = int(log_n / log_m) if n_value > 1 else 0
    layer0_tuples_max = float(2 * int(m)) * float(ef_value)
    layer0_selectivity = (
        float(active.hnsw_scan_scaling_factor)
        * float(log_n)
        / (float(log_m) * (1.0 + float(log_ef)))
        if n_value > 1
        else 1.0
    )
    estimated_index_tuples = float(entry_level * int(m)) + float(layer0_tuples_max) * float(layer0_selectivity)
    ratio = min(1.0, max(0.0, float(estimated_index_tuples) / float(n_value)))

    index_pages = max(1.0, float(n_value) / max(1.0, float(active.pg_tuples_per_page_approx)))
    index_io_cost = float(index_pages) * float(active.pg_random_page_cost)
    index_cpu_cost = float(n_value) * (
        float(active.pg_cpu_index_tuple_cost)
        + float(active.pg_cpu_operator_cost) * max(1.0, float(active.pg_vector_distance_ops_per_tuple))
    )
    generic_index_total_cost = float(index_io_cost) + float(index_cpu_cost)
    startup_cost = float(generic_index_total_cost) * float(ratio)
    total_cost = max(0.0, float(startup_cost))
    return PgHnswCostBreakdown(
        cost=float(total_cost),
        ratio=float(ratio),
        entry_level=int(entry_level),
        layer0_tuples_max=float(layer0_tuples_max),
        layer0_selectivity=float(layer0_selectivity),
        estimated_index_tuples=float(estimated_index_tuples),
        route_ef=float(ef_value),
    )


def estimate_partition_query_cost(
    *,
    partition_vectors: int,
    accessible_vectors: int,
    tenant_weight: float = 1.0,
    ef_search: int | None = None,
    topk: int | None = None,
    target_recall: float | None = None,
    use_adaptive_ef: bool = True,
    model: KMeansCostModel | None = None,
) -> float:
    if int(partition_vectors) <= 0 or int(accessible_vectors) <= 0 or float(tenant_weight) <= 0.0:
        return 0.0

    active = model or DEFAULT_COST_MODEL
    n_value = max(1, int(partition_vectors))
    accessible = max(1, int(accessible_vectors))
    k_value = max(1, int(topk if topk is not None else active.topk))
    route_ef = _route_ef_for_cost(
        partition_vectors=n_value,
        accessible_vectors=accessible,
        ef_search=ef_search,
        topk=k_value,
        target_recall=target_recall,
        use_adaptive_ef=bool(use_adaptive_ef),
        model=active,
    )
    log_n = math.log(float(max(2, n_value)))
    cost_ms = (
        float(active.cost_a) * float(log_n)
        + float(active.cost_b_graph) * float(route_ef)
        + float(active.cost_c)
    )
    return float(tenant_weight) * max(0.0, float(cost_ms))


def cost_model_metadata(model: KMeansCostModel | None = None) -> dict[str, object]:
    active = model or DEFAULT_COST_MODEL
    return {
        "cost_model": "kmeans_formula_with_wiki_stage_constrained_coefficients: cost_ms=a*ln(N)+b*route_ef+c, route_ef=base_ef_from_recall_model/rho",
        "adaptive_ef": True,
        "cost_unit": "milliseconds",
        "cost_a": float(active.cost_a),
        "cost_b_graph": float(active.cost_b_graph),
        "cost_d_filter": 0.0,
        "cost_c": float(active.cost_c),
        "cost_compat_note": "estimate_partition_query_cost uses the KMeans/SQUID latency formula with wiki stage-constrained latency coefficients",
        "hnsw_m": int(active.hnsw_m),
        "hnsw_scan_scaling_factor": float(active.hnsw_scan_scaling_factor),
        "pg_seq_page_cost": float(active.pg_seq_page_cost),
        "pg_random_page_cost": float(active.pg_random_page_cost),
        "pg_cpu_tuple_cost": float(active.pg_cpu_tuple_cost),
        "pg_cpu_index_tuple_cost": float(active.pg_cpu_index_tuple_cost),
        "pg_cpu_operator_cost": float(active.pg_cpu_operator_cost),
        "pg_tuples_per_page_approx": float(active.pg_tuples_per_page_approx),
        "pg_vector_distance_ops_per_tuple": float(active.pg_vector_distance_ops_per_tuple),
        "efs_h": float(active.efs_h),
        "efs_lambda0": float(active.efs_lambda0),
        "efs_lambda1": float(active.efs_lambda1),
        "target_recall": float(active.target_recall),
        "hnsw_max_ef_search": int(active.hnsw_max_ef_search),
        "topk_for_cost": int(active.topk),
        "cost_model_source": str(active.source),
        "efs_model_source": str(active.efs_source),
    }


DEFAULT_COST_MODEL = load_latency_cost_model()
DEFAULT_COST_A = float(DEFAULT_COST_MODEL.cost_a)
DEFAULT_COST_B = float(DEFAULT_COST_MODEL.cost_b_graph)
DEFAULT_COST_C = float(DEFAULT_COST_MODEL.cost_c)
DEFAULT_COST_D_FILTER = 0.0
DEFAULT_SCAN_S1 = 0.0
DEFAULT_SCAN_S2 = 0.0
DEFAULT_COST_TOPK = int(DEFAULT_COST_MODEL.topk)
COST_MODEL_SOURCE = str(DEFAULT_COST_MODEL.source)
EFS_MODEL_SOURCE = str(DEFAULT_COST_MODEL.efs_source)
