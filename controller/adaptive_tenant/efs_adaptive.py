"""Adaptive ef_search prediction helpers.

This file keeps the runtime ef prediction API small so online callers can use
it without importing the full planner.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .cost_model import (
    DEFAULT_PARAMETER_FILE,
    DEFAULT_POLLUTION_WEIGHT,
    HoneybeeHNSWParameters,
    load_honeybee_hnsw_parameters,
    predict_adaptive_ef_search,
    predict_honeybee_ef_search,
)


def get_adaptive_ef_search(
    *,
    sel_whole: float,
    topk: int,
    recall_target: Optional[float] = None,
    pollution: float = 0.0,
    pollution_weight: float = DEFAULT_POLLUTION_WEIGHT,
    parameter_file: Optional[str | Path] = None,
    parameters: Optional[HoneybeeHNSWParameters] = None,
) -> float:
    """Return adaptive ef_search using fitted HNSW parameters.

    When ``pollution`` is zero this reduces to Honeybee's original predictor.
    """
    params = parameters or load_honeybee_hnsw_parameters(parameter_file or DEFAULT_PARAMETER_FILE)
    if pollution <= 0:
        return predict_honeybee_ef_search(
            sel_whole=sel_whole,
            topk=topk,
            k=params.k,
            beta=params.beta,
            recall=recall_target,
        )
    return predict_adaptive_ef_search(
        selectivity=sel_whole,
        topk=topk,
        parameters=params,
        recall_target=recall_target,
        pollution=pollution,
        pollution_weight=pollution_weight,
    )
