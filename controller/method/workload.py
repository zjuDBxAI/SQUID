from __future__ import annotations

from collections import Counter
import json
import os
from typing import Optional

from .common import DEFAULT_QUERY_DATASET_PATH, WorkloadQuery, _parse_vector


def load_workload_queries(
    *,
    query_dataset_path: Optional[str] = None,
    limit: Optional[int] = None,
) -> tuple[list[WorkloadQuery], dict[int, float]]:
    path = query_dataset_path or DEFAULT_QUERY_DATASET_PATH
    if not os.path.exists(path):
        return [], {}

    with open(path, "r", encoding="utf-8") as file:
        payload = json.load(file)

    queries: list[WorkloadQuery] = []
    tenant_weights: dict[int, float] = Counter()
    items = payload if limit is None else payload[: int(limit)]
    for item in items:
        if "user_id" not in item and "tenant_id" not in item:
            continue
        tenant_id = int(item.get("tenant_id", item.get("user_id")))
        weight = item.get("query_frequency", item.get("frequency", item.get("weight", 1)))
        query_weight = float(weight if weight is not None else 1.0)
        queries.append(
            WorkloadQuery(
                tenant_id=tenant_id,
                query_vector=_parse_vector(item.get("query_vector", [])),
                topk=int(item.get("topk", 10)),
                weight=query_weight,
            )
        )
        tenant_weights[tenant_id] += query_weight
    return queries, {int(tenant_id): float(weight) for tenant_id, weight in tenant_weights.items()}
