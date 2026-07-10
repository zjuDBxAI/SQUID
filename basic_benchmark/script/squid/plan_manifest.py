"""Stable names for one versioned benchmark materialization.

This module is PostgreSQL-free. Builders use it to obtain table prefixes
before they create any relation.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass


METHODS = ("ours", "honeybee", "veda", "effveda")
POSTGRES_IDENTIFIER_LIMIT = 63


def _identifier(value: object) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9_]+", "_", str(value)).strip("_").lower()
    return normalized or "default"


def memory_label(memory_ratio: float) -> str:
    rendered = f"{float(memory_ratio):.4f}".rstrip("0")
    if rendered.endswith("."):
        rendered += "0"
    return "m" + rendered.replace(".", "p")


def _truncate(prefix: str, suffix: str = "") -> str:
    available = POSTGRES_IDENTIFIER_LIMIT - len(suffix)
    if available <= 0:
        raise ValueError("Suffix exceeds PostgreSQL identifier limit")
    return prefix[:available] + suffix


@dataclass(frozen=True)
class PlanManifest:
    method: str
    memory_ratio: float
    version: str
    table_prefix: str
    route_relation: str
    pattern_relation: str
    plan_relation: str

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def create_manifest(method: str, memory_ratio: float, version: str | int) -> PlanManifest:
    normalized_method = _identifier(method)
    if normalized_method not in METHODS:
        raise ValueError(f"Unsupported method: {method}")
    ratio = float(memory_ratio)
    if ratio < 1.0:
        raise ValueError("memory_ratio must be at least 1.0")

    version_label = _identifier(version)
    table_prefix = _truncate(f"{normalized_method}_{memory_label(ratio)}_v{version_label}")
    return PlanManifest(
        method=normalized_method,
        memory_ratio=ratio,
        version=version_label,
        table_prefix=table_prefix,
        route_relation=_truncate(f"{table_prefix}_routes"),
        pattern_relation=_truncate(f"{table_prefix}_patterns"),
        plan_relation=_truncate(f"{table_prefix}_plan"),
    )
