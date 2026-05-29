from __future__ import annotations

import argparse
import ast
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

RESULT_DIR = Path(__file__).resolve().parent / "result"
DEFAULT_OUTPUT_DIR = RESULT_DIR / "tenant_group_validation"
DEFAULT_GROUP_FILES = (
    Path(__file__).resolve().parent / "tenant_groups.json",
    Path(__file__).resolve().parent / "tenant_groups.csv",
    RESULT_DIR / "tenant_groups.json",
    RESULT_DIR / "tenant_groups.csv",
)

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


@dataclass(slots=True)
class AccessModel:
    source: str
    total_vectors: int
    pattern_vector_counts: dict[int, int]
    pattern_tenants: dict[int, set[int]]
    tenant_patterns: dict[int, set[int]]
    tenant_vector_counts: dict[int, int]
    partition_by_pattern: dict[int, str]
    partition_vector_counts: dict[str, int]
    plan_id: Optional[int] = None
    warning: str = ""


class ProgressBar:
    def __init__(self, total: int, *, disabled: bool = False) -> None:
        self.total = max(1, int(total))
        self.current = 0
        self.disabled = bool(disabled)
        self._bar = None if disabled or tqdm is None else tqdm(total=self.total, unit="step")

    def update(self, message: str) -> None:
        if self.disabled:
            return
        if self._bar is not None:
            self._bar.set_description(str(message))
            self._bar.update(1)
            if self._bar.n >= self.total:
                self._bar.close()
            return

        self.current = min(self.total, self.current + 1)
        width = 32
        filled = int(round(width * self.current / self.total))
        bar = "#" * filled + "-" * (width - filled)
        percent = 100.0 * self.current / self.total
        sys.stdout.write(f"\r[{bar}] {self.current}/{self.total} {percent:5.1f}%  {message}")
        sys.stdout.flush()
        if self.current >= self.total:
            sys.stdout.write("\n")
            sys.stdout.flush()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return float(ordered[0])
    rank = (len(ordered) - 1) * float(percentile)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return float(ordered[lower])
    weight = rank - lower
    return float(ordered[lower] * (1.0 - weight) + ordered[upper] * weight)


def _distribution(values: list[float]) -> dict[str, float]:
    if not values:
        return {
            "count": 0,
            "min": 0.0,
            "p10": 0.0,
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "stdev": 0.0,
        }
    normalized = [float(value) for value in values]
    return {
        "count": int(len(normalized)),
        "min": float(min(normalized)),
        "p10": _percentile(normalized, 0.10),
        "p25": _percentile(normalized, 0.25),
        "p50": _percentile(normalized, 0.50),
        "p75": _percentile(normalized, 0.75),
        "p90": _percentile(normalized, 0.90),
        "p95": _percentile(normalized, 0.95),
        "max": float(max(normalized)),
        "mean": float(statistics.fmean(normalized)),
        "stdev": float(statistics.pstdev(normalized)) if len(normalized) > 1 else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _parse_tenant_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return sorted({int(item) for item in value})
    text = str(value).strip()
    if not text:
        return []
    try:
        parsed = ast.literal_eval(text)
    except Exception:
        parsed = None
    if isinstance(parsed, (list, tuple, set)):
        return sorted({int(item) for item in parsed})
    normalized = (
        text.replace("[", " ")
        .replace("]", " ")
        .replace("(", " ")
        .replace(")", " ")
        .replace(";", " ")
        .replace("|", " ")
        .replace(",", " ")
    )
    return sorted({int(item) for item in normalized.split() if item.strip()})


def _search_cost(table_vectors: int, accessible_vectors: int, *, alpha: float) -> float:
    table_vectors = max(1, int(table_vectors))
    accessible_vectors = max(1, min(int(accessible_vectors), table_vectors))
    selectivity = max(float(accessible_vectors) / float(table_vectors), 1e-9)
    return float(alpha + math.log1p(float(table_vectors)) * math.sqrt(1.0 / selectivity))


def _load_access_model_from_plan() -> Optional[AccessModel]:
    try:
        from controller.method.storage import (
            get_current_plan_summary,
            load_current_logical_patterns,
            load_current_partitions,
        )
    except Exception:
        return None

    try:
        plan_summary = get_current_plan_summary()
        if plan_summary is None:
            return None
        patterns = load_current_logical_patterns()
        partitions = load_current_partitions()
    except Exception:
        return None

    if not patterns:
        return None

    pattern_vector_counts: dict[int, int] = {}
    pattern_tenants: dict[int, set[int]] = {}
    tenant_patterns: dict[int, set[int]] = defaultdict(set)
    tenant_vector_counts: dict[int, int] = defaultdict(int)
    partition_by_pattern: dict[int, str] = {}
    partition_vector_counts: dict[str, int] = {}

    for partition in partitions:
        partition_id = str(partition.partition_id)
        partition_vector_counts[partition_id] = int(partition.vector_count)
        for pattern_id in partition.logical_pattern_ids:
            partition_by_pattern[int(pattern_id)] = partition_id

    for pattern in patterns:
        pattern_id = int(pattern.pattern_id)
        vector_count = int(pattern.vector_count)
        tenants = {int(tenant_id) for tenant_id in pattern.tenant_ids}
        pattern_vector_counts[pattern_id] = vector_count
        pattern_tenants[pattern_id] = tenants
        for tenant_id in tenants:
            tenant_patterns[int(tenant_id)].add(pattern_id)
            tenant_vector_counts[int(tenant_id)] += vector_count

    return AccessModel(
        source="current_plan",
        total_vectors=max(1, int(sum(pattern_vector_counts.values()))),
        pattern_vector_counts=pattern_vector_counts,
        pattern_tenants=pattern_tenants,
        tenant_patterns={int(k): set(v) for k, v in tenant_patterns.items()},
        tenant_vector_counts={int(k): int(v) for k, v in tenant_vector_counts.items()},
        partition_by_pattern=partition_by_pattern,
        partition_vector_counts=partition_vector_counts,
        plan_id=int(plan_summary.get("plan_id", 0) or 0),
    )


def _load_access_model_from_acl_csv() -> AccessModel:
    acl_path = RESULT_DIR / "acl_stats.csv"
    if not acl_path.exists():
        raise FileNotFoundError(f"Missing ACL statistics file: {acl_path}")

    pattern_vector_counts: dict[int, int] = {}
    pattern_tenants: dict[int, set[int]] = {}
    tenant_patterns: dict[int, set[int]] = defaultdict(set)
    tenant_vector_counts: dict[int, int] = defaultdict(int)

    with acl_path.open("r", encoding="utf-8", newline="") as file:
        reader = csv.DictReader(file)
        for index, row in enumerate(reader, start=1):
            pattern_id = int(row.get("acl_id") or index)
            tenants = set(_parse_tenant_list(row.get("tenant_ids")))
            vector_count = int(float(row.get("vector_count") or 0))
            pattern_vector_counts[pattern_id] = vector_count
            pattern_tenants[pattern_id] = tenants
            for tenant_id in tenants:
                tenant_patterns[int(tenant_id)].add(pattern_id)
                tenant_vector_counts[int(tenant_id)] += vector_count

    total_vectors = sum(pattern_vector_counts.values())
    summary_path = RESULT_DIR / "summary.json"
    if summary_path.exists():
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
            total_vectors = int(payload.get("summary", {}).get("block_count", total_vectors) or total_vectors)
        except Exception:
            pass

    return AccessModel(
        source="acl_stats_csv",
        total_vectors=max(1, int(total_vectors)),
        pattern_vector_counts=pattern_vector_counts,
        pattern_tenants=pattern_tenants,
        tenant_patterns={int(k): set(v) for k, v in tenant_patterns.items()},
        tenant_vector_counts={int(k): int(v) for k, v in tenant_vector_counts.items()},
        partition_by_pattern={},
        partition_vector_counts={},
        warning="No current plan metadata was loaded; route fanout replay is unavailable.",
    )


def load_access_model(*, prefer_plan: bool) -> AccessModel:
    if prefer_plan:
        model = _load_access_model_from_plan()
        if model is not None:
            return model
    return _load_access_model_from_acl_csv()


def _find_default_group_file() -> Optional[Path]:
    for path in DEFAULT_GROUP_FILES:
        if path.exists():
            return path
    return None


def _normalize_group_rows(rows: list[dict[str, Any]], all_tenants: set[int]) -> tuple[list[dict[str, Any]], list[str]]:
    groups: list[dict[str, Any]] = []
    warnings: list[str] = []
    seen_membership: dict[int, str] = {}

    for index, row in enumerate(rows, start=1):
        group_id = str(row.get("group_id") or row.get("group") or row.get("cluster_id") or f"g{index}")
        tenants = sorted({int(tenant_id) for tenant_id in row.get("tenant_ids", [])})
        unknown = [tenant_id for tenant_id in tenants if tenant_id not in all_tenants]
        if unknown:
            warnings.append(f"group {group_id} contains {len(unknown)} tenants not present in access data")
        tenants = [tenant_id for tenant_id in tenants if tenant_id in all_tenants]
        if not tenants:
            warnings.append(f"group {group_id} is empty after normalization")
            continue
        for tenant_id in tenants:
            if tenant_id in seen_membership:
                warnings.append(
                    f"tenant {tenant_id} appears in both {seen_membership[tenant_id]} and {group_id}; query validation uses the first group"
                )
            else:
                seen_membership[tenant_id] = group_id
        groups.append({"group_id": group_id, "tenant_ids": tenants})
    return groups, warnings


def _groups_from_json(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict) and "groups" in payload:
        payload = payload["groups"]

    rows: list[dict[str, Any]] = []
    if isinstance(payload, dict):
        for group_id, value in payload.items():
            if isinstance(value, dict):
                tenants = value.get("tenant_ids", value.get("tenants", value.get("user_ids", value.get("users", []))))
            else:
                tenants = value
            rows.append({"group_id": str(group_id), "tenant_ids": _parse_tenant_list(tenants)})
        return rows

    if isinstance(payload, list):
        for index, item in enumerate(payload, start=1):
            if isinstance(item, dict):
                group_id = item.get("group_id", item.get("group", item.get("cluster_id", f"g{index}")))
                tenants = item.get("tenant_ids", item.get("tenants", item.get("user_ids", item.get("users", []))))
                rows.append({"group_id": str(group_id), "tenant_ids": _parse_tenant_list(tenants)})
            else:
                rows.append({"group_id": f"g{index}", "tenant_ids": _parse_tenant_list(item)})
        return rows

    raise ValueError(f"Unsupported JSON group format: {path}")


def _groups_from_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        sample = file.read(4096)
        file.seek(0)
        has_header = csv.Sniffer().has_header(sample) if sample.strip() else True
        if not has_header:
            reader = csv.reader(file)
            grouped: dict[str, list[int]] = defaultdict(list)
            for row in reader:
                if len(row) < 2:
                    continue
                grouped[str(row[0]).strip()].extend(_parse_tenant_list(row[1]))
            return [{"group_id": group_id, "tenant_ids": tenants} for group_id, tenants in grouped.items()]

        reader = csv.DictReader(file)
        field_map = {str(field).lower(): field for field in (reader.fieldnames or [])}
        group_field = (
            field_map.get("group_id")
            or field_map.get("group")
            or field_map.get("cluster_id")
            or field_map.get("cluster")
        )
        tenant_field = (
            field_map.get("tenant_id")
            or field_map.get("user_id")
            or field_map.get("tenant")
            or field_map.get("user")
        )
        tenants_field = (
            field_map.get("tenant_ids")
            or field_map.get("user_ids")
            or field_map.get("tenants")
            or field_map.get("users")
        )
        if group_field is None:
            raise ValueError(f"CSV group file needs a group_id/group/cluster_id column: {path}")
        grouped: dict[str, list[int]] = defaultdict(list)
        for row in reader:
            group_id = str(row.get(group_field, "")).strip()
            if not group_id:
                continue
            if tenant_field is not None:
                grouped[group_id].extend(_parse_tenant_list(row.get(tenant_field)))
            elif tenants_field is not None:
                grouped[group_id].extend(_parse_tenant_list(row.get(tenants_field)))
            else:
                raise ValueError(f"CSV group file needs tenant_id/user_id or tenant_ids/user_ids: {path}")
        return [{"group_id": group_id, "tenant_ids": tenants} for group_id, tenants in grouped.items()]


def _auto_exact_signature_groups(model: AccessModel) -> list[dict[str, Any]]:
    by_signature: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for tenant_id, pattern_ids in model.tenant_patterns.items():
        by_signature[tuple(sorted(pattern_ids))].append(int(tenant_id))
    groups: list[dict[str, Any]] = []
    for index, (_, tenant_ids) in enumerate(
        sorted(by_signature.items(), key=lambda item: (-len(item[1]), item[1])),
        start=1,
    ):
        if len(tenant_ids) <= 1:
            continue
        groups.append({"group_id": f"exact_signature_{index}", "tenant_ids": sorted(tenant_ids)})
    return groups


def load_groups(path: Optional[Path], model: AccessModel) -> tuple[list[dict[str, Any]], str, list[str]]:
    all_tenants = set(model.tenant_patterns)
    if path is None:
        default_path = _find_default_group_file()
        if default_path is not None:
            path = default_path

    if path is None:
        rows = _auto_exact_signature_groups(model)
        groups, warnings = _normalize_group_rows(rows, all_tenants)
        warnings.append(
            "No tenant group file was found; validated auto exact-access-signature groups as a baseline."
        )
        return groups, "auto_exact_access_signature", warnings

    if str(path).strip().lower() in {"auto", "auto-exact", "auto-exact-signature"}:
        rows = _auto_exact_signature_groups(model)
        groups, warnings = _normalize_group_rows(rows, all_tenants)
        return groups, "auto_exact_access_signature", warnings

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Tenant group file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix == ".json":
        rows = _groups_from_json(path)
    elif suffix == ".csv":
        rows = _groups_from_csv(path)
    else:
        raise ValueError(f"Unsupported tenant group file type: {path}")
    groups, warnings = _normalize_group_rows(rows, all_tenants)
    return groups, str(path), warnings


def _weighted_jaccard(left: set[int], right: set[int], pattern_vector_counts: dict[int, int]) -> float:
    union = left | right
    if not union:
        return 1.0
    intersection = left & right
    numerator = sum(int(pattern_vector_counts.get(pattern_id, 0)) for pattern_id in intersection)
    denominator = sum(int(pattern_vector_counts.get(pattern_id, 0)) for pattern_id in union)
    return float(numerator / max(denominator, 1))


def _pairwise_jaccard_summary(
    tenant_ids: list[int],
    model: AccessModel,
    *,
    max_pairs: int,
) -> dict[str, float]:
    if len(tenant_ids) <= 1:
        return {"pair_count": 0, "sampled_pair_count": 0, "mean": 1.0, "min": 1.0, "p10": 1.0}
    pair_total = len(tenant_ids) * (len(tenant_ids) - 1) // 2
    stride = max(1, int(math.ceil(pair_total / max(1, max_pairs))))
    values: list[float] = []
    pair_index = 0
    for left_index, left_tenant in enumerate(tenant_ids):
        left_patterns = model.tenant_patterns.get(int(left_tenant), set())
        for right_tenant in tenant_ids[left_index + 1:]:
            if pair_index % stride == 0:
                right_patterns = model.tenant_patterns.get(int(right_tenant), set())
                values.append(_weighted_jaccard(left_patterns, right_patterns, model.pattern_vector_counts))
            pair_index += 1
    dist = _distribution(values)
    return {
        "pair_count": int(pair_total),
        "sampled_pair_count": int(len(values)),
        "mean": float(dist["mean"]),
        "min": float(dist["min"]),
        "p10": float(dist["p10"]),
    }


def _build_tenant_partition_vectors(model: AccessModel) -> dict[int, dict[str, int]]:
    tenant_partition_vectors: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    if not model.partition_by_pattern:
        return {}
    for pattern_id, tenants in model.pattern_tenants.items():
        partition_id = model.partition_by_pattern.get(int(pattern_id))
        if partition_id is None:
            continue
        vector_count = int(model.pattern_vector_counts.get(int(pattern_id), 0))
        for tenant_id in tenants:
            tenant_partition_vectors[int(tenant_id)][str(partition_id)] += vector_count
    return {int(k): dict(v) for k, v in tenant_partition_vectors.items()}


def evaluate_groups(
    groups: list[dict[str, Any]],
    model: AccessModel,
    *,
    alpha: float,
    max_jaccard_pairs: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    tenant_partition_vectors = _build_tenant_partition_vectors(model)
    group_rows: list[dict[str, Any]] = []
    tenant_rows: list[dict[str, Any]] = []

    for group in groups:
        group_id = str(group["group_id"])
        tenant_ids = [int(tenant_id) for tenant_id in group["tenant_ids"]]
        group_patterns = sorted(
            {
                int(pattern_id)
                for tenant_id in tenant_ids
                for pattern_id in model.tenant_patterns.get(int(tenant_id), set())
            }
        )
        group_vector_count = int(sum(int(model.pattern_vector_counts.get(pattern_id, 0)) for pattern_id in group_patterns))
        per_tenant_vector_sum = int(
            sum(int(model.tenant_vector_counts.get(int(tenant_id), 0)) for tenant_id in tenant_ids)
        )
        selectivities = [
            float(int(model.tenant_vector_counts.get(int(tenant_id), 0)) / max(group_vector_count, 1))
            for tenant_id in tenant_ids
        ]
        jaccard = _pairwise_jaccard_summary(tenant_ids, model, max_pairs=max_jaccard_pairs)

        static_base_cost_sum = 0.0
        static_group_cost_sum = 0.0
        static_gain_sum = 0.0
        static_base_branch_counts: list[int] = []

        for tenant_id in tenant_ids:
            tenant_vector_count = int(model.tenant_vector_counts.get(int(tenant_id), 0))
            tenant_patterns = model.tenant_patterns.get(int(tenant_id), set())
            tenant_partition_map = tenant_partition_vectors.get(int(tenant_id), {})
            if tenant_partition_map:
                static_base_branch_count = int(len(tenant_partition_map))
                static_base_cost = sum(
                    _search_cost(
                        int(model.partition_vector_counts.get(str(partition_id), matched_vectors)),
                        int(matched_vectors),
                        alpha=alpha,
                    )
                    for partition_id, matched_vectors in tenant_partition_map.items()
                )
            else:
                static_base_branch_count = 0
                static_base_cost = 0.0
            static_group_cost = _search_cost(group_vector_count, tenant_vector_count, alpha=alpha)
            static_gain = float(static_base_cost - static_group_cost)
            static_base_cost_sum += static_base_cost
            static_group_cost_sum += static_group_cost
            static_gain_sum += static_gain
            if static_base_branch_count:
                static_base_branch_counts.append(static_base_branch_count)

            tenant_rows.append(
                {
                    "group_id": group_id,
                    "user_id": int(tenant_id),
                    "tenant_vector_count": int(tenant_vector_count),
                    "tenant_pattern_count": int(len(tenant_patterns)),
                    "group_vector_count": int(group_vector_count),
                    "group_pattern_count": int(len(group_patterns)),
                    "selectivity_in_group": float(tenant_vector_count / max(group_vector_count, 1)),
                    "static_base_branch_count": int(static_base_branch_count),
                    "static_group_branch_count": 1,
                    "static_base_cost": float(static_base_cost),
                    "static_group_cost": float(static_group_cost),
                    "static_gain": float(static_gain),
                }
            )

        group_rows.append(
            {
                "group_id": group_id,
                "tenant_count": int(len(tenant_ids)),
                "tenant_ids": ";".join(str(tenant_id) for tenant_id in tenant_ids),
                "group_pattern_count": int(len(group_patterns)),
                "group_vector_count": int(group_vector_count),
                "per_tenant_vector_sum": int(per_tenant_vector_sum),
                "space_saving_vs_per_tenant": float(
                    1.0 - float(group_vector_count) / float(max(per_tenant_vector_sum, 1))
                ),
                "overlay_copy_ratio_to_total_vectors": float(group_vector_count / max(model.total_vectors, 1)),
                "selectivity_mean": float(statistics.fmean(selectivities)) if selectivities else 0.0,
                "selectivity_p10": _percentile(selectivities, 0.10),
                "selectivity_min": min(selectivities) if selectivities else 0.0,
                "pairwise_weighted_jaccard_mean": float(jaccard["mean"]),
                "pairwise_weighted_jaccard_p10": float(jaccard["p10"]),
                "pairwise_weighted_jaccard_min": float(jaccard["min"]),
                "pairwise_count": int(jaccard["pair_count"]),
                "pairwise_sampled_count": int(jaccard["sampled_pair_count"]),
                "static_base_branch_mean": float(statistics.fmean(static_base_branch_counts)) if static_base_branch_counts else 0.0,
                "static_base_branch_p95": _percentile([float(v) for v in static_base_branch_counts], 0.95),
                "static_group_branch_count": 1,
                "static_base_cost_sum": float(static_base_cost_sum),
                "static_group_cost_sum": float(static_group_cost_sum),
                "static_gain_sum": float(static_gain_sum),
            }
        )

    protected_tenants = sorted({int(tenant_id) for group in groups for tenant_id in group["tenant_ids"]})
    summary = {
        "group_count": int(len(groups)),
        "protected_tenant_count": int(len(protected_tenants)),
        "total_overlay_vectors_sum": int(sum(int(row["group_vector_count"]) for row in group_rows)),
        "total_per_tenant_overlay_vectors_sum": int(sum(int(row["per_tenant_vector_sum"]) for row in group_rows)),
        "total_static_base_cost_sum": float(sum(float(row["static_base_cost_sum"]) for row in group_rows)),
        "total_static_group_cost_sum": float(sum(float(row["static_group_cost_sum"]) for row in group_rows)),
        "total_static_gain_sum": float(sum(float(row["static_gain_sum"]) for row in group_rows)),
    }
    total_per_tenant = max(1, int(summary["total_per_tenant_overlay_vectors_sum"]))
    summary["total_space_saving_vs_per_tenant"] = float(
        1.0 - float(summary["total_overlay_vectors_sum"]) / float(total_per_tenant)
    )
    summary["total_overlay_copy_ratio_to_total_vectors"] = float(
        int(summary["total_overlay_vectors_sum"]) / max(int(model.total_vectors), 1)
    )
    return group_rows, tenant_rows, summary


def replay_workload_routes(
    groups: list[dict[str, Any]],
    model: AccessModel,
    *,
    query_dataset_path: Optional[str],
    workload_limit: Optional[int],
    alpha: float,
    progress: ProgressBar,
) -> tuple[list[dict[str, Any]], dict[str, Any], str]:
    if not model.partition_by_pattern:
        return [], {"replayed_query_count": 0}, "route replay skipped because no current plan partitions were loaded"

    try:
        from controller.method.search import get_tenant_partition_route
        from controller.method.workload import load_workload_queries
    except Exception as exc:
        return [], {"replayed_query_count": 0}, f"route replay skipped because imports failed: {exc}"

    try:
        queries, _ = load_workload_queries(query_dataset_path=query_dataset_path, limit=workload_limit)
    except Exception as exc:
        return [], {"replayed_query_count": 0}, f"route replay skipped because workload loading failed: {exc}"

    group_by_tenant: dict[int, dict[str, Any]] = {}
    for group in groups:
        for tenant_id in group["tenant_ids"]:
            group_by_tenant.setdefault(int(tenant_id), group)

    group_vectors: dict[str, int] = {}
    for group in groups:
        group_patterns = {
            int(pattern_id)
            for tenant_id in group["tenant_ids"]
            for pattern_id in model.tenant_patterns.get(int(tenant_id), set())
        }
        group_vectors[str(group["group_id"])] = int(
            sum(int(model.pattern_vector_counts.get(pattern_id, 0)) for pattern_id in group_patterns)
        )

    rows: list[dict[str, Any]] = []
    protected_queries = [query for query in queries if int(query.tenant_id) in group_by_tenant]
    progress.total = max(progress.total, len(protected_queries))
    for query_index, query in enumerate(protected_queries):
        tenant_id = int(query.tenant_id)
        group = group_by_tenant[tenant_id]
        group_id = str(group["group_id"])
        tenant_vector_count = int(model.tenant_vector_counts.get(tenant_id, 0))
        group_vector_count = int(group_vectors.get(group_id, 0))
        try:
            route = get_tenant_partition_route(tenant_id, query.query_vector, topk=int(query.topk))
        except Exception as exc:
            rows.append(
                {
                    "query_index": int(query_index),
                    "user_id": int(tenant_id),
                    "group_id": group_id,
                    "error": str(exc),
                }
            )
            continue

        candidates = tuple(route.selected_candidates or ())
        if candidates:
            base_cost = sum(
                _search_cost(
                    int(candidate.partition_vector_count),
                    int(candidate.matched_vector_count),
                    alpha=alpha,
                )
                for candidate in candidates
            )
        else:
            base_cost = 0.0
        group_cost = _search_cost(group_vector_count, tenant_vector_count, alpha=alpha)
        weighted_gain = float(float(query.weight) * (base_cost - group_cost))
        rows.append(
            {
                "query_index": int(query_index),
                "user_id": int(tenant_id),
                "group_id": group_id,
                "query_weight": float(query.weight),
                "base_branch_count": int(route.partition_count),
                "group_branch_count": 1,
                "base_cost": float(base_cost),
                "group_cost": float(group_cost),
                "weighted_gain": float(weighted_gain),
                "tenant_vector_count": int(tenant_vector_count),
                "group_vector_count": int(group_vector_count),
                "selectivity_in_group": float(tenant_vector_count / max(group_vector_count, 1)),
                "route_coverage_guard_used": bool((route.metadata or {}).get("route_coverage_guard_used", False)),
                "base_selected_accessible_vector_coverage": float(
                    (route.metadata or {}).get("base_selected_accessible_vector_coverage", 0.0) or 0.0
                ),
                "selected_accessible_vector_coverage": float(
                    (route.metadata or {}).get("selected_accessible_vector_coverage", 0.0) or 0.0
                ),
                "candidate_partition_count": int((route.metadata or {}).get("candidate_partition_count", 0) or 0),
                "matched_vector_counts": ";".join(str(v) for v in ((route.metadata or {}).get("matched_vector_counts", []) or [])),
            }
        )
        if query_index == 0 or (query_index + 1) % 50 == 0 or query_index + 1 == len(protected_queries):
            progress.update(f"replayed {query_index + 1}/{len(protected_queries)} protected workload queries")

    valid_rows = [row for row in rows if "error" not in row]
    summary = {
        "replayed_query_count": int(len(valid_rows)),
        "error_query_count": int(len(rows) - len(valid_rows)),
        "base_branch_distribution": _distribution([float(row["base_branch_count"]) for row in valid_rows]),
        "base_cost_distribution": _distribution([float(row["base_cost"]) for row in valid_rows]),
        "group_cost_distribution": _distribution([float(row["group_cost"]) for row in valid_rows]),
        "weighted_gain_sum": float(sum(float(row["weighted_gain"]) for row in valid_rows)),
        "positive_gain_query_share": float(
            sum(1 for row in valid_rows if float(row["weighted_gain"]) > 0.0) / max(len(valid_rows), 1)
        ),
        "coverage_guard_query_share": float(
            sum(1 for row in valid_rows if bool(row["route_coverage_guard_used"])) / max(len(valid_rows), 1)
        ),
    }
    return rows, summary, ""


def write_group_template(path: Path) -> None:
    payload = {
        "groups": [
            {"group_id": "g1", "tenant_ids": [1, 2, 3]},
            {"group_id": "g2", "tenant_ids": [10, 11]},
        ]
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_markdown_report(
    path: Path,
    *,
    model: AccessModel,
    group_source: str,
    warnings: list[str],
    group_rows: list[dict[str, Any]],
    tenant_rows: list[dict[str, Any]],
    static_summary: dict[str, Any],
    query_summary: dict[str, Any],
    route_replay_note: str,
) -> None:
    selectivity_dist = _distribution([float(row["selectivity_in_group"]) for row in tenant_rows])
    group_gain_rows = sorted(group_rows, key=lambda row: float(row["static_gain_sum"]), reverse=True)
    lines = [
        "# Tenant Group Validation",
        "",
        "## Input",
        "",
        f"- Group source: `{group_source}`",
        f"- Access model source: `{model.source}`",
        f"- Current plan id: `{model.plan_id}`",
        f"- Total vectors: {model.total_vectors}",
        f"- Groups: {static_summary['group_count']}",
        f"- Protected tenants: {static_summary['protected_tenant_count']}",
        "",
        "## Space",
        "",
        f"- Shared group overlay vectors: {static_summary['total_overlay_vectors_sum']}",
        f"- Per-tenant overlay vectors: {static_summary['total_per_tenant_overlay_vectors_sum']}",
        f"- Space saving vs per-tenant overlays: {static_summary['total_space_saving_vs_per_tenant']:.4f}",
        f"- Extra copy ratio to base vectors: {static_summary['total_overlay_copy_ratio_to_total_vectors']:.4f}",
        "",
        "## Filter Dilution",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| mean selectivity | {selectivity_dist['mean']:.4f} |",
        f"| p50 selectivity | {selectivity_dist['p50']:.4f} |",
        f"| p10 selectivity | {selectivity_dist['p10']:.4f} |",
        f"| min selectivity | {selectivity_dist['min']:.4f} |",
        "",
        "## Static Cost Model",
        "",
        f"- Static base cost sum: {static_summary['total_static_base_cost_sum']:.4f}",
        f"- Static group cost sum: {static_summary['total_static_group_cost_sum']:.4f}",
        f"- Static estimated gain: {static_summary['total_static_gain_sum']:.4f}",
    ]

    if query_summary.get("replayed_query_count", 0):
        branch_dist = query_summary["base_branch_distribution"]
        lines.extend(
            [
                "",
                "## Workload Route Replay",
                "",
                f"- Replayed protected queries: {query_summary['replayed_query_count']}",
                f"- Base fanout mean: {branch_dist['mean']:.4f}",
                f"- Base fanout p95: {branch_dist['p95']:.4f}",
                f"- Group overlay fanout: 1.0000",
                f"- Weighted query gain sum: {query_summary['weighted_gain_sum']:.4f}",
                f"- Positive-gain query share: {query_summary['positive_gain_query_share']:.4f}",
                f"- Coverage-guard query share in base route: {query_summary['coverage_guard_query_share']:.4f}",
            ]
        )
    else:
        lines.extend(["", "## Workload Route Replay", "", f"- {route_replay_note or 'not run'}"])

    lines.extend(
        [
            "",
            "## Top Groups By Static Gain",
            "",
            "| group_id | tenants | vectors | saving | selectivity_mean | static_gain |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in group_gain_rows[:20]:
        lines.append(
            "| {group_id} | {tenant_count} | {group_vector_count} | {space_saving_vs_per_tenant:.4f} | {selectivity_mean:.4f} | {static_gain_sum:.4f} |".format(
                **row
            )
        )

    if warnings or model.warning:
        lines.extend(["", "## Warnings", ""])
        if model.warning:
            lines.append(f"- {model.warning}")
        for warning in warnings[:50]:
            lines.append(f"- {warning}")

    lines.extend(
        [
            "",
            "## Output Files",
            "",
            "- `summary.json`",
            "- `group_metrics.csv`",
            "- `tenant_metrics.csv`",
            "- `query_metrics.csv`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate tenant groups for shared protection overlays.")
    parser.add_argument(
        "--groups",
        default=None,
        help=(
            "Path to tenant group JSON/CSV. If omitted, the script tries debug/tenant_groups.* "
            "and then falls back to auto exact-access-signature groups."
        ),
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--query-dataset-path", default=None)
    parser.add_argument("--workload-limit", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=1.0, help="Fixed cost for one physical table access.")
    parser.add_argument("--prefer-plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--route-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-jaccard-pairs", type=int, default=20000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressBar(total=8, disabled=bool(args.quiet))

    model = load_access_model(prefer_plan=bool(args.prefer_plan))
    progress.update("loaded access model")

    group_path = Path(args.groups) if args.groups and str(args.groups).lower() not in {"auto", "auto-exact", "auto-exact-signature"} else None
    if args.groups and group_path is None:
        groups, group_source, warnings = load_groups(Path(str(args.groups)), model)
    else:
        groups, group_source, warnings = load_groups(group_path, model)
    progress.update("loaded tenant groups")

    write_group_template(output_dir / "tenant_groups_template.json")
    if not groups:
        raise RuntimeError("No tenant groups were loaded; see tenant_groups_template.json for the expected format.")

    group_rows, tenant_rows, static_summary = evaluate_groups(
        groups,
        model,
        alpha=float(args.alpha),
        max_jaccard_pairs=int(args.max_jaccard_pairs),
    )
    progress.update("evaluated group space and static cost")

    query_rows: list[dict[str, Any]] = []
    query_summary: dict[str, Any] = {"replayed_query_count": 0}
    route_replay_note = "disabled"
    if bool(args.route_replay):
        route_progress = ProgressBar(total=1, disabled=bool(args.quiet))
        query_rows, query_summary, route_replay_note = replay_workload_routes(
            groups,
            model,
            query_dataset_path=args.query_dataset_path,
            workload_limit=args.workload_limit,
            alpha=float(args.alpha),
            progress=route_progress,
        )
    progress.update("replayed workload routes")

    summary_payload = {
        "group_source": group_source,
        "warnings": warnings,
        "access_model": {
            "source": model.source,
            "plan_id": model.plan_id,
            "total_vectors": model.total_vectors,
            "tenant_count": len(model.tenant_patterns),
            "pattern_count": len(model.pattern_vector_counts),
            "partition_count": len(model.partition_vector_counts),
            "warning": model.warning,
        },
        "static_summary": static_summary,
        "query_summary": query_summary,
        "route_replay_note": route_replay_note,
    }

    (output_dir / "summary.json").write_text(
        json.dumps(summary_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    progress.update("wrote summary.json")
    _write_csv(output_dir / "group_metrics.csv", group_rows)
    progress.update("wrote group_metrics.csv")
    _write_csv(output_dir / "tenant_metrics.csv", tenant_rows)
    _write_csv(output_dir / "query_metrics.csv", query_rows)
    write_markdown_report(
        output_dir / "summary.md",
        model=model,
        group_source=group_source,
        warnings=warnings,
        group_rows=group_rows,
        tenant_rows=tenant_rows,
        static_summary=static_summary,
        query_summary=query_summary,
        route_replay_note=route_replay_note,
    )
    progress.update("wrote tenant/query csv and summary.md")

    print("Tenant group validation finished.")
    print(f"Output directory: {output_dir}")
    print(f"Group source: {group_source}")
    print(f"Groups: {static_summary['group_count']}")
    print(f"Protected tenants: {static_summary['protected_tenant_count']}")
    print(f"Space saving vs per-tenant overlays: {static_summary['total_space_saving_vs_per_tenant']:.4f}")
    print(f"Static estimated gain: {static_summary['total_static_gain_sum']:.4f}")
    if query_summary.get("replayed_query_count", 0):
        print(f"Replayed queries: {query_summary['replayed_query_count']}")
        print(f"Weighted query gain: {query_summary['weighted_gain_sum']:.4f}")


if __name__ == "__main__":
    main()
