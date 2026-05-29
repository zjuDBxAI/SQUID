from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

RESULT_DIR = Path(__file__).resolve().parent / "result" / "protection_overlay_analysis"

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover
    tqdm = None


class ProgressBar:
    def __init__(self, total: int, *, disabled: bool = False) -> None:
        self.total = max(1, int(total))
        self.current = 0
        self.disabled = bool(disabled)
        self._bar = None if self.disabled or tqdm is None else tqdm(total=self.total, unit="step")

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
    rank = float(len(ordered) - 1) * float(percentile)
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
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "max": 0.0,
            "mean": 0.0,
        }
    normalized = [float(value) for value in values]
    return {
        "count": int(len(normalized)),
        "min": float(min(normalized)),
        "p25": _percentile(normalized, 0.25),
        "p50": _percentile(normalized, 0.50),
        "p75": _percentile(normalized, 0.75),
        "p90": _percentile(normalized, 0.90),
        "p95": _percentile(normalized, 0.95),
        "max": float(max(normalized)),
        "mean": float(statistics.fmean(normalized)),
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


def _json_default(value: Any) -> Any:
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return str(value)


def _load_method_state() -> dict[str, Any]:
    from controller.method.storage import (
        get_current_plan_summary,
        load_current_access_overlays,
        load_current_logical_patterns,
        load_current_partitions,
    )

    plan_summary = get_current_plan_summary(refresh=True)
    if plan_summary is None:
        raise RuntimeError("No current method plan found. Build the method plan before running this analysis.")

    partitions = load_current_partitions(refresh=True)
    patterns = load_current_logical_patterns(refresh=True)
    access_overlays = load_current_access_overlays(refresh=True)
    return {
        "plan_summary": plan_summary,
        "partitions": partitions,
        "patterns": patterns,
        "access_overlays": access_overlays,
    }


def _build_access_stats(partitions, patterns) -> dict[str, Any]:
    partition_by_pattern: dict[int, str] = {}
    partition_vector_counts: dict[str, int] = {}
    tenant_pattern_ids: dict[int, set[int]] = defaultdict(set)
    tenant_partition_vectors: dict[int, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    tenant_vector_counts: dict[int, int] = defaultdict(int)

    for partition in partitions:
        partition_id = str(partition.partition_id)
        partition_vector_counts[partition_id] = int(partition.vector_count)
        for pattern_id in partition.logical_pattern_ids:
            partition_by_pattern[int(pattern_id)] = partition_id

    for pattern in patterns:
        pattern_id = int(pattern.pattern_id)
        vector_count = int(pattern.vector_count)
        partition_id = partition_by_pattern.get(pattern_id)
        for tenant_id in pattern.tenant_ids:
            tenant_id = int(tenant_id)
            tenant_pattern_ids[tenant_id].add(pattern_id)
            tenant_vector_counts[tenant_id] += vector_count
            if partition_id is not None:
                tenant_partition_vectors[tenant_id][str(partition_id)] += vector_count

    return {
        "partition_by_pattern": partition_by_pattern,
        "partition_vector_counts": partition_vector_counts,
        "tenant_pattern_ids": {int(k): set(v) for k, v in tenant_pattern_ids.items()},
        "tenant_partition_vectors": {int(k): dict(v) for k, v in tenant_partition_vectors.items()},
        "tenant_vector_counts": {int(k): int(v) for k, v in tenant_vector_counts.items()},
    }


def _normalize_overlay_rows(access_overlays: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[int, dict[str, Any]], dict[tuple[int, str], dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    by_tenant: dict[int, dict[str, Any]] = {}
    by_tenant_partition: dict[tuple[int, str], dict[str, Any]] = {}
    for overlay in access_overlays:
        tenant_id = int(overlay["tenant_id"])
        row = dict(overlay)
        row["tenant_id"] = tenant_id
        row["partition_id"] = str(row.get("partition_id", ""))
        row["partition_ids"] = [str(value) for value in (row.get("partition_ids", []) or [])]
        row["table_name"] = str(row.get("table_name", ""))
        row["vector_count"] = int(row.get("vector_count", 0) or 0)
        row["tenant_vector_count"] = int(row.get("tenant_vector_count", 0) or 0)
        row["covered_partition_count"] = int(row.get("covered_partition_count", len(row["partition_ids"])) or 0)
        row["route_partition_count"] = int(row.get("route_partition_count", row["covered_partition_count"]) or 0)
        row["requires_pattern_filter"] = bool(row.get("requires_pattern_filter", False))
        row["protection_group_id"] = str(row.get("protection_group_id", row.get("signature_id", "")) or "")
        row["selectivity"] = float(row["tenant_vector_count"] / max(row["vector_count"], 1))
        rows.append(row)
        by_tenant[tenant_id] = row
        for partition_id in row["partition_ids"]:
            by_tenant_partition[(tenant_id, str(partition_id))] = row
    return rows, by_tenant, by_tenant_partition


def _tenant_rows(
    *,
    access_stats: dict[str, Any],
    overlay_by_tenant: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    tenant_partition_vectors = access_stats["tenant_partition_vectors"]
    tenant_pattern_ids = access_stats["tenant_pattern_ids"]
    tenant_vector_counts = access_stats["tenant_vector_counts"]
    for tenant_id in sorted(tenant_pattern_ids):
        overlay = overlay_by_tenant.get(int(tenant_id))
        partition_map = tenant_partition_vectors.get(int(tenant_id), {})
        rows.append(
            {
                "user_id": int(tenant_id),
                "protected": bool(overlay is not None),
                "branch_count": int(len(partition_map)),
                "pattern_count": int(len(tenant_pattern_ids.get(int(tenant_id), set()))),
                "vector_count": int(tenant_vector_counts.get(int(tenant_id), 0)),
                "protection_table_name": str(overlay.get("table_name", "")) if overlay else "",
                "protection_group_id": str(overlay.get("protection_group_id", "")) if overlay else "",
                "protection_vector_count": int(overlay.get("vector_count", 0) or 0) if overlay else 0,
                "protection_selectivity": float(overlay.get("selectivity", 0.0) or 0.0) if overlay else 0.0,
                "requires_pattern_filter": bool(overlay.get("requires_pattern_filter", False)) if overlay else False,
                "covered_partition_count": int(overlay.get("covered_partition_count", 0) or 0) if overlay else 0,
            }
        )
    rows.sort(key=lambda row: (-int(row["branch_count"]), -int(row["vector_count"]), int(row["user_id"])))
    return rows


def _group_rows(overlay_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in overlay_rows:
        grouped[str(row["table_name"])].append(row)

    result: list[dict[str, Any]] = []
    for table_name, rows in grouped.items():
        tenant_ids = sorted(int(row["tenant_id"]) for row in rows)
        selectivities = [float(row["selectivity"]) for row in rows]
        first = rows[0]
        result.append(
            {
                "table_name": str(table_name),
                "protection_group_id": str(first.get("protection_group_id", "")),
                "tenant_count": int(len(tenant_ids)),
                "tenant_ids": ";".join(str(value) for value in tenant_ids),
                "vector_count": int(first.get("vector_count", 0) or 0),
                "covered_partition_count": int(first.get("covered_partition_count", 0) or 0),
                "route_partition_count_min": int(min(int(row.get("route_partition_count", 0) or 0) for row in rows)),
                "route_partition_count_max": int(max(int(row.get("route_partition_count", 0) or 0) for row in rows)),
                "selectivity_min": float(min(selectivities) if selectivities else 0.0),
                "selectivity_mean": float(statistics.fmean(selectivities) if selectivities else 0.0),
                "requires_filter_tenant_count": int(sum(1 for row in rows if bool(row.get("requires_pattern_filter", False)))),
                "no_filter_tenant_count": int(sum(1 for row in rows if not bool(row.get("requires_pattern_filter", False)))),
            }
        )
    result.sort(
        key=lambda row: (
            -int(row["route_partition_count_max"]),
            -float(row["selectivity_min"]),
            -int(row["tenant_count"]),
            int(row["vector_count"]),
        )
    )
    return result


def _replay_queries(
    *,
    query_dataset_path: str | None,
    workload_limit: int | None,
    route_limit: int | None,
    overlay_by_tenant_partition: dict[tuple[int, str], dict[str, Any]],
    progress: ProgressBar,
) -> list[dict[str, Any]]:
    try:
        from basic_benchmark import efconfig
        if route_limit is not None:
            efconfig.dynamic_partition_route_limit = int(route_limit)
            efconfig.method_route_limit = int(route_limit)
    except Exception:
        pass

    from controller.method.search import get_tenant_partition_route
    from controller.method.workload import load_workload_queries

    queries, _ = load_workload_queries(query_dataset_path=query_dataset_path, limit=workload_limit)
    rows: list[dict[str, Any]] = []
    progress.total = max(1, len(queries))
    for query_index, query in enumerate(queries):
        tenant_id = int(query.tenant_id)
        route = get_tenant_partition_route(tenant_id, query.query_vector, topk=int(query.topk))
        physical_tables: set[str] = set()
        overlay_tables: set[str] = set()
        base_tables: set[str] = set()
        requires_filter = False
        for candidate in tuple(route.selected_candidates or ()):
            overlay = overlay_by_tenant_partition.get((tenant_id, str(candidate.partition_id)))
            if overlay is None:
                table_name = str(candidate.table_name)
                physical_tables.add(table_name)
                base_tables.add(table_name)
                continue
            table_name = str(overlay.get("table_name", ""))
            physical_tables.add(table_name)
            overlay_tables.add(table_name)
            requires_filter = requires_filter or bool(overlay.get("requires_pattern_filter", False))

        rows.append(
            {
                "query_index": int(query_index),
                "user_id": int(tenant_id),
                "query_weight": float(query.weight),
                "protected_route": bool(overlay_tables),
                "physical_table_count": int(len(physical_tables)),
                "overlay_table_count": int(len(overlay_tables)),
                "base_table_count": int(len(base_tables)),
                "candidate_partition_count": int((route.metadata or {}).get("candidate_partition_count", 0) or 0),
                "configured_route_limit": int(route_limit) if route_limit is not None else 0,
                "route_partition_count": int(route.partition_count),
                "coverage_guard_used": bool((route.metadata or {}).get("route_coverage_guard_used", False)),
                "selected_accessible_vector_coverage": float(
                    (route.metadata or {}).get("selected_accessible_vector_coverage", 0.0) or 0.0
                ),
                "base_selected_accessible_vector_coverage": float(
                    (route.metadata or {}).get("base_selected_accessible_vector_coverage", 0.0) or 0.0
                ),
                "overlay_requires_pattern_filter": bool(requires_filter),
                "physical_tables": ";".join(sorted(physical_tables)),
            }
        )
        if query_index == 0 or (query_index + 1) % 100 == 0 or query_index + 1 == len(queries):
            progress.update(f"replayed {query_index + 1}/{len(queries)} routes")
    return rows


def _query_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    protected_rows = [row for row in rows if bool(row["protected_route"])]
    unprotected_rows = [row for row in rows if not bool(row["protected_route"])]
    return {
        "query_count": int(len(rows)),
        "protected_query_count": int(len(protected_rows)),
        "unprotected_query_count": int(len(unprotected_rows)),
        "protected_physical_table_distribution": _distribution([float(row["physical_table_count"]) for row in protected_rows]),
        "unprotected_physical_table_distribution": _distribution([float(row["physical_table_count"]) for row in unprotected_rows]),
        "all_physical_table_distribution": _distribution([float(row["physical_table_count"]) for row in rows]),
        "coverage_guard_query_share": float(
            sum(1 for row in rows if bool(row["coverage_guard_used"])) / max(len(rows), 1)
        ),
        "protected_requires_filter_query_share": float(
            sum(1 for row in protected_rows if bool(row["overlay_requires_pattern_filter"])) / max(len(protected_rows), 1)
        ),
    }


def _write_markdown(
    path: Path,
    *,
    plan_summary: dict[str, Any],
    tenant_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
    query_summary: dict[str, Any],
) -> None:
    metadata = dict(plan_summary.get("metadata", {}) or {})
    protected_tenants = [row for row in tenant_rows if bool(row["protected"])]
    unprotected_tenants = [row for row in tenant_rows if not bool(row["protected"])]
    fanout_all = _distribution([float(row["branch_count"]) for row in tenant_rows])
    fanout_protected = _distribution([float(row["branch_count"]) for row in protected_tenants])
    fanout_unprotected = _distribution([float(row["branch_count"]) for row in unprotected_tenants])
    selectivity = _distribution([float(row["protection_selectivity"]) for row in protected_tenants])

    lines = [
        "# Protection Overlay Analysis",
        "",
        "## Current Plan",
        "",
        f"- plan_id: {plan_summary.get('plan_id')}",
        f"- partition_count: {plan_summary.get('partition_count')}",
        f"- target_partition_count: {metadata.get('target_partition_count')}",
        f"- protection_overlay_space_ratio: {metadata.get('protection_overlay_space_ratio')}",
        f"- overlay_budget_vectors: {metadata.get('overlay_budget_vectors')}",
        f"- protection_overlay_selected_vectors: {metadata.get('protection_overlay_selected_vectors')}",
        f"- shared_protection_group_count: {metadata.get('shared_protection_group_count')}",
        f"- shared_protection_protected_tenant_count: {metadata.get('shared_protection_protected_tenant_count')}",
        f"- shared_protection_no_filter_mapping_count: {metadata.get('shared_protection_no_filter_mapping_count')}",
        f"- shared_protection_exact_signature_candidate_count: {metadata.get('shared_protection_exact_signature_candidate_count')}",
        "",
        "## Tenant Fanout",
        "",
        "| scope | count | p50 | p75 | p90 | p95 | max | mean |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        "| all | {count} | {p50:.2f} | {p75:.2f} | {p90:.2f} | {p95:.2f} | {max:.2f} | {mean:.2f} |".format(**fanout_all),
        "| protected | {count} | {p50:.2f} | {p75:.2f} | {p90:.2f} | {p95:.2f} | {max:.2f} | {mean:.2f} |".format(**fanout_protected),
        "| unprotected | {count} | {p50:.2f} | {p75:.2f} | {p90:.2f} | {p95:.2f} | {max:.2f} | {mean:.2f} |".format(**fanout_unprotected),
        "",
        "## Protection Quality",
        "",
        f"- protection tables: {len(group_rows)}",
        f"- protected tenants: {len(protected_tenants)}",
        f"- selectivity p50/p10/min: {selectivity['p50']:.4f} / {_percentile([float(row['protection_selectivity']) for row in protected_tenants], 0.10):.4f} / {selectivity['min']:.4f}",
        f"- no-filter protected tenants: {sum(1 for row in protected_tenants if not bool(row['requires_pattern_filter']))}",
        f"- pattern-filter protected tenants: {sum(1 for row in protected_tenants if bool(row['requires_pattern_filter']))}",
        "",
        "## Query Replay",
        "",
        f"- replayed queries: {query_summary['query_count']}",
        f"- protected queries: {query_summary['protected_query_count']}",
        f"- unprotected queries: {query_summary['unprotected_query_count']}",
        f"- all physical table p50/p95: {query_summary['all_physical_table_distribution']['p50']:.2f} / {query_summary['all_physical_table_distribution']['p95']:.2f}",
        f"- protected physical table p50/p95: {query_summary['protected_physical_table_distribution']['p50']:.2f} / {query_summary['protected_physical_table_distribution']['p95']:.2f}",
        f"- unprotected physical table p50/p95: {query_summary['unprotected_physical_table_distribution']['p50']:.2f} / {query_summary['unprotected_physical_table_distribution']['p95']:.2f}",
        f"- coverage_guard_query_share: {query_summary['coverage_guard_query_share']:.4f}",
        f"- protected_requires_filter_query_share: {query_summary['protected_requires_filter_query_share']:.4f}",
        "",
        "## Top High-Fanout Tenants",
        "",
        "| user_id | protected | fanout | vectors | protection_vectors | selectivity | needs_filter |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- |",
    ]
    for row in tenant_rows[:20]:
        lines.append(
            "| {user_id} | {protected} | {branch_count} | {vector_count} | {protection_vector_count} | {protection_selectivity:.4f} | {requires_pattern_filter} |".format(
                **row
            )
        )

    lines.extend(
        [
            "",
            "## Protection Tables",
            "",
            "| table | tenants | vectors | partitions | selectivity_min | no_filter | filter |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in group_rows[:20]:
        lines.append(
            "| `{table_name}` | {tenant_count} | {vector_count} | {covered_partition_count} | {selectivity_min:.4f} | {no_filter_tenant_count} | {requires_filter_tenant_count} |".format(
                **row
            )
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze current method protection overlays and route fanout.")
    parser.add_argument("--output-dir", default=str(RESULT_DIR))
    parser.add_argument("--query-dataset-path", default=None)
    parser.add_argument("--workload-limit", type=int, default=1000)
    parser.add_argument("--route-limit", type=int, default=None)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressBar(total=7, disabled=bool(args.quiet))

    state = _load_method_state()
    progress.update("loaded current method plan")

    access_stats = _build_access_stats(state["partitions"], state["patterns"])
    progress.update("built tenant fanout maps")

    overlay_rows, overlay_by_tenant, overlay_by_tenant_partition = _normalize_overlay_rows(state["access_overlays"])
    tenant_rows = _tenant_rows(access_stats=access_stats, overlay_by_tenant=overlay_by_tenant)
    group_rows = _group_rows(overlay_rows)
    progress.update("computed protection overlay quality")

    query_progress = ProgressBar(total=1, disabled=bool(args.quiet))
    query_rows = _replay_queries(
        query_dataset_path=args.query_dataset_path,
        workload_limit=args.workload_limit,
        route_limit=args.route_limit,
        overlay_by_tenant_partition=overlay_by_tenant_partition,
        progress=query_progress,
    )
    progress.update("replayed workload routes")

    query_summary = _query_summary(query_rows)
    summary = {
        "plan": state["plan_summary"],
        "tenant_fanout_distribution": _distribution([float(row["branch_count"]) for row in tenant_rows]),
        "protected_tenant_fanout_distribution": _distribution(
            [float(row["branch_count"]) for row in tenant_rows if bool(row["protected"])]
        ),
        "unprotected_tenant_fanout_distribution": _distribution(
            [float(row["branch_count"]) for row in tenant_rows if not bool(row["protected"])]
        ),
        "protection_selectivity_distribution": _distribution(
            [float(row["protection_selectivity"]) for row in tenant_rows if bool(row["protected"])]
        ),
        "protection_table_count": int(len(group_rows)),
        "protected_tenant_count": int(sum(1 for row in tenant_rows if bool(row["protected"]))),
        "query_summary": query_summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, default=_json_default),
        encoding="utf-8",
    )
    progress.update("wrote summary.json")

    _write_csv(output_dir / "tenant_fanout.csv", tenant_rows)
    _write_csv(output_dir / "protection_tables.csv", group_rows)
    progress.update("wrote tenant and protection csv")

    _write_csv(output_dir / "query_route_replay.csv", query_rows)
    _write_markdown(
        output_dir / "summary.md",
        plan_summary=state["plan_summary"],
        tenant_rows=tenant_rows,
        group_rows=group_rows,
        query_summary=query_summary,
    )
    progress.update("wrote query replay csv and summary.md")

    print("Protection overlay analysis finished.")
    print(f"Output directory: {output_dir}")
    print(f"Protected tenants: {summary['protected_tenant_count']}")
    print(f"Protection tables: {summary['protection_table_count']}")
    print(f"Replayed queries: {query_summary['query_count']}")


if __name__ == "__main__":
    main()
