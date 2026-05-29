from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEBUG_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
if str(DEBUG_DIR) not in sys.path:
    sys.path.append(str(DEBUG_DIR))

from validate_tenant_groups import (  # noqa: E402
    AccessModel,
    ProgressBar,
    _build_tenant_partition_vectors,
    _distribution,
    _search_cost,
    _write_csv,
    evaluate_groups,
    load_access_model,
    replay_workload_routes,
    write_markdown_report,
)


RESULT_ROOT = Path(__file__).resolve().parent / "result"


@dataclass(slots=True)
class TenantSummary:
    tenant_id: int
    pattern_ids: frozenset[int]
    vector_count: int
    base_branch_count: int
    base_cost: float
    singleton_cost: float
    singleton_gain: float


@dataclass(slots=True)
class CandidateGroup:
    group_id: str
    group_type: str
    tenant_ids: tuple[int, ...]
    group_vector_count: int
    per_tenant_vector_sum: int
    static_gain: float
    benefit_density: float
    space_saving_vs_per_tenant: float
    selectivity_mean: float
    selectivity_min: float


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _candidate_to_row(candidate: CandidateGroup, *, selected: bool) -> dict[str, Any]:
    return {
        "selected": bool(selected),
        "group_id": candidate.group_id,
        "group_type": candidate.group_type,
        "tenant_count": int(len(candidate.tenant_ids)),
        "tenant_ids": ";".join(str(tenant_id) for tenant_id in candidate.tenant_ids),
        "group_vector_count": int(candidate.group_vector_count),
        "per_tenant_vector_sum": int(candidate.per_tenant_vector_sum),
        "static_gain": float(candidate.static_gain),
        "benefit_density": float(candidate.benefit_density),
        "space_saving_vs_per_tenant": float(candidate.space_saving_vs_per_tenant),
        "selectivity_mean": float(candidate.selectivity_mean),
        "selectivity_min": float(candidate.selectivity_min),
    }


def _weighted_intersection_count(
    left: frozenset[int],
    right: frozenset[int],
    pattern_vector_counts: dict[int, int],
) -> int:
    if len(left) > len(right):
        left, right = right, left
    return int(sum(int(pattern_vector_counts.get(pattern_id, 0)) for pattern_id in left if pattern_id in right))


def _group_vector_count(
    tenant_ids: tuple[int, ...],
    tenant_summaries: dict[int, TenantSummary],
    pattern_vector_counts: dict[int, int],
) -> int:
    patterns: set[int] = set()
    for tenant_id in tenant_ids:
        patterns.update(tenant_summaries[int(tenant_id)].pattern_ids)
    return int(sum(int(pattern_vector_counts.get(pattern_id, 0)) for pattern_id in patterns))


def build_tenant_summaries(model: AccessModel, *, alpha: float) -> dict[int, TenantSummary]:
    tenant_partition_vectors = _build_tenant_partition_vectors(model)
    summaries: dict[int, TenantSummary] = {}
    for tenant_id in sorted(model.tenant_patterns):
        pattern_ids = frozenset(int(pattern_id) for pattern_id in model.tenant_patterns[int(tenant_id)])
        vector_count = int(model.tenant_vector_counts.get(int(tenant_id), 0))
        partition_map = tenant_partition_vectors.get(int(tenant_id), {})
        base_cost = 0.0
        if partition_map:
            base_cost = sum(
                _search_cost(
                    int(model.partition_vector_counts.get(str(partition_id), matched_vectors)),
                    int(matched_vectors),
                    alpha=alpha,
                )
                for partition_id, matched_vectors in partition_map.items()
            )
        singleton_cost = _search_cost(vector_count, vector_count, alpha=alpha)
        summaries[int(tenant_id)] = TenantSummary(
            tenant_id=int(tenant_id),
            pattern_ids=pattern_ids,
            vector_count=int(vector_count),
            base_branch_count=int(len(partition_map)),
            base_cost=float(base_cost),
            singleton_cost=float(singleton_cost),
            singleton_gain=float(base_cost - singleton_cost),
        )
    return summaries


def make_candidate(
    *,
    group_id: str,
    group_type: str,
    tenant_ids: tuple[int, ...],
    group_vector_count: int,
    tenant_summaries: dict[int, TenantSummary],
    alpha: float,
) -> Optional[CandidateGroup]:
    tenant_ids = tuple(sorted({int(tenant_id) for tenant_id in tenant_ids}))
    if not tenant_ids:
        return None

    per_tenant_vector_sum = int(sum(tenant_summaries[tenant_id].vector_count for tenant_id in tenant_ids))
    if group_vector_count <= 0 or per_tenant_vector_sum <= 0:
        return None

    base_cost_sum = float(sum(tenant_summaries[tenant_id].base_cost for tenant_id in tenant_ids))
    group_cost_sum = float(
        sum(
            _search_cost(
                int(group_vector_count),
                int(tenant_summaries[tenant_id].vector_count),
                alpha=alpha,
            )
            for tenant_id in tenant_ids
        )
    )
    static_gain = float(base_cost_sum - group_cost_sum)
    if static_gain <= 0.0:
        return None

    selectivities = [
        float(tenant_summaries[tenant_id].vector_count / max(int(group_vector_count), 1))
        for tenant_id in tenant_ids
    ]
    return CandidateGroup(
        group_id=str(group_id),
        group_type=str(group_type),
        tenant_ids=tenant_ids,
        group_vector_count=int(group_vector_count),
        per_tenant_vector_sum=int(per_tenant_vector_sum),
        static_gain=float(static_gain),
        benefit_density=float(static_gain / max(int(group_vector_count), 1)),
        space_saving_vs_per_tenant=float(
            1.0 - float(group_vector_count) / float(max(per_tenant_vector_sum, 1))
        ),
        selectivity_mean=float(statistics.fmean(selectivities)) if selectivities else 0.0,
        selectivity_min=float(min(selectivities)) if selectivities else 0.0,
    )


def build_exact_signature_candidates(
    model: AccessModel,
    tenant_summaries: dict[int, TenantSummary],
    *,
    alpha: float,
) -> list[CandidateGroup]:
    by_signature: dict[frozenset[int], list[int]] = defaultdict(list)
    for tenant_id, summary in tenant_summaries.items():
        by_signature[summary.pattern_ids].append(int(tenant_id))

    candidates: list[CandidateGroup] = []
    for index, (signature, tenant_ids) in enumerate(
        sorted(by_signature.items(), key=lambda item: (-len(item[1]), item[1])),
        start=1,
    ):
        if len(tenant_ids) <= 1:
            continue
        vector_count = int(sum(int(model.pattern_vector_counts.get(pattern_id, 0)) for pattern_id in signature))
        candidate = make_candidate(
            group_id=f"v1_exact_{index:04d}",
            group_type="exact_signature",
            tenant_ids=tuple(int(tenant_id) for tenant_id in tenant_ids),
            group_vector_count=int(vector_count),
            tenant_summaries=tenant_summaries,
            alpha=alpha,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def build_pair_candidates(
    model: AccessModel,
    tenant_summaries: dict[int, TenantSummary],
    *,
    alpha: float,
    max_candidates: int,
    progress: ProgressBar,
) -> list[CandidateGroup]:
    import heapq

    tenants = sorted(tenant_summaries)
    heap: list[tuple[float, float, int, int, int, CandidateGroup]] = []
    pair_count = len(tenants) * (len(tenants) - 1) // 2
    report_every = max(1, pair_count // 20)
    visited = 0

    for left_index, left_tenant in enumerate(tenants):
        left = tenant_summaries[int(left_tenant)]
        for right_tenant in tenants[left_index + 1:]:
            right = tenant_summaries[int(right_tenant)]
            visited += 1
            if left.pattern_ids == right.pattern_ids:
                continue
            intersection_vectors = _weighted_intersection_count(
                left.pattern_ids,
                right.pattern_ids,
                model.pattern_vector_counts,
            )
            union_vectors = int(left.vector_count + right.vector_count - intersection_vectors)
            candidate = make_candidate(
                group_id=f"v1_pair_{int(left_tenant)}_{int(right_tenant)}",
                group_type="pair_union",
                tenant_ids=(int(left_tenant), int(right_tenant)),
                group_vector_count=int(union_vectors),
                tenant_summaries=tenant_summaries,
                alpha=alpha,
            )
            if candidate is not None:
                item = (
                    float(candidate.benefit_density),
                    float(candidate.static_gain),
                    -int(candidate.group_vector_count),
                    -int(left_tenant),
                    -int(right_tenant),
                    candidate,
                )
                if len(heap) < int(max_candidates):
                    heapq.heappush(heap, item)
                elif item > heap[0]:
                    heapq.heapreplace(heap, item)
            if visited % report_every == 0:
                progress.update(f"generated pair candidates {visited}/{pair_count}")

    progress.update(f"generated pair candidates {pair_count}/{pair_count}")
    return [item[-1] for item in sorted(heap, key=lambda item: item[:5], reverse=True)]


def build_singleton_candidates(
    tenant_summaries: dict[int, TenantSummary],
    *,
    alpha: float,
) -> list[CandidateGroup]:
    candidates: list[CandidateGroup] = []
    for tenant_id, summary in tenant_summaries.items():
        candidate = make_candidate(
            group_id=f"v1_single_{int(tenant_id)}",
            group_type="single_tenant",
            tenant_ids=(int(tenant_id),),
            group_vector_count=int(summary.vector_count),
            tenant_summaries=tenant_summaries,
            alpha=alpha,
        )
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def select_groups_under_budget(
    candidates: list[CandidateGroup],
    *,
    budget_vectors: int,
) -> list[CandidateGroup]:
    selected: list[CandidateGroup] = []
    used_tenants: set[int] = set()
    used_vectors = 0

    ordered = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.benefit_density),
            -float(candidate.static_gain),
            int(candidate.group_vector_count),
            str(candidate.group_type),
            str(candidate.group_id),
        ),
    )
    for candidate in ordered:
        if used_vectors + int(candidate.group_vector_count) > int(budget_vectors):
            continue
        if any(int(tenant_id) in used_tenants for tenant_id in candidate.tenant_ids):
            continue
        selected.append(candidate)
        used_vectors += int(candidate.group_vector_count)
        used_tenants.update(int(tenant_id) for tenant_id in candidate.tenant_ids)
    return selected


def _selected_groups_payload(selected: list[CandidateGroup]) -> dict[str, Any]:
    return {
        "groups": [
            {
                "group_id": candidate.group_id,
                "tenant_ids": [int(tenant_id) for tenant_id in candidate.tenant_ids],
                "group_type": candidate.group_type,
                "group_vector_count": int(candidate.group_vector_count),
                "static_gain": float(candidate.static_gain),
                "benefit_density": float(candidate.benefit_density),
            }
            for candidate in selected
        ]
    }


def _write_demo_report(
    path: Path,
    *,
    model: AccessModel,
    overlay_space_ratio: float,
    budget_vectors: int,
    candidate_count: int,
    selected: list[CandidateGroup],
    validation_summary: dict[str, Any],
) -> None:
    group_type_counts: dict[str, int] = defaultdict(int)
    for candidate in selected:
        group_type_counts[str(candidate.group_type)] += 1

    lines = [
        "# V1 Shared Protection Group Demo",
        "",
        "## Method",
        "",
        "V1 generates shared protection overlay groups without a similarity threshold.",
        "",
        "1. Build exact-access-signature candidates.",
        "2. Build all positive-gain two-tenant union candidates.",
        "3. Add positive-gain single-tenant candidates.",
        "4. Select non-overlapping candidates by benefit density under the overlay space budget.",
        "5. Validate selected groups with the shared protection overlay cost model and workload route replay.",
        "",
        "## Inputs",
        "",
        f"- Access model source: `{model.source}`",
        f"- Current plan id: `{model.plan_id}`",
        f"- Total vectors: {model.total_vectors}",
        f"- Overlay space ratio: {overlay_space_ratio:.4f}",
        f"- Budget vectors: {budget_vectors}",
        f"- Candidate groups kept: {candidate_count}",
        "",
        "## Selected Groups",
        "",
        f"- Selected groups: {len(selected)}",
        f"- Protected tenants: {len({tenant_id for candidate in selected for tenant_id in candidate.tenant_ids})}",
        f"- Selected overlay vectors: {sum(candidate.group_vector_count for candidate in selected)}",
        "",
        "| group_type | count |",
        "| --- | ---: |",
    ]
    for group_type, count in sorted(group_type_counts.items()):
        lines.append(f"| {group_type} | {count} |")

    static_summary = validation_summary.get("static_summary", {})
    query_summary = validation_summary.get("query_summary", {})
    lines.extend(
        [
            "",
            "## Validation Summary",
            "",
            f"- Space saving vs per-tenant overlays: {float(static_summary.get('total_space_saving_vs_per_tenant', 0.0)):.4f}",
            f"- Extra copy ratio to base vectors: {float(static_summary.get('total_overlay_copy_ratio_to_total_vectors', 0.0)):.4f}",
            f"- Static estimated gain: {float(static_summary.get('total_static_gain_sum', 0.0)):.4f}",
        ]
    )
    if int(query_summary.get("replayed_query_count", 0) or 0):
        branch_dist = query_summary.get("base_branch_distribution", {})
        lines.extend(
            [
                f"- Replayed protected queries: {int(query_summary.get('replayed_query_count', 0) or 0)}",
                f"- Base fanout mean: {float(branch_dist.get('mean', 0.0)):.4f}",
                f"- Base fanout p95: {float(branch_dist.get('p95', 0.0)):.4f}",
                "- Group overlay fanout: 1.0000",
                f"- Weighted query gain: {float(query_summary.get('weighted_gain_sum', 0.0)):.4f}",
                f"- Positive-gain query share: {float(query_summary.get('positive_gain_query_share', 0.0)):.4f}",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_demo(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    progress = ProgressBar(total=30, disabled=bool(args.quiet))

    model = load_access_model(prefer_plan=bool(args.prefer_plan))
    progress.update("loaded access model")
    tenant_summaries = build_tenant_summaries(model, alpha=float(args.alpha))
    progress.update("built tenant summaries")

    exact_candidates = build_exact_signature_candidates(
        model,
        tenant_summaries,
        alpha=float(args.alpha),
    )
    progress.update("built exact-signature candidates")

    pair_candidates = build_pair_candidates(
        model,
        tenant_summaries,
        alpha=float(args.alpha),
        max_candidates=int(args.max_pair_candidates),
        progress=progress,
    )
    singleton_candidates = build_singleton_candidates(
        tenant_summaries,
        alpha=float(args.alpha),
    )
    progress.update("built singleton candidates")

    candidates = exact_candidates + pair_candidates + singleton_candidates
    budget_vectors = int(math.floor(float(args.overlay_space_ratio) * float(model.total_vectors)))
    selected = select_groups_under_budget(candidates, budget_vectors=budget_vectors)
    progress.update("selected groups under budget")

    selected_payload = _selected_groups_payload(selected)
    _write_json(output_dir / "tenant_groups_v1.json", selected_payload)
    _write_csv(
        output_dir / "selected_groups.csv",
        [_candidate_to_row(candidate, selected=True) for candidate in selected],
    )
    selected_ids = {candidate.group_id for candidate in selected}
    top_candidates = sorted(
        candidates,
        key=lambda candidate: (
            -float(candidate.benefit_density),
            -float(candidate.static_gain),
            int(candidate.group_vector_count),
            str(candidate.group_id),
        ),
    )[: int(args.candidate_csv_limit)]
    _write_csv(
        output_dir / "candidate_groups_top.csv",
        [_candidate_to_row(candidate, selected=candidate.group_id in selected_ids) for candidate in top_candidates],
    )
    progress.update("wrote selected and candidate groups")

    groups = [
        {
            "group_id": candidate.group_id,
            "tenant_ids": [int(tenant_id) for tenant_id in candidate.tenant_ids],
        }
        for candidate in selected
    ]
    group_rows, tenant_rows, static_summary = evaluate_groups(
        groups,
        model,
        alpha=float(args.alpha),
        max_jaccard_pairs=int(args.max_jaccard_pairs),
    )
    progress.update("validated selected groups")

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

    validation_dir = output_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(validation_dir / "group_metrics.csv", group_rows)
    _write_csv(validation_dir / "tenant_metrics.csv", tenant_rows)
    _write_csv(validation_dir / "query_metrics.csv", query_rows)

    validation_summary = {
        "group_source": str(output_dir / "tenant_groups_v1.json"),
        "access_model": {
            "source": model.source,
            "plan_id": model.plan_id,
            "total_vectors": model.total_vectors,
            "tenant_count": len(model.tenant_patterns),
            "pattern_count": len(model.pattern_vector_counts),
            "partition_count": len(model.partition_vector_counts),
            "warning": model.warning,
        },
        "candidate_summary": {
            "exact_candidate_count": len(exact_candidates),
            "pair_candidate_count": len(pair_candidates),
            "singleton_candidate_count": len(singleton_candidates),
            "total_candidate_count": len(candidates),
            "budget_vectors": int(budget_vectors),
            "selected_group_count": len(selected),
            "selected_overlay_vectors": int(sum(candidate.group_vector_count for candidate in selected)),
        },
        "static_summary": static_summary,
        "query_summary": query_summary,
        "route_replay_note": route_replay_note,
    }
    _write_json(output_dir / "summary.json", validation_summary)
    _write_json(validation_dir / "summary.json", validation_summary)
    write_markdown_report(
        validation_dir / "summary.md",
        model=model,
        group_source=str(output_dir / "tenant_groups_v1.json"),
        warnings=[],
        group_rows=group_rows,
        tenant_rows=tenant_rows,
        static_summary=static_summary,
        query_summary=query_summary,
        route_replay_note=route_replay_note,
    )
    _write_demo_report(
        output_dir / "README.md",
        model=model,
        overlay_space_ratio=float(args.overlay_space_ratio),
        budget_vectors=int(budget_vectors),
        candidate_count=len(candidates),
        selected=selected,
        validation_summary=validation_summary,
    )
    progress.update("wrote demo reports")

    print("V1 shared protection group demo finished.")
    print(f"Output directory: {output_dir}")
    print(f"Candidate groups: {len(candidates)}")
    print(f"Budget vectors: {budget_vectors}")
    print(f"Selected groups: {len(selected)}")
    print(f"Selected overlay vectors: {sum(candidate.group_vector_count for candidate in selected)}")
    print(f"Protected tenants: {len({tenant_id for candidate in selected for tenant_id in candidate.tenant_ids})}")
    print(f"Space saving vs per-tenant overlays: {static_summary['total_space_saving_vs_per_tenant']:.4f}")
    print(f"Static estimated gain: {static_summary['total_static_gain_sum']:.4f}")
    if int(query_summary.get("replayed_query_count", 0) or 0):
        print(f"Replayed queries: {query_summary['replayed_query_count']}")
        print(f"Weighted query gain: {query_summary['weighted_gain_sum']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="V1 demo for shared tenant protection overlays.")
    parser.add_argument("--output-dir", default=str(RESULT_ROOT / "v1"))
    parser.add_argument("--overlay-space-ratio", type=float, default=0.25)
    parser.add_argument("--alpha", type=float, default=1.0, help="Fixed cost for one physical table access.")
    parser.add_argument("--prefer-plan", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--route-replay", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--query-dataset-path", default=None)
    parser.add_argument("--workload-limit", type=int, default=None)
    parser.add_argument("--max-pair-candidates", type=int, default=200000)
    parser.add_argument("--candidate-csv-limit", type=int, default=20000)
    parser.add_argument("--max-jaccard-pairs", type=int, default=20000)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    run_demo(args)


if __name__ == "__main__":
    main()
