from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
import inspect
import json
import math
from pathlib import Path
import statistics
import sys
import time
from collections import Counter, defaultdict
from typing import Any, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from controller.kmeans.hybrid_planner import HybridACLKMeansPlanner  # noqa: E402
from controller.kmeans.repository import KMeansRepository  # noqa: E402


RESULT_DIR = Path(__file__).resolve().parent / "result"


@dataclass(frozen=True)
class StepTraceRow:
    op_index: int
    operation: str
    left_id: int
    right_id: int
    left_vectors: int
    right_vectors: int
    before_memory: int
    after_memory: int
    memory_saved: int
    delta_latency: float
    unit_latency_cost: float
    before_cost: float
    after_cost: float
    new_group_count: int
    new_group_vectors_max: int
    new_group_vectors_sum: int
    total_storage: int
    private_storage: int
    allowed_total_storage: int
    allowed_private_storage: int
    storage_gap: int
    group_count: int
    max_group_vectors: int
    p90_group_vectors: float
    active_edges: int
    mean_degree: float
    max_degree: int
    heap_entries: int
    stale_candidates: int
    candidate_evaluations: int
    rejected_no_overlap: int
    rejected_no_saving: int
    heap_push_count: int
    edge_cache_size: int
    refreshed_edge_count: int
    graph_rebuild_count: int
    heap_rebuild_count: int


@dataclass(frozen=True)
class EdgeRecallRow:
    op_index: int
    group_id: int
    group_vectors: int
    group_patterns: int
    true_neighbor_count: int
    seen_degree: int
    topk: int
    topk_recall: float
    best_true_shared: float
    best_seen_shared: float
    best_missed_shared: float
    missed_topk_count: int
    seen_has_best_true: bool


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(v) for v in values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * float(q)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    weight = pos - lo
    return float(ordered[lo] * (1.0 - weight) + ordered[hi] * weight)


def _distribution(values: Sequence[float]) -> dict[str, float]:
    if not values:
        return {"count": 0, "min": 0.0, "p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "mean": 0.0}
    vals = [float(v) for v in values]
    return {
        "count": int(len(vals)),
        "min": float(min(vals)),
        "p50": _percentile(vals, 0.50),
        "p90": _percentile(vals, 0.90),
        "p95": _percentile(vals, 0.95),
        "p99": _percentile(vals, 0.99),
        "max": float(max(vals)),
        "mean": float(statistics.fmean(vals)),
    }


def _write_csv(path: Path, rows: Sequence[Any], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(asdict(rows[0]).keys()) if rows else list(fieldnames)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)


def _active_edges_and_degrees(groups: dict[int, dict[str, object]], adjacency: dict[int, set[int]]) -> tuple[int, list[int]]:
    live = set(int(group_id) for group_id in groups)
    edges = 0
    degrees: list[int] = []
    for group_id in live:
        neighbors = {int(n) for n in adjacency.get(int(group_id), set()) if int(n) in live and int(n) != int(group_id)}
        degrees.append(int(len(neighbors)))
        for neighbor_id in neighbors:
            if int(group_id) < int(neighbor_id):
                edges += 1
    return int(edges), degrees


def _group_vector_values(groups: dict[int, dict[str, object]]) -> list[int]:
    return [int(group.get("vector_count", 0) or 0) for group in groups.values()]


def _sample_group_ids(groups: dict[int, dict[str, object]], *, limit: int) -> list[int]:
    ranked = sorted(
        (int(group_id) for group_id in groups),
        key=lambda group_id: (-int(groups[group_id].get("vector_count", 0) or 0), int(group_id)),
    )
    return ranked[: max(0, int(limit))]


def _edge_recall_rows(locals_: dict[str, Any], *, op_index: int, topk: int, sample_groups: int) -> list[EdgeRecallRow]:
    groups: dict[int, dict[str, object]] = locals_.get("groups", {})
    adjacency: dict[int, set[int]] = locals_.get("adjacency", {})
    pattern_group_ids: dict[int, set[int]] = locals_.get("pattern_group_ids", {})
    pattern_weights: dict[int, int] = locals_.get("pattern_weights", {})
    if not groups or not pattern_group_ids:
        return []
    live = set(int(group_id) for group_id in groups)
    rows: list[EdgeRecallRow] = []
    for group_id in _sample_group_ids(groups, limit=int(sample_groups)):
        group = groups[int(group_id)]
        scores: Counter[int] = Counter()
        for pattern_id in group.get("pattern_ids", set()):  # type: ignore[union-attr]
            pattern_id = int(pattern_id)
            weight = float(pattern_weights.get(pattern_id, 0) or 0)
            if weight <= 0.0:
                continue
            for owner_id in pattern_group_ids.get(pattern_id, set()):
                owner_id = int(owner_id)
                if owner_id != int(group_id) and owner_id in live:
                    scores[owner_id] += float(weight)
        if not scores:
            continue
        seen = {int(n) for n in adjacency.get(int(group_id), set()) if int(n) in live and int(n) != int(group_id)}
        ranked_true = [int(neighbor_id) for neighbor_id, _score in scores.most_common(max(1, int(topk)))]
        denom = max(1, min(int(topk), len(ranked_true)))
        hit_count = sum(1 for neighbor_id in ranked_true[:denom] if int(neighbor_id) in seen)
        best_true_shared = float(scores[ranked_true[0]]) if ranked_true else 0.0
        seen_shared = [float(scores[n]) for n in seen if n in scores]
        missed_top = [int(n) for n in ranked_true[:denom] if int(n) not in seen]
        rows.append(
            EdgeRecallRow(
                op_index=int(op_index),
                group_id=int(group_id),
                group_vectors=int(group.get("vector_count", 0) or 0),
                group_patterns=int(len(group.get("pattern_ids", set()))),
                true_neighbor_count=int(len(scores)),
                seen_degree=int(len(seen)),
                topk=int(topk),
                topk_recall=float(hit_count) / float(denom),
                best_true_shared=float(best_true_shared),
                best_seen_shared=float(max(seen_shared)) if seen_shared else 0.0,
                best_missed_shared=float(max((scores[n] for n in missed_top), default=0.0)),
                missed_topk_count=int(len(missed_top)),
                seen_has_best_true=bool(ranked_true and int(ranked_true[0]) in seen),
            )
        )
    return rows


class PrivatePlanTracer:
    def __init__(self, *, edge_recall_every: int, edge_recall_topk: int, sample_groups: int) -> None:
        self.edge_recall_every = max(0, int(edge_recall_every))
        self.edge_recall_topk = max(1, int(edge_recall_topk))
        self.sample_groups = max(0, int(sample_groups))
        self.step_rows: list[StepTraceRow] = []
        self.edge_rows: list[EdgeRecallRow] = []

    def capture_from_progress_update(self) -> None:
        frame = inspect.currentframe()
        if frame is None:
            return
        frame = frame.f_back
        while frame is not None:
            if frame.f_code.co_name == "_cluster_private_tenants_by_core_star_v16":
                loc = frame.f_locals
                if "merge_count" in loc and "candidate" in loc and "groups" in loc:
                    self.capture_step(loc)
                return
            frame = frame.f_back

    def capture_step(self, loc: dict[str, Any]) -> None:
        groups: dict[int, dict[str, object]] = loc.get("groups", {})
        adjacency: dict[int, set[int]] = loc.get("adjacency", {})
        candidate: dict[str, object] = loc.get("candidate", {})
        left: dict[str, object] = loc.get("left", {})
        right: dict[str, object] = loc.get("right", {})
        new_group_ids = {int(group_id) for group_id in loc.get("new_group_ids", set())}
        new_vectors = [int(groups[group_id].get("vector_count", 0) or 0) for group_id in new_group_ids if group_id in groups]
        group_vectors = _group_vector_values(groups)
        active_edges, degrees = _active_edges_and_degrees(groups, adjacency)
        op_index = int(loc.get("merge_count", 0) or 0)
        row = StepTraceRow(
            op_index=int(op_index),
            operation=str(loc.get("operation", candidate.get("operation", ""))),
            left_id=int(loc.get("left_id", candidate.get("left_id", -1)) or -1),
            right_id=int(loc.get("right_id", candidate.get("right_id", -1)) or -1),
            left_vectors=int(left.get("vector_count", 0) or 0),
            right_vectors=int(right.get("vector_count", 0) or 0),
            before_memory=int(loc.get("before_memory", candidate.get("before_memory", 0)) or 0),
            after_memory=int(loc.get("after_memory", 0) or 0),
            memory_saved=int(loc.get("memory_saved", candidate.get("memory_saved", 0)) or 0),
            delta_latency=float(loc.get("delta_latency", candidate.get("delta_latency", 0.0)) or 0.0),
            unit_latency_cost=float(candidate.get("unit_latency_cost", 0.0) or 0.0),
            before_cost=float(loc.get("before_cost", candidate.get("before_cost", 0.0)) or 0.0),
            after_cost=float(loc.get("after_cost", 0.0) or 0.0),
            new_group_count=int(len(new_group_ids)),
            new_group_vectors_max=int(max(new_vectors)) if new_vectors else 0,
            new_group_vectors_sum=int(sum(new_vectors)),
            total_storage=int(loc.get("total_current_storage", 0) or 0),
            private_storage=int(loc.get("private_current_storage", 0) or 0),
            allowed_total_storage=int(loc.get("allowed_total_storage", 0) or 0),
            allowed_private_storage=int(loc.get("allowed_private_storage", 0) or 0),
            storage_gap=int(loc.get("total_current_storage", 0) or 0) - int(loc.get("allowed_total_storage", 0) or 0),
            group_count=int(len(groups)),
            max_group_vectors=int(max(group_vectors)) if group_vectors else 0,
            p90_group_vectors=float(_percentile(group_vectors, 0.90)) if group_vectors else 0.0,
            active_edges=int(active_edges),
            mean_degree=float(statistics.fmean(degrees)) if degrees else 0.0,
            max_degree=int(max(degrees)) if degrees else 0,
            heap_entries=int(len(loc.get("candidate_heap", []))),
            stale_candidates=int(loc.get("stale_candidates", 0) or 0),
            candidate_evaluations=int(loc.get("candidate_evaluations", 0) or 0),
            rejected_no_overlap=int(loc.get("rejected_no_overlap_candidates", 0) or 0),
            rejected_no_saving=int(loc.get("rejected_no_saving_candidates", 0) or 0),
            heap_push_count=int(loc.get("heap_push_count", 0) or 0),
            edge_cache_size=int(len(loc.get("edge_candidate_cache", {}))),
            refreshed_edge_count=int(loc.get("refreshed_edge_count", 0) or 0),
            graph_rebuild_count=int(loc.get("graph_rebuild_count", 0) or 0),
            heap_rebuild_count=int(loc.get("heap_rebuild_count", 0) or 0),
        )
        self.step_rows.append(row)
        if self.edge_recall_every > 0 and self.sample_groups > 0 and int(op_index) % int(self.edge_recall_every) == 0:
            self.edge_rows.extend(
                _edge_recall_rows(
                    loc,
                    op_index=int(op_index),
                    topk=int(self.edge_recall_topk),
                    sample_groups=int(self.sample_groups),
                )
            )


class _TraceProgress:
    def __init__(self, tracer: PrivatePlanTracer, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        self.tracer = tracer
        self.iterable = args[0] if args else None
        self.total = kwargs.get("total")
        self.desc = kwargs.get("desc", "")
        self.n = 0

    def __iter__(self):
        if self.iterable is None:
            return iter(())
        for item in self.iterable:
            yield item

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:  # noqa: ANN001
        return False

    def update(self, n: int = 1) -> None:
        self.n += int(n)
        if str(self.desc).startswith("Private core-star planner"):
            self.tracer.capture_from_progress_update()

    def set_description(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        if args:
            self.desc = str(args[0])

    def set_postfix(self, *args, **kwargs) -> None:  # noqa: ANN002, ANN003
        return None

    def close(self) -> None:
        return None


def _make_trace_tqdm(tracer: PrivatePlanTracer):
    def _factory(*args, **kwargs):  # noqa: ANN002, ANN003, ANN202
        return _TraceProgress(tracer, *args, **kwargs)
    return _factory

def _window_summaries(rows: Sequence[StepTraceRow], *, windows: int) -> list[dict[str, object]]:
    if not rows:
        return []
    windows = max(1, int(windows))
    op_min = int(rows[0].op_index)
    op_max = int(rows[-1].op_index)
    span = max(1, op_max - op_min + 1)
    bucket_size = max(1, int(math.ceil(span / windows)))
    output: list[dict[str, object]] = []
    for start in range(op_min, op_max + 1, bucket_size):
        end = min(op_max, start + bucket_size - 1)
        selected = [row for row in rows if start <= int(row.op_index) <= end]
        if not selected:
            continue
        ops = Counter(row.operation for row in selected)
        output.append(
            {
                "op_start": int(start),
                "op_end": int(end),
                "count": int(len(selected)),
                "operation_counts": dict(sorted(ops.items())),
                "memory_saved": _distribution([row.memory_saved for row in selected]),
                "delta_latency": _distribution([row.delta_latency for row in selected]),
                "unit_latency_cost": _distribution([row.unit_latency_cost for row in selected]),
                "active_edges_first": int(selected[0].active_edges),
                "active_edges_last": int(selected[-1].active_edges),
                "heap_entries_first": int(selected[0].heap_entries),
                "heap_entries_last": int(selected[-1].heap_entries),
                "group_count_first": int(selected[0].group_count),
                "group_count_last": int(selected[-1].group_count),
                "max_group_vectors_first": int(selected[0].max_group_vectors),
                "max_group_vectors_last": int(selected[-1].max_group_vectors),
                "storage_gap_first": int(selected[0].storage_gap),
                "storage_gap_last": int(selected[-1].storage_gap),
                "rejected_no_saving_delta": int(selected[-1].rejected_no_saving - selected[0].rejected_no_saving),
                "rejected_no_overlap_delta": int(selected[-1].rejected_no_overlap - selected[0].rejected_no_overlap),
            }
        )
    return output


def _edge_recall_summary(rows: Sequence[EdgeRecallRow]) -> dict[str, object]:
    if not rows:
        return {"count": 0}
    late_start = _percentile([row.op_index for row in rows], 0.75)
    late = [row for row in rows if float(row.op_index) >= float(late_start)]
    return {
        "count": int(len(rows)),
        "topk_recall": _distribution([row.topk_recall for row in rows]),
        "seen_degree": _distribution([row.seen_degree for row in rows]),
        "true_neighbor_count": _distribution([row.true_neighbor_count for row in rows]),
        "missed_topk_count": _distribution([row.missed_topk_count for row in rows]),
        "late_topk_recall": _distribution([row.topk_recall for row in late]),
        "late_seen_has_best_true_ratio": float(sum(1 for row in late if row.seen_has_best_true) / max(1, len(late))),
    }


def _final_partition_summary(plan) -> dict[str, object]:  # noqa: ANN001
    partitions = list(plan.partitions)
    vectors = [int(p.vector_count) for p in partitions]
    pattern_counts = [int(len(p.pattern_ids)) for p in partitions]
    tenant_counts = [int(len(p.tenant_ids)) for p in partitions]
    return {
        "partition_count": int(len(partitions)),
        "total_partition_vectors": int(sum(vectors)),
        "vector_count": _distribution(vectors),
        "pattern_count": _distribution(pattern_counts),
        "tenant_count": _distribution(tenant_counts),
    }


def _diagnosis(summary: dict[str, object]) -> list[str]:
    messages: list[str] = []
    edge_summary = summary.get("edge_recall_summary", {}) or {}
    late_recall = ((edge_summary.get("late_topk_recall", {}) or {}).get("mean")) if isinstance(edge_summary, dict) else None
    if late_recall is not None and float(late_recall) < 0.5:
        messages.append("late edge top-k recall is low; candidate graph likely misses high-shared neighbors near the final stage")
    windows = summary.get("windows", []) or []
    if windows:
        first_edges = float(windows[0].get("active_edges_first", 0) or 0)
        last_edges = float(windows[-1].get("active_edges_last", 0) or 0)
        if first_edges > 0 and last_edges / first_edges < 0.25:
            messages.append("active edge count collapses strongly during planning")
        first_heap = float(windows[0].get("heap_entries_first", 0) or 0)
        last_heap = float(windows[-1].get("heap_entries_last", 0) or 0)
        if first_heap > 0 and last_heap / first_heap < 0.25:
            messages.append("heap frontier shrinks strongly during planning")
        late_window = windows[-1]
        saving_rejects = int(late_window.get("rejected_no_saving_delta", 0) or 0)
        if saving_rejects > int(late_window.get("count", 0) or 0) * 10:
            messages.append("late stage evaluates many no-saving candidates; graph may expose edges that no longer reduce memory")
    if not messages:
        messages.append("no single obvious graph-collapse signal found; inspect step CSV and edge recall CSV")
    return messages


def run_plan_diagnosis(args: argparse.Namespace) -> dict[str, object]:
    repository = KMeansRepository()
    started = time.perf_counter()
    acl_rows = repository.fetch_acl_rows(document_limit=args.document_limit)
    load_seconds = time.perf_counter() - started
    print(f"[diagnose-plan] loaded acl_rows={len(acl_rows)} in {load_seconds:.2f}s", flush=True)

    planner = HybridACLKMeansPlanner()
    tracer = PrivatePlanTracer(
        edge_recall_every=int(args.edge_recall_every),
        edge_recall_topk=int(args.edge_recall_topk),
        sample_groups=int(args.edge_recall_sample_groups),
    )
    started = time.perf_counter()
    import controller.kmeans.hybrid_planner as hybrid_planner_module

    previous_tqdm = hybrid_planner_module.tqdm
    hybrid_planner_module.tqdm = _make_trace_tqdm(tracer)
    try:
        plan = planner.build_plan(
            acl_rows,
            private_cluster_count=int(args.private_cluster_count),
            shared_cluster_count=int(args.shared_cluster_count),
            shared_score_ratio=float(args.shared_score_ratio),
            shared_route_limit=int(args.shared_route_limit),
            private_replication_budget_ratio=float(args.private_replication_budget_ratio),
            ef_search=int(args.ef_search),
            embedding_dim=args.embedding_dim,
            query_dataset_path=args.query_dataset_path,
            show_progress=bool(args.show_progress),
            enable_split=bool(args.enable_split),
            private_edge_top_d=int(args.private_edge_top_d),
        )
    finally:
        hybrid_planner_module.tqdm = previous_tqdm
    plan_seconds = time.perf_counter() - started
    print(f"[diagnose-plan] plan-only finished in {plan_seconds:.2f}s, steps={len(tracer.step_rows)}", flush=True)

    metadata = dict(plan.metadata or {})
    private_metadata = dict(metadata.get("private_cluster_metadata") or {})
    summary: dict[str, object] = {
        "arguments": vars(args),
        "acl_row_count": int(len(acl_rows)),
        "load_acl_seconds": float(load_seconds),
        "plan_only_seconds": float(plan_seconds),
        "step_count": int(len(tracer.step_rows)),
        "plan_metadata": metadata,
        "private_metadata": private_metadata,
        "final_partitions": _final_partition_summary(plan),
        "step_distributions": {
            "memory_saved": _distribution([row.memory_saved for row in tracer.step_rows]),
            "delta_latency": _distribution([row.delta_latency for row in tracer.step_rows]),
            "unit_latency_cost": _distribution([row.unit_latency_cost for row in tracer.step_rows]),
            "active_edges": _distribution([row.active_edges for row in tracer.step_rows]),
            "heap_entries": _distribution([row.heap_entries for row in tracer.step_rows]),
            "max_group_vectors": _distribution([row.max_group_vectors for row in tracer.step_rows]),
            "group_count": _distribution([row.group_count for row in tracer.step_rows]),
        },
        "operation_counts": dict(sorted(Counter(row.operation for row in tracer.step_rows).items())),
        "windows": _window_summaries(tracer.step_rows, windows=int(args.windows)),
        "edge_recall_summary": _edge_recall_summary(tracer.edge_rows),
    }
    summary["diagnosis"] = _diagnosis(summary)
    return {"summary": summary, "step_rows": tracer.step_rows, "edge_rows": tracer.edge_rows}


def write_outputs(payload: dict[str, object], *, output_prefix: str, result_dir: Path) -> dict[str, Path]:
    result_dir.mkdir(parents=True, exist_ok=True)
    summary = payload["summary"]
    step_rows: list[StepTraceRow] = payload["step_rows"]  # type: ignore[assignment]
    edge_rows: list[EdgeRecallRow] = payload["edge_rows"]  # type: ignore[assignment]
    summary_path = result_dir / f"{output_prefix}_summary.json"
    steps_path = result_dir / f"{output_prefix}_steps.csv"
    edge_path = result_dir / f"{output_prefix}_edge_recall.csv"
    report_path = result_dir / f"{output_prefix}_report.md"
    _write_json(summary_path, summary)
    _write_csv(steps_path, step_rows, StepTraceRow.__dataclass_fields__.keys())
    _write_csv(edge_path, edge_rows, EdgeRecallRow.__dataclass_fields__.keys())
    diagnosis = summary.get("diagnosis", []) if isinstance(summary, dict) else []
    private = summary.get("private_metadata", {}) if isinstance(summary, dict) else {}
    report = [
        "# KMeans Plan-Only Diagnosis",
        "",
        f"- plan-only seconds: {summary.get('plan_only_seconds') if isinstance(summary, dict) else None}",
        f"- step count: {summary.get('step_count') if isinstance(summary, dict) else None}",
        f"- stop reason: {private.get('stop_reason') if isinstance(private, dict) else None}",
        f"- final private storage: {private.get('final_private_storage') if isinstance(private, dict) else None}",
        f"- allowed private storage: {private.get('allowed_private_storage') if isinstance(private, dict) else None}",
        "",
        "## Diagnosis",
        "",
    ]
    report.extend(f"- {item}" for item in diagnosis)
    report.extend(
        [
            "",
            "## Outputs",
            "",
            f"- {summary_path}",
            f"- {steps_path}",
            f"- {edge_path}",
        ]
    )
    report_path.write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"summary": summary_path, "steps": steps_path, "edge_recall": edge_path, "report": report_path}


def _str_to_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    lowered = str(value).strip().lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run kmeans plan only and trace private planner step state without materialization.")
    parser.add_argument("--private-cluster-count", type=int, default=30)
    parser.add_argument("--shared-cluster-count", type=int, default=5)
    parser.add_argument("--shared-score-ratio", type=float, default=0.10)
    parser.add_argument("--shared-route-limit", type=int, default=3)
    parser.add_argument("--private-replication-budget-ratio", type=float, default=2.0)
    parser.add_argument("--ef-search", type=int, default=5000)
    parser.add_argument("--private-edge-top-d", type=int, default=128)
    parser.add_argument("--enable-split", type=_str_to_bool, default=True)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--query-dataset-path", default=None)
    parser.add_argument("--show-progress", type=_str_to_bool, default=False)
    parser.add_argument("--edge-recall-every", type=int, default=200)
    parser.add_argument("--edge-recall-topk", type=int, default=32)
    parser.add_argument("--edge-recall-sample-groups", type=int, default=20)
    parser.add_argument("--windows", type=int, default=10)
    parser.add_argument("--output-prefix", default="kmeans_plan_only_diagnosis")
    parser.add_argument("--result-dir", default=str(RESULT_DIR))
    args = parser.parse_args()
    payload = run_plan_diagnosis(args)
    paths = write_outputs(payload, output_prefix=str(args.output_prefix), result_dir=Path(args.result_dir))
    for name, path in paths.items():
        print(f"[diagnose-plan] wrote {name}: {path}", flush=True)


if __name__ == "__main__":
    main()
