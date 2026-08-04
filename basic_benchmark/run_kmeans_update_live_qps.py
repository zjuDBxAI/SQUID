from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from basic_benchmark import efconfig  # noqa: E402
from basic_benchmark.common_function import prepare_query_dataset  # noqa: E402
from basic_benchmark.test_kmeans_partition import _ensure_index_state  # noqa: E402
from controller.kmeans import (  # noqa: E402
    apply_kmeans_update_batch,
    drop_indexes_for_materialized_partitions,
    kmeans_partition_search,
    load_current_partitions,
    prepare_kmeans_update_schema,
)
from controller.kmeans.storage import get_current_plan_summary  # noqa: E402
from services.config import get_db_connection  # noqa: E402


RESULT_DIR = PROJECT_ROOT / "basic_benchmark" / "result"


def _str_to_bool(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _load_updates(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("update workload must be a JSON list")
    return [dict(item) for item in payload]


class PhaseTracker:
    def __init__(self, started_at: float) -> None:
        self.started_at = float(started_at)
        self._lock = threading.Lock()
        self._phase = "warmup"
        self._generation = 0
        self._main_table_epoch = 0
        self._transitions: list[dict[str, float | str]] = [
            {"phase": "warmup", "offset_seconds": 0.0}
        ]

    def transition(self, phase: str, *, increment_main_table_epoch: bool = False) -> None:
        now = time.perf_counter()
        with self._lock:
            if increment_main_table_epoch:
                self._main_table_epoch += 1
            if str(phase) == self._phase:
                return
            self._phase = str(phase)
            self._generation += 1
            self._transitions.append(
                {
                    "phase": str(phase),
                    "offset_seconds": float(now - self.started_at),
                }
            )
        print(f"[live-qps] phase={phase} t={now - self.started_at:.3f}s", flush=True)

    def current(self) -> str:
        with self._lock:
            return str(self._phase)

    def snapshot(self) -> tuple[str, int, int]:
        """Return an atomic (phase, transition generation, main-table epoch)."""
        with self._lock:
            return (
                str(self._phase),
                int(self._generation),
                int(self._main_table_epoch),
            )

    def transitions(self) -> list[dict[str, float | str]]:
        with self._lock:
            return [dict(item) for item in self._transitions]


class QueryAdmissionGate:
    """Quiesce the load generator while coverage mutates live partition tables."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._open = True
        self._active = 0

    def enter(self, stop_event: threading.Event) -> bool:
        with self._condition:
            while not self._open:
                if stop_event.is_set():
                    return False
                self._condition.wait(timeout=0.1)
            self._active += 1
            return True

    def leave(self) -> None:
        with self._condition:
            self._active -= 1
            self._condition.notify_all()

    def close_and_wait(self) -> None:
        with self._condition:
            self._open = False
            while self._active:
                self._condition.wait(timeout=0.1)

    def open(self) -> None:
        with self._condition:
            self._open = True
            self._condition.notify_all()


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    index = min(len(ordered) - 1, max(0, int(math.ceil(float(fraction) * len(ordered))) - 1))
    return float(ordered[index])


def _exact_ground_truth_keys(user_id: int, query_vector, topk: int) -> set[tuple[int, int]]:
    """Read the exact, current main-table top-k without using a stale cache."""
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SET LOCAL enable_indexscan = off;")
            cur.execute("SET LOCAL enable_bitmapscan = off;")
            cur.execute("SET LOCAL enable_indexonlyscan = off;")
            cur.execute(
                """
                SELECT db.block_id, db.document_id
                FROM documentblocks db
                WHERE EXISTS (
                    SELECT 1
                    FROM PermissionAssignment pa
                    JOIN UserRoles ur ON ur.role_id = pa.role_id
                    WHERE pa.document_id = db.document_id
                      AND ur.user_id = %s
                )
                ORDER BY db.vector <-> %s::vector
                LIMIT %s;
                """,
                [int(user_id), query_vector, int(topk)],
            )
            return {(int(document_id), int(block_id)) for block_id, document_id in cur.fetchall()}
    finally:
        conn.close()


def _recall_at_k(results, ground_truth: set[tuple[int, int]]) -> float:
    predicted = {
        (int(result[1]), int(result[0]))
        for result in results
        if result is not None and len(result) >= 2
    }
    if not ground_truth:
        return 1.0 if not predicted else 0.0
    return float(len(predicted & ground_truth)) / float(len(ground_truth))


def _select_recall_panel(
    queries: list[dict[str, Any]],
    batches: list[list[dict[str, Any]]],
    *,
    panel_size: int,
    affected_only: bool,
) -> tuple[list[tuple[int, dict[str, Any]]], list[dict[str, Any]]]:
    """Select a deterministic panel, preferring users touched by more batches."""
    requested_size = min(max(1, int(panel_size)), len(queries))
    user_ids = sorted({int(query["user_id"]) for query in queries})
    touched_document_ids = sorted(
        {
            int(item["document_id"])
            for batch in batches
            for item in batch
            if item.get("document_id") is not None
        }
    )
    roles_by_user: dict[int, set[int]] = {user_id: set() for user_id in user_ids}
    old_roles_by_document: dict[int, set[int]] = {
        document_id: set() for document_id in touched_document_ids
    }
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT user_id, role_id FROM UserRoles WHERE user_id = ANY(%s);",
                [user_ids],
            )
            for user_id, role_id in cur.fetchall():
                roles_by_user.setdefault(int(user_id), set()).add(int(role_id))
            if touched_document_ids:
                cur.execute(
                    """
                    SELECT document_id, role_id
                    FROM PermissionAssignment
                    WHERE document_id = ANY(%s);
                    """,
                    [touched_document_ids],
                )
                for document_id, role_id in cur.fetchall():
                    old_roles_by_document.setdefault(int(document_id), set()).add(int(role_id))
    finally:
        conn.close()

    affected_roles_by_batch: list[set[int]] = []
    for batch in batches:
        affected_roles: set[int] = set()
        for item in batch:
            affected_roles.update(int(role_id) for role_id in (item.get("role_ids") or ()))
            if item.get("document_id") is not None:
                affected_roles.update(
                    old_roles_by_document.get(int(item["document_id"]), set())
                )
        affected_roles_by_batch.append(affected_roles)

    ranked: list[tuple[int, int, dict[str, Any]]] = []
    selection_metadata: list[dict[str, Any]] = []
    for query_id, query in enumerate(queries):
        user_id = int(query["user_id"])
        user_roles = roles_by_user.get(user_id, set())
        affected_batch_count = sum(
            bool(user_roles & batch_roles)
            for batch_roles in affected_roles_by_batch
        )
        ranked.append((int(affected_batch_count), int(query_id), query))
    ranked.sort(key=lambda item: (-int(item[0]), int(item[1])))

    if affected_only:
        preferred = [item for item in ranked if int(item[0]) > 0]
        fallback = [item for item in ranked if int(item[0]) == 0]
        ranked = preferred + fallback
    selected = ranked[:requested_size]
    panel = [(int(query_id), query) for _score, query_id, query in selected]
    for score, query_id, query in selected:
        selection_metadata.append(
            {
                "query_id": int(query_id),
                "user_id": int(query["user_id"]),
                "affected_batch_count": int(score),
                "topk": int(query.get("topk", 5)),
            }
        )
    return panel, selection_metadata


def _summarize_recall_panel_phases(
    cycles: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cycle in cycles:
        if cycle.get("valid") and cycle.get("recall_mean") is not None:
            by_phase[str(cycle["phase"])].append(cycle)
    result: dict[str, dict[str, Any]] = {}
    for phase, values in sorted(by_phase.items()):
        recalls = [float(value["recall_mean"]) for value in values]
        result[phase] = {
            "cycle_count": int(len(values)),
            "probe_count": int(sum(int(value["probe_count"]) for value in values)),
            "recall_mean": float(statistics.fmean(recalls)),
            "recall_min": float(min(recalls)),
            "recall_max": float(max(recalls)),
        }
    return result


def _phase_durations(
    transitions: list[dict[str, float | str]],
    finished_offset: float,
) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for index, transition in enumerate(transitions):
        phase = str(transition["phase"])
        started = float(transition["offset_seconds"])
        ended = (
            float(transitions[index + 1]["offset_seconds"])
            if index + 1 < len(transitions)
            else float(finished_offset)
        )
        result[phase] += max(0.0, ended - started)
    return dict(result)


def _summarize_phases(
    samples: list[dict[str, Any]],
    durations: dict[str, float],
) -> dict[str, dict[str, float | int]]:
    by_phase: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        by_phase[str(sample["phase"])].append(sample)

    summary: dict[str, dict[str, float | int]] = {}
    for phase in sorted(set(durations) | set(by_phase)):
        phase_samples = by_phase.get(phase, [])
        successful = [sample for sample in phase_samples if sample.get("success")]
        wall_latencies = [float(sample["wall_latency_seconds"]) for sample in successful]
        search_latencies = [float(sample["search_latency_seconds"]) for sample in successful]
        recalls = [
            float(sample["recall"])
            for sample in successful
            if sample.get("recall") is not None
        ]
        duration = float(durations.get(phase, 0.0))
        summary[phase] = {
            "duration_seconds": duration,
            "completed_queries": int(len(successful)),
            "failed_queries": int(len(phase_samples) - len(successful)),
            "qps": float(len(successful) / duration) if duration > 0.0 else 0.0,
            "wall_latency_mean_seconds": float(statistics.fmean(wall_latencies)) if wall_latencies else 0.0,
            "wall_latency_p50_seconds": _percentile(wall_latencies, 0.50),
            "wall_latency_p95_seconds": _percentile(wall_latencies, 0.95),
            "wall_latency_p99_seconds": _percentile(wall_latencies, 0.99),
            "search_latency_mean_seconds": float(statistics.fmean(search_latencies)) if search_latencies else 0.0,
            "recall_sample_count": int(len(recalls)),
            "recall_mean": float(statistics.fmean(recalls)) if recalls else None,
            "recall_p50": _percentile(recalls, 0.50) if recalls else None,
        }
    return summary


def _build_buckets(
    samples: list[dict[str, Any]],
    *,
    bucket_seconds: float,
) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        bucket_index = int(float(sample["finished_offset_seconds"]) // float(bucket_seconds))
        buckets[(bucket_index, str(sample["phase"]))].append(sample)

    result = []
    for (bucket_index, phase), values in sorted(buckets.items()):
        successful = [sample for sample in values if sample.get("success")]
        latencies = [float(sample["wall_latency_seconds"]) for sample in successful]
        recalls = [
            float(sample["recall"])
            for sample in successful
            if sample.get("recall") is not None
        ]
        result.append(
            {
                "bucket_start_seconds": float(bucket_index * bucket_seconds),
                "bucket_end_seconds": float((bucket_index + 1) * bucket_seconds),
                "phase": str(phase),
                "completed_queries": int(len(successful)),
                "failed_queries": int(len(values) - len(successful)),
                "qps": float(len(successful) / bucket_seconds),
                "wall_latency_p50_seconds": _percentile(latencies, 0.50),
                "wall_latency_p95_seconds": _percentile(latencies, 0.95),
                "wall_latency_p99_seconds": _percentile(latencies, 0.99),
                "recall_sample_count": int(len(recalls)),
                "recall_mean": float(statistics.fmean(recalls)) if recalls else None,
                "recall_p50": _percentile(recalls, 0.50) if recalls else None,
            }
        )
    return result


def _build_recall_time_buckets(
    samples: list[dict[str, Any]],
    *,
    bucket_seconds: float,
) -> list[dict[str, Any]]:
    """Aggregate recall strictly by wall-clock bucket, never by phase."""
    buckets: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for sample in samples:
        if sample.get("recall") is None:
            continue
        bucket_index = int(
            float(sample["finished_offset_seconds"]) // float(bucket_seconds)
        )
        buckets[bucket_index].append(sample)
    result: list[dict[str, Any]] = []
    for bucket_index, values in sorted(buckets.items()):
        recalls = [float(sample["recall"]) for sample in values]
        result.append(
            {
                "bucket_start_seconds": float(bucket_index * bucket_seconds),
                "bucket_end_seconds": float((bucket_index + 1) * bucket_seconds),
                "recall_sample_count": int(len(recalls)),
                "recall_mean": float(statistics.fmean(recalls)),
            }
        )
    return result


def run(args: argparse.Namespace) -> dict[str, Any]:
    if int(args.workers) <= 0:
        raise ValueError("workers must be positive")
    if float(args.coverage_seconds) <= 0.0 or float(args.maintained_seconds) <= 0.0:
        raise ValueError("coverage-seconds and maintained-seconds must be positive")
    if float(args.post_maintenance_warmup_seconds) < 0.0:
        raise ValueError("post-maintenance-warmup-seconds must be non-negative")
    if float(args.bucket_seconds) <= 0.0:
        raise ValueError("bucket-seconds must be positive")
    if int(args.recall_sample_every) <= 0:
        raise ValueError("recall-sample-every must be positive")
    if float(args.recall_plot_bucket_seconds) <= 0.0:
        raise ValueError("recall-plot-bucket-seconds must be positive")
    if int(args.recall_panel_size) < 0:
        raise ValueError("recall-panel-size must be non-negative")
    if (
        bool(args.record_recall)
        and int(args.recall_panel_size) > 0
        and float(args.recall_panel_interval_seconds) <= 0.0
    ):
        raise ValueError("recall-panel-interval-seconds must be positive")
    if int(args.batches) <= 0:
        raise ValueError("batches must be positive")
    if int(args.batches) > 1 and float(args.batch_interval_seconds) <= 0.0:
        raise ValueError("batch-interval-seconds must be positive for multiple batches")
    updates = _load_updates(args.update_workload)
    start = (int(args.batch_index) - 1) * int(args.batch_size)
    required = int(args.batch_size) * int(args.batches)
    selected_updates = updates[start : start + required]
    if len(selected_updates) != required:
        raise RuntimeError(
            f"{args.batches} batches starting at {args.batch_index} need {required} updates, "
            f"found {len(selected_updates)}"
        )
    batches = [
        selected_updates[offset : offset + int(args.batch_size)]
        for offset in range(0, required, int(args.batch_size))
    ]

    def batch_phase(batch_number: int, phase: str) -> str:
        if int(args.batches) == 1:
            return str(phase)
        return f"batch_{int(batch_number)}_{phase}"

    # Schema initialization contains ALTER TABLE migrations that require
    # AccessExclusiveLock.  Complete them before query workers start; running
    # this lazily inside the first update can deadlock with continuous readers.
    prepare_kmeans_update_schema()
    partitions = load_current_partitions(refresh=True)
    if not partitions:
        raise RuntimeError("No current kmeans plan. Prepare and materialize a plan first.")
    if bool(args.enable_index):
        _ensure_index_state(partitions, str(args.index_type))
    else:
        drop_indexes_for_materialized_partitions()

    efconfig.kmeans_index_type = str(args.index_type)
    efconfig.ef_search = int(args.ef_search)
    efconfig.kmeans_ef_search = int(args.ef_search)

    queries = prepare_query_dataset(
        regenerate=False,
        num_queries=int(args.query_num),
        query_dataset_path=str(args.query_dataset_path),
    )[: int(args.query_num)]
    if not queries:
        raise RuntimeError("query dataset is empty")
    use_fixed_recall_panel = bool(args.record_recall) and int(args.recall_panel_size) > 0
    recall_sample_stride = int(args.recall_sample_every)
    if bool(args.record_recall) and not use_fixed_recall_panel and len(queries) > 1:
        while recall_sample_stride > 1 and math.gcd(
            recall_sample_stride, len(queries)
        ) != 1:
            recall_sample_stride -= 1
        if recall_sample_stride != int(args.recall_sample_every):
            print(
                "[live-qps] recall sampling stride adjusted "
                f"from {int(args.recall_sample_every)} to {recall_sample_stride} "
                f"to cover all {len(queries)} query identities",
                flush=True,
            )
    recall_panel: list[tuple[int, dict[str, Any]]] = []
    recall_panel_selection: list[dict[str, Any]] = []
    ground_truth_cache: dict[tuple[int, int], set[tuple[int, int]]] = {}
    if use_fixed_recall_panel:
        recall_panel, recall_panel_selection = _select_recall_panel(
            queries,
            batches,
            panel_size=int(args.recall_panel_size),
            affected_only=bool(args.recall_panel_affected_only),
        )
        affected_count = sum(
            int(item["affected_batch_count"]) > 0
            for item in recall_panel_selection
        )
        print(
            "[live-qps] fixed recall panel: "
            f"{len(recall_panel)} queries, {affected_count} affected-query candidates, "
            f"{float(args.recall_panel_interval_seconds):g}s cadence; "
            "--recall-sample-every is ignored in panel mode",
            flush=True,
        )
        print("[live-qps] precomputing initial fixed-panel ground truth", flush=True)
        for query_id, query in recall_panel:
            ground_truth_cache[(0, int(query_id))] = _exact_ground_truth_keys(
                int(query["user_id"]),
                query["query_vector"],
                int(query.get("topk", 5)),
            )

    initial_plan = get_current_plan_summary(refresh=True)
    experiment_started = time.perf_counter()
    tracker = PhaseTracker(experiment_started)
    stop_event = threading.Event()
    query_admission = QueryAdmissionGate()
    samples: list[dict[str, Any]] = []
    sample_lock = threading.Lock()
    worker_errors: Counter[str] = Counter()
    recall_panel_cycles: list[dict[str, Any]] = []
    recall_probe_samples: list[dict[str, Any]] = []
    recall_panel_lock = threading.Lock()

    def query_worker(worker_id: int) -> None:
        query_index = int(worker_id) % len(queries)
        completed_by_worker = 0
        while not stop_event.is_set():
            if not query_admission.enter(stop_event):
                return
            query = queries[query_index]
            query_index = (query_index + int(args.workers)) % len(queries)
            started = time.perf_counter()
            success = True
            error = None
            result_count = 0
            search_latency = 0.0
            results = []
            recall = None
            recall_error = None
            try:
                results, search_latency = kmeans_partition_search(
                    user_id=int(query["user_id"]),
                    query_vector=query["query_vector"],
                    topk=int(query.get("topk", 5)),
                    statistics_type=str(args.statistics_type),
                )
                result_count = int(len(results))
            except Exception as exc:  # keep the load generator alive and report failures
                success = False
                error = f"{type(exc).__name__}: {exc}"
                worker_errors[str(error)] += 1
                time.sleep(0.01)
            finished = time.perf_counter()
            completed_by_worker += 1
            if (
                success
                and bool(args.record_recall)
                and not use_fixed_recall_panel
                and completed_by_worker % int(recall_sample_stride) == 0
            ):
                try:
                    topk = int(query.get("topk", 5))
                    ground_truth = _exact_ground_truth_keys(
                        int(query["user_id"]),
                        query["query_vector"],
                        topk,
                    )
                    recall = _recall_at_k(results, ground_truth)
                except Exception as exc:
                    recall_error = f"{type(exc).__name__}: {exc}"
                    worker_errors[f"recall: {recall_error}"] += 1
            query_admission.leave()
            sample = {
                "worker_id": int(worker_id),
                "user_id": int(query["user_id"]),
                "phase": tracker.current(),
                "started_offset_seconds": float(started - experiment_started),
                "finished_offset_seconds": float(finished - experiment_started),
                "wall_latency_seconds": float(finished - started),
                "search_latency_seconds": float(search_latency),
                "result_count": int(result_count),
                "success": bool(success),
                "error": error,
                "recall": recall,
                "recall_error": recall_error,
            }
            with sample_lock:
                samples.append(sample)

    def recall_panel_worker() -> None:
        interval = float(args.recall_panel_interval_seconds)
        next_scheduled_at = float(experiment_started)
        cycle_index = 0
        while not stop_event.is_set():
            wait_seconds = next_scheduled_at - time.perf_counter()
            if wait_seconds > 0.0 and stop_event.wait(wait_seconds):
                return
            scheduled_at = next_scheduled_at
            cycle_started_at = time.perf_counter()
            cycle_snapshot = tracker.snapshot()
            phase, generation, main_table_epoch = cycle_snapshot
            cycle_probes: list[dict[str, Any]] = []
            invalid_reason = None

            for query_id, query in recall_panel:
                if stop_event.is_set():
                    invalid_reason = "experiment_stopped"
                    break
                if tracker.snapshot() != cycle_snapshot:
                    invalid_reason = "phase_changed_during_panel"
                    break
                if not query_admission.enter(stop_event):
                    invalid_reason = "query_admission_closed"
                    break
                probe_started_at = time.perf_counter()
                probe: dict[str, Any] = {
                    "cycle_index": int(cycle_index),
                    "query_id": int(query_id),
                    "user_id": int(query["user_id"]),
                    "phase": str(phase),
                    "main_table_epoch": int(main_table_epoch),
                    "started_offset_seconds": float(probe_started_at - experiment_started),
                    "topk": int(query.get("topk", 5)),
                    "ground_truth_cache_hit": False,
                    "valid": False,
                    "recall": None,
                    "error": None,
                }
                try:
                    cache_key = (int(main_table_epoch), int(query_id))
                    ground_truth = ground_truth_cache.get(cache_key)
                    if ground_truth is None:
                        ground_truth = _exact_ground_truth_keys(
                            int(query["user_id"]),
                            query["query_vector"],
                            int(query.get("topk", 5)),
                        )
                        if tracker.snapshot() != cycle_snapshot:
                            invalid_reason = "phase_changed_during_ground_truth"
                            probe["error"] = invalid_reason
                        else:
                            ground_truth_cache[cache_key] = ground_truth
                    else:
                        probe["ground_truth_cache_hit"] = True

                    if invalid_reason is None:
                        search_started_at = time.perf_counter()
                        results, search_latency = kmeans_partition_search(
                            user_id=int(query["user_id"]),
                            query_vector=query["query_vector"],
                            topk=int(query.get("topk", 5)),
                            statistics_type=str(args.statistics_type),
                        )
                        search_finished_at = time.perf_counter()
                        probe.update(
                            {
                                "search_started_offset_seconds": float(
                                    search_started_at - experiment_started
                                ),
                                "search_finished_offset_seconds": float(
                                    search_finished_at - experiment_started
                                ),
                                "search_latency_seconds": float(search_latency),
                                "result_count": int(len(results)),
                            }
                        )
                        if tracker.snapshot() != cycle_snapshot:
                            invalid_reason = "phase_changed_during_search"
                            probe["error"] = invalid_reason
                        else:
                            probe["recall"] = _recall_at_k(results, ground_truth)
                            probe["valid"] = True
                except Exception as exc:
                    invalid_reason = f"{type(exc).__name__}: {exc}"
                    probe["error"] = invalid_reason
                    worker_errors[f"recall panel: {invalid_reason}"] += 1
                finally:
                    probe["finished_offset_seconds"] = float(
                        time.perf_counter() - experiment_started
                    )
                    query_admission.leave()
                cycle_probes.append(probe)
                if invalid_reason is not None:
                    break

            cycle_finished_at = time.perf_counter()
            cycle_valid = (
                invalid_reason is None
                and len(cycle_probes) == len(recall_panel)
                and tracker.snapshot() == cycle_snapshot
                and all(bool(probe.get("valid")) for probe in cycle_probes)
            )
            recalls = [
                float(probe["recall"])
                for probe in cycle_probes
                if probe.get("valid") and probe.get("recall") is not None
            ]
            cycle = {
                "cycle_index": int(cycle_index),
                "scheduled_offset_seconds": float(scheduled_at - experiment_started),
                "started_offset_seconds": float(cycle_started_at - experiment_started),
                "finished_offset_seconds": float(cycle_finished_at - experiment_started),
                "duration_seconds": float(cycle_finished_at - cycle_started_at),
                "phase": str(phase),
                "phase_generation": int(generation),
                "main_table_epoch": int(main_table_epoch),
                "probe_count": int(len(recalls)) if cycle_valid else 0,
                "valid": bool(cycle_valid),
                "invalid_reason": None if cycle_valid else str(
                    invalid_reason or "incomplete_panel"
                ),
                "recall_mean": (
                    float(statistics.fmean(recalls))
                    if cycle_valid and recalls
                    else None
                ),
                "recall_min": (
                    float(min(recalls))
                    if cycle_valid and recalls
                    else None
                ),
                "recall_max": (
                    float(max(recalls))
                    if cycle_valid and recalls
                    else None
                ),
            }
            with recall_panel_lock:
                recall_probe_samples.extend(cycle_probes)
                recall_panel_cycles.append(cycle)

            cycle_index += 1
            next_scheduled_at += interval
            while next_scheduled_at <= cycle_finished_at:
                next_scheduled_at += interval

    workers = [
        threading.Thread(target=query_worker, args=(worker_id,), name=f"live-qps-{worker_id}", daemon=True)
        for worker_id in range(int(args.workers))
    ]
    for worker in workers:
        worker.start()
    recall_thread = None
    if use_fixed_recall_panel:
        recall_thread = threading.Thread(
            target=recall_panel_worker,
            name="live-recall-panel",
            daemon=True,
        )
        recall_thread.start()

    repair_plan_snapshots: list[dict[str, Any]] = []
    update_results = []
    batch_timeline: list[dict[str, Any]] = []
    update_result = None
    update_error = None
    try:
        time.sleep(max(0.0, float(args.warmup_seconds)))
        first_batch_started_at = time.perf_counter()
        for batch_offset, batch in enumerate(batches):
            batch_number = int(args.batch_index) + int(batch_offset)
            scheduled_at = first_batch_started_at + float(batch_offset) * float(args.batch_interval_seconds)
            delay = float(scheduled_at - time.perf_counter())
            if delay > 0.0:
                tracker.transition(batch_phase(batch_number, "waiting"))
                time.sleep(delay)
            actual_started_at = time.perf_counter()
            if not bool(args.continuous_queries):
                query_admission.close_and_wait()
            tracker.transition(batch_phase(batch_number, "coverage_updating"))
            repair_snapshot: dict[str, Any] = {"batch_number": int(batch_number)}

            def on_repair_published(repair_plan, *, _batch_number=batch_number, _snapshot=repair_snapshot) -> None:
                _snapshot.update(
                    {
                        "partition_count": int(len(repair_plan.partitions)),
                        "route_count": int(len(repair_plan.tenant_routes)),
                        "vector_count": int(sum(int(partition.vector_count) for partition in repair_plan.partitions)),
                        "metadata": dict(repair_plan.metadata or {}),
                    }
                )
                tracker.transition(batch_phase(_batch_number, "coverage_repair"))
                if not bool(args.continuous_queries):
                    query_admission.open()
                time.sleep(float(args.coverage_seconds))
                tracker.transition(batch_phase(_batch_number, "maintenance_running"))

            def on_main_table_applied(_batch_id: int, *, _batch_number=batch_number) -> None:
                tracker.transition(
                    batch_phase(_batch_number, "main_table_applied"),
                    increment_main_table_epoch=True,
                )

            update_result = apply_kmeans_update_batch(
                batch,
                tau_del=float(args.tau_del),
                max_operations=int(args.max_operations),
                enable_maintenance=True,
                create_indexes=bool(args.enable_index),
                index_type=str(args.index_type),
                main_table_applied_callback=on_main_table_applied,
                repair_published_callback=on_repair_published,
            )
            update_results.append(update_result)
            repair_plan_snapshots.append(repair_snapshot)
            if update_result.accepted_operations:
                tracker.transition(batch_phase(batch_number, "maintenance_warmup"))
                warmup_seconds = float(args.post_maintenance_warmup_seconds)
                if batch_offset + 1 < len(batches):
                    next_scheduled_at = (
                        first_batch_started_at
                        + float(batch_offset + 1) * float(args.batch_interval_seconds)
                    )
                    warmup_seconds = min(
                        warmup_seconds,
                        max(0.0, next_scheduled_at - time.perf_counter()),
                    )
                time.sleep(warmup_seconds)
                tracker.transition(batch_phase(batch_number, "maintained"))
            else:
                tracker.transition(batch_phase(batch_number, "repair_only"))
            if batch_offset + 1 == len(batches):
                time.sleep(float(args.maintained_seconds))
            batch_timeline.append(
                {
                    "batch_number": int(batch_number),
                    "batch_id": int(update_result.batch_id),
                    "scheduled_start_seconds": float(scheduled_at - experiment_started),
                    "actual_start_seconds": float(actual_started_at - experiment_started),
                    "start_lag_seconds": float(max(0.0, actual_started_at - scheduled_at)),
                    "finished_seconds": float(time.perf_counter() - experiment_started),
                }
            )
    except Exception as exc:
        update_error = f"{type(exc).__name__}: {exc}"
        tracker.transition("update_failed")
    finally:
        tracker.transition("done")
        stop_event.set()
        query_admission.open()
        for worker in workers:
            worker.join(timeout=30.0)
        if recall_thread is not None:
            recall_thread.join(timeout=30.0)

    finished_offset = float(time.perf_counter() - experiment_started)
    transitions = tracker.transitions()
    durations = _phase_durations(transitions, finished_offset)
    with sample_lock:
        captured_samples = list(samples)
    with recall_panel_lock:
        captured_recall_cycles = list(recall_panel_cycles)
        captured_recall_probes = list(recall_probe_samples)
    phase_summary = _summarize_phases(captured_samples, durations)
    recall_phase_summary = _summarize_recall_panel_phases(captured_recall_cycles)
    buckets = _build_buckets(captured_samples, bucket_seconds=float(args.bucket_seconds))

    coverage_phases = [
        batch_phase(int(args.batch_index) + offset, "coverage_repair")
        for offset in range(int(args.batches))
    ]
    maintained_phases = [
        batch_phase(int(args.batch_index) + offset, "maintained")
        for offset in range(int(args.batches))
    ]

    def combined_qps(phase_names: list[str]) -> float:
        completed = sum(int(phase_summary.get(name, {}).get("completed_queries", 0) or 0) for name in phase_names)
        duration = sum(float(phase_summary.get(name, {}).get("duration_seconds", 0.0) or 0.0) for name in phase_names)
        return float(completed / duration) if duration > 0.0 else 0.0

    coverage_qps = combined_qps(coverage_phases)
    maintained_qps = combined_qps(maintained_phases)
    measurement_phases = tuple(coverage_phases + maintained_phases)
    measurement_failed_queries = int(sum(
        int(phase_summary.get(phase, {}).get("failed_queries", 0) or 0)
        for phase in measurement_phases
    ))
    measurement_valid = update_error is None and measurement_failed_queries == 0
    qps_gain = (
        float((maintained_qps - coverage_qps) / coverage_qps)
        if measurement_valid and coverage_qps > 0.0 and maintained_qps > 0.0
        else None
    )
    final_plan = get_current_plan_summary(refresh=True)
    accepted_operations = []
    update_metadata: dict[str, Any] = {}
    if update_result is not None:
        accepted_operations = [
            {
                "batch_id": int(result_item.batch_id),
                "op_type": str(candidate.op_type),
                "partition_ids": list(candidate.partition_ids),
                "delta_memory": int(candidate.delta_memory),
                "delta_latency": float(candidate.delta_latency),
            }
            for result_item in update_results
            for candidate in result_item.accepted_operations
        ]
        update_metadata = {
            str(key): value
            for key, value in update_result.metadata.items()
            if isinstance(value, (str, int, float, bool)) or value is None
        }

    result: dict[str, Any] = {
        "experiment": "kmeans_update_live_qps",
        "configuration": {
            "update_workload": str(args.update_workload),
            "batch_index": int(args.batch_index),
            "batch_size": int(args.batch_size),
            "batches": int(args.batches),
            "batch_interval_seconds": float(args.batch_interval_seconds),
            "query_dataset_path": str(args.query_dataset_path),
            "query_num": int(len(queries)),
            "workers": int(args.workers),
            "warmup_seconds": float(args.warmup_seconds),
            "coverage_seconds": float(args.coverage_seconds),
            "maintained_seconds": float(args.maintained_seconds),
            "post_maintenance_warmup_seconds": float(args.post_maintenance_warmup_seconds),
            "bucket_seconds": float(args.bucket_seconds),
            "statistics_type": str(args.statistics_type),
            "index_type": str(args.index_type),
            "ef_search": int(args.ef_search),
            "enable_index": bool(args.enable_index),
            "tau_del": float(args.tau_del),
            "max_operations": int(args.max_operations),
            "continuous_queries": bool(args.continuous_queries),
            "record_recall": bool(args.record_recall),
            "recall_sample_every": int(args.recall_sample_every),
            "effective_recall_sample_stride": int(recall_sample_stride),
            "recall_plot_bucket_seconds": float(args.recall_plot_bucket_seconds),
            "recall_mode": "fixed_panel" if use_fixed_recall_panel else "legacy_stride",
            "recall_panel_size": int(len(recall_panel)),
            "recall_panel_interval_seconds": float(args.recall_panel_interval_seconds),
            "recall_panel_affected_only": bool(args.recall_panel_affected_only),
        },
        "initial_plan": initial_plan,
        "repair_plan": repair_plan_snapshots[-1] if repair_plan_snapshots else {},
        "repair_plans": repair_plan_snapshots,
        "batch_timeline": batch_timeline,
        "final_plan": final_plan,
        "phase_transitions": transitions,
        "phase_summary": phase_summary,
        "recall_panel_selection": recall_panel_selection,
        "recall_panel_cycles": captured_recall_cycles,
        "recall_probe_samples": captured_recall_probes,
        "recall_phase_summary": recall_phase_summary,
        "qps_comparison": {
            "coverage_repair_qps": coverage_qps,
            "maintained_qps": maintained_qps,
            "relative_gain": qps_gain,
            "measurement_valid": bool(measurement_valid),
            "measurement_failed_queries": int(measurement_failed_queries),
            "expectation_met": bool(maintained_qps > coverage_qps)
            if measurement_valid and maintained_qps > 0.0
            else None,
        },
        "accepted_operations": accepted_operations,
        "update_metadata": update_metadata,
        "update_error": update_error,
        "worker_errors": dict(worker_errors.most_common(100)),
        "buckets": buckets,
    }

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    output_json = RESULT_DIR / f"kmeans_update_live_qps_{args.result_tag}.json"
    output_csv = RESULT_DIR / f"kmeans_update_live_qps_{args.result_tag}.csv"
    output_recall_csv = RESULT_DIR / f"kmeans_update_live_recall_{args.result_tag}.csv"
    result["recall_panel_csv"] = (
        str(output_recall_csv) if use_fixed_recall_panel else None
    )
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        fieldnames = list(buckets[0]) if buckets else ["bucket_start_seconds", "phase", "qps"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(buckets)
    if use_fixed_recall_panel:
        recall_fieldnames = [
            "cycle_index",
            "scheduled_offset_seconds",
            "started_offset_seconds",
            "finished_offset_seconds",
            "duration_seconds",
            "phase",
            "phase_generation",
            "main_table_epoch",
            "probe_count",
            "valid",
            "invalid_reason",
            "recall_mean",
            "recall_min",
            "recall_max",
        ]
        with open(output_recall_csv, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=recall_fieldnames)
            writer.writeheader()
            writer.writerows(captured_recall_cycles)

    recall_plot = None
    has_fixed_panel_recall = any(
        row.get("valid") and row.get("recall_mean") is not None
        for row in captured_recall_cycles
    )
    has_legacy_recall = any(row.get("recall_mean") is not None for row in buckets)
    if bool(args.record_recall) and (has_fixed_panel_recall or has_legacy_recall):
        try:
            import matplotlib.pyplot as plt

            if use_fixed_recall_panel:
                recall_rows = [
                    {
                        "time_seconds": (
                            float(row["started_offset_seconds"])
                            + float(row["finished_offset_seconds"])
                        )
                        / 2.0,
                        "recall_mean": float(row["recall_mean"]),
                    }
                    for row in captured_recall_cycles
                    if row.get("valid") and row.get("recall_mean") is not None
                ]
                recall_label = (
                    f"Fixed panel mean "
                    f"(n={len(recall_panel)}, "
                    f"{float(args.recall_panel_interval_seconds):g}s cadence)"
                )
            else:
                plot_bucket_seconds = max(
                    float(args.bucket_seconds),
                    float(args.recall_plot_bucket_seconds),
                )
                plot_buckets = _build_recall_time_buckets(
                    captured_samples,
                    bucket_seconds=plot_bucket_seconds,
                )
                recall_rows = [
                    {
                        "time_seconds": (
                            float(row["bucket_start_seconds"])
                            + float(row["bucket_end_seconds"])
                        )
                        / 2.0,
                        "recall_mean": float(row["recall_mean"]),
                    }
                    for row in plot_buckets
                ]
                recall_label = f"Recall@k ({plot_bucket_seconds:g}s mean)"
            figure, axis = plt.subplots(figsize=(10, 4.5))
            axis.plot(
                [float(row["time_seconds"]) for row in recall_rows],
                [float(row["recall_mean"]) for row in recall_rows],
                linewidth=1.8,
                color="#1f77b4",
                marker="o",
                markersize=3.0,
                label=recall_label,
            )
            marker_specs = [
                ("coverage_updating", "Update", "#d62728", "--"),
                ("main_table_applied", "Main commit", "#ff7f0e", ":"),
                ("coverage_repair", "Repair", "#2ca02c", "-."),
            ]
            for suffix, label, color, linestyle in marker_specs:
                matching = [
                    transition
                    for transition in transitions
                    if str(transition["phase"]).endswith(suffix)
                ]
                for marker_index, transition in enumerate(matching, start=1):
                    marker_time = float(transition["offset_seconds"])
                    axis.axvline(
                        marker_time,
                        color=color,
                        linestyle=linestyle,
                        alpha=0.75,
                        linewidth=1.1,
                        label=label if marker_index == 1 else None,
                    )
                    axis.text(
                        marker_time,
                        0.98,
                        f"{label} {marker_index}",
                        color=color,
                        rotation=90,
                        ha="right",
                        va="top",
                        fontsize=7,
                        transform=axis.get_xaxis_transform(),
                    )
            axis.set_xlabel("Time (seconds)")
            axis.set_ylabel("Recall@k")
            axis.set_ylim(0.0, 1.02)
            axis.grid(alpha=0.2)
            axis.legend(loc="lower right")
            figure.tight_layout()
            recall_plot = RESULT_DIR / f"kmeans_update_live_recall_{args.result_tag}.png"
            figure.savefig(recall_plot, dpi=180)
            plt.close(figure)
        except Exception as exc:
            worker_errors[f"recall plot: {type(exc).__name__}: {exc}"] += 1

    result["recall_plot"] = None if recall_plot is None else str(recall_plot)
    result["worker_errors"] = dict(worker_errors.most_common(100))
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    print(
        f"[live-qps] coverage={coverage_qps:.3f} qps, maintained={maintained_qps:.3f} qps, "
        f"gain={'n/a' if qps_gain is None else f'{qps_gain * 100.0:.2f}%'}, "
        f"valid={measurement_valid}",
        flush=True,
    )
    print(f"[live-qps] accepted operations: {len(accepted_operations)}", flush=True)
    print(f"[live-qps] results: {output_json} and {output_csv}", flush=True)
    if use_fixed_recall_panel:
        print(f"[live-qps] recall panel results: {output_recall_csv}", flush=True)
    if recall_plot is not None:
        print(f"[live-qps] recall plot: {recall_plot}", flush=True)

    if update_error is not None:
        raise RuntimeError(f"update failed during live QPS experiment: {update_error}")
    if not measurement_valid:
        raise RuntimeError(
            "live QPS measurement is invalid: "
            f"{measurement_failed_queries} query failures occurred during coverage_repair/maintained"
        )
    if not accepted_operations:
        print(
            "[live-qps] WARNING: maintenance accepted no operation; "
            "this run cannot compare a maintained plan.",
            flush=True,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Continuously measure KMeans QPS across coverage-repair and maintenance publication."
    )
    parser.add_argument("--update-workload", required=True)
    parser.add_argument(
        "--query-dataset-path",
        default=str(PROJECT_ROOT / "basic_benchmark" / "query_dataset.json"),
    )
    parser.add_argument("--batch-index", type=int, default=1, help="One-based batch index in the workload")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument(
        "--batch-interval-seconds",
        type=float,
        default=100.0,
        help="Target interval between the start times of consecutive update batches.",
    )
    parser.add_argument("--query-num", type=int, default=200)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--warmup-seconds", type=float, default=5.0)
    parser.add_argument("--coverage-seconds", type=float, default=10.0)
    parser.add_argument("--maintained-seconds", type=float, default=10.0)
    parser.add_argument("--post-maintenance-warmup-seconds", type=float, default=10.0)
    parser.add_argument("--bucket-seconds", type=float, default=1.0)
    parser.add_argument("--statistics-type", choices=["sql", "system"], default="system")
    parser.add_argument(
        "--index-type",
        choices=["squidhnsw", "hnsw", "ivfflat", "acorn"],
        default="hnsw",
    )
    parser.add_argument("--ef-search", type=int, default=25)
    parser.add_argument("--enable-index", type=_str_to_bool, default=True)
    parser.add_argument("--tau-del", type=float, default=0.2)
    parser.add_argument("--max-operations", type=int, default=8)
    parser.add_argument("--continuous-queries", type=_str_to_bool, default=False)
    parser.add_argument("--record-recall", type=_str_to_bool, default=False)
    parser.add_argument(
        "--recall-sample-every",
        type=int,
        default=50,
        help="Legacy stride sampler; ignored when recall-panel-size is positive.",
    )
    parser.add_argument(
        "--recall-panel-size",
        type=int,
        default=20,
        help="Number of fixed queries evaluated in every recall cycle; use 0 for legacy stride sampling.",
    )
    parser.add_argument(
        "--recall-panel-interval-seconds",
        type=float,
        default=10.0,
        help="Target cadence between fixed-panel recall cycles.",
    )
    parser.add_argument(
        "--recall-panel-affected-only",
        type=_str_to_bool,
        default=True,
        help="Prefer query users whose roles are touched by the selected update batches.",
    )
    parser.add_argument(
        "--recall-plot-bucket-seconds",
        type=float,
        default=10.0,
        help="Aggregation width used only for the recall plot.",
    )
    parser.add_argument("--result-tag", default="bs50_w8")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
