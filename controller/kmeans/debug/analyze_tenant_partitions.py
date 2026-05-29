from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from services.config import get_db_connection


RESULT_DIR = Path(__file__).resolve().parent / "result"


try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - fallback for non-benchmark envs.
    tqdm = None


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
            "p25": 0.0,
            "p50": 0.0,
            "p75": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "max": 0.0,
            "mean": 0.0,
            "stdev": 0.0,
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
        "p99": _percentile(normalized, 0.99),
        "max": float(max(normalized)),
        "mean": float(statistics.fmean(normalized)),
        "stdev": float(statistics.pstdev(normalized)) if len(normalized) > 1 else 0.0,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _latest_plan(cur) -> dict[str, Any] | None:
    cur.execute(
        """
        SELECT plan_id, cluster_count, tenant_count, partition_count,
               document_count, vector_count, metadata, created_at
        FROM kmeans_current_plan
        ORDER BY plan_id DESC
        LIMIT 1;
        """
    )
    row = cur.fetchone()
    if row is None:
        return None
    return {
        "plan_id": int(row[0]),
        "cluster_count": int(row[1]),
        "tenant_count": int(row[2]),
        "partition_count": int(row[3]),
        "document_count": int(row[4]),
        "partition_vector_count": int(row[5]),
        "metadata": dict(row[6] or {}),
        "created_at": str(row[7]),
    }


def _collect_partition_rows(cur, plan_id: int) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT partition_id, cluster_id, partition_kind, table_name, vector_count,
               tenant_ids, pattern_ids, document_ids, metadata
        FROM kmeans_current_partitions
        WHERE plan_id = %s
        ORDER BY vector_count DESC, partition_kind, cluster_id;
        """,
        [int(plan_id)],
    )
    rows: list[dict[str, Any]] = []
    for partition_id, cluster_id, partition_kind, table_name, vector_count, tenant_ids, pattern_ids, document_ids, metadata in cur.fetchall():
        normalized_tenants = [int(value) for value in (tenant_ids or [])]
        normalized_patterns = [int(value) for value in (pattern_ids or [])]
        normalized_documents = [int(value) for value in (document_ids or [])]
        rows.append(
            {
                "partition_id": str(partition_id),
                "cluster_id": int(cluster_id),
                "partition_kind": str(partition_kind),
                "table_name": str(table_name),
                "vector_count": int(vector_count),
                "tenant_count": int(len(normalized_tenants)),
                "pattern_count": int(len(normalized_patterns)),
                "document_count": int(len(normalized_documents)),
                "tenant_ids": " ".join(str(value) for value in normalized_tenants),
                "pattern_ids": " ".join(str(value) for value in normalized_patterns),
                "metadata": json.dumps(dict(metadata or {}), ensure_ascii=False, sort_keys=True),
            }
        )
    return rows


def _collect_tenant_rows(cur, plan_id: int) -> list[dict[str, Any]]:
    cur.execute(
        """
        SELECT kr.tenant_id, kr.cluster_id, kr.route_kind, kr.partition_id, kr.table_name,
               cardinality(kr.pattern_ids) AS route_pattern_count,
               kp.partition_kind,
               kp.vector_count AS partition_vector_count,
               cardinality(kp.tenant_ids) AS partition_tenant_count,
               cardinality(kp.pattern_ids) AS partition_pattern_count
        FROM kmeans_current_routes kr
        JOIN kmeans_current_partitions kp
          ON kp.plan_id = kr.plan_id
         AND kp.partition_id = kr.partition_id
        WHERE kr.plan_id = %s
        ORDER BY kr.tenant_id, kr.route_kind, kr.cluster_id;
        """,
        [int(plan_id)],
    )
    return [
        {
            "tenant_id": int(row[0]),
            "cluster_id": int(row[1]),
            "route_kind": str(row[2]),
            "partition_id": str(row[3]),
            "table_name": str(row[4]),
            "route_pattern_count": int(row[5]),
            "partition_kind": str(row[6]),
            "partition_vector_count": int(row[7]),
            "partition_tenant_count": int(row[8]),
            "partition_pattern_count": int(row[9]),
        }
        for row in cur.fetchall()
    ]


def collect_current_kmeans_partition_stats() -> dict[str, Any]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            plan = _latest_plan(cur)
            if plan is None:
                raise RuntimeError("No current KMeans plan found. Run test_kmeans_partition.py --prepare true first.")
            plan_id = int(plan["plan_id"])
            partition_rows = _collect_partition_rows(cur, plan_id)
            tenant_rows = _collect_tenant_rows(cur, plan_id)
    finally:
        conn.close()

    vector_counts = [float(row["vector_count"]) for row in partition_rows]
    tenant_counts = [float(row["tenant_count"]) for row in partition_rows]
    pattern_counts = [float(row["pattern_count"]) for row in partition_rows]
    document_counts = [float(row["document_count"]) for row in partition_rows]

    metadata = dict(plan.get("metadata") or {})
    summary = {
        "plan": plan,
        "partition_count": int(len(partition_rows)),
        "tenant_assignment_count": int(len(tenant_rows)),
        "vector_count_distribution": _distribution(vector_counts),
        "tenant_count_distribution": _distribution(tenant_counts),
        "pattern_count_distribution": _distribution(pattern_counts),
        "document_count_distribution": _distribution(document_counts),
        "total_partition_vectors_from_rows": int(sum(int(row["vector_count"]) for row in partition_rows)),
        "original_vector_count": int(metadata.get("original_vector_count", 0) or 0),
        "memory_replication_factor": float(metadata.get("memory_replication_factor", 0.0) or 0.0),
    }
    return {
        "summary": summary,
        "partition_rows": partition_rows,
        "tenant_rows": tenant_rows,
    }


def write_results(result_dir: Path = RESULT_DIR) -> dict[str, Path]:
    result_dir.mkdir(parents=True, exist_ok=True)
    stats = collect_current_kmeans_partition_stats()

    partition_path = result_dir / "partition_tenant_groups.csv"
    tenant_path = result_dir / "tenant_cluster_assignments.csv"
    summary_path = result_dir / "summary.json"

    steps = [
        ("write partition_tenant_groups.csv", lambda: _write_csv(partition_path, stats["partition_rows"])),
        ("write tenant_cluster_assignments.csv", lambda: _write_csv(tenant_path, stats["tenant_rows"])),
        (
            "write summary.json",
            lambda: summary_path.write_text(
                json.dumps(stats["summary"], ensure_ascii=False, indent=2, sort_keys=True),
                encoding="utf-8",
            ),
        ),
    ]

    iterator = tqdm(steps, desc="KMeans partition stats", unit="file") if tqdm is not None else steps
    for _, action in iterator:
        action()

    return {
        "partition_tenant_groups": partition_path,
        "tenant_cluster_assignments": tenant_path,
        "summary": summary_path,
    }


def main() -> None:
    paths = write_results()
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
