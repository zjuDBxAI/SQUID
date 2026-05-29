from __future__ import annotations

import csv
import json
import math
import statistics
import sys
from itertools import combinations
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


def _mean(values: list[float]) -> float:
    return float(statistics.fmean(values)) if values else 0.0


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


def _load_current_clusters(cur, plan_id: int) -> dict[int, list[int]]:
    cur.execute(
        """
        SELECT DISTINCT cluster_id, tenant_id
        FROM kmeans_current_routes
        WHERE plan_id = %s
          AND route_kind = 'private'
        ORDER BY cluster_id, tenant_id;
        """,
        [int(plan_id)],
    )
    clusters: dict[int, list[int]] = {}
    for cluster_id, tenant_id in cur.fetchall():
        clusters.setdefault(int(cluster_id), []).append(int(tenant_id))
    return clusters


def _load_tenant_documents(cur) -> tuple[dict[int, set[int]], dict[int, int]]:
    cur.execute(
        """
        WITH tenant_documents AS (
            SELECT DISTINCT ur.user_id AS tenant_id, pa.document_id
            FROM UserRoles ur
            JOIN PermissionAssignment pa ON pa.role_id = ur.role_id
        ),
        document_vectors AS (
            SELECT document_id, COUNT(*)::BIGINT AS vector_count
            FROM documentblocks
            GROUP BY document_id
        )
        SELECT td.tenant_id, td.document_id, COALESCE(dv.vector_count, 0)::BIGINT AS vector_count
        FROM tenant_documents td
        JOIN document_vectors dv ON dv.document_id = td.document_id
        ORDER BY td.tenant_id, td.document_id;
        """
    )
    tenant_documents: dict[int, set[int]] = {}
    document_vector_counts: dict[int, int] = {}
    for tenant_id, document_id, vector_count in cur.fetchall():
        tenant_id = int(tenant_id)
        document_id = int(document_id)
        tenant_documents.setdefault(tenant_id, set()).add(document_id)
        document_vector_counts[document_id] = int(vector_count)
    return tenant_documents, document_vector_counts


def _load_tenant_patterns(cur, plan_id: int) -> tuple[dict[int, set[int]], dict[int, float]]:
    cur.execute(
        """
        SELECT pattern_id, tenant_ids, weight
        FROM kmeans_current_patterns
        WHERE plan_id = %s
        ORDER BY pattern_id;
        """,
        [int(plan_id)],
    )
    tenant_patterns: dict[int, set[int]] = {}
    pattern_weights: dict[int, float] = {}
    for pattern_id, tenant_ids, weight in cur.fetchall():
        pattern_id = int(pattern_id)
        pattern_weights[pattern_id] = float(weight)
        for tenant_id in tenant_ids or ():
            tenant_patterns.setdefault(int(tenant_id), set()).add(pattern_id)
    return tenant_patterns, pattern_weights


def _sum_by_docs(values_by_doc: dict[int, float] | dict[int, int], docs: set[int]) -> float:
    return float(sum(float(values_by_doc.get(int(document_id), 0.0)) for document_id in docs))


def _pair_metrics(
    tenant_left: int,
    tenant_right: int,
    *,
    tenant_documents: dict[int, set[int]],
    document_vector_counts: dict[int, int],
    tenant_patterns: dict[int, set[int]],
    pattern_weights: dict[int, float],
) -> dict[str, Any]:
    left_docs = tenant_documents.get(int(tenant_left), set())
    right_docs = tenant_documents.get(int(tenant_right), set())
    shared_docs = left_docs & right_docs
    union_docs = left_docs | right_docs

    left_vector_count = _sum_by_docs(document_vector_counts, left_docs)
    right_vector_count = _sum_by_docs(document_vector_counts, right_docs)
    shared_vector_count = _sum_by_docs(document_vector_counts, shared_docs)
    union_vector_count = _sum_by_docs(document_vector_counts, union_docs)

    left_patterns = tenant_patterns.get(int(tenant_left), set())
    right_patterns = tenant_patterns.get(int(tenant_right), set())
    shared_patterns = left_patterns & right_patterns
    union_patterns = left_patterns | right_patterns

    left_weight = _sum_by_docs(pattern_weights, left_patterns)
    right_weight = _sum_by_docs(pattern_weights, right_patterns)
    shared_weight = _sum_by_docs(pattern_weights, shared_patterns)
    union_weight = _sum_by_docs(pattern_weights, union_patterns)

    min_vector_count = min(left_vector_count, right_vector_count)
    min_weight = min(left_weight, right_weight)
    return {
        "tenant_left": int(tenant_left),
        "tenant_right": int(tenant_right),
        "left_document_count": int(len(left_docs)),
        "right_document_count": int(len(right_docs)),
        "shared_document_count": int(len(shared_docs)),
        "union_document_count": int(len(union_docs)),
        "document_jaccard": float(len(shared_docs) / max(len(union_docs), 1)),
        "left_vector_count": int(left_vector_count),
        "right_vector_count": int(right_vector_count),
        "shared_vector_count": int(shared_vector_count),
        "union_vector_count": int(union_vector_count),
        "vector_jaccard": float(shared_vector_count / max(union_vector_count, 1.0)),
        "vector_containment_min": float(shared_vector_count / max(min_vector_count, 1.0)),
        "shared_acl_pattern_count": int(len(shared_patterns)),
        "union_acl_pattern_count": int(len(union_patterns)),
        "acl_pattern_jaccard": float(len(shared_patterns) / max(len(union_patterns), 1)),
        "kmeans_shared_weight": float(shared_weight),
        "kmeans_weighted_jaccard": float(shared_weight / max(union_weight, 1.0)),
        "kmeans_weighted_containment_min": float(shared_weight / max(min_weight, 1.0)),
    }


def _cluster_rows(
    clusters: dict[int, list[int]],
    *,
    tenant_documents: dict[int, set[int]],
    document_vector_counts: dict[int, int],
    tenant_patterns: dict[int, set[int]],
    pattern_weights: dict[int, float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cluster_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []

    iterator = tqdm(sorted(clusters.items()), desc="Cluster similarity", unit="cluster") if tqdm is not None else sorted(clusters.items())
    for cluster_id, tenant_ids in iterator:
        tenant_ids = [int(tenant_id) for tenant_id in tenant_ids]
        cluster_docs: set[int] = set()
        tenant_vector_counts: list[float] = []
        tenant_document_counts: list[float] = []
        for tenant_id in tenant_ids:
            docs = tenant_documents.get(int(tenant_id), set())
            cluster_docs |= docs
            tenant_document_counts.append(float(len(docs)))
            tenant_vector_counts.append(_sum_by_docs(document_vector_counts, docs))

        pair_metrics = [
            _pair_metrics(
                left,
                right,
                tenant_documents=tenant_documents,
                document_vector_counts=document_vector_counts,
                tenant_patterns=tenant_patterns,
                pattern_weights=pattern_weights,
            )
            for left, right in combinations(tenant_ids, 2)
        ]
        for row in pair_metrics:
            row["cluster_id"] = int(cluster_id)
            pair_rows.append(row)

        document_jaccards = [float(row["document_jaccard"]) for row in pair_metrics]
        acl_pattern_jaccards = [float(row["acl_pattern_jaccard"]) for row in pair_metrics]
        vector_jaccards = [float(row["vector_jaccard"]) for row in pair_metrics]
        vector_containments = [float(row["vector_containment_min"]) for row in pair_metrics]
        weighted_jaccards = [float(row["kmeans_weighted_jaccard"]) for row in pair_metrics]
        weighted_containments = [float(row["kmeans_weighted_containment_min"]) for row in pair_metrics]
        shared_vectors = [float(row["shared_vector_count"]) for row in pair_metrics]

        cluster_rows.append(
            {
                "cluster_id": int(cluster_id),
                "tenant_count": int(len(tenant_ids)),
                "tenant_ids": " ".join(str(value) for value in tenant_ids),
                "pair_count": int(len(pair_metrics)),
                "cluster_union_document_count": int(len(cluster_docs)),
                "cluster_union_vector_count": int(_sum_by_docs(document_vector_counts, cluster_docs)),
                "tenant_document_count_mean": _mean(tenant_document_counts),
                "tenant_vector_count_mean": _mean(tenant_vector_counts),
                "pair_document_jaccard_mean": _mean(document_jaccards),
                "pair_document_jaccard_p50": _percentile(document_jaccards, 0.50),
                "pair_document_jaccard_min": min(document_jaccards) if document_jaccards else 0.0,
                "pair_acl_pattern_jaccard_mean": _mean(acl_pattern_jaccards),
                "pair_acl_pattern_jaccard_p50": _percentile(acl_pattern_jaccards, 0.50),
                "pair_acl_pattern_jaccard_min": min(acl_pattern_jaccards) if acl_pattern_jaccards else 0.0,
                "pair_vector_jaccard_mean": _mean(vector_jaccards),
                "pair_vector_jaccard_p50": _percentile(vector_jaccards, 0.50),
                "pair_vector_jaccard_min": min(vector_jaccards) if vector_jaccards else 0.0,
                "pair_vector_containment_min_mean": _mean(vector_containments),
                "pair_vector_containment_min_p50": _percentile(vector_containments, 0.50),
                "pair_kmeans_weighted_jaccard_mean": _mean(weighted_jaccards),
                "pair_kmeans_weighted_jaccard_p50": _percentile(weighted_jaccards, 0.50),
                "pair_kmeans_weighted_containment_min_mean": _mean(weighted_containments),
                "pair_shared_vector_count_mean": _mean(shared_vectors),
                "pair_shared_vector_count_p50": _percentile(shared_vectors, 0.50),
            }
        )
    return cluster_rows, pair_rows


def _markdown_report(summary: dict[str, Any], cluster_rows: list[dict[str, Any]]) -> str:
    top_by_size = sorted(cluster_rows, key=lambda row: int(row["cluster_union_vector_count"]), reverse=True)[:10]
    top_by_overlap = sorted(cluster_rows, key=lambda row: float(row["pair_vector_jaccard_mean"]), reverse=True)[:10]
    low_by_overlap = [
        row
        for row in sorted(cluster_rows, key=lambda row: float(row["pair_vector_jaccard_mean"]))
        if int(row["pair_count"]) > 0
    ][:10]

    def table(rows: list[dict[str, Any]]) -> str:
        lines = [
            "| cluster | tenants | vectors | pairs | vector_jaccard_mean | containment_mean | weighted_jaccard_mean |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for row in rows:
            lines.append(
                "| {cluster_id} | {tenant_count} | {cluster_union_vector_count} | {pair_count} | "
                "{pair_vector_jaccard_mean:.4f} | {pair_vector_containment_min_mean:.4f} | "
                "{pair_kmeans_weighted_jaccard_mean:.4f} |".format(**row)
            )
        return "\n".join(lines)

    return "\n".join(
        [
            "# KMeans Cluster Similarity Analysis",
            "",
            "This report analyzes the current private-route tenant clustering without reading materialized partition tables.",
            "",
            "## Summary",
            "",
            f"- plan_id: {summary['plan']['plan_id']}",
            f"- cluster_scope: {summary['cluster_scope']}",
            f"- cluster_count: {summary['plan']['cluster_count']}",
            f"- tenant_count: {summary['plan']['tenant_count']}",
            f"- pair_count: {summary['pair_count']}",
            f"- mean pair document Jaccard: {summary['pair_document_jaccard_distribution']['mean']:.4f}",
            f"- mean pair ACL-pattern Jaccard: {summary['pair_acl_pattern_jaccard_distribution']['mean']:.4f}",
            f"- mean pair vector Jaccard: {summary['pair_vector_jaccard_distribution']['mean']:.4f}",
            f"- mean pair vector containment-min: {summary['pair_vector_containment_min_distribution']['mean']:.4f}",
            f"- mean pair KMeans weighted Jaccard: {summary['pair_kmeans_weighted_jaccard_distribution']['mean']:.4f}",
            "",
            "## Largest Clusters",
            "",
            table(top_by_size),
            "",
            "## Highest Overlap Clusters",
            "",
            table(top_by_overlap),
            "",
            "## Lowest Overlap Clusters",
            "",
            table(low_by_overlap),
            "",
        ]
    )


def collect_current_cluster_similarity() -> dict[str, Any]:
    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            plan = _latest_plan(cur)
            if plan is None:
                raise RuntimeError("No current KMeans plan found. Run test_kmeans_partition.py --prepare true first.")
            clusters = _load_current_clusters(cur, int(plan["plan_id"]))
            tenant_documents, document_vector_counts = _load_tenant_documents(cur)
            tenant_patterns, pattern_weights = _load_tenant_patterns(cur, int(plan["plan_id"]))
    finally:
        conn.close()

    cluster_rows, pair_rows = _cluster_rows(
        clusters,
        tenant_documents=tenant_documents,
        document_vector_counts=document_vector_counts,
        tenant_patterns=tenant_patterns,
        pattern_weights=pattern_weights,
    )

    pair_document_jaccards = [float(row["document_jaccard"]) for row in pair_rows]
    pair_acl_pattern_jaccards = [float(row["acl_pattern_jaccard"]) for row in pair_rows]
    pair_vector_jaccards = [float(row["vector_jaccard"]) for row in pair_rows]
    pair_vector_containments = [float(row["vector_containment_min"]) for row in pair_rows]
    pair_weighted_jaccards = [float(row["kmeans_weighted_jaccard"]) for row in pair_rows]
    pair_weighted_containments = [float(row["kmeans_weighted_containment_min"]) for row in pair_rows]

    summary = {
        "plan": plan,
        "cluster_scope": "private_routes",
        "cluster_count": int(len(cluster_rows)),
        "tenant_count": int(sum(int(row["tenant_count"]) for row in cluster_rows)),
        "pair_count": int(len(pair_rows)),
        "cluster_union_vector_count_distribution": _distribution(
            [float(row["cluster_union_vector_count"]) for row in cluster_rows]
        ),
        "cluster_tenant_count_distribution": _distribution(
            [float(row["tenant_count"]) for row in cluster_rows]
        ),
        "pair_document_jaccard_distribution": _distribution(pair_document_jaccards),
        "pair_acl_pattern_jaccard_distribution": _distribution(pair_acl_pattern_jaccards),
        "pair_vector_jaccard_distribution": _distribution(pair_vector_jaccards),
        "pair_vector_containment_min_distribution": _distribution(pair_vector_containments),
        "pair_kmeans_weighted_jaccard_distribution": _distribution(pair_weighted_jaccards),
        "pair_kmeans_weighted_containment_min_distribution": _distribution(pair_weighted_containments),
    }
    return {
        "summary": summary,
        "cluster_rows": cluster_rows,
        "pair_rows": pair_rows,
    }


def write_results(result_dir: Path = RESULT_DIR) -> dict[str, Path]:
    result_dir.mkdir(parents=True, exist_ok=True)
    stats = collect_current_cluster_similarity()

    cluster_path = result_dir / "cluster_similarity.csv"
    pair_path = result_dir / "cluster_similarity_pairs.csv"
    summary_path = result_dir / "cluster_similarity_summary.json"
    report_path = result_dir / "cluster_similarity_report.md"

    _write_csv(cluster_path, stats["cluster_rows"])
    _write_csv(pair_path, stats["pair_rows"])
    summary_path.write_text(
        json.dumps(stats["summary"], ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_path.write_text(
        _markdown_report(stats["summary"], stats["cluster_rows"]),
        encoding="utf-8",
    )

    return {
        "cluster_similarity": cluster_path,
        "cluster_similarity_pairs": pair_path,
        "cluster_similarity_summary": summary_path,
        "cluster_similarity_report": report_path,
    }


def main() -> None:
    paths = write_results()
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
