from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from controller.method import (
    build_and_materialize_workload_aware_plan,
    create_indexes_for_materialized_partitions,
    dynamic_partition_search,
    get_current_plan_summary,
    get_tenant_partition_route,
    load_current_partitions,
)
from controller.method.common import DEFAULT_QUERY_DATASET_PATH


def _load_queries(query_dataset_path: str, limit: int | None = None) -> list[dict]:
    with open(query_dataset_path, "r", encoding="utf-8") as file:
        queries = json.load(file)
    if limit is not None:
        return queries[: int(limit)]
    return queries


def _compact_plan_summary(plan_summary: dict | None) -> dict:
    if not plan_summary:
        return {}
    metadata = dict((plan_summary.get("metadata", {}) or {}))
    interesting_metadata_keys = [
        "target_partition_count",
        "dp_effective_partition_count",
        "semantic_primary_partitioning_used",
        "semantic_primary_partition_count",
        "overlay_budget_vectors",
        "overlay_selected_vectors",
        "overlay_selected_tenant_count",
        "overlay_selected_access_count",
        "access_signature_cover_count",
        "query_count",
    ]
    return {
        "plan_id": int(plan_summary.get("plan_id", 0) or 0),
        "logical_pattern_count": int(plan_summary.get("logical_pattern_count", 0) or 0),
        "dag_node_count": int(plan_summary.get("dag_node_count", 0) or 0),
        "partition_count": int(plan_summary.get("partition_count", 0) or 0),
        "document_count": int(plan_summary.get("document_count", 0) or 0),
        "metadata": {
            key: metadata[key]
            for key in interesting_metadata_keys
            if key in metadata
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test for the ACL Prefix-DAG method.")
    parser.add_argument("--query-dataset-path", default=DEFAULT_QUERY_DATASET_PATH)
    parser.add_argument("--query-index", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=10)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--statistics-type", choices=["sql", "system"], default="sql")
    parser.add_argument("--min-pattern-support", type=int, default=16)
    parser.add_argument("--min-pattern-query-mass", type=float, default=0.0)
    parser.add_argument("--safe-density-threshold", type=float, default=0.35)
    parser.add_argument("--supplemental-edge-penalty", type=float, default=0.25)
    parser.add_argument("--supplemental-edge-gain-threshold", type=float, default=0.0)
    parser.add_argument("--target-partition-count", type=int, default=None)
    parser.add_argument("--max-partition-vector-count", type=int, default=None)
    parser.add_argument("--overlay-space-ratio", type=float, default=0.25)
    parser.add_argument("--workload-limit", type=int, default=None)
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--build-plan", action="store_true")
    parser.add_argument("--create-indexes", action="store_true")
    parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    parser.add_argument("--print-full-summary", action="store_true")
    parser.add_argument("--print-route", action="store_true")
    args = parser.parse_args()

    if args.build_plan:
        plan = build_and_materialize_workload_aware_plan(
            min_pattern_support=args.min_pattern_support,
            min_pattern_query_mass=args.min_pattern_query_mass,
            safe_density_threshold=args.safe_density_threshold,
            supplemental_edge_penalty=args.supplemental_edge_penalty,
            supplemental_edge_gain_threshold=args.supplemental_edge_gain_threshold,
            target_partition_count=args.target_partition_count,
            max_partition_vector_count=args.max_partition_vector_count,
            overlay_space_ratio=args.overlay_space_ratio,
            query_dataset_path=args.query_dataset_path,
            workload_limit=args.workload_limit,
            document_limit=args.document_limit,
            create_indexes=False,
            index_type=args.index_type,
        )
        print("[smoke] built ACL Prefix-DAG plan")
        if args.print_full_summary:
            print(json.dumps(plan.metadata, indent=2, ensure_ascii=False))
        else:
            print(json.dumps(_compact_plan_summary({"metadata": plan.metadata}), indent=2, ensure_ascii=False))

    partitions = load_current_partitions()
    if not partitions:
        raise RuntimeError("No materialized partitions found. Run with --build-plan first.")

    if args.create_indexes:
        create_indexes_for_materialized_partitions(index_type=args.index_type)
        print(f"[smoke] created {args.index_type} indexes")

    plan_summary = get_current_plan_summary()
    print("[smoke] current plan summary:")
    if args.print_full_summary:
        print(json.dumps(plan_summary or {}, indent=2, ensure_ascii=False))
    else:
        print(json.dumps(_compact_plan_summary(plan_summary), indent=2, ensure_ascii=False))
    print(f"[smoke] materialized partitions: {len(partitions)}")

    queries = _load_queries(args.query_dataset_path, limit=args.query_limit)
    if not queries:
        raise RuntimeError(f"No queries found in {args.query_dataset_path}")
    if args.query_index < 0 or args.query_index >= len(queries):
        raise IndexError(f"query_index {args.query_index} out of range for {len(queries)} loaded queries")

    query = queries[args.query_index]
    user_id = int(query["user_id"])
    query_vector = query["query_vector"]
    topk = int(query.get("topk", args.topk))

    route = get_tenant_partition_route(user_id, query_vector, topk=topk)
    route_summary = {
        "partition_count": int(route.partition_count),
        "candidate_partition_count": int(route.metadata.get("candidate_partition_count", 0) or 0),
        "selected_accessible_vector_coverage": float(route.metadata.get("selected_accessible_vector_coverage", 0.0) or 0.0),
        "route_coverage_guard_used": bool(route.metadata.get("route_coverage_guard_used", False)),
        "route_semantic_guard_used": bool(route.metadata.get("route_semantic_guard_used", False)),
        "fallback_used": bool(route.metadata.get("fallback_used", False)),
    }
    print("[smoke] route summary:")
    print(json.dumps(route_summary, indent=2, ensure_ascii=False))
    if args.print_route:
        print(json.dumps(route.metadata, indent=2, ensure_ascii=False, default=str))

    results, latency = dynamic_partition_search(
        user_id=user_id,
        query_vector=query_vector,
        topk=topk,
        statistics_type=args.statistics_type,
    )
    print(f"[smoke] query user_id={user_id} topk={topk} statistics_type={args.statistics_type}")
    print(f"[smoke] latency={latency}")
    print("[smoke] results:")
    print(json.dumps(results, indent=2, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
