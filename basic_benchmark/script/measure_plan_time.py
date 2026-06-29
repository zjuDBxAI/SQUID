#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
from collections import defaultdict
from contextlib import contextmanager

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from controller.dynamic_partition.hnsw import AnonySys_dynamic_partition as honeybee
from controller.dynamic_partition.hnsw.helper import (
    clean_empty_partitions,
    fetch_initial_data,
    prepare_background_data,
    reorganize_partitions,
)
from controller.dynamic_partition.hnsw.heavy_partition_refine import (
    rebalance_heavy_partition,
    remap_comb_role_trackers,
)
from controller.baseline.veda.storage import build_veda_plan
from controller.kmeans.hybrid_planner import HybridACLKMeansPlanner
from controller.kmeans.repository import KMeansRepository


@contextmanager
def timed():
    start = time.perf_counter()
    holder = {"seconds": 0.0}
    try:
        yield holder
    finally:
        holder["seconds"] = time.perf_counter() - start


def _bool_arg(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {value}")


def _load_honeybee_parameters(path: Path) -> dict[str, float]:
    if not path.exists():
        raise FileNotFoundError(
            f"HONEYBEE parameter file not found: {path}. "
            "This script only measures plan time and will not fit parameters or build indexes."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    required = ("k", "beta", "a", "b")
    missing = [name for name in required if name not in data]
    if missing:
        raise ValueError(f"HONEYBEE parameter file missing keys: {missing}")
    return {name: float(data[name]) for name in required}


def measure_honeybee(args: argparse.Namespace) -> dict[str, object]:
    params = _load_honeybee_parameters(Path(args.honeybee_parameter_path))
    with timed() as elapsed:
        roles, documents, permissions, _avg_blocks_per_document, user_to_roles = fetch_initial_data()
        role_to_documents, document_to_index = prepare_background_data(roles, documents, permissions)
        role_to_documents_index = {
            role: set(sorted({document_to_index[doc] for doc in docs if doc in document_to_index}))
            for role, docs in role_to_documents.items()
        }

        document_index_to_roles: dict[int, set[int]] = defaultdict(set)
        for role, doc_indices in role_to_documents_index.items():
            for idx in doc_indices:
                document_index_to_roles[int(idx)].add(int(role))

        role_combinations, comb_role_weights = honeybee.init_user_role_combination_data()
        single_role_weights = honeybee.calculate_single_role_weights_from_queries(user_to_roles, role_combinations)

        combination_roles_to_documents: dict[tuple[int, ...], set[int]] = {}
        for comb in role_combinations:
            all_documents: set[int] = set()
            for role in comb:
                if role in role_to_documents_index:
                    all_documents.update(role_to_documents_index[role])
            combination_roles_to_documents[tuple(comb)] = all_documents

        all_roles = {int(role) for comb in role_combinations for role in comb}
        for role in all_roles:
            combination_roles_to_documents.setdefault((role,), role_to_documents_index.get(role, set()))

        expanded_role_combinations = set(tuple(comb) for comb in role_combinations)
        for role in all_roles:
            expanded_role_combinations.add((role,))

        partition_assignment, comb_role_trackers = honeybee.split_comb_roles(
            role_to_documents_index,
            float(args.honeybee_storage),
            int(args.topk),
            float(params["k"]),
            float(params["beta"]),
            float(params["a"]),
            float(params["b"]),
            expanded_role_combinations,
            combination_roles_to_documents,
            comb_role_weights,
            single_role_weights,
            combination_mode=False,
            recall=args.honeybee_recall,
        )

        if bool(args.honeybee_refine):
            sorted_partitions = sorted(
                ((pid, len(docs)) for pid, docs in partition_assignment.items()),
                key=lambda item: item[1],
                reverse=True,
            )
            largest_partition_id = sorted_partitions[0][0] if sorted_partitions else None
            largest_size = sorted_partitions[0][1] if sorted_partitions else 0
            if largest_partition_id is not None and largest_size > 0:
                partition_assignment, comb_role_trackers = rebalance_heavy_partition(
                    partition_assignment,
                    comb_role_trackers,
                    document_index_to_roles,
                    honeybee.logger,
                    target_partitions={largest_partition_id},
                )
                partition_assignment = clean_empty_partitions(partition_assignment)
                partition_assignment, partition_mapping = reorganize_partitions(partition_assignment)
                comb_role_trackers = remap_comb_role_trackers(comb_role_trackers, partition_mapping)

    return {
        "method": "HONEYBEE",
        "plan_seconds": float(elapsed["seconds"]),
        "partition_count": int(len(partition_assignment)),
        "metadata": {
            "storage": float(args.honeybee_storage),
            "recall": args.honeybee_recall,
            "refine": bool(args.honeybee_refine),
            "role_count": int(len(roles)),
            "document_count": int(len(documents)),
            "combination_count": int(len(combination_roles_to_documents)),
        },
    }


def measure_veda(args: argparse.Namespace) -> dict[str, object]:
    with timed() as elapsed:
        plan = build_veda_plan(
            algorithm=str(args.veda_algorithm),
            indexing_threshold=int(args.veda_indexing_threshold),
            storage_amplification=float(args.veda_storage_amplification),
            ef_search=int(args.ef_search),
            document_limit=args.document_limit,
            show_progress=bool(args.show_progress),
        )
    return {
        "method": str(args.veda_algorithm).upper(),
        "plan_seconds": float(elapsed["seconds"]),
        "partition_count": int(len(plan.nodes)),
        "metadata": dict(plan.metadata or {}),
    }


def measure_kmeans(args: argparse.Namespace) -> dict[str, object]:
    with timed() as elapsed:
        repository = KMeansRepository()
        acl_rows = repository.fetch_acl_rows(document_limit=args.document_limit)
        effective_private_cluster_count = (
            int(args.private_cluster_count)
            if args.private_cluster_count is not None
            else int(args.cluster_count)
        )
        planner = HybridACLKMeansPlanner()
        plan = planner.build_plan(
            acl_rows,
            private_cluster_count=int(effective_private_cluster_count),
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
    return {
        "method": "KMEANS",
        "plan_seconds": float(elapsed["seconds"]),
        "partition_count": int(len(plan.partitions)),
        "metadata": dict(plan.metadata or {}),
    }


def _write_outputs(rows: list[dict[str, object]], output_json: Path | None, output_csv: Path | None) -> None:
    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=["method", "plan_seconds", "partition_count", "metadata"])
            writer.writeheader()
            for row in rows:
                writer.writerow({
                    "method": row["method"],
                    "plan_seconds": f"{float(row['plan_seconds']):.6f}",
                    "partition_count": int(row["partition_count"]),
                    "metadata": json.dumps(row.get("metadata", {}), ensure_ascii=False, sort_keys=True),
                })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure planning time for HONEYBEE(dynamic_partition), Veda/EffVeda, and KMeans without materializing partitions."
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["honeybee", "veda", "kmeans"],
        choices=["honeybee", "veda", "kmeans"],
        help="Methods to measure.",
    )
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--show-progress", type=_bool_arg, default=False)
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=40)
    parser.add_argument("--query-dataset-path", default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)

    parser.add_argument("--honeybee-storage", type=float, default=1.5)
    parser.add_argument("--honeybee-recall", type=float, default=None)
    parser.add_argument("--honeybee-refine", type=_bool_arg, default=True)
    parser.add_argument(
        "--honeybee-parameter-path",
        default=str(PROJECT_ROOT / "controller" / "dynamic_partition" / "hnsw" / "parameter_hnsw.json"),
    )

    parser.add_argument("--veda-algorithm", default="effveda", choices=["veda", "effveda"])
    parser.add_argument("--veda-indexing-threshold", type=int, default=1000)
    parser.add_argument("--veda-storage-amplification", type=float, default=1.2)

    parser.add_argument("--cluster-count", type=int, default=30)
    parser.add_argument("--private-cluster-count", type=int, default=None)
    parser.add_argument("--shared-cluster-count", type=int, default=5)
    parser.add_argument("--shared-score-ratio", type=float, default=0.10)
    parser.add_argument("--shared-route-limit", type=int, default=3)
    parser.add_argument("--private-replication-budget-ratio", type=float, default=2.0)
    parser.add_argument("--enable-split", type=_bool_arg, default=True)
    parser.add_argument("--private-edge-top-d", type=int, default=32)

    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    method_fns = {
        "honeybee": measure_honeybee,
        "veda": measure_veda,
        "kmeans": measure_kmeans,
    }
    for iteration in range(1, max(1, int(args.iterations)) + 1):
        for method in args.methods:
            started_at = time.strftime("%Y-%m-%d %H:%M:%S")
            print(f"[plan-time] start method={method} iteration={iteration} at={started_at}", flush=True)
            row = method_fns[method](args)
            row["iteration"] = int(iteration)
            rows.append(row)
            print(
                f"[plan-time] method={row['method']} iteration={iteration} "
                f"seconds={float(row['plan_seconds']):.6f} partitions={int(row['partition_count'])}",
                flush=True,
            )
    output_json = Path(args.output_json) if args.output_json else None
    output_csv = Path(args.output_csv) if args.output_csv else None
    _write_outputs(rows, output_json, output_csv)
    print(json.dumps(rows, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
