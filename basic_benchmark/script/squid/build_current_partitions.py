#!/usr/bin/env python3
"""Build current-style benchmark partitions for the connected database.

This is the companion builder for direct_pg_qps.py with PLAN_MODE=current.
It intentionally uses the existing non-versioned/current metadata paths:

* SQUID/OURS -> kmeans_current_* tables
* VEDA/EffVeda -> veda_current_* tables
* ROLE -> documentblocks_role_* tables
* RLS -> documentblocks row-level security
* HQI -> QD-tree pickle + partition tables
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from controller.baseline.pg_row_security.row_level_security import (  # noqa: E402
    create_database_users,
    disable_row_level_security,
    drop_database_users,
    enable_row_level_security,
)
from controller.baseline.prefilter.initialize_partitions import (  # noqa: E402
    drop_prefilter_partition_tables,
    initialize_role_partitions,
)
from controller.baseline.veda import build_and_materialize_veda_plan  # noqa: E402


DEFAULT_OURS_EFS_MODEL_JSON = (
    PROJECT_ROOT
    / "controller"
    / "kmeans"
    / "train"
    / "result"
    / "yfcc_recall_fit"
    / "yfcc_hnsw_recall_model.json"
)


METHOD_ALIASES = {
    "squid": "ours",
    "ours": "ours",
    "kmeans": "ours",
    "effveda": "effveda",
    "veda": "veda",
    "role": "role",
    "rls": "rls",
    "hqi": "hqi",
    "qdtree": "hqi",
    "qd_tree": "hqi",
}


def parse_methods(values: list[str]) -> list[str]:
    methods: list[str] = []
    for value in values:
        for item in str(value).replace(",", " ").split():
            normalized = METHOD_ALIASES.get(item.strip().lower())
            if normalized is None:
                raise argparse.ArgumentTypeError(f"Unsupported method: {item}")
            if normalized not in methods:
                methods.append(normalized)
    return methods


def build_rls() -> None:
    disable_row_level_security()
    drop_database_users()
    create_database_users()
    enable_row_level_security()


def build_role(index_type: str, workers: int) -> None:
    drop_prefilter_partition_tables(condition="role")
    initialize_role_partitions(enable_index=True, index_type=index_type, max_workers=int(workers))


def build_ours(args: argparse.Namespace) -> None:
    efs_model_json = Path(args.ours_efs_model_json).expanduser()
    if not efs_model_json.is_absolute():
        efs_model_json = PROJECT_ROOT / efs_model_json
    if not efs_model_json.is_file():
        raise FileNotFoundError(f"OURS ef-search recall model not found: {efs_model_json}")

    os.environ["KMEANS_EFS_MODEL_JSON"] = str(efs_model_json)
    os.environ["KMEANS_TARGET_RECALL"] = str(float(args.ours_target_recall))
    private_budget_ratio = max(0.0, float(args.memory_ratio) - 1.0)
    print(
        "[build-current][ours] "
        f"memory_ratio={float(args.memory_ratio):.4f} total storage, "
        f"private_replication_budget_ratio={private_budget_ratio:.4f}",
        flush=True,
    )

    from controller.kmeans import build_and_materialize_kmeans_plan

    build_and_materialize_kmeans_plan(
        private_replication_budget_ratio=private_budget_ratio,
        ef_search=int(args.ours_cost_ef),
        embedding_dim=args.embedding_dim,
        document_limit=args.document_limit,
        query_dataset_path=args.query_dataset_path,
        create_indexes=True,
        index_type=str(args.ours_index_type),
        show_progress=bool(args.show_progress),
        enable_split=bool(args.enable_split),
        private_edge_top_d=int(args.private_edge_top_d),
        replace_current=True,
        drop_stale=True,
    )


def build_veda(method: str, args: argparse.Namespace) -> None:
    build_and_materialize_veda_plan(
        algorithm=method,
        indexing_threshold=int(args.veda_indexing_threshold),
        storage_amplification=float(args.memory_ratio),
        ef_search=int(args.ef_search),
        document_limit=args.document_limit,
        create_indexes=True,
        index_type=str(args.veda_index_type),
        show_progress=bool(args.show_progress),
        replace_current=True,
        drop_stale=True,
    )


def build_hqi(args: argparse.Namespace) -> None:
    python = str(args.python_bin)
    subprocess.run(
        [
            python,
            str(PROJECT_ROOT / "controller" / "baseline" / "HQI" / "build_tree.py"),
            "--min-size",
            str(int(args.hqi_min_size)),
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )
    subprocess.run(
        [
            python,
            str(PROJECT_ROOT / "controller" / "baseline" / "HQI" / "persist_tree.py"),
            "--index-type",
            str(args.hqi_index_type),
            "--workers",
            str(int(args.workers)),
        ],
        check=True,
        cwd=str(PROJECT_ROOT),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build current-style partitions for direct current-mode QPS.")
    parser.add_argument("--methods", nargs="+", required=True, help="Methods: rls role ours effveda veda hqi")
    parser.add_argument("--memory-ratio", type=float, default=2.0,
                        help="Total storage/memory ratio. OURS internally receives max(memory_ratio - 1, 0).")
    parser.add_argument("--ef-search", type=int, default=50,
                        help="Planning/search ef passed to current planners.")
    parser.add_argument("--ours-cost-ef", type=int, default=18,
                        help="Cost ef recorded/used by SQUID/OURS planning; default is the YFCC Recall@0.95 base ef.")
    parser.add_argument("--ours-efs-model-json", type=Path,
                        default=Path(os.environ.get("OURS_EFS_MODEL_JSON", str(DEFAULT_OURS_EFS_MODEL_JSON))),
                        help="Recall model JSON used by SQUID/OURS adaptive ef planning.")
    parser.add_argument("--ours-target-recall", type=float,
                        default=float(os.environ.get("OURS_TARGET_RECALL", "0.95")),
                        help="Target recall used by SQUID/OURS adaptive ef planning.")
    parser.add_argument("--document-limit", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--query-dataset-path", default=None)
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--role-index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    parser.add_argument("--ours-index-type", choices=["squidhnsw", "hnsw", "ivfflat"], default="squidhnsw")
    parser.add_argument("--veda-index-type", choices=["hnsw", "vedahnsw", "ivfflat"], default="hnsw")
    parser.add_argument("--veda-indexing-threshold", type=int, default=1000)
    parser.add_argument("--enable-split", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--private-edge-top-d", type=int, default=32)
    parser.add_argument("--hqi-min-size", type=int, default=512)
    parser.add_argument("--hqi-index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    parser.add_argument("--show-progress", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--python-bin", type=Path, default=PROJECT_ROOT / "venv" / "bin" / "python")
    args = parser.parse_args()

    methods = parse_methods(args.methods)
    for method in methods:
        print(f"[build-current] method={method}", flush=True)
        if method == "rls":
            build_rls()
        elif method == "role":
            build_role(args.role_index_type, args.workers)
        elif method == "ours":
            build_ours(args)
        elif method in {"veda", "effveda"}:
            build_veda(method, args)
        elif method == "hqi":
            build_hqi(args)
        else:
            raise RuntimeError(f"Unhandled method: {method}")
        print(f"[build-current] method={method} done", flush=True)


if __name__ == "__main__":
    main()
