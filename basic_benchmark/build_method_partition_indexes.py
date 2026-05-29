import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from controller.method import create_indexes_for_materialized_partitions


def _parse_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create indexes for materialized method_partition tables.")
    parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    parser.add_argument("--parallel", type=_parse_bool, default=True)
    parser.add_argument("--max-workers", type=int, default=max(1, os.cpu_count() // 2))
    parser.add_argument("--hnsw-m", type=int, default=16)
    parser.add_argument("--hnsw-ef-construction", type=int, default=64)
    parser.add_argument("--hnsw-threads", type=int, default=None)
    parser.add_argument("--disable-sync-commit", type=_parse_bool, default=True)
    args = parser.parse_args()

    create_indexes_for_materialized_partitions(
        index_type=args.index_type,
        parallel=bool(args.parallel),
        max_workers=max(1, int(args.max_workers)),
        hnsw_m=int(args.hnsw_m),
        hnsw_ef_construction=int(args.hnsw_ef_construction),
        hnsw_threads=args.hnsw_threads,
        disable_sync_commit=bool(args.disable_sync_commit),
    )


if __name__ == "__main__":
    main()
