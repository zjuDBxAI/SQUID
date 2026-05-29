import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from controller.latent_access.load_result_to_database import create_indexes_for_materialized_partitions


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Initialize LatentAccess materialized tables for testing.")
    parser.add_argument("--index-type", choices=["hnsw", "ivfflat"], default="hnsw")
    parser.add_argument("--create-indexes", type=_str_to_bool, default=False)
    args = parser.parse_args()

    if args.create_indexes:
        create_indexes_for_materialized_partitions(index_type=args.index_type)
