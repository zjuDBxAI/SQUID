import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from controller.baseline.prefilter.initialize_partitions import create_indexes_for_all_role_tables

if __name__ == "__main__":
    create_indexes_for_all_role_tables(
        index_type="hnsw",
        parallel=True,
        max_workers=max(1, os.cpu_count() // 2),
        hnsw_m=16,
        hnsw_ef_construction=64,
        disable_sync_commit=True,
    )
