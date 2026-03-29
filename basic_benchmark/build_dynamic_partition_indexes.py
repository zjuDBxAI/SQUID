import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from controller.dynamic_partition.load_result_to_database import create_indexes_for_all_partitions

if __name__ == "__main__":
    create_indexes_for_all_partitions(
        index_type="hnsw",
        parallel=True,
        max_workers=max(1, os.cpu_count() // 2),
        disable_sync_commit=True,
    )
