import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from controller.initialize_main_tables import create_indexes


if __name__ == "__main__":
    create_indexes(
        index_type="hnsw",
        hnsw_m=16,
        hnsw_ef_construction=64,
        disable_sync_commit=True,
    )
