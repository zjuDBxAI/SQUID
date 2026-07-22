import os
import random
import sys
import time
import psycopg2
from psycopg2 import sql


# Add the project root to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.logger import get_logger

logger = get_logger(__name__)
logger.debug("sys.path=%s", sys.path)
from controller.dynamic_partition.load_result_to_database import drop_indexes_for_all_partitions, \
    create_indexes_for_all_partitions
from controller.baseline.pg_row_security.row_level_security import drop_database_users, create_database_users
from controller.initialize_main_tables import drop_indexes, create_indexes
from basic_benchmark.common_function import get_index_type, prepare_query_dataset, run_test
from services.config import get_db_connection


def test_dynamic_partition_search(iterations=1, enable_index=True, index_type="hnsw", statistics_type="sql",
                                  generator_type="tree-based", record_recall=True, query_num=1000, warm_up=True):
    """
    Test search across partitions with optional index creation and verification.
    """
    table_prefix = os.environ.get("HONEYBEE_TABLE_PREFIX") or os.environ.get("DYNAMIC_PARTITION_TABLE_PREFIX")
    sample_table = f"{table_prefix}_partition_0" if table_prefix else "documentblocks_partition_0"
    current_index_type = get_index_type(sample_table)
    skip_index_maintenance = str(os.environ.get("SKIP_INDEX_MAINTENANCE", "")).strip().lower() in {
        "1",
        "true",
        "t",
        "yes",
        "y",
        "on",
    }

    if skip_index_maintenance:
        logger.info("Skipping dynamic partition index maintenance; using existing indexes.")
    elif enable_index:
        if current_index_type is not None and current_index_type != index_type:
            logger.info(
                "Index type %s does not match %s. Recreating index.",
                current_index_type,
                index_type,
            )
            drop_indexes_for_all_partitions()
        create_indexes_for_all_partitions(index_type, table_prefix=table_prefix)
    else:
        drop_indexes_for_all_partitions()

    # Generate queries
    queries = prepare_query_dataset(regenerate=False, num_queries=max(1, int(query_num)))

    run_test(queries, f"dynamic_partition", iterations=iterations,
             enable_index=enable_index,
             statistics_type=statistics_type, generator_type=generator_type, index_type=index_type,
             record_recall=record_recall, warm_up=warm_up)


if __name__ == '__main__':
    test_dynamic_partition_search()
