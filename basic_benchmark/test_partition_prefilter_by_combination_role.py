import os
import sys
import json
from operator import index

import psycopg2

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULT_DIR = os.path.join(project_root, "basic_benchmark", "result")
os.makedirs(RESULT_DIR, exist_ok=True)

sys.path.append(project_root)

from services.logger import get_logger

logger = get_logger(__name__)
logger.debug("sys.path=%s", sys.path)

from services.config import get_db_connection
from basic_benchmark.common_function import prepare_query_dataset, run_search_experiment, drop_extra_tables, run_test, \
    get_index_type
from controller.baseline.prefilter.initialize_partitions import initialize_user_partitions, initialize_role_partitions, \
    initialize_combination_partitions, drop_prefilter_partition_tables, create_indexes_for_all_role_tables, \
    drop_indexes_for_all_role_tables, drop_indexes_for_all_combination_tables, create_indexes_for_all_combination_tables
from services.config import get_db_connection


def get_existing_combination_partition_name():
    """
    Retrieve the name of the first existing combination partition table.

    Returns:
        str: The name of the first combination partition table, or None if no such table exists.
    """
    conn = get_db_connection()
    cur = conn.cursor()

    try:
        # Query for all tables matching the combination partition naming pattern
        cur.execute("""
            SELECT tablename
            FROM pg_tables
            WHERE tablename LIKE 'documentblocks_combination_%'
            ORDER BY tablename LIMIT 1;
        """)
        result = cur.fetchone()

        return result[0] if result else None

    except psycopg2.Error as e:
        logger.error("Error retrieving combination partition names: %s", e)
        return None
    finally:
        cur.close()
        conn.close()

def test_partition_prefilter_by_combination_role(iterations=3, enable_index=False, index_type="ivfflat",
                             statistics_type="sql",
                             generator_type="tree-based", record_recall=True, query_num=1000, warm_up=True):
    combination_table = get_existing_combination_partition_name()
    current_index_type = get_index_type(combination_table)

    if enable_index:
        if current_index_type is not None and current_index_type != index_type:
            logger.info(
                "Index type %s does not match requested type %s, recreating index.",
                current_index_type,
                index_type,
            )
            drop_indexes_for_all_combination_tables()

        import time
        start_time = time.time()
        create_indexes_for_all_combination_tables(index_type)
        time_taken = time.time() - start_time
        logger.info(
            "initialize prefilter combination role partition indexing time cost: %.2f seconds",
            time_taken,
        )
        #with open("indexed_time.json", "w") as json_file:
        with open(os.path.join(project_root, "basic_benchmark", "result", "indexed_time.json"), "w") as json_file:
            json.dump(time_taken, json_file, indent=4)
    else:
        drop_indexes_for_all_combination_tables()

    # Generate queries
    queries = prepare_query_dataset(regenerate=False, num_queries=max(1, int(query_num)))

    run_test(queries, f"prefilter_partition_combination", iterations=iterations,
             enable_index=enable_index,
             statistics_type=statistics_type, generator_type=generator_type, index_type=index_type,
             record_recall=record_recall, warm_up=warm_up)


if __name__ == '__main__':
    test_partition_prefilter_by_combination_role(enable_index=False, statistics_type="sql")
