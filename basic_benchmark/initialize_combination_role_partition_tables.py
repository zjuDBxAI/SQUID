import os
import sys
import time
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(sys.path)

from controller.baseline.prefilter.initialize_partitions import initialize_combination_partitions, \
    drop_prefilter_partition_tables

from controller.prepare_database import clear_db
from controller.initialize_main_tables import initialize_database_deduplication

if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description="Initialize combination partitions with optional index type.")
    parser.add_argument('--index_type', choices=['hnsw', 'ivfflat'], default=None,
                        help="Type of index to use. Valid options are 'hnsw' and 'ivfflat'.")

    # Parse arguments
    args = parser.parse_args()

    # Set index_type and enable_index based on input
    index_type = args.index_type
    enable_index = index_type is not None

    # Inform the user of the chosen settings
    if enable_index:
        print(f"Index type: {index_type}, Enable index: {enable_index}")
    else:
        print("No index_type provided, setting enable_index to False.")

    # Drop combination partition tables
    drop_prefilter_partition_tables(condition="combination")

    # Measure time for initializing combination partitions
    start_time = time.time()
    initialize_combination_partitions(enable_index=enable_index, index_type=index_type)
    time_taken = time.time() - start_time
    print(f"initialize_combination_partition time cost: {time_taken} seconds")