import os
import sys
import time
import argparse

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(sys.path)

from controller.baseline.prefilter.initialize_partitions import initialize_role_partitions, \
    drop_prefilter_partition_tables



if __name__ == '__main__':
    # Create argument parser
    parser = argparse.ArgumentParser(description="Initialize role partitions with optional index type.")
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

    # Drop prefilter partition tables
    drop_prefilter_partition_tables(condition="role")

    # Measure time for initializing role partitions
    start_time = time.time()
    initialize_role_partitions(enable_index=enable_index, index_type=index_type)
    time_taken = time.time() - start_time
    print(f"initialize_role_partition time cost: {time_taken} seconds")