import os
import sys
import time
import argparse
from operator import index

from controller.baseline.pg_row_security.row_level_security import drop_database_users, create_database_users
from controller.dynamic_partition.load_result_to_database import disable_rls_for_partitions, \
    create_indexes_for_all_partitions

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(sys.path)

from controller.initialize_main_tables import create_indexes


def initialize_dynamic_partition_tables_in_comb(index_type = None):
    from controller.dynamic_partition.load_result_to_database import initialize_rls_for_partitions

    disable_rls_for_partitions()
    drop_database_users()

    create_database_users()
    initialize_rls_for_partitions()

    if index_type is not None:
        create_indexes_for_all_partitions(index_type)


if __name__ == '__main__':
    initialize_dynamic_partition_tables_in_comb()