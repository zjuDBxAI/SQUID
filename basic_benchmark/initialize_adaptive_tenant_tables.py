import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.logger import get_logger
from controller.adaptive_tenant.load_result_to_database import (
    create_indexes_for_materialized_partitions,
    disable_rls_for_materialized_partitions,
    initialize_rls_for_materialized_partitions,
)

logger = get_logger(__name__)


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if lowered in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Initialize existing AdaptiveTenant materialized tables for testing.')
    parser.add_argument('--index-type', choices=['hnsw', 'ivfflat'], default='hnsw', help='Index type to build on existing adaptive partition tables.')
    parser.add_argument('--create-indexes', type=_str_to_bool, default=False, help='Whether to build ANN indexes for existing adaptive partition tables. Default is false because index build is split into basic_benchmark/build_adaptive_tenant_indexes.py.')
    parser.add_argument('--enable-rls', type=_str_to_bool, default=False, help='Whether to enable PostgreSQL RLS on existing adaptive partition tables.')
    args = parser.parse_args()

    disable_rls_for_materialized_partitions()
    if args.create_indexes:
        create_indexes_for_materialized_partitions(index_type=args.index_type)
    if args.enable_rls:
        initialize_rls_for_materialized_partitions()
    logger.info(
        'AdaptiveTenant initialization complete: create_indexes=%s index_type=%s enable_rls=%s',
        args.create_indexes,
        args.index_type,
        args.enable_rls,
    )
