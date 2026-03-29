import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.logger import get_logger
from controller.adaptive_tenant.load_result_to_database import (
    create_indexes_for_materialized_partitions,
    drop_indexes_for_materialized_partitions,
    load_current_partitions,
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
    parser = argparse.ArgumentParser(description='Build ANN indexes for existing AdaptiveTenant materialized partitions.')
    parser.add_argument('--index-type', choices=['hnsw', 'ivfflat'], default='hnsw', help='ANN index type to build for each adaptive partition table.')
    parser.add_argument('--rebuild', type=_str_to_bool, default=False, help='Whether to drop existing adaptive partition indexes before creating new ones.')
    args = parser.parse_args()

    partitions = load_current_partitions()
    if not partitions:
        raise RuntimeError(
            'No adaptive_tenant partitions are materialized. '
            'Run basic_benchmark/build_adaptive_tenant_partitions.py first.'
        )

    logger.info(
        'AdaptiveTenant index build requested: partitions=%s index_type=%s rebuild=%s',
        len(partitions),
        args.index_type,
        args.rebuild,
    )
    if args.rebuild:
        drop_indexes_for_materialized_partitions()
    create_indexes_for_materialized_partitions(index_type=args.index_type)
