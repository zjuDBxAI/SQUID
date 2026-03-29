import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.logger import get_logger
from controller.adaptive_tenant.load_result_to_database import build_and_materialize_adaptive_plan

logger = get_logger(__name__)


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if lowered in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Build and materialize AdaptiveTenant partitions only.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Global memory budget ratio.')
    parser.add_argument('--topk', type=int, default=10, help='Planner top-k for cost estimation.')
    parser.add_argument('--window-limit', type=int, default=10, help='EMA window depth per tenant.')
    parser.add_argument('--max-split-actions', type=int, default=0, help='How many split actions to apply after initialization.')
    parser.add_argument('--index-type', choices=['hnsw', 'ivfflat'], default='hnsw', help='Index type for materialized adaptive partition tables.')
    parser.add_argument('--create-indexes', type=_str_to_bool, default=False, help='Whether to build ANN indexes during materialization. Default is false because index build is split into basic_benchmark/build_adaptive_tenant_indexes.py.')
    parser.add_argument('--enable-rls', type=_str_to_bool, default=False, help='Whether to enable PostgreSQL RLS on adaptive partition tables.')
    args = parser.parse_args()

    result = build_and_materialize_adaptive_plan(
        alpha=args.alpha,
        topk=args.topk,
        window_limit=args.window_limit,
        index_type=args.index_type,
        create_indexes=args.create_indexes,
        enable_rls=args.enable_rls,
        max_split_actions=args.max_split_actions,
    )
    logger.info(
        'AdaptiveTenant build complete: partitions=%s current_memory=%.2f limit=%.2f total_query_cost=%.4f',
        len(result.partitions),
        result.budget.current_memory,
        result.budget.memory_limit,
        result.total_query_cost,
    )
