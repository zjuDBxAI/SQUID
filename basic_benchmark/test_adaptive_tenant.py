import argparse
import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.logger import get_logger

logger = get_logger(__name__)
logger.debug("sys.path=%s", sys.path)

from controller.adaptive_tenant.load_result_to_database import (
    build_and_materialize_adaptive_plan,
    drop_indexes_for_materialized_partitions,
    get_partition_table_name,
    load_current_partitions,
)
from basic_benchmark.common_function import get_index_type, prepare_query_dataset, run_test
import efconfig


def _collect_index_state(partitions):
    state = {
        'missing': [],
        'by_type': {},
    }
    for partition in partitions:
        table_name = partition.metadata.get('table_name', get_partition_table_name(partition.partition_id))
        index_type = get_index_type(table_name)
        if index_type is None:
            state['missing'].append(table_name)
            continue
        state['by_type'].setdefault(index_type, []).append(table_name)
    return state


def _validate_index_state(partitions, index_type: str) -> None:
    state = _collect_index_state(partitions)
    missing = state['missing']
    wrong_type_tables = []
    for actual_type, table_names in state['by_type'].items():
        if actual_type != index_type:
            wrong_type_tables.extend(table_names)

    if not missing and not wrong_type_tables:
        logger.info(
            'Using existing %s indexes on %s AdaptiveTenant materialized partitions.',
            index_type,
            len(partitions),
        )
        return

    sample_missing = ', '.join(missing[:3])
    sample_wrong = ', '.join(wrong_type_tables[:3])
    parts = []
    if missing:
        parts.append(f'missing indexes on {len(missing)} partitions' + (f' (e.g. {sample_missing})' if sample_missing else ''))
    if wrong_type_tables:
        parts.append(f'wrong index type on {len(wrong_type_tables)} partitions' + (f' (e.g. {sample_wrong})' if sample_wrong else ''))
    raise RuntimeError(
        'AdaptiveTenant indexes are not ready: ' + '; '.join(parts) + '. '
        'Run basic_benchmark/build_adaptive_tenant_indexes.py first.'
    )


def test_adaptive_tenant_search(
    iterations=1,
    enable_index=True,
    index_type='hnsw',
    statistics_type='sql',
    generator_type='',
    record_recall=True,
    warm_up=True,
    alpha=0.5,
    topk=10,
    window_limit=10,
    max_split_actions=0,
    enable_rls=False,
    query_num=1000,
    prepare_before_test=False,
    ef_search='adaptive',
):
    """Run AdaptiveTenant benchmark queries on existing or freshly prepared partitions."""
    result = None
    if prepare_before_test:
        logger.info(
            'Preparing adaptive partitions only: alpha=%s topk=%s window_limit=%s max_split_actions=%s',
            alpha,
            topk,
            window_limit,
            max_split_actions,
        )
        result = build_and_materialize_adaptive_plan(
            alpha=alpha,
            topk=topk,
            window_limit=window_limit,
            index_type=index_type,
            create_indexes=False,
            enable_rls=enable_rls,
            max_split_actions=max_split_actions,
        )
        logger.info(
            'AdaptiveTenant partitions materialized: partitions=%s current_memory=%.2f limit=%.2f total_query_cost=%.4f',
            len(result.partitions),
            result.budget.current_memory,
            result.budget.memory_limit,
            result.total_query_cost,
        )

    partitions = load_current_partitions()
    if not partitions:
        raise RuntimeError(
            'No adaptive_tenant partitions are materialized. '
            'Run basic_benchmark/build_adaptive_tenant_partitions.py first, '
            'or rerun this script with --prepare true.'
        )

    if enable_index:
        _validate_index_state(partitions, index_type)
    else:
        logger.info('Dropping ANN indexes from %s materialized partitions before testing.', len(partitions))
        drop_indexes_for_materialized_partitions()

    logger.info('Running AdaptiveTenant benchmark on %s materialized partitions.', len(partitions))
    efconfig.ef_search = ef_search
    logger.info('AdaptiveTenant ef_search mode: %s', efconfig.ef_search)
    queries = prepare_query_dataset(regenerate=False, num_queries=query_num)
    run_test(
        queries,
        'adaptive_tenant',
        iterations=iterations,
        enable_index=enable_index,
        statistics_type=statistics_type,
        generator_type=generator_type,
        index_type=index_type,
        record_recall=record_recall,
        warm_up=warm_up,
    )


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if lowered in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise argparse.ArgumentTypeError(f'Invalid boolean value: {value}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the AdaptiveTenant benchmark pipeline.')
    parser.add_argument('--iterations', type=int, default=1, help='Repeat each query this many times and average the latency.')
    parser.add_argument('--enable-index', type=_str_to_bool, default=True, help='Whether the benchmark should use existing ANN indexes on adaptive partitions.')
    parser.add_argument('--index-type', choices=['hnsw', 'ivfflat'], default='hnsw', help='Expected ANN index type on each adaptive partition table.')
    parser.add_argument('--statistics-type', choices=['sql', 'system'], default='sql', help='Measure latency using EXPLAIN SQL time or wall-clock system time.')
    parser.add_argument('--generator-type', default='', help='Tag written into output JSON filenames.')
    parser.add_argument('--record-recall', type=_str_to_bool, default=True, help='Whether to compute recall against ground truth.')
    parser.add_argument('--warm-up', type=_str_to_bool, default=True, help='Whether to run warm-up searches before measuring each query.')
    parser.add_argument('--alpha', type=float, default=0.5, help='Global memory budget ratio. Larger alpha allows more dedicated partitions.')
    parser.add_argument('--topk', type=int, default=10, help='Top-k used by the planner when estimating ef_search and query cost.')
    parser.add_argument('--window-limit', type=int, default=10, help='How many recent workload windows to read per tenant when building EMA statistics.')
    parser.add_argument('--max-split-actions', type=int, default=0, help='How many periodic split operations the planner may apply after initialization.')
    parser.add_argument('--enable-rls', type=_str_to_bool, default=False, help='Whether to enable PostgreSQL RLS on adaptive partition tables after materialization.')
    parser.add_argument('--query-num', type=int, default=1000, help='How many benchmark queries to load from query_dataset.json.')
    parser.add_argument('--prepare', type=_str_to_bool, default=False, help='Whether to rebuild/materialize adaptive partitions before testing. This step no longer builds indexes.')
    parser.add_argument('--ef-search', default='adaptive', help='HNSW ef_search used during testing. Use an integer like 40 for fixed ef, or adaptive for model-based ef.')
    args = parser.parse_args()

    test_adaptive_tenant_search(
        iterations=args.iterations,
        enable_index=args.enable_index,
        index_type=args.index_type,
        statistics_type=args.statistics_type,
        generator_type=args.generator_type,
        record_recall=args.record_recall,
        warm_up=args.warm_up,
        alpha=args.alpha,
        topk=args.topk,
        window_limit=args.window_limit,
        max_split_actions=args.max_split_actions,
        enable_rls=args.enable_rls,
        query_num=args.query_num,
        prepare_before_test=args.prepare,
        ef_search=args.ef_search,
    )
