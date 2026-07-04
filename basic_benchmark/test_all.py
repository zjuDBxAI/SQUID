import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.logger import get_logger

logger = get_logger(__name__)
logger.debug("sys.path=%s", sys.path)
from basic_benchmark.test_partition_prefilter_by_combination_role import test_partition_prefilter_by_combination_role
from basic_benchmark.test_partition_prefilter_by_role import test_partition_prefilter_role
from basic_benchmark.space_calculate import (
    calculate_prefilter,
    calculate_rls,
    calculate_dynamic_partition,
    calculate_qd_tree_storage,
)

from basic_benchmark.test_dynamic_partition import test_dynamic_partition_search
from basic_benchmark.test_row_level_security import test_row_level_security
from basic_benchmark.test_qd_tree_partition import test_qd_tree_partition_search
import efconfig


def _str_to_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {'1', 'true', 't', 'yes', 'y', 'on'}:
        return True
    if lowered in {'0', 'false', 'f', 'no', 'n', 'off'}:
        return False
    raise ValueError(f'Invalid boolean value: {value}')


# python test_all.py --algorithm AnonySys --efs 40
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Run benchmark tests with partition and index strategies.')

    parser.add_argument(
        '--algorithm',
        choices=['RLS', 'ROLE', 'USER', 'AnonySys', 'QDTree'],
        required=True,
        help='Select which test to run: RLS, ROLE, USER, AnonySys, or QDTree',
    )
    parser.add_argument(
        '--efs',
        type=int,
        nargs='+',
        required=True,
        help='List of EF search values to use (space-separated integers).',
    )
    parser.add_argument('--enable-index', type=_str_to_bool, default=True, help='Whether to build ANN indexes for the selected method.')
    parser.add_argument('--index-type', choices=['hnsw', 'ivfflat'], default='hnsw', help='ANN index type used by the selected method.')
    parser.add_argument('--statistics-type', choices=['sql', 'system'], default='sql', help='Latency source: EXPLAIN SQL time or wall-clock system time.')
    parser.add_argument('--generator-type', default='', help='Tag included in output filenames.')
    parser.add_argument('--iterations', type=int, default=1, help='Repeat each query this many times and average the result.')
    parser.add_argument('--query-num', type=int, default=1000, help='Number of queries to run for QDTree.')
    parser.add_argument('--record-recall', type=_str_to_bool, default=True, help='Whether to compute recall against ground truth.')
    parser.add_argument('--warm-up', type=_str_to_bool, default=True, help='Whether to warm up each query before measuring.')

    args = parser.parse_args()
    enable_index = args.enable_index
    index_type = args.index_type
    generator_type = args.generator_type
    statistics_type = args.statistics_type
    iterations = args.iterations
    query_num = args.query_num
    record_recall = args.record_recall
    warm_up = args.warm_up

    test_type = args.algorithm
    ef_search_values = args.efs

    logger.info('Test Type: %s', test_type)
    logger.info('EF Search Values: %s', ef_search_values)
    logger.info('Index Type: %s', index_type)
    logger.info('Enable Index: %s', enable_index)

    if test_type == 'RLS':
        for ef in ef_search_values:
            efconfig.ef_search = ef
            logger.info('Running RLS test with ef_search=%s', ef)
            test_row_level_security(
                iterations=iterations,
                enable_index=enable_index,
                statistics_type=statistics_type,
                index_type=index_type,
                generator_type=generator_type,
                warm_up=warm_up,
            )
            rls_space_mb = calculate_rls('row_level_security', enable_index=enable_index)
            logger.info('RLS storage footprint: %.2f MB', rls_space_mb)

    elif test_type == 'ROLE':
        for ef in ef_search_values:
            efconfig.ef_search = ef
            logger.info('Running ROLE test with ef_search=%s', ef)
            test_partition_prefilter_role(
                iterations=iterations,
                enable_index=enable_index,
                index_type=index_type,
                statistics_type=statistics_type,
                generator_type=generator_type,
                record_recall=record_recall,
                warm_up=warm_up,
            )
            role_space_mb = calculate_prefilter('prefilter_partition_role', enable_index=enable_index)
            logger.info('Role partition storage footprint: %.2f MB', role_space_mb)

    elif test_type == 'USER':
        for ef in ef_search_values:
            efconfig.ef_search = ef
            logger.info('Running USER test with ef_search=%s', ef)
            test_partition_prefilter_by_combination_role(
                iterations=iterations,
                enable_index=enable_index,
                index_type=index_type,
                statistics_type=statistics_type,
                generator_type=generator_type,
                record_recall=record_recall,
                warm_up=warm_up,
            )
            comb_space_mb = calculate_prefilter('prefilter_partition_combination', enable_index=enable_index)
            logger.info('Combination partition storage footprint: %.2f MB', comb_space_mb)

    elif test_type == 'AnonySys':
        for ef in ef_search_values:
            efconfig.ef_search = ef
            logger.info('Running AnonySys test with ef_search=%s', ef)
            test_dynamic_partition_search(
                iterations=iterations,
                enable_index=enable_index,
                statistics_type=statistics_type,
                index_type=index_type,
                generator_type=generator_type,
                record_recall=record_recall,
                warm_up=warm_up,
            )
            dynamic_space_mb = calculate_dynamic_partition('dynamic_partition', enable_index=enable_index)
            logger.info('Dynamic partition storage footprint: %.2f MB', dynamic_space_mb)

    elif test_type == 'QDTree':
        for ef in ef_search_values:
            efconfig.ef_search = ef
            logger.info('Running QDTree test with ef_search=%s', ef)
            test_qd_tree_partition_search(
                iterations=iterations,
                enable_index=enable_index,
                statistics_type=statistics_type,
                index_type=index_type,
                generator_type=generator_type,
                record_recall=record_recall,
                warm_up=warm_up,
                query_num=query_num,
            )
            qdt_space_mb = calculate_qd_tree_storage('qd_tree_partition', enable_index=enable_index)
            logger.info('QDTree partition storage footprint: %.2f MB', qdt_space_mb)
