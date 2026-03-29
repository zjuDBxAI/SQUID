"""Benchmark harness for QD-tree based partitions."""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

from services.logger import get_logger

logger = get_logger(__name__)
logger.debug("sys.path=%s", sys.path)

from basic_benchmark.common_function import get_index_type, prepare_query_dataset, run_test
from controller.baseline.HQI.qd_tree import (
    DEFAULT_QD_TREE_PARTITION_PREFIX,
    create_indexes_for_qdtree_partitions,
    drop_indexes_for_qdtree_partitions,
)


def test_qd_tree_partition_search(
    iterations: int = 1,
    enable_index: bool = True,
    index_type: str = "hnsw",
    statistics_type: str = "sql",
    generator_type: str = "tree-based",
    record_recall: bool = True,
    warm_up: bool = True,
    partition_prefix: str = DEFAULT_QD_TREE_PARTITION_PREFIX,
):
    """Run the QD-tree partition benchmark under the basic_benchmark harness."""
    sample_table = f"{partition_prefix}_0"
    current_index_type = get_index_type(sample_table)

    if enable_index:
        if current_index_type is not None and current_index_type != index_type:
            logger.info(
                "Index type %s does not match requested %s on %s. Dropping existing indexes.",
                current_index_type,
                index_type,
                partition_prefix,
            )
            drop_indexes_for_qdtree_partitions(partition_prefix)
        create_indexes_for_qdtree_partitions(index_type=index_type, partition_prefix=partition_prefix)
    else:
        drop_indexes_for_qdtree_partitions(partition_prefix)

    queries = prepare_query_dataset(regenerate=False, num_queries=1000)

    run_test(
        queries,
        "qd_tree_partition",
        iterations=iterations,
        enable_index=enable_index,
        statistics_type=statistics_type,
        generator_type=generator_type,
        index_type=index_type,
        record_recall=record_recall,
        warm_up=warm_up,
    )


if __name__ == "__main__":
    test_qd_tree_partition_search()
