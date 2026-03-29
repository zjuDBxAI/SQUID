import os
import sys
import json

from controller.baseline.pg_row_security.row_level_security import search_documents_rls_statistics_sql

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
print(sys.path)

# Define the mapping outside the function for better organization
CONDITION_CONFIG = {
    "prefilter_partition_role": {
        "search_func_path": "controller.baseline.prefilter.prefilter_role.search_documents_role_partition",
        "space_calc_func_path": "basic_benchmark.space_calculate.calculate_prefilter",
        "extra_params": {"queries_num": 200}
    },
    "prefilter_partition_combination": {
        "search_func_path": "controller.baseline.prefilter.prefilter_combination_role.search_documents_combination_partition",
        "space_calc_func_path": "basic_benchmark.space_calculate.calculate_prefilter",
        "extra_params": {"queries_num": 1000}
    },
    "row_level_security": {
        "search_func_path": "controller.baseline.pg_row_security.row_level_security.search_documents_rls",
        "space_calc_func_path": "basic_benchmark.space_calculate.calculate_rls",
        "extra_params": {"queries_num": 200}
    },
    "dynamic_partition": {
        "search_func_path": "controller.dynamic_partition.search.dynamic_partition_search",
        "space_calc_func_path": "basic_benchmark.space_calculate.calculate_dynamic_partition",
        "extra_params": {"queries_num": 1000}
    },
    "adaptive_tenant": {
        "search_func_path": "controller.adaptive_tenant.search.adaptive_tenant_search",
        "space_calc_func_path": "basic_benchmark.space_calculate.calculate_adaptive_tenant",
        "extra_params": {"queries_num": 1000}
    },
    "qd_tree_partition": {
        "search_func_path": "controller.baseline.HQI.qd_tree.qd_tree_search",
        "space_calc_func_path": "basic_benchmark.space_calculate.calculate_qd_tree_storage",
        "extra_params": {"queries_num": 1000}
    },
}
