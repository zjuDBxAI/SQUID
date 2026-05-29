import os
import sys
import json

from controller.baseline.pg_row_security.row_level_security import search_documents_rls_statistics_sql

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

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
    "method_partition": {
        "search_func_path": "controller.method.dynamic_partition_search",
        "space_calc_func_path": "basic_benchmark.space_calculate.calculate_method_partition",
        "extra_params": {"queries_num": 1000}
    },
    "kmeans_partition": {
        "search_func_path": "controller.kmeans.kmeans_partition_search",
        "space_calc_func_path": "basic_benchmark.space_calculate.calculate_kmeans_partition",
        "extra_params": {"queries_num": 1000}
    },
    "latent_access": {
        "search_func_path": "controller.latent_access.search.latent_access_search",
        "space_calc_func_path": "basic_benchmark.space_calculate.calculate_latent_access",
        "extra_params": {"queries_num": 1000}
    },
    "qd_tree_partition": {
        "search_func_path": "controller.baseline.HQI.qd_tree.qd_tree_search",
        "space_calc_func_path": "basic_benchmark.space_calculate.calculate_qd_tree_storage",
        "extra_params": {"queries_num": 1000}
    },
    "sieve": {
        "search_func_path": "controller.baseline.SIEVE.sieve_search",
        "space_calc_func_path": "basic_benchmark.space_calculate.calculate_sieve",
        "extra_params": {"queries_num": 1000}
    },
}
