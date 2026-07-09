#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/data/Multitenanthakes"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

run_step() {
  local label="$1"
  shift
  echo "[treebase] ${label}"
  "$@"
}

run_step "Generate random RBAC data"   "${PYTHON_BIN}" services/rbac_generator/store_tree_based_rbac_generate_data.py

#run_step "Build HQI tree"   "${PYTHON_BIN}" controller/baseline/HQI/build_tree.py --min-size 10000

#run_step "Persist HQI tree"   "${PYTHON_BIN}" controller/baseline/HQI/persist_tree.py --workers 8

#run_step "Initialize role partition HNSW indexes"   "${PYTHON_BIN}" basic_benchmark/initialize_role_partition_tables.py --index_type hnsw

run_step "Generate benchmark queries"   "${PYTHON_BIN}" basic_benchmark/generate_queries.py --num_queries 1000 --topk 10 --num_threads 4

run_step "Compute ground truth"   "${PYTHON_BIN}" basic_benchmark/compute_ground_truth.py

#run_step "Build AnonySys dynamic partition"   "${PYTHON_BIN}" controller/dynamic_partition/hnsw/AnonySys_dynamic_partition.py --storage 1.5 --recall 0.99
