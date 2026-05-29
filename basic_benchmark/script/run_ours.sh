#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/chenyang/.conda/envs/multitenant/bin/python}"

PREPARE="${PREPARE:-false}"
QUERY_NUM="${QUERY_NUM:-1000}"
INDEX_TYPE="${INDEX_TYPE:-hnsw}"
STATISTICS_TYPE="${STATISTICS_TYPE:-sql}"
RECORD_RECALL="${RECORD_RECALL:-true}"
WARM_UP="${WARM_UP:-true}"
ENABLE_INDEX="${ENABLE_INDEX:-true}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"
USE_GROUND_TRUTH_CACHE="${USE_GROUND_TRUTH_CACHE:-true}"
CLUSTER_COUNT="${CLUSTER_COUNT:-1000}"
PRIVATE_CLUSTER_COUNT="${PRIVATE_CLUSTER_COUNT:-1000}"
SHARED_CLUSTER_COUNT="${SHARED_CLUSTER_COUNT:-1000}"
SHARED_SCORE_RATIO="${SHARED_SCORE_RATIO:-0.20}"
SHARED_ROUTE_LIMIT="${SHARED_ROUTE_LIMIT:-3}"
PRIVATE_REPLICATION_BUDGET_RATIO="${PRIVATE_REPLICATION_BUDGET_RATIO:-1.50}"
FETCH_MULTIPLIER="${FETCH_MULTIPLIER:-1}"
HNSW_ITERATIVE_SCAN="${HNSW_ITERATIVE_SCAN:-off}"
HNSW_MAX_SCAN_TUPLES="${HNSW_MAX_SCAN_TUPLES:-20000}"
RESULT_TAG_PREFIX="${RESULT_TAG_PREFIX:-kmeans_efs}"

# The first ef value is used to build and materialize the plan once.
PLAN_EF_SEARCH="${PLAN_EF_SEARCH:-40}"

# Space-separated list of ef values to benchmark.
EFS_LIST=(450 500 550 600)

LOG_DIR="${ROOT_DIR}/efs_logs"
mkdir -p "${LOG_DIR}"

run_case() {
  local efs="$1"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local log_file="${LOG_DIR}/${RESULT_TAG_PREFIX}_efs${efs}_${ts}.log"
  local result_tag="${RESULT_TAG_PREFIX}_efs${efs}"

  echo "========================================"
  echo "[RUN] ef-search=${efs}"
  echo "[LOG] ${log_file}"

  "${PYTHON_BIN}" test_kmeans_partition.py \
    --prepare "${PREPARE}" \
    --enable-index "${ENABLE_INDEX}" \
    --index-type "${INDEX_TYPE}" \
    --statistics-type "${STATISTICS_TYPE}" \
    --query-num "${QUERY_NUM}" \
    --cluster-count "${CLUSTER_COUNT}" \
    --private-cluster-count "${PRIVATE_CLUSTER_COUNT}" \
    --shared-cluster-count "${SHARED_CLUSTER_COUNT}" \
    --shared-score-ratio "${SHARED_SCORE_RATIO}" \
    --shared-route-limit "${SHARED_ROUTE_LIMIT}" \
    --private-replication-budget-ratio "${PRIVATE_REPLICATION_BUDGET_RATIO}" \
    --fetch-multiplier "${FETCH_MULTIPLIER}" \
    --ef-search "${efs}" \
    --hnsw-iterative-scan "${HNSW_ITERATIVE_SCAN}" \
    --hnsw-max-scan-tuples "${HNSW_MAX_SCAN_TUPLES}" \
    --show-progress "${SHOW_PROGRESS}" \
    --use-ground-truth-cache "${USE_GROUND_TRUTH_CACHE}" \
    --result-tag "${result_tag}" \
    > "${log_file}" 2>&1

  echo "[DONE] ef-search=${efs}"
}

first_run=1
for efs in "${EFS_LIST[@]}"; do
  if [[ "${first_run}" -eq 1 ]]; then
    run_case "${efs}" true
    first_run=0
  else
    run_case "${efs}" false
  fi
done

echo "All kmeans partition ef runs finished."
