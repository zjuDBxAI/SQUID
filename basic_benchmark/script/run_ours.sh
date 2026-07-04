#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/chenyang/.conda/envs/multitenant/bin/python}"

PRIVATE_REPLICATION_BUDGET_RATIO="${PRIVATE_REPLICATION_BUDGET_RATIO:-2.00}"
PRIVATE_EDGE_TOP_D="${PRIVATE_EDGE_TOP_D:-32}"
ENABLE_SPLIT="${ENABLE_SPLIT:-false}"
ENABLE_INDEX="${ENABLE_INDEX:-true}"
INDEX_TYPE="${INDEX_TYPE:-hnsw}"
STATISTICS_TYPE="${STATISTICS_TYPE:-sql}"
QUERY_NUM="${QUERY_NUM:-200}"
ITERATIONS="${ITERATIONS:-1}"
RECORD_RECALL="${RECORD_RECALL:-true}"
WARM_UP="${WARM_UP:-true}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"
USE_GROUND_TRUTH_CACHE="${USE_GROUND_TRUTH_CACHE:-true}"
RESULT_TAG="${RESULT_TAG:-OURS}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/efs_logs}"
EFS_LIST="${EFS_LIST:-25}"

# This script only evaluates existing materialized kmeans partitions.
# It intentionally passes --prepare false for every ef_search value.
IFS=' ' read -r -a EFS_VALUES <<< "${EFS_LIST//,/ }"

METHOD_LOG_DIR="${LOG_DIR}/OURS"
mkdir -p "${METHOD_LOG_DIR}"

for EFS in "${EFS_VALUES[@]}"; do
  EFFECTIVE_RESULT_TAG="${RESULT_TAG}_ef${EFS}"
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${METHOD_LOG_DIR}/${EFFECTIVE_RESULT_TAG}_${TS}.log"

  echo "========================================"
  echo "[run_ours] testing ef_search=${EFS}, result_tag=${EFFECTIVE_RESULT_TAG}, prepare=false"
  echo "[log] ${LOG_FILE}"

 python test_kmeans_partition.py \
    --prepare false \
    --private-replication-budget-ratio "${PRIVATE_REPLICATION_BUDGET_RATIO}" \
    --private-edge-top-d "${PRIVATE_EDGE_TOP_D}" \
    --enable-split "${ENABLE_SPLIT}" \
    --enable-index "${ENABLE_INDEX}" \
    --index-type "${INDEX_TYPE}" \
    --statistics-type "${STATISTICS_TYPE}" \
    --query-num "${QUERY_NUM}" \
    --iterations "${ITERATIONS}" \
    --record-recall "${RECORD_RECALL}" \
    --warm-up "${WARM_UP}" \
    --ef-search "${EFS}" \
    --show-progress "${SHOW_PROGRESS}" \
    --use-ground-truth-cache "${USE_GROUND_TRUTH_CACHE}" \
    --result-tag "${EFFECTIVE_RESULT_TAG}" \
    > "${LOG_FILE}" 2>&1

done
