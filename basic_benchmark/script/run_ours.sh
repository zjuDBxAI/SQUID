#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/chenyang/.conda/envs/multitenant/bin/python}"

PLAN_MODE="${PLAN_MODE:-current}"
MEMORY_RATIO="${MEMORY_RATIO:-1.5}"
BUILD_CURRENT="${BUILD_CURRENT:-false}"
CURRENT_MEMORY_RATIO="${CURRENT_MEMORY_RATIO:-1.5}"
CURRENT_BUILD_EF="${CURRENT_BUILD_EF:-100}"
CURRENT_OURS_COST_EF="${CURRENT_OURS_COST_EF:-18}"
CURRENT_BUILD_WORKERS="${CURRENT_BUILD_WORKERS:-10}"
KMEANS_COST_MODEL_JSON="${KMEANS_COST_MODEL_JSON:-${ROOT_DIR}/../controller/kmeans/train/result/wiki_latency_cost_095_regrouped_20260715/wiki_latency_cost_095_regrouped_20260715_planner_model.json}"
export KMEANS_COST_MODEL_JSON
OURS_EFS_MODEL_JSON="${OURS_EFS_MODEL_JSON:-${ROOT_DIR}/../controller/kmeans/train/result/yfcc_recall_fit/yfcc_hnsw_recall_model.json}"
OURS_TARGET_RECALL="${OURS_TARGET_RECALL:-0.95}"
PRIVATE_REPLICATION_BUDGET_RATIO="${PRIVATE_REPLICATION_BUDGET_RATIO:-0.5}"
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
EFS_LIST="${EFS_LIST:-20}"
QUERY_DATASET_PATH="${QUERY_DATASET_PATH:-}"
# 3 5 10 12 15 18 20 25 28 30 33 35 38 40 45 50 53 55 60 65 70 75 80 85

# This script only evaluates existing materialized kmeans partitions.
# It intentionally passes --prepare false for every ef_search value.
IFS=' ' read -r -a EFS_VALUES <<< "${EFS_LIST//,/ }"

bool_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

if bool_true "${BUILD_CURRENT}"; then
  build_args=(
    "${PYTHON_BIN}" "${ROOT_DIR}/script/squid/build_current_partitions.py"
    --methods ours
    --memory-ratio "${CURRENT_MEMORY_RATIO}"
    --ef-search "${CURRENT_BUILD_EF}"
    --ours-cost-ef "${CURRENT_OURS_COST_EF}"
    --ours-efs-model-json "${OURS_EFS_MODEL_JSON}"
    --ours-target-recall "${OURS_TARGET_RECALL}"
    --workers "${CURRENT_BUILD_WORKERS}"
    --ours-index-type "${INDEX_TYPE}"
    --private-edge-top-d "${PRIVATE_EDGE_TOP_D}"
  )
  if bool_true "${ENABLE_SPLIT}"; then
    build_args+=(--enable-split)
  else
    build_args+=(--no-enable-split)
  fi
  if [[ -n "${QUERY_DATASET_PATH}" ]]; then
    build_args+=(--query-dataset-path "${QUERY_DATASET_PATH}")
  fi

  echo "[run_ours] building current OURS partitions before SQL query-time sweep"
  printf '[CMD]'
  printf ' %q' "${build_args[@]}"
  printf '\n'
  "${build_args[@]}"
fi

if [[ "$(printf '%s' "${PLAN_MODE}" | tr '[:upper:]' '[:lower:]')" == "versioned" ]]; then
  echo "[run_ours] using existing versioned OURS plan: memory_ratio=${MEMORY_RATIO}"
  eval "$("${PYTHON_BIN}" "${ROOT_DIR}/script/squid/resolve_versioned_env.py" \
    --method ours \
    --memory-ratio "${MEMORY_RATIO}")"
  export SKIP_INDEX_MAINTENANCE=true
fi

METHOD_LOG_DIR="${LOG_DIR}/OURS"
mkdir -p "${METHOD_LOG_DIR}"

for EFS in "${EFS_VALUES[@]}"; do
  EFFECTIVE_RESULT_TAG="${RESULT_TAG}_ef${EFS}"
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${METHOD_LOG_DIR}/${EFFECTIVE_RESULT_TAG}_${TS}.log"

  echo "========================================"
  echo "[run_ours] testing ef_search=${EFS}, result_tag=${EFFECTIVE_RESULT_TAG}, prepare=false"
  echo "[log] ${LOG_FILE}"

 cmd=(
    "${PYTHON_BIN}" test_kmeans_partition.py
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
    --result-tag "${EFFECTIVE_RESULT_TAG}"
  )
  if [[ -n "${QUERY_DATASET_PATH}" ]]; then
    cmd+=(--query-dataset-path "${QUERY_DATASET_PATH}")
  fi

  "${cmd[@]}" > "${LOG_FILE}" 2>&1

done
