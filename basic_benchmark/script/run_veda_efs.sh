#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/chenyang/.conda/envs/multitenant/bin/python}"

# Algorithm can be "veda" or "effveda".
ALGORITHM="${ALGORITHM:-veda}"

# Space-separated ef_search values. Override with: EFS_VALUES="40 80 120" ./script/run_veda_efs.sh
EFS_VALUES="${EFS_VALUES:-100}"
read -r -a EFS_LIST <<< "${EFS_VALUES}"

PREPARE="${PREPARE:-true}"
# Veda planning uses ef_search in the cost model, so rebuilding per ef is the safer default.
REBUILD_EACH_EF="${REBUILD_EACH_EF:-false}"

QUERY_NUM="${QUERY_NUM:-200}"
ITERATIONS="${ITERATIONS:-1}"
INDEX_TYPE="${INDEX_TYPE:-hnsw}"
STATISTICS_TYPE="${STATISTICS_TYPE:-sql}"
RECORD_RECALL="${RECORD_RECALL:-true}"
WARM_UP="${WARM_UP:-true}"
ENABLE_INDEX="${ENABLE_INDEX:-true}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"
USE_GROUND_TRUTH_CACHE="${USE_GROUND_TRUTH_CACHE:-true}"

INDEXING_THRESHOLD="${INDEXING_THRESHOLD:-2900}"
STORAGE_AMPLIFICATION="${STORAGE_AMPLIFICATION:-3.0}"
SEARCH_MODE="${SEARCH_MODE:-coordinated}"
SQL_TIMING_MODE="${SQL_TIMING_MODE:-legacy}"
HNSW_ITERATIVE_SCAN="${HNSW_ITERATIVE_SCAN:-off}"
HNSW_MAX_SCAN_TUPLES="${HNSW_MAX_SCAN_TUPLES:-}"
GENERATOR_TYPE="${GENERATOR_TYPE:-tree-based}"
RESULT_TAG_PREFIX="${RESULT_TAG_PREFIX:-veda_efs}"
LOG_DIR="${LOG_DIR:-${ROOT_DIR}/efs_logs}"
USE_GROUND_TRUTH_CACHE="${USE_GROUND_TRUTH_CACHE:-true}"

# Optional. Leave empty to use the full dataset/default query dataset.
DOCUMENT_LIMIT="${DOCUMENT_LIMIT:-}"
QUERY_DATASET_PATH="${QUERY_DATASET_PATH:-}"

mkdir -p "${LOG_DIR}"

_bool_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

run_case() {
  local efs="$1"
  local prepare="$2"
  local ts
  ts="$(date +%Y%m%d_%H%M%S)"
  local result_tag="${RESULT_TAG_PREFIX}_${ALGORITHM}_efs${efs}"
  local log_file="${LOG_DIR}/${result_tag}_${ts}.log"

  local cmd=(
    "${PYTHON_BIN}" test_veda.py
    --prepare "${prepare}"
    --algorithm "${ALGORITHM}"
    --enable-index "${ENABLE_INDEX}"
    --index-type "${INDEX_TYPE}"
    --statistics-type "${STATISTICS_TYPE}"
    --generator-type "${GENERATOR_TYPE}"
    --record-recall "${RECORD_RECALL}"
    --warm-up "${WARM_UP}"
    --query-num "${QUERY_NUM}"
    --iterations "${ITERATIONS}"
    --indexing-threshold "${INDEXING_THRESHOLD}"
    --storage-amplification "${STORAGE_AMPLIFICATION}"
    --ef-search "${efs}"
    --hnsw-iterative-scan "${HNSW_ITERATIVE_SCAN}"
    --search-mode "${SEARCH_MODE}"
    --sql-timing-mode "${SQL_TIMING_MODE}"
    --use-ground-truth-cache "${USE_GROUND_TRUTH_CACHE}"
    --show-progress "${SHOW_PROGRESS}"
    --result-tag "${result_tag}"
    --use-ground-truth-cache "${USE_GROUND_TRUTH_CACHE}"
  )

  if [[ -n "${HNSW_MAX_SCAN_TUPLES}" ]]; then
    cmd+=(--hnsw-max-scan-tuples "${HNSW_MAX_SCAN_TUPLES}")
  fi
  if [[ -n "${DOCUMENT_LIMIT}" ]]; then
    cmd+=(--document-limit "${DOCUMENT_LIMIT}")
  fi
  if [[ -n "${QUERY_DATASET_PATH}" ]]; then
    cmd+=(--query-dataset-path "${QUERY_DATASET_PATH}")
  fi

  echo "========================================"
  echo "[RUN] algorithm=${ALGORITHM} ef-search=${efs} prepare=${prepare}"
  echo "[LOG] ${log_file}"
  printf '[CMD]'
  printf ' %q' "${cmd[@]}"
  printf '
'

  "${cmd[@]}" > "${log_file}" 2>&1

  echo "[DONE] algorithm=${ALGORITHM} ef-search=${efs}"
}

first_run=1
for efs in "${EFS_LIST[@]}"; do
  prepare_for_case="false"
  if _bool_true "${PREPARE}"; then
    if _bool_true "${REBUILD_EACH_EF}"; then
      prepare_for_case="true"
    elif [[ "${first_run}" -eq 1 ]]; then
      prepare_for_case="true"
    fi
  fi
  run_case "${efs}" "${prepare_for_case}"
  first_run=0
done

echo "All Veda/EffVeda ef_search runs finished. Logs: ${LOG_DIR}"
