#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${ROOT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

# Memory values are storage-amplification factors.  VEDA/EFFVEDA and
# HONEYBEE use the value directly; OURS uses memory - 1 as its private
# replication budget ratio.
MEMORY_VALUES="${MEMORY_VALUES:-2.0}"
METHODS="${METHODS:-veda effveda AnonySys OURS}"

# SEARCH_EF is only a fallback. Set per-method values for fair single-point
# memory experiments, for example:
#   VEDA_SEARCH_EF=40 EFFVEDA_SEARCH_EF=40 HONEYBEE_SEARCH_EF=220 OURS_SEARCH_EF=35
SEARCH_EF="${SEARCH_EF:-50}"
VEDA_SEARCH_EF="${VEDA_SEARCH_EF:-${SEARCH_EF}}"
EFFVEDA_SEARCH_EF="${EFFVEDA_SEARCH_EF:-${SEARCH_EF}}"
HONEYBEE_SEARCH_EF="${HONEYBEE_SEARCH_EF:-${SEARCH_EF}}"
OURS_SEARCH_EF="${OURS_SEARCH_EF:-${SEARCH_EF}}"

QUERY_NUM="${QUERY_NUM:-200}"
ITERATIONS="${ITERATIONS:-1}"
STATISTICS_TYPE="${STATISTICS_TYPE:-sql}"
GENERATOR_TYPE="${GENERATOR_TYPE:-erbac}"
RECORD_RECALL="${RECORD_RECALL:-true}"
WARM_UP="${WARM_UP:-true}"
USE_GROUND_TRUTH_CACHE="${USE_GROUND_TRUTH_CACHE:-true}"
SHOW_PROGRESS="${SHOW_PROGRESS:-true}"
PREPARE_PARTITIONS="${PREPARE_PARTITIONS:-true}"
# Optional filters for partition building. Empty means use METHODS/MEMORY_VALUES.
# Examples:
#   PREPARE_METHODS="OURS" PREPARE_MEMORY_VALUES="1.5"
#   PREPARE_METHODS="veda effveda" PREPARE_MEMORY_VALUES="1.2 1.5"
PREPARE_METHODS="${PREPARE_METHODS:-}"
PREPARE_MEMORY_VALUES="${PREPARE_MEMORY_VALUES:-}"

# ROLE and HQI are fixed-layout baselines. Build each once before the
# memory-specific plans. ROLE must exist before AnonySys prepares its plan.
PREPARE_FIXED_BASELINES="${PREPARE_FIXED_BASELINES:-${PREPARE_PARTITIONS}}"
PREPARE_ROLE_TABLES="${PREPARE_ROLE_TABLES:-true}"
PREPARE_HQI_TABLES="${PREPARE_HQI_TABLES:-true}"
# Abort an individual layout build instead of continuing with incomplete tables.
PLANNER_TIMEOUT="${PLANNER_TIMEOUT:-3h}"
ROLE_INDEX_TYPE="${ROLE_INDEX_TYPE:-hnsw}"
ROLE_MAX_WORKERS="${ROLE_MAX_WORKERS:-4}"
HQI_INDEX_TYPE="${HQI_INDEX_TYPE:-hnsw}"
HQI_MIN_SIZE="${HQI_MIN_SIZE:-10000}"
HQI_WORKERS="${HQI_WORKERS:-8}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/efs_logs/memory_sweep}"
BUILD_LOG_DIR="${BUILD_LOG_DIR:-${LOG_DIR}/_build_logs}"

VEDA_PLAN_EF="${VEDA_PLAN_EF:-100}"
VEDA_INDEX_TYPE="${VEDA_INDEX_TYPE:-vedahnsw}"
VEDA_INDEXING_THRESHOLD="${VEDA_INDEXING_THRESHOLD:-2900}"
VEDA_SEARCH_MODE="${VEDA_SEARCH_MODE:-coordinated}"
VEDA_SQL_TIMING_MODE="${VEDA_SQL_TIMING_MODE:-fair}"
VEDA_HNSW_ITERATIVE_SCAN="${VEDA_HNSW_ITERATIVE_SCAN:-off}"
VEDA_HNSW_MAX_SCAN_TUPLES="${VEDA_HNSW_MAX_SCAN_TUPLES:-}"

HONEYBEE_RECALL="${HONEYBEE_RECALL:-0.99}"
OURS_RECALL="${OURS_RECALL:-0.99}"
HONEYBEE_INDEX_TYPE="${HONEYBEE_INDEX_TYPE:-hnsw}"

OURS_INDEX_TYPE="${OURS_INDEX_TYPE:-squidhnsw}"
OURS_ENABLE_SPLIT="${OURS_ENABLE_SPLIT:-false}"
OURS_PRIVATE_EDGE_TOP_D="${OURS_PRIVATE_EDGE_TOP_D:-32}"

DOCUMENT_LIMIT="${DOCUMENT_LIMIT:-}"
QUERY_DATASET_PATH="${QUERY_DATASET_PATH:-}"

mkdir -p "${LOG_DIR}" "${BUILD_LOG_DIR}"

read -r -a MEMORY_LIST <<< "${MEMORY_VALUES//,/ }"
read -r -a METHOD_LIST <<< "${METHODS//,/ }"
if [[ -n "${PREPARE_METHODS}" ]]; then
  read -r -a PREPARE_METHOD_LIST <<< "${PREPARE_METHODS//,/ }"
else
  PREPARE_METHOD_LIST=("${METHOD_LIST[@]}")
fi
if [[ -n "${PREPARE_MEMORY_VALUES}" ]]; then
  read -r -a PREPARE_MEMORY_LIST <<< "${PREPARE_MEMORY_VALUES//,/ }"
else
  PREPARE_MEMORY_LIST=("${MEMORY_LIST[@]}")
fi

bool_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

_normalize_method() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    honeybee) printf 'anonysys' ;;
    squid) printf 'ours' ;;
    *) printf '%s' "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" ;;
  esac
}

_list_has_method() {
  local target
  target="$(_normalize_method "$1")"
  shift
  local item
  for item in "$@"; do
    if [[ "$(_normalize_method "${item}")" == "${target}" ]]; then
      return 0
    fi
  done
  return 1
}

_list_has_memory() {
  local target="$1"
  shift
  local item
  for item in "$@"; do
    if [[ "$(memory_label "${item}")" == "$(memory_label "${target}")" ]]; then
      return 0
    fi
  done
  return 1
}

should_prepare() {
  local method="$1"
  local memory="$2"
  bool_true "${PREPARE_PARTITIONS}" || return 1
  _list_has_method "${method}" "${PREPARE_METHOD_LIST[@]}" || return 1
  _list_has_memory "${memory}" "${PREPARE_MEMORY_LIST[@]}" || return 1
  return 0
}

memory_label() {
  printf '%s' "$1" | sed 's/-/m/g; s/\./p/g'
}

ours_budget_ratio() {
  "${PYTHON_BIN}" -c 'import sys; print(max(0.0, float(sys.argv[1]) - 1.0))' "$1"
}

append_optional_common_args() {
  local -n cmd_ref="$1"
  if [[ -n "${DOCUMENT_LIMIT}" ]]; then
    cmd_ref+=(--document-limit "${DOCUMENT_LIMIT}")
  fi
  if [[ -n "${QUERY_DATASET_PATH}" ]]; then
    cmd_ref+=(--query-dataset-path "${QUERY_DATASET_PATH}")
  fi
}

run_logged() {
  local label="$1"
  local log_file="$2"
  shift 2
  mkdir -p "$(dirname "${log_file}")"
  echo "========================================"
  echo "[RUN] ${label}"
  echo "[LOG] ${log_file}"
  printf '[CMD]'
  printf ' %q' "$@"
  printf '\n'
  "$@" > "${log_file}" 2>&1
  echo "[DONE] ${label}"
}

run_planner_logged() {
  local label="$1"
  local log_file="$2"
  shift 2
  mkdir -p "$(dirname "${log_file}")"
  echo "========================================"
  echo "[PLAN] ${label}"
  echo "[TIMEOUT] ${PLANNER_TIMEOUT}"
  echo "[LOG] ${log_file}"
  printf "[CMD]"
  printf " %q" "$@"
  printf "\n"
  if timeout --foreground "${PLANNER_TIMEOUT}" "$@" > "${log_file}" 2>&1; then
    echo "[DONE] ${label}"
    return 0
  else
    local status=$?
    if [[ ${status} -eq 124 ]]; then
      echo "[TIMEOUT] ${label} exceeded ${PLANNER_TIMEOUT}; stopping sweep." >&2
    fi
    return "${status}"
  fi
}

prepare_veda_like() {
  local algorithm="$1"
  local memory="$2"
  local mem_label="$3"
  local log_file="${BUILD_LOG_DIR}/${algorithm}_mem${mem_label}_prepare_$(date +%Y%m%d_%H%M%S).txt"
  local result_tag="memory_${algorithm}_mem${mem_label}_prepare"
  local cmd=(
    "${PYTHON_BIN}" test_veda.py
    --prepare true
    --algorithm "${algorithm}"
    --enable-index true
    --index-type "${VEDA_INDEX_TYPE}"
    --statistics-type "${STATISTICS_TYPE}"
    --generator-type "${GENERATOR_TYPE}"
    --record-recall "${RECORD_RECALL}"
    --warm-up "${WARM_UP}"
    --query-num "${QUERY_NUM}"
    --iterations "${ITERATIONS}"
    --indexing-threshold "${VEDA_INDEXING_THRESHOLD}"
    --storage-amplification "${memory}"
    --ef-search "${VEDA_PLAN_EF}"
    --hnsw-iterative-scan "${VEDA_HNSW_ITERATIVE_SCAN}"
    --search-mode "${VEDA_SEARCH_MODE}"
    --sql-timing-mode "${VEDA_SQL_TIMING_MODE}"
    --use-ground-truth-cache "${USE_GROUND_TRUTH_CACHE}"
    --show-progress "${SHOW_PROGRESS}"
    --result-tag "${result_tag}"
  )
  if [[ -n "${VEDA_HNSW_MAX_SCAN_TUPLES}" ]]; then
    cmd+=(--hnsw-max-scan-tuples "${VEDA_HNSW_MAX_SCAN_TUPLES}")
  fi
  append_optional_common_args cmd
  run_planner_logged "${algorithm} prepare memory=${memory} plan_ef=${VEDA_PLAN_EF}" "${log_file}" "${cmd[@]}"
}

measure_veda_like() {
  local algorithm="$1"
  local memory="$2"
  local mem_label="$3"
  local search_ef="${VEDA_SEARCH_EF}"
  if [[ "${algorithm}" == "effveda" ]]; then
    search_ef="${EFFVEDA_SEARCH_EF}"
  fi
  local method_dir="${LOG_DIR}/mem_${mem_label}/${algorithm}"
  local result_tag="memory_${algorithm}_mem${mem_label}_ef${search_ef}"
  local log_file="${method_dir}/${result_tag}_$(date +%Y%m%d_%H%M%S).log"
  local cmd=(
    "${PYTHON_BIN}" test_veda.py
    --prepare false
    --algorithm "${algorithm}"
    --enable-index true
    --index-type "${VEDA_INDEX_TYPE}"
    --statistics-type "${STATISTICS_TYPE}"
    --generator-type "${GENERATOR_TYPE}"
    --record-recall "${RECORD_RECALL}"
    --warm-up "${WARM_UP}"
    --query-num "${QUERY_NUM}"
    --iterations "${ITERATIONS}"
    --indexing-threshold "${VEDA_INDEXING_THRESHOLD}"
    --storage-amplification "${memory}"
    --ef-search "${search_ef}"
    --hnsw-iterative-scan "${VEDA_HNSW_ITERATIVE_SCAN}"
    --search-mode "${VEDA_SEARCH_MODE}"
    --sql-timing-mode "${VEDA_SQL_TIMING_MODE}"
    --use-ground-truth-cache "${USE_GROUND_TRUTH_CACHE}"
    --show-progress "${SHOW_PROGRESS}"
    --result-tag "${result_tag}"
  )
  if [[ -n "${VEDA_HNSW_MAX_SCAN_TUPLES}" ]]; then
    cmd+=(--hnsw-max-scan-tuples "${VEDA_HNSW_MAX_SCAN_TUPLES}")
  fi
  append_optional_common_args cmd
  run_logged "${algorithm} measure memory=${memory} search_ef=${search_ef}" "${log_file}" "${cmd[@]}"
}

prepare_honeybee() {
  local memory="$1"
  local mem_label="$2"
  local log_file="${BUILD_LOG_DIR}/AnonySys_mem${mem_label}_prepare_$(date +%Y%m%d_%H%M%S).txt"
  local cmd=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/controller/dynamic_partition/hnsw/AnonySys_dynamic_partition.py"
    --storage "${memory}"
    --recall "${HONEYBEE_RECALL}"
  )
  run_planner_logged "AnonySys prepare memory=${memory} recall=${HONEYBEE_RECALL}" "${log_file}" "${cmd[@]}"
}

prepare_role() {
  local log_file="${BUILD_LOG_DIR}/ROLE_prepare_$(date +%Y%m%d_%H%M%S).txt"
  local cmd=(
    "${PYTHON_BIN}" initialize_role_partition_tables.py
    --index_type "${ROLE_INDEX_TYPE}"
    --max-workers "${ROLE_MAX_WORKERS}"
  )
  run_planner_logged "ROLE prepare index_type=${ROLE_INDEX_TYPE}" "${log_file}" "${cmd[@]}"
}

prepare_hqi() {
  local build_log_file="${BUILD_LOG_DIR}/HQI_tree_prepare_$(date +%Y%m%d_%H%M%S).txt"
  local persist_log_file="${BUILD_LOG_DIR}/HQI_partitions_prepare_$(date +%Y%m%d_%H%M%S).txt"
  local build_cmd=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/controller/baseline/HQI/build_tree.py"
    --min-size "${HQI_MIN_SIZE}"
  )
  local persist_cmd=(
    "${PYTHON_BIN}" "${PROJECT_ROOT}/controller/baseline/HQI/persist_tree.py"
    --index-type "${HQI_INDEX_TYPE}"
    --workers "${HQI_WORKERS}"
  )
  run_planner_logged "HQI build tree min_size=${HQI_MIN_SIZE}" "${build_log_file}" "${build_cmd[@]}"
  run_planner_logged "HQI persist partitions index_type=${HQI_INDEX_TYPE}" "${persist_log_file}" "${persist_cmd[@]}"
}

measure_honeybee() {
  local memory="$1"
  local mem_label="$2"
  local method_dir="${LOG_DIR}/mem_${mem_label}/AnonySys"
  local log_file="${method_dir}/AnonySys_mem${mem_label}_efs${HONEYBEE_SEARCH_EF}_$(date +%Y%m%d_%H%M%S).log"
  local cmd=(
    "${PYTHON_BIN}" test_all.py
    --algorithm AnonySys
    --efs "${HONEYBEE_SEARCH_EF}"
    --enable-index true
    --index-type "${HONEYBEE_INDEX_TYPE}"
    --statistics-type "${STATISTICS_TYPE}"
    --generator-type "${GENERATOR_TYPE}_memory_AnonySys_mem${mem_label}"
    --iterations "${ITERATIONS}"
    --query-num "${QUERY_NUM}"
    --record-recall "${RECORD_RECALL}"
    --warm-up "${WARM_UP}"
  )
  run_logged "AnonySys measure memory=${memory} search_ef=${HONEYBEE_SEARCH_EF}" "${log_file}" "${cmd[@]}"
}

prepare_ours() {
  local memory="$1"
  local mem_label="$2"
  local ratio
  ratio="$(ours_budget_ratio "${memory}")"
  local log_file="${BUILD_LOG_DIR}/OURS_mem${mem_label}_prepare_$(date +%Y%m%d_%H%M%S).txt"
  local result_tag="memory_OURS_mem${mem_label}_prepare"
  local cmd=(
    env "KMEANS_TARGET_RECALL=${OURS_RECALL}"
    "${PYTHON_BIN}" test_kmeans_partition.py
    --prepare true
    --private-replication-budget-ratio "${ratio}"
    --private-edge-top-d "${OURS_PRIVATE_EDGE_TOP_D}"
    --enable-split "${OURS_ENABLE_SPLIT}"
    --enable-index true
    --index-type "${OURS_INDEX_TYPE}"
    --statistics-type "${STATISTICS_TYPE}"
    --query-num "${QUERY_NUM}"
    --iterations "${ITERATIONS}"
    --record-recall "${RECORD_RECALL}"
    --warm-up "${WARM_UP}"
    --show-progress "${SHOW_PROGRESS}"
    --use-ground-truth-cache "${USE_GROUND_TRUTH_CACHE}"
    --result-tag "${result_tag}"
  )
  append_optional_common_args cmd
  run_planner_logged "OURS prepare memory=${memory} ratio=${ratio} cost_ef=recall-model target_recall=${OURS_RECALL}" "${log_file}" "${cmd[@]}"
}

measure_ours() {
  local memory="$1"
  local mem_label="$2"
  local ratio
  ratio="$(ours_budget_ratio "${memory}")"
  local method_dir="${LOG_DIR}/mem_${mem_label}/OURS"
  local result_tag="memory_OURS_mem${mem_label}_ef${OURS_SEARCH_EF}"
  local log_file="${method_dir}/OURS_mem${mem_label}_ef${OURS_SEARCH_EF}_$(date +%Y%m%d_%H%M%S).log"
  local cmd=(
    env "KMEANS_TARGET_RECALL=${OURS_RECALL}"
    "${PYTHON_BIN}" test_kmeans_partition.py
    --prepare false
    --private-replication-budget-ratio "${ratio}"
    --private-edge-top-d "${OURS_PRIVATE_EDGE_TOP_D}"
    --enable-split "${OURS_ENABLE_SPLIT}"
    --enable-index true
    --index-type "${OURS_INDEX_TYPE}"
    --statistics-type "${STATISTICS_TYPE}"
    --query-num "${QUERY_NUM}"
    --iterations "${ITERATIONS}"
    --record-recall "${RECORD_RECALL}"
    --warm-up "${WARM_UP}"
    --ef-search "${OURS_SEARCH_EF}"
    --show-progress "${SHOW_PROGRESS}"
    --use-ground-truth-cache "${USE_GROUND_TRUTH_CACHE}"
    --result-tag "${result_tag}"
  )
  append_optional_common_args cmd
  run_logged "OURS measure memory=${memory} ratio=${ratio} search_ef=${OURS_SEARCH_EF}" "${log_file}" "${cmd[@]}"
}

# Fixed baselines are deliberately materialized before the memory sweep. In
# particular, AnonySys calls into the ROLE tables during its preparation.
if bool_true "${PREPARE_FIXED_BASELINES}"; then
  if bool_true "${PREPARE_ROLE_TABLES}"; then
    prepare_role
  else
    echo "[SKIP] ROLE fixed-table preparation"
  fi
  if bool_true "${PREPARE_HQI_TABLES}"; then
    prepare_hqi
  else
    echo "[SKIP] HQI fixed-table preparation"
  fi
else
  echo "[SKIP] fixed baseline preparation"
fi

for memory in "${MEMORY_LIST[@]}"; do
  mem_label="$(memory_label "${memory}")"
  echo "########################################"
  echo "[MEMORY] ${memory} label=mem_${mem_label}"
  for method in "${METHOD_LIST[@]}"; do
    case "${method}" in
      veda|effveda)
        if should_prepare "${method}" "${memory}"; then
          prepare_veda_like "${method}" "${memory}" "${mem_label}"
        else
          echo "[SKIP] ${method} prepare memory=${memory}"
        fi
        measure_veda_like "${method}" "${memory}" "${mem_label}"
        ;;
      AnonySys|HONEYBEE|honeybee)
        if should_prepare "${method}" "${memory}"; then
          prepare_honeybee "${memory}" "${mem_label}"
        else
          echo "[SKIP] AnonySys prepare memory=${memory}"
        fi
        measure_honeybee "${memory}" "${mem_label}"
        ;;
      OURS|ours|SQUID|squid)
        if should_prepare "${method}" "${memory}"; then
          prepare_ours "${memory}" "${mem_label}"
        else
          echo "[SKIP] OURS prepare memory=${memory}"
        fi
        measure_ours "${memory}" "${mem_label}"
        ;;
      *)
        echo "Unknown method: ${method}" >&2
        exit 2
        ;;
    esac
  done
done

echo "Memory sweep finished. Formal logs: ${LOG_DIR}"
echo "Build/pre-cache logs: ${BUILD_LOG_DIR}"
