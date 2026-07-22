#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/chenyang/.conda/envs/multitenant/bin/python}"

# Space-separated algorithms supported by test_all.py:
#   RLS ROLE USER AnonySys QDTree OURS SQUID VEDA EFFVEDA
# Override example: ALGORITHMS="OURS EFFVEDA RLS" ./script/run_baseline.sh
ALGORITHMS="${ALGORITHMS:-QDTree}"
read -r -a ALGORITHM_LIST <<< "${ALGORITHMS}"

# Space-separated or comma-separated ef_search values.
# Override example: EFS_VALUES="40 60 80 100" ./script/run_baseline.sh
EFS_VALUES="${EFS_VALUES:-5 8 10 12 15 18 20 23 25 28 30 33 35 38}"
EFS_VALUES="${EFS_VALUES//,/ }"
read -r -a EFS_LIST <<< "${EFS_VALUES}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/efs_logs}"
ENABLE_INDEX="${ENABLE_INDEX:-true}"
INDEX_TYPE="${INDEX_TYPE:-hnsw}"
PLAN_MODE="${PLAN_MODE:-current}"
MEMORY_RATIO="${MEMORY_RATIO:-1.5}"
ROLE_INDEX_TYPE="${ROLE_INDEX_TYPE:-hnsw}"
OURS_INDEX_TYPE="${OURS_INDEX_TYPE:-hnsw}"
VEDA_INDEX_TYPE="${VEDA_INDEX_TYPE:-hnsw}"
BUILD_CURRENT="${BUILD_CURRENT:-false}"
CURRENT_MEMORY_RATIO="${CURRENT_MEMORY_RATIO:-1.5}"
CURRENT_BUILD_EF="${CURRENT_BUILD_EF:-${EFS_VALUES%% *}}"
CURRENT_BUILD_WORKERS="${CURRENT_BUILD_WORKERS:-10}"
CURRENT_OURS_COST_EF="${CURRENT_OURS_COST_EF:-18}"
KMEANS_COST_MODEL_JSON="${KMEANS_COST_MODEL_JSON:-${ROOT_DIR}/../controller/kmeans/train/result/wiki_latency_cost_095_regrouped_20260715/wiki_latency_cost_095_regrouped_20260715_planner_model.json}"
export KMEANS_COST_MODEL_JSON
OURS_EFS_MODEL_JSON="${OURS_EFS_MODEL_JSON:-${ROOT_DIR}/../controller/kmeans/train/result/yfcc_recall_fit/yfcc_hnsw_recall_model.json}"
OURS_TARGET_RECALL="${OURS_TARGET_RECALL:-0.95}"
VEDA_INDEXING_THRESHOLD="${VEDA_INDEXING_THRESHOLD:-2900}"
HONEYBEE_RECALL="${HONEYBEE_RECALL:-0.99}"
HQI_MIN_SIZE="${HQI_MIN_SIZE:-512}"
HQI_INDEX_TYPE="${HQI_INDEX_TYPE:-hnsw}"
STATISTICS_TYPE="${STATISTICS_TYPE:-sql}"
ITERATIONS="${ITERATIONS:-1}"
RECORD_RECALL="${RECORD_RECALL:-true}"
WARM_UP="${WARM_UP:-true}"
GENERATOR_TYPE="${GENERATOR_TYPE:-erbac}"
QUERY_NUM="${QUERY_NUM:-200}"
USE_GROUND_TRUTH_CACHE="${USE_GROUND_TRUTH_CACHE:-true}"
VEDA_SEARCH_MODE="${VEDA_SEARCH_MODE:-coordinated}"
VEDA_SQL_TIMING_MODE="${VEDA_SQL_TIMING_MODE:-fair}"
HNSW_ITERATIVE_SCAN="${HNSW_ITERATIVE_SCAN:-off}"
HNSW_MAX_SCAN_TUPLES="${HNSW_MAX_SCAN_TUPLES:-}"

# By default this script only changes ef_search and reuses already materialized
# current partition tables. Set BUILD_CURRENT=true to rebuild the requested
# current layouts before the query-time sweep.

bool_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

current_build_method() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    rls) printf 'rls' ;;
    role) printf 'role' ;;
    anonysys|honeybee|dynamic_partition) printf 'honeybee' ;;
    qdtree|qd_tree|hqi) printf 'hqi' ;;
    ours|squid) printf 'ours' ;;
    veda) printf 'veda' ;;
    effveda) printf 'effveda' ;;
    *) return 1 ;;
  esac
}

if bool_true "${BUILD_CURRENT}"; then
  BUILD_METHODS=()
  for ALGORITHM in "${ALGORITHM_LIST[@]}"; do
    if method="$(current_build_method "${ALGORITHM}")"; then
      already_present=false
      for existing in "${BUILD_METHODS[@]}"; do
        if [[ "${existing}" == "${method}" ]]; then
          already_present=true
          break
        fi
      done
      if ! "${already_present}"; then
        BUILD_METHODS+=("${method}")
      fi
    else
      echo "[WARN] BUILD_CURRENT does not know how to build ${ALGORITHM}; skipping build for it" >&2
    fi
  done

  if [[ "${#BUILD_METHODS[@]}" -gt 0 ]]; then
    build_args=(
      "${PYTHON_BIN}" "${ROOT_DIR}/script/squid/build_current_partitions.py"
      --methods "${BUILD_METHODS[@]}"
      --memory-ratio "${CURRENT_MEMORY_RATIO}"
      --ef-search "${CURRENT_BUILD_EF}"
      --ours-cost-ef "${CURRENT_OURS_COST_EF}"
      --ours-efs-model-json "${OURS_EFS_MODEL_JSON}"
      --ours-target-recall "${OURS_TARGET_RECALL}"
      --workers "${CURRENT_BUILD_WORKERS}"
      --role-index-type "${ROLE_INDEX_TYPE}"
      --ours-index-type "${OURS_INDEX_TYPE}"
      --veda-index-type "${VEDA_INDEX_TYPE}"
      --veda-indexing-threshold "${VEDA_INDEXING_THRESHOLD}"
      --honeybee-recall "${HONEYBEE_RECALL}"
      --hqi-min-size "${HQI_MIN_SIZE}"
      --hqi-index-type "${HQI_INDEX_TYPE}"
      --no-enable-split
    )
    echo "[run_baseline] building current partitions before SQL query-time sweep: ${BUILD_METHODS[*]}"
    printf '[CMD]'
    printf ' %q' "${build_args[@]}"
    printf '\n'
    "${build_args[@]}"
  fi
fi

mkdir -p "${LOG_DIR}"

for ALGORITHM in "${ALGORITHM_LIST[@]}"; do
  METHOD_LOG_DIR="${LOG_DIR}/${ALGORITHM}"
  mkdir -p "${METHOD_LOG_DIR}"
  if [[ "$(printf '%s' "${PLAN_MODE}" | tr '[:upper:]' '[:lower:]')" == "versioned" ]]; then
    case "$(printf '%s' "${ALGORITHM}" | tr '[:upper:]' '[:lower:]')" in
      honeybee|anonysys|dynamic_partition|ours|squid|veda|effveda)
        echo "[run_baseline] using existing versioned ${ALGORITHM} plan: memory_ratio=${MEMORY_RATIO}"
        eval "$("${PYTHON_BIN}" "${ROOT_DIR}/script/squid/resolve_versioned_env.py" \
          --method "${ALGORITHM}" \
          --memory-ratio "${MEMORY_RATIO}")"
        export SKIP_INDEX_MAINTENANCE=true
        ;;
      *)
        ;;
    esac
  fi
  for EFS in "${EFS_LIST[@]}"; do
    TS="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="${METHOD_LOG_DIR}/${ALGORITHM}_efs${EFS}_${TS}.log"

    echo "========================================"
    echo "[RUN] algorithm=${ALGORITHM} ef_search=${EFS} prepare=false"
    echo "[LOG] ${LOG_FILE}"


    cmd=(
      "${PYTHON_BIN}" test_all.py
      --algorithm "${ALGORITHM}" \
      --efs "${EFS}" \
      --enable-index "${ENABLE_INDEX}" \
      --index-type "${INDEX_TYPE}" \
      --statistics-type "${STATISTICS_TYPE}" \
      --generator-type "${GENERATOR_TYPE}" \
      --iterations "${ITERATIONS}" \
      --query-num "${QUERY_NUM}" \
      --record-recall "${RECORD_RECALL}" \
      --warm-up "${WARM_UP}" \
      --use-ground-truth-cache "${USE_GROUND_TRUTH_CACHE}" \
      --veda-search-mode "${VEDA_SEARCH_MODE}" \
      --veda-sql-timing-mode "${VEDA_SQL_TIMING_MODE}" \
      --hnsw-iterative-scan "${HNSW_ITERATIVE_SCAN}"
    )
    if [[ -n "${HNSW_MAX_SCAN_TUPLES}" ]]; then
      cmd+=(--hnsw-max-scan-tuples "${HNSW_MAX_SCAN_TUPLES}")
    fi

    "${cmd[@]}" > "${LOG_FILE}" 2>&1

    echo "[DONE] algorithm=${ALGORITHM} ef_search=${EFS}"
  done
done

echo "All baseline ef_search runs finished. Logs: ${LOG_DIR}"
