#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"

# ours honeybee veda effveda
METHODS="${METHODS:-ours effveda honeybee veda}"
MEMORY_VALUES="${MEMORY_VALUES:-1.0 2.0 3.0 4.0 5.0 6.0}"
#1.0 2.0 3.0 4.0 5.0 6.0
BUILD="${BUILD:-true}"
QPS="${QPS:-false}"

QUERY_COUNT="${QUERY_COUNT:-10}"
QUERY_REPETITIONS="${QUERY_REPETITIONS:-5}"
CONCURRENCY="${CONCURRENCY:-64}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}"
EF_VALUES="${EF_VALUES:-100}"
INDEX_MODE="${INDEX_MODE:-hnsw}"
OURS_INDEX_TYPE="${OURS_INDEX_TYPE:-hnsw}"
VEDA_INDEX_TYPE="${VEDA_INDEX_TYPE:-hnsw}"
JIT="${JIT:-on}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-2}"
AUTH_FILTER="${AUTH_FILTER:-rls}"

VERSION_PREFIX="${VERSION_PREFIX:-sweep}"
PLANNER_TIMEOUT="${PLANNER_TIMEOUT:-3h}"

bool_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

read -r -a METHOD_LIST <<< "${METHODS//,/ }"
read -r -a MEMORY_LIST <<< "${MEMORY_VALUES//,/ }"
read -r -a EF_LIST <<< "${EF_VALUES//,/ }"

for memory in "${MEMORY_LIST[@]}"; do
  for method in "${METHOD_LIST[@]}"; do
    version="${VERSION_PREFIX}_${method}_$(printf '%s' "${memory}" | sed 's/\./p/g')"
    if bool_true "${BUILD}"; then
      timeout --foreground "${PLANNER_TIMEOUT}" \
        "${PYTHON_BIN}" "${SCRIPT_DIR}/build_versioned_plan.py" \
          --method "${method}" \
          --memory-ratio "${memory}" \
          --version "${version}" \
          --ours-index-type "${OURS_INDEX_TYPE}" \
          --veda-index-type "${VEDA_INDEX_TYPE}"
    fi
    if bool_true "${QPS}"; then
      for ef in "${EF_LIST[@]}"; do
        "${PYTHON_BIN}" "${PROJECT_ROOT}/basic_benchmark/direct_pg_qps.py" \
          --methods "${method}" \
          --memory-ratio "${memory}" \
          --query-count "${QUERY_COUNT}" \
          --query-repetitions "${QUERY_REPETITIONS}" \
          --concurrency "${CONCURRENCY}" \
          --warmup-rounds "${WARMUP_ROUNDS}" \
          --ef-search "${ef}" \
          --index-mode "${INDEX_MODE}" \
          --jit "${JIT}" \
          --parallel-workers "${PARALLEL_WORKERS}" \
          --auth-filter "${AUTH_FILTER}"
      done
    fi
  done
done
