#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${ROOT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-/home/chenyang/.conda/envs/multitenant/bin/python}"

# Space-separated algorithms supported by test_all.py:
#   RLS ROLE USER AnonySys QDTree
# Override example: ALGORITHMS="AnonySys RLS" ./script/run_baseline.sh
ALGORITHMS="${ALGORITHMS:-QDTree}"
read -r -a ALGORITHM_LIST <<< "${ALGORITHMS}"

# Space-separated or comma-separated ef_search values.
# Override example: EFS_VALUES="40 60 80 100" ./script/run_baseline.sh
EFS_VALUES="${EFS_VALUES:-15 20 25 28 30 35 40 45 50 55}"
EFS_VALUES="${EFS_VALUES//,/ }"
read -r -a EFS_LIST <<< "${EFS_VALUES}"

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/efs_logs}"
ENABLE_INDEX="${ENABLE_INDEX:-true}"
INDEX_TYPE="${INDEX_TYPE:-hnsw}"
STATISTICS_TYPE="${STATISTICS_TYPE:-sql}"
ITERATIONS="${ITERATIONS:-1}"
RECORD_RECALL="${RECORD_RECALL:-true}"
WARM_UP="${WARM_UP:-true}"
GENERATOR_TYPE="${GENERATOR_TYPE:-erbac}"
QDTREE_QUERY_NUM="${QDTREE_QUERY_NUM:-200}"

# test_all.py has no --prepare flag.  This script intentionally does not build
# partitions; it only changes ef_search and reuses already materialized baseline
# partition tables.  Prepare/build the partition tables once before running this
# script when using ROLE/USER/AnonySys/QDTree.

mkdir -p "${LOG_DIR}"

for ALGORITHM in "${ALGORITHM_LIST[@]}"; do
  METHOD_LOG_DIR="${LOG_DIR}/${ALGORITHM}"
  mkdir -p "${METHOD_LOG_DIR}"
  for EFS in "${EFS_LIST[@]}"; do
    TS="$(date +%Y%m%d_%H%M%S)"
    LOG_FILE="${METHOD_LOG_DIR}/${ALGORITHM}_efs${EFS}_${TS}.log"

    echo "========================================"
    echo "[RUN] algorithm=${ALGORITHM} ef_search=${EFS} prepare=false"
    echo "[LOG] ${LOG_FILE}"

    EXTRA_ARGS=()
    if [[ "${ALGORITHM}" == "QDTree" ]]; then
      EXTRA_ARGS+=(--query-num "${QDTREE_QUERY_NUM}")
    fi

    python test_all.py \
      --algorithm "${ALGORITHM}" \
      --efs "${EFS}" \
      --enable-index "${ENABLE_INDEX}" \
      --index-type "${INDEX_TYPE}" \
      --statistics-type "${STATISTICS_TYPE}" \
      --generator-type "${GENERATOR_TYPE}" \
      --iterations "${ITERATIONS}" \
      --record-recall "${RECORD_RECALL}" \
      --warm-up "${WARM_UP}" \
      "${EXTRA_ARGS[@]}" \
      > "${LOG_FILE}" 2>&1

    echo "[DONE] algorithm=${ALGORITHM} ef_search=${EFS}"
  done
done

echo "All baseline ef_search runs finished. Logs: ${LOG_DIR}"
