#!/usr/bin/env bash
set -euo pipefail

ALGORITHM="RLS"
EFS_LIST=(800  1000  1200 1500 1800 2000 2200)
LOG_DIR="./efs_logs"

mkdir -p "$LOG_DIR"

for EFS in "${EFS_LIST[@]}"; do
  TS="$(date +%Y%m%d_%H%M%S)"
  LOG_FILE="${LOG_DIR}/${ALGORITHM}_efs${EFS}_${TS}.log"

  echo "========================================"
  echo "[RUN] python test_all.py --algorithm ${ALGORITHM} --efs ${EFS}"
  echo "[LOG] ${LOG_FILE}"

  python test_all.py \
    --algorithm "${ALGORITHM}" \
    --efs "${EFS}" \
    > "${LOG_FILE}" 2>&1

  echo "[DONE] efs=${EFS}"
done

echo "All runs finished."