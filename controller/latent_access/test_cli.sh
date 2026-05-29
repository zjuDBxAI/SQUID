#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="/data/Multitenanthakes"
PYTHON_BIN="${PYTHON_BIN:-/home/chenyang/.conda/envs/multitenant/bin/python}"
CLI_PATH="${ROOT_DIR}/controller/latent_access/cli.py"

ACTION="${1:-help}"
shift || true

common_build_args=(
  --training-limit 2000
  --atom-count 16
  --semantic-cell-count 24
  --residual-quantile 0.95
  --access-weight 1.0
  --semantic-weight 0.0
  --semantic-knn 8
  --semantic-knn-weight 0.15
  --max-atoms-per-semantic-cell 3
  --min-partition-documents 8
  --sparsity 2
  --max-iterations 10
  --z-inner-iterations 3
  --momentum-weight 0.05
  --min-atom-support 4.0
  --revive-every 3
  --revive-residual-quantile 0.92
)

common_benchmark_args=(
  --enable-index true
  --index-type hnsw
  --statistics-type sql
  --iterations 1
)

usage() {
  cat <<USAGE
Usage:
  bash controller/latent_access/test_cli.sh init
  bash controller/latent_access/test_cli.sh build
  bash controller/latent_access/test_cli.sh index
  bash controller/latent_access/test_cli.sh summary
  bash controller/latent_access/test_cli.sh smoke
  bash controller/latent_access/test_cli.sh benchmark
  bash controller/latent_access/test_cli.sh full

Optional env vars:
  PYTHON_BIN=/path/to/python
USAGE
}

cd "${ROOT_DIR}"

case "${ACTION}" in
  init)
    "${PYTHON_BIN}" "${CLI_PATH}" init "$@"
    ;;
  build)
    "${PYTHON_BIN}" "${CLI_PATH}" build "${common_build_args[@]}" "$@"
    ;;
  index)
    "${PYTHON_BIN}" "${CLI_PATH}" index --index-type hnsw "$@"
    ;;
  summary)
    "${PYTHON_BIN}" "${CLI_PATH}" summary "$@"
    ;;
  smoke)
    "${PYTHON_BIN}" "${CLI_PATH}" benchmark \
      "${common_build_args[@]}" \
      "${common_benchmark_args[@]}" \
      --query-num 5 \
      --route-limit 16 \
      --partition-fetch-multiplier 6 \
      --use-ground-truth-cache false \
      "$@"
    ;;
  benchmark)
    "${PYTHON_BIN}" "${CLI_PATH}" benchmark \
      "${common_build_args[@]}" \
      "${common_benchmark_args[@]}" \
      --query-num 1000 \
      --route-limit 16 \
      --partition-fetch-multiplier 6 \
      --use-ground-truth-cache false \
      "$@"
    ;;
  full)
    "${PYTHON_BIN}" "${CLI_PATH}" init
    "${PYTHON_BIN}" "${CLI_PATH}" build "${common_build_args[@]}"
    "${PYTHON_BIN}" "${CLI_PATH}" index --index-type hnsw
    "${PYTHON_BIN}" "${CLI_PATH}" summary
    "${PYTHON_BIN}" "${CLI_PATH}" benchmark \
      "${common_build_args[@]}" \
      "${common_benchmark_args[@]}" \
      --query-num 1000 \
      --route-limit 16 \
      --partition-fetch-multiplier 6 \
      --use-ground-truth-cache false \
      "$@"
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    echo "Unknown action: ${ACTION}" >&2
    usage >&2
    exit 1
    ;;
esac
