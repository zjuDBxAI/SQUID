#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/Multitenanthakes}"
cd "${PROJECT_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"
if [[ ! -x "${PYTHON_BIN}" ]]; then
  PYTHON_BIN="python3"
fi

export DBNAME="${DBNAME:-rbacdatabase_wikipedia}"
export DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/dataset}"

LOAD_NUMBER="${LOAD_NUMBER:-100000}"
START_ROW="${START_ROW:-0}"
LOAD_THREADS="${LOAD_THREADS:-4}"
PERMISSION_MODEL="${PERMISSION_MODEL:-treebase}"
QUERY_COUNT="${QUERY_COUNT:-1000}"
QUERY_TOPK="${QUERY_TOPK:-10}"
QUERY_THREADS="${QUERY_THREADS:-4}"
GROUND_TRUTH_WORKERS="${GROUND_TRUTH_WORKERS:-8}"
SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-false}"
SKIP_LOAD="${SKIP_LOAD:-false}"
SKIP_PERMISSION="${SKIP_PERMISSION:-false}"
SKIP_QUERY="${SKIP_QUERY:-false}"
SKIP_GROUND_TRUTH="${SKIP_GROUND_TRUTH:-false}"

DATASET_DIR="${DATASET_PATH}/wikipedia-22-12-simple-embeddings"
CSV_FILE="${DATASET_DIR}/wiki.csv"
DATASET_URL="https://huggingface.co/datasets/timescale/wikipedia-22-12-simple-embeddings/resolve/main/wiki.csv"

bool_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

run_step() {
  local label="$1"
  shift
  echo "[wikipedia-simple] ${label}"
  "$@"
}

if ! bool_true "${SKIP_DOWNLOAD}"; then
  mkdir -p "${DATASET_DIR}"
  if [[ -s "${CSV_FILE}" ]]; then
    echo "[wikipedia-simple] Reusing ${CSV_FILE}"
  else
    run_step "Download Timescale Wikipedia CSV" \
      wget -c --progress=dot:giga -O "${CSV_FILE}" "${DATASET_URL}"
  fi
fi

if ! bool_true "${SKIP_LOAD}"; then
  run_step "Initialize ${DBNAME} and load Wikipedia vectors" \
    "${PYTHON_BIN}" basic_benchmark/common_prepare_pipeline.py \
      --dataset wikipedia-22-12-simple \
      --load-number "${LOAD_NUMBER}" \
      --start-row "${START_ROW}" \
      --num-threads "${LOAD_THREADS}"
fi

if ! bool_true "${SKIP_PERMISSION}"; then
  case "$(printf '%s' "${PERMISSION_MODEL}" | tr '[:upper:]' '[:lower:]')" in
    treebase|tree|tree-based)
      run_step "Generate tree-based RBAC permissions" \
        "${PYTHON_BIN}" services/rbac_generator/store_tree_based_rbac_generate_data.py
      ;;
    erbac)
      run_step "Generate ERBAC permissions" \
        "${PYTHON_BIN}" services/rbac_generator/store_erbac_generate_data.py
      ;;
    saas)
      run_step "Generate SaaS tenant RBAC permissions" \
        "${PYTHON_BIN}" services/rbac_generator/store_saas_workload.py
      ;;
    *)
      echo "Unsupported PERMISSION_MODEL=${PERMISSION_MODEL}. Use treebase, erbac, or saas." >&2
      exit 2
      ;;
  esac
fi

if ! bool_true "${SKIP_QUERY}"; then
  run_step "Generate benchmark queries" \
    "${PYTHON_BIN}" basic_benchmark/generate_queries.py \
      --num_queries "${QUERY_COUNT}" \
      --topk "${QUERY_TOPK}" \
      --num_threads "${QUERY_THREADS}"
fi

if ! bool_true "${SKIP_GROUND_TRUTH}"; then
  run_step "Compute ground truth" \
    "${PYTHON_BIN}" basic_benchmark/compute_ground_truth.py \
      --workers "${GROUND_TRUTH_WORKERS}"
fi

echo "[wikipedia-simple] Done. DBNAME=${DBNAME}, dataset=${CSV_FILE}"
