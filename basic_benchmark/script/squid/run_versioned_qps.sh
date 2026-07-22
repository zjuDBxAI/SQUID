#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/venv/bin/python}"

# ours honeybee veda effveda
METHODS="${METHODS:-ours effveda honeybee veda}"
MEMORY_VALUES="${MEMORY_VALUES:-1.5}"
#1.0 2.0 3.0 4.0 5.0 6.0
BUILD="${BUILD:-true}"
QPS="${QPS:-false}"
PLAN_MODE="${PLAN_MODE:-versioned}"

QUERY_COUNT="${QUERY_COUNT:-10}"
QUERY_SOURCE="${QUERY_SOURCE:-file}"
TOPK="${TOPK:-10}"
QUERY_FILE="${QUERY_FILE:-}"
GROUND_TRUTH_FILE="${GROUND_TRUTH_FILE:-}"
QUERY_REPETITIONS="${QUERY_REPETITIONS:-5}"
DURATION_SECONDS="${DURATION_SECONDS:-0}"
CONCURRENCY="${CONCURRENCY:-64}"
WARMUP_ROUNDS="${WARMUP_ROUNDS:-1}"
EF_VALUES="${EF_VALUES:-100}"
QPS_TRIALS="${QPS_TRIALS:-1}"
INDEX_MODE="${INDEX_MODE:-hnsw}"
OURS_INDEX_TYPE="${OURS_INDEX_TYPE:-hnsw}"
VEDA_INDEX_TYPE="${VEDA_INDEX_TYPE:-hnsw}"
CURRENT_MEMORY_RATIO="${CURRENT_MEMORY_RATIO:-${MEMORY_VALUES%% *}}"
CURRENT_RESULT_LABEL="${CURRENT_RESULT_LABEL:-current}"
CURRENT_BUILD_EF="${CURRENT_BUILD_EF:-${EF_VALUES%% *}}"
CURRENT_OURS_COST_EF="${CURRENT_OURS_COST_EF:-18}"
KMEANS_COST_MODEL_JSON="${KMEANS_COST_MODEL_JSON:-${PROJECT_ROOT}/controller/kmeans/train/result/wiki_latency_cost_095_regrouped_20260715/wiki_latency_cost_095_regrouped_20260715_planner_model.json}"
export KMEANS_COST_MODEL_JSON
OURS_EFS_MODEL_JSON="${OURS_EFS_MODEL_JSON:-${PROJECT_ROOT}/controller/kmeans/train/result/yfcc_recall_fit/yfcc_hnsw_recall_model.json}"
OURS_TARGET_RECALL="${OURS_TARGET_RECALL:-0.95}"
ROLE_INDEX_TYPE="${ROLE_INDEX_TYPE:-hnsw}"
VEDA_INDEXING_THRESHOLD="${VEDA_INDEXING_THRESHOLD:-1000}"
CURRENT_BUILD_WORKERS="${CURRENT_BUILD_WORKERS:-10}"
HONEYBEE_RECALL="${HONEYBEE_RECALL:-0.99}"
HQI_MIN_SIZE="${HQI_MIN_SIZE:-512}"
HQI_INDEX_TYPE="${HQI_INDEX_TYPE:-hnsw}"
JIT="${JIT:-off}"
PARALLEL_WORKERS="${PARALLEL_WORKERS:-0}"
AUTH_FILTER="${AUTH_FILTER:-rls}"
HNSW_ITERATIVE_SCAN="${HNSW_ITERATIVE_SCAN:-off}"
VEDA_NATIVE_ALL_ROUTES="${VEDA_NATIVE_ALL_ROUTES:-false}"
ROUTE_LIMIT="${ROUTE_LIMIT:-0}"
ROUTE_PARALLELISM="${ROUTE_PARALLELISM:-1}"
ROUTE_WORKER_COUNT="${ROUTE_WORKER_COUNT:-64}"
ROUTE_SCHEDULER="${ROUTE_SCHEDULER:-inline}"
ROUTE_OVERFETCH="${ROUTE_OVERFETCH:-1}"
NATIVE_FILTER_LOCATION="${NATIVE_FILTER_LOCATION:-client}"
PG_PARALLEL_ROUTE_SCAN="${PG_PARALLEL_ROUTE_SCAN:-false}"
OURS_DB_FUNCTION="${OURS_DB_FUNCTION:-false}"
OURS_PRECOMPUTE_ACCESS="${OURS_PRECOMPUTE_ACCESS:-true}"

VERSION_PREFIX="${VERSION_PREFIX:-sweep}"
PLANNER_TIMEOUT="${PLANNER_TIMEOUT:-3h}"

bool_true() {
  case "$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')" in
    1|true|t|yes|y|on) return 0 ;;
    *) return 1 ;;
  esac
}

read -r -a METHOD_LIST <<< "${METHODS//,/ }"
if [[ "$(printf '%s' "${PLAN_MODE}" | tr '[:upper:]' '[:lower:]')" == "current" ]]; then
  MEMORY_LIST=("current")
else
  read -r -a MEMORY_LIST <<< "${MEMORY_VALUES//,/ }"
fi
read -r -a EF_LIST <<< "${EF_VALUES//,/ }"
QPS_TRIALS_INT="${QPS_TRIALS}"
if ! [[ "${QPS_TRIALS_INT}" =~ ^[0-9]+$ ]] || [[ "${QPS_TRIALS_INT}" -lt 1 ]]; then
  echo "QPS_TRIALS must be a positive integer, got: ${QPS_TRIALS}" >&2
  exit 2
fi

VEDA_NATIVE_ARGS=()
if bool_true "${VEDA_NATIVE_ALL_ROUTES}"; then
  VEDA_NATIVE_ARGS+=(--veda-native-all-routes)
fi
PG_PARALLEL_ROUTE_SCAN_ARGS=()
if bool_true "${PG_PARALLEL_ROUTE_SCAN}"; then
  PG_PARALLEL_ROUTE_SCAN_ARGS+=(--pg-parallel-route-scan)
fi
OURS_DB_FUNCTION_ARGS=()
if bool_true "${OURS_DB_FUNCTION}"; then
  OURS_DB_FUNCTION_ARGS+=(--ours-db-function)
fi
OURS_PRECOMPUTE_ACCESS_ARGS=()
if bool_true "${OURS_PRECOMPUTE_ACCESS}"; then
  OURS_PRECOMPUTE_ACCESS_ARGS+=(--ours-precompute-access)
else
  OURS_PRECOMPUTE_ACCESS_ARGS+=(--no-ours-precompute-access)
fi
QUERY_FILE_ARGS=()
if [[ -n "${QUERY_FILE}" ]]; then
  QUERY_FILE_ARGS+=(--query-file "${QUERY_FILE}")
fi
if [[ -n "${GROUND_TRUTH_FILE}" ]]; then
  QUERY_FILE_ARGS+=(--ground-truth-file "${GROUND_TRUTH_FILE}")
fi

for memory in "${MEMORY_LIST[@]}"; do
  for method in "${METHOD_LIST[@]}"; do
    version="${VERSION_PREFIX}_${method}_$(printf '%s' "${memory}" | sed 's/\./p/g')"
    if bool_true "${BUILD}"; then
      if [[ "$(printf '%s' "${PLAN_MODE}" | tr '[:upper:]' '[:lower:]')" == "current" ]]; then
        timeout --foreground "${PLANNER_TIMEOUT}" \
          "${PYTHON_BIN}" "${SCRIPT_DIR}/build_current_partitions.py" \
            --methods "${method}" \
            --memory-ratio "${CURRENT_MEMORY_RATIO}" \
            --ef-search "${CURRENT_BUILD_EF}" \
            --ours-cost-ef "${CURRENT_OURS_COST_EF}" \
            --ours-efs-model-json "${OURS_EFS_MODEL_JSON}" \
            --ours-target-recall "${OURS_TARGET_RECALL}" \
            --workers "${CURRENT_BUILD_WORKERS}" \
            --role-index-type "${ROLE_INDEX_TYPE}" \
            --ours-index-type "${OURS_INDEX_TYPE}" \
            --veda-index-type "${VEDA_INDEX_TYPE}" \
            --veda-indexing-threshold "${VEDA_INDEXING_THRESHOLD}" \
            --honeybee-recall "${HONEYBEE_RECALL}" \
            --hqi-min-size "${HQI_MIN_SIZE}" \
            --hqi-index-type "${HQI_INDEX_TYPE}" \
            --no-enable-split
      else
        timeout --foreground "${PLANNER_TIMEOUT}" \
          "${PYTHON_BIN}" "${SCRIPT_DIR}/build_versioned_plan.py" \
            --method "${method}" \
            --memory-ratio "${memory}" \
            --version "${version}" \
            --ours-index-type "${OURS_INDEX_TYPE}" \
            --veda-index-type "${VEDA_INDEX_TYPE}"
      fi
    fi
    if bool_true "${QPS}"; then
      for ef in "${EF_LIST[@]}"; do
        QPS_ARGS=(
          --methods "${method}"
          --query-source "${QUERY_SOURCE}"
          --query-count "${QUERY_COUNT}"
          --topk "${TOPK}"
          "${QUERY_FILE_ARGS[@]}"
          --query-repetitions "${QUERY_REPETITIONS}"
          --duration-seconds "${DURATION_SECONDS}"
          --concurrency "${CONCURRENCY}"
          --warmup-rounds "${WARMUP_ROUNDS}"
          --ef-search "${ef}"
          --index-mode "${INDEX_MODE}"
          --current-label "${CURRENT_RESULT_LABEL}"
          --jit "${JIT}"
          --parallel-workers "${PARALLEL_WORKERS}"
          --auth-filter "${AUTH_FILTER}"
          --hnsw-iterative-scan "${HNSW_ITERATIVE_SCAN}"
          --route-limit "${ROUTE_LIMIT}"
          --route-parallelism "${ROUTE_PARALLELISM}"
          --route-worker-count "${ROUTE_WORKER_COUNT}"
          --route-scheduler "${ROUTE_SCHEDULER}"
          --route-overfetch "${ROUTE_OVERFETCH}"
          --native-filter-location "${NATIVE_FILTER_LOCATION}"
          "${VEDA_NATIVE_ARGS[@]}"
          "${PG_PARALLEL_ROUTE_SCAN_ARGS[@]}"
          "${OURS_DB_FUNCTION_ARGS[@]}"
          "${OURS_PRECOMPUTE_ACCESS_ARGS[@]}"
        )
        if [[ "$(printf '%s' "${PLAN_MODE}" | tr '[:upper:]' '[:lower:]')" != "current" ]]; then
          QPS_ARGS+=(--memory-ratio "${memory}")
        fi
        if [[ "${QPS_TRIALS_INT}" -le 1 ]]; then
          "${PYTHON_BIN}" "${PROJECT_ROOT}/basic_benchmark/direct_pg_qps.py" "${QPS_ARGS[@]}"
        else
          method_label="$(printf '%s' "${method}" | tr '[:upper:]' '[:lower:]')"
          case "${method_label}" in
            squid|kmeans) method_label="ours" ;;
            qdtree|qd_tree) method_label="hqi" ;;
          esac
          if [[ "$(printf '%s' "${PLAN_MODE}" | tr '[:upper:]' '[:lower:]')" == "current" ]]; then
            memory_label="${CURRENT_RESULT_LABEL}"
          else
            memory_label="memory_$(printf '%s' "${memory}" | sed 's/\./p/g')"
          fi
          ef_label="ef_${ef}"
          output_dir="${PROJECT_ROOT}/basic_benchmark/result/direct_pg_qps/${method_label}/${memory_label}/${ef_label}"
          mkdir -p "${output_dir}"
          timestamp="$(date -u +%Y%m%d_%H%M%S)"
          trial_files=()
          for trial in $(seq 1 "${QPS_TRIALS_INT}"); do
            trial_output="${output_dir}/${timestamp}_trial${trial}.json"
            trial_files+=("${trial_output}")
            "${PYTHON_BIN}" "${PROJECT_ROOT}/basic_benchmark/direct_pg_qps.py" "${QPS_ARGS[@]}" --output "${trial_output}"
          done
          median_output="${output_dir}/median_${timestamp}.json"
          "${PYTHON_BIN}" - "${median_output}" "${trial_files[@]}" <<'PY'
import json
import statistics
import sys
from pathlib import Path

output = Path(sys.argv[1])
paths = [Path(value) for value in sys.argv[2:]]
trial_payloads = []
for path in paths:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise SystemExit(f"{path} does not contain a JSON list")
    trial_payloads.append(payload)

if not trial_payloads:
    raise SystemExit("no trial payloads")

max_len = max(len(payload) for payload in trial_payloads)
aggregated = []
for index in range(max_len):
    rows = [payload[index] for payload in trial_payloads if index < len(payload) and isinstance(payload[index], dict)]
    if not rows:
        continue
    keys = sorted({key for row in rows for key in row})
    out = {}
    for key in keys:
        values = [row.get(key) for row in rows if key in row]
        numeric = [
            value
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        if len(numeric) == len(values) and numeric:
            out[key] = statistics.median(numeric)
            continue
        first = values[0] if values else None
        out[key] = first if all(value == first for value in values) else first
    out["aggregation"] = "median"
    out["trial_count"] = len(rows)
    out["trial_files"] = [str(path) for path in paths]
    metric_values = {}
    for metric in ("qps", "recall_at_k", "avg_latency_ms", "p50_latency_ms", "p95_latency_ms", "wall_time_seconds", "effective_concurrency"):
        vals = [row.get(metric) for row in rows if isinstance(row.get(metric), (int, float)) and not isinstance(row.get(metric), bool)]
        if vals:
            metric_values[metric] = vals
    out["trial_metric_values"] = metric_values
    aggregated.append(out)

output.write_text(json.dumps(aggregated, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print(json.dumps({"median_output": str(output), "trial_count": len(paths)}, sort_keys=True))
PY
        fi
      done
    fi
  done
done
