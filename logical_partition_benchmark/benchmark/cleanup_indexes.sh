#!/usr/bin/env bash
# Delete HNSW index directories produced by the logical partition benchmarks/tests.

set -euo pipefail

# Resolve script location even under sudo.
SCRIPT_PATH="$(realpath "${BASH_SOURCE[0]}")"
SCRIPT_DIR="$(cd "$(dirname "${SCRIPT_PATH}")" && pwd)"
CONFIG_FILE="${SCRIPT_DIR}/config.json"
DRY_RUN=0
INDEX_ROOT=""

usage() {
    cat <<'EOF'
Usage: cleanup_indexes.sh [--config PATH] [--index-root PATH] [--dry-run]

  --config PATH       Path to config.json (defaults to benchmark/config.json)
  --index-root PATH   Override index root directly (skips reading config)
  --dry-run           Show what would be removed without deleting anything
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG_FILE="$2"
            shift 2
            ;;
        --index-root)
            INDEX_ROOT="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
    esac
done

if [[ -z "${INDEX_ROOT}" ]]; then
    if [[ ! -f "${CONFIG_FILE}" ]]; then
        ALT_CONFIG="${SCRIPT_DIR}/../config.json"
        if [[ -f "${ALT_CONFIG}" ]]; then
            CONFIG_FILE="${ALT_CONFIG}"
        else
            echo "Config file not found: ${CONFIG_FILE}" >&2
            exit 1
        fi
    fi
    # Extract index_storage_path (basic JSON, single line key).
    INDEX_ROOT="$(grep -E '"index_storage_path"\s*:' "${CONFIG_FILE}" | head -n1 | sed -E 's/.*"index_storage_path"\s*:\s*"([^"]+)".*/\1/')"
    if [[ -z "${INDEX_ROOT}" ]]; then
        echo "Failed to read index_storage_path from ${CONFIG_FILE}" >&2
        exit 1
    fi
fi

INDEX_ROOT="$(cd "${SCRIPT_DIR}" && realpath "${INDEX_ROOT}")"

if [[ "${INDEX_ROOT}" == "/" ]]; then
    echo "Refusing to operate on root directory" >&2
    exit 1
fi

echo "Index storage root: ${INDEX_ROOT}"

TARGETS=(
    "role_partition"
    "postfilter"
    "dynamic_partition"
    "physical_role_partition"
    "physical_postfilter"
    "physical_dynamic_partition"
)

for name in "${TARGETS[@]}"; do
    target="${INDEX_ROOT}/${name}"
    if [[ -d "${target}" ]]; then
        if [[ "${DRY_RUN}" -eq 1 ]]; then
            echo "[dry-run] would remove ${target}"
        else
            echo "Removing ${target}"
            rm -rf "${target}"
        fi
    else
        echo "${target} (missing)"
    fi
done
