#!/usr/bin/env bash
set -euo pipefail

# Edit only these two values for a memory sweep.
MEMORY_VALUES="${MEMORY_VALUES:-1.0 1.5 2.0 2.5 3.0 3.5}"
METHODS="${METHODS:-veda effveda AnonySys OURS}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MEMORY_VALUES METHODS
exec "${SCRIPT_DIR}/run_memory_sweep.sh" "$@"
