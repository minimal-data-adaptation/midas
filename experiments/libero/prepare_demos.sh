#!/usr/bin/env bash
# Prepare the ten LIBERO MiDAS task demonstrations.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
source "$MIDAS_REPO_DIR/scripts/experiment_env.sh"
midas_require_storage_roots
midas_select_python libero
: "${LIBERO_LEROBOT_ROOT:?Set LIBERO_LEROBOT_ROOT to the local physical-intelligence/libero dataset}"

exec "$PYTHON_BIN" "$MIDAS_REPO_DIR/scripts/prepare_libero_demos.py" \
    --dataset-root "$LIBERO_LEROBOT_ROOT" \
    --output-dir "${DEMO_OUTPUT_DIR:-$MIDAS_DATA_ROOT/expert_hdf5}" \
    "$@"
