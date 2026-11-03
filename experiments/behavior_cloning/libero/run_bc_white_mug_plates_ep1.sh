#!/usr/bin/env bash

# One-demo LIBERO-10 task 4. Run directly from any directory.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
exec bash "$MIDAS_REPO_DIR/experiments/behavior_cloning/libero/run_bc.sh" \
    pi05_libero10_white_mug_plates_ep1
