#!/usr/bin/env bash

# LIBERO-10 task 9: yellow white mug microwave. Same paper_v1 MiDAS hyperparameters.
# Set T_CKPT to this task's BC checkpoint; DEMO_HDF5 optionally overrides the demo.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
exec bash "$MIDAS_REPO_DIR/experiments/libero/run_midas.sh" \
    9 \
    pi05_libero10_yellow_white_mug_microwave_ep1 \
    libero10_task9_yellow_white_mug_microwave_ep1.hdf5
