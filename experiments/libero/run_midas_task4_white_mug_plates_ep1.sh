#!/usr/bin/env bash

# LIBERO-10 task 4: white mug plates. Same paper_v1 MiDAS hyperparameters.
# Set T_CKPT to this task's BC checkpoint; DEMO_HDF5 optionally overrides the demo.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
exec bash "$MIDAS_REPO_DIR/experiments/libero/run_midas.sh" \
    4 \
    pi05_libero10_white_mug_plates_ep1 \
    libero10_task4_white_mug_left_yellow_mug_right_ep1.hdf5
