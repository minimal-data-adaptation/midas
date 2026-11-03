#!/usr/bin/env bash

# LIBERO-10 task 7: alphabet soup cream cheese basket. Same paper_v1 MiDAS hyperparameters.
# Set T_CKPT to this task's BC checkpoint; DEMO_HDF5 optionally overrides the demo.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
exec bash "$MIDAS_REPO_DIR/experiments/libero/run_midas.sh" \
    7 \
    pi05_libero10_alphabet_soup_cream_cheese_basket_ep1 \
    libero10_task7_alphabet_soup_cream_cheese_basket_ep1.hdf5
