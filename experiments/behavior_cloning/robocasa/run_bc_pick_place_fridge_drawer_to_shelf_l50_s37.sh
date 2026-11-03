#!/usr/bin/env bash

set -euo pipefail

# Same fridge task/scene filters as MIDAS: layout 50, style 37, one demo.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
exec bash "$MIDAS_REPO_DIR/experiments/behavior_cloning/robocasa/run_bc.sh" \
    pi05_robocasa_bc_fridge_drawer_to_shelf_l50_s37
