#!/usr/bin/env bash

set -euo pipefail

# Same cabinet demonstration as MIDAS: layout 1, style 1, episode 32.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
exec bash "$MIDAS_REPO_DIR/experiments/behavior_cloning/robocasa/run_bc.sh" \
    pi05_robocasa_bc_counter_to_cabinet_l1_s1_ep32
