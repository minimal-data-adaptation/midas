#!/usr/bin/env bash

set -euo pipefail

# Reuse the canonical launcher so every training hyperparameter, including
# seed 0, remains identical. Only the experiment/W&B ID and output path change.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MIDAS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export MIDAS_REPO_DIR="$REPO_DIR"
export MIDAS_RUN_ID="${MIDAS_RUN_ID:-$(date -u +%Y%m%dT%H%M%S)_$$}"
export MIDAS_EXP_NAME_OVERRIDE="${MIDAS_EXP_NAME_OVERRIDE:-midas_rc_pnp_cab_er_l1s1_ep32_paper_v2_s0_a7_r1_seed_repro_test_repeat_${MIDAS_RUN_ID}}"
exec bash "$REPO_DIR/experiments/robocasa/run_midas_pick_place_counter_to_cabinet_exact_replay_l1_s1_ep32_paper_v2.sh"
