#!/usr/bin/env bash

set -euo pipefail

# Fifty-rollout, base-policy-only evaluation of the standalone RoboCasa
# fridge-drawer-to-shelf BC checkpoint. Replay training episode 40 exactly and
# save every rollout as an MP4 in a new run-specific directory.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MIDAS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
export MIDAS_REPO_DIR="$REPO_DIR"
# shellcheck source=../../../scripts/experiment_env.sh
source "$REPO_DIR/scripts/experiment_env.sh"
midas_require_storage_roots
RUN_ID="${MIDAS_RUN_ID:-$(date -u +%Y%m%dT%H%M%S)_$$}"
midas_select_python robocasa

CONFIG_NAME=pi05_robocasa_bc_fridge_drawer_to_shelf_l50_s37
CHECKPOINT_STEP="${CHECKPOINT_STEP:-12000}"
OPENPI_CHECKPOINT_ROOT="${OPENPI_CHECKPOINT_ROOT:-$MIDAS_DATA_ROOT/openpi_bc_checkpoints}"
: "${BC_CHECKPOINT:?Set BC_CHECKPOINT to the BC checkpoint directory to evaluate}"
readonly ROBOCASA_EVAL_OBJECT_POSE_NOISE=0.0

for required_path in "$BC_CHECKPOINT/params" "$BC_CHECKPOINT/assets/$CONFIG_NAME/norm_stats.json"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Incomplete BC checkpoint; missing: $required_path" >&2
        exit 2
    fi
done

EVAL_ROOT="${EVAL_ROOT:-$MIDAS_DATA_ROOT/midas/evals/behavior_cloning}"
EVAL_NAME="${BC_EVAL_NAME:-robocasa_bc_pick_place_fridge_drawer_to_shelf_s0_checkpoint${CHECKPOINT_STEP}_exact_replay_ep40_50rollouts_${RUN_ID}}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_ROOT/$EVAL_NAME}"

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENPI_DATASET_ROOT="${OPENPI_DATASET_ROOT:-$MIDAS_DATA_ROOT/robocasa_assets}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$OPENPI_DATASET_ROOT/openpi_cache}"
export OPENPI_ASSETS_ROOT="${OPENPI_ASSETS_ROOT:-$MIDAS_DATA_ROOT/openpi_bc_assets}"
export OPENPI_CHECKPOINT_ROOT
export ROBOCASA_EVAL_DATA_ROOT="${ROBOCASA_EVAL_DATA_ROOT:-$OPENPI_DATASET_ROOT/data_dumps/robocasa}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
ROBOCASA_SOURCE_DIR="${ROBOCASA_SOURCE_DIR:-$REPO_DIR/robocasa}"
export PYTHONPATH="$ROBOCASA_SOURCE_DIR:$REPO_DIR/openpi/src:$REPO_DIR/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

echo "Evaluating BC checkpoint: $BC_CHECKPOINT"
echo "Writing 50-rollout evaluation to: $OUTPUT_DIR"

JAX_PLATFORMS=cpu "$PYTHON_BIN" -u -m training.evaluation.verify_robocasa_replay \
    --config "$CONFIG_NAME" \
    --env-name PickPlaceFridgeDrawerToShelf \
    --output-dir "$OUTPUT_DIR"

"$PYTHON_BIN" -u -m training.evaluation.evaluate_sim \
    --eval_base_only 1 \
    --env robocasa \
    --exp_name "$EVAL_NAME" \
    --wandb 0 \
    --seed 0 \
    --num_evals 50 \
    --output_dir "$OUTPUT_DIR" \
    --save_eval_videos 1 \
    --eval_video_fps 20 \
    --resize_image 224 \
    --query_freq 10 \
    --chunk_len 10 \
    --requery_base_policy 1 \
    --pi_05_config "$CONFIG_NAME" \
    --pi_05_ckpt_dir "$BC_CHECKPOINT" \
    --robocasa_env_name PickPlaceFridgeDrawerToShelf \
    --robocasa_split target \
    --robocasa_horizon_scale 1.5 \
    --robocasa_horizon_cap 500 \
    --robocasa_use_right_view 1 \
    --robocasa_eval_object_pose_noise "$ROBOCASA_EVAL_OBJECT_POSE_NOISE"
