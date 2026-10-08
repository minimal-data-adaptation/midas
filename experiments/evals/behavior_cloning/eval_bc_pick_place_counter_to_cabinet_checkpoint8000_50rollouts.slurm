#!/usr/bin/env bash
#SBATCH --job-name=eval_bc_rc_cab_ckpt8k
#SBATCH --nodes=1
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=12:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Fifty-rollout, base-policy-only evaluation of the standalone RoboCasa
# counter-to-cabinet BC checkpoint. Replay training episode 32, optionally with
# object-position noise; every rollout is saved as an MP4 in a new job-specific
# directory.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MIDAS_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}}"
export MIDAS_REPO_DIR="$REPO_DIR"
# shellcheck source=../../../scripts/experiment_env.sh
source "$REPO_DIR/scripts/experiment_env.sh"
midas_require_storage_roots
if [[ -n "${MIDAS_ROBOCASA_PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$MIDAS_ROBOCASA_PYTHON_BIN"
elif [[ -x "$MIDAS_DATA_ROOT/conda_envs/midas-robocasa/bin/python" ]]; then
    PYTHON_BIN="$MIDAS_DATA_ROOT/conda_envs/midas-robocasa/bin/python"
else
    midas_activate_conda "${MIDAS_ROBOCASA_CONDA_ENV:-$MIDAS_DATA_ROOT/conda_envs/midas-robocasa}"
    PYTHON_BIN=python
fi

CONFIG_NAME=pi05_robocasa_bc_counter_to_cabinet_l1_s1_ep32
CHECKPOINT_STEP="${CHECKPOINT_STEP:-8000}"
OPENPI_CHECKPOINT_ROOT="${OPENPI_CHECKPOINT_ROOT:-$MIDAS_DATA_ROOT/openpi_bc_checkpoints}"
BC_RUN_NAME="${BC_RUN_NAME:-robocasa_bc_counter_to_cabinet_l1_s1_ep32_v1_s0_10691811}"
BC_CHECKPOINT="${BC_CHECKPOINT:-$OPENPI_CHECKPOINT_ROOT/robocasa_bc/$CONFIG_NAME/$BC_RUN_NAME/$CHECKPOINT_STEP}"
ROBOCASA_EVAL_OBJECT_POSE_NOISE="${ROBOCASA_EVAL_OBJECT_POSE_NOISE:-0.0}"

for required_path in "$BC_CHECKPOINT/params" "$BC_CHECKPOINT/assets/$CONFIG_NAME/norm_stats.json"; do
    if [[ ! -e "$required_path" ]]; then
        echo "Incomplete BC checkpoint; missing: $required_path" >&2
        exit 2
    fi
done

EVAL_ROOT="${EVAL_ROOT:-$MIDAS_DATA_ROOT/midas/evals/behavior_cloning}"
NOISE_TAG="${ROBOCASA_EVAL_OBJECT_POSE_NOISE//./p}"
EVAL_NAME="${BC_EVAL_NAME:-robocasa_bc_pick_place_counter_to_cabinet_s0_checkpoint${CHECKPOINT_STEP}_exact_replay_ep32_object_pose_noise${NOISE_TAG}_50rollouts_job${SLURM_JOB_ID:-manual}}"
OUTPUT_DIR="${OUTPUT_DIR:-$EVAL_ROOT/$EVAL_NAME}"

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export DISPLAY=:0
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

# The pixel-exact replay preflight is meaningful only without perturbations.
if [[ "$ROBOCASA_EVAL_OBJECT_POSE_NOISE" == "0" || "$ROBOCASA_EVAL_OBJECT_POSE_NOISE" == "0.0" ]]; then
    JAX_PLATFORMS=cpu "$PYTHON_BIN" -u -m training.evaluation.verify_robocasa_replay \
        --config "$CONFIG_NAME" \
        --env-name PickPlaceCounterToCabinet \
        --output-dir "$OUTPUT_DIR"
else
    echo "Skipping pixel-exact replay preflight with object pose noise=$ROBOCASA_EVAL_OBJECT_POSE_NOISE"
fi

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
    --robocasa_env_name PickPlaceCounterToCabinet \
    --robocasa_split target \
    --robocasa_horizon_scale 1.5 \
    --robocasa_horizon_cap 500 \
    --robocasa_use_right_view 1 \
    --robocasa_eval_object_pose_noise "$ROBOCASA_EVAL_OBJECT_POSE_NOISE"
