#!/usr/bin/env bash

set -euo pipefail

# RoboCasa MIDAS equivalent of the residual PA-RL paper-v2 cabinet job.
# PickPlaceCounterToCabinet, layout 1, style 1, exact replay of episode 32 with
# 0.025 m object-position noise.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MIDAS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
export MIDAS_REPO_DIR="$REPO_DIR"
# shellcheck source=../../scripts/experiment_env.sh
source "$REPO_DIR/scripts/experiment_env.sh"
midas_require_storage_roots
RUN_ID="${MIDAS_RUN_ID:-$(date -u +%Y%m%dT%H%M%S)_$$}"
midas_select_python robocasa

EXP_NAME="${MIDAS_EXP_NAME_OVERRIDE:-midas_rc_pnp_cab_er_l1s1_ep32_paper_v2_s0_a7_r1_seed_repro_test_${RUN_ID}}"
export MIDAS_EXP_DIR="${MIDAS_EXP_DIR:-$MIDAS_DATA_ROOT/midas_exps/robocasa/pi05_base_v1}"

cd "$REPO_DIR"
mkdir -p "$MIDAS_EXP_DIR"

export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$MIDAS_DATA_ROOT/openpi_cache}"
export OPENPI_DATASET_ROOT="${OPENPI_DATASET_ROOT:-$MIDAS_DATA_ROOT/robocasa_assets}"
: "${ROBOCASA_EVAL_DATA_ROOT:?Set ROBOCASA_EVAL_DATA_ROOT before running this script}"
export ROBOCASA_EVAL_DATA_ROOT
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
ROBOCASA_SOURCE_DIR="${ROBOCASA_SOURCE_DIR:-$REPO_DIR/robocasa}"
export PYTHONPATH="$ROBOCASA_SOURCE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_ENTITY="${WANDB_ENTITY:-${MIDAS_WANDB_ENTITY:-}}"
: "${WANDB_ENTITY:?Set WANDB_ENTITY to your W&B username or team}"

T_CONFIG=midas_pi05_robocasa_pick_place_counter_to_cabinet_exact_replay_l1_s1_ep32
: "${T_CKPT:?Set T_CKPT to the frozen BC checkpoint directory for this task}"

# Reusing the experiment name resumes a complete agent/replay snapshot.
RESUME_ARGS=()
RUN_DIR="$MIDAS_EXP_DIR/$EXP_NAME"
if compgen -G "$RUN_DIR/train_state/*.json" > /dev/null; then
    RESUME_ARGS=(--resume_dir "$RUN_DIR")
    echo "Resuming MIDAS run from $RUN_DIR"
else
    echo "Starting fresh MIDAS run in $RUN_DIR"
fi

"$PYTHON_BIN" -u -m training.launch_train_sim \
    --algo midas \
    --env robocasa \
    --prefix midas-robocasa-pick_place_counter_to_cabinet-exact_replay_l1_s1_ep32-seed0 \
    --exp_name "$EXP_NAME" \
    --launch_group_id paper_v2 \
    --wandb 1 \
    --wandb_project robocasa-midas-pi05-base-v1 \
    --batch_size 64 \
    --discount 0.999 \
    --seed 0 \
    --max_steps 2500000 \
    --eval_interval 25000 \
    --log_interval 500 \
    --checkpoint_interval 10000 \
    --keep_checkpoint_interval 50000 \
    --eval_episodes 50 \
    --multi_grad_step 1 \
    --encoder_type small \
    --start_online_updates 500 \
    --resize_image 100 \
    --action_magnitude 1.0 \
    --query_freq 10 \
    --chunk_len 10 \
    --hidden_dims 512 \
    --pi_05_config "$T_CONFIG" \
    --pi_05_ckpt_dir "$T_CKPT" \
    --action_dim 7 \
    --residual_alpha 0.5 \
    --use_zero_residual_initially 1 \
    --use_huber_loss 0 \
    --num_critic_updates 30 \
    --num_actor_updates 10 \
    --bc_reg_coeff 0.0 \
    --bc_on_success_only 1 \
    --success_buffer_ratio 0.20 \
    --success_buffer_min_size 20 \
    --predict_a_exec 1 \
    --midas_num_samples 16 \
    --midas_num_elites 8 \
    --midas_num_grad_steps 30 \
    --midas_step_size 0.001 \
    --midas_use_trust_region 0 \
    --bc_warmup_steps 10000 \
    --bc_warmup_num_critic_updates 8 \
    --bc_warmup_num_actor_updates 8 \
    --tau 0.05 \
    --use_vlm_embedding 1 \
    --actor_pop_base_actions 0 \
    --robocasa_env_name PickPlaceCounterToCabinet \
    --robocasa_split target \
    --robocasa_horizon_scale 1.5 \
    --robocasa_horizon_cap 500 \
    --robocasa_use_right_view 1 \
    --robocasa_eval_object_pose_noise 0.025 \
    "${RESUME_ARGS[@]}"
