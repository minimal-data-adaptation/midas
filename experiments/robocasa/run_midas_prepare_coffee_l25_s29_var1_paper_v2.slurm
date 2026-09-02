#!/usr/bin/env bash
#SBATCH --job-name=midas_rc_coffee_a7_s0_r1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --requeue
#SBATCH --exclude=babel-w9-26,babel-y9-08,babel-n5-20
#SBATCH --output=/home/skowshik/vla/codebase/midas/midas/experiments/robocasa/%x_%j.out
#SBATCH --error=/home/skowshik/vla/codebase/midas/midas/experiments/robocasa/%x_%j.err

set -euo pipefail

# RoboCasa MIDAS equivalent of the residual PA-RL paper-v2 coffee var1 job.
# PrepareCoffee, layout 25, style 29, seed 0.

source /data/user_data/skowshik/anaconda3/etc/profile.d/conda.sh
conda activate /data/group_data/maxlab/common_datasets/skowshik/conda_envs/midas-robocasa

REPO_DIR=/home/skowshik/vla/codebase/midas/midas
EXP_NAME="${MIDAS_EXP_NAME_OVERRIDE:-midas_rc_coffee_l25s29_var1_paper_v2_s0_a7_r1}"
export MIDAS_EXP_DIR=/data/group_data/maxlab/common_datasets/skowshik/midas_exps/robocasa/pi05_base_v1

cd "$REPO_DIR"
mkdir -p "$MIDAS_EXP_DIR"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export DISPLAY=:0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export OPENPI_DATA_HOME=/data/hf_cache/pi-models/openpi
export OPENPI_DATASET_ROOT=/data/group_data/maxlab/common_datasets/skowshik/robocasa_assets
export ROBOCASA_EVAL_DATA_ROOT=/home/skowshik/vla/codebase/openpi/data_dumps/robocasa
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="/home/skowshik/vla/codebase/openpi_robocasa/robocasa${PYTHONPATH:+:$PYTHONPATH}"
export WANDB_ENTITY=skowshik-carnegie-mellon-university

T_CONFIG=midas_pi05_robocasa_prepare_coffee_l25_s29_var1
SOURCE_CONFIG=pi05_robocasa_single_task_lora_vision_fullft_action_prepare_coffee_l25_s29
T_CKPT=/data/group_data/maxlab/common_datasets/skowshik/pi05_robocasa/${SOURCE_CONFIG}/${SOURCE_CONFIG}-v1/12000

# Requeued jobs resume once a complete agent/replay snapshot exists.
RESUME_ARGS=()
RUN_DIR="$MIDAS_EXP_DIR/$EXP_NAME"
if compgen -G "$RUN_DIR/train_state/*.json" > /dev/null; then
    RESUME_ARGS=(--resume_dir "$RUN_DIR")
    echo "Resuming MIDAS run from $RUN_DIR"
else
    echo "Starting fresh MIDAS run in $RUN_DIR"
fi

python -u -m training.launch_train_sim \
    --algo midas \
    --env robocasa \
    --prefix midas-robocasa-prepare_coffee-l25_s29-var1-seed0 \
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
    --robocasa_env_name PrepareCoffee \
    --robocasa_split target \
    --robocasa_horizon_scale 1.0 \
    --robocasa_horizon_cap 800 \
    --robocasa_use_right_view 1 \
    "${RESUME_ARGS[@]}"
