#!/usr/bin/env bash
#SBATCH --job-name=midas_ep1_t8_pv1_s42
#SBATCH --nodes=1
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# LIBERO-10, episode-1 Pi-0.5 base, task 8: put both moka pots on the stove.
# Seed-42 replicate of run_midas_task8_both_mokapots_paper_v1.slurm.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MIDAS_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}}"
export MIDAS_REPO_DIR="$REPO_DIR"
# shellcheck source=../../scripts/experiment_env.sh
source "$REPO_DIR/scripts/experiment_env.sh"
midas_require_storage_roots
midas_activate_conda "${MIDAS_LIBERO_CONDA_ENV:-$MIDAS_DATA_ROOT/conda_envs/midas-libero}"

EXP_NAME="${MIDAS_EXP_NAME_OVERRIDE:-midas_pi05_base-ep1-both_mokapots-vlm-a-exec-paper_v1-s42-r2}"
export MIDAS_EXP_DIR="${MIDAS_EXP_DIR:-$MIDAS_DATA_ROOT/midas_exps/pi05_base_v1}"

cd "$REPO_DIR"
mkdir -p "$MIDAS_EXP_DIR"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$HOME/.config/libero-pro"
export OPENPI_DATA_HOME=/data/hf_cache/pi-models/openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONNOUSERSITE=1
: "${WANDB_ENTITY:?Set WANDB_ENTITY before submitting this job}"
export WANDB_ENTITY
mkdir -p "$LIBERO_CONFIG_PATH"

T_CONFIG=midas_pi05_libero10_both_mokapots_stove_ep1
T_CKPT="${T_CKPT:-$MIDAS_SHARED_DATA_ROOT/pi05_common/pi05_libero10_both_mokapots_stove_ep1_ep86_filtered-v1/16000}"
DEMO_HDF5="${DEMO_HDF5:-$MIDAS_DATA_ROOT/expert_hdf5/libero10_task8_put_both_moka_pots_on_the_stove_ep86.hdf5}"

# A requeued preempt job resumes the latest complete agent + replay-buffer
# snapshot. If preemption happens before the first checkpoint, it starts fresh.
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
    --env libero \
    --exp_name "$EXP_NAME" \
    --launch_group_id paper_v1 \
    --wandb 1 \
    --wandb_project libero-midas-pi05-base-v1 \
    --batch_size 64 \
    --discount 0.999 \
    --seed 42 \
    --max_steps 250000 \
    --eval_interval 25000 \
    --log_interval 500 \
    --checkpoint_interval 10000 \
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
    --task_suite_name libero_10 \
    --task_id 8 \
    --demo_hdf5_path "$DEMO_HDF5" \
    --num_demos 1 \
    --demo_bc_warmup_utd 10 \
    --actor_pop_base_actions 0 \
    "${RESUME_ARGS[@]}"
