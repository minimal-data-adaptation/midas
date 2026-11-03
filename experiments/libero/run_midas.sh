#!/usr/bin/env bash
# Shared MiDAS settings from the original task-8 paper_v1 launcher.
# Per-task .slurm files provide the task ID, OpenPI BC config, and demo filename.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: bash run_midas.sh <task-id> <openpi-config> <demo-filename>" >&2
    echo "Normally submit a task-specific .slurm file; see README.md." >&2
    exit 2
fi
TASK_ID="$1"
T_CONFIG="$2"
DEMO_FILENAME="$3"
if [[ ! "$TASK_ID" =~ ^[0-9]$ || "$T_CONFIG" != pi05_libero10_*_ep1 || "$DEMO_FILENAME" == */* ]]; then
    echo "Expected a LIBERO-10 task ID (0-9), one-demo config, and demo filename." >&2
    exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}"
# shellcheck source=../../scripts/experiment_env.sh
source "$MIDAS_REPO_DIR/scripts/experiment_env.sh"
midas_require_storage_roots

# Checkpoint directory must include params/ and assets/ (not just params/).
# Require an explicit choice rather than silently choosing another task's weights.
: "${T_CKPT:?Set T_CKPT to the frozen OpenPI BC checkpoint directory for this task}"
DEMO_HDF5="${DEMO_HDF5:-$MIDAS_DATA_ROOT/expert_hdf5/$DEMO_FILENAME}"
for path in "$MIDAS_DATA_ROOT" "$T_CKPT" "$DEMO_HDF5"; do
    if [[ "$path" != /* ]]; then
        echo "Storage, checkpoint, and demo paths must be absolute: $path" >&2
        exit 2
    fi
done
if [[ ! -d "$T_CKPT/params" || ! -d "$T_CKPT/assets" ]]; then
    echo "Invalid BC checkpoint (expected params/ and assets/): $T_CKPT" >&2
    exit 2
fi
if [[ ! -f "$DEMO_HDF5" ]]; then
    echo "Demo file not found: $DEMO_HDF5; set DEMO_HDF5 to override." >&2
    exit 2
fi

SEED="${MIDAS_SEED:-0}"
if [[ ! "$SEED" =~ ^[0-9]+$ ]]; then
    echo "MIDAS_SEED must be a non-negative integer." >&2
    exit 2
fi
TASK_SLUG="${T_CONFIG#pi05_libero10_}"
TASK_SLUG="${TASK_SLUG%_ep1}"
RUN_ID="${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%S)_$$}"
EXP_NAME="${MIDAS_EXP_NAME_OVERRIDE:-midas_pi05_base-ep1-${TASK_SLUG}-vlm-a-exec-paper_v1-s${SEED}-${RUN_ID}}"
if [[ "$EXP_NAME" == */* || "$EXP_NAME" == . || "$EXP_NAME" == .. ]]; then
    echo "MIDAS_EXP_NAME_OVERRIDE must be a single directory name." >&2
    exit 2
fi
export MIDAS_EXP_DIR="${MIDAS_EXP_DIR:-$MIDAS_DATA_ROOT/midas_exps/libero10/pi05_base_v1}"
if [[ "$MIDAS_EXP_DIR" != /* ]]; then
    echo "MIDAS_EXP_DIR must be an absolute path." >&2
    exit 2
fi
: "${WANDB_ENTITY:?Set WANDB_ENTITY before submitting this job}"
export WANDB_ENTITY

if [[ -n "${MIDAS_PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$MIDAS_PYTHON_BIN"
else
    midas_activate_conda "${MIDAS_LIBERO_CONDA_ENV:-$MIDAS_DATA_ROOT/conda_envs/midas-libero}"
    PYTHON_BIN=python
fi

cd "$MIDAS_REPO_DIR"
mkdir -p "$MIDAS_EXP_DIR"
# Keep the GPU mask assigned by Slurm; EGL uses device 0 within that allocation.
export MUJOCO_EGL_DEVICE_ID="${MUJOCO_EGL_DEVICE_ID:-0}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-$HOME/.config/libero-pro}"
export OPENPI_DATA_HOME="${OPENPI_DATA_HOME:-$MIDAS_DATA_ROOT/openpi_cache}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONNOUSERSITE=1
unset JAX_PLATFORMS
mkdir -p "$LIBERO_CONFIG_PATH"

# Requeues reuse their job ID/name and resume the latest saved training state.
RESUME_ARGS=()
RUN_DIR="$MIDAS_EXP_DIR/$EXP_NAME"
if compgen -G "$RUN_DIR/train_state/*.json" > /dev/null; then
    RESUME_ARGS=(--resume_dir "$RUN_DIR")
    echo "Resuming MiDAS run from $RUN_DIR"
else
    echo "Starting fresh MiDAS run in $RUN_DIR"
fi
echo "LIBERO-10 task: $TASK_ID; config: $T_CONFIG; seed: $SEED"
echo "Frozen BC checkpoint: $T_CKPT"
echo "Demonstration: $DEMO_HDF5"

exec "$PYTHON_BIN" -u -m training.launch_train_sim \
    --algo midas \
    --env libero \
    --exp_name "$EXP_NAME" \
    --launch_group_id paper_v1 \
    --wandb 1 \
    --wandb_project libero-midas-pi05-base-v1 \
    --batch_size 64 \
    --discount 0.999 \
    --seed "$SEED" \
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
    --task_id "$TASK_ID" \
    --demo_hdf5_path "$DEMO_HDF5" \
    --num_demos 1 \
    --demo_bc_warmup_utd 10 \
    --actor_pop_base_actions 0 \
    "${RESUME_ARGS[@]}"
