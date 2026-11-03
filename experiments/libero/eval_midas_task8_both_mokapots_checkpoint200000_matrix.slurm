#!/usr/bin/env bash
#SBATCH --job-name=eval_midas_t8_ck200k
#SBATCH --nodes=1
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -euo pipefail

# Parameterized 50-rollout evaluation of the seed-11/seed-42 task-8 MIDAS
# checkpoint at step 200k. MODEL_SEED identifies the trained checkpoint;
# PERTURBATION is one of clean, object, language, or position_5cm.

: "${MODEL_SEED:?Submit with MODEL_SEED=11 or MODEL_SEED=42}"
: "${PERTURBATION:?Submit with PERTURBATION=clean, object, language, or position_5cm}"

case "$MODEL_SEED" in
    11|42) ;;
    *) echo "Unsupported MODEL_SEED=$MODEL_SEED" >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MIDAS_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}}"
export MIDAS_REPO_DIR="$REPO_DIR"
# shellcheck source=../../scripts/experiment_env.sh
source "$REPO_DIR/scripts/experiment_env.sh"
midas_require_storage_roots
midas_activate_conda "${MIDAS_LIBERO_CONDA_ENV:-$MIDAS_DATA_ROOT/conda_envs/midas-libero}"

CHECKPOINT_DIR="${CHECKPOINT_DIR:-$MIDAS_DATA_ROOT/midas_exps/pi05_base_v1/midas_pi05_base-ep1-both_mokapots-vlm-a-exec-paper_v1-s${MODEL_SEED}-r2/checkpoint200000}"
BASE_POLICY_CHECKPOINT="${BASE_POLICY_CHECKPOINT:-$MIDAS_SHARED_DATA_ROOT/pi05_common/pi05_libero10_both_mokapots_stove_ep1_ep86_filtered-v1/16000}"
EVAL_ROOT="${EVAL_ROOT:-$MIDAS_DATA_ROOT/midas_evals/pi05_base_v1}"
PERTURB_SUITE_ROOT="$EVAL_ROOT/perturbation_suites"
LIBERO_TASK=KITCHEN_SCENE8_put_both_moka_pots_on_the_stove

# Keep evaluation randomness fixed across trained seeds for paired comparison.
EVAL_SEED=0
PERTURB_ARGS=()
case "$PERTURBATION" in
    clean)
        CASE_LABEL=no_perturbations
        PERTURB_ARGS=(--pos_perturb_radius 0.0)
        ;;
    object)
        CASE_LABEL=object_perturbation
        PERTURB_ARGS=(
            --pos_perturb_radius 0.0
            --eval_config_path "$REPO_DIR/LIBERO-PRO/evaluation_config.yaml"
            --perturb_input_dir "$REPO_DIR/LIBERO-PRO/libero/libero/bddl_files/libero_10"
            --perturb_output_root "$PERTURB_SUITE_ROOT"
            --perturb_seed "$EVAL_SEED"
            --use_object 1
            --object_config "$REPO_DIR/LIBERO-PRO/libero_ood/ood_object.yaml"
        )
        ;;
    language)
        CASE_LABEL=language_perturbation
        PERTURB_ARGS=(
            --pos_perturb_radius 0.0
            --eval_config_path "$REPO_DIR/LIBERO-PRO/evaluation_config.yaml"
            --perturb_input_dir "$REPO_DIR/LIBERO-PRO/libero/libero/bddl_files/libero_10"
            --perturb_output_root "$PERTURB_SUITE_ROOT"
            --perturb_seed "$EVAL_SEED"
            --use_language 1
            --language_config "$REPO_DIR/LIBERO-PRO/libero_ood/ood_language.yaml"
        )
        ;;
    position_5cm)
        CASE_LABEL=position_5cm_perturbation
        PERTURB_ARGS=(
            --pos_perturb_radius 0.05
            --pos_perturb_objects auto
            --pos_perturb_settle_secs 5.0
        )
        ;;
    *)
        echo "Unsupported PERTURBATION=$PERTURBATION" >&2
        exit 2
        ;;
esac

EVAL_NAME=midas_t8_both_mokapots_s${MODEL_SEED}_ckpt200000_${CASE_LABEL}_50rollouts
OUTPUT_DIR="$EVAL_ROOT/$EVAL_NAME"

if [[ ! -d "$CHECKPOINT_DIR" ]]; then
    echo "Checkpoint directory does not exist: $CHECKPOINT_DIR" >&2
    exit 1
fi

cd "$REPO_DIR"
mkdir -p "$OUTPUT_DIR" "$PERTURB_SUITE_ROOT"

export CUDA_VISIBLE_DEVICES=0
export MUJOCO_EGL_DEVICE_ID=0
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export LIBERO_CONFIG_PATH="$HOME/.config/libero-pro"
export OPENPI_DATA_HOME=/data/hf_cache/pi-models/openpi
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export PYTHONNOUSERSITE=1
mkdir -p "$LIBERO_CONFIG_PATH"

python -u -m training.evaluation.evaluate_sim \
    --algo midas \
    --env libero \
    --exp_name "$EVAL_NAME" \
    --wandb 0 \
    --seed "$EVAL_SEED" \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --num_evals 50 \
    --output_dir "$OUTPUT_DIR" \
    --save_eval_videos 1 \
    --eval_video_fps 20 \
    --round_robin_init_states 1 \
    "${PERTURB_ARGS[@]}" \
    --batch_size 64 \
    --encoder_type small \
    --resize_image 100 \
    --action_magnitude 1.0 \
    --query_freq 10 \
    --chunk_len 10 \
    --hidden_dims 512 \
    --pi_05_config midas_pi05_libero10_both_mokapots_stove_ep1 \
    --pi_05_ckpt_dir "$BASE_POLICY_CHECKPOINT" \
    --residual_alpha 0.5 \
    --predict_a_exec 1 \
    --midas_num_samples 16 \
    --midas_num_elites 8 \
    --midas_num_grad_steps 30 \
    --midas_step_size 0.001 \
    --midas_use_trust_region 0 \
    --use_huber_loss 0 \
    --num_critic_updates 30 \
    --num_actor_updates 10 \
    --bc_reg_coeff 0.0 \
    --bc_on_success_only 1 \
    --tau 0.05 \
    --use_vlm_embedding 1 \
    --actor_pop_base_actions 0 \
    --task_suite_name libero_10 \
    --libero_task "$LIBERO_TASK"
