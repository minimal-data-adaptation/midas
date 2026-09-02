#!/usr/bin/env bash
#SBATCH --job-name=eval_midas_t8_ck250k_lang
#SBATCH --nodes=1
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=01:00:00
#SBATCH --requeue
#SBATCH --output=/home/skowshik/vla/codebase/midas/midas/experiments/libero/%x_%j.out
#SBATCH --error=/home/skowshik/vla/codebase/midas/midas/experiments/libero/%x_%j.err

set -euo pipefail

# 50-rollout language-perturbation evaluation of the seed-0 LIBERO-10 task-8
# MIDAS checkpoint. The generated suite paraphrases the moka-pot instruction
# according to LIBERO-PRO's OOD language configuration.

source /data/user_data/skowshik/anaconda3/etc/profile.d/conda.sh
conda activate /data/group_data/maxlab/common_datasets/skowshik/conda_envs/midas-libero

REPO_DIR=/home/skowshik/vla/codebase/midas/midas
CHECKPOINT_DIR=/data/group_data/maxlab/common_datasets/skowshik/midas_exps/pi05_base_v1/midas_pi05_base-ep1-both_mokapots-vlm-a-exec-paper_v1-s0-r2/checkpoint250000
BASE_POLICY_CHECKPOINT=/data/group_data/maxlab/common_datasets/pi05_common/pi05_libero10_both_mokapots_stove_ep1_ep86_filtered-v1/16000
EVAL_NAME=midas_t8_both_mokapots_s0_ckpt250000_language_perturbation_50rollouts
EVAL_ROOT=/data/group_data/maxlab/common_datasets/skowshik/midas_evals/pi05_base_v1
OUTPUT_DIR="$EVAL_ROOT/$EVAL_NAME"
PERTURB_SUITE_ROOT="$EVAL_ROOT/perturbation_suites"
LIBERO_TASK=KITCHEN_SCENE8_put_both_moka_pots_on_the_stove

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
    --seed 0 \
    --checkpoint_dir "$CHECKPOINT_DIR" \
    --num_evals 50 \
    --output_dir "$OUTPUT_DIR" \
    --save_eval_videos 1 \
    --eval_video_fps 20 \
    --round_robin_init_states 1 \
    --pos_perturb_radius 0.0 \
    --eval_config_path "$REPO_DIR/LIBERO-PRO/evaluation_config.yaml" \
    --perturb_input_dir "$REPO_DIR/LIBERO-PRO/libero/libero/bddl_files/libero_10" \
    --perturb_output_root "$PERTURB_SUITE_ROOT" \
    --perturb_seed 0 \
    --use_language 1 \
    --language_config "$REPO_DIR/LIBERO-PRO/libero_ood/ood_language.yaml" \
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
