#!/usr/bin/env bash

set -euo pipefail

REPO_DIR=/home/skowshik/vla/codebase/midas/midas
LAUNCHER="$REPO_DIR/experiments/libero/eval_midas_task8_both_mokapots_checkpoint200000_matrix.slurm"

cd "$REPO_DIR"

for model_seed in 11 42; do
    for perturbation in clean object language position_5cm; do
        case "$perturbation" in
            clean) short=clean ;;
            object) short=obj ;;
            language) short=lang ;;
            position_5cm) short=pos5cm ;;
        esac
        job_id=$(sbatch --parsable \
            --job-name="eval_t8_s${model_seed}_c200_${short}" \
            --export="ALL,MODEL_SEED=${model_seed},PERTURBATION=${perturbation}" \
            "$LAUNCHER")
        job_id=${job_id%%;*}
        printf 'seed=%s case=%s job=%s\n' "$model_seed" "$perturbation" "$job_id"
    done
done
