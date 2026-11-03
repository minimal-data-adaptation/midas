#!/usr/bin/env bash
#SBATCH --job-name=openpi_bc_libero_t7
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=256G
#SBATCH --time=48:00:00
#SBATCH --partition=preempt
#SBATCH --qos=preempt_qos
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# One-demo LIBERO-10 task 7. Submit from the repo root or export MIDAS_REPO_DIR.
# For four GPUs: sbatch --gres=gpu:4 --export=ALL <this-file>
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}}"
exec bash "$MIDAS_REPO_DIR/experiments/behavior_cloning/libero/run_bc.sh" \
    pi05_libero10_alphabet_soup_cream_cheese_basket_ep1
