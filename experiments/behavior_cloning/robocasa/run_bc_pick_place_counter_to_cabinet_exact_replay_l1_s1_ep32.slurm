#!/usr/bin/env bash
#SBATCH --job-name=openpi_bc_rc_cab_l1s1_ep32
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

set -euo pipefail

# Same cabinet demonstration as MIDAS: layout 1, style 1, episode 32.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}}"
exec bash "$MIDAS_REPO_DIR/experiments/behavior_cloning/robocasa/run_bc.sh" \
    pi05_robocasa_bc_counter_to_cabinet_l1_s1_ep32
