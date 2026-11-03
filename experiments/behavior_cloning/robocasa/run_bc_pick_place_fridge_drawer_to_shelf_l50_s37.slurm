#!/usr/bin/env bash
#SBATCH --job-name=openpi_bc_rc_fridge_l50s37
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

# Same fridge task/scene filters as MIDAS: layout 50, style 37, one demo.
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}}"
exec bash "$MIDAS_REPO_DIR/experiments/behavior_cloning/robocasa/run_bc.sh" \
    pi05_robocasa_bc_fridge_drawer_to_shelf_l50_s37
