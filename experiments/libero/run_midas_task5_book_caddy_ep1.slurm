#!/usr/bin/env bash
#SBATCH --job-name=midas_libero10_t5
#SBATCH --nodes=1
#SBATCH --partition=preempt
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

# LIBERO-10 task 5: book caddy. Same paper_v1 MiDAS hyperparameters.
# Set T_CKPT to this task's BC checkpoint; DEMO_HDF5 optionally overrides the demo.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}}"
exec bash "$MIDAS_REPO_DIR/experiments/libero/run_midas.sh" \
    5 \
    pi05_libero10_book_caddy_ep1 \
    libero10_task5_book_caddy_ep1.hdf5
