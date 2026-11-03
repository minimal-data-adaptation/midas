#!/usr/bin/env bash

# Shared full-training runner for the three task-specific Slurm launchers.
# Configure machine-specific paths in your external experiments.env file.
set -euo pipefail

CONFIG_NAME="${1:?Usage: bash run_bc.sh <robocasa-bc-config>}"
case "$CONFIG_NAME" in
    pi05_robocasa_bc_counter_to_cabinet_l1_s1_ep32)
        RUN_PREFIX=robocasa_bc_counter_to_cabinet_l1_s1_ep32
        ;;
    pi05_robocasa_bc_fridge_drawer_to_shelf_l50_s37)
        RUN_PREFIX=robocasa_bc_fridge_drawer_to_shelf_l50_s37
        ;;
    pi05_robocasa_bc_prepare_coffee_l25_s29)
        RUN_PREFIX=robocasa_bc_prepare_coffee_l25_s29
        ;;
    *)
        echo "Unsupported RoboCasa BC config: $CONFIG_NAME" >&2
        exit 2
        ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
# shellcheck source=../../../scripts/experiment_env.sh
source "$MIDAS_REPO_DIR/scripts/experiment_env.sh"

# Explicit OpenPI paths also work without MIDAS_DATA_ROOT. Otherwise use the
# existing experiment storage layout, in directories separate from MIDAS runs.
export OPENPI_DATASET_ROOT="${OPENPI_DATASET_ROOT:-${MIDAS_DATA_ROOT:+$MIDAS_DATA_ROOT/robocasa_assets}}"
export OPENPI_ASSETS_ROOT="${OPENPI_ASSETS_ROOT:-${MIDAS_DATA_ROOT:+$MIDAS_DATA_ROOT/openpi_bc_assets}}"
export OPENPI_CHECKPOINT_ROOT="${OPENPI_CHECKPOINT_ROOT:-${MIDAS_DATA_ROOT:+$MIDAS_DATA_ROOT/openpi_bc_checkpoints}}"
: "${OPENPI_DATASET_ROOT:?Set OPENPI_DATASET_ROOT or MIDAS_DATA_ROOT}"
: "${OPENPI_ASSETS_ROOT:?Set OPENPI_ASSETS_ROOT or MIDAS_DATA_ROOT}"
: "${OPENPI_CHECKPOINT_ROOT:?Set OPENPI_CHECKPOINT_ROOT or MIDAS_DATA_ROOT}"

if [[ -n "${OPENPI_PYTHON_BIN:-}" ]]; then
    PYTHON_BIN="$OPENPI_PYTHON_BIN"
else
    midas_activate_conda "${MIDAS_ROBOCASA_CONDA_ENV:-${MIDAS_DATA_ROOT:+$MIDAS_DATA_ROOT/conda_envs/}midas-robocasa}"
    PYTHON_BIN=python
fi

export PYTHONPATH="$MIDAS_REPO_DIR/openpi/src:$MIDAS_REPO_DIR/openpi/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export POLARS_MAX_THREADS="${POLARS_MAX_THREADS:-2}"
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
# A CPU setting used for normalization/trace must not disable GPU training.
unset JAX_PLATFORMS
cd "$MIDAS_REPO_DIR/openpi"

SEED="${OPENPI_BC_SEED:-0}"
RUN_ID="${SLURM_JOB_ID:-$(date -u +%Y%m%dT%H%M%S)_$$}"
EXP_NAME="${OPENPI_BC_EXP_NAME:-${RUN_PREFIX}_v1_s${SEED}_${RUN_ID}}"
if [[ "$EXP_NAME" == */* || "$EXP_NAME" == . || "$EXP_NAME" == .. ]]; then
    echo "OPENPI_BC_EXP_NAME must be a single directory name." >&2
    exit 2
fi
RUN_DIR="$OPENPI_CHECKPOINT_ROOT/robocasa_bc/$CONFIG_NAME/$EXP_NAME"
TRAIN_ARGS=(scripts/train.py "$CONFIG_NAME" --exp-name "$EXP_NAME" --seed "$SEED")

# With no overrides, retain the configs' batch size 64, 100,000 steps, four
# loader workers, 4,000-step checkpoint interval, and remaining BC settings.
for override in \
    batch-size:OPENPI_BC_BATCH_SIZE \
    num-train-steps:OPENPI_BC_NUM_TRAIN_STEPS \
    num-workers:OPENPI_BC_NUM_WORKERS \
    save-interval:OPENPI_BC_SAVE_INTERVAL; do
    option="${override%%:*}"
    variable="${override#*:}"
    if [[ -n "${!variable:-}" ]]; then
        TRAIN_ARGS+=("--$option" "${!variable}")
    fi
done

case "${OPENPI_BC_WANDB_ENABLED:-1}" in
    1)
        export WANDB_ENTITY="${WANDB_ENTITY:-${MIDAS_WANDB_ENTITY:-}}"
        : "${WANDB_ENTITY:?Set WANDB_ENTITY, or set OPENPI_BC_WANDB_ENABLED=0}"
        ;;
    0)
        TRAIN_ARGS+=(--no-wandb-enabled)
        ;;
    *)
        echo "OPENPI_BC_WANDB_ENABLED must be 0 or 1." >&2
        exit 2
        ;;
esac

# A requeue keeps the same Slurm job ID/run name and restores any completed
# checkpoint. A separately submitted job receives a new name. Never overwrite.
if [[ -d "$RUN_DIR" ]]; then
    TRAIN_ARGS+=(--resume)
    echo "Resuming RoboCasa BC run: $RUN_DIR"
else
    echo "Starting RoboCasa BC run: $RUN_DIR"
fi

NORM_STATS="$OPENPI_ASSETS_ROOT/$CONFIG_NAME/$CONFIG_NAME/norm_stats.json"
if [[ ! -f "$NORM_STATS" ]]; then
    echo "Computing missing normalization statistics: $NORM_STATS"
    JAX_PLATFORMS=cpu "$PYTHON_BIN" scripts/compute_norm_stats.py --config-name "$CONFIG_NAME"
fi
JAX_PLATFORMS=cpu "$PYTHON_BIN" scripts/trace_robocasa_bc.py --config-name "$CONFIG_NAME"

echo "Training config: $CONFIG_NAME; seed: $SEED; experiment: $EXP_NAME"
exec "$PYTHON_BIN" "${TRAIN_ARGS[@]}"
