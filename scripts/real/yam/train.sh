#!/usr/bin/env bash
set -euo pipefail

: "${YAM_ENV_CONFIG:?set YAM_ENV_CONFIG to the yam_teleop environment YAML}"
: "${PI_CKPT_DIR:?set PI_CKPT_DIR to the trained OpenPI checkpoint directory}"
: "${DEMO_DATA_ROOT:?set DEMO_DATA_ROOT to the LeRobot home directory}"
: "${DEMO_NORM_STATS_PATH:?set DEMO_NORM_STATS_PATH to norm_stats.json}"

task_config="${TASK_CONFIG:-configs/real/yam/pickplace_a.yaml}"
python_bin="${PYTHON_BIN:-python}"

"$python_bin" -u -m training.launch_train_real \
  --task_config "$task_config" \
  --yam_env_config_path "$YAM_ENV_CONFIG" \
  --pi_05_ckpt_dir "$PI_CKPT_DIR" \
  --demo_data_root "$DEMO_DATA_ROOT" \
  --demo_norm_stats_path "$DEMO_NORM_STATS_PATH" \
  --output_dir "${OUTPUT_DIR:-experiments/real}" \
  --server_host "${SERVER_HOST:-localhost}" \
  --server_port "${SERVER_PORT:-8000}" \
  --server_api_key "${SERVER_API_KEY:-}" \
  "$@"
