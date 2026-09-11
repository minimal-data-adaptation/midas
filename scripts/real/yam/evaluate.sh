#!/usr/bin/env bash
set -euo pipefail

: "${RUN_SPEC:?set RUN_SPEC to a run real_run_spec.json}"
: "${EVAL_OUTPUT_DIR:?set EVAL_OUTPUT_DIR to a new evaluation output directory}"
python_bin="${PYTHON_BIN:-python}"

"$python_bin" -u -m training.evaluation.evaluate_real \
  --run_spec "$RUN_SPEC" \
  --output_dir "$EVAL_OUTPUT_DIR" \
  --server_host "${SERVER_HOST:-localhost}" \
  --server_port "${SERVER_PORT:-8000}" \
  --server_api_key "${SERVER_API_KEY:-}" \
  "$@"
