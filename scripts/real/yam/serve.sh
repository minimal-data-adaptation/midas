#!/usr/bin/env bash
set -euo pipefail

: "${RUN_SPEC:?set RUN_SPEC to a trainer-generated real_run_spec.json}"
python_bin="${PYTHON_BIN:-python}"

args=(
  --run_spec "$RUN_SPEC"
  --host "${SERVER_BIND_HOST:-0.0.0.0}"
  --port "${SERVER_PORT:-8000}"
  --api_key "${SERVER_API_KEY:-}"
)
if [[ -n "${RESIDUAL_CHECKPOINT:-}" ]]; then
  args+=(--residual_checkpoint "$RESIDUAL_CHECKPOINT")
fi

"$python_bin" -u -m training.serve_real "${args[@]}" "$@"
