#!/usr/bin/env bash
set -euo pipefail

export XLA_PYTHON_CLIENT_PREALLOCATE=false
python -m compileall -q midas training envs
python -m pytest -q tests
python - <<'PY'
import jax
import midas

print("midas import: ok")
print("jax devices:", jax.devices())
PY
