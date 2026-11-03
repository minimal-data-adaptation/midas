#!/usr/bin/env bash
# Run the ten one-demo LIBERO-10 configs. See ../README.md.
set -euo pipefail

case "${1:-}" in
    pi05_libero10_*|--help|-h|"") ;;
    *) echo "Expected a pi05_libero10_* config; see ../README.md." >&2; exit 2 ;;
esac
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export MIDAS_REPO_DIR="${MIDAS_REPO_DIR:-$(cd -- "$SCRIPT_DIR/../../.." && pwd)}"
exec bash "$MIDAS_REPO_DIR/experiments/behavior_cloning/run_bc.sh" "$@"
