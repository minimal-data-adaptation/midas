#!/usr/bin/env bash
set -euo pipefail

profile="${1:-libero}"
case "$profile" in
  libero|ci-cpu)
    simulator="LIBERO-PRO"
    ;;
  robocasa)
    simulator="robocasa"
    ;;
  real)
    simulator=""
    ;;
  *)
    echo "usage: $0 {libero|robocasa|real|ci-cpu}" >&2
    exit 2
    ;;
esac

python -m pip install --require-hashes -r "locks/lock-${profile}.txt"
editables=(-e . -e openpi -e openpi/packages/openpi-client)
if [[ -n "$simulator" ]]; then
  editables+=(-e "$simulator")
fi
python -m pip install --no-deps --no-build-isolation "${editables[@]}"
python tools/verify_lock_coverage.py --profile "$profile"
