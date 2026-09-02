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
  *)
    echo "usage: $0 {libero|robocasa|ci-cpu}" >&2
    exit 2
    ;;
esac

python -m pip install --require-hashes -r "locks/lock-${profile}.txt"
python -m pip install --no-deps --no-build-isolation -e . -e openpi \
  -e openpi/packages/openpi-client -e "$simulator"
python tools/verify_lock_coverage.py --profile "$profile"
