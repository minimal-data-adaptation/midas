#!/usr/bin/env bash

# Shared environment bootstrap for experiment launchers. This file is intended
# to be sourced after a launcher has derived and exported MIDAS_REPO_DIR.

midas_require_storage_roots() {
    if [[ -z "${MIDAS_DATA_ROOT:-}" ]]; then
        echo "MIDAS_DATA_ROOT is required; set it to your writable MIDAS data directory." >&2
        return 2
    fi

    MIDAS_SHARED_DATA_ROOT="${MIDAS_SHARED_DATA_ROOT:-$(dirname -- "$MIDAS_DATA_ROOT")}"
    export MIDAS_DATA_ROOT MIDAS_SHARED_DATA_ROOT
}

midas_activate_conda() {
    local environment=$1

    if [[ -n "${MIDAS_CONDA_SH:-}" ]]; then
        if [[ ! -f "$MIDAS_CONDA_SH" ]]; then
            echo "MIDAS_CONDA_SH does not exist: $MIDAS_CONDA_SH" >&2
            return 2
        fi
        # shellcheck disable=SC1090
        source "$MIDAS_CONDA_SH"
    elif command -v conda >/dev/null 2>&1; then
        eval "$(conda shell.bash hook)"
    else
        echo "Conda is unavailable; set MIDAS_CONDA_SH to conda.sh." >&2
        return 2
    fi

    conda activate "$environment"
}
