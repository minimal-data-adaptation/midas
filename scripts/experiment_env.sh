#!/usr/bin/env bash

# Shared environment bootstrap for experiment launchers. This file is intended
# to be sourced after a launcher has derived and exported MIDAS_REPO_DIR.

midas_require_storage_roots() {
    if [[ -z "${MIDAS_DATA_ROOT:-}" ]]; then
        echo "MIDAS_DATA_ROOT is required; set it to your writable MIDAS data directory." >&2
        return 2
    fi

    if [[ "$MIDAS_DATA_ROOT" != /* ]]; then
        echo "MIDAS_DATA_ROOT must be an absolute path." >&2
        return 2
    fi
    export MIDAS_DATA_ROOT
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

# Use the active environment unless a Python binary or Conda environment was
# explicitly selected. PYTHON_BIN is consumed by the calling launch script.
midas_select_python() {
    local profile=$1
    local python_variable conda_variable
    case "$profile" in
        libero)
            python_variable=MIDAS_LIBERO_PYTHON_BIN
            conda_variable=MIDAS_LIBERO_CONDA_ENV
            ;;
        robocasa)
            python_variable=MIDAS_ROBOCASA_PYTHON_BIN
            conda_variable=MIDAS_ROBOCASA_CONDA_ENV
            ;;
        *)
            echo "Unknown environment profile: $profile" >&2
            return 2
            ;;
    esac
    if [[ -n "${MIDAS_PYTHON_BIN:-}" ]]; then
        PYTHON_BIN="$MIDAS_PYTHON_BIN"
    elif [[ -n "${!python_variable:-}" ]]; then
        PYTHON_BIN="${!python_variable}"
    else
        if [[ -n "${!conda_variable:-}" ]]; then
            midas_activate_conda "${!conda_variable}" || return
        fi
        PYTHON_BIN=python
    fi
    if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
        echo "Python is unavailable: $PYTHON_BIN; activate an environment or set MIDAS_PYTHON_BIN." >&2
        return 2
    fi
}
