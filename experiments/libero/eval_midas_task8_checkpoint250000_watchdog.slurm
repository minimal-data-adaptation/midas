#!/usr/bin/env bash
#SBATCH --job-name=midas_t8_eval_watch
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --partition=maxlab-cpu
#SBATCH --qos=maxlab_cpu_qos
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MIDAS_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/../.." && pwd)}}"
export MIDAS_REPO_DIR="$REPO_DIR"
# shellcheck source=../../scripts/experiment_env.sh
source "$REPO_DIR/scripts/experiment_env.sh"
midas_require_storage_roots

EVAL_LAUNCHER="$REPO_DIR/experiments/libero/eval_midas_task8_both_mokapots_checkpoint250000_no_perturbations.slurm"
OUTPUT_DIR="${OUTPUT_DIR:-$MIDAS_DATA_ROOT/midas_evals/pi05_base_v1/midas_t8_both_mokapots_s0_ckpt250000_no_perturbations_50rollouts}"
STATE_DIR="${STATE_DIR:-$MIDAS_DATA_ROOT/midas_evals/watchdog}"
STATE_FILE="$STATE_DIR/midas_t8_ckpt250000_clean.state"
WATCHDOG_SCRIPT="$REPO_DIR/experiments/libero/eval_midas_task8_checkpoint250000_watchdog.slurm"
CHECK_INTERVAL=${MIDAS_EVAL_WATCHDOG_INTERVAL:-300}
RUN_SECONDS=10740
INITIAL_JOB_ID=10282313

mkdir -p "$STATE_DIR"
cd "$REPO_DIR"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

record_job() {
    printf '%s\n' "$1" > "$STATE_FILE"
}

current_job_id() {
    if [[ -s "$STATE_FILE" ]]; then
        head -1 "$STATE_FILE"
    else
        printf '%s\n' "$INITIAL_JOB_ID"
    fi
}

active_state() {
    local job_id=$1
    squeue -h -j "$job_id" -o '%T' 2>/dev/null | head -1
}

accounting_state() {
    local job_id=$1
    sacct -j "$job_id" -X -n -o State 2>/dev/null \
        | awk 'NF {sub(/\+.*/, "", $1); print $1; exit}'
}

artifacts_complete() {
    python - "$OUTPUT_DIR" <<'PY'
import csv
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
summary_json = root / "summary.json"
summary_csv = root / "summary.csv"
videos = root / "videos"
if not summary_json.is_file() or not summary_csv.is_file() or not videos.is_dir():
    raise SystemExit(1)
metadata = json.loads(summary_json.read_text())
if metadata.get("episodes") != 50:
    raise SystemExit(1)
with summary_csv.open(newline="") as handle:
    rows = list(csv.DictReader(handle))
episode_rows = [row for row in rows if row.get("rollout_id") != "AGGREGATE"]
if len(episode_rows) != 50:
    raise SystemExit(1)
if len(list(videos.glob("rollout_*.mp4"))) != 50:
    raise SystemExit(1)
PY
}

check_once() {
    local job_id state
    job_id=$(current_job_id)
    state=$(active_state "$job_id")
    case "$state" in
        PENDING|RUNNING|CONFIGURING|COMPLETING|REQUEUED|RESIZING|SUSPENDED)
            record_job "$job_id"
            log "evaluation job $job_id healthy in Slurm: $state"
            return 0
            ;;
    esac

    state=$(accounting_state "$job_id")
    case "$state" in
        COMPLETED)
            if artifacts_complete; then
                log "evaluation job $job_id completed with verified 50-rollout artifacts"
                return 10
            fi
            log "evaluation job $job_id completed but artifacts are incomplete; resubmitting"
            ;;
        FAILED|CANCELLED|TIMEOUT|PREEMPTED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|REVOKED)
            log "evaluation job $job_id ended as $state; resubmitting"
            ;;
        "")
            log "evaluation job $job_id is absent from squeue but not visible in sacct; deferring"
            return 0
            ;;
        *)
            log "evaluation job $job_id has transitional accounting state $state; deferring"
            return 0
            ;;
    esac

    local new_job_id
    if new_job_id=$(sbatch --parsable "$EVAL_LAUNCHER"); then
        new_job_id=${new_job_id%%;*}
        record_job "$new_job_id"
        log "submitted replacement evaluation job $new_job_id"
    else
        log "replacement submission failed; will retry next check"
    fi
    return 0
}

if [[ ! -s "$STATE_FILE" ]]; then
    record_job "$INITIAL_JOB_ID"
fi

check_once
initial_status=$?
if (( initial_status == 10 )); then
    exit 0
fi

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    if next_job_id=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" "$WATCHDOG_SCRIPT"); then
        next_job_id=${next_job_id%%;*}
        log "queued successor watchdog $next_job_id after $SLURM_JOB_ID"
    else
        log "WARNING: failed to queue successor watchdog"
    fi
fi

while true; do
    if (( SECONDS >= RUN_SECONDS )); then
        log "three-hour handoff reached; exiting for successor watchdog"
        break
    fi
    remaining=$((RUN_SECONDS - SECONDS))
    sleep_for=$CHECK_INTERVAL
    if (( sleep_for > remaining )); then
        sleep_for=$remaining
    fi
    sleep "$sleep_for"
    check_once
    status=$?
    if (( status == 10 )); then
        log "verified evaluation complete; exiting"
        break
    fi
done
