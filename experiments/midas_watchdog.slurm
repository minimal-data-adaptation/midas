#!/usr/bin/env bash
#SBATCH --job-name=midas_watchdog
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=2G
#SBATCH --time=03:00:00
#SBATCH --partition=maxlab-cpu
#SBATCH --output=midas_watchdog_%j.out
#SBATCH --error=midas_watchdog_%j.err

set -uo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${MIDAS_REPO_DIR:-${SLURM_SUBMIT_DIR:-$(cd -- "$SCRIPT_DIR/.." && pwd)}}"
export MIDAS_REPO_DIR="$REPO_DIR"
# shellcheck source=../scripts/experiment_env.sh
source "$REPO_DIR/scripts/experiment_env.sh"
midas_require_storage_roots

WATCHDOG_SCRIPT="$REPO_DIR/experiments/midas_watchdog.slurm"
STATE_DIR="${STATE_DIR:-$MIDAS_DATA_ROOT/midas_exps/watchdog}"
LIBERO_EXP_DIR="${LIBERO_EXP_DIR:-$MIDAS_DATA_ROOT/midas_exps/pi05_base_v1}"
ROBOCASA_EXP_DIR="${ROBOCASA_EXP_DIR:-$MIDAS_DATA_ROOT/midas_exps/robocasa/pi05_base_v1}"
WATCHDOG_USER="${SLURM_JOB_USER:-${USER:-$(id -un)}}"
CHECK_INTERVAL=${MIDAS_WATCHDOG_INTERVAL:-300}
RUN_SECONDS=10740

mkdir -p "$STATE_DIR"
cd "$REPO_DIR"

log() {
    printf '[%s] %s\n' "$(date '+%Y-%m-%d %H:%M:%S %Z')" "$*"
}

record_state() {
    local key=$1
    local exp_name=$2
    local job_id=$3
    printf '%s|%s\n' "$exp_name" "$job_id" > "$STATE_DIR/$key.state"
}

active_job_ids() {
    local job_name=$1
    squeue -h -u "$WATCHDOG_USER" -n "$job_name" \
        -t PENDING,RUNNING,CONFIGURING,COMPLETING,RESIZING,REQUEUED,SUSPENDED \
        -o '%A' 2>/dev/null | sort -u
}

recorded_job_state() {
    local job_id=$1
    sacct -j "$job_id" -X -n -o State 2>/dev/null \
        | awk 'NF {sub(/\+.*/, "", $1); print $1; exit}'
}

has_checkpoint() {
    local run_dir=$1
    compgen -G "$run_dir/train_state/*.json" > /dev/null
}

ensure_job() {
    local key=$1
    local job_name=$2
    local launcher=$3
    local default_exp=$4
    local exp_root=$5
    local state_file="$STATE_DIR/$key.state"
    local active_ids
    active_ids=$(active_job_ids "$job_name")

    if [[ -n "$active_ids" ]]; then
        if [[ ! -s "$state_file" ]]; then
            record_state "$key" "$default_exp" "$(printf '%s\n' "$active_ids" | head -1)"
        fi
        log "$key healthy in Slurm: $(printf '%s' "$active_ids" | tr '\n' ' ')"
        return
    fi

    local exp_name=$default_exp
    local prior_job_id=
    if [[ -s "$state_file" ]]; then
        IFS='|' read -r exp_name prior_job_id < "$state_file"
    fi

    if [[ -n "$prior_job_id" ]]; then
        local prior_state
        prior_state=$(recorded_job_state "$prior_job_id")
        case "$prior_state" in
            COMPLETED)
                log "$key completed normally as job $prior_job_id; not relaunching"
                return
                ;;
            FAILED|CANCELLED|TIMEOUT|PREEMPTED|NODE_FAIL|OUT_OF_MEMORY|BOOT_FAIL|DEADLINE|REVOKED)
                ;;
            "")
                log "$key job $prior_job_id is absent from squeue but not yet visible in sacct; deferring"
                return
                ;;
            *)
                log "$key job $prior_job_id has transitional state $prior_state; deferring"
                return
                ;;
        esac

        if has_checkpoint "$exp_root/$exp_name"; then
            log "$key will resume $exp_name after $prior_state"
        else
            exp_name="${default_exp}-auto-$(date '+%Y%m%d_%H%M%S')"
            log "$key had no checkpoint after $prior_state; rotating to fresh run $exp_name"
        fi
    fi

    local new_job_id
    if new_job_id=$(sbatch --parsable \
        --export="ALL,MIDAS_EXP_NAME_OVERRIDE=$exp_name" "$launcher"); then
        new_job_id=${new_job_id%%;*}
        record_state "$key" "$exp_name" "$new_job_id"
        log "$key submitted as job $new_job_id with experiment $exp_name"
    else
        log "$key submission failed; will retry next check"
    fi
}

check_all_jobs() {
    ensure_job \
        libero_s0 midas_ep1_t8_paper_v1 \
        "$REPO_DIR/experiments/libero/run_midas_task8_both_mokapots_paper_v1.slurm" \
        midas_pi05_base-ep1-both_mokapots-vlm-a-exec-paper_v1-s0-r2 \
        "$LIBERO_EXP_DIR"
    ensure_job \
        libero_s11 midas_ep1_t8_pv1_s11 \
        "$REPO_DIR/experiments/libero/run_midas_task8_both_mokapots_paper_v1_s11.slurm" \
        midas_pi05_base-ep1-both_mokapots-vlm-a-exec-paper_v1-s11-r2 \
        "$LIBERO_EXP_DIR"
    ensure_job \
        libero_s42 midas_ep1_t8_pv1_s42 \
        "$REPO_DIR/experiments/libero/run_midas_task8_both_mokapots_paper_v1_s42.slurm" \
        midas_pi05_base-ep1-both_mokapots-vlm-a-exec-paper_v1-s42-r2 \
        "$LIBERO_EXP_DIR"
    ensure_job \
        robocasa_cabinet_a7 midas_rc_cab_a7_s0_r1 \
        "$REPO_DIR/experiments/robocasa/run_midas_pick_place_counter_to_cabinet_exact_replay_l1_s1_ep32_paper_v2.sh" \
        midas_rc_pnp_cab_er_l1s1_ep32_paper_v2_s0_a7_r1 \
        "$ROBOCASA_EXP_DIR"
    ensure_job \
        robocasa_fridge_a7 midas_rc_fridge_a7_s0_r1 \
        "$REPO_DIR/experiments/robocasa/run_midas_pick_place_fridge_drawer_to_shelf_l50_s37_paper_v2.slurm" \
        midas_rc_ppfridge_l50s37_paper_v2_s0_a7_r1 \
        "$ROBOCASA_EXP_DIR"
    ensure_job \
        robocasa_coffee_a7 midas_rc_coffee_a7_s0_r1 \
        "$REPO_DIR/experiments/robocasa/run_midas_prepare_coffee_l25_s29_var1_paper_v2.slurm" \
        midas_rc_coffee_l25s29_var1_paper_v2_s0_a7_r1 \
        "$ROBOCASA_EXP_DIR"
}

if [[ "${MIDAS_WATCHDOG_ONCE:-0}" != 1 && -n "${SLURM_JOB_ID:-}" ]]; then
    other_running=$(squeue -h -u "$WATCHDOG_USER" -n midas_watchdog \
        -t RUNNING -o '%A' 2>/dev/null | awk -v self="$SLURM_JOB_ID" '$1 != self')
    if [[ -n "$other_running" ]]; then
        log "Another watchdog is already running ($other_running); exiting duplicate"
        exit 0
    fi

    if next_job_id=$(sbatch --parsable --dependency="afterany:$SLURM_JOB_ID" \
        "$WATCHDOG_SCRIPT"); then
        next_job_id=${next_job_id%%;*}
        log "Queued successor watchdog $next_job_id for after job $SLURM_JOB_ID"
    else
        log "WARNING: failed to queue successor watchdog"
    fi
fi

while true; do
    check_all_jobs
    if [[ "${MIDAS_WATCHDOG_ONCE:-0}" == 1 ]]; then
        break
    fi
    if (( SECONDS >= RUN_SECONDS )); then
        log "Three-hour handoff reached; exiting for successor watchdog"
        break
    fi
    remaining=$((RUN_SECONDS - SECONDS))
    sleep_for=$CHECK_INTERVAL
    if (( sleep_for > remaining )); then
        sleep_for=$remaining
    fi
    sleep "$sleep_for"
done
