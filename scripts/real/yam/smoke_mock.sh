#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$repo_root"
# This smoke is intentionally hardware-free, including accelerator hardware.
export JAX_PLATFORMS=cpu
smoke_root="$(mktemp -d "${TMPDIR:-/tmp}/midas-real-smoke.XXXXXX")"
port="${MIDAS_REAL_SMOKE_PORT:-18765}"
python_bin="${PYTHON_BIN:-python}"
trainer_log="$smoke_root/trainer.log"
server_log="$smoke_root/server.log"
evaluator_log="$smoke_root/evaluator.log"
trainer_pid=""
server_pid=""

cleanup() {
  if [[ -n "$trainer_pid" ]]; then kill "$trainer_pid" 2>/dev/null || true; fi
  if [[ -n "$server_pid" ]]; then kill "$server_pid" 2>/dev/null || true; fi
}
trap cleanup EXIT INT TERM

"$python_bin" -u -m training.launch_train_real \
  --mock_env 1 --mock_episode_len 4 --rollout_interactive_success_label 0 \
  --num_demos 0 --bc_warmup_steps 0 --midas_use_trust_region 0 \
  --use_vlm_embedding 0 --resize_image 16 --chunk_len 2 --query_freq 2 \
  --max_traj_len 4 --max_steps 1 --batch_size 2 --online_buffer_capacity 32 \
  --success_buffer_ratio 0 --num_initial_traj_collect 1 --start_online_updates 0 \
  --num_online_gradsteps_batch 1 --num_critic_updates 1 --num_actor_updates 1 \
  --actor_push_interval 1 --checkpoint_interval 1 --keep_checkpoint_interval 1 \
  --hidden_dims 16 16 --cnn_features 8 8 --cnn_strides 2 2 --cnn_padding SAME \
  --latent_dim 8 --num_qs 2 --midas_num_samples 2 --midas_num_elites 1 \
  --midas_num_grad_steps 1 --color_jitter 0 --aug_next 0 --wandb 0 \
  --server_port "$port" --server_connect_timeout 600 \
  --output_dir "$smoke_root" --exp_name run >"$trainer_log" 2>&1 &
trainer_pid="$!"

spec="$smoke_root/run/real_run_spec.json"
for _ in $(seq 1 120); do
  [[ -f "$spec" ]] && break
  kill -0 "$trainer_pid" 2>/dev/null || { sed -n '1,240p' "$trainer_log"; exit 1; }
  sleep 1
done
[[ -f "$spec" ]] || { echo "trainer did not publish $spec" >&2; exit 1; }

"$python_bin" -u -m training.serve_real --run_spec "$spec" --host 127.0.0.1 \
  --port "$port" --mock_base_policy >"$server_log" 2>&1 &
server_pid="$!"

if ! wait "$trainer_pid"; then
  sed -n '1,260p' "$trainer_log"
  sed -n '1,260p' "$server_log"
  exit 1
fi
trainer_pid=""
kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
server_pid=""

test -f "$smoke_root/run/train_state/1.json"
test -d "$smoke_root/run/checkpoint1"

"$python_bin" -u -m training.serve_real --run_spec "$spec" --host 127.0.0.1 \
  --port "$port" --mock_base_policy \
  --residual_checkpoint "$smoke_root/run/checkpoint1" >>"$server_log" 2>&1 &
server_pid="$!"
if ! "$python_bin" -u -m training.evaluation.evaluate_real \
  --run_spec "$spec" --server_host 127.0.0.1 --server_port "$port" \
  --server_connect_timeout 600 --mock_env 1 --mock_episode_len 4 \
  --interactive_labels 0 --num_episodes 1 --max_episode_steps 4 \
  --save_video 0 --output_dir "$smoke_root/evaluation" >"$evaluator_log" 2>&1; then
  sed -n '1,260p' "$evaluator_log"
  sed -n '1,320p' "$server_log"
  exit 1
fi
kill "$server_pid" 2>/dev/null || true
wait "$server_pid" 2>/dev/null || true
server_pid=""
test -f "$smoke_root/evaluation/summary.json"
echo "MIDAS real mock smoke passed; artifacts: $smoke_root/run"
