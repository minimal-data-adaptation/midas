# YAM real-world training

The YAM workflow is deliberately separate from simulation. It has dedicated
trainer, server, evaluator, configuration, scripts, dependency profile, and
mock environment. No simulation launcher imports `yam_teleop` or LeRobot.

## Install

Create the pinned Python 3.11 environment from the repository root:

```bash
conda env create -f environment-real.yml
conda activate midas-real
bash scripts/install_profile.sh real
python -m pip check
```

The real lock installs MIDAS/OpenPI runtime dependencies, LeRobot 0.3.3, the
WebSocket client/server, and video support. `yam_teleop` is site-specific and is
not vendored or locked here. Install the package that provides
`yam_teleop.env.YAMBimanualEnv` separately in this environment. Its constructor
must accept the environment YAML path and expose `reset`, `step`, and either
`get_observation` or `_get_obs`.

`pip check` reports that editable OpenPI declares `gym-aloha`. That package is
for a different robot and is intentionally absent from the YAM lock; this is
the only expected metadata complaint before the external `yam_teleop` install.

Before using hardware, run the CPU-only end-to-end smoke:

```bash
PYTHON_BIN=python bash scripts/real/yam/smoke_mock.sh
```

The smoke starts separate trainer and server processes, synchronizes the actor,
collects from the deterministic mock YAM, performs an update, disarms the
server, and verifies a v3 checkpoint/replay manifest. It never imports
`yam_teleop` or uses a GPU.

## Data and policy preparation

The expected robot contract is fixed:

- cameras, in order: `top`, `left_wrist`, `right_wrist`;
- live camera frames: BGR HWC; LeRobot demonstration frames: RGB;
- state/action: 14 values (`left joints x6`, left gripper, `right joints x6`,
  right gripper);
- Pi context: `(chunk_len, 14)` normalized actions;
- learned/executed chunk: `(query_freq, 14)`, where
  `0 < query_freq <= chunk_len <= 60`.

Demonstrations use a LeRobot repository with `observation.state`, `action`, the
three `observation.images.*` fields, task metadata, and (for the `a`/`b`
profiles) `orig_traj_id_6`. Norm statistics must contain 14-D action `q01` and
`q99` arrays. The trainer hashes the norm-stat file into the immutable run spec.

Convert the original task-directory tree into the supported combined dataset:

```bash
python tools/yam/convert_to_lerobot.py \
  --mode combined \
  --input_dir /path/to/yam-recordings \
  --lerobot_home /path/to/lerobot_home \
  --repo_id local/yam_combined
```

The converter refuses to overwrite an existing dataset. A custom YAML mapping
from task-directory name to prompt can be supplied with `--task_prompts`.
OpenPI fine-tuning reads the same location when
`HF_LEROBOT_HOME=/path/to/lerobot_home` is exported before launch.

Evaluation output can be converted with `--mode evaluation`; aborted/retried
episodes are skipped and `is_success` is preserved. Seed those rollouts into a
later run with `--eval_rollout_repo_id`, `--eval_rollout_data_root`, and
optionally `--eval_rollout_norm_stats_path` (otherwise the demo norm stats are
used). Successful evaluation episodes are also routed to the success replay.

OpenPI provides these YAM fine-tuning configurations:

```text
pi05_yam_combined_lora
pi05_yam_pickplace_a_lora   pi05_yam_pickplace_b_lora
pi05_yam_arrange_a_lora     pi05_yam_arrange_all_lora
pi05_yam_wipe_a_lora        pi05_yam_wipe_b_lora
pi05_yam_wipe_all_lora
```

The matching portable runtime YAMLs live in `configs/real/yam/`. They contain
only the OpenPI config, prompt, repository ID, and trajectory filter. Machine
paths remain command-line arguments or environment variables.

## Safe launch order

Use a unique API key when trainer/server traffic crosses machines. Ensure the
port is reachable only on the intended network.

1. Start the trainer. It creates the output directory, atomically writes
   `real_run_spec.json`, initializes the learner, and waits for the server.

   ```bash
   export TASK_CONFIG=configs/real/yam/pickplace_a.yaml
   export YAM_ENV_CONFIG=/path/to/yam_env.yaml
   export PI_CKPT_DIR=/path/to/pi05/checkpoint/19999
   export DEMO_DATA_ROOT=/path/to/lerobot_home
   export DEMO_NORM_STATS_PATH=/path/to/norm_stats.json
   export OUTPUT_DIR=/path/to/midas-real-runs
   export SERVER_HOST=robot-server.example
   export SERVER_PORT=8000
   export SERVER_API_KEY='replace-with-a-secret'
   bash scripts/real/yam/train.sh --exp_name pickplace_a_run1
   ```

2. After `OUTPUT_DIR/pickplace_a_run1/real_run_spec.json` exists, start the
   policy server on the machine with the frozen Pi checkpoint:

   ```bash
   export RUN_SPEC=/path/to/midas-real-runs/pickplace_a_run1/real_run_spec.json
   export SERVER_PORT=8000
   export SERVER_API_KEY='replace-with-the-same-secret'
   bash scripts/real/yam/serve.sh
   ```

The server starts disarmed. Frozen-policy inference is available only for demo
preprocessing; robot inference fails closed until the trainer completes BC (or
restores a checkpoint) and pushes a structurally validated actor. Actor params
and batch statistics are swapped atomically. Protocol, spec hash, actor
signature, array shapes, dtypes, finite values, camera order, and normalized
action bounds are validated on both sides. The trainer disarms the server on a
normal or exceptional shutdown.

Hardware training is rejected unless it has either a compatible restored
MIDAS actor or non-empty demonstrations plus BC warmup. For a first run, keep
the default real-world trust region enabled. Its joint caps are converted from
the physical radian radius using the supplied norm stats; gripper dimensions
remain uncapped. Simulation keeps this option disabled by default. To perform a
deliberate no-trust-region real ablation, pass `--midas_use_trust_region 0`.

Operator keys during collection are `1` success, `0` failure, `q` or `2`
abort/discard, and `r` retry/discard. Dense mode uses intermediate success keys
for subtask completion and requires `--num_subtasks`.

## Resume

Resume from the run directory, not an individual checkpoint, and use the same
task/model/actor flags:

```bash
bash scripts/real/yam/train.sh \
  --resume_dir /path/to/midas-real-runs/pickplace_a_run1
```

The latest complete checkpoint/manifest pair is selected. v3 manifests record
the reward schema, actor signature, base-policy identity, norm-stat hash,
query/chunk lengths, server actor version, environment steps, and incremental
replay chains. Incompatible reward, actor, policy, or normalization contracts
are rejected rather than migrated implicitly.

## Read-only evaluation

Stop training and start a fresh server with the saved residual checkpoint:

```bash
export RUN_SPEC=/path/to/run/real_run_spec.json
export RESIDUAL_CHECKPOINT=/path/to/run/checkpoint10000
bash scripts/real/yam/serve.sh
```

Then evaluate from the robot client:

```bash
export RUN_SPEC=/path/to/run/real_run_spec.json
export EVAL_OUTPUT_DIR=/path/to/evaluations/run1
export SERVER_HOST=robot-server.example
export SERVER_PORT=8000
bash scripts/real/yam/evaluate.sh --instruction 'put the green block in the right bin and the blue block in the left bin'
```

`training.evaluation.evaluate_real` does not import a trainer or learner and
has no actor-update method. It writes one HDF5 record (and optionally a top-view
MP4) per attempt plus CSV/JSON summaries. To evaluate the frozen base policy
without MIDAS, start `training.serve_real` manually with `--allow_base_only` and
without `--residual_checkpoint`.

## Lock maintenance

Regenerate the real dependency lock only when `requirements/real.in` changes:

```bash
uv pip compile requirements/real.in -o locks/lock-real.txt \
  --generate-hashes --python-version 3.11 \
  --index-strategy unsafe-best-match --emit-index-url
```
