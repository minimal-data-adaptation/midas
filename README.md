# MIDAS

MIDAS is residual policy-agnostic reinforcement learning for adapting a frozen
vision-language-action policy. The frozen Pi-0.5 policy proposes action chunks;
MIDAS learns bounded residual actions from online interaction. This release
supports LIBERO and RoboCasa simulation.

Real-world/YAM training, other RL algorithms, plotting utilities, and
research-era checkpoints are intentionally out of scope. Checkpoints created by
the pre-release research package are not compatible with the renamed MIDAS state.

## Environment setup

Prerequisites are Linux, Git, Git LFS, Conda, a working MuJoCo graphics setup,
and (for GPU runs) an NVIDIA driver compatible with CUDA 12. Python is fixed at
3.11.11. Clone with all three dependency forks:

```bash
git clone --recurse-submodules https://github.com/shreyas-kowshik/midas.git
cd midas
git submodule update --init --recursive
```

Create one simulator-specific environment. The lock installs every Python
dependency; editable submodules are installed without dependency resolution so
their conflicting metadata cannot override the lock.

LIBERO:

```bash
conda env create -f environment-libero.yml
conda activate midas-libero
export LIBERO_CONFIG_PATH="$HOME/.config/libero-pro"
mkdir -p "$LIBERO_CONFIG_PATH"
bash scripts/install_profile.sh libero
python -m pip check  # the omitted real-world extra is documented in docs/SETUP.md
bash scripts/smoke_cpu.sh
```

RoboCasa:

```bash
conda env create -f environment-robocasa.yml
conda activate midas-robocasa
bash scripts/install_profile.sh robocasa
# Download all kitchen assets (~10 GB, extracted into
# robocasa/robocasa/models/assets; the script asks for a y/n confirmation).
# Required before running RoboCasa simulations, but not for the smoke tests.
python robocasa/robocasa/scripts/download_kitchen_assets.py --type all
python -m pip check  # see the documented fork-metadata exceptions below
bash scripts/smoke_cpu.sh
```

For an NVIDIA GPU, install the CUDA JAX overlay after the profile lock:

```bash
python -m pip install -r requirements/jax-cuda.in
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

See [docs/SETUP.md](docs/SETUP.md) for system packages, environment variables,
CPU CI setup, lock regeneration, and troubleshooting.

## Running

The launcher accepts only `--algo midas`. A LIBERO run looks like:

```bash
python -m training.launch_train_sim \
  --env libero \
  --task_suite_name libero_10 \
  --task_id 8 \
  --pi_05_config <openpi-train-config> \
  --pi_05_ckpt_dir <pi-checkpoint> \
  --max_steps 1000000
```

For RoboCasa:

```bash
python -m training.launch_train_sim \
  --env robocasa \
  --robocasa_env_name <task-or-single-task-set> \
  --pi_05_config <openpi-train-config> \
  --pi_05_ckpt_dir <pi-checkpoint> \
  --max_steps 1000000
```

Add `--wandb 1` to opt into W&B logging. Output defaults to `experiments/` and
can be redirected with `MIDAS_EXP_DIR`.

Evaluate a saved checkpoint with the same architecture and policy flags used
for training. The evaluator accepts either a run directory (restores its latest
checkpoint) or a specific `checkpoint<step>` directory, and writes per-rollout
metrics plus MP4s to `--output_dir`:

```bash
python -m training.evaluation.evaluate_sim \
  --env libero \
  --checkpoint_dir <run-or-checkpoint-directory> \
  --output_dir <evaluation-output-directory> \
  --num_evals 50 \
  --task_suite_name libero_10 --task_id 8 \
  --pi_05_config <openpi-train-config> \
  --pi_05_ckpt_dir <pi-checkpoint> \
  --resize_image 100 --hidden_dims 512 \
  --query_freq 10 --chunk_len 10 \
  --residual_alpha 0.5 --predict_a_exec 1 \
  --use_vlm_embedding 1
```

For object, language, spatial-swap, environment, or task perturbations, either
pass a previously generated `--suite_manifest`, or generate it in the evaluator
with `--use_object 1`, `--use_language 1`, `--use_swap 1`,
`--use_environment 1`, or `--use_task 1`. Defaults come from
`LIBERO-PRO/evaluation_config.yaml`; individual YAMLs can be overridden with,
for example, `--object_config <yaml>`. The task transform cannot be combined
with another BDDL transform.

Position and yaw perturbations are applied after reset and can be used alone or
combined with a BDDL suite:

```bash
python -m training.evaluation.evaluate_sim \
  <the-same-checkpoint-and-model-flags-as-above> \
  --pos_perturb_radius 0.05 \
  --pos_perturb_objects auto
```

`auto` currently covers the known LIBERO tasks listed in
`training/evaluation/position_perturbation.py`; otherwise pass exact BDDL object
names such as `--pos_perturb_objects moka_pot_1,moka_pot_2`. See
[`docs/LIBERO.md`](docs/LIBERO.md) for perturbation examples and reset semantics.

Actor-target trust-region clipping is disabled by default for simulation. It
can be enabled for real-world-style training or a controlled ablation by
supplying both the flag and one normalized-action radius per action dimension:

```bash
--midas_use_trust_region 1 \
--midas_a_star_delta_clip_norm 0.05 0.05 0.05 0.05 0.05 0.05 1000000
```

The final large value in this seven-dimensional example leaves the gripper
effectively uncapped. A clip norm supplied while the flag is `0` is ignored.

The zero-base-policy CartPole environment is a fast integration check and does
not represent benchmark performance:

```bash
python -m training.launch_train_sim --env cartpole --max_steps 1 \
  --batch_size 2 --resize_image 16 --chunk_len 2 --query_freq 2 \
  --cartpole_horizon 4 --start_online_updates 1 --eval_episodes 1 \
  --eval_interval 1000 --hidden_dims 16 16 --cnn_features 8 8 \
  --cnn_strides 2 2 --cnn_padding SAME --latent_dim 8 --num_qs 2 \
  --midas_num_samples 2 --midas_num_elites 1 --midas_num_grad_steps 1 \
  --num_critic_updates 1 --num_actor_updates 1 --color_jitter 0
```

Simulator details are in [docs/LIBERO.md](docs/LIBERO.md) and
[docs/ROBOCASA.md](docs/ROBOCASA.md). Public artifact records and hashes live in
`midas/artifacts_index.json`; no trained artifacts are bundled in v0.1.0.

## License and attribution

Owned code is released under the [MIT License](LICENSE). The code has lineage
from JAXRL2/PTR and uses separately licensed submodules; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Release publication remains
subject to the rights-holder and dataset sign-offs described in
[docs/RELEASING.md](docs/RELEASING.md).
