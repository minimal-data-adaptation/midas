# MIDAS

MIDAS is residual policy-agnostic reinforcement learning for adapting a frozen
vision-language-action policy. The frozen Pi-0.5 policy proposes action chunks;
MIDAS learns bounded residual actions from online interaction. This release
supports LIBERO and RoboCasa simulation plus a separately isolated YAM
bimanual real-world workflow.

Other RL algorithms, plotting utilities, and research-era checkpoints are
intentionally out of scope. Checkpoints created by the pre-release research
package are not compatible with the renamed MIDAS state.

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

YAM real-world training (the robot driver is supplied separately):

```bash
conda env create -f environment-real.yml
conda activate midas-real
bash scripts/install_profile.sh real
# Install the site-specific yam_teleop package into this environment.
python -m pip check  # the unrelated gym-aloha metadata exception is documented below
PYTHON_BIN=python bash scripts/real/yam/smoke_mock.sh
```

For an NVIDIA GPU, install the CUDA JAX overlay after the profile lock:

```bash
python -m pip install -r requirements/jax-cuda.in
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
```

### Configure machine-specific experiment paths

Do not edit the checked-in launchers to insert usernames, checkout locations,
or cluster storage paths. Keep those values in a per-user shell file outside
the repository and source it before calling `sbatch`. For example:

```bash
mkdir -p "${XDG_CONFIG_HOME:-$HOME/.config}/midas"
${EDITOR:-vi} "${XDG_CONFIG_HOME:-$HOME/.config}/midas/experiments.env"
```

Put the following template in `experiments.env`, replacing every value in
angle brackets with a path or account name appropriate for your machine:

```bash
# Core settings. MIDAS_DATA_ROOT is required. MIDAS_REPO_DIR is strongly
# recommended for Slurm, and WANDB_ENTITY is required by training launchers.
export MIDAS_REPO_DIR="<absolute-path-to-this-midas-checkout>"
export MIDAS_DATA_ROOT="<absolute-path-to-your-writable-midas-data>"
# Required on compute nodes where Conda is not already initialized.
export MIDAS_CONDA_SH="<absolute-path-to-conda>/etc/profile.d/conda.sh"
export WANDB_ENTITY="<your-wandb-entity>"

# Shared, read-only datasets and base checkpoints. If omitted, this defaults
# to the parent directory of MIDAS_DATA_ROOT.
export MIDAS_SHARED_DATA_ROOT="<absolute-path-to-shared-datasets>"

# Required by the checked-in RoboCasa jobs because evaluation reset data is
# not distributed in this repository.
export ROBOCASA_EVAL_DATA_ROOT="<absolute-path-to-robocasa-evaluation-data>"

# Optional overrides; these defaults are described below.
export MIDAS_LIBERO_CONDA_ENV="<conda-env-name-or-absolute-prefix>"
export MIDAS_ROBOCASA_CONDA_ENV="<conda-env-name-or-absolute-prefix>"
export OPENPI_DATASET_ROOT="<absolute-path-to-robocasa-assets>"
export MIDAS_ROBOCASA_POLICY_ROOT="<absolute-path-to-pi05-robocasa-checkpoints>"
export ROBOCASA_SOURCE_DIR="<absolute-path-to-robocasa-source>"
```

The defaults assume this layout beneath `MIDAS_DATA_ROOT`:

```text
<MIDAS_DATA_ROOT>/
├── conda_envs/
│   ├── midas-libero/
│   └── midas-robocasa/
├── expert_hdf5/
├── midas_evals/
├── midas_exps/
├── pi05_robocasa/
└── robocasa_assets/
```

`MIDAS_SHARED_DATA_ROOT` is expected to contain shared resources such as
`pi05_common/`. `ROBOCASA_SOURCE_DIR` defaults to the bundled `robocasa/`
checkout. You may omit an optional variable when that default matches your
layout.

Load the settings and verify the important locations before submitting:

```bash
source "${XDG_CONFIG_HOME:-$HOME/.config}/midas/experiments.env"

test -f "$MIDAS_REPO_DIR/scripts/experiment_env.sh"
test -f "$MIDAS_CONDA_SH"
test -d "$MIDAS_DATA_ROOT"
test -d "$MIDAS_SHARED_DATA_ROOT"

cd "$MIDAS_REPO_DIR"
sbatch --export=ALL experiments/libero/run_midas_task8_both_mokapots_paper_v1.slurm
```

Submitting from the repository root is recommended. `MIDAS_REPO_DIR` makes
the checkout location unambiguous when Slurm executes its spooled copy of a
script, and `--export=ALL` ensures the configured values reach the job and any
successor jobs submitted by a watchdog. Slurm output and error files use
`%x_%j.out` and `%x_%j.err` in the submission directory.

The most useful per-run overrides are `MIDAS_EXP_DIR`, `CHECKPOINT_DIR`,
`BASE_POLICY_CHECKPOINT`, `EVAL_ROOT`, `T_CKPT`, and `DEMO_HDF5`. Set one only
when a particular run does not follow the directory layout above; no launcher
source edit is necessary. See
[docs/EXPERIMENT_PATHS.md](docs/EXPERIMENT_PATHS.md) for the complete path
configuration reference.

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

### Seed and resume reproducibility

`--seed` controls the learner, replay samplers, simulator/reset streams, and
the frozen OpenPI policy (including its private PyTorch generator). Periodic
evaluation uses a separate simulator stream and restores shared policy/global
RNG state, so changing evaluation cadence does not change later training.

Simulation v3 and real-world v4 manifests continue all software RNG streams
after preemption. Older checkpoints remain loadable, but because they did not
record those streams their resumed trajectory cannot exactly match an
uninterrupted run. Bitwise equality still requires the same accelerator,
driver, JAX/XLA stack, and deterministic kernels; physical robot rollouts are
not expected to be bitwise repeatable.

Simulator details are in [docs/LIBERO.md](docs/LIBERO.md) and
[docs/ROBOCASA.md](docs/ROBOCASA.md). Real-world setup, safety gates, task
profiles, launch ordering, resume, and evaluation are in
[docs/REAL_WORLD.md](docs/REAL_WORLD.md). Public artifact records and hashes live in
`midas/artifacts_index.json`; no trained artifacts are bundled in v0.1.0.

## License and attribution

Owned code is released under the [MIT License](LICENSE). The code has lineage
from JAXRL2/PTR and uses separately licensed submodules; see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md). Release publication remains
subject to the rights-holder and dataset sign-offs described in
[docs/RELEASING.md](docs/RELEASING.md).
