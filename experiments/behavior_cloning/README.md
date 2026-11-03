# Behavior cloning

Uses the 10 one-demo LIBERO-10 configs and 3 RoboCasa BC configs in
`openpi/src/openpi/training/config.py`. No MiDAS residual training is run.

## Setup

Install the matching environment using the repository README, then:

```bash
source "${XDG_CONFIG_HOME:-$HOME/.config}/midas/experiments.env"
cd "$MIDAS_REPO_DIR"
```

Set absolute `OPENPI_ASSETS_ROOT` and `OPENPI_CHECKPOINT_ROOT` paths, or set
`MIDAS_DATA_ROOT` to use its `openpi_bc_assets` and `openpi_bc_checkpoints`
subdirectories. RoboCasa also needs `OPENPI_DATASET_ROOT` (defaults to
`$MIDAS_DATA_ROOT/robocasa_assets`), containing the `robocasa/v1.0/...` datasets.
LIBERO loads `physical-intelligence/libero` through the Hugging Face cache
(`HF_LEROBOT_HOME`); it does not need local RoboCasa data.

Set `MIDAS_LIBERO_CONDA_ENV=midas-libero` and
`MIDAS_ROBOCASA_CONDA_ENV=midas-robocasa`, or set `OPENPI_PYTHON_BIN` to the
absolute path of the matching environment's Python. Set `WANDB_ENTITY` and run
`wandb login`, or use `OPENPI_BC_WANDB_ENABLED=0`.

## LIBERO

Run the both-moka-pots task directly on an allocated GPU:

```bash
bash experiments/behavior_cloning/libero/run_bc.sh pi05_libero10_both_mokapots_stove_ep1
```

Or submit to preempt (one GPU by default; this example requests four):

```bash
sbatch --gres=gpu:4 --export=ALL experiments/behavior_cloning/libero/run_bc_both_mokapots_stove_ep1.slurm
```

Each task has a launcher under `libero/`. Pass `pi05_libero10_<task>_ep1` to
`libero/run_bc.sh`, or submit `libero/run_bc_<task>_ep1.slurm`:

| Task | `<task>` |
| --- | --- |
| 0 | `alphabet_soup_tomato_sauce_basket` |
| 1 | `cream_cheese_butter_basket` |
| 2 | `stove_mokapot` |
| 3 | `black_bowl_bottom_drawer` |
| 4 | `white_mug_plates` |
| 5 | `book_caddy` |
| 6 | `white_mug_chocolate_pudding` |
| 7 | `alphabet_soup_cream_cheese_basket` |
| 8 | `both_mokapots_stove` |
| 9 | `yellow_white_mug_microwave` |

Defaults: seed 42, batch size 32, 50,000 training steps, zero loader workers.
These configs select one demo using the task filter; they do not pin episode 86.

## RoboCasa

Run directly on an allocated GPU:

```bash
bash experiments/behavior_cloning/robocasa/run_bc.sh pi05_robocasa_bc_counter_to_cabinet_l1_s1_ep32
bash experiments/behavior_cloning/robocasa/run_bc.sh pi05_robocasa_bc_fridge_drawer_to_shelf_l50_s37
bash experiments/behavior_cloning/robocasa/run_bc.sh pi05_robocasa_bc_prepare_coffee_l25_s29
```

Or submit separate preempt jobs:

```bash
sbatch --gres=gpu:4 --export=ALL experiments/behavior_cloning/robocasa/run_bc_pick_place_counter_to_cabinet_exact_replay_l1_s1_ep32.slurm
sbatch --gres=gpu:4 --export=ALL experiments/behavior_cloning/robocasa/run_bc_pick_place_fridge_drawer_to_shelf_l50_s37.slurm
sbatch --gres=gpu:4 --export=ALL experiments/behavior_cloning/robocasa/run_bc_prepare_coffee_l25_s29.slurm
```

Defaults: seed 0, batch size 64, 100,000 training steps, four loader workers.
The runner checks the dataset/prompt trace before training. Instructions come
from the demonstrations; none are supplied as launcher overrides.

## Run names and outputs

Missing normalization statistics are computed on CPU before GPU training.
All hyperparameters stay at their config values unless explicitly overridden.
Slurm `.out`/`.err` logs go in the submission directory.

```text
LIBERO:   $OPENPI_CHECKPOINT_ROOT/pi05_checkpoints_libero10/<config>/<experiment>/<step>/
RoboCasa: $OPENPI_CHECKPOINT_ROOT/robocasa_bc/<config>/<experiment>/<step>/
```

Run names include the seed and Slurm job ID (timestamp/PID for direct runs).
New submissions use new directories. Requeues resume the same directory;
existing checkpoints are never overwritten.

Optional overrides:

```bash
export OPENPI_BC_SEED=11
export OPENPI_BC_EXP_NAME=both_mokapots_seed11  # Reuse this name only to resume this run.
export OPENPI_BC_NUM_TRAIN_STEPS=16000
sbatch --gres=gpu:4 --export=ALL experiments/behavior_cloning/libero/run_bc_both_mokapots_stove_ep1.slurm
```

Also supported: `OPENPI_BC_BATCH_SIZE`, `OPENPI_BC_NUM_WORKERS`,
`OPENPI_BC_SAVE_INTERVAL`, and `OPENPI_BC_FSDP_DEVICES` (default 1; data parallel
across the allocated GPUs). Global batch size must divide evenly across GPUs;
FSDP devices must divide the allocated GPU count.
