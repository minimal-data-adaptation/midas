# Behavior cloning

Scripts use the 10 one-demo LIBERO-10 configs and 3 standalone RoboCasa BC
configs in `openpi/src/openpi/training/config.py`.

## Setup

From the repository root, set the paths shown in the main README:

```bash
export MIDAS_REPO_DIR="$PWD"
export MIDAS_DATA_ROOT="/absolute/path/to/writable/storage"
export WANDB_ENTITY="your-wandb-username-or-team"
```

Activate the matching Python environment before running. Scripts use its
`python` by default; `OPENPI_PYTHON_BIN` can select a different interpreter.
Use `OPENPI_BC_WANDB_ENABLED=0` to disable W&B.

By default, assets and checkpoints go under `$MIDAS_DATA_ROOT/openpi_bc_assets`
and `$MIDAS_DATA_ROOT/openpi_bc_checkpoints`. Override with absolute
`OPENPI_ASSETS_ROOT` and `OPENPI_CHECKPOINT_ROOT` paths if desired.

RoboCasa needs `OPENPI_DATASET_ROOT` (default:
`$MIDAS_DATA_ROOT/robocasa_assets`), containing `robocasa/v1.0/...`.
LIBERO loads `physical-intelligence/libero` through the Hugging Face cache.

## LIBERO

```bash
conda activate midas-libero
bash experiments/behavior_cloning/libero/run_bc_both_mokapots_stove_ep1.sh
```

Or pass a config directly:

```bash
bash experiments/behavior_cloning/libero/run_bc.sh pi05_libero10_both_mokapots_stove_ep1
```

Each task has `libero/run_bc_<task>_ep1.sh` and config
`pi05_libero10_<task>_ep1`:

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

Defaults: seed 42, batch size 32, 50,000 steps. The moka-pot config pins episode 86;
the mug/chocolate-pudding config pins episode 11. The other configs select the
first matching demonstration.

## RoboCasa

Choose one task:

```bash
conda activate midas-robocasa
bash experiments/behavior_cloning/robocasa/run_bc_pick_place_counter_to_cabinet_exact_replay_l1_s1_ep32.sh
bash experiments/behavior_cloning/robocasa/run_bc_pick_place_fridge_drawer_to_shelf_l50_s37.sh
bash experiments/behavior_cloning/robocasa/run_bc_prepare_coffee_l25_s29.sh
```

Defaults: seed 0, batch size 64, 100,000 steps. Instructions come from the
demonstrations; the runner checks the dataset/prompt trace before training.

## Outputs and overrides

Missing normalization statistics are computed on CPU before GPU training.
Hyperparameters stay at their config values unless overridden. Training uses
the visible GPUs; select them with `CUDA_VISIBLE_DEVICES` when needed.

```text
LIBERO:   $OPENPI_CHECKPOINT_ROOT/pi05_checkpoints_libero10/<config>/<experiment>/<step>/
RoboCasa: $OPENPI_CHECKPOINT_ROOT/robocasa_bc/<config>/<experiment>/<step>/
```

Default experiment names include the seed and a timestamp/PID. New invocations
use separate directories. Set `OPENPI_BC_EXP_NAME` to reuse and resume an
existing run. Output is printed to the terminal; redirect it to a file if needed.

```bash
OPENPI_BC_SEED=11 OPENPI_BC_EXP_NAME=both_mokapots_seed11 \
OPENPI_BC_NUM_TRAIN_STEPS=16000 CUDA_VISIBLE_DEVICES=0,1,2,3 \
  bash experiments/behavior_cloning/libero/run_bc_both_mokapots_stove_ep1.sh
```

Also supported: `OPENPI_BC_BATCH_SIZE`, `OPENPI_BC_NUM_WORKERS`,
`OPENPI_BC_SAVE_INTERVAL`, and `OPENPI_BC_FSDP_DEVICES` (default 1).
The global batch size must be divisible by the visible GPU count.
