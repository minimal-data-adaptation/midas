# LIBERO-10 MiDAS

One portable training script per task. BC scripts are in
`../behavior_cloning/libero/`.

## Run

From the repository root, after setting the paths in the main README:

Prepare the required HDF5 demos first; see [demo preparation](../../docs/DEMO_PREPARATION.md).

```bash
conda activate midas-libero
export T_CKPT="/absolute/path/to/this-task-BC-checkpoint/<step>"
bash experiments/libero/run_midas_task8_both_mokapots_stove_ep1.sh
```

`T_CKPT` must contain `params/` and `assets/`. Scripts use the active Python
environment. Set `MIDAS_PYTHON_BIN` to choose another interpreter.

| Task | Script / OpenPI config suffix |
| --- | --- |
| 0 | `alphabet_soup_tomato_sauce_basket_ep1` |
| 1 | `cream_cheese_butter_basket_ep1` |
| 2 | `stove_mokapot_ep1` |
| 3 | `black_bowl_bottom_drawer_ep1` |
| 4 | `white_mug_plates_ep1` |
| 5 | `book_caddy_ep1` |
| 6 | `white_mug_chocolate_pudding_ep1` |
| 7 | `alphabet_soup_cream_cheese_basket_ep1` |
| 8 | `both_mokapots_stove_ep1` |
| 9 | `yellow_white_mug_microwave_ep1` |

Script: `run_midas_task<id>_<suffix>.sh`.
Config: `pi05_libero10_<suffix>`.

Demonstrations default to the task-specific HDF5 filenames shown in each
script, under `$MIDAS_DATA_ROOT/expert_hdf5/`. Set `DEMO_HDF5` to match the
demo used for your BC checkpoint:

```bash
T_CKPT="/absolute/path/to/BC-checkpoint" DEMO_HDF5="/absolute/path/to/demo.hdf5" MIDAS_SEED=11 \
  bash experiments/libero/run_midas_task0_alphabet_soup_tomato_sauce_basket_ep1.sh
```

## Settings and outputs

`run_midas.sh` uses seed 0, 250,000
steps, batch size 64, 50-episode evaluation every 25,000 steps, checkpoints
every 10,000 steps, and MiDAS/BC warmup.

```text
$MIDAS_DATA_ROOT/midas_exps/libero10/pi05_base_v1/<experiment>/
```

Experiment names include the task, seed, and timestamp/PID. Override the parent
with `MIDAS_EXP_DIR`. To resume a saved run, set `MIDAS_EXP_NAME_OVERRIDE` to
its directory name and keep the same seed, BC checkpoint, demo, and output parent.
Output is printed to the terminal.
