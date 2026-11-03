# LIBERO-10 MiDAS

One preempt training launcher per task. Behavior-cloning launchers live in
`../behavior_cloning/libero/` instead.

## Run

```bash
source "${XDG_CONFIG_HOME:-$HOME/.config}/midas/experiments.env"
cd "$MIDAS_REPO_DIR"

export T_CKPT="<absolute-path-to-this-task-BC-checkpoint>/<step>"
sbatch --export=ALL experiments/libero/run_midas_task8_both_mokapots_stove_ep1.slurm
```

`T_CKPT` is required and must contain `params/` and `assets/`. The script does
not guess which task's weights to use. Use the matching OpenPI BC config below.
Set `MIDAS_LIBERO_CONDA_ENV` or `MIDAS_PYTHON_BIN`, `MIDAS_DATA_ROOT`, and
`WANDB_ENTITY` in your environment.

| Task | Launcher suffix / OpenPI config suffix |
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

File: `run_midas_task<id>_<suffix>.slurm`.
Config: `pi05_libero10_<suffix>` (the retained one-demo BC configs).

Demonstrations default to the existing task-specific HDF5 files under
`$MIDAS_DATA_ROOT/expert_hdf5/`; each launcher shows its filename. Task 8 keeps
the original episode-86 demo. Override `DEMO_HDF5` if needed, and make sure it
matches the demonstration used for your chosen BC checkpoint:

```bash
T_CKPT="<absolute-BC-checkpoint>" DEMO_HDF5="<absolute-demo.hdf5>" MIDAS_SEED=11 \
  sbatch --export=ALL experiments/libero/run_midas_task0_alphabet_soup_tomato_sauce_basket_ep1.slurm
```

## Settings and outputs

`run_midas.sh` retains the original task-8 `paper_v1` hyperparameters for all
tasks: seed 0, 250,000 steps, batch size 64, evaluation every 25,000 steps
(50 episodes), checkpoints every 10,000 steps, and the same MiDAS/BC warmup settings.
Only task ID, BC config/checkpoint, demo, and output name change by task.

```text
$MIDAS_DATA_ROOT/midas_exps/libero10/pi05_base_v1/<experiment>/
```

Override the parent with `MIDAS_EXP_DIR`. Experiment names include the task,
seed, and job ID, so new submissions do not overwrite the old task-8 runs.
Requeues keep their run name and resume saved training state; preemption before
the first checkpoint restarts training. For a separate submission to resume
a run, set `MIDAS_EXP_NAME_OVERRIDE` to that run's directory name (and keep the
same seed, checkpoint, demo, and output parent).

Slurm logs go in the submission directory as `<job-name>_<job-id>.out/.err`.

## Archived scripts

The previous task-8-only seed launchers, perturbation evaluations, and evaluation
watchdog are in `legacy/`. Their saved-run settings are unchanged; only internal
script paths were updated. The existing global watchdog still targets those
legacy runs, not the ten new launchers. No checkpoints or rollouts were removed.
