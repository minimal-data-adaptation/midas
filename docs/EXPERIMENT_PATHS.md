# Experiment paths

Run the scripts under `experiments/` with Bash from any directory. They find
the repository from their own location; `MIDAS_REPO_DIR` can override it.

```bash
export MIDAS_DATA_ROOT="/absolute/path/to/writable/storage"
export WANDB_ENTITY="your-wandb-username-or-team"
```

Scripts use `python` from your active environment. Optionally set
`MIDAS_PYTHON_BIN` (or `OPENPI_PYTHON_BIN` for BC), or explicitly choose
`MIDAS_LIBERO_CONDA_ENV` / `MIDAS_ROBOCASA_CONDA_ENV`. Conda activation is
only used when a Conda environment is explicitly selected; `MIDAS_CONDA_SH`
can point to its activation script.

RoboCasa data paths:

```bash
export OPENPI_DATASET_ROOT="/absolute/path/to/robocasa-assets"
export ROBOCASA_EVAL_DATA_ROOT="/absolute/path/to/robocasa-evaluation-data"
```

`OPENPI_DATASET_ROOT` defaults to `$MIDAS_DATA_ROOT/robocasa_assets`.
`ROBOCASA_SOURCE_DIR` defaults to the bundled RoboCasa checkout.

Checkpoint inputs are explicit:
`T_CKPT` for MiDAS training, `BC_CHECKPOINT` for BC evaluation, and
`CHECKPOINT_DIR` plus `BASE_POLICY_CHECKPOINT` for MiDAS evaluation.
LIBERO demonstrations use `DEMO_HDF5`.

Output/cache paths can be overridden with `MIDAS_EXP_DIR`, `EVAL_ROOT`,
`OUTPUT_DIR`, `OPENPI_ASSETS_ROOT`, `OPENPI_CHECKPOINT_ROOT`, and
`OPENPI_DATA_HOME`. Training names are unique by default. Reuse
`OPENPI_BC_EXP_NAME` (BC) or `MIDAS_EXP_NAME_OVERRIDE` (MiDAS) to resume.
`MIDAS_RUN_ID` can provide a deterministic naming suffix.
