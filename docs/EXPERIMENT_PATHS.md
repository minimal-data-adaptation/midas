# Experiment path configuration

The launchers under `experiments/` use `MIDAS_REPO_DIR` when set, then the
Slurm submission directory, and finally derive the repository path from their
own location when run directly. Submit from the repository root or export
`MIDAS_REPO_DIR` when submitting elsewhere. Machine- and user-specific storage
paths are supplied through the environment instead of being committed.

Set these variables before submitting a job:

```bash
export MIDAS_DATA_ROOT=/path/to/your/writable/midas-data
export MIDAS_SHARED_DATA_ROOT=/path/to/shared/datasets
export MIDAS_CONDA_SH=/path/to/conda/etc/profile.d/conda.sh
export WANDB_ENTITY=your-wandb-entity
```

`MIDAS_SHARED_DATA_ROOT` defaults to the parent of `MIDAS_DATA_ROOT`. The
default Conda environments are
`$MIDAS_DATA_ROOT/conda_envs/midas-libero` and
`$MIDAS_DATA_ROOT/conda_envs/midas-robocasa`; override them with
`MIDAS_LIBERO_CONDA_ENV` and `MIDAS_ROBOCASA_CONDA_ENV` respectively.

RoboCasa launchers additionally accept:

```bash
export OPENPI_DATASET_ROOT=/path/to/robocasa-assets
export ROBOCASA_EVAL_DATA_ROOT=/path/to/robocasa/evaluation-data
export ROBOCASA_SOURCE_DIR=/path/to/robocasa/source
export MIDAS_ROBOCASA_POLICY_ROOT=/path/to/pi05-robocasa-checkpoints
```

`OPENPI_DATASET_ROOT` and `MIDAS_ROBOCASA_POLICY_ROOT` default beneath
`MIDAS_DATA_ROOT`, while `ROBOCASA_SOURCE_DIR` defaults to the bundled
RoboCasa checkout. `ROBOCASA_EVAL_DATA_ROOT` must be set because its data is
not part of this repository.

Individual evaluation and training paths can still be overridden with the
variables assigned in each launcher, such as `MIDAS_EXP_DIR`, `EVAL_ROOT`,
`CHECKPOINT_DIR`, `BASE_POLICY_CHECKPOINT`, `T_CKPT`, and `DEMO_HDF5`.
