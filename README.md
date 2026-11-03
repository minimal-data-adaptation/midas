# MiDAS: Minimal-Data Adaptation for Generalist Robot Policies

[Paper](https://arxiv.org/abs/2608.11363) · [PDF](https://arxiv.org/pdf/2608.11363)

Shreyas Kowshik, Sreyas Venkataraman, Leo Wang, Niharika Pant, Max Simchowitz, and Aviral Kumar

MiDAS adapts a frozen vision-language-action policy to a new task from one
demonstration and autonomous online interaction. It combines behavior cloning
with value-based residual reinforcement learning over frozen VLA features.

![MiDAS overview](main_fig_midas.jpg)

## Install

```bash
git clone --recurse-submodules https://github.com/minimal-data-adaptation/midas.git
cd midas
git submodule update --init --recursive
```

### LIBERO

```bash
conda env create -f environment-libero.yml
conda activate midas-libero
bash scripts/install_profile.sh libero
python -m pip install -r requirements/jax-cuda.in

export LIBERO_CONFIG_PATH="$HOME/.config/libero-pro"
mkdir -p "$LIBERO_CONFIG_PATH"
bash scripts/smoke_cpu.sh
```

### RoboCasa

```bash
conda env create -f environment-robocasa.yml
conda activate midas-robocasa
bash scripts/install_profile.sh robocasa
python -m pip install -r requirements/jax-cuda.in
python robocasa/robocasa/scripts/download_kitchen_assets.py --type all
bash scripts/smoke_cpu.sh
```

## Configure

Create `${XDG_CONFIG_HOME:-$HOME/.config}/midas/experiments.env`:

```bash
export MIDAS_REPO_DIR="<absolute-path-to-midas>"
export MIDAS_DATA_ROOT="<absolute-path-to-writable-storage>"


export ROBOCASA_EVAL_DATA_ROOT="<absolute-path-to-robocasa-evaluation-data>"
export OPENPI_DATASET_ROOT="$MIDAS_DATA_ROOT/robocasa_assets"
export OPENPI_ASSETS_ROOT="$MIDAS_DATA_ROOT/openpi_bc_assets"
export OPENPI_CHECKPOINT_ROOT="$MIDAS_DATA_ROOT/openpi_bc_checkpoints"
export OPENPI_DATA_HOME="$OPENPI_DATASET_ROOT/openpi_cache"

export WANDB_ENTITY="<wandb-entity>"
export MIDAS_WANDB_ENTITY="$WANDB_ENTITY"
```

```bash
source "${XDG_CONFIG_HOME:-$HOME/.config}/midas/experiments.env"
cd "$MIDAS_REPO_DIR"
```

### Prepare Libero Demos

Download the dataset:

```bash
conda activate midas-libero
export HF_LEROBOT_HOME="$MIDAS_DATA_ROOT/lerobot"
export LIBERO_LEROBOT_ROOT="$HF_LEROBOT_HOME/physical-intelligence/libero"
hf download physical-intelligence/libero --repo-type dataset --revision v2.0 --local-dir "$LIBERO_LEROBOT_ROOT"
```

Create the manifest:

```bash
CONFIG=pi05_libero10_both_mokapots_stove_ep1
python scripts/create_libero_demo_manifest.py \
  --config-name "$CONFIG" \
  --output "$MIDAS_DATA_ROOT/manifests/$CONFIG.json" \
  --filename libero10_task8_put_both_moka_pots_on_the_stove_ep86.hdf5
```

Export the HDF5:

```bash
python scripts/prepare_libero_demos.py \
  --manifest "$MIDAS_DATA_ROOT/manifests/$CONFIG.json" \
  --dataset-root "$LIBERO_LEROBOT_ROOT" \
  --output-dir "$MIDAS_DATA_ROOT/expert_hdf5"
```

## Train

### LIBERO behavior cloning

Run from the repository root after the Configure step. Missing normalization
statistics are computed automatically. Scripts use your active Python environment.
Select GPUs with `CUDA_VISIBLE_DEVICES` if needed (for example, `0,1,2,3`).

Run both moka pots:

```bash
conda activate midas-libero
bash experiments/behavior_cloning/libero/run_bc.sh pi05_libero10_both_mokapots_stove_ep1
```

Or choose another task (one command per task):

```bash
bash experiments/behavior_cloning/libero/run_bc_alphabet_soup_tomato_sauce_basket_ep1.sh
bash experiments/behavior_cloning/libero/run_bc_cream_cheese_butter_basket_ep1.sh
bash experiments/behavior_cloning/libero/run_bc_stove_mokapot_ep1.sh
bash experiments/behavior_cloning/libero/run_bc_black_bowl_bottom_drawer_ep1.sh
bash experiments/behavior_cloning/libero/run_bc_white_mug_plates_ep1.sh
bash experiments/behavior_cloning/libero/run_bc_book_caddy_ep1.sh
bash experiments/behavior_cloning/libero/run_bc_white_mug_chocolate_pudding_ep1.sh
bash experiments/behavior_cloning/libero/run_bc_alphabet_soup_cream_cheese_basket_ep1.sh
bash experiments/behavior_cloning/libero/run_bc_both_mokapots_stove_ep1.sh
bash experiments/behavior_cloning/libero/run_bc_yellow_white_mug_microwave_ep1.sh
```

Checkpoints: `$OPENPI_CHECKPOINT_ROOT/pi05_checkpoints_libero10/<config>/<experiment>/<step>/`.
More options: [BC scripts](experiments/behavior_cloning/README.md).

### LIBERO MiDAS

Use the named BC run above once training finishes:

```bash
cd "$MIDAS_REPO_DIR"
CONFIG=pi05_libero10_both_mokapots_stove_ep1
EXP_NAME=both_mokapots_ep1_bc
# Training step indices start at zero: a 16,000-step run ends at 15999.
BC_CHECKPOINT="$OPENPI_CHECKPOINT_ROOT/pi05_checkpoints_libero10/$CONFIG/$EXP_NAME/15999"
T_CKPT="$BC_CHECKPOINT" bash \
  experiments/libero/run_midas_task8_both_mokapots_stove_ep1.sh
```

One launcher per task: [experiments/libero](experiments/libero/README.md).
Set `T_CKPT` to the matching task's BC checkpoint.

```bash
bash experiments/libero/run_midas_task8_both_mokapots_stove_ep1.sh
MIDAS_SEED=11 bash experiments/libero/run_midas_task8_both_mokapots_stove_ep1.sh
MIDAS_SEED=42 bash experiments/libero/run_midas_task8_both_mokapots_stove_ep1.sh
```

### RoboCasa behavior cloning

Choose a task:

```bash
conda activate midas-robocasa
bash experiments/behavior_cloning/robocasa/run_bc.sh pi05_robocasa_bc_counter_to_cabinet_l1_s1_ep32
bash experiments/behavior_cloning/robocasa/run_bc.sh pi05_robocasa_bc_fridge_drawer_to_shelf_l50_s37
bash experiments/behavior_cloning/robocasa/run_bc.sh pi05_robocasa_bc_prepare_coffee_l25_s29
```

Or use the task-specific scripts:

```bash
bash experiments/behavior_cloning/robocasa/run_bc_pick_place_counter_to_cabinet_exact_replay_l1_s1_ep32.sh
bash experiments/behavior_cloning/robocasa/run_bc_pick_place_fridge_drawer_to_shelf_l50_s37.sh
bash experiments/behavior_cloning/robocasa/run_bc_prepare_coffee_l25_s29.sh
```

Defaults: 100,000 steps, batch size 64, seed 0. Normalization is computed if
missing, and the dataset/prompt trace is checked before training.

Checkpoints: `$OPENPI_CHECKPOINT_ROOT/robocasa_bc/<config>/<experiment>/<step>/`.
Run names are generated automatically; each new run gets a separate directory.

### RoboCasa MiDAS

Set `T_CKPT` to the matching task's frozen BC checkpoint before each command.

```bash
conda activate midas-robocasa
bash experiments/robocasa/run_midas_pick_place_counter_to_cabinet_exact_replay_l1_s1_ep32_paper_v2.sh
bash experiments/robocasa/run_midas_pick_place_fridge_drawer_to_shelf_l50_s37_paper_v2.sh
bash experiments/robocasa/run_midas_prepare_coffee_l25_s29_var1_paper_v2.sh
```

## Evaluate

### LIBERO

```bash
conda activate midas-libero
export CHECKPOINT_DIR="<absolute-path-to-MiDAS-checkpoint>"
export BASE_POLICY_CHECKPOINT="<absolute-path-to-frozen-BC-checkpoint>"
bash experiments/libero/legacy/eval_midas_task8_both_mokapots_checkpoint250000_no_perturbations.sh
bash experiments/libero/legacy/eval_midas_task8_both_mokapots_checkpoint250000_object_perturbation.sh
bash experiments/libero/legacy/eval_midas_task8_both_mokapots_checkpoint250000_language_perturbation.sh
bash experiments/libero/legacy/eval_midas_task8_both_mokapots_checkpoint250000_position_5cm_perturbation.sh
CHECKPOINT_DIR_S11="<seed-11-MiDAS-checkpoint>" CHECKPOINT_DIR_S42="<seed-42-MiDAS-checkpoint>" \
  bash experiments/libero/legacy/run_midas_task8_checkpoint200000_eval_matrix.sh
```

### RoboCasa behavior cloning

```bash
conda activate midas-robocasa
export BC_CHECKPOINT="<absolute-path-to-BC-checkpoint>"
bash experiments/evals/behavior_cloning/eval_bc_pick_place_counter_to_cabinet_checkpoint8000_50rollouts.sh
```

## Test

```bash
conda activate midas-libero
bash scripts/smoke_cpu.sh
```

```bash
conda activate midas-robocasa
bash scripts/smoke_cpu.sh
```

## Citation

```bibtex
@misc{kowshik2026midas,
  title         = {What Does It Take For A Generalist Robot Policy To Adapt From Minimal Data?},
  author        = {Kowshik, Shreyas and Venkataraman, Sreyas and Wang, Leo and Pant, Niharika and Simchowitz, Max and Kumar, Aviral},
  year          = {2026},
  eprint        = {2608.11363},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2608.11363}
}
```

## License

MIT. See [LICENSE](LICENSE).
