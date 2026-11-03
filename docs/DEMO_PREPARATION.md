# Prepare demos

Run these commands from the repository root after installation and path setup.

## LIBERO

Download the dataset and prepare demos:

```bash
conda activate midas-libero
export MIDAS_DATA_ROOT="/absolute/path/to/writable/storage"
export HF_LEROBOT_HOME="$MIDAS_DATA_ROOT/lerobot"
export LIBERO_LEROBOT_ROOT="$HF_LEROBOT_HOME/physical-intelligence/libero"
hf download physical-intelligence/libero --repo-type dataset --revision v2.0 --local-dir "$LIBERO_LEROBOT_ROOT"
bash experiments/libero/prepare_demos.sh
```

Outputs go to `$MIDAS_DATA_ROOT/expert_hdf5/`. Existing files are not overwritten.
Use demo episodes that match your BC checkpoint's training data.

List available selections:

```bash
python scripts/prepare_libero_demos.py --list
```

Export the episodes selected by your BC config (run in the same environment and
with the same `HF_LEROBOT_HOME` as training):

```bash
CONFIG=pi05_libero10_both_mokapots_stove_ep1
python scripts/create_libero_demo_manifest.py \
  --config-name "$CONFIG" \
  --output "$MIDAS_DATA_ROOT/manifests/$CONFIG.json" \
  --filename libero10_task8_put_both_moka_pots_on_the_stove_ep86.hdf5
python scripts/prepare_libero_demos.py \
  --manifest "$MIDAS_DATA_ROOT/manifests/$CONFIG.json" \
  --dataset-root "$LIBERO_LEROBOT_ROOT" \
  --output-dir "$MIDAS_DATA_ROOT/expert_hdf5_custom"
```

Keep the manifest with your BC run. It records the training loader's selected
episodes, instruction, frame counts, and source checksums; export rejects changed
sources. It resolves the **current** config, so it cannot establish which data an
older checkpoint used. It records eligible episodes, not sampled minibatch history.

Validate the default demo files:

```bash
python scripts/prepare_libero_demos.py --verify-dir "$MIDAS_DATA_ROOT/expert_hdf5"
```

## RoboCasa

Use the LeRobot datasets described in the [dataset download guide](../robocasa/docs/datasets/using_datasets.md).

```bash
export OPENPI_DATASET_ROOT="/absolute/path/to/robocasa-assets"
```

This directory should contain `robocasa/v1.0/...`. No HDF5 export is needed.
