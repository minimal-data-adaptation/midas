#!/usr/bin/env python3
"""Resolve an OpenPI LIBERO training config into an exportable demo manifest.

Run in the BC environment, with the same HF_LEROBOT_HOME as training. Uses the
actual training loader, not a second implementation of its episode selection.
This records the current eligible training data, not the history of an old run.
"""

import argparse
from datetime import datetime, timezone
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys

from prepare_libero_demos import load_manifest, parquet_path, sha256_file


ROOT = Path(__file__).resolve().parents[1]


def raw_selection(dataset, loader):
    """Map every selected sample through the loader's wrappers to its raw row."""
    indices = list(range(len(dataset)))
    while isinstance(dataset, (
        loader.FilteredDataset, loader.EpisodeFilteredDataset,
        loader.FeatureRangeFilteredDataset, loader.TransformedDataset,
    )):
        if not isinstance(dataset, loader.TransformedDataset):
            indices = [int(dataset._valid_indices[index]) for index in indices]
        dataset = dataset._dataset
    if not hasattr(dataset, "hf_dataset") or not hasattr(dataset, "meta"):
        raise ValueError("Expected a LeRobot dataset with supported OpenPI filtering wrappers")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("Selection is empty or contains duplicate source rows")
    return dataset, indices


def build_manifest(config, data_config, dataset, loader, filename):
    """Validate full, single-task episodes and fingerprint their source files."""
    import polars as pl

    if Path(filename).name != filename or not filename.endswith(".hdf5"):
        raise ValueError("--filename must be a single .hdf5 filename")
    raw, indices = raw_selection(dataset, loader)
    root = Path(raw.root).resolve()
    table = raw.hf_dataset
    required = {"episode_index", "task_index", "frame_index"}
    if not required.issubset(table.column_names):
        raise ValueError(f"Dataset is missing columns: {required - set(table.column_names)}")
    episodes = table["episode_index"]
    tasks = table["task_index"]
    frames = table["frame_index"]
    selected = {}
    prompts = set()
    for index in indices:
        episode = int(episodes[index])
        selected.setdefault(episode, []).append(index)
        prompts.add(raw.meta.tasks[int(tasks[index])])
    if len(prompts) != 1 or not next(iter(prompts)).strip():
        raise ValueError("HDF5 export requires exactly one nonempty task instruction")
    prompt = next(iter(prompts))
    requested = set(data_config.episode_ids)
    if requested and requested != set(selected):
        raise ValueError(f"Requested episodes missing after filtering: {sorted(requested - set(selected))}")

    sources = []
    lengths = []
    for episode in sorted(selected):
        path = parquet_path(root, episode)
        before = sha256_file(path)
        source = pl.read_parquet(path, columns=sorted(required)).to_dict(as_series=False)
        rows = selected[episode]
        # A flat HDF5 export always copies whole parquet episodes. Reject partial
        # selection, reordered frames, or a stale HF/Arrow cache of another file.
        for field, column in (("episode_index", episodes), ("task_index", tasks), ("frame_index", frames)):
            if [int(column[index]) for index in rows] != source[field]:
                raise ValueError(f"Episode {episode}: selected rows do not match the full source parquet")
        if any(value != episode for value in source["episode_index"]):
            raise ValueError(f"Source file contains another episode: {path}")
        if sha256_file(path) != before:
            raise ValueError(f"Source changed while creating manifest: {path}")
        lengths.append(len(rows))
        sources.append({"path": path.relative_to(root).as_posix(), "sha256": before})

    # Fingerprint metadata as well: language annotations and paths are part of
    # the data identity. Missing metadata files must remain absent at export.
    metadata = []
    for name in ("meta/info.json", "meta/tasks.jsonl", "meta/episodes.jsonl"):
        path = root / name
        metadata.append({"path": name, "sha256": sha256_file(path) if path.exists() else None})
    settings = load_manifest()
    settings["upstream_repo"] = data_config.repo_id
    settings["files"] = [{
        "filename": filename, "default": True, "task_description": prompt,
        "episode_indices": sorted(selected), "episode_lengths": lengths,
    }]
    selection_keys = (
        "repo_id", "filter_prompt", "exclude_filter_prompt", "num_episodes", "episode_ids",
        "filter_orig_traj_id_6_eq", "filter_orig_traj_id_6_min", "filter_orig_traj_id_6_max",
        "prompt_from_task",
    )
    config_path = Path(inspect.getfile(type(config)))
    loader_path = Path(inspect.getfile(loader.create_torch_dataset))
    settings["provenance"] = {
        "kind": "openpi_training_selection", "config_name": config.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": {key: getattr(data_config, key) for key in selection_keys},
        "config_sha256": sha256_file(config_path), "loader_sha256": sha256_file(loader_path),
        # LeRobot can open local v2.0 files while raw.revision remains its
        # default requested v2.1 tag. Do not present that as a resolved commit.
        "loader_requested_revision": getattr(raw, "revision", None),
        "dataset_format_version": getattr(raw.meta, "info", {}).get("codebase_version"),
        "num_samples": len(indices), "source_files": sources + metadata,
    }
    return settings, root


def write_manifest(path, manifest):
    """Never replace an existing run's record."""
    path = Path(path)
    content = json.dumps(manifest, indent=2) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(content)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-name", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--filename", help="HDF5 basename (default: <config-name>.hdf5)")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    if importlib.util.find_spec("lerobot") is None:
        parser.error("LeRobot is missing from this Python environment; use the BC training environment")
    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    sys.path[:0] = [str(ROOT / "openpi/src"), str(ROOT / "openpi/packages/openpi-client/src")]
    from openpi.training import config as configs
    from openpi.training import data_loader as loader

    config = configs.get_config(args.config_name)
    if not isinstance(config.data, configs.LeRobotLiberoDataConfig):
        parser.error("Only LeRobotLiberoDataConfig is supported; RoboCasa needs no HDF5 export")
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.repo_id != "physical-intelligence/libero":
        parser.error("This exporter supports physical-intelligence/libero only")
    if not data_config.prompt_from_task:
        parser.error("Expected prompt_from_task=True so the exported instruction matches BC")
    dataset = loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    manifest, root = build_manifest(
        config, data_config, dataset, loader, args.filename or f"{config.name}.hdf5",
    )
    write_manifest(args.output, manifest)
    entry = manifest["files"][0]
    print(f"Saved: {args.output}")
    print(f"Dataset root: {root}")
    print(f"Episodes: {entry['episode_indices']}; frames: {sum(entry['episode_lengths'])}")
    print(f"Instruction: {entry['task_description']}")
    print("Keep this manifest with the BC run. It describes today's config/data, not an old checkpoint's history.")


if __name__ == "__main__":
    main()
