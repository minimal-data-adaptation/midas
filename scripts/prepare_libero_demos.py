#!/usr/bin/env python3
"""Prepare LIBERO demo HDF5 files from selected LeRobot parquet episodes.

Episode selections are defined in scripts/data/libero_demo_manifest.json.
Preparation runs on CPU and does not overwrite existing output files.
"""

import argparse
import hashlib
from io import BytesIO
import json
import os
from pathlib import Path
import tempfile


MANIFEST = Path(__file__).with_name("data") / "libero_demo_manifest.json"
FIELDS = ("image", "wrist_image", "state", "actions", "rewards")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_source_files(root, settings):
    """Reject source/version drift for manifests resolved from BC configs."""
    provenance = settings.get("provenance")
    if provenance is None:
        return  # Handwritten manifests remain supported.
    if provenance.get("kind") != "openpi_training_selection":
        raise ValueError("Unsupported manifest provenance")
    files = provenance["source_files"]
    recorded = {}
    for entry in files:
        relative = Path(entry["path"])
        if relative.is_absolute() or ".." in relative.parts or entry["path"] in recorded:
            raise ValueError("Invalid or duplicate source path in manifest")
        path = Path(root) / relative
        expected = entry["sha256"]
        recorded[entry["path"]] = expected
        actual = sha256_file(path) if path.exists() else None
        if actual != expected:
            raise ValueError(f"Source checksum mismatch: {path}; use the dataset used for BC")
    for entry in settings["files"]:
        for episode in entry["episode_indices"]:
            relative = parquet_path(Path(root), episode).relative_to(root).as_posix()
            if not recorded.get(relative):
                raise ValueError(f"Manifest has no source checksum for episode {episode}")


def load_manifest(path=MANIFEST):
    manifest = json.loads(Path(path).read_text())
    if manifest["format_version"] != 1:
        raise ValueError("Unsupported demo manifest version")
    names = [entry["filename"] for entry in manifest["files"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate demo filenames")
    for entry in manifest["files"]:
        if (Path(entry["filename"]).name != entry["filename"]
                or not entry["filename"].endswith(".hdf5")):
            raise ValueError("Demo filenames must not contain directories")
        ids = entry["episode_indices"]
        if not ids or ids != sorted(set(ids)) or min(ids) < 0:
            raise ValueError(f"Invalid episode selection for {entry['filename']}")
        if len(ids) != len(entry["episode_lengths"]):
            raise ValueError("Episode IDs and lengths must have the same size")
    return manifest


def selection_from_json(path, filename):
    """Load a JSON selection containing a repository, task prompt, and episode IDs."""
    selection = json.loads(Path(path).read_text())
    ids = sorted(set(selection["episode_indices"]))
    prompt = selection.get("filter_prompt", "")
    if not ids or min(ids) < 0 or not prompt.strip() or selection.get("exclude_filter_prompt"):
        raise ValueError("Expected a nonempty, single-task inclusion selection")
    if Path(filename).name != filename or not filename.endswith(".hdf5"):
        raise ValueError("--filename must be a single .hdf5 filename")
    ranges = selection.get("episode_to_sample_indices", {})
    lengths = [len(ranges[str(index)]) if str(index) in ranges else None for index in ids]
    return selection["repo_id"], {
        "filename": filename, "task_description": prompt,
        "episode_indices": ids, "episode_lengths": lengths,
    }


def parquet_path(root, episode_id):
    info_path = root / "meta/info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    # physical-intelligence/libero uses the episode-per-parquet v2 layout.
    pattern = info.get("data_path", "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet")
    chunk_size = info.get("chunks_size", 1000)
    if not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("Invalid LeRobot chunks_size")
    relative = Path(pattern.format(episode_chunk=episode_id // chunk_size, episode_index=episode_id))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Dataset parquet paths must stay within the dataset root")
    return root / relative


def resize_image(value, root, size):
    """Decode embedded parquet images; match OpenPI's PIL bilinear pad-resize."""
    import numpy as np
    from PIL import Image

    if isinstance(value, dict):
        if value.get("bytes") is not None:
            with Image.open(BytesIO(value["bytes"])) as image:
                array = np.asarray(image, dtype=np.uint8)
        elif value.get("path"):
            path = Path(value["path"])
            with Image.open(path if path.is_absolute() else root / path) as image:
                array = np.asarray(image, dtype=np.uint8)
        else:
            raise ValueError("Image has neither embedded bytes nor a path")
    else:
        array = np.asarray(value, dtype=np.uint8)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an RGB image, got {array.shape}")
    if array.shape[:2] == (size, size):
        return array
    image = Image.fromarray(array)
    width, height = image.size
    ratio = max(width / size, height / size)
    resized_width, resized_height = int(width / ratio), int(height / ratio)
    image = image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)
    padded = Image.new(image.mode, (size, size), 0)
    padded.paste(image, (max(0, (size - resized_width) // 2), max(0, (size - resized_height) // 2)))
    return np.asarray(padded, dtype=np.uint8)


def read_episode(root, episode_id, expected_length, prompt, settings):
    import numpy as np
    import polars as pl

    path = parquet_path(root, episode_id)
    if not path.is_file():
        raise FileNotFoundError(f"Missing selected episode {episode_id}: {path}")
    frame = pl.read_parquet(path)
    length = frame.height
    if length == 0 or (expected_length is not None and length != expected_length):
        raise ValueError(f"Episode {episode_id}: expected {expected_length} frames, found {length}")
    required = {"image", "wrist_image", "state", "actions", "episode_index", "task_index"}
    if not required.issubset(frame.columns):
        raise ValueError(f"Episode {episode_id}: missing columns {required - set(frame.columns)}")
    table = frame.to_dict(as_series=False)
    if any(index != episode_id for index in table["episode_index"]):
        raise ValueError(f"Parquet contains frames from another episode: {path}")

    # When metadata is available, reject a mislabeled language instruction.
    tasks_path = root / "meta/tasks.jsonl"
    if tasks_path.exists():
        tasks = {row["task_index"]: row["task"] for row in
                 (json.loads(line) for line in tasks_path.read_text().splitlines() if line.strip())}
        if any(tasks.get(index) != prompt for index in table["task_index"]):
            raise ValueError(f"Episode {episode_id}: task instruction does not match selection")

    values = {}
    for field in ("image", "wrist_image"):
        values[field] = np.stack([
            resize_image(value, root, settings["image_size"])
            for value in table[field]
        ])
    for field, dim in (("state", settings["state_dim"]), ("actions", settings["action_dim"])):
        values[field] = np.asarray(table[field], dtype=np.float32)
        if values[field].shape != (length, dim) or not np.isfinite(values[field]).all():
            raise ValueError(f"Episode {episode_id}: invalid {field} array")
    # Use source rewards when available; otherwise fill with 1.0.
    rewards = table["reward"] if "reward" in table else [1.0] * length
    values["rewards"] = np.asarray(rewards, dtype=np.float32)
    if values["rewards"].shape != (length,) or not np.isfinite(values["rewards"]).all():
        raise ValueError(f"Episode {episode_id}: invalid rewards")
    return values


def validate_hdf5(path, entry, settings):
    """Check episode selections, dimensions, and required flat-layout fields."""
    import h5py

    with h5py.File(path, "r") as source:
        ids = entry["episode_indices"]
        names = [f"episode_{index:07d}" for index in range(len(ids))]
        if set(source) != {"metadata", *names}:
            raise ValueError(f"Unexpected episode groups: {path}")
        expected_meta = {key: settings[key] for key in
                         ("image_size", "query_frequency", "state_dim", "action_dim")}
        expected_meta.update(task_description=entry["task_description"], num_episodes=len(ids))
        for key, value in expected_meta.items():
            if source["metadata"].attrs.get(key) != value:
                raise ValueError(f"Metadata mismatch for {key}: {path}")
        for index, (episode_id, expected_length) in enumerate(
            zip(ids, entry["episode_lengths"], strict=True)
        ):
            episode = source[names[index]]
            length = len(episode["actions"])
            if (expected_length is not None and length != expected_length) or length == 0:
                raise ValueError(f"Incorrect frame count for episode {episode_id}: {path}")
            attrs = {"source_ep_idx": episode_id, "num_steps": length, "repo_id": settings["upstream_repo"]}
            for key, value in attrs.items():
                if episode.attrs.get(key) != value:
                    raise ValueError(f"Provenance mismatch for {key}: {path}")
            if set(episode) != set(FIELDS) or "is_success" not in episode.attrs:
                raise ValueError(f"Invalid episode schema: {path}")
            for key in FIELDS:
                tail = {"image": (settings["image_size"], settings["image_size"], 3),
                        "wrist_image": (settings["image_size"], settings["image_size"], 3),
                        "state": (settings["state_dim"],), "actions": (settings["action_dim"],),
                        "rewards": ()}[key]
                dtype = "uint8" if key in ("image", "wrist_image") else "float32"
                if episode[key].shape != (length, *tail) or str(episode[key].dtype) != dtype:
                    raise ValueError(f"Incorrect shape/dtype for {key}: {path}")


def compare_hdf5(actual, reference):
    """Compare array contents and attributes, not HDF5 container byte layout."""
    import h5py
    import numpy as np

    with h5py.File(actual, "r") as left, h5py.File(reference, "r") as right:
        def compare_group(a, b):
            if set(a) != set(b) or set(a.attrs) != set(b.attrs):
                raise ValueError(f"Schema/attribute mismatch at {a.name}")
            for key in a.attrs:
                np.testing.assert_array_equal(a.attrs[key], b.attrs[key])
            for key in a:
                if isinstance(a[key], h5py.Group):
                    compare_group(a[key], b[key])
                else:
                    if a[key].shape != b[key].shape or a[key].dtype != b[key].dtype:
                        raise ValueError(f"Shape/dtype mismatch at {a[key].name}")
                    for start in range(0, len(a[key]), 32):
                        np.testing.assert_array_equal(a[key][start:start + 32], b[key][start:start + 32])
        compare_group(left, right)


def export_hdf5(root, destination, entry, settings, reference=None):
    import h5py

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite {destination}; choose a new output directory")
    verify_source_files(root, settings)
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(prefix=f".{destination.name}.", suffix=".tmp",
                                         dir=destination.parent, delete=False)
    temporary = Path(handle.name)
    handle.close()
    try:
        with h5py.File(temporary, "w") as target:
            metadata = target.create_group("metadata")
            for key in ("query_frequency", "image_size", "state_dim", "action_dim"):
                metadata.attrs[key] = settings[key]
            metadata.attrs["task_description"] = entry["task_description"]
            metadata.attrs["num_episodes"] = len(entry["episode_indices"])
            for index, (episode_id, length) in enumerate(
                zip(entry["episode_indices"], entry["episode_lengths"], strict=True)
            ):
                values = read_episode(root, episode_id, length, entry["task_description"], settings)
                episode = target.create_group(f"episode_{index:07d}")
                for key, value in values.items():
                    compression = "lzf" if key in ("image", "wrist_image") else None
                    episode.create_dataset(key, data=value, compression=compression)
                episode.attrs["source_ep_idx"] = episode_id
                episode.attrs["repo_id"] = settings["upstream_repo"]
                episode.attrs["num_steps"] = len(values["actions"])
                episode.attrs["is_success"] = int(values["rewards"][-1] > 0.5)
        validate_hdf5(temporary, entry, settings)
        if reference is not None:
            compare_hdf5(temporary, reference)
        # Also catch a source mutation during the export before publishing it.
        verify_source_files(root, settings)
        # Atomic no-clobber publication, even if two exporters race for the same name.
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root", type=Path,
        help="Local LeRobot root containing data/chunk-* parquet files",
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="New exports; existing HDF5 files are never overwritten",
    )
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    choice = parser.add_mutually_exclusive_group()
    choice.add_argument("--files", nargs="+", help="Demo filenames from --list")
    choice.add_argument("--all", action="store_true", help="Export all selections in the manifest")
    choice.add_argument("--selection-json", type=Path, help="JSON task prompt and episode selection")
    parser.add_argument("--filename", help="Output filename when using --selection-json")
    parser.add_argument(
        "--list", action="store_true",
        help="List available demo selections",
    )
    parser.add_argument("--verify-dir", type=Path, help="Validate existing HDF5s without writing files")
    parser.add_argument(
        "--reference-dir", type=Path,
        help="Compare arrays and attributes against reference HDF5s before saving",
    )
    args = parser.parse_args(argv)
    settings = load_manifest(args.manifest)
    entries = settings["files"]
    if args.list:
        for entry in entries:
            marker = "*" if entry["default"] else " "
            print(f"{marker} {entry['filename']}: {entry['episode_indices']}")
        return
    if args.selection_json:
        if settings.get("provenance"):
            parser.error("Use config-generated manifests directly, without --selection-json")
        if not args.filename:
            parser.error("--selection-json requires --filename")
        repo_id, entry = selection_from_json(args.selection_json, args.filename)
        settings["upstream_repo"] = repo_id
        entries = [entry]
    elif args.files:
        selected = set(args.files)
        entries = [entry for entry in entries if entry["filename"] in selected]
        missing = selected - {entry["filename"] for entry in entries}
        if missing:
            parser.error(f"Unknown demo filenames: {sorted(missing)}")
    elif not args.all:
        entries = [entry for entry in entries if entry["default"]]
    if args.verify_dir:
        for entry in entries:
            validate_hdf5(args.verify_dir / entry["filename"], entry, settings)
            print(f"Verified: {entry['filename']}")
        return
    if not args.dataset_root or not args.output_dir:
        parser.error("Export requires --dataset-root and --output-dir")
    # Refuse an entire selection up front rather than exporting a partial bundle.
    for entry in entries:
        if (args.output_dir / entry["filename"]).exists():
            parser.error(f"Output already exists: {args.output_dir / entry['filename']}")
        for episode_id in entry["episode_indices"]:
            if not parquet_path(args.dataset_root, episode_id).is_file():
                parser.error(f"Missing selected episode {episode_id} in {args.dataset_root}")
    for entry in entries:
        reference = args.reference_dir / entry["filename"] if args.reference_dir else None
        export_hdf5(args.dataset_root, args.output_dir / entry["filename"], entry, settings, reference)
        print(f"Exported: {entry['filename']} <- {entry['episode_indices']}")


if __name__ == "__main__":
    main()
