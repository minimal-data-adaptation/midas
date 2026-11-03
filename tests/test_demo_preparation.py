"""The portable exporter preserves the historical LIBERO HDF5 contract."""

import importlib.util
from io import BytesIO
import json
from pathlib import Path

import pytest


np = pytest.importorskip("numpy")
h5py = pytest.importorskip("h5py")
pl = pytest.importorskip("polars")
Image = pytest.importorskip("PIL.Image")
ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "prepare_libero_demos", ROOT / "scripts/prepare_libero_demos.py"
)
exporter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exporter)


@pytest.fixture
def dataset(tmp_path):
    root = tmp_path / "dataset"
    parquet = root / "data/chunk-000/episode_000008.parquet"
    parquet.parent.mkdir(parents=True)
    image = np.zeros((6, 10, 3), dtype=np.uint8)
    image[0] = 200
    image[-1] = 40
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    table = pl.DataFrame({
        "image": [{"bytes": buffer.getvalue(), "path": None}] * 3,
        "wrist_image": [{"bytes": buffer.getvalue(), "path": None}] * 3,
        "state": [list(np.arange(8, dtype=float))] * 3,
        "actions": [list(np.arange(7, dtype=float))] * 3,
        "episode_index": [8] * 3,
        "task_index": [5] * 3,
    })
    table.write_parquet(parquet)
    metadata = root / "meta"
    metadata.mkdir()
    (metadata / "tasks.jsonl").write_text(json.dumps({"task_index": 5, "task": "move the object"}) + "\n")
    entry = {"filename": "demo.hdf5", "episode_indices": [8], "episode_lengths": [3],
             "task_description": "move the object"}
    settings = {"upstream_repo": "physical-intelligence/libero", "state_dim": 8,
                "action_dim": 7, "image_size": 224, "query_frequency": 5}
    return root, entry, settings, table, image


def test_manifest_covers_historical_files_and_current_launchers():
    manifest = exporter.load_manifest()
    entries = manifest["files"]
    assert len(entries) == 31
    current = {entry["filename"] for entry in entries if entry["default"]}
    assert len(current) == 10
    for script in (ROOT / "experiments/libero").glob("run_midas_task*.sh"):
        filename = script.read_text().strip().splitlines()[-1].strip()
        assert filename in current
    special = next(entry for entry in entries if entry["filename"].endswith("stove_ep86.hdf5"))
    assert special["episode_indices"] == [86]
    assert special["episode_lengths"] == [421]
    assert manifest["query_frequency"] == 5


def test_export_schema_reward_fallback_and_no_subsampling(dataset, tmp_path):
    root, entry, settings, _, image = dataset
    output = tmp_path / "export/demo.hdf5"
    exporter.export_hdf5(root, output, entry, settings)
    exporter.validate_hdf5(output, entry, settings)
    with h5py.File(output, "r") as source:
        episode = source["episode_0000000"]
        assert episode["actions"].shape == (3, 7)  # query_frequency=5 never drops frames.
        np.testing.assert_array_equal(episode["rewards"][:], np.ones(3))
        np.testing.assert_array_equal(episode["actions"][0], np.arange(7))
        expected = exporter.resize_image(image, root, 224)
        np.testing.assert_array_equal(episode["image"][0], expected)
        assert episode["image"].compression == "lzf"
        assert episode.attrs["source_ep_idx"] == 8
        assert episode.attrs["is_success"] == 1
        assert source["metadata"].attrs["task_description"] == "move the object"


def test_non_square_padding_and_no_vertical_flip(dataset):
    root, _, _, _, image = dataset
    result = exporter.resize_image(image, root, 20)
    assert result.shape == (20, 20, 3)
    assert not result[:4].any()
    assert result[4].mean() > result[15].mean()


def test_existing_output_is_never_overwritten(dataset, tmp_path):
    root, entry, settings, _, _ = dataset
    output = tmp_path / "demo.hdf5"
    output.write_bytes(b"user-owned file")
    with pytest.raises(FileExistsError):
        exporter.export_hdf5(root, output, entry, settings)
    assert output.read_bytes() == b"user-owned file"


@pytest.mark.parametrize("failure", ["missing", "length", "prompt", "episode_id", "actions"])
def test_invalid_inputs_fail_without_publishing(dataset, tmp_path, failure):
    root, entry, settings, table, _ = dataset
    if failure == "missing":
        entry["episode_indices"] = [86]
    elif failure == "length":
        entry["episode_lengths"] = [4]
    elif failure == "prompt":
        entry["task_description"] = "wrong instruction"
    else:
        values = table.to_dict(as_series=False)
        if failure == "episode_id":
            values["episode_index"] = [9] * 3
        else:
            values["actions"] = [[1.0, 2.0]] * 3
        pl.DataFrame(values).write_parquet(root / "data/chunk-000/episode_000008.parquet")
    output = tmp_path / "export/demo.hdf5"
    with pytest.raises((ValueError, FileNotFoundError)):
        exporter.export_hdf5(root, output, entry, settings)
    assert not output.exists()
    assert not list(output.parent.glob("*.tmp"))


def test_source_reward_column_is_preserved(dataset, tmp_path):
    root, entry, settings, table, _ = dataset
    table = table.with_columns(pl.Series("reward", [0.0, 0.5, 0.0]))
    table.write_parquet(root / "data/chunk-000/episode_000008.parquet")
    output = tmp_path / "demo.hdf5"
    exporter.export_hdf5(root, output, entry, settings)
    with h5py.File(output, "r") as source:
        np.testing.assert_array_equal(source["episode_0000000/rewards"][:], [0.0, 0.5, 0.0])
        assert source["episode_0000000"].attrs["is_success"] == 0


def test_reference_mismatch_prevents_publication(dataset, tmp_path):
    root, entry, settings, _, _ = dataset
    reference = tmp_path / "reference.hdf5"
    exporter.export_hdf5(root, reference, entry, settings)
    with h5py.File(reference, "r+") as source:
        source["episode_0000000/actions"][0, 0] += 1.0
    output = tmp_path / "new/demo.hdf5"
    with pytest.raises(AssertionError):
        exporter.export_hdf5(root, output, entry, settings, reference=reference)
    assert not output.exists()


def test_original_selection_json_is_supported(tmp_path):
    selection = tmp_path / "selection.json"
    selection.write_text(json.dumps({
        "repo_id": "physical-intelligence/libero", "filter_prompt": "move the object",
        "episode_indices": [86, 10], "episode_to_sample_indices": {"10": [0, 1], "86": [2, 3, 4]},
    }))
    repo, entry = exporter.selection_from_json(selection, "custom.hdf5")
    assert repo == "physical-intelligence/libero"
    assert entry["episode_indices"] == [10, 86]
    assert entry["episode_lengths"] == [2, 3]
    with pytest.raises(ValueError):
        exporter.selection_from_json(selection, "../escape.hdf5")
