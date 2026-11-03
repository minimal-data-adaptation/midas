"""Config manifests use the training loader's actual filtering implementations."""

import ast
from dataclasses import dataclass
from io import BytesIO
import json
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest


pl = pytest.importorskip("polars")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import create_libero_demo_manifest as creator
import prepare_libero_demos as exporter
sys.path.pop(0)


@dataclass
class Config:
    name: str = "test_libero_config"


@pytest.fixture
def loader():
    # Exercise the real fast-filter classes without importing JAX or models.
    path = ROOT / "openpi/src/openpi/training/data_loader.py"
    names = {
        "FilteredDataset", "EpisodeFilteredDataset", "FeatureRangeFilteredDataset",
        "TransformedDataset", "_raw_dataset_index", "create_torch_dataset",
    }
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if getattr(node, "name", None) in names]
    prefix = ast.parse(
        "from __future__ import annotations\n"
        "import logging\n"
        "from typing import TypeVar, Protocol\n"
        "T_co = TypeVar('T_co')\n"
        "class Dataset(Protocol[T_co]): pass\n"
    ).body
    namespace = {}
    exec(compile(ast.Module(body=prefix + nodes, type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(**namespace)


class Table:
    def __init__(self, **columns):
        self.columns = columns
        self.column_names = list(columns)

    def __getitem__(self, key):
        return self.columns[key]


class RawDataset:
    def __init__(self, root, table):
        self.root = root
        self.hf_dataset = table
        self.meta = SimpleNamespace(
            tasks={0: "move the mug", 1: "move the pot"}, info={"codebase_version": "v2.0"},
        )
        self.revision = "local-test-revision"

    def __len__(self):
        return len(self.hf_dataset["episode_index"])


@pytest.fixture
def selection(tmp_path):
    np = pytest.importorskip("numpy")
    Image = pytest.importorskip("PIL.Image")

    root = tmp_path / "dataset"
    episodes = [10, 10, 11, 11, 86, 86, 86]
    tasks = [1, 1, 0, 0, 1, 1, 1]
    frames = [0, 1, 0, 1, 0, 1, 2]
    buffer = BytesIO()
    Image.fromarray(np.zeros((6, 10, 3), dtype=np.uint8)).save(buffer, format="PNG")
    image = {"bytes": buffer.getvalue(), "path": None}
    raw = RawDataset(root, Table(
        episode_index=episodes, task_index=tasks, frame_index=frames,
        image=[image] * len(episodes), wrist_image=[image] * len(episodes),
        state=[[0.0] * 8] * len(episodes), actions=[[0.0] * 7] * len(episodes),
    ))
    for episode in set(episodes):
        path = exporter.parquet_path(root, episode)
        path.parent.mkdir(parents=True, exist_ok=True)
        positions = [i for i, value in enumerate(episodes) if value == episode]
        pl.DataFrame({key: [raw.hf_dataset[key][i] for i in positions]
                      for key in raw.hf_dataset.column_names}).write_parquet(path)
    data = SimpleNamespace(
        repo_id="physical-intelligence/libero", filter_prompt="move the pot",
        exclude_filter_prompt=False, num_episodes=-1, episode_ids=[86],
        filter_orig_traj_id_6_eq=None, filter_orig_traj_id_6_min=None,
        filter_orig_traj_id_6_max=None, prompt_from_task=True,
    )
    return raw, data


def filtered_dataset(loader, raw, data):
    dataset = loader.FilteredDataset(
        raw, data.filter_prompt, lerobot_dataset=raw, dataset_meta=raw.meta,
        num_episodes=data.num_episodes, exclude_filter_prompt=data.exclude_filter_prompt,
    )
    if data.episode_ids:
        dataset = loader.EpisodeFilteredDataset(dataset, data.episode_ids, lerobot_dataset=raw)
    # Construct the identity-index transform wrapper without needing ML imports.
    wrapped = object.__new__(loader.TransformedDataset)
    wrapped._dataset = dataset
    return wrapped


@pytest.mark.parametrize(("prompt", "ids", "count", "expected"), [
    ("move the pot", [86], -1, [86]),
    ("move the mug", [11], -1, [11]),
    ("move the pot", [], 1, [10]),
    ("move the pot", [], -1, [10, 86]),
])
def test_actual_loader_selection(loader, selection, prompt, ids, count, expected):
    raw, data = selection
    data.filter_prompt, data.episode_ids, data.num_episodes = prompt, ids, count
    dataset = filtered_dataset(loader, raw, data)
    manifest, root = creator.build_manifest(Config(), data, dataset, loader, "demo.hdf5")
    assert root == raw.root
    assert manifest["files"][0]["episode_indices"] == expected
    assert manifest["files"][0]["task_description"] == prompt
    assert manifest["provenance"]["num_samples"] == len(dataset)
    assert manifest["provenance"]["loader_requested_revision"] == "local-test-revision"
    assert manifest["provenance"]["dataset_format_version"] == "v2.0"
    assert "dataset_revision" not in manifest["provenance"]
    exporter.verify_source_files(root, manifest)


def test_reject_missing_requested_episode(loader, selection):
    raw, data = selection
    data.episode_ids = [86, 999]
    dataset = filtered_dataset(loader, raw, data)
    with pytest.raises(ValueError, match="missing after filtering"):
        creator.build_manifest(Config(), data, dataset, loader, "demo.hdf5")


def test_reject_partial_episode(loader, selection):
    raw, data = selection
    dataset = filtered_dataset(loader, raw, data)
    dataset._dataset._valid_indices.pop()
    with pytest.raises(ValueError, match="full source parquet"):
        creator.build_manifest(Config(), data, dataset, loader, "demo.hdf5")


def test_reject_multiple_tasks(loader, selection):
    raw, data = selection
    data.episode_ids = []
    with pytest.raises(ValueError, match="exactly one"):
        creator.build_manifest(Config(), data, raw, loader, "demo.hdf5")


@pytest.mark.parametrize("change", ["parquet", "new_metadata", "missing_checksum"])
def test_reject_source_drift(loader, selection, change):
    raw, data = selection
    manifest, root = creator.build_manifest(
        Config(), data, filtered_dataset(loader, raw, data), loader, "demo.hdf5",
    )
    if change == "parquet":
        exporter.parquet_path(root, 86).write_bytes(b"modified")
    elif change == "new_metadata":
        (root / "meta").mkdir()
        (root / "meta/tasks.jsonl").write_text('{}\n')
    else:
        manifest["provenance"]["source_files"] = []
    with pytest.raises(ValueError, match="checksum"):
        exporter.verify_source_files(root, manifest)


def test_write_roundtrip_and_no_overwrite(loader, selection, tmp_path):
    raw, data = selection
    manifest, _ = creator.build_manifest(
        Config(), data, filtered_dataset(loader, raw, data), loader, "demo.hdf5",
    )
    output = tmp_path / "run/demo_manifest.json"
    creator.write_manifest(output, manifest)
    assert exporter.load_manifest(output) == manifest
    with pytest.raises(FileExistsError):
        creator.write_manifest(output, {})
    assert json.loads(output.read_text()) == manifest


def test_generated_manifest_exports_only_selected_episode(loader, selection, tmp_path):
    h5py = pytest.importorskip("h5py")
    raw, data = selection
    manifest, root = creator.build_manifest(
        Config(), data, filtered_dataset(loader, raw, data), loader, "demo.hdf5",
    )
    path = tmp_path / "manifest.json"
    creator.write_manifest(path, manifest)
    output = tmp_path / "output"
    exporter.main(["--manifest", str(path), "--dataset-root", str(root), "--output-dir", str(output)])
    with h5py.File(output / "demo.hdf5", "r") as source:
        assert set(source) == {"metadata", "episode_0000000"}
        assert source["episode_0000000"].attrs["source_ep_idx"] == 86
        assert len(source["episode_0000000/actions"]) == 3
        assert source["metadata"].attrs["task_description"] == "move the pot"


def test_source_mutation_during_export_does_not_publish(loader, selection, tmp_path, monkeypatch):
    pytest.importorskip("h5py")
    raw, data = selection
    manifest, root = creator.build_manifest(
        Config(), data, filtered_dataset(loader, raw, data), loader, "demo.hdf5",
    )
    original = exporter.read_episode

    def mutate(*args):
        result = original(*args)
        exporter.parquet_path(root, 86).write_bytes(b"changed during export")
        return result

    monkeypatch.setattr(exporter, "read_episode", mutate)
    destination = tmp_path / "output/demo.hdf5"
    with pytest.raises(ValueError, match="checksum"):
        exporter.export_hdf5(root, destination, manifest["files"][0], manifest)
    assert not destination.exists()
    assert not list(destination.parent.glob("*.tmp"))


def test_cli_reports_missing_bc_dependency_without_writing(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(creator.importlib.util, "find_spec", lambda name: None)
    path = tmp_path / "manifest.json"
    with pytest.raises(SystemExit) as error:
        creator.main(["--config-name", Config().name, "--output", str(path)])
    assert error.value.code == 2
    assert "LeRobot is missing" in capsys.readouterr().err
    assert not path.exists()


@pytest.mark.parametrize("ids", [[86], []])
def test_training_loader_only_loads_requested_global_episodes(loader, selection, monkeypatch, ids):
    raw, data = selection
    data.episode_ids = ids
    data.data_dirs = None
    data.action_sequence_keys = ("actions",)
    # Keep prompt filtering active but avoid model/tokenizer transforms here.
    data.prompt_from_task = False
    seen = []

    def construct(repo_id, **kwargs):
        seen.append((repo_id, kwargs))
        return raw

    for name in ("lerobot", "lerobot.datasets", "lerobot.datasets.lerobot_dataset"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    module = sys.modules["lerobot.datasets.lerobot_dataset"]
    raw.meta.fps = 10
    module.LeRobotDatasetMetadata = lambda repo_id: raw.meta
    module.LeRobotDataset = construct
    result = loader.create_torch_dataset(data, 10, None)
    assert len(result) == (3 if ids else 5)
    assert seen[0][1]["episodes"] == (ids or None)
    assert seen[0][1]["delta_timestamps"]["actions"] == [i / 10 for i in range(10)]


@pytest.mark.parametrize(("name", "episode"), [
    ("pi05_libero10_both_mokapots_stove_ep1", 86),
    ("pi05_libero10_white_mug_chocolate_pudding_ep1", 11),
])
def test_updated_configs_pin_correct_global_episodes(name, episode):
    tree = ast.parse((ROOT / "openpi/src/openpi/training/config.py").read_text())
    config = next(node for node in ast.walk(tree) if isinstance(node, ast.Call)
                  and any(kw.arg == "name" and isinstance(kw.value, ast.Constant)
                          and kw.value.value == name for kw in node.keywords))
    base = next(node for node in ast.walk(config) if isinstance(node, ast.Call)
                and getattr(node.func, "id", None) == "DataConfig")
    kwargs = {kw.arg: kw.value for kw in base.keywords}
    assert ast.literal_eval(kwargs["episode_ids"]) == [episode]
    assert "num_episodes" not in kwargs
