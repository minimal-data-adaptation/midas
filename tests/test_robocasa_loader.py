import json

import h5py
import numpy as np

from midas.utils.general_utils import AttrDict
from midas.utils.robocasa_utils import (
    load_robocasa_hdf5_demos_to_buffer,
    resolve_robocasa_demo_episode_ids,
)
from training.robocasa_eval_reset import RoboCasaEvalResetController


class _ZeroPolicy:
    def infer(self, observation, return_vlm_embedding=False):
        assert observation["prompt"] == "move the object"
        assert return_vlm_embedding is False
        return {"actions": np.zeros((1, 12), dtype=np.float32)}


class _RecordingBuffer:
    def __init__(self):
        self.transitions = []
        self.trajectories = 0

    def insert(self, transition):
        self.transitions.append(transition)

    def increment_traj_counter(self):
        self.trajectories += 1

    def __len__(self):
        return len(self.transitions)


def test_flat_robocasa_hdf5_loader(tmp_path):
    path = tmp_path / "demo.hdf5"
    with h5py.File(path, "w") as file:
        metadata = file.create_group("metadata")
        metadata.attrs["task_description"] = "move the object"
        episode = file.create_group("episode_0000000")
        episode.create_dataset("image", data=np.zeros((3, 8, 8, 3), dtype=np.uint8))
        episode.create_dataset("wrist_image", data=np.zeros((3, 8, 8, 3), dtype=np.uint8))
        episode.create_dataset("state", data=np.zeros((3, 16), dtype=np.float32))
        episode.create_dataset("actions", data=np.ones((3, 12), dtype=np.float32) * 0.2)
        episode.create_dataset("rewards", data=np.array([0, 0, 1], dtype=np.float32))
    variant = AttrDict(
        query_freq=1,
        chunk_len=1,
        action_dim=12,
        residual_alpha=0.1,
        resize_image=8,
        add_states=True,
        predict_a_exec=False,
        reward_type="sparse",
        discount=0.99,
        use_vlm_embedding=False,
        actor_pop_base_actions=False,
        critic_pop_base_actions=True,
    )
    buffer = _RecordingBuffer()
    load_robocasa_hdf5_demos_to_buffer(str(path), buffer, _ZeroPolicy(), variant)
    assert len(buffer) == 2
    assert buffer.trajectories == 1
    np.testing.assert_allclose(buffer.transitions[0]["actions"], 2.0)
    assert buffer.transitions[-1]["masks"] == 0.0


def test_scene_filtered_episode_resolution_does_not_require_openpi_groot_utils(
    tmp_path,
):
    dataset = tmp_path / "lerobot"
    (dataset / "meta").mkdir(parents=True)
    episodes = [{"episode_index": 2}, {"episode_index": 7}]
    (dataset / "meta" / "episodes.jsonl").write_text(
        "".join(json.dumps(episode) + "\n" for episode in episodes)
    )
    for episode_id, layout_id in [(2, 11), (7, 50)]:
        extra = dataset / "extras" / f"episode_{episode_id:06d}"
        extra.mkdir(parents=True)
        (extra / "ep_meta.json").write_text(json.dumps({
            "layout_id": layout_id,
            "style_id": 37,
            "fixture_refs": {"fridge": "fridge_left"},
            "object_cfgs": [{"info": {"cat": "apple"}}],
        }))

    config = AttrDict(
        layout_and_style_ids=[(50, 37)],
        eval_pool_episode_ids=None,
        eval_pool_fixture_refs={"fridge": "fridge_left"},
        eval_pool_object_categories=["apple"],
    )
    assert resolve_robocasa_demo_episode_ids(dataset, config) == [7]
    controller = RoboCasaEvalResetController(
        dataset_path=dataset,
        eval_init_mode="fixture_pair_fresh_placement",
        layout_and_style_ids=[(50, 37)],
        eval_pool_fixture_refs={"fridge": "fridge_left"},
        eval_pool_object_categories=["apple"],
    )
    assert controller._pool_ids == [7]

    (dataset / "meta" / "episodes.jsonl").unlink()
    (dataset / "manifest.json").write_text(json.dumps({
        "episodes": [{"episode_id": 2}, {"episode_id": 7}],
    }))
    manifest_controller = RoboCasaEvalResetController(
        dataset_path=dataset,
        eval_init_mode="fixture_pair_fresh_placement",
        layout_and_style_ids=[(50, 37)],
    )
    assert manifest_controller._pool_ids == [7]
