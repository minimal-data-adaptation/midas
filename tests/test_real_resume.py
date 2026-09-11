import json

import gym
import numpy as np

from midas.data.replay_buffer import ReplayBuffer
from midas.utils.resume import resolve_resume
from midas.utils.snapshots import BufferSnapshotState, save_buffer_delta, write_json_manifest


def _transition(value):
    observation = {"state": np.array([value], dtype=np.float32)}
    return {
        "observations": observation,
        "next_observations": observation,
        "actions": np.array([value], dtype=np.float32),
        "next_actions": np.array([value], dtype=np.float32),
        "rewards": np.float32(-1),
        "masks": np.float32(1),
        "discount": np.float32(0.9),
        "success_flag": np.float32(0),
        "old_log_probs": np.float32(0),
        "mc_returns": np.float32(-1),
    }


def test_v3_resume_preserves_version_and_incremental_chain(tmp_path):
    observation_space = gym.spaces.Dict({"state": gym.spaces.Box(-10, 10, (1,), np.float32)})
    action_space = gym.spaces.Box(-10, 10, (1,), np.float32)
    replay = ReplayBuffer(observation_space, action_space, 2)
    replay.insert(_transition(1))
    replay.increment_traj_counter()
    snapshot = BufferSnapshotState()
    save_buffer_delta(replay, "online", tmp_path, snapshot)
    (tmp_path / "checkpoint7").mkdir()
    manifest = {
        "format_version": 3,
        "online": {"traj_count": 1, "delta_files": snapshot.prev_delta_files},
        "success": {"traj_count": 0, "delta_files": []},
        "reward_type": "sparse",
    }
    write_json_manifest(manifest, tmp_path / "train_state" / "7.json")
    info = resolve_resume(str(tmp_path))
    assert info["format_version"] == 3
    assert info["online_traj_count"] == 1
    assert len(info["online_delta_paths"]) == 1
    assert json.loads((tmp_path / "train_state" / "7.json").read_text()) == manifest
