from types import SimpleNamespace

import numpy as np

from envs.mock_yam_env import MockYamEnv
from midas.real.yam_env import YamGymEnv
from training.train_utils_real import collect_trajectory, trajectory_transitions


class FakeClient:
    def __init__(self):
        self.version = 3

    def infer(self, observation):
        del observation
        return {
            "actions": np.zeros((2, 14), np.float32),
            "base_action": np.zeros((4, 14), np.float32),
            "a_exec_norm": np.zeros((2, 14), np.float32),
            "actor_version": self.version,
        }


def test_mock_collection_is_query_aligned_and_terminal():
    variant = SimpleNamespace(
        query_freq=2,
        chunk_len=4,
        max_traj_len=8,
        reward_type="sparse",
        num_subtasks=1,
        instruction="task",
        resize_image=8,
        use_vlm_embedding=False,
        label_timeout_seconds=0.0,
        discount=0.99,
    )
    environment_variant = SimpleNamespace(
        seed=0,
        mock_episode_len=4,
        mock_step_seconds=0.0,
        resize_image=8,
        chunk_len=4,
        query_freq=2,
        use_vlm_embedding=False,
    )
    env = YamGymEnv(MockYamEnv(environment_variant), environment_variant)
    trajectory = collect_trajectory(variant, env, FakeClient(), 0, interactive=False)
    assert trajectory["status"] == "success"
    assert trajectory["actions"].shape == (2, 2, 14)
    assert trajectory["rewards"].tolist() == [-1.0, 0.0]
    assert trajectory["masks"].tolist() == [1.0, 0.0]
    transitions = trajectory_transitions(variant, trajectory)
    assert len(transitions) == 2
    assert transitions[-1]["next_observations"]["base_action"].shape == (4, 14, 1)
