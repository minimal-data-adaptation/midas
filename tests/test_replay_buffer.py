import gym
import numpy as np

from midas.data import ReplayBuffer


def _transition(value):
    observation = {"state": np.array([value, value + 1], dtype=np.float32)}
    return {
        "observations": observation,
        "next_observations": observation,
        "actions": np.array([value], dtype=np.float32),
        "next_actions": np.array([value], dtype=np.float32),
        "rewards": float(value),
        "masks": 1.0,
        "discount": 0.9,
        "success_flag": 0.0,
        "old_log_probs": 0.0,
        "mc_returns": 0.0,
    }


def test_replay_buffer_insert_grows_and_samples():
    observations = gym.spaces.Dict({"state": gym.spaces.Box(-10, 10, (2,), np.float32)})
    actions = gym.spaces.Box(-10, 10, (1,), np.float32)
    buffer = ReplayBuffer(observations, actions, capacity=1)
    buffer.insert(_transition(1))
    buffer.insert(_transition(2))
    buffer.increment_traj_counter()
    assert len(buffer) == 2
    assert buffer.capacity == 2
    sample = buffer.sample(2, indx=np.array([0, 1]))
    np.testing.assert_allclose(sample["rewards"], [1, 2])
