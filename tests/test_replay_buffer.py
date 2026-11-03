import gym
import numpy as np

from midas.data import ReplayBuffer
from midas.data.dataset import Dataset


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


def _filled_buffer(seed):
    observations = gym.spaces.Dict({"state": gym.spaces.Box(-10, 10, (2,), np.float32)})
    actions = gym.spaces.Box(-10, 10, (1,), np.float32)
    buffer = ReplayBuffer(observations, actions, capacity=10)
    for value in range(10):
        buffer.insert(_transition(value))
    buffer.increment_traj_counter()
    buffer.seed(seed)
    return buffer


def test_replay_buffer_sampling_uses_its_private_seed():
    np.random.seed(1)
    first = _filled_buffer(123).sample(8)["rewards"]
    np.random.seed(999)
    second = _filled_buffer(123).sample(8)["rewards"]
    np.testing.assert_array_equal(first, second)

    np.random.seed(1)
    different = _filled_buffer(124).sample(8)["rewards"]
    assert not np.array_equal(first, different)


def test_replay_buffer_full_save_restores_next_sample(tmp_path):
    original = _filled_buffer(7)
    original.sample(5)
    path = tmp_path / "replay.pkl"
    original.save(path)
    expected = original.sample(12)["rewards"]

    restored = _filled_buffer(999)
    restored.restore(path)
    actual = restored.sample(12)["rewards"]
    np.testing.assert_array_equal(actual, expected)


def test_dataset_split_derives_reproducible_child_streams():
    data = {"value": np.arange(20)}
    first_train, first_test = Dataset(data, seed=31).split(0.5)
    second_train, second_test = Dataset(data, seed=31).split(0.5)

    np.testing.assert_array_equal(
        first_train.sample(8)["value"], second_train.sample(8)["value"]
    )
    np.testing.assert_array_equal(
        first_test.sample(8)["value"], second_test.sample(8)["value"]
    )
