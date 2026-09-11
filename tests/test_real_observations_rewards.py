import numpy as np
import pytest

from midas.real.observations import (
    pixels_from_observation,
    policy_observation,
    training_observation,
)
from midas.real.rewards import normalize_action_chunk, normalized_trust_region, transition_reward


def _observation():
    colors = {
        "top": np.array([1, 2, 3], dtype=np.uint8),
        "left_wrist": np.array([4, 5, 6], dtype=np.uint8),
        "right_wrist": np.array([7, 8, 9], dtype=np.uint8),
    }
    return {
        "robot": {
            "left/joint_pos": np.arange(6, dtype=np.float32),
            "left/gripper_pos": np.array([6], np.float32),
            "right/joint_pos": np.arange(7, 13, dtype=np.float32),
            "right/gripper_pos": np.array([13], np.float32),
        },
        "images": {key: np.tile(value, (2, 3, 1)) for key, value in colors.items()},
    }


def test_yam_camera_order_and_bgr_to_rgb_conversion():
    pixels = pixels_from_observation(_observation(), resize=2, source_color="bgr")
    assert pixels.shape == (2, 2, 9)
    np.testing.assert_array_equal(pixels[0, 0], [3, 2, 1, 6, 5, 4, 9, 8, 7])
    policy = policy_observation(_observation(), "task", resize=2, source_color="bgr")
    assert tuple(policy["images"]) == ("top", "left_wrist", "right_wrist")
    assert policy["images"]["top"].shape == (3, 2, 2)


def test_training_observation_shapes_are_replay_compatible():
    result = training_observation(
        _observation(), np.zeros((4, 14), np.float32), np.zeros(5, np.float32), 2
    )
    assert result["pixels"].shape == (2, 2, 9, 1)
    assert result["state"].shape == (14, 1)
    assert result["base_action"].shape == (4, 14, 1)
    assert result["vlm_embedding"].shape == (5, 1)


def test_action_normalization_and_real_trust_cap():
    q01 = np.full(14, -2.0, np.float32)
    q99 = np.full(14, 2.0, np.float32)
    state = np.arange(14, dtype=np.float32)
    actions = np.tile(state, (2, 1))
    normalized = normalize_action_chunk(actions, state, q01, q99)
    np.testing.assert_allclose(normalized[:, [0, 5, 7, 12]], 0.0, atol=1e-6)
    cap = normalized_trust_region(q01, q99, 0.2)
    np.testing.assert_allclose(cap[[0, 12]], 0.1, atol=1e-6)
    assert cap[6] == cap[13] == pytest.approx(1_000_000.0)
    assert transition_reward(success=True, terminal=True, reward_type="sparse") == (0.0, 0.0)
    assert transition_reward(success=True, terminal=True, reward_type="dense", num_subtasks=3) == (
        7.0,
        0.0,
    )
