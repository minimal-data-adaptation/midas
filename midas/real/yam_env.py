"""Gym-space adapter for YAM hardware and its mock implementation."""

from __future__ import annotations

import gym
import numpy as np

from midas.real.config import YAM_ACTION_DIM, YAM_CAMERAS, YAM_STATE_DIM


class YamGymEnv:
    """Small adapter exposing the spaces required by ``MidasLearner``."""

    def __init__(self, env, variant) -> None:
        self.raw_env = env
        resize = int(getattr(variant, "resize_image", 224))
        chunk_len = int(getattr(variant, "chunk_len", 60))
        query_freq = int(getattr(variant, "query_freq", 30))
        spaces = {
            "pixels": gym.spaces.Box(0, 255, (resize, resize, 3 * len(YAM_CAMERAS), 1), np.uint8),
            "state": gym.spaces.Box(-np.inf, np.inf, (YAM_STATE_DIM, 1), np.float32),
            "base_action": gym.spaces.Box(-1.0, 1.0, (chunk_len, YAM_ACTION_DIM, 1), np.float32),
        }
        if bool(getattr(variant, "use_vlm_embedding", False)):
            dimension = int(getattr(variant, "vlm_embedding_dim", 2048))
            spaces["vlm_embedding"] = gym.spaces.Box(-np.inf, np.inf, (dimension, 1), np.float32)
        self.observation_space = gym.spaces.Dict(spaces)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (query_freq, YAM_ACTION_DIM), np.float32)

    def reset(self):
        return self.raw_env.reset()

    def step(self, action):
        return self.raw_env.step(action)

    def get_observation(self) -> dict:
        if hasattr(self.raw_env, "get_observation"):
            return self.raw_env.get_observation()
        return self.raw_env._get_obs()

    def close(self) -> None:
        if hasattr(self.raw_env, "close"):
            self.raw_env.close()


def create_yam_env(variant) -> YamGymEnv:
    """Create mock or hardware YAM without importing hardware in mock mode."""

    if bool(getattr(variant, "mock_env", False)):
        from envs.mock_yam_env import MockYamEnv

        return YamGymEnv(MockYamEnv(variant), variant)
    config_path = str(getattr(variant, "yam_env_config_path", "") or "")
    if not config_path:
        raise ValueError("--yam_env_config_path is required unless --mock_env=1")
    try:
        from yam_teleop.env import YAMBimanualEnv
    except ImportError as error:
        raise ImportError(
            "Real YAM training requires the optional yam_teleop package; see docs/REAL_WORLD.md"
        ) from error
    return YamGymEnv(YAMBimanualEnv(config_path), variant)


__all__ = ["YamGymEnv", "create_yam_env"]
