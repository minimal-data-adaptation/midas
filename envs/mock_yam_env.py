"""Deterministic, hardware-free stand-in for the YAM bimanual robot."""

from __future__ import annotations

import time

import numpy as np

from midas.real.config import YAM_ACTION_DIM, YAM_CAMERAS


class MockYamEnv:
    def __init__(self, variant) -> None:
        self._rng = np.random.default_rng(int(getattr(variant, "seed", 0)))
        self._step = 0
        self._episode_len = int(getattr(variant, "mock_episode_len", 8))
        self._step_seconds = float(getattr(variant, "mock_step_seconds", 0.0))
        self.last_action = None

    def reset(self) -> None:
        self._step = 0
        self.last_action = None

    def step(self, action) -> None:
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (YAM_ACTION_DIM,) or not np.all(np.isfinite(action)):
            raise ValueError(f"Expected finite action ({YAM_ACTION_DIM},), got {action.shape}")
        self.last_action = action
        self._step += 1
        if self._step_seconds:
            time.sleep(self._step_seconds)

    def get_observation(self) -> dict:
        return {
            "robot": {
                "left/joint_pos": self._rng.uniform(-1, 1, 6).astype(np.float32),
                "left/gripper_pos": self._rng.uniform(0, 1, 1).astype(np.float32),
                "right/joint_pos": self._rng.uniform(-1, 1, 6).astype(np.float32),
                "right/gripper_pos": self._rng.uniform(0, 1, 1).astype(np.float32),
            },
            "images": {
                camera: self._rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
                for camera in YAM_CAMERAS
            },
        }

    _get_obs = get_observation

    def is_done(self) -> bool:
        return self._step >= self._episode_len

    def close(self) -> None:
        return None


__all__ = ["MockYamEnv"]
