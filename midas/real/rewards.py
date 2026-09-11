"""Reward schemas and YAM action normalization helpers."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


JOINT_MASK = np.asarray([True] * 6 + [False] + [True] * 6 + [False], dtype=bool)
GRIPPER_NO_CAP = np.float32(1_000_000.0)


def load_action_norm_stats(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    stats = raw.get("norm_stats", raw).get("actions")
    if stats is None:
        raise ValueError(f"No actions norm stats in {path}")
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    if q01.shape != (14,) or q99.shape != (14,):
        raise ValueError(f"Expected 14-D action quantiles, got {q01.shape}/{q99.shape}")
    if np.any(q99 <= q01):
        raise ValueError("Every action q99 must be greater than q01")
    return q01, q99


def normalized_trust_region(q01: np.ndarray, q99: np.ndarray, radius_radians: float) -> np.ndarray:
    if radius_radians <= 0:
        raise ValueError("Trust-region radius must be positive")
    cap = (2.0 * float(radius_radians) / (q99 - q01 + 1e-6)).astype(np.float32)
    cap[[6, 13]] = GRIPPER_NO_CAP
    return cap


def normalize_action_chunk(
    actions: np.ndarray, state: np.ndarray, q01: np.ndarray, q99: np.ndarray
) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32).copy()
    state = np.asarray(state, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[-1] != 14 or state.shape != (14,):
        raise ValueError(
            f"Expected actions (T,14) and state (14,), got {actions.shape}/{state.shape}"
        )
    actions -= np.where(JOINT_MASK, state, 0.0)
    return ((actions - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).astype(np.float32)


def transition_reward(
    *, success: bool, terminal: bool, reward_type: str, num_subtasks: int = 1
) -> tuple[np.float32, np.float32]:
    if reward_type == "dense" and success and terminal:
        return np.float32(4 + num_subtasks), np.float32(0.0)
    if reward_type == "sparse" and success and terminal:
        return np.float32(0.0), np.float32(0.0)
    if reward_type not in {"sparse", "dense"}:
        raise ValueError(f"Unknown reward type: {reward_type}")
    return np.float32(-1.0), np.float32(1.0)


def validate_reward_manifest(manifest: dict, reward_type: str, num_subtasks: int) -> None:
    saved_type = manifest.get("reward_type")
    if saved_type is None:
        if reward_type != "sparse":
            raise ValueError("Legacy reward manifest may only be resumed explicitly as sparse")
        return
    if saved_type != reward_type:
        raise ValueError(
            f"Reward schema mismatch: checkpoint={saved_type}, requested={reward_type}"
        )
    if reward_type == "dense" and int(manifest.get("num_subtasks", 0)) != num_subtasks:
        raise ValueError("Dense reward num_subtasks does not match the checkpoint")


__all__ = [
    "JOINT_MASK",
    "load_action_norm_stats",
    "normalize_action_chunk",
    "normalized_trust_region",
    "transition_reward",
    "validate_reward_manifest",
]
