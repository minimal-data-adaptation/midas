"""YAM observation conversion shared by real training and evaluation."""

from __future__ import annotations

import numpy as np
from PIL import Image

from midas.real.config import YAM_ACTION_DIM, YAM_CAMERAS, YAM_STATE_DIM


def preprocess_image(image: np.ndarray, resize: int, *, source_color: str) -> np.ndarray:
    """Convert a camera frame to direct-resized RGB HWC uint8."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected HWC three-channel image, got {array.shape}")
    if source_color == "bgr":
        array = array[..., ::-1]
    elif source_color != "rgb":
        raise ValueError(f"source_color must be 'bgr' or 'rgb', got {source_color!r}")
    if np.issubdtype(array.dtype, np.floating):
        scale = 255.0 if array.size and float(np.nanmax(array)) <= 1.0 else 1.0
        array = np.nan_to_num(array * scale).clip(0, 255).astype(np.uint8)
    else:
        array = array.clip(0, 255).astype(np.uint8, copy=False)
    array = np.ascontiguousarray(array)
    if array.shape[:2] != (resize, resize):
        array = np.asarray(
            Image.fromarray(array).resize((resize, resize), Image.Resampling.BICUBIC)
        )
    return np.ascontiguousarray(array, dtype=np.uint8)


def state_from_observation(observation: dict) -> np.ndarray:
    robot = observation["robot"]
    state = np.concatenate(
        [
            np.asarray(robot["left/joint_pos"], dtype=np.float32),
            np.asarray(robot["left/gripper_pos"], dtype=np.float32),
            np.asarray(robot["right/joint_pos"], dtype=np.float32),
            np.asarray(robot["right/gripper_pos"], dtype=np.float32),
        ]
    )
    if state.shape != (YAM_STATE_DIM,) or not np.all(np.isfinite(state)):
        raise ValueError(f"Expected finite YAM state ({YAM_STATE_DIM},), got {state.shape}")
    return state


def pixels_from_observation(
    observation: dict, resize: int = 224, *, source_color: str = "bgr"
) -> np.ndarray:
    images = observation["images"]
    missing = [camera for camera in YAM_CAMERAS if camera not in images]
    if missing:
        raise ValueError(f"YAM observation is missing cameras: {missing}")
    return np.concatenate(
        [
            preprocess_image(images[camera], resize, source_color=source_color)
            for camera in YAM_CAMERAS
        ],
        axis=-1,
    )


def policy_observation(
    observation: dict,
    prompt: str,
    resize: int = 224,
    *,
    source_color: str = "bgr",
) -> dict:
    images = {
        camera: np.transpose(
            preprocess_image(observation["images"][camera], resize, source_color=source_color),
            (2, 0, 1),
        )
        for camera in YAM_CAMERAS
    }
    output = {"images": images, "state": state_from_observation(observation)}
    if prompt:
        output["prompt"] = prompt
    return output


def training_observation(
    observation: dict,
    base_action: np.ndarray,
    vlm_embedding: np.ndarray | None,
    resize: int = 224,
    *,
    source_color: str = "bgr",
) -> dict[str, np.ndarray]:
    base = np.asarray(base_action, dtype=np.float32)
    if base.ndim != 2 or base.shape[-1] != YAM_ACTION_DIM or not np.all(np.isfinite(base)):
        raise ValueError(
            f"Expected finite base action (chunk_len, {YAM_ACTION_DIM}), got {base.shape}"
        )
    result = {
        "pixels": pixels_from_observation(observation, resize, source_color=source_color)[
            ..., np.newaxis
        ],
        "state": state_from_observation(observation)[..., np.newaxis],
        "base_action": base[..., np.newaxis],
    }
    if vlm_embedding is not None:
        embedding = np.asarray(vlm_embedding, dtype=np.float32).reshape(-1)
        if not np.all(np.isfinite(embedding)):
            raise ValueError("VLM embedding contains non-finite values")
        result["vlm_embedding"] = embedding[..., np.newaxis]
    return result


# Source-compatible aliases ease review without exposing old algorithm names.
obs_to_img_yam = pixels_from_observation
obs_to_pi_zero_input_yam = policy_observation
obs_to_qpos_yam = state_from_observation
build_training_obs = training_observation


__all__ = [
    "pixels_from_observation",
    "policy_observation",
    "preprocess_image",
    "state_from_observation",
    "training_observation",
]
