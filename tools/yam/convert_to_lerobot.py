#!/usr/bin/env python3
"""Convert supported YAM HDF5/video rollouts to a LeRobot 0.3.3 dataset.

``combined`` mode expects ``INPUT/<task-name>/<episode>/episode.hdf5`` with
nested robot state and three camera MP4s. ``evaluation`` mode expects
``INPUT/<episode>/episode.hdf5`` from ``evaluate_real``. Existing output is
never overwritten.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import av
import h5py
import numpy as np
from PIL import Image
import torch
import yaml


CAMERAS = ("top", "left_wrist", "right_wrist")
MOTORS = [
    *(f"left_joint_{index}" for index in range(6)),
    "left_gripper",
    *(f"right_joint_{index}" for index in range(6)),
    "right_gripper",
]
DEFAULT_PROMPTS = {
    "pick-place": "put the green block in the right bin and the blue block in the left bin",
    "arrange-corn-knife": "place the knife and the donut on the plate",
    "wipe-the-tray-with-the-cloth": "Wipe the black tray with the white cloth",
}


def _decode(path: Path, image_size: int) -> np.ndarray:
    frames = []
    with av.open(str(path)) as container:
        for frame in container.decode(video=0):
            image = frame.to_ndarray(format="rgb24")
            if image.shape[:2] != (image_size, image_size):
                image = np.asarray(
                    Image.fromarray(image).resize(
                        (image_size, image_size), Image.Resampling.BICUBIC
                    )
                )
            frames.append(image)
    if not frames:
        raise ValueError(f"Video has no frames: {path}")
    return np.stack(frames)


def _trajectory_id(name: str, fallback: int) -> int:
    suffix = name.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else fallback


def _read_episode(path: Path, mode: str, image_size: int, fallback_index: int) -> dict:
    with h5py.File(path / "episode.hdf5", "r") as handle:
        actions = np.asarray(handle["actions"], dtype=np.float32)
        attrs = dict(handle.attrs)
        if mode == "evaluation":
            state = np.asarray(handle["state"], dtype=np.float32)
        else:
            state = np.concatenate(
                [
                    np.asarray(handle["robot/left/joint_pos"]),
                    np.asarray(handle["robot/left/gripper_pos"]),
                    np.asarray(handle["robot/right/joint_pos"]),
                    np.asarray(handle["robot/right/gripper_pos"]),
                ],
                axis=1,
            ).astype(np.float32)
    if state.ndim != 2 or state.shape[1] != 14:
        raise ValueError(f"Expected state (T,14) in {path}, got {state.shape}")
    if actions.ndim != 2 or actions.shape[1] < 14:
        raise ValueError(f"Expected actions (T,>=14) in {path}, got {actions.shape}")
    if len(state) != len(actions):
        raise ValueError(f"State/action length mismatch in {path}")
    videos = {camera: _decode(path / f"{camera}.mp4", image_size) for camera in CAMERAS}
    for camera, frames in videos.items():
        if len(frames) != len(state):
            raise ValueError(
                f"{camera} frame count {len(frames)} != state count {len(state)} in {path}"
            )
    return {
        "state": state,
        "actions": actions[:, :14],
        "videos": videos,
        "attrs": attrs,
        "trajectory_id": _trajectory_id(path.name, fallback_index),
    }


def _episode_paths(root: Path, mode: str) -> list[tuple[Path, str | None]]:
    if mode == "evaluation":
        return [(path, None) for path in sorted(root.iterdir()) if path.is_dir()]
    return [
        (episode, task.name)
        for task in sorted(root.iterdir())
        if task.is_dir()
        for episode in sorted(task.iterdir())
        if episode.is_dir()
    ]


def _features(image_size: int) -> dict:
    features = {
        "observation.state": {"dtype": "float32", "shape": (14,), "names": [MOTORS]},
        "action": {"dtype": "float32", "shape": (14,), "names": [MOTORS]},
        "orig_traj_id_6": {"dtype": "int64", "shape": (1,), "names": ["orig_traj_id_6"]},
        "is_success": {"dtype": "int64", "shape": (1,), "names": ["is_success"]},
    }
    for camera in CAMERAS:
        features[f"observation.images.{camera}"] = {
            "dtype": "video",
            "shape": (3, image_size, image_size),
            "names": ["channels", "height", "width"],
        }
    return features


def convert(args) -> None:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    input_root = Path(args.input_dir).expanduser().resolve()
    output_root = Path(args.lerobot_home).expanduser().resolve() / args.repo_id
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing LeRobot dataset: {output_root}")
    prompts = dict(DEFAULT_PROMPTS)
    if args.task_prompts:
        with open(args.task_prompts, encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle) or {}
        if not isinstance(loaded, dict):
            raise ValueError("--task_prompts must contain a YAML mapping")
        prompts.update({str(key): str(value) for key, value in loaded.items()})
    candidates = _episode_paths(input_root, args.mode)
    if not candidates:
        raise ValueError(f"No episodes found under {input_root}")
    dataset = LeRobotDataset.create(
        repo_id=args.repo_id,
        root=output_root,
        fps=args.fps,
        robot_type="yam",
        features=_features(args.image_size),
        use_videos=True,
        tolerance_s=0.0001,
        image_writer_processes=args.image_writer_processes,
        image_writer_threads=args.image_writer_threads,
    )
    kept = 0
    for index, (episode_path, task_name) in enumerate(candidates):
        episode = _read_episode(episode_path, args.mode, args.image_size, index)
        attrs = episode["attrs"]
        status = str(attrs.get("status", "success"))
        success = bool(attrs.get("is_success", status == "success"))
        if args.mode == "evaluation" and status not in {"success", "failure"}:
            continue
        if args.success_only and not success:
            continue
        prompt = str(attrs.get("prompt", "") or prompts.get(task_name, "") or args.fallback_prompt)
        if not prompt:
            raise ValueError(f"No task prompt for {episode_path}")
        trajectory_id = np.asarray([episode["trajectory_id"]], dtype=np.int64)
        success_value = np.asarray([int(success)], dtype=np.int64)
        for frame_index in range(len(episode["state"])):
            frame = {
                "observation.state": torch.from_numpy(episode["state"][frame_index]),
                "action": torch.from_numpy(episode["actions"][frame_index]),
                "orig_traj_id_6": trajectory_id,
                "is_success": success_value,
            }
            for camera in CAMERAS:
                frame[f"observation.images.{camera}"] = episode["videos"][camera][frame_index]
            dataset.add_frame(frame, task=prompt)
        dataset.save_episode()
        kept += 1
    dataset.stop_image_writer()
    if not kept:
        raise ValueError("All candidate episodes were filtered out")
    print(f"Saved {kept} YAM episodes to {output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--lerobot_home", required=True)
    parser.add_argument("--repo_id", required=True)
    parser.add_argument("--mode", choices=("combined", "evaluation"), required=True)
    parser.add_argument("--task_prompts", default="")
    parser.add_argument("--fallback_prompt", default="")
    parser.add_argument("--success_only", action="store_true")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--image_writer_processes", type=int, default=4)
    parser.add_argument("--image_writer_threads", type=int, default=4)
    convert(parser.parse_args())


if __name__ == "__main__":
    main()
