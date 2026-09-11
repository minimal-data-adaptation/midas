"""Standalone, read-only evaluation client for YAM base or MIDAS policies."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import h5py
import numpy as np

from midas.real.client import RealPolicyClient
from midas.real.config import RealRunSpec
from midas.real.observations import policy_observation, state_from_observation
from midas.real.operator import cbreak_stdin, poll_label, read_final_label
from midas.real.yam_env import create_yam_env
from midas.utils.general_utils import AttrDict


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a fixed real-world policy; never updates weights"
    )
    parser.add_argument("--run_spec", required=True)
    parser.add_argument("--server_host", default="localhost")
    parser.add_argument("--server_port", default=8000, type=int)
    parser.add_argument("--server_api_key", default="")
    parser.add_argument("--server_connect_timeout", default=300.0, type=float)
    parser.add_argument("--mock_env", default=0, type=int)
    parser.add_argument("--mock_episode_len", default=8, type=int)
    parser.add_argument("--mock_step_seconds", default=0.0, type=float)
    parser.add_argument("--yam_env_config_path", default="")
    parser.add_argument("--instruction", default="perform the task")
    parser.add_argument("--num_episodes", default=10, type=int)
    parser.add_argument("--max_episode_steps", default=2500, type=int)
    parser.add_argument("--interactive_labels", default=1, type=int)
    parser.add_argument("--label_timeout_seconds", default=300.0, type=float)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--save_video", default=1, type=int)
    parser.add_argument("--video_fps", default=60, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser


def _episode(variant, spec, env, client, episode_index: int) -> dict:
    states = []
    actions = []
    frames = {camera: [] for camera in spec.cameras}
    status = None
    steps = 0
    env.reset()
    with cbreak_stdin():
        while steps < variant.max_episode_steps and status is None:
            observation = env.get_observation()
            response = client.infer(
                policy_observation(
                    observation, variant.instruction, spec.resize_image, source_color="bgr"
                )
            )
            chunk = np.asarray(response["actions"], dtype=np.float32)
            for action in chunk:
                if steps >= variant.max_episode_steps:
                    break
                states.append(state_from_observation(observation))
                for camera in spec.cameras:
                    frames[camera].append(
                        np.asarray(observation["images"][camera])[..., ::-1].copy()
                    )
                actions.append(action.copy())
                env.step(action)
                steps += 1
                status = poll_label() if variant.interactive_labels else None
                if status is not None:
                    break
                if not variant.interactive_labels and bool(
                    getattr(env.raw_env, "is_done", lambda: False)()
                ):
                    status = "success"
                    break
                observation = env.get_observation()
    if status is None:
        status = (
            read_final_label(variant.label_timeout_seconds)
            if variant.interactive_labels
            else "failure"
        )
    return {
        "episode": episode_index,
        "prompt": variant.instruction,
        "status": status,
        "success": status == "success",
        "counted": status not in {"aborted", "retry"},
        "steps": steps,
        "states": np.asarray(states, dtype=np.float32),
        "actions": np.asarray(actions, dtype=np.float32),
        "frames": frames,
    }


def _save_episode(path: Path, episode: dict, save_video: bool, fps: int) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with h5py.File(path / "episode.hdf5", "w") as handle:
        handle.create_dataset("state", data=episode["states"])
        handle.create_dataset("actions", data=episode["actions"])
        handle.attrs["status"] = episode["status"]
        handle.attrs["is_success"] = episode["success"]
        handle.attrs["num_steps"] = episode["steps"]
        handle.attrs["episode_index"] = episode["episode"]
        handle.attrs["prompt"] = episode["prompt"]
    if save_video and episode["frames"]["top"]:
        import imageio.v2 as imageio

        for camera, frames in episode["frames"].items():
            imageio.mimwrite(path / f"{camera}.mp4", frames, fps=fps)


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    variant = AttrDict(vars(args))
    for key in ("mock_env", "interactive_labels", "save_video"):
        variant[key] = bool(variant[key])
    spec = RealRunSpec.read(variant.run_spec)
    variant.resize_image = spec.resize_image
    variant.chunk_len = spec.chunk_len
    variant.query_freq = spec.query_freq
    variant.use_vlm_embedding = spec.use_vlm_embedding
    variant.vlm_embedding_dim = spec.vlm_embedding_dim
    output = Path(variant.output_dir).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    env = create_yam_env(variant)
    rows = []
    try:
        with RealPolicyClient(
            spec,
            variant.server_host,
            variant.server_port,
            api_key=variant.server_api_key or None,
            connect_timeout=variant.server_connect_timeout,
        ) as client:
            episode_index = 0
            attempts = 0
            while len(rows) < variant.num_episodes:
                episode = _episode(variant, spec, env, client, episode_index)
                attempts += 1
                if episode["status"] == "retry":
                    if attempts >= 10 * variant.num_episodes:
                        raise RuntimeError("Too many retried/aborted real evaluation episodes")
                    continue
                _save_episode(
                    output / f"episode_{episode_index:04d}",
                    episode,
                    variant.save_video,
                    variant.video_fps,
                )
                episode_index += 1
                if episode["counted"]:
                    rows.append(
                        {
                            "episode": episode["episode"],
                            "status": episode["status"],
                            "success": int(episode["success"]),
                            "steps": episode["steps"],
                        }
                    )
    finally:
        env.close()
    with open(output / "summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode", "status", "success", "steps"])
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "episodes": len(rows),
        "successes": sum(row["success"] for row in rows),
        "success_rate": sum(row["success"] for row in rows) / len(rows) if rows else 0.0,
        "rows": rows,
    }
    with open(output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")


if __name__ == "__main__":
    main()
