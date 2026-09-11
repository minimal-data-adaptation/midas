"""LeRobot YAM demonstration ingestion for MIDAS replay pre-seeding."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from midas.real.observations import policy_observation, training_observation
from midas.real.rewards import load_action_norm_stats, normalize_action_chunk, transition_reward


LOGGER = logging.getLogger(__name__)


def _image_hwc_uint8(image) -> np.ndarray:
    value = np.asarray(image)
    if np.issubdtype(value.dtype, np.floating):
        value = (value * 255.0).clip(0, 255).astype(np.uint8)
    if value.ndim == 3 and value.shape[0] == 3:
        value = np.transpose(value, (1, 2, 0))
    return np.ascontiguousarray(value, dtype=np.uint8)


def frame_to_observation(frame: dict) -> dict:
    state = np.asarray(frame["observation.state"], dtype=np.float32)
    if state.shape != (14,):
        raise ValueError(f"Expected 14-D LeRobot state, got {state.shape}")
    return {
        "robot": {
            "left/joint_pos": state[:6],
            "left/gripper_pos": state[6:7],
            "right/joint_pos": state[7:13],
            "right/gripper_pos": state[13:14],
        },
        "images": {
            camera: _image_hwc_uint8(frame[f"observation.images.{camera}"])
            for camera in ("top", "left_wrist", "right_wrist")
        },
    }


def filtered_episode_index(
    metadata,
    dataset,
    *,
    prompt: str | None = None,
    trajectory_id_equal: int | None = None,
    trajectory_id_min: int | None = None,
    trajectory_id_max: int | None = None,
) -> list[tuple[int, int, int, str]]:
    need_id = any(
        x is not None for x in (trajectory_id_equal, trajectory_id_min, trajectory_id_max)
    )
    episodes = (
        metadata.episodes.values() if isinstance(metadata.episodes, dict) else metadata.episodes
    )
    result = []
    start = 0
    for episode in episodes:
        length = int(episode["length"])
        tasks = episode.get("tasks", [])
        task = tasks[0] if tasks else ""
        keep = prompt is None or prompt in tasks
        if keep and need_id:
            if "orig_traj_id_6" not in getattr(metadata, "features", {}):
                raise ValueError("Dataset lacks orig_traj_id_6 required by the selected filter")
            trajectory_id = int(np.asarray(dataset[start]["orig_traj_id_6"]).reshape(-1)[0])
            keep = (
                (trajectory_id_equal is None or trajectory_id == trajectory_id_equal)
                and (trajectory_id_min is None or trajectory_id >= trajectory_id_min)
                and (trajectory_id_max is None or trajectory_id <= trajectory_id_max)
            )
        if keep:
            result.append((int(episode["episode_index"]), start, length, task))
        start += length
    return result


def load_lerobot_demos(
    *,
    repo_id: str,
    replay,
    client,
    variant,
    data_root: str,
    norm_stats_path: str,
    num_demos: int = -1,
    success_replay=None,
    filter_prompt: str | None = None,
    filter_orig_traj_id_6_eq: int | None = None,
    filter_orig_traj_id_6_min: int | None = None,
    filter_orig_traj_id_6_max: int | None = None,
) -> int:
    """Load successful YAM demonstrations as stride-one query transitions."""

    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    except ImportError:
        try:
            from lerobot.common.datasets.lerobot_dataset import (
                LeRobotDataset,
                LeRobotDatasetMetadata,
            )
        except ImportError as error:
            raise ImportError(
                "Demo loading requires the real profile's pinned LeRobot package"
            ) from error
    dataset_root = Path(data_root).expanduser() / repo_id if data_root else None
    metadata = LeRobotDatasetMetadata(repo_id, root=dataset_root)
    lerobot = LeRobotDataset(repo_id, root=dataset_root, delta_timestamps=None)
    dataset = lerobot.hf_dataset
    episodes = filtered_episode_index(
        metadata,
        dataset,
        prompt=filter_prompt,
        trajectory_id_equal=filter_orig_traj_id_6_eq,
        trajectory_id_min=filter_orig_traj_id_6_min,
        trajectory_id_max=filter_orig_traj_id_6_max,
    )
    if not episodes:
        raise ValueError(f"YAM demo filters matched no episodes in {repo_id}")
    if num_demos > 0:
        episodes = episodes[:num_demos]
    q01, q99 = load_action_norm_stats(norm_stats_path)
    query_freq = int(variant.query_freq)
    chunk_len = int(variant.chunk_len)
    has_success = "is_success" in getattr(lerobot, "features", getattr(dataset, "features", {}))
    inserted = 0
    for episode_index, start, length, task in episodes:
        if length <= query_freq:
            LOGGER.warning("Skipping short YAM episode %d", episode_index)
            continue
        prompt = task or str(variant.instruction)
        frames = [lerobot[start + index] for index in range(length)]
        observations = [frame_to_observation(frame) for frame in frames]
        states = [np.asarray(frame["observation.state"], np.float32) for frame in frames]
        physical_actions = [np.asarray(frame["action"], np.float32)[:14] for frame in frames]
        bases = []
        features = []
        for observation in observations:
            result = client.infer_base(
                policy_observation(observation, prompt, variant.resize_image, source_color="rgb")
            )
            base = np.asarray(result["base_action"], np.float32)[:chunk_len, :14]
            if base.shape != (chunk_len, 14):
                raise ValueError(f"Server returned insufficient base context: {base.shape}")
            bases.append(base)
            features.append(result.get("vlm_embedding"))
        training = [
            training_observation(
                observation,
                bases[index],
                features[index] if variant.use_vlm_embedding else None,
                variant.resize_image,
                source_color="rgb",
            )
            for index, observation in enumerate(observations)
        ]
        success = True
        if has_success:
            success = bool(int(np.asarray(frames[0]["is_success"]).reshape(-1)[0]))
        transitions = []
        success_transitions = []
        count = length - query_freq
        for index in range(count):
            next_index = index + query_freq
            actions = normalize_action_chunk(
                physical_actions[index : index + query_freq], states[index], q01, q99
            )
            if next_index < count:
                next_actions = normalize_action_chunk(
                    physical_actions[next_index : next_index + query_freq],
                    states[next_index],
                    q01,
                    q99,
                )
            else:
                next_actions = actions
            reward, mask = transition_reward(
                success=success,
                terminal=next_index >= length - 1,
                reward_type=variant.reward_type,
                num_subtasks=variant.num_subtasks,
            )
            transition = {
                "observations": training[index],
                "next_observations": training[next_index],
                "actions": actions,
                "next_actions": next_actions,
                "rewards": reward,
                "masks": mask,
                "discount": np.float32(variant.discount**query_freq),
                "success_flag": np.float32(success),
                "old_log_probs": np.float32(0.0),
            }
            transitions.append(transition)
            if success:
                success_transitions.append(transition)
        replay.insert_trajectory(transitions)
        if success_replay is not None and success_transitions:
            success_replay.insert_trajectory(success_transitions)
        inserted += len(transitions)
    return inserted


__all__ = ["filtered_episode_index", "frame_to_observation", "load_lerobot_demos"]
