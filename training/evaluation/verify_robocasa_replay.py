"""Validate an exact replay against the actual BC data before policy evaluation."""

import argparse
from pathlib import Path
import json

import gymnasium
import numpy as np

from openpi.training import config
from openpi.training.robocasa_dataset import RoboCasaDataset, RobocasaRepack
import robocasa.wrappers.gym_wrapper  # noqa: F401; register the environments
from training.robocasa_eval_reset import RoboCasaEvalResetController


def verify(config_name, env_name, output_dir):
    train_config = config.get_config(config_name)
    settings = train_config.data
    if settings.eval_init_mode != "exact_state_replay":
        raise ValueError("Replay verification requires eval_init_mode=exact_state_replay")
    if any((settings.eval_robot_pose_noise, settings.eval_object_pose_noise, settings.eval_object_ori_noise)):
        raise ValueError("Replay verification requires zero perturbation")
    data_config = settings.create(train_config.assets_dirs, train_config.model)
    dataset = RoboCasaDataset(
        data_config.data_dirs, train_config.model.action_horizon, prompt_from_task=True,
    )
    controller = RoboCasaEvalResetController(
        dataset_path=Path(settings.eval_dataset_path), eval_init_mode=settings.eval_init_mode,
        eval_pool_episode_ids=settings.eval_pool_episode_ids, keep_robot_pose=True,
    )
    training_episodes = [episode[1] for episode in dataset.episodes]
    if training_episodes != controller._pool_ids:
        raise ValueError(f"Replay pool {controller._pool_ids} differs from training episodes {training_episodes}")
    env = gymnasium.make(
        f"robocasa/{env_name}", split=None, layout_and_style_ids=settings.layout_and_style_ids,
        seed=0, eval_reset_controller=controller,
    )
    report = []
    try:
        index = 0
        for _, episode_id, length in dataset.episodes:
            training = RobocasaRepack()(dataset[index])
            obs, reset_info = env.reset()
            if reset_info["episode_id"] != episode_id:
                raise ValueError(f"Unexpected replay episode: {reset_info}")
            if obs["annotation.human.task_description"] != training["prompt"]:
                raise ValueError("Replay prompt differs from BC training prompt")
            state = np.concatenate([
                obs[f"state.{key}"] for key in (
                    "end_effector_position_relative", "end_effector_rotation_relative",
                    "base_position", "base_rotation", "gripper_qpos",
                )
            ])
            # Quaternion sign is preserved by the same RoboCasa observation path.
            np.testing.assert_allclose(state, training["observation/state"], rtol=0, atol=1e-4)
            image_errors = {}
            for target, camera in (
                ("observation/image", "robot0_agentview_left"),
                ("observation/wrist_image", "robot0_eye_in_hand"),
                ("observation/image_right", "robot0_agentview_right"),
            ):
                actual = obs[f"video.{camera}"]
                expected = training[target]
                if actual.shape != expected.shape:
                    raise ValueError(f"Camera {camera} shape {actual.shape} differs from BC {expected.shape}")
                # Stored MP4s are lossy: compare appearance, not identical bytes.
                error = float(np.mean(np.abs(actual.astype(float) - expected.astype(float))))
                image_errors[camera] = error
                if error > 10.0:
                    raise ValueError(f"Camera {camera} differs from BC frame 0: mean pixel error={error:.3f}")
                import imageio.v2 as imageio

                imageio.imwrite(output_dir / f"episode_{episode_id:06d}_{camera}_reset.png", actual)
            # Every query after stepping must retain the recorded instruction.
            obs, *_ = env.step({
                "action.end_effector_position": np.zeros(3),
                "action.end_effector_rotation": np.zeros(3),
                "action.gripper_close": np.array([-1.0]),
                "action.base_motion": np.zeros(4),
                "action.control_mode": np.array([-1.0]),
            })
            if obs["annotation.human.task_description"] != training["prompt"]:
                raise ValueError("Replay prompt changed after step")
            report.append({
                "episode_id": episode_id, "prompt": training["prompt"],
                "state_max_abs_error": float(np.max(np.abs(state - training["observation/state"]))),
                "camera_mean_pixel_errors": image_errors,
            })
            index += length
        result = {"config": config_name, "verified": True, "episodes": report}
        (output_dir / "replay_verification.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    finally:
        env.close()
        dataset.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--env-name", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    verify(args.config, args.env_name, args.output_dir)
