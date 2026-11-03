import csv
import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from training.evaluation import evaluate_sim
from training.evaluation.position_perturbation import (
    get_perturb_objects,
    sample_disk_perturbation,
)
from midas.utils.general_utils import AttrDict


@pytest.mark.parametrize("requery", [False, True])
@pytest.mark.parametrize("action_dim", [7, 12])
def test_base_only_robocasa_executes_bc_actions_without_residual(
    tmp_path, monkeypatch, requery, action_dim
):
    from training import train_utils_sim

    class Env:
        def __init__(self):
            self.rng = np.random.default_rng(0)
            self.episode = 0
            self.actions = []

        def observation(self):
            return {
                "video.robot0_agentview_left": np.zeros((16, 16, 3), dtype=np.uint8),
                "video.robot0_eye_in_hand": np.zeros((16, 16, 3), dtype=np.uint8),
                "video.robot0_agentview_right": np.zeros((16, 16, 3), dtype=np.uint8),
                "state.end_effector_position_relative": np.zeros(3),
                "state.end_effector_rotation_relative": np.zeros(4),
                "state.base_position": np.zeros(3),
                "state.base_rotation": np.zeros(4),
                "state.gripper_qpos": np.zeros(2),
                "annotation.human.task_description": f"instruction for episode {self.episode}",
            }

        def reset(self):
            self.episode += 1
            self.t = 0
            return self.observation(), {"episode_id": 32, "reset_mode": "exact_state_replay"}

        def step(self, action):
            self.actions.append(np.array(action))
            self.t += 1
            done = self.t == 6
            return self.observation(), int(done), done, False, {"success": done}

    class BasePolicy:
        def __init__(self):
            self.inputs = []

        def infer(self, observation, return_vlm_embedding=False):
            assert not return_vlm_embedding
            self.inputs.append(observation)
            # Includes a value outside action bounds to check the existing clipping.
            return {"actions": np.tile(np.array([0.2, 0.4, 0.6, 1.2])[:, None], (1, 12))}

        def get_prefix_rep(self, *args):
            pytest.fail("Base-only evaluation must not request VLM features")

    def forbidden(*args, **kwargs):
        pytest.fail("Base-only evaluation must not build residual observations")

    monkeypatch.setattr(train_utils_sim, "obs_to_qpos", forbidden)
    # Verify the 12-D policy action passed to RoboCasa before its control conversion.
    monkeypatch.setitem(
        sys.modules, "robocasa.utils.env_utils",
        SimpleNamespace(convert_action=lambda action: action),
    )
    variant = AttrDict(
        env="robocasa", seed=0, eval_base_only=True, eval_episodes=2,
        query_freq=2, chunk_len=4, action_dim=action_dim, env_action_dim=12,
        max_timesteps=6, env_max_reward=1, resize_image=16,
        output_dir=str(tmp_path), save_eval_videos=False,
        # Residual settings must have no effect on BC-only evaluation.
        predict_a_exec=True, use_vlm_embedding=True, freeze_vision_encoder=True,
        actor_pop_base_actions=True, critic_pop_base_actions=True,
        requery_base_policy=requery, add_states=True, robocasa_use_right_view=True,
        task_description="stale instruction", pi_05_config="bc-config",
        pi_05_ckpt_dir="/tmp/bc/8000", restore_checkpoint_path=None,
        robocasa_env_name_resolved="PickPlaceCounterToCabinet",
    )
    env, policy = Env(), BasePolicy()
    logger = SimpleNamespace(wandb_logging=False, log=lambda *args, **kwargs: None)
    result = train_utils_sim.perform_control_eval_residual(
        None, env, 1, variant, logger, policy
    )

    expected_values = [0.2, 0.4, 0.2, 0.4, 0.2, 0.4] if requery else [0.2, 0.4, 0.6, 1.0, 0.2, 0.4]
    expected = np.zeros((6, 12))
    expected[:, :action_dim] = np.array(expected_values)[:, None]
    np.testing.assert_allclose(env.actions, np.tile(expected, (2, 1)))
    calls_per_episode = 3 if requery else 2
    assert [item["prompt"] for item in policy.inputs] == (
        ["instruction for episode 1"] * calls_per_episode
        + ["instruction for episode 2"] * calls_per_episode
    )
    assert all("observation/image_right" in item for item in policy.inputs)
    assert result["success_rate"] == 1.0
    assert all(row["delta_norm_mean"] == 0 for row in result["episodes"])
    metadata = json.loads((tmp_path / "summary.json").read_text())
    assert metadata["checkpoint"] == "/tmp/bc/8000"
    assert metadata["eval_base_only"] is True
    assert metadata["pi_05_config"] == "bc-config"
    assert [row["init_state_index"] for row in result["episodes"]] == [32, 32]
    assert [row["task_description"] for row in result["episodes"]] == [
        "instruction for episode 1", "instruction for episode 2",
    ]
    with (tmp_path / "summary.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 3  # two rollouts plus aggregate


@pytest.mark.parametrize("base_only", [False, True])
def test_robocasa_inference_always_uses_current_episode_language(base_only):
    from training.train_utils_sim import obs_to_pi_zero_input

    obs = {
        "video.robot0_agentview_left": np.zeros((4, 4, 3), dtype=np.uint8),
        "video.robot0_eye_in_hand": np.zeros((4, 4, 3), dtype=np.uint8),
        **{f"state.{name}": np.zeros(size) for name, size in (
            ("end_effector_position_relative", 3), ("end_effector_rotation_relative", 4),
            ("base_position", 3), ("base_rotation", 4), ("gripper_qpos", 2),
        )},
        "annotation.human.task_description": "Pick the hot dog from the counter.",
    }
    variant = AttrDict(env="robocasa", task_description="Pick the ice cube.", eval_base_only=base_only)
    assert obs_to_pi_zero_input(obs, variant)["prompt"] == obs["annotation.human.task_description"]
    obs["annotation.human.task_description"] = ""
    with pytest.raises(ValueError, match="natural-language"):
        obs_to_pi_zero_input(obs, variant)


def test_base_only_launcher_skips_midas_setup(tmp_path, monkeypatch):
    from training import train_sim

    def forbidden(*args, **kwargs):
        pytest.fail("Base-only evaluation must not construct or train MIDAS")

    monkeypatch.setattr(train_sim, "MidasLearner", forbidden)
    monkeypatch.setattr(train_sim, "DummyEnvResidual", forbidden)
    monkeypatch.setattr(train_sim, "ReplayBuffer", forbidden)
    monkeypatch.setattr(train_sim, "trajwise_alternating_training_loop_residual", forbidden)
    variant = evaluate_sim.prepare_arguments([
        "--env", "cartpole", "--eval_base_only", "1", "--num_evals", "2",
        "--output_dir", str(tmp_path), "--save_eval_videos", "0",
        "--chunk_len", "2", "--query_freq", "2", "--cartpole_horizon", "4",
        "--resize_image", "16", "--use_vlm_embedding", "1",
        "--freeze_vision_encoder", "1", "--vlm_base_config", "must-not-load",
    ])
    from training.launch_train_sim import parse_args

    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    result = train_sim.main_residual(parse_args(variant))
    assert len(result["episodes"]) == 2
    assert (tmp_path / "summary.json").is_file()


@pytest.mark.parametrize("predict_a_exec", [False, True])
def test_midas_eval_still_executes_residual_actor(predict_a_exec):
    from training import train_utils_sim

    class Env:
        def reset(self):
            return {"image": np.zeros((16, 16, 3), dtype=np.uint8), "state": np.zeros(4)}

        def step(self, action):
            self.action = np.array(action)
            return self.reset(), 0, True, {"success": True}

    class Actor:
        _residual_alpha = 0.5

        def eval_actions(self, observation):
            self.called = True
            assert "base_action" in observation
            return np.array([0.4, 0.4])

    env, actor = Env(), Actor()
    policy = SimpleNamespace(infer=lambda *args, **kwargs: {"actions": np.full((2, 1), 0.1)})
    logger = SimpleNamespace(wandb_logging=False, log=lambda *args, **kwargs: None)
    variant = AttrDict(
        env="cartpole", seed=0, query_freq=2, chunk_len=2, action_dim=1,
        max_timesteps=1, env_max_reward=0, resize_image=16, eval_episodes=1,
        output_dir=None, add_states=True, predict_a_exec=predict_a_exec,
    )
    train_utils_sim.perform_control_eval_residual(actor, env, 1, variant, logger, policy)
    assert actor.called
    np.testing.assert_allclose(env.action, [0.4 if predict_a_exec else 0.3])


def test_prepare_arguments_forces_eval_only_and_preserves_training_flags():
    arguments = evaluate_sim.prepare_arguments(
        ["--env", "cartpole", "--checkpoint_dir", "/tmp/checkpoint"]
    )
    assert arguments[:4] == [
        "--env",
        "cartpole",
        "--checkpoint_dir",
        "/tmp/checkpoint",
    ]
    assert arguments[-2:] == ["--eval_only", "1"]


def test_inline_perturbation_generation_forwards_manifest(tmp_path, monkeypatch):
    input_dir = tmp_path / "bddl" / "libero_10"
    input_dir.mkdir(parents=True)
    config = tmp_path / "language.yaml"
    config.write_text("libero_10: {}\n", encoding="utf-8")
    eval_config = tmp_path / "evaluation.yaml"
    eval_config.write_text(
        yaml.safe_dump(
            {
                "bddl_files_path": str(tmp_path / "bddl"),
                "ood_task_configs": {"language": str(config)},
            }
        ),
        encoding="utf-8",
    )
    generated = tmp_path / "generated"
    generated.mkdir()
    calls = []

    def fake_generate_suite(**kwargs):
        calls.append(kwargs)
        return generated

    monkeypatch.setattr("training.perturbation.generate_suite", fake_generate_suite)
    arguments = evaluate_sim.prepare_arguments(
        [
            "--checkpoint_dir",
            "/tmp/checkpoint",
            "--eval_config_path",
            str(eval_config),
            "--use_language",
            "1",
            "--perturb_seed",
            "7",
        ]
    )

    assert calls[0]["input_dir"] == input_dir
    assert calls[0]["flags"].language is True
    assert calls[0]["seed"] == 7
    assert arguments[-4:] == [
        "--suite_manifest",
        str(generated / "suite_meta.json"),
        "--eval_only",
        "1",
    ]


def test_position_samples_stay_inside_requested_disk():
    import numpy as np

    rng = np.random.RandomState(3)
    samples = [sample_disk_perturbation(rng, 0.05) for _ in range(100)]
    assert all(sample[2] == 0 for sample in samples)
    assert all(np.linalg.norm(sample[:2]) <= 0.05 for sample in samples)


def test_known_moka_pot_task_has_auto_position_targets():
    objects = get_perturb_objects(
        "KITCHEN_SCENE8_put_both_moka_pots_on_the_stove"
    )
    assert objects == ["moka_pot_1", "moka_pot_2"]
