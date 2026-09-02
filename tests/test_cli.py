import pytest

from training.launch_train_sim import parse_args


def test_cartpole_cli_defaults_to_midas_without_wandb():
    variant = parse_args(["--env", "cartpole", "--chunk_len", "2", "--query_freq", "2"])
    assert variant.algo == "midas"
    assert variant.wandb is False
    assert variant.eval_only is False
    assert variant.midas_use_trust_region is False


def test_eval_only_requires_checkpoint():
    with pytest.raises(SystemExit):
        parse_args(["--env", "cartpole", "--eval_only", "1"])


def test_evaluation_aliases_and_artifact_flags():
    variant = parse_args(
        [
            "--env",
            "cartpole",
            "--eval_only",
            "1",
            "--checkpoint_dir",
            "/tmp/checkpoint",
            "--num_evals",
            "3",
            "--output_dir",
            "/tmp/evaluation",
            "--save_eval_videos",
            "0",
        ]
    )
    assert variant.restore_checkpoint_path == "/tmp/checkpoint"
    assert variant.eval_episodes == 3
    assert variant.output_dir == "/tmp/evaluation"
    assert variant.save_eval_videos is False


def test_trust_region_requires_clip_norm():
    with pytest.raises(SystemExit):
        parse_args(["--env", "cartpole", "--midas_use_trust_region", "1"])


def test_trust_region_can_be_enabled_with_clip_norm():
    variant = parse_args(
        [
            "--env",
            "cartpole",
            "--midas_use_trust_region",
            "1",
            "--midas_a_star_delta_clip_norm",
            "0.1",
        ]
    )
    assert variant.midas_use_trust_region is True
    assert variant.midas_a_star_delta_clip_norm == [0.1]


def test_robocasa_right_view_flag_is_boolean():
    variant = parse_args(
        [
            "--env",
            "robocasa",
            "--robocasa_env_name",
            "PrepareCoffee",
            "--pi_05_config",
            "test-config",
            "--pi_05_ckpt_dir",
            "/tmp/test-checkpoint",
            "--robocasa_use_right_view",
            "1",
        ]
    )
    assert variant.robocasa_use_right_view is True


def test_action_dim_is_optional_and_can_be_overridden():
    default_variant = parse_args(
        ["--env", "cartpole", "--chunk_len", "2", "--query_freq", "2"]
    )
    assert default_variant.action_dim == -1

    overridden_variant = parse_args(
        [
            "--env",
            "robocasa",
            "--robocasa_env_name",
            "PrepareCoffee",
            "--pi_05_config",
            "test-config",
            "--pi_05_ckpt_dir",
            "/tmp/test-checkpoint",
            "--action_dim",
            "7",
        ]
    )
    assert overridden_variant.action_dim == 7
