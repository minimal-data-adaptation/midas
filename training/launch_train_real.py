"""Dedicated command-line launcher for real-world YAM MIDAS training."""

from __future__ import annotations

import argparse
import os

import yaml

from midas.real.config import RealRunSpec, sha256_file
from midas.real.rewards import load_action_norm_stats, normalized_trust_region
from midas.utils.launch_util import parse_training_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MIDAS on the YAM bimanual robot")
    parser.add_argument("--task_config", default="")
    parser.add_argument("--algo", default="midas", choices=["midas"])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--prefix", default="yam")
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--output_dir", default="")
    parser.add_argument("--resume_dir", default="")
    parser.add_argument("--restore_checkpoint_path", default="")
    parser.add_argument("--wandb", default=0, type=int)
    parser.add_argument("--wandb_project", default="midas-real")
    parser.add_argument("--launch_group_id", default="")

    parser.add_argument("--server_host", default="localhost")
    parser.add_argument("--server_port", default=8000, type=int)
    parser.add_argument("--server_api_key", default="")
    parser.add_argument("--server_connect_timeout", default=300.0, type=float)
    parser.add_argument("--actor_push_interval", default=0, type=int)
    parser.add_argument("--actor_push_at_traj_boundary", default=1, type=int)

    parser.add_argument("--mock_env", default=0, type=int)
    parser.add_argument("--mock_episode_len", default=8, type=int)
    parser.add_argument("--mock_step_seconds", default=0.0, type=float)
    parser.add_argument("--yam_env_config_path", default="")
    parser.add_argument("--instruction", default="perform the task")
    parser.add_argument("--resize_image", default=224, type=int)
    parser.add_argument("--chunk_len", default=60, type=int)
    parser.add_argument("--query_freq", default=30, type=int)
    parser.add_argument("--max_traj_len", default=2500, type=int)
    parser.add_argument("--label_timeout_seconds", default=300.0, type=float)
    parser.add_argument("--rollout_interactive_success_label", default=1, type=int)

    parser.add_argument("--pi_05_config", default="")
    parser.add_argument("--pi_05_ckpt_dir", default="")
    parser.add_argument("--demo_repo_id", default="")
    parser.add_argument("--demo_data_root", default="")
    parser.add_argument("--demo_norm_stats_path", default="")
    parser.add_argument("--num_demos", default=1, type=int)
    parser.add_argument("--demo_filter_prompt", default="")
    parser.add_argument("--demo_filter_orig_traj_id_6_eq", default=None, type=int)
    parser.add_argument("--demo_filter_orig_traj_id_6_min", default=None, type=int)
    parser.add_argument("--demo_filter_orig_traj_id_6_max", default=None, type=int)
    parser.add_argument("--eval_rollout_repo_id", default="")
    parser.add_argument("--eval_rollout_data_root", default="")
    parser.add_argument("--eval_rollout_norm_stats_path", default="")

    parser.add_argument("--reward_type", default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--num_subtasks", default=1, type=int)
    parser.add_argument("--online_buffer_capacity", default=100_000, type=int)
    parser.add_argument("--success_buffer_ratio", default=0.5, type=float)
    parser.add_argument("--success_buffer_min_size", default=1, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_steps", default=1_000_000, type=int)
    parser.add_argument("--start_online_updates", default=0, type=int)
    parser.add_argument("--num_initial_traj_collect", default=1, type=int)
    parser.add_argument("--multi_grad_step", default=1, type=int)
    parser.add_argument("--num_online_gradsteps_batch", default=-1, type=int)
    parser.add_argument("--bc_warmup_steps", default=5000, type=int)
    parser.add_argument("--bc_warmup_num_critic_updates", default=10, type=int)
    parser.add_argument("--bc_warmup_num_actor_updates", default=1, type=int)
    parser.add_argument("--log_interval", default=100, type=int)
    parser.add_argument("--checkpoint_interval", default=1000, type=int)
    parser.add_argument("--keep_checkpoint_interval", default=20_000, type=int)

    parser.add_argument("--predict_a_exec", default=1, type=int)
    parser.add_argument("--learn_std", default=1, type=int)
    parser.add_argument("--use_vlm_embedding", default=1, type=int)
    parser.add_argument("--freeze_vision_encoder", default=1, type=int)
    parser.add_argument("--vlm_embedding_dim", default=2048, type=int)
    parser.add_argument("--midas_num_samples", default=16, type=int)
    parser.add_argument("--midas_num_elites", default=4, type=int)
    parser.add_argument("--midas_num_grad_steps", default=5, type=int)
    parser.add_argument("--midas_step_size", default=0.01, type=float)
    parser.add_argument("--midas_use_trust_region", default=1, type=int)
    parser.add_argument("--midas_a_star_clip_rad", default=0.2, type=float)
    parser.add_argument("--midas_a_star_delta_clip_norm", nargs=14, type=float)
    parser.add_argument("--b_o_n", default=1, type=int)
    parser.add_argument("--grad_a_q", default=1, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--use_huber_loss", default=0, type=int)
    parser.add_argument("--huber_delta", default=1.0, type=float)
    parser.add_argument("--num_critic_updates", default=2, type=int)
    parser.add_argument("--num_actor_updates", default=4, type=int)
    parser.add_argument("--bc_reg_coeff", default=0.0, type=float)
    parser.add_argument("--bc_on_success_only", default=0, type=int)
    parser.add_argument("--actor_arch", default="tanh_gaussian", choices=["tanh_gaussian", "mip"])
    parser.add_argument(
        "--critic_arch", default="mlp_ensemble", choices=["mlp_ensemble", "mip_ensemble"]
    )
    parser.add_argument("--mip_t_star", default=0.9, type=float)
    parser.add_argument("--mip_noise_std", default=0.01, type=float)
    parser.add_argument("--mip_use_film", default=0, type=int)
    parser.add_argument("--mip_q_noise_scale", default=1.0, type=float)
    return parser


TASK_CONFIG_FIELDS = {
    "instruction",
    "pi_05_config",
    "demo_repo_id",
    "demo_filter_prompt",
    "demo_filter_orig_traj_id_6_eq",
    "demo_filter_orig_traj_id_6_min",
    "demo_filter_orig_traj_id_6_max",
}


def _task_defaults(path: str) -> dict:
    if not path:
        return {}
    with open(path, encoding="utf-8") as handle:
        values = yaml.safe_load(handle) or {}
    if not isinstance(values, dict):
        raise ValueError(f"Real task config must be a mapping: {path}")
    unknown = set(values) - TASK_CONFIG_FIELDS
    if unknown:
        raise ValueError(f"Unknown or machine-specific real task config fields: {sorted(unknown)}")
    return values


TRAIN_DEFAULTS = {
    "actor_lr": 1e-4,
    "critic_lr": 3e-4,
    "hidden_dims": (1024, 1024, 1024, 1024),
    "cnn_features": (64, 64, 64, 64),
    "cnn_strides": (2, 1, 1, 1),
    "cnn_padding": "VALID",
    "latent_dim": 200,
    "discount": 0.999,
    "tau": 0.05,
    "critic_reduction": "mean",
    "dropout_rate": 0.0,
    "aug_next": True,
    "color_jitter": True,
    "use_bottleneck": True,
    "encoder_type": "small",
    "encoder_norm": "group",
    "use_spatial_softmax": True,
    "softmax_temperature": -1.0,
    "num_qs": 10,
    "action_magnitude": 1.0,
    # Three physical cameras are concatenated into one encoder input.
    "num_cameras": 1,
    "critic_pop_base_actions": True,
    "actor_pop_base_actions": False,
}


def parse_args(argv: list[str] | None = None):
    parser = build_parser()
    probe = argparse.ArgumentParser(add_help=False)
    probe.add_argument("--task_config", default="")
    task_args, _ = probe.parse_known_args(argv)
    try:
        parser.set_defaults(**_task_defaults(task_args.task_config))
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if argv is not None:
        parser.parse_args = lambda: argparse.ArgumentParser.parse_args(parser, argv)
    variant, _ = parse_training_args(TRAIN_DEFAULTS, parser)
    for key in (
        "wandb",
        "mock_env",
        "actor_push_at_traj_boundary",
        "rollout_interactive_success_label",
        "predict_a_exec",
        "learn_std",
        "use_vlm_embedding",
        "freeze_vision_encoder",
        "midas_use_trust_region",
        "b_o_n",
        "grad_a_q",
        "use_huber_loss",
        "bc_on_success_only",
        "mip_use_film",
    ):
        variant[key] = bool(variant[key])
    for key in ("aug_next", "color_jitter", "use_bottleneck", "use_spatial_softmax"):
        variant.train_kwargs[key] = bool(variant.train_kwargs[key])
    if not variant.predict_a_exec:
        parser.error("Real-world MIDAS requires --predict_a_exec=1")
    if not 0 < variant.query_freq <= variant.chunk_len <= 60:
        parser.error("Expected 0 < --query_freq <= --chunk_len <= 60")
    if variant.max_traj_len < variant.query_freq:
        parser.error("--max_traj_len must be at least --query_freq")
    if variant.reward_type == "dense" and variant.num_subtasks < 1:
        parser.error("Dense rewards require --num_subtasks >= 1")
    if variant.resume_dir and variant.restore_checkpoint_path:
        parser.error("--resume_dir and --restore_checkpoint_path are mutually exclusive")
    if not 0.0 <= variant.success_buffer_ratio <= 1.0:
        parser.error("--success_buffer_ratio must be in [0, 1]")
    if not variant.mock_env:
        for key in ("yam_env_config_path", "pi_05_config", "pi_05_ckpt_dir"):
            if not variant[key]:
                parser.error(f"--{key} is required for hardware training")
        safe_restore = bool(variant.resume_dir or variant.restore_checkpoint_path)
        safe_warmup = (
            variant.num_demos > 0 or bool(variant.eval_rollout_repo_id)
        ) and variant.bc_warmup_steps > 0
        if not (safe_restore or safe_warmup):
            parser.error(
                "Hardware training requires a restored actor or demonstrations plus BC warmup"
            )
        if not os.path.isfile(variant.yam_env_config_path):
            parser.error(f"YAM environment config is not readable: {variant.yam_env_config_path}")
        if not os.path.isdir(variant.pi_05_ckpt_dir):
            parser.error(f"Pi checkpoint directory does not exist: {variant.pi_05_ckpt_dir}")
    if variant.num_demos > 0:
        for key in ("demo_repo_id", "demo_data_root", "demo_norm_stats_path"):
            if not variant[key]:
                parser.error(f"--{key} is required when --num_demos > 0")
        if not os.path.isdir(variant.demo_data_root):
            parser.error(f"Demo data root does not exist: {variant.demo_data_root}")
        if not os.path.isfile(variant.demo_norm_stats_path):
            parser.error(f"Demo norm stats file does not exist: {variant.demo_norm_stats_path}")
    if variant.eval_rollout_repo_id:
        if not variant.eval_rollout_data_root:
            parser.error("--eval_rollout_data_root is required with --eval_rollout_repo_id")
        if not os.path.isdir(variant.eval_rollout_data_root):
            parser.error(f"Eval rollout data root does not exist: {variant.eval_rollout_data_root}")
        if not variant.eval_rollout_norm_stats_path:
            variant.eval_rollout_norm_stats_path = variant.demo_norm_stats_path
        if not os.path.isfile(variant.eval_rollout_norm_stats_path):
            parser.error(
                f"Eval rollout norm stats file does not exist: {variant.eval_rollout_norm_stats_path}"
            )
    if variant.midas_use_trust_region:
        if variant.midas_a_star_delta_clip_norm is not None:
            cap = variant.midas_a_star_delta_clip_norm
        else:
            if not variant.demo_norm_stats_path:
                parser.error(
                    "Trust-region conversion requires --demo_norm_stats_path or an explicit cap"
                )
            q01, q99 = load_action_norm_stats(variant.demo_norm_stats_path)
            cap = normalized_trust_region(q01, q99, variant.midas_a_star_clip_rad).tolist()
        variant.midas_a_star_delta_clip_norm = cap
    else:
        variant.midas_a_star_delta_clip_norm = None
    return variant


def build_run_spec(variant) -> RealRunSpec:
    norm_path = variant.demo_norm_stats_path or variant.eval_rollout_norm_stats_path
    norm_hash = sha256_file(norm_path) if norm_path else ""
    actor_kwargs = dict(variant.train_kwargs)
    for key in (
        "midas_num_samples",
        "midas_num_elites",
        "midas_num_grad_steps",
        "midas_step_size",
        "b_o_n",
        "grad_a_q",
        "max_grad_norm",
        "use_huber_loss",
        "huber_delta",
        "num_critic_updates",
        "num_actor_updates",
        "learn_std",
        "bc_reg_coeff",
        "bc_on_success_only",
        "actor_arch",
        "critic_arch",
        "mip_t_star",
        "mip_noise_std",
        "mip_use_film",
        "mip_q_noise_scale",
        "freeze_vision_encoder",
    ):
        actor_kwargs[key] = variant[key]
    return RealRunSpec(
        resize_image=variant.resize_image,
        chunk_len=variant.chunk_len,
        query_freq=variant.query_freq,
        use_vlm_embedding=variant.use_vlm_embedding,
        vlm_embedding_dim=variant.vlm_embedding_dim,
        midas_use_trust_region=variant.midas_use_trust_region,
        midas_a_star_delta_clip_norm=(
            tuple(variant.midas_a_star_delta_clip_norm)
            if variant.midas_a_star_delta_clip_norm is not None
            else None
        ),
        reward_type=variant.reward_type,
        num_subtasks=variant.num_subtasks,
        pi_config=variant.pi_05_config,
        pi_checkpoint=variant.pi_05_ckpt_dir,
        norm_stats_sha256=norm_hash,
        actor_kwargs=actor_kwargs,
    )


def main(argv: list[str] | None = None) -> None:
    variant = parse_args(argv)
    from training.train_real import main_real

    main_real(variant, build_run_spec(variant))


if __name__ == "__main__":
    main()
