"""Command-line launcher for MIDAS simulation training."""

from __future__ import annotations

import argparse

from midas.utils.launch_util import parse_training_args


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train MIDAS in simulation")
    parser.add_argument("--algo", default="midas", choices=["midas"])
    parser.add_argument("--env", default="libero", choices=["libero", "robocasa", "cartpole"])
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--launch_group_id", default="")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--suffix", default="")
    parser.add_argument("--exp_name", default="")
    parser.add_argument("--wandb_project", default="midas")
    parser.add_argument("--wandb", default=0, type=int, help="Enable Weights & Biases logging")
    parser.add_argument("--eval_only", default=0, type=int)
    parser.add_argument(
        "--eval_episodes",
        "--num_evals",
        dest="eval_episodes",
        default=10,
        type=int,
        help="Number of evaluation rollouts.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Evaluation artifact directory (summary.csv and videos/).",
    )
    parser.add_argument(
        "--save_eval_videos",
        default=1,
        type=int,
        help="Save every evaluation rollout as an MP4 when --output_dir is set.",
    )
    parser.add_argument("--eval_video_fps", default=20, type=int)
    parser.add_argument(
        "--round_robin_init_states",
        default=1,
        type=int,
        help="Cycle through fixed LIBERO init states; 0 samples them reproducibly.",
    )
    parser.add_argument(
        "--pos_perturb_radius",
        default=0.0,
        type=float,
        help="LIBERO object XY perturbation radius in metres; 0 disables it.",
    )
    parser.add_argument(
        "--pos_perturb_objects",
        default="",
        help="Comma-separated LIBERO object names, or 'auto' for known tasks.",
    )
    parser.add_argument(
        "--pos_perturb_settle_secs",
        default=5.0,
        type=float,
        help="MuJoCo settle time after position/yaw perturbation.",
    )
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--eval_interval", default=5000, type=int)
    parser.add_argument("--checkpoint_interval", default=-1, type=int)
    parser.add_argument("--keep_checkpoint_interval", default=None, type=int)
    parser.add_argument("--batch_size", default=16, type=int)
    parser.add_argument("--max_steps", default=1_000_000, type=int)
    parser.add_argument("--start_online_updates", default=1000, type=int)
    parser.add_argument("--multi_grad_step", default=1, type=int)
    parser.add_argument("--num_online_gradsteps_batch", default=-1, type=int)
    parser.add_argument("--resize_image", default=224, type=int)
    parser.add_argument("--query_freq", default=10, type=int)
    parser.add_argument(
        "--action_dim",
        default=-1,
        type=int,
        help=(
            "Residual-policy action dimension. Values greater than zero override "
            "the environment action dimension; executed actions are zero-padded "
            "back to the full environment dimension."
        ),
    )
    parser.add_argument("--add_states", default=1, type=int)
    parser.add_argument(
        "--restore_checkpoint_path",
        "--checkpoint_dir",
        dest="restore_checkpoint_path",
        default=None,
        help="MIDAS run directory or a specific checkpoint directory.",
    )
    parser.add_argument("--resume_dir", default=None)

    parser.add_argument("--pi_05_config", default="")
    parser.add_argument("--pi_05_ckpt_dir", default="")
    parser.add_argument("--reward_type", default="sparse", choices=["sparse", "dense"])
    parser.add_argument("--residual_alpha", default=0.1, type=float)
    parser.add_argument("--chunk_len", default=10, type=int)
    parser.add_argument("--use_zero_residual_initially", default=1, type=int)
    parser.add_argument("--predict_a_exec", default=0, type=int)
    parser.add_argument("--learn_std", default=1, type=int)
    parser.add_argument("--requery_base_policy", default=1, type=int)

    parser.add_argument("--midas_num_samples", default=16, type=int)
    parser.add_argument("--midas_num_elites", default=4, type=int)
    parser.add_argument("--midas_num_grad_steps", default=5, type=int)
    parser.add_argument("--midas_step_size", default=0.01, type=float)
    parser.add_argument(
        "--midas_use_trust_region",
        default=0,
        type=int,
        help="Enable actor-target trust-region clipping (intended for real-world training)",
    )
    parser.add_argument("--midas_a_star_delta_clip_norm", nargs="+", type=float)
    parser.add_argument("--b_o_n", default=1, type=int)
    parser.add_argument("--grad_a_q", default=1, type=int)
    parser.add_argument("--max_grad_norm", default=1.0, type=float)
    parser.add_argument("--use_huber_loss", default=0, type=int)
    parser.add_argument("--huber_delta", default=1.0, type=float)
    parser.add_argument("--num_critic_updates", default=2, type=int)
    parser.add_argument("--num_actor_updates", default=4, type=int)

    parser.add_argument("--bc_reg_coeff", default=0.0, type=float)
    parser.add_argument("--bc_on_success_only", default=0, type=int)
    parser.add_argument("--success_buffer_ratio", default=0.0, type=float)
    parser.add_argument("--success_buffer_min_size", default=100, type=int)
    parser.add_argument("--bc_warmup_steps", default=0, type=int)
    parser.add_argument("--bc_warmup_num_critic_updates", default=10, type=int)
    parser.add_argument("--bc_warmup_num_actor_updates", default=1, type=int)
    parser.add_argument("--demo_hdf5_path", default="")
    parser.add_argument("--num_demos", default=-1, type=int)
    parser.add_argument("--demo_buffer_path", default="")
    parser.add_argument("--demo_bc_warmup_utd", default=-1, type=int)

    parser.add_argument("--use_vlm_embedding", default=0, type=int)
    parser.add_argument("--freeze_vision_encoder", default=0, type=int)
    parser.add_argument("--vlm_embedding_dim", default=2048, type=int)
    parser.add_argument("--vlm_seq_len", default=16, type=int)
    parser.add_argument("--vlm_base_config", default=None)

    parser.add_argument("--task_suite_name", default="libero_10")
    parser.add_argument("--task_id", default=8, type=int)
    parser.add_argument(
        "--libero_task",
        default="",
        help="Task name override; when set, it is resolved within the selected suite.",
    )
    parser.add_argument("--suite_manifest", default=None)
    parser.add_argument("--robocasa_env_name", default="")
    parser.add_argument("--robocasa_split", default="target")
    parser.add_argument("--robocasa_horizon_scale", default=1.5, type=float)
    parser.add_argument("--robocasa_horizon_cap", default=0, type=int)
    parser.add_argument("--robocasa_use_right_view", default=0, type=int)
    parser.add_argument("--cartpole_horizon", default=100, type=int)

    parser.add_argument("--actor_arch", default="tanh_gaussian", choices=["tanh_gaussian", "mip"])
    parser.add_argument("--critic_arch", default="mlp_ensemble", choices=["mlp_ensemble", "mip_ensemble"])
    parser.add_argument("--mip_t_star", default=0.9, type=float)
    parser.add_argument("--mip_noise_std", default=0.01, type=float)
    parser.add_argument("--mip_use_film", default=0, type=int)
    parser.add_argument("--mip_q_noise_scale", default=1.0, type=float)
    return parser


def parse_args(argv: list[str] | None = None):
    train_kwargs = dict(
        actor_lr=1e-4,
        critic_lr=3e-4,
        hidden_dims=(256, 256, 256),
        cnn_features=(64, 64, 64, 64),
        cnn_strides=(2, 1, 1, 1),
        cnn_padding="VALID",
        latent_dim=200,
        discount=0.999,
        tau=0.005,
        critic_reduction="mean",
        dropout_rate=0.0,
        aug_next=True,
        color_jitter=True,
        use_bottleneck=True,
        encoder_type="small",
        encoder_norm="group",
        use_spatial_softmax=True,
        softmax_temperature=-1.0,
        num_qs=10,
        action_magnitude=1.0,
        num_cameras=1,
        critic_pop_base_actions=True,
        actor_pop_base_actions=False,
    )
    parser = build_parser()
    if argv is not None:
        parser.parse_args = lambda: argparse.ArgumentParser.parse_args(parser, argv)
    variant, _ = parse_training_args(train_kwargs, parser)

    for key in (
        "add_states",
        "use_zero_residual_initially",
        "predict_a_exec",
        "learn_std",
        "requery_base_policy",
        "b_o_n",
        "grad_a_q",
        "midas_use_trust_region",
        "use_huber_loss",
        "bc_on_success_only",
        "use_vlm_embedding",
        "freeze_vision_encoder",
        "wandb",
        "eval_only",
        "save_eval_videos",
        "round_robin_init_states",
        "mip_use_film",
        "robocasa_use_right_view",
    ):
        variant[key] = bool(variant[key])
    if variant.demo_bc_warmup_utd <= 0:
        variant.demo_bc_warmup_utd = variant.multi_grad_step
    if variant.resume_dir and variant.restore_checkpoint_path:
        parser.error("--resume_dir and --restore_checkpoint_path are mutually exclusive")
    if variant.env != "cartpole" and not variant.pi_05_config:
        parser.error("--pi_05_config is required for LIBERO and RoboCasa")
    if variant.env != "cartpole" and not variant.pi_05_ckpt_dir:
        parser.error("--pi_05_ckpt_dir is required for LIBERO and RoboCasa")
    if variant.env == "robocasa" and not variant.robocasa_env_name:
        parser.error("--robocasa_env_name is required for RoboCasa")
    if variant.eval_only and not variant.restore_checkpoint_path:
        parser.error("--restore_checkpoint_path is required with --eval_only")
    if variant.eval_episodes <= 0:
        parser.error("--eval_episodes/--num_evals must be greater than zero")
    if variant.pos_perturb_radius < 0:
        parser.error("--pos_perturb_radius cannot be negative")
    if variant.pos_perturb_radius and variant.env != "libero":
        parser.error("--pos_perturb_radius is supported only for LIBERO")
    if (
        variant.midas_use_trust_region
        and variant.midas_a_star_delta_clip_norm is None
    ):
        parser.error(
            "--midas_a_star_delta_clip_norm is required when "
            "--midas_use_trust_region=1"
        )
    if not variant.requery_base_policy and variant.chunk_len % variant.query_freq:
        parser.error("--chunk_len must be divisible by --query_freq")
    return variant


def main(argv: list[str] | None = None) -> None:
    variant = parse_args(argv)
    from training.train_sim import main_residual

    main_residual(variant)


if __name__ == "__main__":
    main()
