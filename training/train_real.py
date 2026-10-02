"""Setup and lifecycle for the dedicated real-world MIDAS trainer."""

from __future__ import annotations

import contextlib
import dataclasses
import os
from pathlib import Path
import signal

from midas.real.config import RealRunSpec
from midas.real.rewards import validate_reward_manifest
from midas.utils.resume import gc_old_snapshots, resolve_resume, sweep_orphan_deltas
from midas.utils.reproducibility import (
    capture_training_rng_state,
    decode_rng_state,
    encode_rng_state,
    restore_training_rng_state,
    seed_process,
)
from midas.utils.snapshots import (
    SnapshotState,
    save_buffer_delta,
    seed_snapshot_state_from_resume,
    write_json_manifest,
)


def _output_directory(variant) -> Path:
    if variant.resume_dir:
        output = Path(variant.resume_dir).expanduser().resolve()
        if not output.is_dir():
            raise FileNotFoundError(f"Resume directory does not exist: {output}")
        return output
    if variant.exp_name:
        name = variant.exp_name
    else:
        from midas.utils.wandb_logger import create_exp_name

        name = create_exp_name(variant.prefix, seed=variant.seed)
    root = Path(variant.output_dir or os.environ.get("MIDAS_EXP_DIR", "experiments"))
    output = (root / name).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=False)
    return output


def _manifest(
    *,
    variant,
    spec: RealRunSpec,
    replay,
    success_replay,
    snapshots: SnapshotState,
    step: int,
    total_env_steps: int,
    actor_version: int,
    env,
    base_policy_rng_state=None,
) -> dict:
    def persisted_size(buffer, snapshot_state) -> int:
        count = int(snapshot_state.prev_traj_count)
        if buffer is None or count == 0:
            return 0
        return int(buffer.buffer.traj_bounds[count - 1][1])

    reproducibility_state = capture_training_rng_state(
        env=env,
        replay_buffer=replay,
        success_replay_buffer=success_replay,
    )
    if base_policy_rng_state is not None:
        # The frozen policy lives in the server process, so the trainer asks
        # the server for this state explicitly at each trajectory boundary.
        reproducibility_state["base_policy"] = base_policy_rng_state

    return {
        "format_version": 4,
        "step": int(step),
        "total_env_steps": int(total_env_steps),
        "reward_type": variant.reward_type,
        "num_subtasks": int(variant.num_subtasks),
        "actor_signature": spec.actor_signature,
        "real_run_spec_hash": spec.spec_hash,
        "base_policy_config": spec.pi_config,
        "base_policy_checkpoint": spec.pi_checkpoint,
        "norm_stats_sha256": spec.norm_stats_sha256,
        "predict_a_exec": True,
        "chunk_len": spec.chunk_len,
        "query_freq": spec.query_freq,
        "server_actor_version": int(actor_version),
        "reproducibility_state": encode_rng_state(reproducibility_state),
        "online": {
            "traj_count": snapshots.online.prev_traj_count,
            "size": persisted_size(replay, snapshots.online),
            "delta_files": list(snapshots.online.prev_delta_files),
        },
        "success": {
            "traj_count": snapshots.success.prev_traj_count,
            "size": persisted_size(success_replay, snapshots.success),
            "delta_files": list(snapshots.success.prev_delta_files),
        },
    }


def _restore_replays(info, replay, success_replay) -> None:
    if int(info["format_version"]) < 2:
        replay.buffer.restore(info["online_buffer_path"])
        if info["success_buffer_path"]:
            if success_replay is None:
                raise ValueError("Snapshot has a success buffer but this run disabled it")
            success_replay.buffer.restore(info["success_buffer_path"])
        return
    for path in info["online_delta_paths"]:
        replay.append_delta(path)
    if info["success_delta_paths"] and success_replay is None:
        raise ValueError("Snapshot has success deltas but this run disabled the success buffer")
    for path in info["success_delta_paths"]:
        success_replay.append_delta(path)


def _save_initial_buffer(
    output, variant, spec, replay, success_replay, snapshots, env, client
) -> None:
    with replay.lock:
        save_buffer_delta(replay.buffer, "online", output, snapshots.online)
    if success_replay is not None:
        with success_replay.lock:
            save_buffer_delta(success_replay.buffer, "success", output, snapshots.success)
    manifest = _manifest(
        variant=variant,
        spec=spec,
        replay=replay,
        success_replay=success_replay,
        snapshots=snapshots,
        step=0,
        total_env_steps=0,
        actor_version=-1,
        env=env,
        base_policy_rng_state=client.get_base_rng_state(),
    )
    manifest["phase"] = "pre_bc_warmup"
    write_json_manifest(manifest, output / "initial_buffer.json")


def _bc_warmup(variant, agent, replay, logger) -> int:
    import jax

    steps = int(variant.bc_warmup_steps)
    if steps <= 0:
        return 0
    if not len(replay):
        raise ValueError("BC warmup requested with an empty replay buffer")
    for step in range(steps):
        critic_info = {}
        for _ in range(max(1, int(variant.bc_warmup_num_critic_updates))):
            critic_info = agent.update_critic(replay.sample(int(variant.batch_size)))
        actor_info = {}
        for _ in range(max(1, int(variant.bc_warmup_num_actor_updates))):
            actor_info = agent.update_actor_bc(replay.sample(int(variant.batch_size)))
        if logger is not None and step % int(variant.log_interval) == 0:
            values = {**critic_info, **actor_info}
            logger.log(
                {
                    f"bc_warmup/{key}": jax.device_get(value)
                    for key, value in values.items()
                    if not hasattr(value, "ndim") or value.ndim == 0
                },
                step=step,
            )
    return steps


def main_real(variant, spec: RealRunSpec) -> None:
    seed_process(variant.seed)
    output = _output_directory(variant)
    variant.outputdir = str(output)
    spec_path = output / "real_run_spec.json"
    if spec_path.exists():
        saved_spec = RealRunSpec.read(spec_path)
        if saved_spec.spec_hash != spec.spec_hash:
            legacy_compatible = (
                saved_spec.policy_seed is None
                and saved_spec.spec_hash
                == dataclasses.replace(spec, policy_seed=None).spec_hash
            )
            if not legacy_compatible:
                raise ValueError("Resume configuration does not match real_run_spec.json")
        spec = saved_spec
    else:
        spec.write(spec_path)

    # Publish the immutable trainer/server contract before loading JAX or the
    # optional real data stack. This lets a separately managed server start
    # from the exact contract while the trainer initializes its learner.
    from midas.real.client import RealPolicyClient
    from midas.real.demos import load_lerobot_demos
    from midas.real.learner import create_learner
    from midas.real.replay import ThreadSafeReplayBuffer
    from midas.real.yam_env import create_yam_env
    from midas.utils.wandb_logger import WandBLogger
    from training.train_utils_real import run_async_training

    resume_info = resolve_resume(str(output)) if variant.resume_dir else None
    if resume_info is not None:
        validate_reward_manifest(
            resume_info["train_state"], variant.reward_type, int(variant.num_subtasks)
        )
        manifest = resume_info["train_state"]
        if manifest.get("actor_signature") not in {None, spec.actor_signature}:
            raise ValueError("Resume actor signature does not match this run")
        if manifest.get("real_run_spec_hash") not in {None, spec.spec_hash}:
            raise ValueError("Resume manifest does not match real_run_spec.json")

    env = create_yam_env(variant)
    agent = create_learner(spec, variant.seed)
    replay = ThreadSafeReplayBuffer(
        env.observation_space, env.action_space, variant.online_buffer_capacity
    )
    replay.seed(variant.seed)
    success_replay = None
    if variant.success_buffer_ratio > 0:
        success_replay = ThreadSafeReplayBuffer(
            env.observation_space,
            env.action_space,
            max(10_000, variant.online_buffer_capacity // 2),
        )
        success_replay.seed(variant.seed + 1)

    logger = WandBLogger(
        variant.wandb,
        variant,
        variant.wandb_project,
        output.name,
        output_dir=str(output.parent),
        group_name=variant.launch_group_id,
        resume=resume_info is not None,
        run_id=(resume_info or {}).get("train_state", {}).get("wandb_run_id"),
    )
    snapshots = seed_snapshot_state_from_resume(resume_info)
    start_step = 0
    start_total_env_steps = 0

    def interrupt_on_termination(_signum, _frame):
        raise KeyboardInterrupt("Received SIGTERM")

    previous_sigterm = signal.getsignal(signal.SIGTERM)
    signal.signal(signal.SIGTERM, interrupt_on_termination)
    with contextlib.ExitStack() as cleanup:
        cleanup.callback(signal.signal, signal.SIGTERM, previous_sigterm)
        cleanup.callback(env.close)
        client = cleanup.enter_context(
            RealPolicyClient(
                spec,
                variant.server_host,
                variant.server_port,
                api_key=variant.server_api_key or None,
                connect_timeout=variant.server_connect_timeout,
            )
        )
        if resume_info is not None:
            agent.restore_checkpoint(resume_info["agent_dir"])
            _restore_replays(resume_info, replay, success_replay)
            encoded_rng_state = resume_info["train_state"].get("reproducibility_state")
            if encoded_rng_state:
                reproducibility_state = decode_rng_state(encoded_rng_state)
                restore_training_rng_state(
                    reproducibility_state,
                    env=env,
                    replay_buffer=replay,
                    success_replay_buffer=success_replay,
                )
                if reproducibility_state.get("base_policy") is not None:
                    client.set_base_rng_state(reproducibility_state["base_policy"])
            start_step = int(resume_info["step"])
            start_total_env_steps = int(resume_info["train_state"].get("total_env_steps", 0))
        elif variant.restore_checkpoint_path:
            agent.restore_checkpoint(variant.restore_checkpoint_path)
        elif variant.num_demos > 0 or variant.eval_rollout_repo_id:
            if variant.num_demos > 0:
                load_lerobot_demos(
                    repo_id=variant.demo_repo_id,
                    replay=replay,
                    success_replay=success_replay,
                    client=client,
                    variant=variant,
                    data_root=variant.demo_data_root,
                    norm_stats_path=variant.demo_norm_stats_path,
                    num_demos=variant.num_demos,
                    filter_prompt=variant.demo_filter_prompt or None,
                    filter_orig_traj_id_6_eq=variant.demo_filter_orig_traj_id_6_eq,
                    filter_orig_traj_id_6_min=variant.demo_filter_orig_traj_id_6_min,
                    filter_orig_traj_id_6_max=variant.demo_filter_orig_traj_id_6_max,
                )
            if variant.eval_rollout_repo_id:
                load_lerobot_demos(
                    repo_id=variant.eval_rollout_repo_id,
                    replay=replay,
                    success_replay=success_replay,
                    client=client,
                    variant=variant,
                    data_root=variant.eval_rollout_data_root,
                    norm_stats_path=variant.eval_rollout_norm_stats_path,
                    num_demos=-1,
                )
            _save_initial_buffer(
                output, variant, spec, replay, success_replay, snapshots, env, client
            )
            _bc_warmup(variant, agent, replay, logger)

        actor_version = client.update_actor_state(agent.export_actor_state())
        variant.initial_actor_version = actor_version
        variant.start_step = start_step
        variant.start_total_env_steps = start_total_env_steps
        sweep_orphan_deltas(str(output))

        if resume_info is None and variant.bc_warmup_steps > 0 and variant.checkpoint_interval > 0:
            agent.save_checkpoint(str(output), 0, int(variant.keep_checkpoint_interval))
            agent.wait_for_checkpoints()
            write_json_manifest(
                _manifest(
                    variant=variant,
                    spec=spec,
                    replay=replay,
                    success_replay=success_replay,
                    snapshots=snapshots,
                    step=0,
                    total_env_steps=0,
                    actor_version=actor_version,
                    env=env,
                    base_policy_rng_state=client.get_base_rng_state(),
                ),
                output / "train_state" / "0.json",
            )

        def save_snapshot(step: int, total_env_steps: int, server_actor_version: int) -> None:
            with replay.lock:
                save_buffer_delta(replay.buffer, "online", output, snapshots.online)
            if success_replay is not None:
                with success_replay.lock:
                    save_buffer_delta(success_replay.buffer, "success", output, snapshots.success)
            agent.wait_for_checkpoints()
            write_json_manifest(
                _manifest(
                    variant=variant,
                    spec=spec,
                    replay=replay,
                    success_replay=success_replay,
                    snapshots=snapshots,
                    step=step,
                    total_env_steps=total_env_steps,
                    actor_version=server_actor_version,
                    env=env,
                    base_policy_rng_state=client.get_base_rng_state(),
                ),
                output / "train_state" / f"{step}.json",
            )
            gc_old_snapshots(str(output), step, int(variant.keep_checkpoint_interval))

        try:
            run_async_training(
                variant,
                agent,
                env,
                replay,
                client,
                success_replay=success_replay,
                logger=logger,
                snapshot_callback=save_snapshot,
            )
        finally:
            agent.wait_for_checkpoints()


__all__ = ["main_real"]
