"""Collection and asynchronous update utilities for real-world MIDAS."""

from __future__ import annotations

import logging
import queue
import threading
from typing import Callable

import jax
import numpy as np
from flax.core import frozen_dict

from midas.data.dataset import concat_recursive
from midas.real.observations import policy_observation, training_observation
from midas.real.operator import cbreak_stdin, poll_label, read_final_label


LOGGER = logging.getLogger(__name__)


def _mc_returns(rewards: np.ndarray, discount: float) -> np.ndarray:
    result = np.zeros_like(rewards, dtype=np.float32)
    running = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        running = float(rewards[index]) + discount * running
        result[index] = running
    return result


def collect_trajectory(
    variant,
    env,
    client,
    trajectory_id: int,
    *,
    interactive: bool,
    stop_event: threading.Event | None = None,
) -> dict | None:
    """Collect query-aligned transitions, dropping incomplete action chunks."""

    query_freq = int(variant.query_freq)
    chunk_len = int(variant.chunk_len)
    max_steps = int(variant.max_traj_len)
    dense = variant.reward_type == "dense"
    observations = []
    base_actions = []
    learned_actions = []
    completion_chunks: list[int] = []
    actor_versions: list[int] = []
    env_steps = 0
    status = None

    env.reset()
    with cbreak_stdin():
        while env_steps + query_freq <= max_steps:
            if stop_event is not None and stop_event.is_set():
                return None
            raw = env.get_observation()
            response = client.infer(
                policy_observation(
                    raw, variant.instruction, variant.resize_image, source_color="bgr"
                )
            )
            physical = np.asarray(response["actions"], dtype=np.float32)
            base = np.asarray(response["base_action"], dtype=np.float32)
            learned = np.asarray(response["a_exec_norm"], dtype=np.float32)
            feature = response.get("vlm_embedding") if variant.use_vlm_embedding else None
            current = training_observation(
                raw, base[:chunk_len], feature, variant.resize_image, source_color="bgr"
            )

            chunk_status = None
            complete_chunk = True
            for offset in range(query_freq):
                if stop_event is not None and stop_event.is_set():
                    return None
                env.step(physical[offset])
                env_steps += 1
                key_status = poll_label() if interactive else None
                if key_status in {"aborted", "retry", "failure"}:
                    status = key_status
                    complete_chunk = False
                    break
                if key_status == "success":
                    chunk_status = "success"
                    if not dense or len(completion_chunks) + 1 >= int(variant.num_subtasks):
                        status = "success"
                        # Finish this chunk so success credit is attached to the
                        # transition that executed the successful action.
                    else:
                        completion_chunks.append(len(learned_actions))
            if not complete_chunk:
                break
            observations.append({key: value[np.newaxis] for key, value in current.items()})
            base_actions.append(base[:chunk_len])
            learned_actions.append(learned[:query_freq])
            actor_versions.append(int(response.get("actor_version", -1)))
            if chunk_status == "success" and status == "success":
                completion_chunks.append(len(learned_actions) - 1)
                break
            if not interactive and bool(getattr(env.raw_env, "is_done", lambda: False)()):
                status = "success"
                break

    if status is None:
        status = read_final_label(variant.label_timeout_seconds) if interactive else "failure"
    if status in {"aborted", "retry"}:
        return {
            "discard": True,
            "status": status,
            "env_steps": env_steps,
            "trajectory_id": trajectory_id,
        }
    if not learned_actions:
        return {
            "discard": True,
            "status": "empty",
            "env_steps": env_steps,
            "trajectory_id": trajectory_id,
        }

    # Capture the query-boundary next observation and its base context.
    raw = env.get_observation()
    response = client.infer(
        policy_observation(raw, variant.instruction, variant.resize_image, source_color="bgr")
    )
    next_feature = response.get("vlm_embedding") if variant.use_vlm_embedding else None
    final_observation = training_observation(
        raw,
        np.asarray(response["base_action"])[:chunk_len],
        next_feature,
        variant.resize_image,
        source_color="bgr",
    )
    observations.append({key: value[np.newaxis] for key, value in final_observation.items()})

    success = status == "success"
    rewards = np.full(len(learned_actions), -1.0, dtype=np.float32)
    masks = np.ones(len(learned_actions), dtype=np.float32)
    if dense:
        for index in set(completion_chunks[:-1] if success else completion_chunks):
            if 0 <= index < len(rewards):
                rewards[index] += 1.0
        if success:
            rewards[-1] += 5.0
            masks[-1] = 0.0
    elif success:
        rewards[-1] = 0.0
        masks[-1] = 0.0

    return {
        "discard": False,
        "status": status,
        "is_success": success,
        "observations": observations,
        "base_actions": np.asarray(base_actions, dtype=np.float32),
        "actions": np.asarray(learned_actions, dtype=np.float32),
        "rewards": rewards,
        "masks": masks,
        "env_steps": env_steps,
        "episode_return": float(np.sum(rewards)),
        "trajectory_id": trajectory_id,
        "completed_subtasks": len(completion_chunks),
        "subtask_completion_chunks": completion_chunks,
        "server_actor_version_first": actor_versions[0] if actor_versions else -1,
        "server_actor_version_last": actor_versions[-1] if actor_versions else -1,
        "server_actor_version_changed": len(set(actor_versions)) > 1,
    }


def trajectory_transitions(variant, trajectory: dict) -> list[dict]:
    actions = np.asarray(trajectory["actions"])
    bases = np.asarray(trajectory["base_actions"])
    rewards = np.asarray(trajectory["rewards"], dtype=np.float32)
    masks = np.asarray(trajectory["masks"], dtype=np.float32)
    gamma = float(variant.discount) ** int(variant.query_freq)
    returns = _mc_returns(rewards, gamma)
    output = []
    for index in range(len(actions)):
        observation = {
            key: value[0].copy() for key, value in trajectory["observations"][index].items()
        }
        next_observation = {
            key: value[0].copy() for key, value in trajectory["observations"][index + 1].items()
        }
        observation["base_action"] = bases[index][..., np.newaxis]
        if index + 1 < len(bases):
            next_observation["base_action"] = bases[index + 1][..., np.newaxis]
        output.append(
            {
                "observations": observation,
                "next_observations": next_observation,
                "actions": actions[index],
                "next_actions": actions[index + 1] if index + 1 < len(actions) else actions[index],
                "rewards": rewards[index],
                "masks": masks[index],
                "discount": np.float32(gamma),
                "success_flag": np.float32(trajectory["is_success"]),
                "old_log_probs": np.float32(0.0),
                "mc_returns": returns[index],
            }
        )
    return output


def insert_trajectory(variant, trajectory: dict, replay, success_replay=None) -> None:
    transitions = trajectory_transitions(variant, trajectory)
    replay.insert_trajectory(transitions)
    if success_replay is not None and trajectory["is_success"]:
        success_replay.insert_trajectory(transitions)


def _mixed_batch(replay, success_replay, variant):
    ratio = float(variant.success_buffer_ratio)
    if (
        success_replay is None
        or ratio <= 0
        or len(success_replay) < int(variant.success_buffer_min_size)
    ):
        return replay.sample(int(variant.batch_size))
    success_size = max(1, int(int(variant.batch_size) * ratio))
    main_size = int(variant.batch_size) - success_size
    if main_size == 0:
        return success_replay.sample(success_size)
    combined = concat_recursive([replay.sample(main_size), success_replay.sample(success_size)])
    return frozen_dict.freeze(combined)


def run_async_training(
    variant,
    agent,
    env,
    replay,
    client,
    *,
    success_replay=None,
    logger=None,
    snapshot_callback: Callable[[int, int, int], None] | None = None,
) -> None:
    """Run one collector thread and the learner update loop in the caller."""

    stop = threading.Event()
    # One completed trajectory may wait for the learner. Bounded backpressure
    # prevents a fast mock (or a faster-than-training robot) from growing the
    # replay indefinitely while an update is compiling.
    lengths: queue.Queue[tuple[int, threading.Event | None]] = queue.Queue(maxsize=1)
    state = {
        "trajectories": 0,
        "env_steps": int(getattr(variant, "start_total_env_steps", 0)),
        "error": None,
    }
    state_lock = threading.Lock()

    def collector() -> None:
        trajectory_id = 0
        try:
            while not stop.is_set():
                trajectory = collect_trajectory(
                    variant,
                    env,
                    client,
                    trajectory_id,
                    interactive=bool(variant.rollout_interactive_success_label),
                    stop_event=stop,
                )
                if trajectory is None:
                    return
                if trajectory.get("discard"):
                    continue
                insert_trajectory(variant, trajectory, replay, success_replay)
                trajectory_id += 1
                with state_lock:
                    state["trajectories"] += 1
                    state["env_steps"] += int(trajectory["env_steps"])
                boundary_ack = (
                    threading.Event() if bool(variant.actor_push_at_traj_boundary) else None
                )
                while not stop.is_set():
                    try:
                        lengths.put((len(trajectory["actions"]), boundary_ack), timeout=0.2)
                        break
                    except queue.Full:
                        continue
                if boundary_ack is not None:
                    while not stop.is_set() and not boundary_ack.wait(timeout=0.2):
                        pass
        except BaseException as error:
            LOGGER.exception("Real collector failed")
            with state_lock:
                state["error"] = error
            stop.set()
            try:
                lengths.put_nowait((-1, None))
            except queue.Full:
                pass

    thread = threading.Thread(target=collector, name="midas-real-collector", daemon=False)
    thread.start()
    step = int(getattr(variant, "start_step", 0))
    last_version = int(getattr(variant, "initial_actor_version", -1))
    try:
        while step < int(variant.max_steps) and not stop.is_set():
            try:
                length, boundary_ack = lengths.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                if length < 0:
                    break
                with state_lock:
                    enough = state["trajectories"] >= int(variant.num_initial_traj_collect)
                if not enough or len(replay) <= int(variant.start_online_updates):
                    continue
                updates = int(variant.num_online_gradsteps_batch)
                if updates <= 0:
                    updates = length * max(1, int(variant.multi_grad_step))
                for _ in range(updates):
                    if step >= int(variant.max_steps) or stop.is_set():
                        break
                    critic_info = {}
                    for _ in range(max(1, int(variant.num_critic_updates))):
                        critic_info = agent.update_critic(replay.sample(int(variant.batch_size)))
                    actor_info = {}
                    for _ in range(max(1, int(variant.num_actor_updates))):
                        actor_info = agent.update_actor(
                            _mixed_batch(replay, success_replay, variant)
                        )
                    step += 1
                    if (
                        int(variant.actor_push_interval) > 0
                        and step % int(variant.actor_push_interval) == 0
                    ):
                        last_version = client.update_actor_state(agent.export_actor_state())
                    if logger is not None and step % int(variant.log_interval) == 0:
                        values = {**critic_info, **actor_info}
                        payload = {
                            f"training/{key}": jax.device_get(value)
                            for key, value in values.items()
                            if not hasattr(value, "ndim") or value.ndim == 0
                        }
                        payload["training/actor_version_pushed"] = last_version
                        payload["training/replay_buffer_size"] = len(replay)
                        logger.log(payload, step=step)
                    if (
                        int(variant.checkpoint_interval) > 0
                        and step % int(variant.checkpoint_interval) == 0
                    ):
                        agent.save_checkpoint(
                            variant.outputdir, step, int(variant.keep_checkpoint_interval)
                        )
                        if snapshot_callback is not None:
                            with state_lock:
                                snapshot_callback(step, state["env_steps"], last_version)
                if bool(variant.actor_push_at_traj_boundary):
                    last_version = client.update_actor_state(agent.export_actor_state())
            finally:
                if boundary_ack is not None:
                    boundary_ack.set()
    finally:
        stop.set()
        try:
            client.disarm()
        except Exception:
            LOGGER.exception("Could not disarm real policy server during shutdown")
        thread.join(timeout=30)
        env.close()
        with state_lock:
            error = state["error"]
        if thread.is_alive():
            raise RuntimeError("Real collector did not stop within 30 seconds")
        if error is not None:
            raise error


__all__ = [
    "collect_trajectory",
    "insert_trajectory",
    "run_async_training",
    "trajectory_transitions",
]
