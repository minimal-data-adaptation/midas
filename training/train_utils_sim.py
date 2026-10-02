"""Training utilities for MIDAS in simulation.

This module implements the MIDAS training loop where:
- A frozen base policy (Pi-0.5) produces base action chunks
- SAC predicts residual actions in environment action space
- Executed actions are: a_exec = clip(base_action + alpha * delta, -1, 1)

The key difference from DSRL-pi is that SAC operates in env action space,
not diffusion noise space.
"""

import csv
import json
import os
from pathlib import Path
from typing import Optional
from tqdm import tqdm
import numpy as np
import wandb
import jax
import jax.numpy as jnp
from openpi_client import image_tools
import math
import PIL
from midas.data.dataset import concat_recursive
from midas.utils.resume import gc_old_snapshots
from midas.utils.reproducibility import (
    EVAL_SEED_OFFSET,
    capture_component_rng_state,
    capture_process_rng_state,
    capture_training_rng_state,
    encode_rng_state,
    restore_component_rng_state,
    restore_process_rng_state,
    seed_component,
    seed_environment,
    seed_process,
)
from midas.utils.snapshots import (
    SnapshotState,
    _atomic_save,
    _maybe_save_buffer_delta,
    _seed_snapshot_state_from_resume,
)
from flax.core import frozen_dict


def _zero_pad_actions(actions, env_action_dim):
    """Pad a reduced action with trailing zeros for the full environment."""
    actions = np.asarray(actions)
    if actions.shape[-1] == env_action_dim:
        return actions
    if actions.shape[-1] > env_action_dim:
        raise ValueError(
            f"Action dimension {actions.shape[-1]} exceeds environment action "
            f"dimension {env_action_dim}"
        )
    padded = np.zeros(actions.shape[:-1] + (env_action_dim,), dtype=actions.dtype)
    padded[..., :actions.shape[-1]] = actions
    return padded


def _save_resume_snapshot(variant, agent, online_replay_buffer, success_replay_buffer,
                          wandb_logger, step, total_env_steps,
                          snapshot_state: SnapshotState, env=None, agent_dp=None):
    """Persist agent ckpt + replay-buffer deltas + train_state JSON for pre-emption resume.

    Layout under ``variant.outputdir``:
        checkpoint<step>/                                  (Orbax — atomic tmp-dir + rename internally)
        replay_buffer/online/<lo>_<hi>.pkl                 (global delta pool; atomic .tmp + rename;
        replay_buffer/success/<lo>_<hi>.pkl                  cumulative chain across all saves)
        train_state/<step>.json                            (manifest with cumulative delta files and
                                                            reproducibility state; written LAST)

    The strict write order makes resume robust to mid-save pre-emption:
    ``resolve_resume`` keys off ``train_state/<step>.json``, which is renamed
    into place only after every delta listed in the manifest is on disk.
    """
    save_dir = variant.outputdir
    agent.save_checkpoint(save_dir, step, variant.checkpoint_interval)

    _maybe_save_buffer_delta(online_replay_buffer, 'online', save_dir, snapshot_state.online)
    _maybe_save_buffer_delta(success_replay_buffer, 'success', save_dir, snapshot_state.success)

    ts_dir = os.path.join(save_dir, 'train_state')
    os.makedirs(ts_dir, exist_ok=True)
    wandb_run_id = wandb.run.id if (wandb_logger.wandb_logging and wandb.run is not None) else None
    launch_group_id = variant.get('launch_group_id', '')
    group_name = f"{variant.prefix}_{launch_group_id}" if launch_group_id else ''
    reproducibility_state = capture_training_rng_state(
        env=env,
        base_policy=agent_dp,
        replay_buffer=online_replay_buffer,
        success_replay_buffer=success_replay_buffer,
    )
    train_state = {
        'step': int(step),
        'total_env_steps': int(total_env_steps),
        'exp_name': os.path.basename(save_dir.rstrip('/')),
        'wandb_run_id': wandb_run_id,
        'group_name': group_name,
        'format_version': 3,
        'reproducibility_state': encode_rng_state(reproducibility_state),
        'online': {
            'traj_count': int(online_replay_buffer._traj_counter),
            'size': int(online_replay_buffer.size),
            'delta_files': list(snapshot_state.online.prev_delta_files),
        },
        'success': {
            'traj_count': int(success_replay_buffer._traj_counter) if success_replay_buffer is not None else 0,
            'size': int(success_replay_buffer.size) if success_replay_buffer is not None else 0,
            'delta_files': list(snapshot_state.success.prev_delta_files),
        },
    }

    def _write_json(p):
        with open(p, 'w') as f:
            json.dump(train_state, f, indent=2)

    # The agent checkpoint write is async (Orbax AsyncCheckpointer). Block
    # here so the train_state JSON marker — which downstream resume keys off
    # to mean "this snapshot is complete and durable" — is only renamed into
    # place after the checkpoint files actually finish writing.
    agent.wait_for_checkpoints()
    _atomic_save(_write_json, os.path.join(ts_dir, f'{step}.json'))

    gc_old_snapshots(save_dir, current_step=step,
                     keep_every_n=variant.get('keep_checkpoint_interval', None))


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def obs_to_img(obs, variant):
    """Convert raw observation to resized image for DSRL actor/critic."""
    if variant.env == 'libero':
        curr_image = obs["agentview_image"][::-1, ::-1]
    elif variant.env == 'robocasa':
        curr_image = obs["video.robot0_agentview_left"]
    elif variant.env == 'cartpole':
        curr_image = obs["image"]
    else:
        raise NotImplementedError(f"Unknown env: {variant.env}")
    if variant.resize_image > 0:
        curr_image = np.array(PIL.Image.fromarray(curr_image).resize((variant.resize_image, variant.resize_image)))
    return curr_image


def obs_to_pi_zero_input(obs, variant):
    """Convert raw observation to Pi-0/Pi-0.5 input format."""
    if variant.env == 'libero':
        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        
        obs_pi_zero = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": np.concatenate(
                (
                    obs["robot0_eef_pos"],
                    _quat2axisangle(obs["robot0_eef_quat"]),
                    obs["robot0_gripper_qpos"],
                )
            ),
            "prompt": str(variant.task_description),
        }
    elif variant.env == 'robocasa':
        img = np.ascontiguousarray(obs["video.robot0_agentview_left"])
        wrist_img = np.ascontiguousarray(obs["video.robot0_eye_in_hand"])
        img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(img, 224, 224)
        )
        wrist_img = image_tools.convert_to_uint8(
            image_tools.resize_with_pad(wrist_img, 224, 224)
        )
        state = np.concatenate((
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ), axis=0)
        obs_pi_zero = {
            "observation/image": img,
            "observation/wrist_image": wrist_img,
            "observation/state": state,
            "prompt": str(variant.task_description),
        }
        if variant.get("robocasa_use_right_view", False):
            right_img = np.ascontiguousarray(obs["video.robot0_agentview_right"])
            obs_pi_zero["observation/image_right"] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(right_img, 224, 224)
            )
    elif variant.env == 'cartpole':
        # CartPole uses ZeroBasePolicy, so this is never actually called
        # for inference, but we define it for consistency
        obs_pi_zero = {
            "state": obs["state"],
            "image": obs["image"],
        }
    else:
        raise NotImplementedError(f"Unknown env: {variant.env}")
    return obs_pi_zero


def obs_to_qpos(obs, variant):
    """Extract qpos (proprioceptive state) from observation."""
    if variant.env == 'libero':
        qpos = np.concatenate(
            (
                obs["robot0_eef_pos"],
                _quat2axisangle(obs["robot0_eef_quat"]),
                obs["robot0_gripper_qpos"],
            )
        )
    elif variant.env == 'robocasa':
        qpos = np.concatenate((
            obs["state.end_effector_position_relative"],
            obs["state.end_effector_rotation_relative"],
            obs["state.base_position"],
            obs["state.base_rotation"],
            obs["state.gripper_qpos"],
        ), axis=0)
    elif variant.env == 'cartpole':
        qpos = obs["state"]
    else:
        raise NotImplementedError(f"Unknown env: {variant.env}")
    return qpos



def _compute_mc_returns(rewards, gamma):
    """Discounted Monte-Carlo return-to-go for one trajectory's query-level rewards.

    ``mc[t] = rewards[t] + gamma * mc[t+1]`` with ``mc[T-1] = rewards[T-1]``. This
    is a pure within-trajectory return with no bootstrap past truncation.
    """
    rewards = np.asarray(rewards, dtype=np.float32)
    mc = np.zeros_like(rewards)
    running = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        running = rewards[t] + gamma * running
        mc[t] = running
    return mc


def _sample_actor_batch(
    replay_buffer, success_replay_buffer,
    batch_size, success_buffer_ratio,
    success_buffer_min_size, use_success_buffer,
    shard_fn=None,
):
    """Sample a batch for actor updates, optionally mixing in success buffer data.
    
    If success buffer is enabled and has enough data, composes the batch as:
        - (1 - success_buffer_ratio) * batch_size samples from main buffer
        - success_buffer_ratio * batch_size samples from success buffer
    Otherwise, samples entirely from the main buffer.
    
    Args:
        replay_buffer: Main replay buffer (all data).
        success_replay_buffer: Success-only replay buffer (may be None).
        batch_size: Total batch size.
        success_buffer_ratio: Fraction of batch from success buffer (e.g., 0.2).
        success_buffer_min_size: Minimum samples in success buffer before using it.
        use_success_buffer: Whether success buffer mixing is enabled.
        shard_fn: Optional function to shard batch across devices.
        
    Returns:
        FrozenDict batch for actor update.
    """
    # Check if success buffer is ready
    success_buffer_ready = (
        use_success_buffer
        and success_replay_buffer is not None
        and len(success_replay_buffer) >= success_buffer_min_size
    )
    
    if success_buffer_ready:
        success_batch_size = max(1, int(batch_size * success_buffer_ratio))
        main_batch_size = batch_size - success_batch_size
        
        main_batch = replay_buffer.sample(main_batch_size)
        success_batch = success_replay_buffer.sample(success_batch_size)
        
        # Concatenate: concat_recursive handles nested dicts
        mixed = concat_recursive([main_batch, success_batch])
        batch = frozen_dict.freeze(mixed)
    else:
        batch = replay_buffer.sample(batch_size)
    
    if shard_fn is not None:
        batch = shard_fn(batch)
    
    return batch


def trajwise_alternating_training_loop_residual(
    variant, agent, env, eval_env, online_replay_buffer, replay_buffer, wandb_logger,
    perform_control_evals=True, shard_fn=None, agent_dp=None, success_replay_buffer=None,
    agent_vlm=None, start_step=0, start_total_env_steps=0,
    snapshot_state: Optional[SnapshotState] = None,
):
    """Main training loop for MIDAS.
    
    Args:
        variant: Training configuration.
        agent: MIDAS agent (MidasLearner).
        env: Training environment.
        eval_env: Evaluation environment.
        online_replay_buffer: Replay buffer for online data.
        replay_buffer: Main replay buffer (same as online for pure online RL).
        wandb_logger: WandB logger.
        perform_control_evals: Whether to run policy evaluations.
        shard_fn: Function to shard batches across devices.
        agent_dp: Frozen base policy (Pi-0.5 / Pi-0).
        success_replay_buffer: Optional separate buffer for successful trajectories.
    """
    # Success buffer configuration
    success_buffer_ratio = variant.get('success_buffer_ratio', 0.0)
    success_buffer_min_size = variant.get('success_buffer_min_size', 100)
    use_success_buffer = (success_replay_buffer is not None and success_buffer_ratio > 0.0)
    
    if use_success_buffer:
        # Compute split sizes for mixed actor batches
        success_batch_size = max(1, int(variant.batch_size * success_buffer_ratio))
        main_batch_size = variant.batch_size - success_batch_size
        print(f'[Success Buffer] Enabled: ratio={success_buffer_ratio}, '
              f'main_batch={main_batch_size}, success_batch={success_batch_size}, '
              f'min_size={success_buffer_min_size}')

    total_env_steps = start_total_env_steps
    i = start_step
    is_resumed = start_step > 0
    if snapshot_state is None:
        snapshot_state = SnapshotState()
    if is_resumed:
        print(f'[Resume] start_step={start_step}, start_total_env_steps={start_total_env_steps}')
    # BC warmup configuration
    bc_warmup_steps = variant.get('bc_warmup_steps', 0)
    bc_warmup_num_critic_updates = variant.get('bc_warmup_num_critic_updates', 10)
    bc_warmup_num_actor_updates = variant.get('bc_warmup_num_actor_updates', 1)
    if bc_warmup_steps > 0:
        print(f'[BC Warmup] Enabled: warmup_steps={bc_warmup_steps}, '
              f'critic_updates={bc_warmup_num_critic_updates}, '
              f'actor_updates={bc_warmup_num_actor_updates}')
    
    wandb_logger.log({'num_online_samples': 0}, step=i)
    wandb_logger.log({'num_online_trajs': 0}, step=i)
    wandb_logger.log({'env_steps': 0}, step=i)

    # Demo BC warmup: if buffer was pre-seeded with demo data, run BC updates
    # over all demo transitions before any online collection.
    # Skipped on resume — the restored `i` already accounts for these steps.
    demo_buffer_size = len(replay_buffer)
    num_critic_updates = getattr(variant, 'num_critic_updates', 1)
    num_actor_updates = getattr(variant, 'num_actor_updates', 1)
    demo_bc_steps = 0
    if not is_resumed and demo_buffer_size > 0:
        demo_bc_utd = variant.demo_bc_warmup_utd
        demo_bc_steps = demo_buffer_size * demo_bc_utd
        print(f'[Demo BC Warmup] Buffer has {demo_buffer_size} demo transitions, '
              f'running {demo_bc_steps} BC grad steps '
              f'(UTD={demo_bc_utd}, '
              f'critic_updates={bc_warmup_num_critic_updates}, '
              f'actor_updates={bc_warmup_num_actor_updates})')
        for step in tqdm(range(demo_bc_steps), desc='demo BC warmup'):
            critic_info = {}
            for _ in range(bc_warmup_num_critic_updates):
                batch = replay_buffer.sample(variant.batch_size)
                if shard_fn is not None:
                    batch = shard_fn(batch)
                critic_info = agent.update_critic(batch)

            actor_info = {}
            for _ in range(bc_warmup_num_actor_updates):
                actor_batch = replay_buffer.sample(variant.batch_size)
                if shard_fn is not None:
                    actor_batch = shard_fn(actor_batch)
                actor_info = agent.update_actor_bc(actor_batch)

            if step % variant.log_interval == 0:
                combined_info = {**critic_info, **actor_info}
                combined_info = {k: jax.device_get(v) for k, v in combined_info.items()}
                for k, v in combined_info.items():
                    if hasattr(v, 'ndim') and v.ndim == 0:
                        wandb_logger.log({f'demo_bc_warmup/{k}': v}, step=step)
                    elif not hasattr(v, 'ndim'):
                        wandb_logger.log({f'demo_bc_warmup/{k}': v}, step=step)
        print(f'[Demo BC Warmup] Complete.')
        i += demo_bc_steps

    with tqdm(total=variant.max_steps, initial=i) as pbar:
        while i <= variant.max_steps:
            traj = collect_traj_residual(variant, agent, env, i, agent_dp, agent_vlm=agent_vlm)
            traj_id = online_replay_buffer._traj_counter
            add_online_data_to_buffer_residual(variant, traj, online_replay_buffer, success_replay_buffer)
            total_env_steps += traj['env_steps']
            print('online buffer timesteps length:', len(online_replay_buffer))
            print('online buffer num traj:', traj_id + 1)
            if success_replay_buffer is not None:
                print('success buffer timesteps length:', len(success_replay_buffer))
            print('total env steps:', total_env_steps)
            
            if variant.get("num_online_gradsteps_batch", -1) > 0:
                num_gradsteps = variant.num_online_gradsteps_batch
            else:
                num_gradsteps = len(traj["rewards"]) * variant.multi_grad_step

            # UTD ratios: num_critic_updates and num_actor_updates per collected step
            num_critic_updates = getattr(variant, 'num_critic_updates', 1)
            num_actor_updates = getattr(variant, 'num_actor_updates', 1)

            if len(online_replay_buffer) > variant.start_online_updates:
                # Evaluate the untrained residual before the first update.
                if i == demo_bc_steps:
                    print('Performing evaluation for initial checkpoint...')
                    if perform_control_evals:
                        perform_control_eval_residual(agent, eval_env, i, variant, wandb_logger, agent_dp, agent_vlm=agent_vlm)

                for _ in tqdm(range(num_gradsteps), desc='gradsteps', leave=False):
                    # Determine if we're in BC warmup phase
                    in_bc_warmup = (bc_warmup_steps > 0 and i < bc_warmup_steps)

                    # Select update ratios based on phase
                    if in_bc_warmup:
                        curr_num_critic_updates = bc_warmup_num_critic_updates
                        curr_num_actor_updates = bc_warmup_num_actor_updates
                    else:
                        curr_num_critic_updates = num_critic_updates
                        curr_num_actor_updates = num_actor_updates

                    current_ratio = success_buffer_ratio

                    # Critic updates: always TD learning (aggressive during warmup)
                    critic_info = {}
                    for _ in range(curr_num_critic_updates):
                        critic_batch = _sample_actor_batch(
                                replay_buffer, success_replay_buffer,
                                variant.batch_size, current_ratio,
                                success_buffer_min_size, use_success_buffer,
                                shard_fn,
                            )
                        critic_info = agent.update_critic(critic_batch)

                    # Actor updates: BC during warmup, RL after
                    actor_info = {}
                    if in_bc_warmup:
                        # BC warmup: distill base policy actions via MSE
                        for _ in range(curr_num_actor_updates):
                            # Sample synchronously so the replay RNG state in a
                            # checkpoint identifies the exact next batch. The
                            # old prefetch iterator kept two already-sampled
                            # batches in an uncheckpointed host queue.
                            actor_batch = replay_buffer.sample(variant.batch_size)
                            if shard_fn is not None:
                                actor_batch = shard_fn(actor_batch)
                            actor_info = agent.update_actor_bc(actor_batch)
                    else:
                        # MIDAS is trained off-policy from the replay buffers.
                        for _ in range(curr_num_actor_updates):
                            actor_batch = _sample_actor_batch(
                                replay_buffer, success_replay_buffer,
                                variant.batch_size, current_ratio,
                                success_buffer_min_size, use_success_buffer,
                                shard_fn,
                            )
                            actor_info = agent.update_actor(actor_batch)

                    # Combine info for logging
                    update_info = {**critic_info, **actor_info}
                    update_info['residual/alpha'] = float(agent._residual_alpha)
                    update_info['algo'] = agent.algo
                    update_info['bc_warmup/is_warmup'] = 1.0 if in_bc_warmup else 0.0
                    
                    # Log phase transition
                    if bc_warmup_steps > 0 and i == bc_warmup_steps:
                        print(f'\n{"="*60}')
                        print(f'[BC Warmup -> RL] Transitioning at step {i}')
                        print(f'{"="*60}\n')

                    pbar.update()
                    i += 1

                    if i % variant.log_interval == 0:
                        # Only the keys actually consumed by the wandb logging
                        # branches below are worth pulling off device. Anything
                        # with ndim > 2 would be device_get'd and then silently
                        # dropped, which is wasted host sync inside the gradient
                        # loop.
                        def _is_logged(v):
                            if not hasattr(v, 'ndim'):
                                return True
                            return v.ndim <= 2
                        update_info = {
                            k: jax.device_get(v)
                            for k, v in update_info.items() if _is_logged(v)
                        }
                        for k, v in update_info.items():
                            if hasattr(v, 'ndim'):
                                if v.ndim == 0:
                                    wandb_logger.log({f'training/{k}': v}, step=i)
                                elif v.ndim <= 2:
                                    wandb_logger.log_histogram(f'training/{k}', v, i)
                            else:
                                # Scalar value (e.g., residual_alpha)
                                wandb_logger.log({f'training/{k}': v}, step=i)
                        
                        wandb_logger.log({
                            'replay_buffer_size': len(online_replay_buffer),
                            'success_buffer_size': len(success_replay_buffer) if success_replay_buffer is not None else 0,
                            'success_buffer_active': float(
                                use_success_buffer 
                                and success_replay_buffer is not None 
                                and len(success_replay_buffer) >= success_buffer_min_size
                            ),
                            'episode_return (exploration)': traj['episode_return'],
                            'is_success (exploration)': int(traj['is_success']),
                        }, i)

                    if i % variant.eval_interval == 0:
                        wandb_logger.log({'num_online_samples': len(online_replay_buffer)}, step=i)
                        wandb_logger.log({'num_online_trajs': traj_id + 1}, step=i)
                        wandb_logger.log({'env_steps': total_env_steps}, step=i)
                        if perform_control_evals:
                            perform_control_eval_residual(agent, eval_env, i, variant, wandb_logger, agent_dp, agent_vlm=agent_vlm)

                    if variant.checkpoint_interval != -1 and i % variant.checkpoint_interval == 0:
                        _save_resume_snapshot(
                            variant, agent, online_replay_buffer, success_replay_buffer,
                            wandb_logger, step=i, total_env_steps=total_env_steps,
                            snapshot_state=snapshot_state,
                            env=env, agent_dp=agent_dp,
                        )


def add_online_data_to_buffer_residual(variant, traj, online_replay_buffer, success_replay_buffer=None):
    """Add collected trajectory to replay buffer for MIDAS.
    
    Stores:
        - observations with 'base_action' (chunk from frozen policy)
        - actions = delta_actions (residual) or a_exec (if predict_a_exec=True)
        - rewards, masks, discount
        - success_flag: 1.0 if this episode was successful, 0.0 otherwise
        - old_log_probs: log probability of the action under the behavior policy
    
    If success_replay_buffer is provided and the trajectory was successful,
    transitions are also added to the success buffer.
    """
    discount_horizon = variant.query_freq
    actions = np.array(traj['actions'])  # (B, query_freq, action_dim) - delta or a_exec depending on predict_a_exec
    base_actions = np.array(traj['base_actions'])  # (B, chunk_len, action_dim)
    episode_len = len(actions)
    rewards = np.array(traj['rewards'])
    masks = np.array(traj['masks'])
    is_success = float(traj['is_success'])
    old_log_probs = np.array(traj.get('old_log_probs', np.zeros(episode_len)))

    # Preserve return-to-go in the common replay schema for checkpoint parity.
    gamma_q = variant.discount ** discount_horizon
    mc_returns = _compute_mc_returns(rewards, gamma_q)

    for t in range(episode_len):
        obs = traj['observations'][t]
        next_obs = traj['observations'][t + 1]
        
        # Remove batch dimension
        obs = {k: v[0] for k, v in obs.items()}
        next_obs = {k: v[0] for k, v in next_obs.items()}
        
        # Add base_action to observations (with trailing dimension)
        obs['base_action'] = base_actions[t][..., np.newaxis]  # (chunk_len, action_dim, 1)
        if t < episode_len - 1:
            next_obs['base_action'] = base_actions[t + 1][..., np.newaxis]
        else:
            # Last step: use same base_action (will be masked anyway)
            next_obs['base_action'] = base_actions[t][..., np.newaxis]
        
        if not variant.add_states:
            obs.pop('state', None)
            next_obs.pop('state', None)
        
        insert_dict = dict(
            observations=obs,
            next_observations=next_obs,
            actions=actions[t],  # delta_action or a_exec (depending on predict_a_exec)
            next_actions=actions[t + 1] if t < episode_len - 1 else actions[t],
            rewards=rewards[t],
            masks=masks[t],
            discount=variant.discount ** discount_horizon,
            success_flag=is_success,
            old_log_probs=old_log_probs[t],
            mc_returns=mc_returns[t],
        )
        online_replay_buffer.insert(insert_dict)

        if success_replay_buffer is not None and is_success > 0.5:
            success_replay_buffer.insert(insert_dict)

    online_replay_buffer.increment_traj_counter()
    if success_replay_buffer is not None and is_success > 0.5:
        success_replay_buffer.increment_traj_counter()


def collect_traj_residual(variant, agent, env, i, agent_dp=None, agent_vlm=None):
    """Collect a trajectory using MIDAS.
    
    At each query step:
    1. Query frozen Pi-0.5 (with internal random noise) -> base_actions
    2. Build MIDAS observation with base_action
    3. SAC samples delta_actions (residual)
    4. Execute: a_exec = clip(base_actions + alpha * delta_actions, -1, 1)
    
    When requery_base_policy=False, the base policy is queried only every
    chunk_len steps.  The resulting chunk is cached, and the residual policy
    is invoked every query_freq steps on the appropriate slice of the cached
    chunk (chunk_len must be divisible by query_freq).
    
    Args:
        variant: Training configuration (must have residual_alpha).
        agent: MIDAS agent.
        env: Environment.
        i: Current training step (for exploration strategy).
        agent_dp: Frozen base policy (Pi-0.5).
        
    Returns:
        Trajectory dict with observations, base_actions, actions (delta), rewards, etc.
    """
    query_frequency = variant.query_freq
    env_action_dim = variant.get('env_action_dim', variant.action_dim)
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    residual_alpha =  float(agent._residual_alpha)
    chunk_len = variant.chunk_len  # e.g., 10 for Pi-0.5
    predict_a_exec = variant.get('predict_a_exec', False)
    use_vlm_embedding = variant.get('use_vlm_embedding', False)
    # VLM mode and cached-feature mode are mutually exclusive in encode():
    # PixelMultiplexer prioritizes vlm_embedding and silently ignores
    # pixel_features. Suppress the cache compute under VLM mode so we don't
    # burn an extra encoder forward per rollout step for output that's
    # never read.
    cache_pixel_features = (
        variant.get('freeze_vision_encoder', False) and not use_vlm_embedding
    )
    is_mip_actor = variant.get('actor_arch', 'tanh_gaussian') == 'mip'
    requery_base_policy = variant.get('requery_base_policy', True)
    actor_pop_base_actions = variant.get('actor_pop_base_actions', False)
    critic_pop_base_actions = variant.get('critic_pop_base_actions', True)
    skip_infer = actor_pop_base_actions and critic_pop_base_actions

    # Flag to control initial exploration behavior
    use_zero_residual_initially = variant.get('use_zero_residual_initially', True)
    
    # BC warmup: force zero residual for ALL trajectories during warmup
    bc_warmup_steps = variant.get('bc_warmup_steps', 0)
    in_bc_warmup = (bc_warmup_steps > 0 and i < bc_warmup_steps)
    force_zero_residual = (i == 0 and use_zero_residual_initially) or in_bc_warmup

    if 'libero' in variant.env:
        obs = env.reset()
    elif variant.env == 'robocasa':
        obs, _ = env.reset()
    elif variant.env == 'cartpole':
        obs = env.reset()

    image_list = []  # for visualization
    rewards = []
    action_list = []  # delta actions
    base_action_list = []  # base actions from Pi-0.5
    obs_list = []
    old_log_probs_list = []  # retained as zeros for checkpoint schema compatibility
    
    # Cached base actions for requery_base_policy=False mode.
    # When requery_base_policy=True this is refreshed every query_freq (== every residual query).
    cached_base_actions = None  # (chunk_len, action_dim)
    vlm_hidden_state = None  # (W,) — populated per-step when use_vlm_embedding=True

    for t in tqdm(range(max_timesteps)):
        curr_image = obs_to_img(obs, variant)
        qpos = obs_to_qpos(obs, variant)

        # -----------------------------------------------------------------
        # Step A: Decide whether to (re-)query the base policy this timestep
        # -----------------------------------------------------------------
        need_base_query = False
        if requery_base_policy:
            # Original behaviour: query base policy at every residual query step
            need_base_query = (t % query_frequency == 0)
        else:
            # Cached mode: query base policy only every chunk_len steps
            need_base_query = (t % chunk_len == 0)

        if need_base_query:
            assert agent_dp is not None, "Frozen base policy (agent_dp) is required for MIDAS"
            obs_pi_zero = obs_to_pi_zero_input(obs, variant)
            vlm_source = agent_vlm or agent_dp
            if skip_infer and not force_zero_residual:
                # After BC warmup: both actor and critic pop base actions, so only
                # need VLM embedding (no base action generation via infer()).
                cached_base_actions = np.zeros((chunk_len, variant.action_dim))
                if use_vlm_embedding:
                    vlm_hs = vlm_source.get_prefix_rep(obs_pi_zero)
                    vlm_hs = vlm_hs[0]  # hidden_state; kv_cache not used
                    if vlm_hs.ndim == 3 and vlm_hs.shape[0] == 1:
                        vlm_hs = vlm_hs[0]  # (S, W)
                    vlm_hidden_state = np.mean(vlm_hs, axis=0)  # Mean-pool → (W,)
            else:
                # When agent_vlm is set, get VLM embeddings from it separately
                return_vlm = use_vlm_embedding and agent_vlm is None
                infer_result = agent_dp.infer(obs_pi_zero, return_vlm_embedding=return_vlm)
                cached_base_actions = infer_result["actions"][
                    :chunk_len, :variant.action_dim
                ]  # (chunk_len, residual_action_dim)
                if use_vlm_embedding:
                    if agent_vlm is not None:
                        vlm_hs = vlm_source.get_prefix_rep(obs_pi_zero)
                        vlm_hs = vlm_hs[0]
                    else:
                        vlm_hs = infer_result["vlm_embedding"][0]  # hidden_state from (hidden_state, kv_cache)
                    if vlm_hs.ndim == 3 and vlm_hs.shape[0] == 1:
                        vlm_hs = vlm_hs[0]  # (S, W)
                    vlm_hidden_state = np.mean(vlm_hs, axis=0)  # Mean-pool → (W,)

        # -----------------------------------------------------------------
        # Step B: At every query_freq boundary, invoke the residual policy
        # -----------------------------------------------------------------
        if t % query_frequency == 0:
            assert cached_base_actions is not None, (
                "Base policy must be queried before the residual can act"
            )
            
            # Determine which slice of the cached chunk to use for composition.
            # When requery_base_policy=True  → always slice [0 : query_freq] (fresh chunk each time)
            # When requery_base_policy=False → slice depends on position within the cached chunk
            if requery_base_policy:
                slice_start = 0
            else:
                slice_start = (t % chunk_len)  # e.g., 0, query_freq, 2*query_freq, ...
            slice_end = slice_start + query_frequency
            base_actions_slice = cached_base_actions[slice_start:slice_end]  # (query_freq, action_dim)
            
            # Build MIDAS observation — always uses the full cached chunk as context
            base_actions = cached_base_actions  # alias for readability
            
            # VLM embedding: reuse from infer() if base was queried this step,
            # otherwise get a fresh one via get_prefix_rep() (cheap, no action generation)
            if use_vlm_embedding:
                if not need_base_query:
                    # Intermediate query_freq boundary (requery_base_policy=False):
                    # infer() was not called, so get fresh VLM for current obs
                    obs_pi_zero_vlm = obs_to_pi_zero_input(obs, variant)
                    vlm_hs = vlm_source.get_prefix_rep(obs_pi_zero_vlm)  # (hidden_state, kv_cache)
                    vlm_hs = vlm_hs[0]  # hidden_state; kv_cache not used for SAC input
                    if vlm_hs.ndim == 3 and vlm_hs.shape[0] == 1:
                        vlm_hs = vlm_hs[0]  # (S, W)
                    vlm_hidden_state = np.mean(vlm_hs, axis=0)  # Mean-pool → (W,)
                # else: vlm_hidden_state was already set in Step A from infer()
                obs_dict = {
                    'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    'vlm_embedding': vlm_hidden_state[np.newaxis, ..., np.newaxis],  # (1, W, 1)
                    'base_action': base_actions[np.newaxis, ..., np.newaxis],
                }
                if variant.add_states:
                    obs_dict['state'] = qpos[np.newaxis, ..., np.newaxis]
            elif variant.add_states:
                obs_dict = {
                    'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    'state': qpos[np.newaxis, ..., np.newaxis],
                    'base_action': base_actions[np.newaxis, ..., np.newaxis],  # (1, chunk_len, action_dim, 1)
                }
            else:
                obs_dict = {
                    'pixels': curr_image[np.newaxis, ..., np.newaxis],
                    'base_action': base_actions[np.newaxis, ..., np.newaxis],
                }

            if cache_pixel_features:
                # Cache the frozen encoder's output once per residual query so
                # the train graph can read pre-computed features and skip the
                # full encoder forward. Stored shape mirrors the VLM-embedding
                # convention: (1, D, 1) at rollout time → squeezed to (B, D)
                # inside ``PixelMultiplexer.encode``.
                feat = agent.compute_pixel_features(obs_dict['pixels'])
                obs_dict['pixel_features'] = np.asarray(feat)[..., np.newaxis]

            # Sample actions from SAC
            if force_zero_residual:
                # Zero residual: evaluate base policy (used for first traj or BC warmup)
                delta_actions = np.zeros((query_frequency, variant.action_dim))
                
                if in_bc_warmup:
                    log_prob = 0.0
                    if t == 0:
                        print(f"[BC warmup] Forcing zero residual (log_prob={log_prob:.4f})")
                else:
                    log_prob = 0.0
                    if t == 0:
                        print(f"[t={t}] Using zero residual (initial eval) (log_prob={log_prob:.4f})")
                
                # Compose executed action (base only)
                actions = np.clip(base_actions_slice, -1.0, 1.0)
            else:
                # Use deterministic (mode) actions for rollouts.
                # Exploration comes from the stochastic base policy, not from
                # sampling noise in the residual — sampling adds jitter.
                actions_flat = agent.eval_actions(obs_dict)
                log_prob = 0.0
                raw_actions = np.reshape(actions_flat, (query_frequency, variant.action_dim))
                
                # NaN guard: if action contains NaN/Inf, replace with zeros
                if not np.all(np.isfinite(raw_actions)):
                    print(f"[WARNING] NaN/Inf detected in actions at t={t}, replacing with zeros")
                    raw_actions = np.nan_to_num(raw_actions, nan=0.0, posinf=0.0, neginf=0.0)
                    log_prob = 0.0  # invalidated by NaN replacement
                
                if predict_a_exec:
                    # Actor predicts a_exec directly
                    actions = np.clip(raw_actions, -1.0, 1.0)
                    delta_actions = raw_actions  # store raw actor output (which IS a_exec)
                else:
                    # Original: actor predicts delta, compose a_exec
                    delta_actions = raw_actions
                    actions = np.clip(base_actions_slice + residual_alpha * delta_actions, -1.0, 1.0)
            
            # Store for replay buffer
            if predict_a_exec:
                action_list.append(actions)  # Store a_exec
            else:
                action_list.append(delta_actions)  # Store residual
            base_action_list.append(base_actions)  # Store full cached chunk for replay
            obs_list.append(obs_dict)
            old_log_probs_list.append(log_prob)  # Store log_prob from behavior policy
            
            # Log residual stats occasionally
            if t == 0:
                if predict_a_exec:
                    delta_from_base = actions - base_actions_slice
                    delta_norm = np.linalg.norm(delta_from_base)
                else:
                    delta_norm = np.linalg.norm(delta_actions)
                base_norm = np.linalg.norm(base_actions)
                print(f"[t={t}] base_norm={base_norm:.4f}, delta_norm={delta_norm:.4f}, alpha={residual_alpha}, predict_a_exec={predict_a_exec}, requery_base_policy={requery_base_policy}")
     
        action_t = actions[t % query_frequency]
        action_t_env = _zero_pad_actions(action_t, env_action_dim)

        if 'libero' in variant.env:
            obs, reward, done, _ = env.step(action_t_env)
        elif variant.env == 'robocasa':
            from robocasa.utils.env_utils import convert_action
            env_action = convert_action(action_t_env)
            obs, reward, terminated, truncated, info = env.step(env_action)
            done = terminated or truncated or bool(info.get("success", False))
        elif variant.env == 'cartpole':
            obs, reward, done, info = env.step(action_t_env)

        rewards.append(reward)
        image_list.append(curr_image)
        if done:
            break

    # Add last observation
    curr_image = obs_to_img(obs, variant)
    qpos = obs_to_qpos(obs, variant)
    
    # For last obs, we need a base_action - use the last one (will be masked)
    last_base_action = base_action_list[-1] if base_action_list else np.zeros((chunk_len, variant.action_dim))
    if use_vlm_embedding:
        # Get fresh VLM embedding for last obs (cheap, no action generation)
        obs_pi_zero = obs_to_pi_zero_input(obs, variant)
        vlm_hidden_state = (agent_vlm or agent_dp).get_prefix_rep(obs_pi_zero)  # (hidden_state, kv_cache)
        vlm_hidden_state = vlm_hidden_state[0]  # First is hidden state, second is kv_cache (not used)
        if vlm_hidden_state.ndim == 3 and vlm_hidden_state.shape[0] == 1:
            vlm_hidden_state = vlm_hidden_state[0]
        # Mean-pool over sequence tokens → (W,)
        vlm_hidden_state = np.mean(vlm_hidden_state, axis=0)  # (W,)
        obs_dict = {
            'pixels': curr_image[np.newaxis, ..., np.newaxis],
            'vlm_embedding': vlm_hidden_state[np.newaxis, ..., np.newaxis],  # (1, W, 1)
            'base_action': last_base_action[np.newaxis, ..., np.newaxis],
        }
        if variant.add_states:
            obs_dict['state'] = qpos[np.newaxis, ..., np.newaxis]
    else:
        obs_dict = {
            'pixels': curr_image[np.newaxis, ..., np.newaxis],
            'state': qpos[np.newaxis, ..., np.newaxis],
            'base_action': last_base_action[np.newaxis, ..., np.newaxis],
        }
    if not variant.add_states:
        obs_dict.pop('state', None)
    if cache_pixel_features:
        feat = agent.compute_pixel_features(obs_dict['pixels'])
        obs_dict['pixel_features'] = np.asarray(feat)[..., np.newaxis]
    obs_list.append(obs_dict)
    image_list.append(curr_image)
    
    # Per episode stats
    rewards = np.array(rewards)
    episode_return = np.sum(rewards[rewards != None])
    if variant.env == 'cartpole':
        # CartPole: success if most recent info shows success (pole was upright)
        is_success = bool(info.get('success', 0))
    elif variant.env == 'robocasa':
        is_success = bool(info.get("success", False))
    else:
        is_success = (reward == env_max_reward)
    print(f'Rollout Done: {episode_return=}, Success: {is_success}')
    
    reward_type = variant.get('reward_type', 'sparse')
    query_steps = len(action_list)
    if reward_type == 'dense':
        # Keep raw environment rewards; set terminal mask on done
        rewards = np.array(rewards, dtype=np.float32)
        if is_success:
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            masks = np.ones(query_steps)
    else:
        # Sparse -1/0 reward for SAC training
        if is_success:
            rewards = np.concatenate([-np.ones(query_steps - 1), [0]])
            masks = np.concatenate([np.ones(query_steps - 1), [0]])
        else:
            rewards = -np.ones(query_steps)
            masks = np.ones(query_steps)

    return {
        'observations': obs_list,
        'actions': action_list,  # delta actions (if predict_a_exec=False) or a_exec (if predict_a_exec=True)
        'base_actions': base_action_list,  # base actions from Pi-0.5
        'rewards': rewards,
        'masks': masks,
        'is_success': is_success,
        'episode_return': episode_return,
        'images': image_list,
        'env_steps': t + 1,
        'old_log_probs': old_log_probs_list,  # log probs from behavior policy
    }


def perform_control_eval_residual(agent, env, i, variant, wandb_logger, agent_dp=None, agent_vlm=None):
    """Run evaluation on an isolated, repeatable random stream.

    The frozen policy is intentionally shared with training because loading a
    second VLA model can exceed GPU memory. Its RNG and all process-global RNGs
    are restored afterward, so changing evaluation cadence cannot change later
    training samples or base actions.
    """

    process_state = capture_process_rng_state()
    policy_state = capture_component_rng_state(agent_dp)
    vlm_policy_state = (
        capture_component_rng_state(agent_vlm)
        if agent_vlm is not None and agent_vlm is not agent_dp
        else None
    )
    eval_seed = int(variant.seed) + EVAL_SEED_OFFSET
    try:
        seed_process(eval_seed)
        seed_environment(env, eval_seed)
        seed_component(agent_dp, eval_seed)
        if agent_vlm is not None and agent_vlm is not agent_dp:
            seed_component(agent_vlm, eval_seed + 2)
        return _perform_control_eval_residual(
            agent, env, i, variant, wandb_logger, agent_dp, agent_vlm
        )
    finally:
        if agent_vlm is not None and agent_vlm is not agent_dp:
            restore_component_rng_state(agent_vlm, vlm_policy_state)
        restore_component_rng_state(agent_dp, policy_state)
        restore_process_rng_state(process_state)


def _perform_control_eval_residual(agent, env, i, variant, wandb_logger, agent_dp=None, agent_vlm=None):
    """Evaluate MIDAS policy.
    
    Uses the same logic as collect_traj_residual but without exploration noise.
    When requery_base_policy=False, the base policy is queried every chunk_len
    steps and the cached chunk is sliced for each query_freq window.
    """
    query_frequency = variant.query_freq
    env_action_dim = variant.get('env_action_dim', variant.action_dim)
    print(f'[Eval] query frequency: {query_frequency}')
    max_timesteps = variant.max_timesteps
    env_max_reward = variant.env_max_reward
    residual_alpha =  float(agent._residual_alpha)
    chunk_len = variant.chunk_len
    predict_a_exec = variant.get('predict_a_exec', False)
    use_vlm_embedding = variant.get('use_vlm_embedding', False)
    cache_pixel_features = (
        variant.get('freeze_vision_encoder', False) and not use_vlm_embedding
    )
    requery_base_policy = variant.get('requery_base_policy', True)
    actor_pop_base_actions = variant.get('actor_pop_base_actions', False)
    critic_pop_base_actions = variant.get('critic_pop_base_actions', True)
    skip_infer = actor_pop_base_actions and critic_pop_base_actions

    output_dir = Path(variant.output_dir).expanduser() if variant.get('output_dir') else None
    videos_dir = output_dir / 'videos' if output_dir is not None else None
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        if variant.get('save_eval_videos', True):
            videos_dir.mkdir(parents=True, exist_ok=True)

    init_state_rng = np.random.RandomState(variant.seed)
    perturb_rng = None
    perturb_objects = None
    if variant.env == 'libero' and variant.get('pos_perturb_radius', 0.0) > 0:
        from training.evaluation.position_perturbation import get_perturb_objects

        object_spec = variant.get('pos_perturb_objects', '')
        if not object_spec or object_spec == 'auto':
            perturb_objects = get_perturb_objects(variant.libero_task_name)
        else:
            perturb_objects = [name.strip() for name in object_spec.split(',') if name.strip()]
        if not perturb_objects:
            raise ValueError('--pos_perturb_objects did not contain any object names')
        perturb_rng = np.random.RandomState(variant.seed + 999)
        print(
            f"[Eval] position perturbation: radius={variant.pos_perturb_radius}m, "
            f"objects={perturb_objects}"
        )

    episode_returns = []
    highest_rewards = []
    success_rates = []
    episode_lens = []

    # Track residual statistics across evaluation
    all_delta_norms = []
    all_base_norms = []
    all_clipping_rates = []
    episode_rows = []

    # Track at most one video per eval: the first successful rollout, or the
    # last rollout if none succeed.
    first_success_video = None
    first_success_rollout_id = None
    last_video = None
    last_rollout_id = None

    for rollout_id in range(variant.eval_episodes):
        if 'libero' in variant.env:
            obs = env.reset()
            init_states = variant.get("eval_init_states", None)
            if init_states is not None:
                if variant.get('round_robin_init_states', True):
                    init_index = rollout_id % len(init_states)
                else:
                    init_index = init_state_rng.randint(len(init_states))
                init_state = init_states[init_index]
                obs = env.set_init_state(init_state)
            else:
                init_index = None
            if perturb_rng is not None:
                from training.evaluation.position_perturbation import perturb_object_positions

                obs = perturb_object_positions(
                    env,
                    object_names=perturb_objects,
                    radius=variant.pos_perturb_radius,
                    rng=perturb_rng,
                    settle_secs=variant.get('pos_perturb_settle_secs', 5.0),
                )
        elif variant.env == 'robocasa':
            obs, _ = env.reset()
            init_index = None
        elif variant.env == 'cartpole':
            obs = env.reset()
            init_index = None

        image_list = []
        rewards = []
        rollout_delta_norms = []
        rollout_base_norms = []
        rollout_clipping_rates = []
        
        # Cached base actions for requery_base_policy=False mode
        cached_base_actions = None
        vlm_hidden_state = None
        
        for t in tqdm(range(max_timesteps)):
            curr_image = obs_to_img(obs, variant)

            # -----------------------------------------------------------
            # Step A: Decide whether to (re-)query the base policy
            # -----------------------------------------------------------
            need_base_query = False
            if requery_base_policy:
                need_base_query = (t % query_frequency == 0)
            else:
                need_base_query = (t % chunk_len == 0)

            if need_base_query:
                assert agent_dp is not None
                vlm_source = agent_vlm or agent_dp
                qpos = obs_to_qpos(obs, variant)
                obs_pi_zero = obs_to_pi_zero_input(obs, variant)
                if skip_infer and i != 0:
                    # After initial eval: only need VLM embedding, no base actions
                    cached_base_actions = np.zeros((chunk_len, variant.action_dim))
                    if use_vlm_embedding:
                        vlm_hs = vlm_source.get_prefix_rep(obs_pi_zero)
                        vlm_hs = vlm_hs[0]
                        if vlm_hs.ndim == 3 and vlm_hs.shape[0] == 1:
                            vlm_hs = vlm_hs[0]
                        vlm_hidden_state = np.mean(vlm_hs, axis=0)
                else:
                    return_vlm = use_vlm_embedding and agent_vlm is None
                    infer_result = agent_dp.infer(obs_pi_zero, return_vlm_embedding=return_vlm)
                    cached_base_actions = infer_result["actions"][
                        :chunk_len, :variant.action_dim
                    ]
                    if use_vlm_embedding:
                        if agent_vlm is not None:
                            vlm_hs = vlm_source.get_prefix_rep(obs_pi_zero)
                            vlm_hs = vlm_hs[0]
                        else:
                            vlm_hs = infer_result["vlm_embedding"][0]
                        if vlm_hs.ndim == 3 and vlm_hs.shape[0] == 1:
                            vlm_hs = vlm_hs[0]
                        vlm_hidden_state = np.mean(vlm_hs, axis=0)

            # -----------------------------------------------------------
            # Step B: At every query_freq boundary, invoke residual policy
            # -----------------------------------------------------------
            if t % query_frequency == 0:
                assert cached_base_actions is not None
                # Need qpos for obs_dict even if base wasn't re-queried this step
                if not need_base_query:
                    qpos = obs_to_qpos(obs, variant)
                
                # Determine which slice of the cached chunk to use
                if requery_base_policy:
                    slice_start = 0
                else:
                    slice_start = (t % chunk_len)
                slice_end = slice_start + query_frequency
                base_actions_slice = cached_base_actions[slice_start:slice_end]
                
                base_actions = cached_base_actions  # alias for obs building
                
                # VLM embedding: reuse from infer() if base was queried this step,
                # otherwise get a fresh one via get_prefix_rep() (cheap, no action generation)
                if use_vlm_embedding:
                    if not need_base_query:
                        # Intermediate query_freq boundary (requery_base_policy=False):
                        # infer() was not called, so get fresh VLM for current obs
                        obs_pi_zero_vlm = obs_to_pi_zero_input(obs, variant)
                        vlm_hs = vlm_source.get_prefix_rep(obs_pi_zero_vlm)  # (hidden_state, kv_cache)
                        vlm_hs = vlm_hs[0]  # hidden_state; kv_cache not used for SAC input
                        if vlm_hs.ndim == 3 and vlm_hs.shape[0] == 1:
                            vlm_hs = vlm_hs[0]  # (S, W)
                        vlm_hidden_state = np.mean(vlm_hs, axis=0)  # Mean-pool → (W,)
                    # else: vlm_hidden_state was already set in Step A from infer()
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'vlm_embedding': vlm_hidden_state[np.newaxis, ..., np.newaxis],  # (1, W, 1)
                        'base_action': base_actions[np.newaxis, ..., np.newaxis],
                    }
                    if variant.add_states:
                        obs_dict['state'] = qpos[np.newaxis, ..., np.newaxis]
                elif variant.add_states:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'state': qpos[np.newaxis, ..., np.newaxis],
                        'base_action': base_actions[np.newaxis, ..., np.newaxis],
                    }
                else:
                    obs_dict = {
                        'pixels': curr_image[np.newaxis, ..., np.newaxis],
                        'base_action': base_actions[np.newaxis, ..., np.newaxis],
                    }

                if cache_pixel_features:
                    # Same caching path as the training rollout — required so
                    # eval-time obs structure matches what the agent traced
                    # against during training (cached-feature branch).
                    feat = agent.compute_pixel_features(obs_dict['pixels'])
                    obs_dict['pixel_features'] = np.asarray(feat)[..., np.newaxis]

                if i == 0:
                    # Initial evaluation: zero residual to test base policy
                    delta_actions = np.zeros((query_frequency, variant.action_dim))
                    actions = np.clip(base_actions_slice, -1.0, 1.0)
                else:
                    # SAC samples residual (deterministic: use mean)
                    actions_flat = agent.eval_actions(obs_dict)  # Use eval_actions for deterministic
                    raw_actions = np.reshape(actions_flat, (query_frequency, variant.action_dim))
                    
                    # NaN guard: if action contains NaN/Inf, replace with zeros
                    if not np.all(np.isfinite(raw_actions)):
                        print(f"[WARNING] NaN/Inf detected in eval actions at t={t}, replacing with zeros")
                        raw_actions = np.nan_to_num(raw_actions, nan=0.0, posinf=0.0, neginf=0.0)
                    
                    if predict_a_exec:
                        # Actor predicts a_exec directly
                        actions = np.clip(raw_actions, -1.0, 1.0)
                        delta_actions = actions - base_actions_slice
                    else:
                        # Original: actor predicts delta, compose a_exec
                        delta_actions = raw_actions
                        actions = np.clip(base_actions_slice + residual_alpha * delta_actions, -1.0, 1.0)
                
                # Track statistics
                delta_norm = np.linalg.norm(delta_actions.flatten())
                base_norm = np.linalg.norm(base_actions.flatten())
                clipping_rate = np.mean(np.abs(actions) > 0.999)
                all_delta_norms.append(delta_norm)
                all_base_norms.append(base_norm)
                all_clipping_rates.append(clipping_rate)
                rollout_delta_norms.append(delta_norm)
                rollout_base_norms.append(base_norm)
                rollout_clipping_rates.append(clipping_rate)
              
            action_t = actions[t % query_frequency]
            action_t_env = _zero_pad_actions(action_t, env_action_dim)

            if 'libero' in variant.env:
                obs, reward, done, _ = env.step(action_t_env)
            elif variant.env == 'robocasa':
                from robocasa.utils.env_utils import convert_action
                env_action = convert_action(action_t_env)
                obs, reward, terminated, truncated, eval_info = env.step(env_action)
                done = terminated or truncated or bool(eval_info.get("success", False))
            elif variant.env == 'cartpole':
                obs, reward, done, eval_info = env.step(action_t_env)

            rewards.append(reward)
            image_list.append(curr_image)
            if done:
                break

        # Per episode stats
        episode_lens.append(t + 1)
        rewards = np.array(rewards)
        episode_return = np.sum(rewards)
        episode_returns.append(episode_return)
        episode_highest_reward = np.max(rewards)
        highest_rewards.append(episode_highest_reward)
        if variant.env == 'cartpole':
            is_success = bool(eval_info.get('success', 0))
        elif variant.env == 'robocasa':
            is_success = bool(eval_info.get("success", False))
        else:
            is_success = (reward == env_max_reward)
        success_rates.append(is_success)

        episode_rows.append({
            'rollout_id': rollout_id,
            'init_state_index': '' if init_index is None else init_index,
            'is_success': int(is_success),
            'episode_return': float(episode_return),
            'episode_len': t + 1,
            'highest_reward': float(episode_highest_reward),
            'delta_norm_mean': float(np.mean(rollout_delta_norms)),
            'base_norm_mean': float(np.mean(rollout_base_norms)),
            'clipping_rate_mean': float(np.mean(rollout_clipping_rates)),
        })
                
        print(f'Rollout {rollout_id}: {episode_return=}, Success: {is_success}')
        video = np.stack(image_list).transpose(0, 3, 1, 2)
        last_video = video
        last_rollout_id = rollout_id
        if is_success and first_success_video is None:
            first_success_video = video
            first_success_rollout_id = rollout_id
        if videos_dir is not None and variant.get('save_eval_videos', True):
            import imageio.v2 as imageio

            video_path = videos_dir / f'rollout_{rollout_id:04d}.mp4'
            imageio.mimwrite(video_path, image_list, fps=variant.eval_video_fps)
            print(f'[Eval] video: {video_path}')

    # Log a single video per eval: the first successful rollout if any,
    # otherwise the last rollout.
    if first_success_video is not None:
        sampled_video = first_success_video
        sampled_rollout_id = first_success_rollout_id
        sampled_is_success = True
    else:
        sampled_video = last_video
        sampled_rollout_id = last_rollout_id
        sampled_is_success = False
    if sampled_video is not None and wandb_logger.wandb_logging:
        wandb_logger.log({'eval_video/sample': wandb.Video(sampled_video, fps=50)}, step=i)
        wandb_logger.log({'eval_video/sample_rollout_id': sampled_rollout_id}, step=i)
        wandb_logger.log({'eval_video/sample_is_success': int(sampled_is_success)}, step=i)

    # Log aggregate statistics
    success_rate = np.mean(np.array(success_rates))
    avg_return = np.mean(episode_returns)
    avg_episode_len = np.mean(episode_lens)

    summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
    wandb_logger.log({'evaluation/avg_return': avg_return}, step=i)
    wandb_logger.log({'evaluation/success_rate': success_rate}, step=i)
    wandb_logger.log({'evaluation/avg_episode_len': avg_episode_len}, step=i)
    
    # Log residual-specific evaluation metrics
    wandb_logger.log({'evaluation/delta_norm_mean': np.mean(all_delta_norms)}, step=i)
    wandb_logger.log({'evaluation/delta_norm_std': np.std(all_delta_norms)}, step=i)
    wandb_logger.log({'evaluation/base_norm_mean': np.mean(all_base_norms)}, step=i)
    wandb_logger.log({'evaluation/clipping_rate_mean': np.mean(all_clipping_rates)}, step=i)
    
    for r in range(env_max_reward + 1):
        more_or_equal_r = (np.array(highest_rewards) >= r).sum()
        more_or_equal_r_rate = more_or_equal_r / variant.eval_episodes
        wandb_logger.log({f'evaluation/Reward >= {r}': more_or_equal_r_rate}, step=i)
        summary_str += f'Reward >= {r}: {more_or_equal_r}/{variant.eval_episodes} = {more_or_equal_r_rate*100}%\n'

    if output_dir is not None:
        summary_path = output_dir / 'summary.csv'
        fieldnames = list(episode_rows[0])
        with summary_path.open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(episode_rows)
            writer.writerow({
                'rollout_id': 'AGGREGATE',
                'init_state_index': '',
                'is_success': float(success_rate),
                'episode_return': float(avg_return),
                'episode_len': float(avg_episode_len),
                'highest_reward': float(np.mean(highest_rewards)),
                'delta_norm_mean': float(np.mean(all_delta_norms)),
                'base_norm_mean': float(np.mean(all_base_norms)),
                'clipping_rate_mean': float(np.mean(all_clipping_rates)),
            })
        metadata = {
            'checkpoint': str(variant.restore_checkpoint_path),
            'suite': variant.get('task_suite_name_resolved', variant.get('task_suite_name')),
            'task_id': variant.get('task_id'),
            'task_name': variant.get('libero_task_name'),
            'task_description': variant.get('task_description'),
            'suite_manifest': variant.get('suite_manifest'),
            'episodes': variant.eval_episodes,
            'success_rate': float(success_rate),
            'average_return': float(avg_return),
            'average_episode_len': float(avg_episode_len),
            'position_perturbation': {
                'radius': float(variant.get('pos_perturb_radius', 0.0)),
                'objects': perturb_objects or [],
            },
        }
        with (output_dir / 'summary.json').open('w', encoding='utf-8') as handle:
            json.dump(metadata, handle, indent=2)
            handle.write('\n')
        print(f'[Eval] summary: {summary_path}')

    print(summary_str)
    return {
        'episodes': episode_rows,
        'success_rate': float(success_rate),
        'average_return': float(avg_return),
        'average_episode_len': float(avg_episode_len),
    }
