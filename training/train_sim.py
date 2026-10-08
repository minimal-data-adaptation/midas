#! /usr/bin/env python
"""Main MIDAS training script for LIBERO, RoboCasa, and CartPole smoke tests.

This script sets up the Residual RL training pipeline where:
- A frozen base policy (Pi-0.5) produces base action chunks
- Residual policy predicts residual actions in environment action space
- Executed actions are: a_exec = clip(base_action + alpha * delta, -1, 1)

MIDAS is the only supported learning algorithm.
"""

import os
# Tell XLA to use Triton GEMM, this improves steps/sec by ~30% on some GPUs
xla_flags = os.environ.get('XLA_FLAGS', '')
xla_flags += ' --xla_gpu_triton_gemm_any=True'
os.environ['XLA_FLAGS'] = xla_flags

import pathlib

import jax
from midas.agents.midas.midas_learner import MidasLearner
from midas.utils.general_utils import add_batch_dim
import numpy as np

import gym
from gym.spaces import Dict, Box

# Heavy environment/model imports are deferred for the CartPole smoke path.

from midas.data import ReplayBuffer
from midas.utils.wandb_logger import WandBLogger, create_exp_name
from midas.utils.resume import resolve_resume, sweep_orphan_deltas
from midas.utils.reproducibility import (
    EVAL_SEED_OFFSET,
    decode_rng_state,
    restore_training_rng_state,
    seed_process,
)
import tempfile
from functools import partial
from training.train_utils_sim import (
    trajwise_alternating_training_loop_residual,
    _seed_snapshot_state_from_resume,
)
from jax.experimental.compilation_cache import compilation_cache

home_dir = os.environ['HOME']
compilation_cache.initialize_cache(os.path.join(home_dir, 'jax_compilation_cache'))


def _get_libero_env(task, bddl_file_path, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    from libero.libero.envs import OffScreenRenderEnv
    task_description = task.language
    env_args = {"bddl_file_name": pathlib.Path(bddl_file_path), "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def shard_batch(batch, sharding):
    """Shards a batch across devices along its first dimension."""
    return jax.tree_util.tree_map(
        lambda x: jax.device_put(
            x, sharding.reshape(sharding.shape[0], *((1,) * (x.ndim - 1)))
        ),
        batch,
    )


class DummyEnvResidual(gym.ObservationWrapper):
    """Dummy environment for MIDAS with base_action in observation space.
    
    Observation space includes:
        - pixels: (H, W, 3 * num_cameras, 1)
        - state: (state_dim, 1) [optional]
        - base_action: (chunk_len, action_dim, 1) - base action from frozen policy
        
    Action space:
        - (chunk_len, action_dim) - residual/delta actions in env action space
    """

    def __init__(self, variant):
        self.variant = variant
        self.image_shape = (variant.resize_image, variant.resize_image, 3 * variant.num_cameras, 1)
        self.use_vlm_embedding = variant.get('use_vlm_embedding', False)
        
        # Determine the full environment action dimension. The residual policy
        # may intentionally control only a leading subset (for example, the
        # first 7 of RoboCasa's 12 policy-action dimensions).
        if variant.env == 'libero':
            state_dim = 8
            env_action_dim = 7
        elif variant.env == 'robocasa':
            state_dim = 16
            env_action_dim = 12
        elif variant.env == 'cartpole':
            state_dim = 4
            env_action_dim = 1
        else:
            raise NotImplementedError(f"Unknown env: {variant.env}")

        requested_action_dim = variant.get('action_dim', -1)
        action_dim = requested_action_dim if requested_action_dim > 0 else env_action_dim
        if action_dim > env_action_dim:
            raise ValueError(
                f"--action_dim ({action_dim}) cannot exceed the {variant.env} "
                f"environment action dimension ({env_action_dim})"
            )
        
        chunk_len = variant.chunk_len  # e.g., 10 for Pi-0.5
        query_freq = variant.query_freq  # e.g., 5 or 10
        obs_dict = {}
        
        obs_dict['pixels'] = Box(low=0, high=255, shape=self.image_shape, dtype=np.uint8)
        if self.use_vlm_embedding:
            vlm_embedding_dim = variant.get('vlm_embedding_dim', 2048)
            obs_dict['vlm_embedding'] = Box(
                low=-np.inf, high=np.inf,
                shape=(vlm_embedding_dim, 1),
                dtype=np.float32
            )
        
        if variant.add_states:
            obs_dict['state'] = Box(low=-1.0, high=1.0, shape=(state_dim, 1), dtype=np.float32)
        
        # Base action from frozen policy (clipped to [-1, 1])
        obs_dict['base_action'] = Box(
            low=-1.0, high=1.0, 
            shape=(chunk_len, action_dim, 1), 
            dtype=np.float32
        )
        
        self.observation_space = Dict(obs_dict)
        
        # Action space: residual actions (delta) for the full chunk
        # Shape: (query_freq, action_dim)
        self.action_space = Box(
            low=-1.0, high=1.0, 
            shape=(query_freq, action_dim), 
            dtype=np.float32
        )
        
        # Store both dimensions for rollout composition and training.
        variant.env_action_dim = env_action_dim
        variant.action_dim = action_dim


def main_residual(variant):
    """Main function for MIDAS training."""

    seed_process(variant.seed)
    eval_base_only = variant.get('eval_base_only', False)

    devices = jax.local_devices()
    num_devices = len(devices)
    if not eval_base_only:
        assert variant.batch_size % num_devices == 0
    print('num devices', num_devices)
    if not eval_base_only:
        print('batch size', variant.batch_size)
        # Shard training batches across all devices evenly.
        sharding = jax.sharding.PositionalSharding(devices)
        shard_fn = partial(shard_batch, sharding=sharding)

    kwargs = variant['train_kwargs']
    if kwargs.pop('cosine_decay', False):
        kwargs['decay_steps'] = variant.max_steps

    resume_dir = variant.get('resume_dir', None)
    if resume_dir and variant.get('restore_checkpoint_path', None):
        raise ValueError(
            "--resume_dir and --restore_checkpoint_path are mutually exclusive; "
            "--resume_dir restores agent + buffers + counters automatically, "
            "while --restore_checkpoint_path is a manual agent-only restore."
        )

    keep_every = variant.get('keep_checkpoint_interval', None)
    if keep_every is not None:
        if keep_every <= 0:
            raise ValueError(
                f"--keep_checkpoint_interval must be > 0 (got {keep_every})."
            )
        if variant.checkpoint_interval <= 0:
            raise ValueError(
                "--keep_checkpoint_interval requires --checkpoint_interval > 0; "
                f"got checkpoint_interval={variant.checkpoint_interval}."
            )
        if keep_every % variant.checkpoint_interval != 0:
            raise ValueError(
                f"--keep_checkpoint_interval ({keep_every}) must be divisible by "
                f"--checkpoint_interval ({variant.checkpoint_interval}) so that every "
                f"milestone step is also a save step."
            )

    resume_info = resolve_resume(resume_dir) if resume_dir else None
    if resume_info is not None:
        # Override exp_name with the saved one so wandb run id and outputdir
        # both line up with the original run.
        saved_exp_name = resume_info['train_state']['exp_name']
        if variant.get('exp_name', '') and variant.exp_name != saved_exp_name:
            raise ValueError(
                f"--exp_name={variant.exp_name!r} disagrees with saved "
                f"exp_name={saved_exp_name!r} in {resume_dir}; omit --exp_name on resume."
            )
        variant.exp_name = saved_exp_name

    if not variant.prefix:
        import uuid
        variant.prefix = str(uuid.uuid4().fields[-1])[:5]

    if getattr(variant, 'exp_name', ''):
        expname = variant.exp_name
    elif variant.suffix:
        expname = create_exp_name(variant.prefix, seed=variant.seed) + f"_{variant.suffix}"
    else:
        expname = create_exp_name(variant.prefix, seed=variant.seed)

    experiment_root = os.environ.get("MIDAS_EXP_DIR", "experiments")
    outputdir = (
        variant.output_dir
        if variant.get("eval_only", False) and variant.get("output_dir")
        else os.path.join(experiment_root, expname)
    )
    variant.outputdir = outputdir
    if resume_info is not None:
        # Sanity check: the resolved outputdir should equal the resume_dir the
        # user passed. If it doesn't, the saved exp_name and the resume_dir
        # location have drifted, which would silently misroute the next save.
        resolved = pathlib.Path(outputdir).resolve()
        requested = pathlib.Path(resume_dir).resolve()
        if resolved != requested:
            raise ValueError(
                f"Resolved outputdir {resolved} does not match --resume_dir {requested}. "
                f"Ensure $EXP/<exp_name> matches the resume directory."
            )
    if not os.path.exists(outputdir):
        os.makedirs(outputdir)
    print('writing to output dir ', outputdir)
    
    # Environment setup
    if variant.env == 'libero':
        from midas.utils.libero_utils import (
            get_suite_init_policy,
            get_task_suite,
            load_suite_manifest,
        )

        task_suite_name = (
            load_suite_manifest(variant.suite_manifest)
            if variant.get("suite_manifest")
            else variant.task_suite_name
        )
        task_suite = get_task_suite(task_suite_name)
        if variant.get("libero_task"):
            task_names = task_suite.get_task_names()
            matches = [
                index
                for index, name in enumerate(task_names)
                if name == variant.libero_task
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Task {variant.libero_task!r} was not found exactly once in "
                    f"{task_suite_name!r}. Available tasks: {task_names}"
                )
            task_id = matches[0]
        else:
            task_id = variant.task_id
        task = task_suite.get_task(task_id)
        env, task_description = _get_libero_env(
            task,
            task_suite.get_task_bddl_file_path(task_id),
            224,
            variant.seed,
        )
        eval_env, _ = _get_libero_env(
            task,
            task_suite.get_task_bddl_file_path(task_id),
            224,
            variant.seed + EVAL_SEED_OFFSET,
        )
        # LIBERO uses process-global NumPy rather than an environment-local
        # generator. Leave the process on the training stream after creating
        # the independent evaluation simulator.
        env.seed(variant.seed)
        variant.task_description = task_description
        variant.libero_task_name = task_suite.get_task_names()[task_id]
        variant.task_id = task_id
        variant.task_suite_name_resolved = task_suite_name
        init_policy = get_suite_init_policy(task_suite_name)
        if init_policy in {"builtin", "provided"}:
            variant.eval_init_states = task_suite.get_task_init_states(task_id)
        else:
            variant.eval_init_states = None
        variant.env_max_reward = 1
        variant.max_timesteps = 500
        print(f"Libero environment initialized (suite={task_suite_name}, task_id={task_id}): {task_description}")
    elif variant.env == 'robocasa':
        import gymnasium
        from robocasa.utils.dataset_registry import TASK_SET_REGISTRY
        from robocasa.utils.dataset_registry_utils import get_task_horizon
        import robocasa.wrappers.gym_wrapper  # registers robocasa/ gymnasium envs

        # Load pi config early to read scene restriction for env creation
        from openpi.training import config as openpi_config
        variant._pi_config = openpi_config.get_config(variant.pi_05_config)
        pi_data_config = variant._pi_config.data

        # Resolve robocasa_env_name: may be a task set name or a direct env name
        robocasa_env_name = variant.robocasa_env_name
        if robocasa_env_name in TASK_SET_REGISTRY:
            resolved_names = TASK_SET_REGISTRY[robocasa_env_name]
            assert len(resolved_names) == 1, (
                f"Task set '{robocasa_env_name}' contains {len(resolved_names)} envs "
                f"({resolved_names}); single-task residual RL requires exactly one env. "
                f"Pass a direct env name instead."
            )
            robocasa_env_name = resolved_names[0]
        variant.robocasa_env_name_resolved = robocasa_env_name

        # Build env kwargs with scene restriction from pi config
        env_kwargs = {"seed": variant.seed}
        if hasattr(pi_data_config, 'layout_and_style_ids') and pi_data_config.layout_and_style_ids is not None:
            env_kwargs["split"] = None
            env_kwargs["layout_and_style_ids"] = pi_data_config.layout_and_style_ids
        else:
            env_kwargs["split"] = variant.robocasa_split

        env = gymnasium.make(
            f"robocasa/{robocasa_env_name}",
            **env_kwargs,
        )
        eval_env_kwargs = dict(env_kwargs)
        eval_env_kwargs["seed"] = variant.seed + EVAL_SEED_OFFSET
        eval_env = gymnasium.make(
            f"robocasa/{robocasa_env_name}",
            **eval_env_kwargs,
        )
        obs, info = env.reset()

        # Verify env accepted the scene restriction
        if "layout_and_style_ids" in env_kwargs:
            requested = set(map(tuple, env_kwargs["layout_and_style_ids"]))
            env_ids = getattr(env.unwrapped, "layout_and_style_ids", None)
            if env_ids is not None:
                actual = set(map(tuple, env_ids))
                assert actual == requested, (
                    f"Env scene restriction mismatch: requested {requested}, "
                    f"but env has {actual}. Check that the RoboCasa gym wrapper "
                    f"supports the layout_and_style_ids kwarg."
                )
            else:
                print(
                    "WARNING: Cannot verify env scene restriction — "
                    "env.unwrapped.layout_and_style_ids not found. "
                    "Ensure the RoboCasa wrapper applies the restriction."
                )
        variant.task_description = obs["annotation.human.task_description"]

        # Construct independent reset controllers for online and evaluation
        # environments. They use distinct streams but the same reset distribution.
        eval_init_mode = getattr(pi_data_config, 'eval_init_mode', None)
        eval_controller = None
        variant.robocasa_eval_setup = {
            'reset_mode': eval_init_mode or 'random',
            'layout_and_style_ids': pi_data_config.layout_and_style_ids,
        }
        if eval_init_mode is not None:
            from training.robocasa_eval_reset import RoboCasaEvalResetController
            dataset_path = pathlib.Path(
                getattr(pi_data_config, "eval_dataset_path", None)
                or pi_data_config.data_dirs[0]["path"]
            )
            robot_pose_noise = getattr(variant, "robocasa_eval_robot_pose_noise", None)
            if robot_pose_noise is None:
                robot_pose_noise = getattr(pi_data_config, "eval_robot_pose_noise", 0.0)
            object_pose_noise = getattr(variant, "robocasa_eval_object_pose_noise", None)
            if object_pose_noise is None:
                object_pose_noise = getattr(pi_data_config, "eval_object_pose_noise", 0.0)
            object_ori_noise = getattr(variant, "robocasa_eval_object_ori_noise", None)
            if object_ori_noise is None:
                object_ori_noise = getattr(pi_data_config, "eval_object_ori_noise", 0.0)

            def make_reset_controller(seed):
                return RoboCasaEvalResetController(
                    dataset_path=dataset_path,
                    eval_init_mode=eval_init_mode,
                    layout_and_style_ids=pi_data_config.layout_and_style_ids,
                    eval_pool_episode_ids=getattr(pi_data_config, 'eval_pool_episode_ids', None),
                    eval_pool_fixture_refs=getattr(pi_data_config, 'eval_pool_fixture_refs', None),
                    eval_pool_object_categories=getattr(pi_data_config, 'eval_pool_object_categories', None),
                    keep_robot_pose=getattr(pi_data_config, 'eval_keep_robot_pose', False),
                    robot_pose_noise=robot_pose_noise,
                    object_pose_noise=object_pose_noise,
                    object_ori_noise=object_ori_noise,
                    rng_seed=seed,
                )

            eval_controller = make_reset_controller(variant.seed)
            evaluation_reset_controller = make_reset_controller(
                variant.seed + EVAL_SEED_OFFSET
            )
            env.unwrapped._eval_reset_controller = eval_controller
            eval_env.unwrapped._eval_reset_controller = evaluation_reset_controller
            variant.robocasa_eval_setup.update(
                dataset_path=str(dataset_path),
                episode_ids=evaluation_reset_controller._pool_ids,
                keep_robot_pose=evaluation_reset_controller._keep_robot_pose,
                robot_pose_noise=evaluation_reset_controller._robot_pose_noise,
                object_pose_noise=evaluation_reset_controller._object_pose_noise,
                object_ori_noise=evaluation_reset_controller._object_ori_noise,
            )

        task_horizon = get_task_horizon(robocasa_env_name)
        horizon = int(task_horizon * variant.robocasa_horizon_scale)
        if variant.get('robocasa_horizon_cap', 0) > 0:
            horizon = min(horizon, variant.robocasa_horizon_cap)
        variant.max_timesteps = horizon
        variant.env_max_reward = 1
        env_action_dim = 12
        requested_action_dim = variant.get('action_dim', -1)
        variant.action_dim = (
            requested_action_dim if requested_action_dim > 0 else env_action_dim
        )
        if variant.action_dim > env_action_dim:
            raise ValueError(
                f"--action_dim ({variant.action_dim}) cannot exceed the RoboCasa "
                f"environment action dimension ({env_action_dim})"
            )
        variant.env_action_dim = env_action_dim
        print(f"RoboCasa env: {robocasa_env_name}, horizon={horizon}, "
              f"action_dim={variant.action_dim}/{env_action_dim}, "
              f"task: {variant.task_description}")
    elif variant.env == 'cartpole':
        from envs.cartpole_env import CartPoleEnv
        render_size = variant.resize_image if variant.resize_image > 0 else 100
        env = CartPoleEnv(render_size=render_size, horizon=variant.get('cartpole_horizon', 100))
        env.seed(variant.seed)
        eval_env = CartPoleEnv(render_size=render_size, horizon=variant.get('cartpole_horizon', 100))
        eval_env.seed(variant.seed + EVAL_SEED_OFFSET)
        variant.env_max_reward = 0  # best reward is 0 (theta=0)
        variant.max_timesteps = variant.get('cartpole_horizon', 100)
        variant.task_description = 'Balance the pole upright'
        print("CartPole test environment initialized.")
    else:
        raise NotImplementedError(f"Unknown env: {variant.env}")

    # WandB setup
    group_name = variant.prefix + '_' + variant.launch_group_id
    wandb_output_dir = tempfile.mkdtemp(prefix="wandb_", dir=outputdir)
    if resume_info is not None and variant.wandb:
        saved_run_id = resume_info['train_state'].get('wandb_run_id')
        if not saved_run_id:
            raise ValueError(
                f"resume_dir snapshot at step {resume_info['step']} has no "
                f"wandb_run_id; the original run was either offline or did not "
                f"log to wandb. Cannot guarantee same-run resume."
            )
    else:
        saved_run_id = None

    wandb_logger = WandBLogger(
        variant.wandb, variant, variant.wandb_project,
        experiment_id=expname, output_dir=wandb_output_dir, group_name=group_name,
        resume=resume_info is not None and variant.wandb,
        run_id=saved_run_id,
    )

    if eval_base_only:
        env_action_dim = {'libero': 7, 'robocasa': 12, 'cartpole': 1}[variant.env]
        requested_action_dim = variant.get('action_dim', -1)
        variant.action_dim = requested_action_dim if requested_action_dim > 0 else env_action_dim
        if variant.action_dim > env_action_dim:
            raise ValueError(
                f"--action_dim ({variant.action_dim}) cannot exceed the {variant.env} "
                f"environment action dimension ({env_action_dim})"
            )
        variant.env_action_dim = env_action_dim
    else:
        # Create dummy env for MIDAS observation/action space specs.
        dummy_env = DummyEnvResidual(variant)
        dummy_env.observation_space.seed(variant.seed)
        dummy_env.action_space.seed(variant.seed + 1)
        sample_obs = add_batch_dim(dummy_env.observation_space.sample())
        sample_action = add_batch_dim(dummy_env.action_space.sample())

        print('MIDAS sample obs shapes:', [(k, v.shape) for k, v in sample_obs.items()])
        print('MIDAS sample action shape:', sample_action.shape)
    
    # Load frozen base policy (Pi-0.5 or zero policy for test envs)
    if variant.env == 'cartpole':
        from envs.zero_base_policy import ZeroBasePolicy
        agent_dp = ZeroBasePolicy(
            action_dim=variant.action_dim,
            chunk_len=variant.chunk_len,
            vlm_embedding_dim=variant.get('vlm_embedding_dim', 2048),
            vlm_seq_len=variant.get('vlm_seq_len', 16),
        )
        print(f"Using ZeroBasePolicy for CartPole (action_dim={variant.action_dim}, chunk_len={variant.chunk_len})")
    else:
        from openpi.training import config as openpi_config
        from openpi.policies import policy_config
        from openpi.shared import download
        if variant.env == 'libero':
            config = openpi_config.get_config(variant.pi_05_config)
            checkpoint_dir = download.maybe_download(variant.pi_05_ckpt_dir)
        elif variant.env == 'robocasa':
            config = getattr(variant, '_pi_config', None) or openpi_config.get_config(variant.pi_05_config)
            checkpoint_dir = download.maybe_download(variant.pi_05_ckpt_dir)
        else:
            raise NotImplementedError()
        
        agent_dp = policy_config.create_trained_policy(
            config, checkpoint_dir, seed=variant.seed
        )
        print(f"Loaded frozen Pi-0.5 policy from {checkpoint_dir}")

    if eval_base_only:
        from training.train_utils_sim import perform_control_eval_residual

        print(f"[Eval] base policy only: {variant.pi_05_ckpt_dir or 'ZeroBasePolicy'}")
        try:
            return perform_control_eval_residual(
                None, eval_env, 1, variant, wandb_logger, agent_dp
            )
        finally:
            for candidate in (eval_env, env):
                if hasattr(candidate, 'close'):
                    candidate.close()

    # Optionally load a separate Pi model for VLM prefix representations
    vlm_base_config = variant.get('vlm_base_config', None)
    if vlm_base_config:
        from openpi.training import config as openpi_config
        from openpi.policies import policy_config
        from openpi.shared import download
        from openpi.training.weight_loaders import CheckpointWeightLoader
        vlm_config = openpi_config.get_config(vlm_base_config)
        if not isinstance(vlm_config.weight_loader, CheckpointWeightLoader):
            raise ValueError(
                f"vlm_base_config '{vlm_base_config}' does not have a CheckpointWeightLoader, "
                f"got {type(vlm_config.weight_loader).__name__}"
            )
        vlm_params_path = vlm_config.weight_loader.params_path
        vlm_ckpt_dir = download.maybe_download(vlm_params_path.removesuffix("/params"))
        agent_vlm = policy_config.create_trained_policy(
            vlm_config, vlm_ckpt_dir, seed=variant.seed + 2
        )
        print(f"Loaded separate VLM prefix rep model from config '{vlm_base_config}'")
    else:
        agent_vlm = None

    # Add residual_alpha to kwargs for the learner
    kwargs['residual_alpha'] = variant.residual_alpha
    
    # Get algorithm selection
    algo = variant.get('algo', 'midas')
    
    # Create Residual RL agent based on algorithm
    if algo != 'midas':
        raise ValueError(f"MIDAS is the only supported algorithm, got {algo!r}")
    # MIDAS: Policy-Agnostic RL (Best-of-N + Grad-Q + BC distillation)
    midas_kwargs = {k: v for k, v in kwargs.items() if k not in ['temp_lr', 'init_temperature', 'backup_entropy', 'clip_temp', 'clip_min_temp', 'clip_max_temp', 'target_entropy']}
    midas_kwargs['midas_num_samples'] = variant.get('midas_num_samples', 16)
    midas_kwargs['midas_num_elites'] = variant.get('midas_num_elites', 4)
    midas_kwargs['midas_num_grad_steps'] = variant.get('midas_num_grad_steps', 5)
    midas_kwargs['midas_step_size'] = variant.get('midas_step_size', 0.01)
    midas_kwargs['max_grad_norm'] = variant.get('max_grad_norm', 1.0)
    midas_kwargs['use_huber_loss'] = variant.get('use_huber_loss', False)
    midas_kwargs['huber_delta'] = variant.get('huber_delta', 1.0)
    midas_kwargs['num_critic_updates'] = variant.get('num_critic_updates', 2)
    midas_kwargs['num_actor_updates'] = variant.get('num_actor_updates', 4)
    midas_kwargs['predict_a_exec'] = variant.get('predict_a_exec', False)
    midas_kwargs['log_std_min'] = variant.get('log_std_min', -5.0)
    midas_kwargs['log_std_max'] = variant.get('log_std_max', 2.0)
    midas_kwargs['learn_std'] = variant.get('learn_std', True)
    midas_kwargs['use_vlm_embedding'] = variant.get('use_vlm_embedding', False)
    midas_kwargs['freeze_vision_encoder'] = variant.get('freeze_vision_encoder', False)
    midas_kwargs['bc_reg_coeff'] = variant.get('bc_reg_coeff', 0.0)
    midas_kwargs['bc_on_success_only'] = variant.get('bc_on_success_only', False)
    midas_kwargs['actor_arch'] = variant.get('actor_arch', 'tanh_gaussian')
    midas_kwargs['critic_arch'] = variant.get('critic_arch', 'mlp_ensemble')
    midas_kwargs['mip_t_star'] = variant.get('mip_t_star', 0.9)
    midas_kwargs['mip_noise_std'] = variant.get('mip_noise_std', 0.01)
    midas_kwargs['mip_use_film'] = variant.get('mip_use_film', False)
    midas_kwargs['mip_q_noise_scale'] = variant.get('mip_q_noise_scale', 1.0)
    midas_kwargs['b_o_n'] = variant.get('b_o_n', True)
    midas_kwargs['grad_a_q'] = variant.get('grad_a_q', True)
    midas_kwargs['midas_use_trust_region'] = variant.get(
        'midas_use_trust_region', False
    )
    midas_kwargs['midas_a_star_delta_clip_norm'] = variant.get(
        'midas_a_star_delta_clip_norm', None
    )
    agent = MidasLearner(variant.seed, sample_obs, sample_action, **midas_kwargs)
    print(f"Initialized Residual MIDAS with alpha={variant.residual_alpha}, "
          f"N={midas_kwargs['midas_num_samples']}, K={midas_kwargs['midas_num_elites']}, "
          f"grad_steps={midas_kwargs['midas_num_grad_steps']}, step_size={midas_kwargs['midas_step_size']}, "
          f"trust_region={midas_kwargs['midas_use_trust_region']}, "
          f"actor_arch={midas_kwargs['actor_arch']}, critic_arch={midas_kwargs['critic_arch']}")

    if variant.restore_checkpoint_path is not None:
        print(f"Restoring agent from checkpoint: {variant.restore_checkpoint_path}")
        agent.restore_checkpoint(variant.restore_checkpoint_path)
        print("Checkpoint restored successfully.")

    if variant.get("eval_only", False):
        from training.train_utils_sim import perform_control_eval_residual

        try:
            perform_control_eval_residual(
                agent, eval_env, 1, variant, wandb_logger, agent_dp, agent_vlm=agent_vlm
            )
        finally:
            for candidate in (eval_env, env):
                if hasattr(candidate, "close"):
                    candidate.close()
        return

    # If the agent is caching frozen-encoder features at rollout time, extend
    # the observation space with a ``pixel_features`` Box (sized from the
    # agent's just-probed encoder output dim) BEFORE building the replay
    # buffer so the buffer pre-allocates space for the cached feature stream.
    # The agent itself was init'd from an obs_space without this key, so its
    # ``encode()`` traced through the encoder branch and the encoder params
    # are still allocated for rollout-time feature precomputation; train-time
    # apply receives obs WITH ``pixel_features`` and takes the cached branch.
    if getattr(agent, 'pixel_features_dim', None) is not None:
        feat_dim = agent.pixel_features_dim
        new_obs_dict = dict(dummy_env.observation_space.spaces)
        new_obs_dict['pixel_features'] = Box(
            low=-np.inf, high=np.inf,
            shape=(feat_dim, 1),
            dtype=np.float32,
        )
        dummy_env.observation_space = Dict(new_obs_dict)
        print(f'Extended observation space with pixel_features (dim={feat_dim}) '
              f'for cached frozen-encoder features.')

    # Replay buffer
    online_buffer_size = variant.max_steps // variant.multi_grad_step
    online_replay_buffer = ReplayBuffer(dummy_env.observation_space, dummy_env.action_space, int(online_buffer_size))
    replay_buffer = online_replay_buffer
    replay_buffer.seed(variant.seed)
    
    # Success replay buffer (stores only transitions from successful episodes)
    success_buffer_ratio = variant.get('success_buffer_ratio', 0.0)
    if success_buffer_ratio > 0.0:
        # Smaller capacity since only success data goes here
        success_buffer_size = max(10000, int(online_buffer_size * 0.5))
        success_replay_buffer = ReplayBuffer(
            dummy_env.observation_space, dummy_env.action_space, int(success_buffer_size)
        )
        success_replay_buffer.seed(variant.seed + 1)
        print(f'Created success replay buffer with capacity {success_buffer_size}')
    else:
        success_replay_buffer = None
    
    # Demo data pre-seeding (skipped on resume — restored buffer already includes it)
    demo_hdf5_path = variant.get('demo_hdf5_path', '')
    demo_buffer_path = variant.get('demo_buffer_path', '')
    if not demo_buffer_path:
        demo_buffer_path = os.path.join(outputdir, 'initial_demo_replay_buffer.pkl')

    if resume_info is not None:
        print("Resuming: skipping demo pre-seeding (restored buffer already contains it).")
    elif demo_buffer_path and os.path.exists(demo_buffer_path):
        print(f"Restoring demo buffer from {demo_buffer_path}")
        replay_buffer.restore(demo_buffer_path)
        print(f"Demo buffer restored: {len(replay_buffer)} transitions, "
              f"{replay_buffer._traj_counter} trajectories")
    elif demo_hdf5_path:
        if variant.env == "libero":
            from midas.utils.libero_utils import load_libero_demos_to_buffer

            loader = load_libero_demos_to_buffer
        else:
            from midas.utils.robocasa_utils import load_robocasa_hdf5_demos_to_buffer

            loader = load_robocasa_hdf5_demos_to_buffer
        loader(
            hdf5_path=demo_hdf5_path,
            replay_buffer=replay_buffer,
            agent_dp=agent_dp,
            variant=variant,
            num_demos=variant.get('num_demos', -1),
            task_description=variant.get('task_description', None),
            agent_vlm=agent_vlm,
            **({"agent": agent} if variant.env == "libero" else {}),
        )
    elif (variant.env == 'robocasa'
          and hasattr(pi_data_config, 'eval_pool_episode_ids')
          and pi_data_config.eval_pool_episode_ids is not None):
        from midas.utils.robocasa_utils import load_robocasa_demos_to_buffer
        rc_dataset_path = pathlib.Path(pi_data_config.data_dirs[0]["path"])
        load_robocasa_demos_to_buffer(
            dataset_path=rc_dataset_path,
            replay_buffer=replay_buffer,
            agent_dp=agent_dp,
            variant=variant,
            num_demos=variant.get('num_demos', -1),
            task_description=variant.get('task_description', None),
            pi_data_config=pi_data_config,
            eval_controller=(
                eval_controller
                if 'eval_controller' in dir() else None
            ),
            agent_vlm=agent_vlm,
            agent=agent,
        )

    # Seed success buffer with demo data (demos are successful trajectories).
    # Skipped on resume: the success buffer is restored from disk below.
    if (resume_info is None
            and success_buffer_ratio > 0.0
            and success_replay_buffer is not None
            and len(replay_buffer) > 0):
        demo_size = len(replay_buffer)
        # Copy only the filled portion of each array, not the full capacity
        for key in replay_buffer.data:
            if isinstance(replay_buffer.data[key], dict):
                for sub_key in replay_buffer.data[key]:
                    success_replay_buffer.data[key][sub_key][:demo_size] = (
                        replay_buffer.data[key][sub_key][:demo_size]
                    )
            else:
                success_replay_buffer.data[key][:demo_size] = (
                    replay_buffer.data[key][:demo_size]
                )
        success_replay_buffer.size = demo_size
        success_replay_buffer._traj_counter = replay_buffer._traj_counter
        success_replay_buffer._start = replay_buffer._start
        success_replay_buffer.traj_bounds = dict(replay_buffer.traj_bounds)
        success_replay_buffer._last_traj_indices = (
            replay_buffer._last_traj_indices.copy()
            if replay_buffer._last_traj_indices is not None else None
        )
        print(f'Success buffer seeded: {len(success_replay_buffer)} transitions '
              f'(capacity {success_replay_buffer.capacity})')

    snapshot_state = _seed_snapshot_state_from_resume(resume_info)
    if resume_info is not None:
        format_version = resume_info.get('format_version', 1)
        print(f"Resuming from step {resume_info['step']} ({resume_info['agent_dir']}) [format v{format_version}]")
        agent.restore_checkpoint(resume_info['agent_dir'])

        if format_version >= 2:
            online_deltas = resume_info['online_delta_paths']
            if online_deltas:
                # The first delta must start at traj_lo == 0 so the demo prefix
                # (saved into the very first delta after preseed) is reconstructed.
                first_lo = int(os.path.basename(online_deltas[0]).split('_')[0])
                if first_lo != 0:
                    raise RuntimeError(
                        f"resume online delta chain does not start at traj 0: "
                        f"{online_deltas[0]} — demo prefix would be missing."
                    )
            for path in online_deltas:
                online_replay_buffer.append_delta(path)
            print(f"Online replay buffer restored from {len(online_deltas)} deltas: "
                  f"{len(online_replay_buffer)} transitions, "
                  f"{online_replay_buffer._traj_counter} trajectories")

            success_deltas = resume_info['success_delta_paths']
            if success_deltas:
                if success_replay_buffer is None:
                    raise ValueError(
                        "Saved snapshot contains a success buffer but the current run "
                        "has success_buffer_ratio=0; rerun with success_buffer_ratio > 0."
                    )
                for path in success_deltas:
                    success_replay_buffer.append_delta(path)
                print(f"Success replay buffer restored from {len(success_deltas)} deltas: "
                      f"{len(success_replay_buffer)} transitions")
            elif success_replay_buffer is not None:
                print("WARNING: success buffer enabled but no success deltas in snapshot; "
                      "starting from empty success buffer.")
        else:
            online_replay_buffer.restore(resume_info['online_buffer_path'])
            print(f"Online replay buffer restored (legacy v1): {len(online_replay_buffer)} transitions, "
                  f"{online_replay_buffer._traj_counter} trajectories")
            if resume_info['success_buffer_path'] is not None:
                if success_replay_buffer is None:
                    raise ValueError(
                        "Saved snapshot contains a success buffer but the current run "
                        "has success_buffer_ratio=0; rerun with success_buffer_ratio > 0."
                    )
                success_replay_buffer.restore(resume_info['success_buffer_path'])
                print(f"Success replay buffer restored (legacy v1): {len(success_replay_buffer)} transitions")
            elif success_replay_buffer is not None:
                print("WARNING: success buffer enabled but no success_replay_buffer.pkl in snapshot; "
                      "starting from empty success buffer.")

        encoded_rng_state = resume_info['train_state'].get('reproducibility_state')
        if encoded_rng_state:
            restore_training_rng_state(
                decode_rng_state(encoded_rng_state),
                env=env,
                base_policy=agent_dp,
                replay_buffer=online_replay_buffer,
                success_replay_buffer=success_replay_buffer,
            )
            print("Restored learner-adjacent RNG and environment state.")
        else:
            print(
                "WARNING: legacy snapshot has no reproducibility state; "
                "resume is compatible but cannot match an uninterrupted run exactly."
            )

        start_step = resume_info['step']
        start_total_env_steps = int(resume_info['train_state'].get('total_env_steps', 0))
    else:
        start_step = 0
        start_total_env_steps = 0

    # Clean up delta files from any prior crash whose JSON manifest never
    # landed; safe to call regardless of format because v1 directories live
    # outside the global pool.
    orphans = sweep_orphan_deltas(variant.outputdir)
    if orphans:
        print(f"Removed {len(orphans)} orphan delta files: "
              f"{[os.path.basename(p) for p in orphans[:5]]}{'...' if len(orphans) > 5 else ''}")

    # Start training
    try:
        trajwise_alternating_training_loop_residual(
            variant, agent, env, eval_env, online_replay_buffer, replay_buffer,
            wandb_logger, shard_fn=shard_fn, agent_dp=agent_dp,
            success_replay_buffer=success_replay_buffer,
            agent_vlm=agent_vlm,
            start_step=start_step,
            start_total_env_steps=start_total_env_steps,
            snapshot_state=snapshot_state,
        )
    finally:
        # Drain any in-flight async checkpoint writes before the process exits.
        # _save_resume_snapshot waits internally before its JSON marker, so any
        # save that ran to completion is already durable; this is the belt-and-
        # suspenders pass for an interrupted run or for the case where the loop
        # exited mid-save.
        if hasattr(agent, 'wait_for_checkpoints'):
            agent.wait_for_checkpoints()
        for candidate in (eval_env, env):
            if hasattr(candidate, "close"):
                candidate.close()
