"""Residual MIDAS (Policy-Agnostic RL) Learner for Pixel Observations.

This module implements a Residual MIDAS agent where:
- A frozen base policy (e.g., Pi-0.5) produces base action chunks
- The actor predicts residual actions in environment action space
- Executed actions are: a_exec = clip(base_action + alpha * delta, -1, 1)
- Critic learns Q(s, a_exec) via TD learning (standard)
- Actor is trained by:
  1. Sampling N actions from the actor
  2. Evaluating with Q, keeping top-K elites
  3. Refining elites via gradient ascent on Q w.r.t. action
  4. Distilling the best refined action back into the actor via MSE (BC loss)

The actor NEVER differentiates through the critic. The critic only provides
stop-gradient targets, avoiding issues with tanh squashing, action clipping, etc.
"""

from flax.training import checkpoints
import pathlib
import numpy as np
import functools
from typing import Dict, Optional, Sequence, Tuple, Union, Any

import jax
import jax.numpy as jnp
import optax
from flax.core.frozen_dict import FrozenDict
from flax.training import train_state

from midas.agents.agent import Agent, get_batch_stats
from midas.data.augmentations import batched_random_crop, color_transform
from midas.networks.encoders.networks import Encoder, PixelMultiplexer
from midas.networks.encoders.impala_encoder import ImpalaEncoder, SmallerImpalaEncoder
from midas.networks.encoders.resnet_encoderv1 import ResNet18, ResNet34, ResNetSmall
from midas.networks.encoders.resnet_encoderv2 import ResNetV2Encoder
from midas.networks.encoders.dinov2_encoder import DINOv2Encoder
from midas.networks.encoders.resnet_imagenet_encoder import (
    ResNetImageNetEncoder,
    load_resnet_imagenet_pretrained_params,
)
from midas.agents.midas.actor_updater import update_actor_midas, update_actor_bc_residual
from midas.agents.midas.critic_updater import update_critic_residual
from midas.agents.midas.shared_jit import (
    _sample_via_method_jit,
    _mode_via_method_jit,
    _sample_with_logprob_via_method_jit,
    _compute_log_prob_via_method_jit,
    _encode_pixels_via_method_jit,
    _VALID_ACTOR_ARCHS,
    _VALID_CRITIC_ARCHS,
    _freeze_encoder_optimizer,
)
from midas.data.dataset import DatasetDict
from midas.networks.learned_std_normal_policy import LearnedStdTanhNormalPolicy, FixedStdTanhNormalPolicy
from midas.networks.mip import MIPActor, MIPCritic
from midas.networks.values import StateActionEnsemble
from midas.types import Params, PRNGKey
from midas.utils.target_update import (
    soft_target_update,
    soft_target_update_skip_encoder,
)


class TrainState(train_state.TrainState):
    batch_stats: Any


# ============================================================
# JIT-compiled update functions
# ============================================================

@functools.partial(jax.jit, static_argnames=(
    'critic_reduction', 'color_jitter', 'aug_next', 'num_cameras',
    'query_frequency', 'use_huber_loss', 'predict_a_exec', 'use_vlm_embedding',
    'freeze_vision_encoder', 'is_mip_actor', 'is_mip_critic',
))
def _update_critic_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    target_critic_params: Params,
    batch: DatasetDict,
    discount: float,
    tau: float,
    residual_alpha: float,
    critic_reduction: str,
    color_jitter: bool,
    aug_next: bool,
    num_cameras: int,
    query_frequency: int,
    use_huber_loss: bool,
    huber_delta: float,
    predict_a_exec: bool = False,
    use_vlm_embedding: bool = False,
    freeze_vision_encoder: bool = False,
    is_mip_actor: bool = False,
    is_mip_critic: bool = False,
) -> Tuple[PRNGKey, TrainState, Params, Dict[str, float]]:
    """JIT-compiled critic update (standard TD learning)."""
    
    if not (use_vlm_embedding or freeze_vision_encoder):
        # Data augmentation for pixels (skip when using VLM embeddings)
        aug_pixels = batch['observations']['pixels']
        aug_next_pixels = batch['next_observations']['pixels']
        
        if batch['observations']['pixels'].squeeze().ndim != 2:
            rng, key = jax.random.split(rng)
            aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

            if color_jitter:
                rng, key = jax.random.split(rng)
                if num_cameras > 1:
                    for i in range(num_cameras):
                        aug_pixels = aug_pixels.at[:, :, :, i*3:(i+1)*3].set(
                            (color_transform(key, aug_pixels[:, :, :, i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8)
                        )
                else:
                    aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

        observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
        batch = batch.copy(add_or_replace={'observations': observations})

        if aug_next:
            rng, key = jax.random.split(rng)
            aug_next_pixels = batched_random_crop(key, batch['next_observations']['pixels'])
            if color_jitter:
                rng, key = jax.random.split(rng)
                if num_cameras > 1:
                    for i in range(num_cameras):
                        aug_next_pixels = aug_next_pixels.at[:, :, :, i*3:(i+1)*3].set(
                            (color_transform(key, aug_next_pixels[:, :, :, i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8)
                        )
                else:
                    aug_next_pixels = (color_transform(key, aug_next_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)
            next_observations = batch['next_observations'].copy(add_or_replace={'pixels': aug_next_pixels})
            batch = batch.copy(add_or_replace={'next_observations': next_observations})
    
    # Critic update via TD learning
    key, rng = jax.random.split(rng)
    target_critic = critic.replace(params=target_critic_params)
    temp_dummy = None
    new_critic, critic_info = update_critic_residual(
        key, actor, critic, target_critic, temp_dummy, batch,
        discount, residual_alpha, query_frequency,
        critic_reduction=critic_reduction, backup_entropy=False,
        use_huber_loss=use_huber_loss, huber_delta=huber_delta,
        predict_a_exec=predict_a_exec,
        is_mip_actor=is_mip_actor,
        is_mip_critic=is_mip_critic,
    )
    if freeze_vision_encoder:
        # See shared_jit: skip encoder subtree in EMA when
        # frozen so the per-step DINOv2-sized tree walk doesn't dominate.
        new_target_critic_params = soft_target_update_skip_encoder(
            new_critic.params, target_critic_params, tau
        )
    else:
        new_target_critic_params = soft_target_update(
            new_critic.params, target_critic_params, tau
        )

    return rng, new_critic, new_target_critic_params, critic_info


@functools.partial(jax.jit, static_argnames=(
    'color_jitter', 'num_cameras',
    'query_frequency', 'action_dim',
    'midas_num_samples', 'midas_num_elites', 'midas_num_grad_steps',
    'critic_reduction', 'predict_a_exec', 'use_vlm_embedding',
    'freeze_vision_encoder',
    'bc_flag', 'bc_on_success_only', 'is_mip_actor',
    'b_o_n', 'grad_a_q', 'use_trust_region',
))
def _update_actor_midas_jit(
    rng: PRNGKey,
    actor: TrainState,
    critic: TrainState,
    batch: DatasetDict,
    residual_alpha: float,
    color_jitter: bool,
    num_cameras: int,
    query_frequency: int,
    action_dim: int,
    midas_num_samples: int,
    midas_num_elites: int,
    midas_num_grad_steps: int,
    midas_step_size: float,
    critic_reduction: str,
    predict_a_exec: bool = False,
    use_vlm_embedding: bool = False,
    freeze_vision_encoder: bool = False,
    bc_flag: bool = False,
    bc_reg_coeff: float = 0.0,
    bc_on_success_only: bool = False,
    is_mip_actor: bool = False,
    b_o_n: bool = True,
    grad_a_q: bool = True,
    use_trust_region: bool = False,
    a_star_delta_clip_norm: Optional[jnp.ndarray] = None,
) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:
    """JIT-compiled MIDAS actor update (Best-of-N + Grad-Q + BC distillation)."""
    
    if not (use_vlm_embedding or freeze_vision_encoder):
        # Data augmentation for pixels (skip when using VLM embeddings)
        aug_pixels = batch['observations']['pixels']
        
        if batch['observations']['pixels'].squeeze().ndim != 2:
            rng, key = jax.random.split(rng)
            aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

            if color_jitter:
                rng, key = jax.random.split(rng)
                if num_cameras > 1:
                    for i in range(num_cameras):
                        aug_pixels = aug_pixels.at[:, :, :, i*3:(i+1)*3].set(
                            (color_transform(key, aug_pixels[:, :, :, i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8)
                        )
                else:
                    aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

        observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
        batch = batch.copy(add_or_replace={'observations': observations})
    
    # MIDAS actor update
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor_midas(
        key, actor, critic, batch,
        residual_alpha, query_frequency, action_dim,
        midas_num_samples=midas_num_samples,
        midas_num_elites=midas_num_elites,
        midas_num_grad_steps=midas_num_grad_steps,
        midas_step_size=midas_step_size,
        critic_reduction=critic_reduction,
        predict_a_exec=predict_a_exec,
        bc_flag=bc_flag,
        bc_reg_coeff=bc_reg_coeff,
        bc_on_success_only=bc_on_success_only,
        is_mip_actor=is_mip_actor,
        b_o_n=b_o_n,
        grad_a_q=grad_a_q,
        use_trust_region=use_trust_region,
        a_star_delta_clip_norm=a_star_delta_clip_norm,
    )
    
    return rng, new_actor, actor_info


@functools.partial(jax.jit, static_argnames=(
    'color_jitter', 'num_cameras', 'query_frequency', 'predict_a_exec', 'use_vlm_embedding',
    'freeze_vision_encoder', 'is_mip_actor',
))
def _update_actor_bc_jit(
    rng: PRNGKey,
    actor: TrainState,
    batch: DatasetDict,
    color_jitter: bool,
    num_cameras: int,
    query_frequency: int,
    predict_a_exec: bool = False,
    use_vlm_embedding: bool = False,
    freeze_vision_encoder: bool = False,
    is_mip_actor: bool = False,
) -> Tuple[PRNGKey, TrainState, Dict[str, float]]:
    """JIT-compiled BC warmup actor update."""
    
    if not (use_vlm_embedding or freeze_vision_encoder):
        # Data augmentation for pixels (skip when using VLM embeddings)
        aug_pixels = batch['observations']['pixels']
        
        if batch['observations']['pixels'].squeeze().ndim != 2:
            rng, key = jax.random.split(rng)
            aug_pixels = batched_random_crop(key, batch['observations']['pixels'])

            if color_jitter:
                rng, key = jax.random.split(rng)
                if num_cameras > 1:
                    for i in range(num_cameras):
                        aug_pixels = aug_pixels.at[:, :, :, i*3:(i+1)*3].set(
                            (color_transform(key, aug_pixels[:, :, :, i*3:(i+1)*3].astype(jnp.float32)/255.)*255).astype(jnp.uint8)
                        )
                else:
                    aug_pixels = (color_transform(key, aug_pixels.astype(jnp.float32)/255.)*255).astype(jnp.uint8)

        observations = batch['observations'].copy(add_or_replace={'pixels': aug_pixels})
        batch = batch.copy(add_or_replace={'observations': observations})
    
    # BC warmup actor update
    key, rng = jax.random.split(rng)
    new_actor, actor_info = update_actor_bc_residual(
        key, actor, batch, query_frequency,
        predict_a_exec=predict_a_exec,
        is_mip_actor=is_mip_actor,
    )
    
    return rng, new_actor, actor_info


# ============================================================
# MIDAS Learner
# ============================================================

class MidasLearner(Agent):
    """Residual MIDAS Learner for pixel observations.
    
    Policy-Agnostic RL: the actor is trained by distilling Q-optimized
    actions rather than by differentiating through the critic.
    
    Training loop:
    - Critic: standard TD learning 
    - Actor: Best-of-N sampling → Q-gradient refinement → MSE distillation
    """

    def __init__(
        self,
        seed: int,
        observations: FrozenDict,
        actions: jnp.ndarray,
        # Actor
        actor_lr: float = 3e-4,
        hidden_dims: Sequence[int] = (256, 256, 256),
        latent_dim: int = 50,
        dropout_rate: float = 0.0,
        encoder_type: str = 'resnet_small',
        encoder_norm: str = 'batch',
        use_bottleneck: bool = True,
        use_spatial_softmax: bool = False,
        softmax_temperature: float = 1.0,
        actor_pop_base_actions: bool = False,
        # Critic
        critic_lr: float = 3e-4,
        critic_pop_base_actions: bool = True,
        num_qs: int = 2,
        # Target networks
        tau: float = 0.005,
        discount: float = 0.99,
        critic_reduction: str = 'min',
        # Residual
        residual_alpha: float = 1.0,
        action_magnitude: float = 0.1,
        # Policy std bounds
        log_std_min: float = -5.0,
        log_std_max: float = 2.0,
        learn_std: bool = True,
        fixed_log_std: float = -0.5,
        # Data augmentation
        color_jitter: bool = True,
        aug_next: bool = True,
        num_cameras: int = 1,
        # MIDAS-specific parameters
        midas_num_samples: int = 16,
        midas_num_elites: int = 4,
        midas_num_grad_steps: int = 5,
        midas_step_size: float = 0.01,
        # PA-RL ablation flags + optional trust region on the distillation target.
        b_o_n: bool = True,
        grad_a_q: bool = True,
        midas_use_trust_region: bool = False,
        midas_a_star_delta_clip_norm: Optional[Sequence[float]] = None,
        # Stability
        max_grad_norm: float = 1.0,
        use_huber_loss: bool = False,
        huber_delta: float = 1.0,
        # Update ratio control
        num_critic_updates: int = 2,
        num_actor_updates: int = 4,
        # Action prediction mode
        predict_a_exec: bool = False,
        # VLM embedding mode
        use_vlm_embedding: bool = False,
        # Vision encoder freezing
        freeze_vision_encoder: bool = False,
        # BC regularization
        bc_reg_coeff: float = 0.0,
        bc_on_success_only: bool = False,
        # Architecture selection
        actor_arch: str = 'tanh_gaussian',
        critic_arch: str = 'mlp_ensemble',
        mip_t_star: float = 0.9,
        mip_noise_std: float = 0.01,
        mip_use_film: bool = False,
        mip_q_noise_scale: float = 1.0,
        # Other
        decay_steps: Optional[int] = None,
        cnn_features: Sequence[int] = (32, 64, 128, 256),
        cnn_strides: Sequence[int] = (2, 2, 2, 2),
        cnn_padding: str = 'VALID',
    ):
        """Initialize Residual MIDAS learner.
        
        Args:
            midas_num_samples: N — number of action candidates sampled from actor.
            midas_num_elites: K — number of top-Q actions kept for refinement.
            midas_num_grad_steps: Number of gradient ascent steps on Q w.r.t. action.
            midas_step_size: Learning rate for gradient ascent on actions.
            midas_use_trust_region: Clip refined actor targets around the base
                action. Disabled by default for simulation training.
            midas_a_star_delta_clip_norm: Per-action-dimension trust-region
                radii. Required when ``midas_use_trust_region`` is enabled.
        """
        self._residual_alpha = jnp.asarray(residual_alpha, dtype=jnp.float32)
        self.color_jitter = color_jitter
        self.aug_next = aug_next
        self.num_cameras = num_cameras
        
        self.query_frequency = actions.shape[1]
        self.action_dim = np.prod(actions.shape[-2:])
        self.action_chunk_shape = actions.shape[-2:]
        self.action_dim_per_step = actions.shape[-1]

        self.tau = tau
        self.discount = discount
        self.critic_reduction = critic_reduction
        self.algo = 'midas'
        
        # MIDAS parameters
        self.midas_num_samples = midas_num_samples
        self.midas_num_elites = midas_num_elites
        self.midas_num_grad_steps = midas_num_grad_steps
        self.midas_step_size = midas_step_size

        # PA-RL ablation flags (static at JIT time) and trust-region cap on the
        # final distillation target. The cap is per-joint in the normalized
        # action space the critic and BC loss live in; a large finite radius
        # can be used to leave selected dimensions effectively uncapped.
        self.b_o_n = bool(b_o_n)
        self.grad_a_q = bool(grad_a_q)
        self.midas_use_trust_region = bool(midas_use_trust_region)
        if self.midas_use_trust_region and midas_a_star_delta_clip_norm is None:
            raise ValueError(
                "midas_a_star_delta_clip_norm is required when "
                "midas_use_trust_region=True"
            )
        action_dim_per_step = int(actions.shape[-1])
        if midas_a_star_delta_clip_norm is None:
            self._a_star_delta_clip_norm = None
        else:
            cap_arr = jnp.asarray(midas_a_star_delta_clip_norm, dtype=jnp.float32)
            if cap_arr.shape != (action_dim_per_step,):
                raise ValueError(
                    f"midas_a_star_delta_clip_norm shape {tuple(cap_arr.shape)} != "
                    f"(action_dim_per_step={action_dim_per_step},)"
                )
            self._a_star_delta_clip_norm = cap_arr
        
        # Stability
        self.max_grad_norm = max_grad_norm
        self.use_huber_loss = use_huber_loss
        self.huber_delta = huber_delta
        
        # Update ratios
        self.num_critic_updates = num_critic_updates
        self.num_actor_updates = num_actor_updates
        
        # Action prediction mode
        self.predict_a_exec = predict_a_exec
        
        # VLM embedding mode
        self.use_vlm_embedding = use_vlm_embedding
        self.freeze_vision_encoder = freeze_vision_encoder
        self.pixel_features_dim: Optional[int] = None

        # BC regularization
        self.bc_reg_coeff = bc_reg_coeff
        self.bc_on_success_only = bc_on_success_only

        # Architecture selection
        if actor_arch not in _VALID_ACTOR_ARCHS:
            raise ValueError(
                f'actor_arch={actor_arch!r} not in {_VALID_ACTOR_ARCHS}'
            )
        if critic_arch not in _VALID_CRITIC_ARCHS:
            raise ValueError(
                f'critic_arch={critic_arch!r} not in {_VALID_CRITIC_ARCHS}'
            )
        if use_huber_loss and critic_arch == 'mip_ensemble':
            raise ValueError(
                "use_huber_loss is incompatible with critic_arch='mip_ensemble'; "
                "MIP-Q TD loss has its own regression+denoising structure."
            )
        self.actor_arch = actor_arch
        self.critic_arch = critic_arch
        self.mip_t_star = mip_t_star
        self.mip_noise_std = mip_noise_std
        self.mip_use_film = mip_use_film
        self.mip_q_noise_scale = mip_q_noise_scale

        if predict_a_exec:
            print(f'[WARNING] predict_a_exec=True: residual_alpha={residual_alpha} is IGNORED '
                  f'for action composition. Actor predicts a_exec directly.')

        rng = jax.random.PRNGKey(seed)
        rng, actor_key, critic_key = jax.random.split(rng, 3)

        # ----- Encoder -----
        if encoder_type == 'small':
            encoder_def = Encoder(cnn_features, cnn_strides, cnn_padding)
        elif encoder_type == 'impala':
            encoder_def = ImpalaEncoder()
        elif encoder_type == 'impala_small':
            encoder_def = SmallerImpalaEncoder()
        elif encoder_type == 'resnet_small':
            encoder_def = ResNetSmall(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_18_v1':
            encoder_def = ResNet18(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_34_v1':
            encoder_def = ResNet34(norm=encoder_norm, use_spatial_softmax=use_spatial_softmax, softmax_temperature=softmax_temperature)
        elif encoder_type == 'resnet_small_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(1, 1, 1, 1), norm=encoder_norm)
        elif encoder_type == 'resnet_18_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(2, 2, 2, 2), norm=encoder_norm)
        elif encoder_type == 'resnet_34_v2':
            encoder_def = ResNetV2Encoder(stage_sizes=(3, 4, 6, 3), norm=encoder_norm)
        elif encoder_type == 'dinov2':
            encoder_def = DINOv2Encoder()
        elif encoder_type == 'resnet_imagenet':
            encoder_def = ResNetImageNetEncoder()
        else:
            raise ValueError(f'Encoder type not found: {encoder_type}')

        if decay_steps is not None:
            actor_lr_schedule = optax.cosine_decay_schedule(actor_lr, decay_steps)
            critic_lr_schedule = optax.cosine_decay_schedule(critic_lr, decay_steps)
        else:
            actor_lr_schedule = actor_lr
            critic_lr_schedule = critic_lr

        if len(hidden_dims) == 1:
            hidden_dims = (hidden_dims[0], hidden_dims[0], hidden_dims[0])
        
        # ----- Actor -----
        if actor_arch == 'tanh_gaussian':
            if learn_std:
                policy_def = LearnedStdTanhNormalPolicy(
                    hidden_dims, self.action_dim,
                    dropout_rate=dropout_rate,
                    log_std_min=log_std_min,
                    log_std_max=log_std_max,
                    low=-action_magnitude,
                    high=action_magnitude
                )
            else:
                policy_def = FixedStdTanhNormalPolicy(
                    hidden_dims, self.action_dim,
                    dropout_rate=dropout_rate,
                    fixed_log_std=fixed_log_std,
                    low=-action_magnitude,
                    high=action_magnitude,
                )
        else:  # 'mip'
            policy_def = MIPActor(
                hidden_dims=hidden_dims,
                action_dim=self.action_dim,
                layer_norm=False,
                use_film=self.mip_use_film,
                mip_t_star=self.mip_t_star,
                use_constant_noise=True,
                constant_noise_std=self.mip_noise_std,
                act_min=-action_magnitude,
                act_max=action_magnitude,
            )

        actor_def = PixelMultiplexer(
            encoder=encoder_def,
            network=policy_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
            pop_base_actions=actor_pop_base_actions,
            use_vlm_embedding=use_vlm_embedding,
            freeze_vision_encoder=freeze_vision_encoder,
        )
        print(f"MIDAS Actor: {actor_def}")
        if actor_arch == 'mip':
            rng, init_key = jax.random.split(rng)
            actor_def_init = actor_def.init(actor_key, observations, init_key, method='sample')
        else:
            actor_def_init = actor_def.init(actor_key, observations)
        actor_params = actor_def_init['params']
        actor_batch_stats = actor_def_init.get('batch_stats', None)

        actor_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=actor_lr_schedule),
        )
        actor_optimizer = _freeze_encoder_optimizer(actor_optimizer, freeze_vision_encoder)
        actor = TrainState.create(
            apply_fn=actor_def.apply,
            params=actor_params,
            tx=actor_optimizer,
            batch_stats=actor_batch_stats,
        )

        # ----- Critic -----
        if critic_arch == 'mlp_ensemble':
            critic_def = StateActionEnsemble(hidden_dims, num_qs=num_qs)
        else:  # 'mip_ensemble'
            critic_def = MIPCritic(
                hidden_dims=hidden_dims,
                num_qs=num_qs,
                mip_t_star=self.mip_t_star,
                mip_q_noise_scale=self.mip_q_noise_scale,
                layer_norm=True,
            )
        critic_def = PixelMultiplexer(
            encoder=encoder_def,
            network=critic_def,
            latent_dim=latent_dim,
            use_bottleneck=use_bottleneck,
            pop_base_actions=critic_pop_base_actions,
            use_vlm_embedding=use_vlm_embedding,
            freeze_vision_encoder=freeze_vision_encoder,
        )
        print(f"MIDAS Critic: {critic_def}")
        
        actions_flat = actions.reshape(actions.shape[0], -1)
        critic_def_init = critic_def.init(critic_key, observations, actions_flat)

        critic_params = critic_def_init['params']
        critic_batch_stats = critic_def_init.get('batch_stats', None)
        
        critic_optimizer = optax.chain(
            optax.clip_by_global_norm(max_grad_norm),
            optax.adam(learning_rate=critic_lr_schedule),
        )
        critic_optimizer = _freeze_encoder_optimizer(critic_optimizer, freeze_vision_encoder)
        critic = TrainState.create(
            apply_fn=critic_def.apply,
            params=critic_params,
            tx=critic_optimizer,
            batch_stats=critic_batch_stats
        )
        # Structural copy via tree_map; see SAC residual learner for rationale.
        target_critic_params = jax.tree_util.tree_map(lambda x: x, critic_params)
        
        # Load pretrained DINOv2 weights into encoder subtree
        if encoder_type == 'dinov2' and not use_vlm_embedding:
            from midas.networks.encoders.dinov2_encoder import load_dinov2_pretrained_params
            pretrained = load_dinov2_pretrained_params()
            actor = actor.replace(
                params={**actor.params, 'encoder': {**actor.params['encoder'], 'dinov2': pretrained}}
            )
            critic = critic.replace(
                params={**critic.params, 'encoder': {**critic.params['encoder'], 'dinov2': pretrained}}
            )
            target_critic_params = {**target_critic_params, 'encoder': {**target_critic_params['encoder'], 'dinov2': pretrained}}
            print('Loaded pretrained DINOv2 weights into actor/critic encoders')

        # Load pretrained ImageNet ResNet-50 weights into encoder subtree
        if encoder_type == 'resnet_imagenet' and not use_vlm_embedding:
            pretrained = load_resnet_imagenet_pretrained_params()
            pretrained_params = pretrained["params"]
            actor = actor.replace(
                params={**actor.params, 'encoder': {**actor.params['encoder'], 'resnet_imagenet': pretrained_params}}
            )
            critic = critic.replace(
                params={**critic.params, 'encoder': {**critic.params['encoder'], 'resnet_imagenet': pretrained_params}}
            )
            target_critic_params = {**target_critic_params, 'encoder': {**target_critic_params['encoder'], 'resnet_imagenet': pretrained_params}}
            pbs = pretrained.get("batch_stats") or {}
            if pbs:
                # Force-install pretrained BN running stats regardless of
                # whether init produced a batch_stats collection. See SAC
                # residual learner for the same fix and rationale.
                actor_bs = dict(actor.batch_stats) if actor.batch_stats is not None else {}
                actor_enc = dict(actor_bs.get('encoder', {}))
                actor_enc['resnet_imagenet'] = pbs
                actor_bs['encoder'] = actor_enc
                actor = actor.replace(batch_stats=actor_bs)

                critic_bs = dict(critic.batch_stats) if critic.batch_stats is not None else {}
                critic_enc = dict(critic_bs.get('encoder', {}))
                critic_enc['resnet_imagenet'] = pbs
                critic_bs['encoder'] = critic_enc
                critic = critic.replace(batch_stats=critic_bs)
            print('Loaded pretrained ResNet-50 ImageNet weights into actor/critic encoders')

        self._rng = rng
        self._actor = actor
        self._critic = critic
        self._target_critic_params = target_critic_params

        # Probe encoder output dim so the training loop can size the
        # ``pixel_features`` Box on the observation space before constructing
        # the replay buffer. See ``compute_pixel_features`` below.
        if freeze_vision_encoder and not use_vlm_embedding:
            feat = _encode_pixels_via_method_jit(
                self._actor.apply_fn,
                self._actor.params,
                observations['pixels'],
                self._actor.batch_stats,
            )
            self.pixel_features_dim = int(feat.shape[-1])
            print(f'  pixel_features_dim: {self.pixel_features_dim}')

        print(f'MIDAS Residual Learner initialized:')
        print(f'  residual_alpha: {self._residual_alpha}')
        print(f'  query_frequency: {self.query_frequency}')
        print(f'  action_dim: {self.action_dim}')
        print(f'  action_dim_per_step: {self.action_dim_per_step}')
        print(f'  critic_reduction: {self.critic_reduction}')
        print(f'  midas_num_samples (N): {self.midas_num_samples}')
        print(f'  midas_num_elites (K): {self.midas_num_elites}')
        print(f'  midas_num_grad_steps: {self.midas_num_grad_steps}')
        print(f'  midas_step_size: {self.midas_step_size}')
        print(f'  midas_use_trust_region: {self.midas_use_trust_region}')
        print(f'  use_huber_loss: {self.use_huber_loss}')
        print(f'  huber_delta: {self.huber_delta}')
        print(f'  max_grad_norm: {self.max_grad_norm}')
        print(f'  num_critic_updates: {self.num_critic_updates}')
        print(f'  num_actor_updates: {self.num_actor_updates}')
        print(f'  predict_a_exec: {self.predict_a_exec}')
        print(f'  use_vlm_embedding: {self.use_vlm_embedding}')
        print(f'  bc_reg_coeff: {self.bc_reg_coeff}')
        print(f'  bc_on_success_only: {self.bc_on_success_only}')
        print(f'  learn_std: {learn_std}')
        print(f'  actor_arch: {self.actor_arch}')
        print(f'  critic_arch: {self.critic_arch}')
        print(f'  mip_t_star: {self.mip_t_star}')
        print(f'  mip_noise_std: {self.mip_noise_std}')
        print(f'  mip_use_film: {self.mip_use_film}')
        print(f'  mip_q_noise_scale: {self.mip_q_noise_scale}')

    # ============================================================
    # Update methods
    # ============================================================

    def update_critic(self, batch: FrozenDict) -> Dict[str, float]:
        """Perform a single critic TD update."""
        new_rng, new_critic, new_target_critic, critic_info = _update_critic_jit(
            self._rng,
            self._actor,
            self._critic,
            self._target_critic_params,
            batch,
            self.discount,
            self.tau,
            self._residual_alpha,
            self.critic_reduction,
            self.color_jitter,
            self.aug_next,
            self.num_cameras,
            self.query_frequency,
            self.use_huber_loss,
            self.huber_delta,
            self.predict_a_exec,
            self.use_vlm_embedding,
            self.freeze_vision_encoder,
            self.actor_arch == 'mip',
            self.critic_arch == 'mip_ensemble',
        )
        self._rng = new_rng
        self._critic = new_critic
        self._target_critic_params = new_target_critic
        return critic_info

    def update_actor(self, batch: FrozenDict) -> Dict[str, float]:
        """Perform a single MIDAS actor update (Best-of-N + Grad-Q + BC distill)."""
        new_rng, new_actor, actor_info = _update_actor_midas_jit(
            self._rng,
            self._actor,
            self._critic,
            batch,
            self._residual_alpha,
            self.color_jitter,
            self.num_cameras,
            self.query_frequency,
            self.action_dim_per_step,
            self.midas_num_samples,
            self.midas_num_elites,
            self.midas_num_grad_steps,
            self.midas_step_size,
            self.critic_reduction,
            self.predict_a_exec,
            self.use_vlm_embedding,
            self.freeze_vision_encoder,
            bc_flag=bool(self.bc_reg_coeff > 0.0),
            bc_reg_coeff=self.bc_reg_coeff,
            bc_on_success_only=self.bc_on_success_only,
            is_mip_actor=self.actor_arch == 'mip',
            b_o_n=self.b_o_n,
            grad_a_q=self.grad_a_q,
            use_trust_region=self.midas_use_trust_region,
            a_star_delta_clip_norm=self._a_star_delta_clip_norm,
        )
        self._rng = new_rng
        self._actor = new_actor
        return actor_info

    def update_actor_bc(self, batch: FrozenDict) -> Dict[str, float]:
        """Perform a single BC warmup actor update."""
        new_rng, new_actor, actor_info = _update_actor_bc_jit(
            self._rng,
            self._actor,
            batch,
            self.color_jitter,
            self.num_cameras,
            self.query_frequency,
            self.predict_a_exec,
            self.use_vlm_embedding,
            self.freeze_vision_encoder,
            self.actor_arch == 'mip',
        )
        self._rng = new_rng
        self._actor = new_actor
        return actor_info

    def update(self, batch: FrozenDict) -> Dict[str, float]:
        """Perform one critic + one actor update (backward compat)."""
        critic_info = self.update_critic(batch)
        actor_info = self.update_actor(batch)

        all_info = {**critic_info, **actor_info}
        all_info['residual/alpha'] = float(self._residual_alpha)
        all_info['algo'] = self.algo

        return all_info

    # ------------------------------------------------------------------
    # Polymorphic action surface (overrides Agent base class; routes MIP
    # through ``method='sample'`` / ``method='mode'`` since MIPActor's
    # ``__call__`` raises ``NotImplementedError``).
    # ------------------------------------------------------------------

    def sample_actions(self, observations):
        rng, actions = _sample_via_method_jit(
            self._rng, self._actor.apply_fn, self._actor.params,
            observations, get_batch_stats(self._actor),
        )
        self._rng = rng
        return np.asarray(actions)

    def eval_actions(self, observations):
        actions = _mode_via_method_jit(
            self._actor.apply_fn, self._actor.params,
            observations, get_batch_stats(self._actor),
        )
        return np.asarray(actions)

    def compute_pixel_features(self, pixels):
        """Run the actor's frozen vision encoder on raw pixels.

        Used at rollout time when ``freeze_vision_encoder=True`` to cache
        encoder output in the replay buffer's ``pixel_features`` slot so the
        encoder forward is skipped on the train graph. See
        ``MidasLearner.compute_pixel_features`` for rationale.
        """
        feat = _encode_pixels_via_method_jit(
            self._actor.apply_fn, self._actor.params,
            pixels, get_batch_stats(self._actor),
        )
        return np.asarray(feat)

    def sample_actions_with_log_prob(self, observations):
        rng, actions, log_probs = _sample_with_logprob_via_method_jit(
            self._rng, self._actor.apply_fn, self._actor.params,
            observations, get_batch_stats(self._actor),
        )
        self._rng = rng
        return np.asarray(actions), np.asarray(log_probs)

    def compute_log_prob(self, observations, actions):
        if self.actor_arch == 'mip':
            raise NotImplementedError(
                "compute_log_prob is unavailable for actor_arch='mip'; "
                "MIPActor does not expose a closed-form log-density on "
                "arbitrary actions. Guard call sites accordingly."
            )
        log_probs = _compute_log_prob_via_method_jit(
            self._actor.apply_fn, self._actor.params,
            observations, actions, get_batch_stats(self._actor),
        )
        return np.asarray(log_probs)

    # ============================================================
    # Evaluation & checkpointing
    # ============================================================

    @property
    def _save_dict(self):
        return {
            'rng': self._rng,
            'critic': self._critic,
            'target_critic_params': self._target_critic_params,
            'actor': self._actor,
            'residual_alpha': self._residual_alpha,
            'algo': self.algo,
            'predict_a_exec': self.predict_a_exec,
            'actor_arch': self.actor_arch,
            'critic_arch': self.critic_arch,
            'mip_t_star': self.mip_t_star,
            'mip_noise_std': self.mip_noise_std,
            'mip_use_film': self.mip_use_film,
            'mip_q_noise_scale': self.mip_q_noise_scale,
        }

    def restore_checkpoint(self, dir):
        assert pathlib.Path(dir).exists(), f"Checkpoint {dir} does not exist."
        output_dict = checkpoints.restore_checkpoint(dir, self._save_dict)
        for key in ('actor_arch', 'critic_arch'):
            if key in output_dict and output_dict[key] != getattr(self, key):
                raise ValueError(
                    f"Checkpoint {key}={output_dict[key]!r} does not match "
                    f"learner {key}={getattr(self, key)!r}"
                )
        self._actor = output_dict['actor']
        self._critic = output_dict['critic']
        self._target_critic_params = output_dict['target_critic_params']
        # Older checkpoints do not contain ``rng``. Flax retains the target
        # value in that case, preserving the historical seed-on-restore
        # behavior while new checkpoints continue the exact PRNG stream.
        self._rng = output_dict.get('rng', self._rng)
        if 'residual_alpha' in output_dict:
            self._residual_alpha = jnp.asarray(output_dict['residual_alpha'], dtype=jnp.float32)
        if 'algo' in output_dict:
            self.algo = output_dict['algo']
        if 'predict_a_exec' in output_dict:
            self.predict_a_exec = bool(output_dict['predict_a_exec'])
        print(f'Restored MIDAS checkpoint from {dir} (algo: {self.algo}, predict_a_exec: {self.predict_a_exec})')
