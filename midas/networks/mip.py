"""Minimum Iterative Policy (MIP) networks and helpers.

Self-contained port of the MIP + MIP-Q building blocks from
``mipq_single_file.py`` plus two wrapper ``nn.Module`` classes that expose the
polymorphic method surface used by the residual learners:

  * :class:`MIPActor`  — two-step flow-matching actor (wraps :class:`ActorVectorField`).
  * :class:`MIPCritic` — explicit-ensemble MIP-Q critic (wraps :class:`ValueMIPEnsemble`).

The pure-function helpers (target construction, sampling, log-probability,
MIP-Q prediction) are kept as module-level callables so they can be unit-tested
against ``mipq_single_file.py`` as an oracle.

Out of scope for v1 and intentionally not ported:
  * Implicit-ensemble ``ValueMIP`` and its helpers (``sample_mip_q_values``,
    ``compute_mip_q_predictions``).
  * Learnable ``noise_fn``; only ``use_constant_noise=True`` is exercised.
  * HL-Gauss categorical TD loss for the MIP critic.
"""

import math
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.linen import initializers

from midas.networks.mlp import _flatten_dict
from midas.networks.normal_tanh_policy import TanhMultivariateNormalDiag


def _obs_to_flat(observations):
    """Accept either a 2D tensor or a dict-of-arrays; return a 2D tensor.

    The Gaussian path feeds dict observations into ``MLP._flatten_dict``; MIP
    helpers (``sample_mip_actions_sde`` etc.) assume flat 2D tensors and read
    ``observations.shape[0]``. This helper bridges the two so the MIP wrapper
    modules can sit behind ``PixelMultiplexer`` without custom plumbing.
    """
    if hasattr(observations, "values") and not hasattr(observations, "shape"):
        return _flatten_dict(observations)
    return observations


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------


def default_init(scale: float = 1.0):
    """Variance-scaling kernel initializer (fan_avg, uniform)."""
    return nn.initializers.variance_scaling(scale, "fan_avg", "uniform")


class SinusoidalTimeEmbedding(nn.Module):
    """Scalar t -> sinusoidal features -> Dense -> GELU -> Dense."""

    embed_dim: int = 32
    max_freq_log: float = math.log(10000)

    @nn.compact
    def __call__(self, t):
        if t.shape[-1] == 1:
            t = t[..., 0]
        half_dim = self.embed_dim // 2
        freqs = jnp.exp(
            jnp.arange(half_dim, dtype=jnp.float32)
            * -(self.max_freq_log / (half_dim - 1))
        )
        angles = t[..., None] * freqs
        emb = jnp.concatenate([jnp.sin(angles), jnp.cos(angles)], axis=-1)
        emb = nn.Dense(self.embed_dim * 2)(emb)
        emb = nn.gelu(emb)
        emb = nn.Dense(self.embed_dim)(emb)
        return emb


class _FlowMLP(nn.Module):
    """MLP used inside the MIP flow backbone.

    Kept local to this module (rather than reusing ``midas.networks.MLP``) so
    the parameter tree and feature-sow behaviour match ``mipq_single_file.py``
    exactly for parity tests.
    """

    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = self.activations(x)
            if i == len(self.hidden_dims) - 2:
                self.sow("intermediates", "feature", x)
        return x


class MLPResidualBlock(nn.Module):
    """Residual block: LN -> Dense(4d) -> GELU -> Dropout -> LN -> Dense(d) -> Dropout + residual."""

    dim: int
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool) -> jnp.ndarray:
        residual = x
        ortho_init = initializers.orthogonal()
        zeros_init = initializers.constant(0.0)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.dim * 4, kernel_init=ortho_init, bias_init=zeros_init)(x)
        x = nn.gelu(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)

        x = nn.LayerNorm()(x)
        x = nn.Dense(self.dim, kernel_init=ortho_init, bias_init=zeros_init)(x)
        x = nn.Dropout(rate=self.dropout)(x, deterministic=deterministic)

        return x + residual


class FilmConditioning(nn.Module):
    """FiLM modulation (MLP-adapted): out = x * (1 + gamma(cond)) + beta(cond).

    Both projections are zero-initialized so FiLM is identity at init.
    """

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        gamma = nn.Dense(
            features=x.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(cond)
        beta = nn.Dense(
            features=x.shape[-1],
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros,
        )(cond)
        return x * (1 + gamma) + beta


class FilmMLP(nn.Module):
    """MLP stack with a FiLM block applied after each residual block.

    ``hidden_dims`` follows the MLP convention: the final element is the
    output dimensionality. Residual blocks all operate at the first hidden
    dim; if subsequent hidden dims differ, a projection is inserted.
    """

    hidden_dims: Sequence[int]
    layer_norm: bool = True
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x: jnp.ndarray, cond: jnp.ndarray) -> jnp.ndarray:
        *block_dims, output_dim = self.hidden_dims

        if not block_dims:
            return nn.Dense(output_dim, kernel_init=default_init())(x)

        x = nn.Dense(block_dims[0], kernel_init=default_init())(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.gelu(x)

        prev_dim = block_dims[0]
        for dim in block_dims:
            if dim != prev_dim:
                x = nn.Dense(dim, kernel_init=default_init())(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
                x = nn.gelu(x)
            x = MLPResidualBlock(dim=dim, dropout=self.dropout)(x, deterministic=True)
            x = FilmConditioning()(x, cond)
            prev_dim = dim

        x = nn.Dense(output_dim, kernel_init=default_init())(x)
        return x


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    """Wrap a Flax module with nn.vmap to produce an ensemble of independent params."""
    return nn.vmap(
        cls,
        variable_axes={"params": 0, "intermediates": 0},
        split_rngs={"params": True, "dropout": True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# ActorVectorField
# ---------------------------------------------------------------------------


class ActorVectorField(nn.Module):
    """Flow-matching policy. Supports plain MLP and functional FiLM backbone.

    Attributes:
        hidden_dims: Hidden layer dimensions for the MLP backbone.
        action_dim: Dimension of the (flat) action vector the field predicts.
        layer_norm: Whether to apply layer normalization in the backbone.
        encoder: Optional Flax module for observation encoding.
        use_film: Use the FiLM-conditioned backbone (FilmMLP) instead of plain MLP.
        use_denoiser: Expose a second output head that predicts injected noise.
        time_embedding: Optional module applied to ``times`` before use.
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: Optional[nn.Module] = None
    use_film: bool = False
    use_denoiser: bool = False
    time_embedding: Optional[nn.Module] = None

    def setup(self) -> None:
        if self.use_film:
            self.mlp = FilmMLP(
                (*self.hidden_dims, self.action_dim),
                layer_norm=self.layer_norm,
            )
        else:
            self.mlp = _FlowMLP(
                (*self.hidden_dims, self.action_dim),
                activate_final=False,
                layer_norm=self.layer_norm,
            )

        if self.use_denoiser:
            if self.use_film:
                self.denoiser_mlp = FilmMLP(
                    (*self.hidden_dims, self.action_dim),
                    layer_norm=self.layer_norm,
                )
            else:
                self.denoiser_mlp = _FlowMLP(
                    (*self.hidden_dims, self.action_dim),
                    activate_final=False,
                    layer_norm=self.layer_norm,
                )

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(
        self,
        observations,
        actions,
        times=None,
        dt=None,
        is_encoded: bool = False,
        return_denoiser: bool = False,
        images=None,
    ):
        if not is_encoded and self.encoder is not None:
            observations = self.encode(observations, images)

        if times is not None and self.time_embedding is not None:
            times = self.time_embedding(times)

        if self.use_film:
            cond = observations
            if times is not None:
                cond = jnp.concatenate([cond, times], axis=-1)
            if dt is not None:
                cond = jnp.concatenate([cond, dt], axis=-1)
            v = self.mlp(actions, cond)
        else:
            if times is None:
                inputs = jnp.concatenate([observations, actions], axis=-1)
            else:
                inputs = jnp.concatenate([observations, actions, times], axis=-1)
            if dt is not None:
                inputs = jnp.concatenate([inputs, dt], axis=-1)
            v = self.mlp(inputs)

        if return_denoiser:
            if not self.use_denoiser:
                raise ValueError("Actor was initialized with use_denoiser=False")
            if self.use_film:
                z = self.denoiser_mlp(actions, cond)
            else:
                z = self.denoiser_mlp(inputs)
            return v, z

        return v


# ---------------------------------------------------------------------------
# MIP-Q critic (explicit ensemble)
# ---------------------------------------------------------------------------


class ValueMIPEnsemble(nn.Module):
    """Explicit ensemble of MIP-Q networks (num_ensembles vmapped MLPs, shared noise).

    ``scalar_input`` forms:
      - ``(B, 1)``: step-1 shared noise across all members.
      - ``(B, num_ensembles, 1)``: step-2 per-member q_0 predictions.
      - ``(B, chunk, 1)``: action-chunking step-1 (broadcast across chunk).

    Output: ``(num_ensembles, B)`` (or plus ``(num_ensembles, B, num_bins)`` logits).
    """

    hidden_dims: Sequence[int]
    num_ensembles: int = 2
    mip_q_noise_scale: float = 1.0
    mip_t_star: float = 0.9
    layer_norm: bool = True
    encoder: Optional[nn.Module] = None
    critic_loss_type: str = "mse"
    num_bins: int = 256
    q_min: Optional[float] = None
    q_max: Optional[float] = None

    def setup(self):
        num_output = self.num_bins if self.critic_loss_type == "hlgauss" else 1

        mlp_class = _FlowMLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles, in_axes=0, out_axes=0)

        self.mlp = mlp_class(
            (*self.hidden_dims, num_output),
            activate_final=False,
            layer_norm=self.layer_norm,
        )

    def encode(self, observations, images=None):
        if self.encoder is None:
            return observations
        if images is not None:
            return self.encoder(images, observations)
        return self.encoder(observations)

    def __call__(
        self,
        observations,
        actions=None,
        scalar_input=None,
        time=None,
        return_logits: bool = False,
        is_encoded: bool = False,
        images=None,
        rng=None,
    ):
        if self.encoder is not None and not is_encoded:
            observations = self.encode(observations, images)

        obs_shape = observations.shape
        batch_size = obs_shape[0]

        if scalar_input is None:
            if rng is None:
                raise ValueError("Must provide either scalar_input or rng")
            scalar_input = jax.random.uniform(
                rng,
                (batch_size, 1),
                minval=-self.mip_q_noise_scale,
                maxval=self.mip_q_noise_scale,
            )

        if time is None:
            if len(obs_shape) == 3:
                chunk_size = obs_shape[1]
                time = jnp.zeros((batch_size, chunk_size, 1))
            else:
                time = jnp.zeros((batch_size, 1))

        if len(obs_shape) == 3:
            chunk_size = obs_shape[1]
            if scalar_input.ndim == 2 and scalar_input.shape[1] == 1:
                scalar_input = jnp.tile(scalar_input[:, None, :], (1, chunk_size, 1))

        if scalar_input.ndim == 3:
            if len(obs_shape) == 3:
                scalar_input = jnp.tile(
                    scalar_input[None, :, :, :], (self.num_ensembles, 1, 1, 1)
                )
            elif scalar_input.shape[1] == self.num_ensembles:
                scalar_input = jnp.transpose(scalar_input, (1, 0, 2))
            else:
                raise ValueError(
                    f"Unexpected scalar_input shape: {scalar_input.shape}"
                )
        elif scalar_input.ndim == 2:
            scalar_input = jnp.tile(
                scalar_input[None, :, :], (self.num_ensembles, 1, 1)
            )
        else:
            raise ValueError(
                f"scalar_input must be 2D or 3D, got shape: {scalar_input.shape}"
            )

        if observations.ndim == 2:
            observations = jnp.tile(
                observations[None, :, :], (self.num_ensembles, 1, 1)
            )
            actions = jnp.tile(actions[None, :, :], (self.num_ensembles, 1, 1))
            time = jnp.tile(time[None, :, :], (self.num_ensembles, 1, 1))
        else:
            observations = jnp.tile(
                observations[None, :, :, :], (self.num_ensembles, 1, 1, 1)
            )
            actions = jnp.tile(
                actions[None, :, :, :], (self.num_ensembles, 1, 1, 1)
            )
            time = jnp.tile(time[None, :, :, :], (self.num_ensembles, 1, 1, 1))

        mlp_input = jnp.concatenate(
            [observations, actions, scalar_input, time], axis=-1
        )
        q_output = self.mlp(mlp_input)

        if self.critic_loss_type == "hlgauss":
            q_logits = q_output
            q_values = jnp.sum(
                jax.nn.softmax(q_logits, axis=-1)
                * jnp.linspace(self.q_min, self.q_max, self.num_bins),
                axis=-1,
            )
            if return_logits:
                return q_values, q_logits
        else:
            q_values = q_output.squeeze(-1)

        return q_values


# ---------------------------------------------------------------------------
# Actor BC and target construction
# ---------------------------------------------------------------------------


def preprocess_actions(
    batch: Dict[str, jnp.ndarray],
    action_chunking: bool,
    action_key: str = "actions",
) -> jnp.ndarray:
    """Flatten actions for BC loss: ``(B, H, A) -> (B, H*A)`` or ``(B, A)``."""
    actions = batch[action_key]
    if action_chunking:
        return jnp.reshape(actions, (actions.shape[0], -1))
    else:
        return actions[..., 0, :]


def apply_chunking_mask(
    loss_per_element: jnp.ndarray,
    valid_mask: jnp.ndarray,
    batch_size: int,
    horizon_length: int,
    action_dim: int,
    action_chunking: bool,
) -> jnp.ndarray:
    """Apply valid mask for action chunking or compute plain mean."""
    if action_chunking:
        loss_reshaped = jnp.reshape(
            loss_per_element, (batch_size, horizon_length, action_dim)
        )
        return jnp.mean(loss_reshaped * valid_mask[..., None])
    else:
        return jnp.mean(loss_per_element)


def get_mip_targets(
    actions: jnp.ndarray,
    rng: jnp.ndarray,
    t_star: float = 0.999,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """MIP actor targets.

    Returns:
        x_0: ``(B, A)`` noise ~ N(0, I).
        x_t_star: ``(1 - t_star) * x_0 + actions``.
        t_0_arr: zeros ``(B, 1)``.
        t_star_arr: full ``t_star`` ``(B, 1)``.
    """
    batch_size = actions.shape[0]

    rng, x0_rng = jax.random.split(rng)
    x_0 = jax.random.normal(x0_rng, actions.shape)

    t_0_arr = jnp.zeros((batch_size, 1))
    t_star_arr = jnp.full((batch_size, 1), t_star)
    x_t_star = (1 - t_star) * x_0 + actions

    return x_0, x_t_star, t_0_arr, t_star_arr


def compute_mip_bc_loss(
    batch: Dict[str, jnp.ndarray],
    rng: jnp.ndarray,
    model_fn: Callable,
    action_chunking: bool,
    horizon_length: int,
    action_dim: int,
    t_star: float = 0.999,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """MIP BC loss: ``|| f(x_0, 0) - x_1 ||^2 + || f(x_t*, t*) - x_1 ||^2``."""
    batch_actions = preprocess_actions(batch, action_chunking)
    batch_size = batch_actions.shape[0]

    x_0, x_t_star, t_0_arr, t_star_arr = get_mip_targets(batch_actions, rng, t_star)

    pred_0 = model_fn(batch["observations"], x_0, t_0_arr)
    loss_regression_per_element = jnp.square(pred_0 - batch_actions)

    pred_t_star = model_fn(batch["observations"], x_t_star, t_star_arr)
    loss_denoising_per_element = jnp.square(pred_t_star - batch_actions)

    combined_loss_per_element = (
        loss_regression_per_element + loss_denoising_per_element
    )

    valid_mask = batch.get("valid", jnp.ones((batch_size, horizon_length)))

    bc_loss = apply_chunking_mask(
        combined_loss_per_element,
        valid_mask,
        batch_size,
        horizon_length,
        action_dim,
        action_chunking,
    )
    loss_regression = apply_chunking_mask(
        loss_regression_per_element,
        valid_mask,
        batch_size,
        horizon_length,
        action_dim,
        action_chunking,
    )
    loss_denoising = apply_chunking_mask(
        loss_denoising_per_element,
        valid_mask,
        batch_size,
        horizon_length,
        action_dim,
        action_chunking,
    )

    info = {
        "bc_loss": bc_loss,
        "bc_mip_loss_regression": loss_regression,
        "bc_mip_loss_denoising": loss_denoising,
    }
    return bc_loss, info


# ---------------------------------------------------------------------------
# Actor sampling and log-probability
# ---------------------------------------------------------------------------


def sample_mip_actions_ode(
    actor_fn: Callable,
    observations: jnp.ndarray,
    rng: jnp.ndarray,
    mip_t_star: float,
    act_dim: int,
    act_min: float = -1.0,
    act_max: float = 1.0,
    noises: Optional[jnp.ndarray] = None,
    is_encoded: bool = False,
    use_tanh_squash: bool = True,
) -> jnp.ndarray:
    """MIP 2-step deterministic sampling: ``actor(obs, x_0, 0)`` -> ``actor(obs, a_0_hat, t*)``.

    ``use_tanh_squash`` (default ``True``) enforces action bounds via a
    differentiable ``tanh`` squash so gradients continue to flow when the
    actor output pushes past ``[act_min, act_max]``. Passing ``False``
    restores the legacy hard ``jnp.clip`` for parity with the oracle
    implementation in ``mipq_single_file.py``.
    """
    batch_size = observations.shape[0]

    if noises is None:
        actions = jax.random.normal(rng, (batch_size, act_dim))
    else:
        actions = noises

    t_0 = jnp.zeros((batch_size, 1))
    a_0_hat = actor_fn(observations, actions, t_0, is_encoded=is_encoded)

    t_star = jnp.full((batch_size, 1), mip_t_star)
    actions = actor_fn(observations, a_0_hat, t_star, is_encoded=is_encoded)

    if use_tanh_squash:
        mid = (act_max + act_min) / 2.0
        half_range = (act_max - act_min) / 2.0
        actions = mid + half_range * jnp.tanh((actions - mid) / half_range)
    else:
        actions = jnp.clip(actions, act_min, act_max)
    return actions


def sample_mip_actions_sde(
    actor_fn: Callable,
    noise_fn: Optional[Callable],
    observations: jnp.ndarray,
    rng: jnp.ndarray,
    mip_t_star: float,
    act_dim: int,
    use_constant_noise: bool = False,
    constant_noise_std: float = 0.01,
    min_noise_std: float = 0.08,
    randn_clip_value: float = 3.0,
    act_min: float = -1.0,
    act_max: float = 1.0,
    is_encoded: bool = False,
    params: Any = None,
    use_tanh_squash: bool = True,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """MIP 2-step stochastic sampling. Returns ``(actions, chains, logprob, sigmas)``.

    ``chains`` has shape ``(B, 3, A) = (x_0, a_0_hat, a_final)``.

    With ``use_tanh_squash=True`` (the default) the final step samples from a
    :class:`TanhMultivariateNormalDiag` so the returned action is a
    differentiable squash onto ``[act_min, act_max]`` and the accumulated
    ``logprob`` includes the ``-log|det(tanh')|`` Jacobian correction. This
    replaces the hard ``jnp.clip(actions, act_min, act_max)`` that previously
    zeroed the critic-path gradient under saturation and, together with the
    constant-sigma Normal reparam identity, killed actor learning in SAC.

    Passing ``use_tanh_squash=False`` restores the legacy hard-clipped path
    for parity against the oracle in ``mipq_single_file.py``.
    """
    batch_size = observations.shape[0]

    rng, key = jax.random.split(rng)
    x_0 = jax.random.normal(key, (batch_size, act_dim))
    chains = [x_0]
    sigmas = []

    init_dist = distrax.Normal(jnp.zeros((batch_size, act_dim)), 1.0)
    logprob = init_dist.log_prob(x_0).sum(-1)

    t_0 = jnp.zeros((batch_size, 1))
    if params is not None:
        mean_0 = actor_fn(observations, x_0, t_0, params=params, is_encoded=is_encoded)
    else:
        mean_0 = actor_fn(observations, x_0, t_0, is_encoded=is_encoded)

    if use_constant_noise:
        sigma_0 = constant_noise_std * jnp.ones((batch_size, act_dim))
    else:
        if params is not None:
            sigma_0 = noise_fn(observations, t_0, params=params)
        else:
            sigma_0 = noise_fn(observations, t_0)
        sigma_0 = jnp.maximum(sigma_0, min_noise_std)

    sigmas.append(sigma_0)

    step_0_dist = distrax.Normal(mean_0, sigma_0)
    rng, noise_key = jax.random.split(rng)
    a_0_hat = step_0_dist.sample(seed=noise_key)
    a_0_hat = jnp.clip(
        a_0_hat,
        mean_0 - randn_clip_value * sigma_0,
        mean_0 + randn_clip_value * sigma_0,
    )

    logprob = logprob + step_0_dist.log_prob(a_0_hat).sum(-1)
    chains.append(a_0_hat)

    t_star = jnp.full((batch_size, 1), mip_t_star)
    if params is not None:
        mean_final = actor_fn(
            observations, a_0_hat, t_star, params=params, is_encoded=is_encoded
        )
    else:
        mean_final = actor_fn(observations, a_0_hat, t_star, is_encoded=is_encoded)

    if use_constant_noise:
        sigma_final = constant_noise_std * jnp.ones((batch_size, act_dim))
    else:
        if params is not None:
            sigma_final = noise_fn(observations, t_star, params=params)
        else:
            sigma_final = noise_fn(observations, t_star)
        sigma_final = jnp.maximum(sigma_final, min_noise_std)

    sigmas.append(sigma_final)

    rng, noise_key = jax.random.split(rng)
    if use_tanh_squash:
        # Squash + proper Jacobian correction via the repo's TanhNormal wrapper.
        # ``sample_and_log_prob`` returns the post-squash action together with
        # the full transformed log-density (Normal log-prob minus
        # ``log|det(tanh')|``), so critic-path gradients survive saturation and
        # the entropy term is no longer a constant under ``mean_final``.
        step_final_dist = TanhMultivariateNormalDiag(
            loc=mean_final,
            scale_diag=sigma_final,
            low=jnp.asarray(act_min),
            high=jnp.asarray(act_max),
        )
        actions, lp_final = step_final_dist.sample_and_log_prob(seed=noise_key)
        logprob = logprob + lp_final
    else:
        step_final_dist = distrax.Normal(mean_final, sigma_final)
        actions = step_final_dist.sample(seed=noise_key)
        actions = jnp.clip(
            actions,
            mean_final - randn_clip_value * sigma_final,
            mean_final + randn_clip_value * sigma_final,
        )
        logprob = logprob + step_final_dist.log_prob(actions).sum(-1)
        actions = jnp.clip(actions, act_min, act_max)
    chains.append(actions)

    chains = jnp.stack(chains, axis=1)
    sigmas = jnp.stack(sigmas, axis=1)

    return actions, chains, logprob, sigmas


def compute_mip_log_prob(
    actor_fn: Callable,
    noise_fn: Optional[Callable],
    observations: jnp.ndarray,
    chains: jnp.ndarray,
    mip_t_star: float,
    use_constant_noise: bool = False,
    constant_noise_std: float = 0.01,
    min_noise_std: float = 0.08,
    normalize_horizon: bool = False,
    normalize_dim: bool = False,
    is_encoded: bool = False,
    params: Any = None,
    get_entropy: bool = False,
) -> Tuple[jnp.ndarray, Optional[jnp.ndarray], Dict]:
    """Log-prob of a MIP chain ``(B, 3, A) -> (x_0, a_0_hat, a_final)``."""
    batch_size = chains.shape[0]
    act_dim = chains.shape[-1]

    x_0 = chains[:, 0, :]
    a_0_hat = chains[:, 1, :]
    a_final = chains[:, 2, :]

    logprob = 0.0
    joint_entropy = 0.0
    logprob_steps = 0

    init_dist = distrax.Normal(jnp.zeros((batch_size, act_dim)), 1.0)
    logprob_init = init_dist.log_prob(x_0).sum(-1)
    logprob = logprob + logprob_init
    logprob_steps += 1

    if get_entropy:
        entropy_init = init_dist.entropy().sum(-1)
        joint_entropy = joint_entropy + entropy_init

    t_0 = jnp.zeros((batch_size, 1))
    if params is not None:
        mean_0 = actor_fn(observations, x_0, t_0, params=params, is_encoded=is_encoded)
    else:
        mean_0 = actor_fn(observations, x_0, t_0, is_encoded=is_encoded)

    if use_constant_noise:
        sigma_0 = constant_noise_std * jnp.ones((batch_size, act_dim))
    else:
        if params is not None:
            sigma_0 = noise_fn(observations, t_0, params=params)
        else:
            sigma_0 = noise_fn(observations, t_0)
        sigma_0 = jnp.maximum(sigma_0, min_noise_std)

    dist_0 = distrax.Normal(mean_0, sigma_0)
    logprob_0 = dist_0.log_prob(a_0_hat).sum(-1)
    logprob = logprob + logprob_0
    logprob_steps += 1

    if get_entropy:
        entropy_0 = dist_0.entropy().sum(-1)
        joint_entropy = joint_entropy + entropy_0

    t_star = jnp.full((batch_size, 1), mip_t_star)
    if params is not None:
        mean_final = actor_fn(
            observations, a_0_hat, t_star, params=params, is_encoded=is_encoded
        )
    else:
        mean_final = actor_fn(observations, a_0_hat, t_star, is_encoded=is_encoded)

    if use_constant_noise:
        sigma_final = constant_noise_std * jnp.ones((batch_size, act_dim))
    else:
        if params is not None:
            sigma_final = noise_fn(observations, t_star, params=params)
        else:
            sigma_final = noise_fn(observations, t_star)
        sigma_final = jnp.maximum(sigma_final, min_noise_std)

    dist_final = distrax.Normal(mean_final, sigma_final)
    logprob_final = dist_final.log_prob(a_final).sum(-1)
    logprob = logprob + logprob_final
    logprob_steps += 1

    if get_entropy:
        entropy_final = dist_final.entropy().sum(-1)
        joint_entropy = joint_entropy + entropy_final
        entropy_rate = joint_entropy / logprob_steps / act_dim
    else:
        entropy_rate = None

    if normalize_horizon:
        logprob = logprob / logprob_steps
    if normalize_dim:
        logprob = logprob / act_dim

    info = {
        "mean_0": mean_0.mean(),
        "mean_final": mean_final.mean(),
        "sigma_0": sigma_0.mean(),
        "sigma_final": sigma_final.mean(),
        "logprob_mean": logprob.mean(),
        "logprob_steps": logprob_steps,
    }

    if get_entropy:
        info["joint_entropy"] = joint_entropy.mean()
        info["entropy_rate"] = entropy_rate.mean()

    return logprob, entropy_rate, info


# ---------------------------------------------------------------------------
# MIP-Q training and inference
# ---------------------------------------------------------------------------


def get_mip_q_targets(
    target_q: jnp.ndarray,
    rng: jnp.ndarray,
    noise_scale: float = 1.0,
    t_star: float = 0.9,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """MIP-Q training targets (scalar space).

    Returns ``z_0`` (uniform noise), ``z_t_star = (1 - t_star) * z_0 + target_q``,
    and the two time arrays.
    """
    batch_size = target_q.shape[0]

    rng, noise_rng = jax.random.split(rng)
    z_0 = jax.random.uniform(
        noise_rng, (batch_size, 1), minval=-noise_scale, maxval=noise_scale
    )

    target_q_expanded = target_q[:, None]
    z_t_star = (1 - t_star) * z_0 + target_q_expanded

    t_0_arr = jnp.zeros((batch_size, 1))
    t_star_arr = jnp.full((batch_size, 1), t_star)

    return z_0, z_t_star, t_0_arr, t_star_arr


def sample_mip_q_ensemble_values(
    critic_fn: Callable,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    noise_sample: jnp.ndarray,
    mip_t_star: float = 0.9,
    is_encoded: bool = False,
) -> jnp.ndarray:
    """Two-step autoregressive sampling for :class:`ValueMIPEnsemble`.

    ``noise_sample`` has shape ``(B, 1)`` and is shared across ensemble members.
    Returns an array of shape ``(num_ensembles, B)``.
    """
    batch_size = observations.shape[0]

    t_0 = jnp.zeros((batch_size, 1))
    q_0 = critic_fn(
        observations,
        actions=actions,
        scalar_input=noise_sample,
        time=t_0,
        is_encoded=is_encoded,
    )  # (num_ensembles, B)

    t_star = jnp.full((batch_size, 1), mip_t_star)
    q_0_expanded = q_0.T[..., None]  # (B, num_ensembles, 1)

    q_final = critic_fn(
        observations,
        actions=actions,
        scalar_input=q_0_expanded,
        time=t_star,
        is_encoded=is_encoded,
    )
    return q_final


def compute_mip_q_ensemble_predictions(
    critic_fn: Callable,
    observations: jnp.ndarray,
    actions: jnp.ndarray,
    target_q: jnp.ndarray,
    rng: jnp.ndarray,
    noise_scale: float = 1.0,
    mip_t_star: float = 0.9,
    is_encoded: bool = False,
    return_logits: bool = False,
) -> Tuple:
    """Training-time predictions for explicit-ensemble MIP-Q (shared noise)."""
    z_0, z_t_star, t_0_arr, t_star_arr = get_mip_q_targets(
        target_q=target_q, rng=rng, noise_scale=noise_scale, t_star=mip_t_star
    )

    if return_logits:
        q_0, q_0_logits = critic_fn(
            observations,
            actions=actions,
            scalar_input=z_0,
            time=t_0_arr,
            is_encoded=is_encoded,
            return_logits=True,
        )
    else:
        q_0 = critic_fn(
            observations,
            actions=actions,
            scalar_input=z_0,
            time=t_0_arr,
            is_encoded=is_encoded,
            return_logits=False,
        )
        q_0_logits = None

    if return_logits:
        q_final, q_final_logits = critic_fn(
            observations,
            actions=actions,
            scalar_input=z_t_star,
            time=t_star_arr,
            is_encoded=is_encoded,
            return_logits=True,
        )
    else:
        q_final = critic_fn(
            observations,
            actions=actions,
            scalar_input=z_t_star,
            time=t_star_arr,
            is_encoded=is_encoded,
            return_logits=False,
        )
        q_final_logits = None

    if return_logits:
        return q_0, q_final, q_0_logits, q_final_logits
    else:
        return q_0, q_final


# ---------------------------------------------------------------------------
# Q aggregation and TD utilities
# ---------------------------------------------------------------------------


def aggregate_q_values(
    q_values: jnp.ndarray,
    method: str = "mean",
    rng: Optional[jnp.ndarray] = None,
    num_qs: Optional[int] = None,
    subsample_size: int = 2,
) -> jnp.ndarray:
    """Aggregate ``(num_qs, B) -> (B)``. ``method`` in ``{mean, min, subsample}``."""
    if method == "min":
        return q_values.min(axis=0)
    elif method == "subsample":
        if rng is None or num_qs is None:
            raise ValueError("rng and num_qs required for subsample aggregation")
        subsample_idxs = jax.random.randint(rng, (subsample_size,), 0, num_qs)
        return q_values[subsample_idxs].min(axis=0)
    else:
        return q_values.mean(axis=0)


def compute_td_target(
    rewards: jnp.ndarray,
    masks: jnp.ndarray,
    next_q: jnp.ndarray,
    discount: float,
    horizon_length: int = 1,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
) -> jnp.ndarray:
    """TD target: ``r + gamma^H * mask * Q(s', a')``."""
    if rewards.ndim > 1:
        rewards = rewards[..., -1]
    if masks.ndim > 1:
        masks = masks[..., -1]

    discount_factor = discount ** horizon_length
    target = rewards + discount_factor * masks * next_q

    if clip_min is not None or clip_max is not None:
        target = jnp.clip(
            target,
            a_min=clip_min if clip_min is not None else -jnp.inf,
            a_max=clip_max if clip_max is not None else jnp.inf,
        )
    return target


def compute_mip_mse_td_loss(
    q_0: jnp.ndarray,
    q_final: jnp.ndarray,
    target_q: jnp.ndarray,
    valid_mask: Optional[jnp.ndarray] = None,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """Two-step MSE TD loss for :class:`ValueMIPEnsemble` predictions.

    ``q_0`` and ``q_final`` each have shape ``(num_ensembles, B)``; ``target_q``
    has shape ``(B,)``; ``valid_mask`` has shape ``(B,)`` (or None). Returns
    ``(loss, info)`` where ``loss`` is the sum of step-1 regression and step-2
    denoising MSEs.
    """
    target_bc = target_q[None, :]
    if valid_mask is None:
        mask_bc = jnp.ones_like(target_bc)
    else:
        mask_bc = valid_mask[None, :]

    loss_regression = (jnp.square(q_0 - target_bc) * mask_bc).mean()
    loss_denoising = (jnp.square(q_final - target_bc) * mask_bc).mean()
    loss = loss_regression + loss_denoising

    info = {
        "mip_td_loss": loss,
        "mip_td_loss_regression": loss_regression,
        "mip_td_loss_denoising": loss_denoising,
        "q_0_mean": q_0.mean(),
        "q_final_mean": q_final.mean(),
    }
    return loss, info


# ---------------------------------------------------------------------------
# Wrapper modules
# ---------------------------------------------------------------------------


class MIPActor(nn.Module):
    """Two-step flow-matching actor.

    The inner backbone is an :class:`ActorVectorField`; this wrapper exposes
    the common actor method surface used by the residual learners:

      * ``sample(obs, rng)``         — stochastic 2-step SDE rollout, returns actions.
      * ``sample_with_logprob(obs, rng)`` — same rollout, returns ``(actions, logp)``.
      * ``mode(obs)``                — deterministic ODE rollout from zero noise.
      * ``compute_log_prob(obs, a)`` — raises :class:`NotImplementedError`.
      * ``bc_loss(obs, targets, valid_mask, rng)`` — two-time BC loss with
        action chunking over the target horizon.

    ``action_dim`` is the flat dimensionality of the predicted action vector
    (i.e. ``horizon_length * per_step_action_dim`` under chunking).
    """

    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    use_film: bool = False
    mip_t_star: float = 0.9
    use_constant_noise: bool = True
    constant_noise_std: float = 0.01
    min_noise_std: float = 0.08
    randn_clip_value: float = 3.0
    act_min: float = -1.0
    act_max: float = 1.0
    bc_t_star: float = 0.999

    def setup(self) -> None:
        self.vector_field = ActorVectorField(
            hidden_dims=self.hidden_dims,
            action_dim=self.action_dim,
            layer_norm=self.layer_norm,
            use_film=self.use_film,
        )

    def _actor_fn(self, obs, act, t, is_encoded: bool = False):
        return self.vector_field(obs, act, t, is_encoded=is_encoded)

    def __call__(self, observations, training: bool = False):
        raise NotImplementedError(
            "MIPActor has no distributional forward; "
            "use sample / sample_with_logprob / mode instead."
        )

    def sample(self, observations, rng, training: bool = False):
        observations = _obs_to_flat(observations)
        actions, _chains, _logprob, _sigmas = sample_mip_actions_sde(
            actor_fn=self._actor_fn,
            noise_fn=None,
            observations=observations,
            rng=rng,
            mip_t_star=self.mip_t_star,
            act_dim=self.action_dim,
            use_constant_noise=self.use_constant_noise,
            constant_noise_std=self.constant_noise_std,
            min_noise_std=self.min_noise_std,
            randn_clip_value=self.randn_clip_value,
            act_min=self.act_min,
            act_max=self.act_max,
        )
        return actions

    def sample_with_logprob(self, observations, rng, training: bool = False):
        observations = _obs_to_flat(observations)
        actions, _chains, logprob, _sigmas = sample_mip_actions_sde(
            actor_fn=self._actor_fn,
            noise_fn=None,
            observations=observations,
            rng=rng,
            mip_t_star=self.mip_t_star,
            act_dim=self.action_dim,
            use_constant_noise=self.use_constant_noise,
            constant_noise_std=self.constant_noise_std,
            min_noise_std=self.min_noise_std,
            randn_clip_value=self.randn_clip_value,
            act_min=self.act_min,
            act_max=self.act_max,
        )
        return actions, logprob

    def mode(self, observations):
        observations = _obs_to_flat(observations)
        batch_size = observations.shape[0]
        zero_noise = jnp.zeros((batch_size, self.action_dim))
        return sample_mip_actions_ode(
            actor_fn=self._actor_fn,
            observations=observations,
            rng=jax.random.PRNGKey(0),
            mip_t_star=self.mip_t_star,
            act_dim=self.action_dim,
            act_min=self.act_min,
            act_max=self.act_max,
            noises=zero_noise,
        )

    def compute_log_prob(self, observations, actions):
        raise NotImplementedError(
            "MIP actor does not support compute_log_prob on arbitrary actions"
        )

    def bc_loss(self, observations, targets, valid_mask, rng):
        observations = _obs_to_flat(observations)
        horizon_length = targets.shape[-2]
        per_step_action_dim = targets.shape[-1]
        batch = {"observations": observations, "actions": targets}
        if valid_mask is not None:
            batch["valid"] = valid_mask
        loss, _info = compute_mip_bc_loss(
            batch=batch,
            rng=rng,
            model_fn=self._actor_fn,
            action_chunking=True,
            horizon_length=horizon_length,
            action_dim=per_step_action_dim,
            t_star=self.bc_t_star,
        )
        return loss


class MIPCritic(nn.Module):
    """Explicit-ensemble MIP-Q critic.

    Wraps a :class:`ValueMIPEnsemble` and exposes:

      * ``__call__(obs, actions, rng=None)`` — step-2 Q-stack of shape
        ``(num_qs, B)``, used for actor Q evaluation and for critic reductions.
        When ``rng`` is ``None`` the scalar input is zero (deterministic eval).
      * ``td_loss(obs, actions, target_q, valid_mask, rng)`` — MSE TD loss
        computed via :func:`compute_mip_q_ensemble_predictions`.
    """

    hidden_dims: Sequence[int]
    num_qs: int = 2
    mip_t_star: float = 0.9
    mip_q_noise_scale: float = 1.0
    layer_norm: bool = True

    def setup(self) -> None:
        self.value_ensemble = ValueMIPEnsemble(
            hidden_dims=self.hidden_dims,
            num_ensembles=self.num_qs,
            mip_q_noise_scale=self.mip_q_noise_scale,
            mip_t_star=self.mip_t_star,
            layer_norm=self.layer_norm,
            critic_loss_type="mse",
        )

    def _critic_fn(
        self,
        observations,
        actions=None,
        scalar_input=None,
        time=None,
        is_encoded: bool = False,
        return_logits: bool = False,
    ):
        return self.value_ensemble(
            observations,
            actions=actions,
            scalar_input=scalar_input,
            time=time,
            is_encoded=is_encoded,
            return_logits=return_logits,
        )

    def __call__(self, observations, actions, rng=None, training: bool = False):
        observations = _obs_to_flat(observations)
        batch_size = observations.shape[0]
        if rng is None:
            noise_sample = jnp.zeros((batch_size, 1))
        else:
            noise_sample = jax.random.uniform(
                rng,
                (batch_size, 1),
                minval=-self.mip_q_noise_scale,
                maxval=self.mip_q_noise_scale,
            )
        return sample_mip_q_ensemble_values(
            critic_fn=self._critic_fn,
            observations=observations,
            actions=actions,
            noise_sample=noise_sample,
            mip_t_star=self.mip_t_star,
        )

    def td_loss(self, observations, actions, target_q, valid_mask, rng):
        observations = _obs_to_flat(observations)
        q_0, q_final = compute_mip_q_ensemble_predictions(
            critic_fn=self._critic_fn,
            observations=observations,
            actions=actions,
            target_q=target_q,
            rng=rng,
            noise_scale=self.mip_q_noise_scale,
            mip_t_star=self.mip_t_star,
        )
        return compute_mip_mse_td_loss(q_0, q_final, target_q, valid_mask)
