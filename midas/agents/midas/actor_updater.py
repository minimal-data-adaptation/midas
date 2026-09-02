"""MIDAS actor updates.

This module implements residual policy-agnostic updates where:
- The observation contains base_action from a frozen base policy (e.g., Pi-0.5)
- The actor outputs residual (delta) actions
- The executed action is a_exec = clip(base_action + alpha * delta, -1, 1)

Only MIDAS distillation and its optional BC warmup are retained.
"""

from typing import Dict, Tuple, Optional
import chex
import jax
import jax.numpy as jnp
from flax.training.train_state import TrainState

from midas.data.dataset import DatasetDict
from midas.types import Params, PRNGKey


_CAP_DISABLED_THRESHOLD = 1e5


def _clip_a_star_delta_from_base(a_star, base, cap, tanh_eps: float = 1e-6):
    """Clip the refined action to a per-dimension trust region around the base."""
    delta = a_star - base
    if cap is None:
        abs_max = jnp.max(jnp.abs(delta))
        zero = jnp.array(0.0, dtype=jnp.float32)
        return a_star, delta, {
            "pre_clip_abs_max": abs_max,
            "post_clip_abs_max": abs_max,
            "post_clip_abs_max_capped_dims": zero,
            "clip_fraction": zero,
        }

    cap_b = cap.reshape((1,) * (delta.ndim - 1) + (-1,))
    delta_capped = jnp.clip(delta, -cap_b, cap_b)
    a_star_capped = jnp.clip(base + delta_capped, -1.0 + tanh_eps, 1.0 - tanh_eps)
    capped_mask = (cap < _CAP_DISABLED_THRESHOLD).astype(jnp.float32)
    capped_mask_b = capped_mask.reshape((1,) * (delta.ndim - 1) + (-1,))
    over = (jnp.abs(delta) > cap_b).astype(jnp.float32) * capped_mask_b
    leading = 1
    for dimension in delta.shape[:-1]:
        leading *= int(dimension)
    denominator = capped_mask.sum() * jnp.asarray(leading, dtype=jnp.float32)
    clip_fraction = over.sum() / jnp.maximum(
        denominator, jnp.array(1.0, dtype=jnp.float32)
    )
    post_clip_abs = jnp.abs(a_star_capped - base)
    return a_star_capped, delta, {
        "pre_clip_abs_max": jnp.max(jnp.abs(delta)),
        "post_clip_abs_max": jnp.max(post_clip_abs),
        "post_clip_abs_max_capped_dims": jnp.max(post_clip_abs * capped_mask_b),
        "clip_fraction": clip_fraction,
    }


def _apply_actor_trust_region(
    a_star,
    base,
    cap,
    use_trust_region: bool,
    tanh_eps: float = 1e-6,
):
    """Apply actor-target clipping only when the trust region is enabled."""
    if not use_trust_region:
        return _clip_a_star_delta_from_base(a_star, base, None, tanh_eps)
    if cap is None:
        raise ValueError("A trust-region cap is required when clipping is enabled")
    return _clip_a_star_delta_from_base(a_star, base, cap, tanh_eps)


def _apply_inputs(ts, params=None):
    """Build variables dict for apply_fn, threading batch_stats if present."""
    coll = {'params': ts.params if params is None else params}
    if getattr(ts, 'batch_stats', None) is not None:
        coll['batch_stats'] = ts.batch_stats
    return coll


def _flat(x):
    return x.reshape(x.shape[0], -1)

def _cosine_sim(a, b, eps=1e-8):
    a = _flat(a)
    b = _flat(b)
    an = jnp.linalg.norm(a, axis=-1)
    bn = jnp.linalg.norm(b, axis=-1)
    return (jnp.sum(a * b, axis=-1) / (an * bn + eps))


def _soft_clip(x, lo=-1.0, hi=1.0, margin=0.05):
    """Differentiable soft-clip using tanh at boundaries.
    
    Inside [lo+margin, hi-margin] this is identity.
    Outside, it smoothly saturates toward lo/hi via tanh.
    Gradient is always non-zero, unlike jnp.clip.
    """
    mid = (hi + lo) / 2.0
    half_range = (hi - lo) / 2.0
    # Normalize to [-1, 1] range
    x_norm = (x - mid) / half_range
    # Apply tanh-based soft saturation
    return mid + half_range * jnp.tanh(x_norm)


def _safe_clip_for_log_prob(x, eps=1e-6):
    """Clamp actions to (-1+eps, 1-eps) to avoid atanh(±1) = ±inf in log_prob.

    This is used when computing log_prob of stored/clipped actions that may
    sit exactly at ±1.0 due to hard clipping during collection.
    """
    return jnp.clip(x, -1.0 + eps, 1.0 - eps)


def _actor_variables(actor, actor_params):
    """Variables dict for an actor ``apply_fn`` call (threads batch_stats)."""
    coll = {'params': actor_params}
    if getattr(actor, 'batch_stats', None) is not None:
        coll['batch_stats'] = actor.batch_stats
    return coll


def _actor_sample_with_logprob(actor, actor_params, observations, key, is_mip_actor):
    """Sample actions + log-probs for either a Gaussian or MIP actor.

    For Gaussian actors, returns ``(actions, log_probs, dist, new_model_state)``
    where ``dist`` is the distrax distribution (used for diagnostics). For MIP
    actors, ``dist`` is ``None`` — the MIP wrapper has no closed-form
    distribution. Either path threads ``batch_stats`` when present.
    """
    coll = _actor_variables(actor, actor_params)
    has_bs = 'batch_stats' in coll
    if is_mip_actor:
        if has_bs:
            (actions, log_probs), new_model_state = actor.apply_fn(
                coll, observations, key,
                mutable=['batch_stats'], method='sample_with_logprob',
            )
        else:
            actions, log_probs = actor.apply_fn(
                coll, observations, key, method='sample_with_logprob',
            )
            new_model_state = {}
        return actions, log_probs, None, new_model_state

    if has_bs:
        dist, new_model_state = actor.apply_fn(
            coll, observations, mutable=['batch_stats'],
        )
    else:
        dist = actor.apply_fn(coll, observations)
        new_model_state = {}
    if isinstance(dist, tuple):
        dist = dist[0]
    actions, log_probs = dist.sample_and_log_prob(seed=key)
    return actions, log_probs, dist, new_model_state


def _actor_sample_only(actor, actor_params, observations, key, is_mip_actor):
    """Sample-only variant — used where log-prob is not needed (e.g. MIDAS candidates)."""
    coll = _actor_variables(actor, actor_params)
    if is_mip_actor:
        return actor.apply_fn(coll, observations, key, method='sample')
    dist = actor.apply_fn(coll, observations)
    if isinstance(dist, tuple):
        dist = dist[0]
    return dist.sample(seed=key)


def compute_bc_loss_residual(
    dist,
    batch: DatasetDict,
    query_frequency: int,
    bc_on_success_only: bool = False,
    key: PRNGKey | None = None,
) -> Tuple[jnp.ndarray, Dict[str, float]]:
    """Compute sample-based BC MSE against actions stored in the replay buffer."""
    del query_frequency
    if key is None:
        raise ValueError("A PRNG key is required for reparameterized BC sampling")
    stored_actions = batch["actions"]
    stored_actions_flat = stored_actions.reshape(stored_actions.shape[0], -1)
    policy_sample = dist.sample(seed=key)
    policy_mode = dist.mode()
    mse_per_sample = jnp.mean((policy_sample - stored_actions_flat) ** 2, axis=-1)
    if bc_on_success_only:
        success_mask = batch["success_flag"]
        bc_loss = jnp.sum(mse_per_sample * success_mask) / (jnp.sum(success_mask) + 1e-8)
        bc_fraction = jnp.mean(success_mask)
    else:
        bc_loss = jnp.mean(mse_per_sample)
        bc_fraction = jnp.array(1.0, dtype=bc_loss.dtype)
    return bc_loss, {
        "bc/loss": bc_loss,
        "bc/mse_mean": jnp.mean(mse_per_sample),
        "bc/mse_max": jnp.max(mse_per_sample),
        "bc/mse_min": jnp.min(mse_per_sample),
        "bc/sample_mean": jnp.mean(policy_sample),
        "bc/sample_std": jnp.std(policy_sample),
        "bc/mode_mean": jnp.mean(policy_mode),
        "bc/mode_std": jnp.std(policy_mode),
        "bc/stored_action_mean": jnp.mean(stored_actions_flat),
        "bc/stored_action_std": jnp.std(stored_actions_flat),
        "bc/success_frac_in_batch": bc_fraction,
    }


def _compute_bc_reg_loss(
    actor, actor_params, dist, batch, query_frequency,
    bc_on_success_only, key, is_mip_actor,
):
    """Dispatch BC regularization loss to the correct path.

    Gaussian: the existing ``compute_bc_loss_residual`` (sample-MSE against the
    already-constructed ``dist``). MIP: ``method='bc_loss'``, passing the
    stored actions in chunked ``(B, H, A)`` form that the MIP helpers expect.
    """
    if is_mip_actor:
        stored_actions_chunked = batch['actions']  # (B, H, A)
        if bc_on_success_only:
            valid_mask = batch['success_flag']  # (B,) — MIP BC ignores this axis
            if valid_mask.ndim == 1:
                valid_mask = jnp.broadcast_to(
                    valid_mask[:, None], stored_actions_chunked.shape[:2]
                )
        else:
            valid_mask = jnp.ones(stored_actions_chunked.shape[:2])
        loss = actor.apply_fn(
            _actor_variables(actor, actor_params),
            batch['observations'], stored_actions_chunked, valid_mask, key,
            method='bc_loss',
        )
        info = {'bc/loss': loss}
        return loss, info

    return compute_bc_loss_residual(
        dist, batch, query_frequency, bc_on_success_only, key=key,
    )


def _empty_gaussian_log_stats(actions_sampled, B):
    """Placeholder Gaussian-style diagnostics when the actor has no distrax dist.

    The keys are kept so the WandB schema stays consistent across arches;
    values are zero or NaN where no meaningful quantity exists.
    """
    zero_scalar = jnp.zeros(())
    zero_b = jnp.zeros((B,))
    actions_flat = actions_sampled.reshape(B, -1)
    return {
        'mean_dist': actions_flat,
        'std_diag_dist': jnp.zeros_like(actions_flat),
        'mean_dist_norm': jnp.linalg.norm(actions_flat, axis=-1),
        'std_dist_norm': zero_b,
        'base_entropy_mean': zero_scalar,
    }


def _nan_to_num_tree(tree):
    """Apply nan_to_num to all leaves in a pytree.
    
    Safety net only — this should ideally never be triggered.
    Uses posinf=0, neginf=0 to avoid injecting large values that
    cause secondary parameter explosions.
    """
    return jax.tree_util.tree_map(
        lambda x: jnp.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), 
        tree
    )


def update_actor_midas(
        key: PRNGKey,
        actor: TrainState,
        critic: TrainState,
        batch: DatasetDict,
        residual_alpha: float,
        query_frequency: int,
        action_dim: int,
        midas_num_samples: int = 16,
        midas_num_elites: int = 4,
        midas_num_grad_steps: int = 5,
        midas_step_size: float = 0.01,
        critic_reduction: str = 'min',
        predict_a_exec: bool = False,
        bc_flag: bool = False,
        bc_reg_coeff: float = 0.0,
        bc_on_success_only: bool = False,
        is_mip_actor: bool = False,
        b_o_n: bool = True,
        grad_a_q: bool = True,
        use_trust_region: bool = False,
        a_star_delta_clip_norm: Optional[jnp.ndarray] = None,
) -> Tuple[TrainState, Dict[str, float]]:
    """Update actor via Policy-Agnostic RL (MIDAS) for residual setting.
    
    MIDAS decouples actor training from policy gradient entirely:
    1. Sample N action candidates from the current actor pi_theta(s),
       plus the base policy action as an additional (N+1)-th candidate
    2. Evaluate all N+1 with critic Q(s, a_exec), keep top-K elites
    3. Refine elites via gradient ascent: a <- a + eta * grad_a Q(s, a)
    4. Pick the best refined action as the distillation target a*
    5. Train the actor to imitate a* via MSE loss (BC distillation)
    
    Including the base policy action ensures the residual never degrades
    below the base policy's quality — if the base action has highest Q,
    it will be selected as the distillation target.
    
    The actor never differentiates through the critic — the critic only
    provides stop-gradient targets. This avoids issues with tanh squashing,
    action clipping killing gradients, etc.
    
    Args:
        key: PRNG key.
        actor: Current actor TrainState.
        critic: Critic TrainState (used for Q-evaluation and gradient ascent).
        batch: Batch of transitions.
        residual_alpha: Scaling factor for residual actions.
        query_frequency: Chunk length for action queries.
        action_dim: Action dimension per step.
        midas_num_samples: N, number of action candidates to sample from actor.
        midas_num_elites: K, number of top-Q actions to keep for refinement.
        midas_num_grad_steps: Number of gradient ascent steps on Q w.r.t. action.
        midas_step_size: Step size (learning rate) for gradient ascent on actions.
        critic_reduction: How to reduce Q ensemble ('min' or 'mean').
        predict_a_exec: If True, actor predicts a_exec directly (not delta).
        use_trust_region: Whether to constrain refined actor targets around the
            base action. This is intended for real-world training and defaults
            to disabled for simulation.
        a_star_delta_clip_norm: Per-action-dimension trust-region radii.
        
    Returns:
        Updated actor TrainState and info dict.
    """
    key, sample_key, select_key = jax.random.split(key, 3)
    
    # Extract base actions from observations
    base_action_raw = batch['observations']['base_action']
    base_action = jnp.squeeze(base_action_raw, axis=-1)  # (B, T, A)
    B, T, A = base_action.shape
    action_dim_flat = query_frequency * action_dim
    N = midas_num_samples
    K = midas_num_elites
    
    # ----------------------------------------------------------------
    # Step 1: Sample N action candidates from current actor (frozen)
    # ----------------------------------------------------------------
    # Polymorphic sampling: Gaussian builds a dist and samples; MIP routes
    # through ``method='head_sample'``. In both cases we stop-gradient actor
    # params so the proposals are treated as a frozen distribution.
    #
    # The actor encoder runs once on raw observations; the N candidates then
    # share the cached features. ResNet-50 / DINOv2 encoders are 25M / 86M
    # params, so collapsing N encoder fwds → 1 fwd is a meaningful win even
    # before the dominant savings inside the refinement loop below.
    #
    # When ``b_o_n=False`` we skip stochastic candidate sampling entirely and
    # use only the base policy action as the candidate set. The actor encoder
    # forward is also skipped in that branch.
    if b_o_n:
        frozen_actor_params = jax.lax.stop_gradient(actor.params)
        sample_keys = jax.random.split(sample_key, N)

        actor_sample_coll = _actor_variables(actor, frozen_actor_params)
        actor_features_frozen = actor.apply_fn(
            actor_sample_coll, batch['observations'],
            method='encode_only', training=False,
        )

        if is_mip_actor:
            def sample_one(k):
                return actor.apply_fn(
                    actor_sample_coll, actor_features_frozen, k,
                    method='head_sample', training=False,
                )  # (B, action_dim_flat)
            delta_candidates = jax.vmap(sample_one)(sample_keys)
        else:
            dist = actor.apply_fn(
                actor_sample_coll, actor_features_frozen, method='head_call',
            )
            if isinstance(dist, tuple):
                dist = dist[0]
            delta_candidates = jax.vmap(lambda k: dist.sample(seed=k))(sample_keys)

        delta_candidates = jax.lax.stop_gradient(delta_candidates)
    else:
        delta_candidates = None
    
    # ----------------------------------------------------------------
    # Step 2: Evaluate Q for all candidates → keep top-K elites
    # ----------------------------------------------------------------
    # Include the base policy action as an additional candidate.
    # The base action IS already a valid a_exec (no residual needed).
    base_a_exec_flat = jnp.clip(
        base_action[:, :query_frequency, :], -1.0, 1.0
    ).reshape(B, action_dim_flat)  # (B, action_dim_flat)
    
    def delta_to_a_exec_flat(delta_flat):
        """Convert actor output (delta or a_exec) to executed action for critic."""
        delta_chunked = delta_flat.reshape(B, query_frequency, action_dim)
        if predict_a_exec:
            a_exec = jnp.clip(delta_chunked, -1.0, 1.0)
        else:
            a_exec = jnp.clip(
                base_action[:, :query_frequency, :] + residual_alpha * delta_chunked,
                -1.0, 1.0
            )
        return a_exec.reshape(B, action_dim_flat)
    
    # Encode critic observations once. Inside this actor update the critic is
    # treated as constant (no gradient w.r.t. critic params), so the encoded
    # features can be reused across all N+1 candidate Q-evaluations, the
    # vmap'd refinement loop's gradient ascent, and the post-refinement Q
    # eval. This collapses ~265 critic encoder fwds + ~240 bwds per outer
    # actor step down to a single forward.
    critic_coll = _apply_inputs(critic)
    critic_features = critic.apply_fn(
        critic_coll, batch['observations'],
        method='encode_only', training=False,
    )

    def compute_q(a_exec_flat):
        """Evaluate critic head on a_exec_flat (B, action_dim_flat) → (B,)."""
        qs = critic.apply_fn(
            critic_coll, critic_features, a_exec_flat,
            method='head_call', training=False,
        )  # (num_qs, B)
        if critic_reduction == 'min':
            return qs.min(axis=0)
        else:
            return qs.mean(axis=0)
    
    # Always evaluate the base policy candidate — it's appended either as the
    # (N+1)-th candidate (when b_o_n=True) or as the only candidate (b_o_n=False).
    q_base = compute_q(base_a_exec_flat)  # (B,)

    if b_o_n:
        # Evaluate all N actor candidates: (N, B, action_dim_flat) → (N, B)
        a_exec_actor = jax.vmap(delta_to_a_exec_flat)(delta_candidates)  # (N, B, action_dim_flat)
        q_actor = jax.vmap(compute_q)(a_exec_actor)  # (N, B)

        # Append base policy action as the (N+1)-th candidate.
        a_exec_candidates = jnp.concatenate(
            [a_exec_actor, base_a_exec_flat[None, :, :]], axis=0
        )  # (N+1, B, action_dim_flat)
        q_candidates = jnp.concatenate(
            [q_actor, q_base[None, :]], axis=0
        )  # (N+1, B)
        K_eff = K
        base_candidate_index = N      # base is the (N+1)-th candidate
    else:
        # Best-of-N disabled: base policy is the only candidate.
        a_exec_candidates = base_a_exec_flat[None, :, :]   # (1, B, action_dim_flat)
        q_candidates = q_base[None, :]                     # (1, B)
        K_eff = 1
        base_candidate_index = 0      # base is the only candidate

    # Select top-K per batch element. Transpose to (B, M) for per-batch sorting.
    q_candidates_T = q_candidates.T                                          # (B, M)
    a_exec_candidates_T = jnp.transpose(a_exec_candidates, (1, 0, 2))         # (B, M, A_flat)

    top_k_indices = jnp.argsort(q_candidates_T, axis=-1)[:, -K_eff:]          # (B, K_eff)

    # Gather elite actions: (B, K_eff, action_dim_flat)
    elite_actions = jnp.take_along_axis(
        a_exec_candidates_T,
        top_k_indices[:, :, None],
        axis=1,
    )

    # Q values before refinement (for logging)
    elite_q_before = jnp.take_along_axis(q_candidates_T, top_k_indices, axis=1)  # (B, K_eff)

    # Track how often the base policy action is among elites. Its index in the
    # candidate set is ``N`` when b_o_n=True and ``0`` when b_o_n=False.
    base_in_elites = (top_k_indices == base_candidate_index).any(axis=-1).mean()
    
    # ----------------------------------------------------------------
    # Step 3: Gradient ascent on Q w.r.t. a_exec for each elite
    # ----------------------------------------------------------------
    # We work in a_exec space (flattened). The gradient ascent is:
    #   a_exec <- clip(a_exec + step_size * grad_a Q(s, a_exec), -1, 1)
    
    def q_sum_fn(a_exec_flat):
        """Scalar Q sum for gradient computation. Uses cached critic features."""
        qs = critic.apply_fn(
            critic_coll, critic_features, a_exec_flat,
            method='head_call', training=False,
        )
        if critic_reduction == 'min':
            return qs.min(axis=0).sum()
        else:
            return qs.mean(axis=0).sum()

    grad_q_fn = jax.grad(q_sum_fn, argnums=0)

    def refine_one_elite(a_exec_flat_BK):
        """Run gradient ascent on a single elite action set (B, action_dim_flat).

        Body is head-only — the encoder forward and backward are hoisted out of
        the loop via the closed-over ``critic_features``.
        """
        def body_fn(_, a):
            grad = grad_q_fn(a)
            a = a + midas_step_size * grad
            a = jnp.clip(a, -1.0, 1.0)
            return a

        return jax.lax.fori_loop(0, midas_num_grad_steps, body_fn, a_exec_flat_BK)
    
    # Reshape elites for vmap over K_eff: (K_eff, B, action_dim_flat)
    elite_actions_KBD = jnp.transpose(elite_actions, (1, 0, 2))

    # Refine all K_eff elites in parallel — unless grad_a_q is disabled, in
    # which case we distill directly to the best pre-refinement candidate.
    if grad_a_q:
        refined_elites_KBD = jax.vmap(refine_one_elite)(elite_actions_KBD)
    else:
        refined_elites_KBD = elite_actions_KBD

    # ----------------------------------------------------------------
    # Optional trust-region cap defines the feasible action set.
    # ----------------------------------------------------------------
    # Cap every refined elite BEFORE Q evaluation and the argmax so that the
    # actor distills toward an action that actually lives inside the cap, not
    # toward an unbounded high-Q action which is then post-projected to a
    # different point. Applied uniformly for both grad_a_q=True and =False
    # (and b_o_n=False, where the only elite is the base candidate — capping
    # the base against itself is a no-op, so no harm).
    #
    # Use the CLIPPED base (a_exec space, [-1, 1]) as the cap reference. The
    # candidates already live in the tanh-safe range; if raw base ever leaves
    # [-1, 1] then capping against raw base, followed by the helper's
    # tanh-safe re-clip, would let the realized delta exceed the nominal cap.
    # base_a_exec_flat is the same clipped base used to build the candidates,
    # so this keeps the cap and candidate spaces aligned.
    base_chunked = base_a_exec_flat.reshape(B, query_frequency, action_dim)
    refined_elites_BKTA = jnp.transpose(
        refined_elites_KBD.reshape(K_eff, B, query_frequency, action_dim),
        (1, 0, 2, 3),
    )                                                                # (B, K, T, A)
    base_BKTA = base_chunked[:, None, :, :]                          # (B, 1, T, A) broadcasts
    refined_targets_BKTA, _, candidate_clip_stats = _apply_actor_trust_region(
        refined_elites_BKTA,
        base_BKTA,
        a_star_delta_clip_norm,
        use_trust_region,
    )
    refined_targets_KBD = jnp.transpose(
        refined_targets_BKTA, (1, 0, 2, 3),
    ).reshape(K_eff, B, action_dim_flat)

    # ----------------------------------------------------------------
    # Step 4: Pick the best refined action as the distillation target
    # ----------------------------------------------------------------
    refined_q = jax.vmap(compute_q)(refined_targets_KBD)              # (K_eff, B)
    refined_q_T = refined_q.T                                         # (B, K_eff)
    refined_elites_BKD = jnp.transpose(refined_targets_KBD, (1, 0, 2)) # (B, K_eff, A_flat)

    best_idx = jnp.argmax(refined_q_T, axis=-1)                       # (B,)
    best_a_exec = jnp.take_along_axis(
        refined_elites_BKD,
        best_idx[:, None, None],
        axis=1,
    ).squeeze(axis=1)                                                  # (B, A_flat)

    best_q = refined_q_T[jnp.arange(B), best_idx]                     # (B,)

    # Selected-target diagnostics. ``candidate_clip_stats`` describes the
    # K_eff *candidates* (pre-selection) — useful for "how aggressive were
    # the proposals?". For "how far did the actually-distilled target sit
    # from base?" we recompute on the selected best_a_exec only. Both stats
    # use the clipped base, matching the cap reference. The selected target
    # is inside the feasible set when the trust region is enabled, so its
    # capped-dims abs-max is the realized trust-region utilisation (and is
    # <= cap whenever the cap is on).
    _, _, selected_clip_stats = _apply_actor_trust_region(
        best_a_exec.reshape(B, query_frequency, action_dim),
        base_chunked,
        a_star_delta_clip_norm,
        use_trust_region,
    )

    # Convert best a_exec back to delta (the actor's output space) for distillation.
    # ``best_a_exec`` has already passed through the optional trust region;
    # no further projection is needed.
    best_a_exec_chunked = best_a_exec.reshape(B, query_frequency, action_dim)
    if predict_a_exec:
        # Actor predicts a_exec directly; target IS a_exec (clipped to tanh range)
        bc_target = jnp.clip(best_a_exec_chunked, -1.0 + 1e-6, 1.0 - 1e-6).reshape(B, action_dim_flat)
    else:
        # Actor predicts delta; back-derive: delta = (a_exec - base) / alpha
        bc_target_delta = (best_a_exec_chunked - base_action[:, :query_frequency, :]) / (residual_alpha + 1e-8)
        # Clip to actor output range (TanhNormal outputs in (-1, 1))
        bc_target = jnp.clip(bc_target_delta, -1.0 + 1e-6, 1.0 - 1e-6).reshape(B, action_dim_flat)
    
    # Stop gradient on target
    bc_target = jax.lax.stop_gradient(bc_target)
    
    # ----------------------------------------------------------------
    # Step 5: Distill into actor via MSE loss
    # ----------------------------------------------------------------
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[Dict[str, float], Dict]]:
        key_sample, key_bc, key_diag = jax.random.split(key, 3)

        # Distillation loss — polymorphic between Gaussian and MIP.
        # - Gaussian: sample-based MSE against the refined elite (reparam so
        #   gradient flows to both mean and std).
        # - MIP: the actor has no closed-form distribution to MSE against, and
        #   plain sample-MSE undertrains the flow. Use ``method='bc_loss'`` on
        #   the refined target instead — two-time regression+denoising is the
        #   native MIP-BC objective, applied here with ``best_a_exec`` as the
        #   target chunk.
        if is_mip_actor:
            dist = None
            new_model_state = {}
            bc_target_chunked = bc_target.reshape(B, query_frequency, action_dim)
            valid_mask_mip = jnp.ones((B, query_frequency))
            mse_loss = actor.apply_fn(
                _actor_variables(actor, actor_params),
                batch['observations'],
                bc_target_chunked,
                valid_mask_mip,
                key_sample,
                method='bc_loss',
            )
            # Sample a rollout for diagnostics only (mode/std stats etc.).
            sampled_actions = _actor_sample_only(
                actor, actor_params, batch['observations'], key_diag, True,
            )
        else:
            dist = actor.apply_fn(_apply_inputs(actor, actor_params), batch['observations'])
            if isinstance(dist, tuple):
                dist = dist[0]
            sampled_actions = dist.sample(seed=key_sample)  # (B, action_dim_flat)
            new_model_state = {}
            mse_loss = jnp.mean((sampled_actions - bc_target) ** 2)

        # L1 (MAE) to the refined-elite distillation target — logging-only,
        # used for the tqdm postfix during the async loop. MIP path runs its
        # own internal regression so we surface mse_loss as the L1 proxy
        # (sampled_actions is the diagnostic rollout, NOT the trained signal).
        if is_mip_actor:
            l1_loss = mse_loss
        else:
            l1_loss = jnp.mean(jnp.abs(sampled_actions - bc_target))

        # BC regularization (MSE to stored actions in the replay buffer)
        bc_loss_val = jnp.array(0.0)
        bc_info = {}
        if bc_flag:
            bc_loss_val, bc_info = _compute_bc_reg_loss(
                actor, actor_params, dist, batch, query_frequency,
                bc_on_success_only, key_bc, is_mip_actor,
            )
        actor_loss = mse_loss + bc_reg_coeff * bc_loss_val

        # Also compute mode-based MSE for logging (Gaussian only; MIP mode
        # requires a separate deterministic ODE rollout we skip here).
        if is_mip_actor:
            policy_mode = sampled_actions
        else:
            policy_mode = dist.mode()
        mode_mse = jnp.mean((policy_mode - bc_target) ** 2)

        # Distribution diagnostics — Gaussian-only; MIP emits zero placeholders.
        if is_mip_actor:
            gauss_stats = _empty_gaussian_log_stats(sampled_actions, B)
            mean_dist = gauss_stats['mean_dist']
            std_diag_dist = gauss_stats['std_diag_dist']
            log_std_dist = jnp.zeros_like(std_diag_dist)
            mean_entropy = jnp.zeros(())
        else:
            mean_dist = dist.distribution._loc
            std_diag_dist = dist.distribution._scale_diag
            log_std_dist = jnp.log(std_diag_dist + 1e-8)
            base_entropy = dist.distribution.entropy()
            mean_entropy = base_entropy.mean()

        # Compute a_exec from sampled actions for diagnostics
        sampled_chunked = sampled_actions.reshape(B, query_frequency, action_dim)
        if predict_a_exec:
            a_exec_diag = jnp.clip(sampled_chunked, -1.0, 1.0)
            delta_diag = a_exec_diag - base_action[:, :query_frequency, :]
        else:
            delta_diag = sampled_chunked
            a_exec_diag = jnp.clip(
                base_action[:, :query_frequency, :] + residual_alpha * sampled_chunked,
                -1.0, 1.0
            )

        delta_norm = jnp.linalg.norm(delta_diag.reshape(B, -1), axis=-1)
        base_flat = base_action[:, :query_frequency, :].reshape(B, -1)
        base_norm = jnp.linalg.norm(base_flat, axis=-1)
        eff_delta_norm = jnp.where(predict_a_exec, delta_norm, jnp.abs(residual_alpha) * delta_norm)
        clipping_rate = (jnp.abs(a_exec_diag) >= 1.0).mean()

        # Target diagnostics
        bc_target_chunked = bc_target.reshape(B, query_frequency, action_dim)
        target_norm = jnp.linalg.norm(bc_target, axis=-1)

        info = {
            'actor_loss': actor_loss,
            'actor_l1_loss': l1_loss,
            'parl/mse_loss': mse_loss,
            'parl/l1_loss': l1_loss,
            'parl/mode_mse': mode_mse,
            'parl/q_before_refinement': elite_q_before.mean(),
            'parl/q_after_refinement': best_q.mean(),
            # ``q_improvement`` is best_q (post-refinement, post-cap) minus the
            # best elite Q before refinement. With the trust-region cap active
            # this can be negative — the cap can project a high-Q refined
            # candidate down into the feasible set, lowering its Q below the
            # unconstrained pre-refinement best.
            'parl/q_improvement': (best_q - elite_q_before.max(axis=-1)).mean(),
            'parl/target_norm_mean': target_norm.mean(),
            'parl/target_mean': bc_target.mean(),
            'parl/target_std': jnp.std(bc_target),
            'parl/num_samples': float(N),
            'parl/num_elites': float(K),
            'parl/num_elites_effective': jnp.float32(K_eff),
            'parl/num_grad_steps': float(midas_num_grad_steps),
            'parl/step_size': midas_step_size,
            'parl/base_in_elites_frac': base_in_elites,
            'parl/base_q': q_base.mean(),
            'parl/b_o_n': jnp.float32(b_o_n),
            'parl/grad_a_q': jnp.float32(grad_a_q),
            'parl/trust_region_enabled': jnp.float32(use_trust_region),
            # Candidate-level stats: aggregated across all K_eff refined
            # elites, before argmax-Q selection. Useful for "how aggressive
            # were the proposals?"
            'parl/candidate_delta_pre_clip_abs_max_all': candidate_clip_stats['pre_clip_abs_max'],
            'parl/candidate_delta_post_clip_abs_max_all': candidate_clip_stats['post_clip_abs_max'],
            'parl/candidate_delta_post_clip_abs_max_capped_dims': candidate_clip_stats['post_clip_abs_max_capped_dims'],
            'parl/candidate_clip_fraction': candidate_clip_stats['clip_fraction'],
            # Selected-target stats: realized distillation target only. This
            # is the metric that should be ``<= cap`` on capped dims whenever
            # the cap is active.
            'parl/a_star_delta_pre_clip_abs_max_all': selected_clip_stats['pre_clip_abs_max'],
            'parl/a_star_delta_post_clip_abs_max_all': selected_clip_stats['post_clip_abs_max'],
            'parl/a_star_delta_post_clip_abs_max_capped_dims': selected_clip_stats['post_clip_abs_max_capped_dims'],
            'parl/a_star_clip_fraction': selected_clip_stats['clip_fraction'],
            'bc/reg_coeff': bc_reg_coeff,
            'bc/weighted_loss': bc_reg_coeff * bc_loss_val,
            **bc_info,
            'entropy': mean_entropy,
            'mean_pi_norm': jnp.linalg.norm(mean_dist, axis=-1).mean(),
            'std_pi_norm': jnp.linalg.norm(std_diag_dist, axis=-1).mean(),
            'mean_pi_avg': mean_dist.mean(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_min': std_diag_dist.min(),
            'std_pi_max': std_diag_dist.max(),
            'log_std_mean': log_std_dist.mean(),
            'log_std_min': log_std_dist.min(),
            'log_std_max': log_std_dist.max(),
            'actor/delta_norm_mean': delta_norm.mean(),
            'actor/clipping_rate': clipping_rate,
            'actor/effective_residual_norm': eff_delta_norm.mean(),
            'collapse/ratio_eff_delta_to_base_mean': (eff_delta_norm / (base_norm + 1e-8)).mean(),
        }

        return actor_loss, (info, new_model_state)
    
    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    # NaN guard
    grads = _nan_to_num_tree(grads)
    
    new_actor = actor.apply_gradients(grads=grads)
    
    return new_actor, info


def update_actor_bc_residual(
        key: PRNGKey,
        actor: TrainState,
        batch: DatasetDict,
        query_frequency: int,
        predict_a_exec: bool = False,
        is_mip_actor: bool = False,
) -> Tuple[TrainState, Dict[str, float]]:
    """BC warmup actor update: train actor to reproduce batch['actions'] via MSE.

    The BC target comes directly from batch['actions'] — whatever was stored
    in the replay buffer at collection time.  This handles both online and
    demo data correctly:

    - Online BC warmup (force_zero_residual=True, delta mode): stored actions
      are zeros → actor learns zero residual.
    - Online BC warmup (force_zero_residual=True, a_exec mode): stored actions
      are clip(base, -1, 1) → actor learns base.
    - Demo data (delta mode): stored actions are (demo - base) / alpha →
      actor learns the residual needed to reproduce the demonstration.
    - Demo data (a_exec mode): stored actions are demo_action → actor learns
      the demo action directly.

    Args:
        key: PRNG key.
        actor: Actor TrainState.
        batch: Batch of transitions containing:
            - actions: (B, query_freq, action_dim) stored actions (BC target)
            - observations['base_action']: (B, chunk_len, action_dim, 1)
        query_frequency: Chunk length for action queries.
        predict_a_exec: Whether actor predicts a_exec (True) or delta (False).
            Used for diagnostic computation and target clipping.

    Returns:
        Updated actor TrainState and info dict with BC diagnostics.
    """
    # Extract base actions from observations (for diagnostics)
    base_action_raw = batch['observations']['base_action']
    base_action = jnp.squeeze(base_action_raw, axis=-1)  # (B, T, A)
    B, T, A = base_action.shape

    # BC target = stored actions from the replay buffer, flattened.
    stored_actions_flat = batch['actions'].reshape(B, query_frequency * A)
    bc_target = jnp.clip(stored_actions_flat, -1.0, 1.0)
    
    def actor_loss_fn(actor_params: Params) -> Tuple[jnp.ndarray, Tuple[Dict[str, float], Dict]]:
        new_model_state = {}

        if is_mip_actor:
            # MIPActor.bc_loss expects chunked targets (B, H, A) and a
            # (B, H) validity mask — we treat every step as valid here.
            targets_chunked = batch['actions']  # (B, H, A)
            valid_mask = jnp.ones(targets_chunked.shape[:2])
            bc_loss_val = actor.apply_fn(
                _actor_variables(actor, actor_params),
                batch['observations'], targets_chunked, valid_mask, key,
                method='bc_loss',
            )
            # MIP has no distrax distribution — surface stored actions as
            # stand-ins so the WandB schema stays consistent.
            sampled_actions = batch['actions'].reshape(B, query_frequency * A)
            policy_mode = sampled_actions
            dist = None
        else:
            if hasattr(actor, 'batch_stats') and actor.batch_stats is not None:
                dist, new_model_state = actor.apply_fn(
                    {'params': actor_params, 'batch_stats': actor.batch_stats},
                    batch['observations'],
                    mutable=['batch_stats']
                )
                if isinstance(dist, tuple):
                    dist = dist[0]
            else:
                dist = actor.apply_fn({'params': actor_params}, batch['observations'])
                new_model_state = {}

            # Sample-based MSE: trains both mean and std via reparameterization trick
            # Avoids atanh boundary issues from log_prob while providing gradient to std
            sampled_actions = dist.sample(seed=key)  # (B, action_dim_flat)
            bc_loss_val = jnp.mean(
                jnp.mean((sampled_actions - bc_target) ** 2, axis=-1)
            )
            # Mode for diagnostics
            policy_mode = dist.mode()  # (B, action_dim_flat)

        # Per-sample diagnostic (Gaussian path: real MSE; MIP: zero).
        mse_per_sample = jnp.mean((sampled_actions - bc_target) ** 2, axis=-1)
        bc_loss = bc_loss_val
        # L1 (MAE) per batch — surfaced to the tqdm postfix during BC warmup.
        # Trained against MSE; this L1 is logging-only, easier to read in action space.
        l1_per_sample = jnp.mean(jnp.abs(sampled_actions - bc_target), axis=-1)
        bc_l1_loss = l1_per_sample.mean()

        # Distribution diagnostics — Gaussian path only.
        if is_mip_actor:
            gauss_stats = _empty_gaussian_log_stats(sampled_actions, B)
            mean_dist = gauss_stats['mean_dist']
            std_diag_dist = gauss_stats['std_diag_dist']
            log_std_dist = jnp.zeros_like(std_diag_dist)
            mean_entropy = jnp.zeros(())
        else:
            mean_dist = dist.distribution._loc
            std_diag_dist = dist.distribution._scale_diag
            log_std_dist = jnp.log(std_diag_dist + 1e-8)
            base_entropy = dist.distribution.entropy()  # (B,)
            mean_entropy = base_entropy.mean()
        
        # Compute a_exec from sampled actions for diagnostics
        sampled_actions_chunked = sampled_actions.reshape(B, query_frequency, A)
        if predict_a_exec:
            a_exec = jnp.clip(sampled_actions_chunked, -1.0, 1.0)
            delta_from_base = a_exec - base_action[:, :query_frequency, :]
        else:
            a_exec = jnp.clip(
                base_action[:, :query_frequency, :] + sampled_actions_chunked, -1.0, 1.0
            )
            delta_from_base = sampled_actions_chunked
        
        delta_norm = jnp.linalg.norm(delta_from_base.reshape(B, -1), axis=-1)
        base_flat = base_action[:, :query_frequency, :].reshape(B, -1)
        base_norm = jnp.linalg.norm(base_flat, axis=-1)
        a_exec_norm = jnp.linalg.norm(a_exec.reshape(B, -1), axis=-1)
        clipping_rate = (jnp.abs(a_exec) >= 1.0).mean()
        bc_target_norm = jnp.linalg.norm(bc_target, axis=-1)
        # Fraction of raw base_action values outside [-1, 1] (before clipping)
        base_out_of_range = (jnp.abs(base_flat) > 1.0).mean()
        
        info = {
            'actor_loss': bc_loss,
            'actor_l1_loss': bc_l1_loss,
            'bc_warmup/mse_loss': bc_loss,
            'bc_warmup/l1_loss': bc_l1_loss,
            'bc_warmup/l1_per_sample_max': l1_per_sample.max(),
            'bc_warmup/l1_per_sample_min': l1_per_sample.min(),
            'bc_warmup/mse_per_sample_mean': mse_per_sample.mean(),
            'bc_warmup/mse_per_sample_max': mse_per_sample.max(),
            'bc_warmup/mse_per_sample_min': mse_per_sample.min(),
            'bc_warmup/sampled_action_mean': sampled_actions.mean(),
            'bc_warmup/sampled_action_std': jnp.std(sampled_actions),
            'bc_warmup/policy_mode_mean': policy_mode.mean(),
            'bc_warmup/policy_mode_std': jnp.std(policy_mode),
            'bc_warmup/target_mean': bc_target.mean(),
            'bc_warmup/target_std': jnp.std(bc_target),
            'bc_warmup/delta_norm_mean': delta_norm.mean(),
            'bc_warmup/base_norm_mean': base_norm.mean(),
            'bc_warmup/base_out_of_range_frac': base_out_of_range,
            'bc_warmup/clipping_rate': clipping_rate,
            'bc_warmup/is_warmup': 1.0,
            # Standard actor diagnostics (for continuity in wandb)
            'entropy': mean_entropy,
            'mean_pi_norm': jnp.linalg.norm(mean_dist, axis=-1).mean(),
            'std_pi_norm': jnp.linalg.norm(std_diag_dist, axis=-1).mean(),
            'mean_pi_avg': mean_dist.mean(),
            'std_pi_avg': std_diag_dist.mean(),
            'std_pi_min': std_diag_dist.min(),
            'std_pi_max': std_diag_dist.max(),
            'log_std_mean': log_std_dist.mean(),
            'log_std_min': log_std_dist.min(),
            'log_std_max': log_std_dist.max(),
            'actor/delta_norm_mean': delta_norm.mean(),
            'actor/clipping_rate': clipping_rate,
            'actor/predict_a_exec': predict_a_exec, 
            'actor/a_exec_norm_mean': a_exec_norm.mean(), 
            'actor/bc_target_norm_mean': bc_target_norm.mean()
        }
        
        return bc_loss, (info, new_model_state)
    
    grads, (info, new_model_state) = jax.grad(actor_loss_fn, has_aux=True)(actor.params)
    
    # NaN guard
    grads = _nan_to_num_tree(grads)
    
    if 'batch_stats' in new_model_state and new_model_state.get('batch_stats'):
        new_actor = actor.apply_gradients(grads=grads, batch_stats=new_model_state['batch_stats'])
    else:
        new_actor = actor.apply_gradients(grads=grads)

    return new_actor, info
    
