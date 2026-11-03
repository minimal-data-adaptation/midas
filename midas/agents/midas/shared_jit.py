"""Small JIT helpers shared by the MIDAS learner."""

import functools
from typing import Any

import jax
import jax.numpy as jnp
import optax
import flax.traverse_util
from flax.core import FrozenDict, freeze


def _freeze_encoder_optimizer(
    base_chain: optax.GradientTransformation,
    freeze_vision_encoder: bool,
) -> optax.GradientTransformation:
    """Wrap ``base_chain`` so the ``encoder`` subtree is excluded from the
    optimizer when ``freeze_vision_encoder`` is True.

    Pairs with ``jax.lax.stop_gradient`` on the encoder output: that already
    zeros the cotangent through the CNN tower, but the encoder params remain
    in the optimizer's param tree, so Adam still allocates ``m``/``v`` moments
    for every encoder param and ``clip_by_global_norm`` walks them. Routing
    the encoder subtree through ``optax.set_to_zero()`` skips both costs.
    """
    if not freeze_vision_encoder:
        return base_chain

    def label_fn(params):
        flat = flax.traverse_util.flatten_dict(params)
        labels = {
            path: ('frozen' if path[0] == 'encoder' else 'trainable')
            for path in flat
        }
        label_tree = flax.traverse_util.unflatten_dict(labels)
        return freeze(label_tree) if isinstance(params, FrozenDict) else label_tree

    return optax.multi_transform(
        {'trainable': base_chain, 'frozen': optax.set_to_zero()},
        label_fn,
    )

_VALID_ACTOR_ARCHS = ('tanh_gaussian', 'mip')
_VALID_CRITIC_ARCHS = ('mlp_ensemble', 'mip_ensemble')


@functools.partial(jax.jit, static_argnames='actor_apply_fn')
def _sample_via_method_jit(rng, actor_apply_fn, actor_params, observations, actor_batch_stats):
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    rng, key = jax.random.split(rng)
    actions = actor_apply_fn(input_collections, observations, key, method='sample')
    return rng, actions


@functools.partial(jax.jit, static_argnames='actor_apply_fn')
def _mode_via_method_jit(actor_apply_fn, actor_params, observations, actor_batch_stats):
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    return actor_apply_fn(input_collections, observations, method='mode')


@functools.partial(jax.jit, static_argnames='actor_apply_fn')
def _sample_with_logprob_via_method_jit(rng, actor_apply_fn, actor_params, observations, actor_batch_stats):
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    rng, key = jax.random.split(rng)
    actions, log_probs = actor_apply_fn(
        input_collections, observations, key, method='sample_with_logprob'
    )
    return rng, actions, log_probs


@functools.partial(jax.jit, static_argnames='actor_apply_fn')
def _encode_pixels_via_method_jit(actor_apply_fn, actor_params, pixels, actor_batch_stats):
    """Run the actor's vision encoder on ``pixels`` only.

    Pairs with ``PixelMultiplexer.encode_pixels``: when
    ``freeze_vision_encoder=True`` the rollout pre-computes encoder features
    once per env step using this JIT'd helper, and the cached output is
    written into the replay buffer at ``observations['pixel_features']``.
    The encoder is then skipped entirely on the train graph.
    """
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    return actor_apply_fn(input_collections, pixels, False, method='encode_pixels')


@functools.partial(jax.jit, static_argnames='actor_apply_fn')
def _compute_log_prob_via_method_jit(actor_apply_fn, actor_params, observations, actions, actor_batch_stats):
    input_collections = {'params': actor_params}
    if actor_batch_stats is not None:
        input_collections['batch_stats'] = actor_batch_stats
    safe_actions = jnp.clip(actions, -1.0 + 1e-6, 1.0 - 1e-6)
    log_probs = actor_apply_fn(
        input_collections, observations, safe_actions, method='compute_log_prob'
    )
    return jnp.clip(log_probs, -50.0, 50.0)


