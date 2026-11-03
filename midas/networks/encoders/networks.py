from typing import Any, Dict, Optional, Sequence, Union

import flax.linen as nn
import jax
import jax.numpy as jnp
from flax.core.frozen_dict import FrozenDict

from midas.networks.constants import default_init, xavier_init

ModuleDef = Any

class Encoder(nn.Module):
    features: Sequence[int] = (32, 32, 32, 32)
    strides: Sequence[int] = (2, 1, 1, 1)
    padding: str = 'VALID'

    @nn.compact
    def __call__(self, observations: jnp.ndarray, training=False) -> jnp.ndarray:
        assert len(self.features) == len(self.strides)

        x = observations.astype(jnp.float32) / 255.0
        x = jnp.reshape(x, (*x.shape[:-2], -1))

        for features, stride in zip(self.features, self.strides):
            x = nn.Conv(features,
                        kernel_size=(3, 3),
                        strides=(stride, stride),
                        kernel_init=default_init(),
                        padding=self.padding)(x)
            x = nn.relu(x)

        return x.reshape((*x.shape[:-3], -1))


class PixelMultiplexer(nn.Module):
    encoder: Union[nn.Module, list]
    network: nn.Module
    latent_dim: int
    use_bottleneck: bool=True
    pop_base_actions: bool=True
    use_vlm_embedding: bool=False
    freeze_vision_encoder: bool=False

    @nn.compact
    def encode(self,
               observations: Union[FrozenDict, Dict],
               training: bool = False):
        """Shared encoder trunk used by ``__call__`` and all proxy methods.

        Returns the encoded observation dict the inner ``network`` head consumes.
        Separating this from ``__call__`` is what lets ``PixelMultiplexer``
        expose the polymorphic method surface (``sample``, ``mode``, etc.) that
        the residual learners dispatch against via ``method=`` — both distrax
        and non-distrax (MIP) inner networks share the same trunk.
        """
        observations = FrozenDict(observations)

        if self.use_vlm_embedding:
            # VLM embedding mode: pop 'pixels' (raw images), use 'vlm_embedding' instead.
            # observations['vlm_embedding'] has shape (B, W, 1) — already mean-pooled at collection time.
            vlm_emb = observations['vlm_embedding']
            x = jnp.squeeze(vlm_emb, axis=-1)  # (B, W)
            # Drop both 'pixels' and 'vlm_embedding' from observations
            observations = FrozenDict({k: v for k, v in observations.items() if k not in ('pixels', 'vlm_embedding')})
        elif self.freeze_vision_encoder and 'pixel_features' in observations:
            # Cached-feature mode: encoder forward already ran at rollout time
            # (see compute_pixel_features on the residual learner) and the
            # output was stored in the replay buffer alongside 'pixels'. Skip
            # the encoder here entirely — this is the main speedup over running
            # a frozen encoder every train step. Stored shape is (B, D, 1) to
            # mirror the VLM-embedding storage convention.
            feat = observations['pixel_features']
            x = jnp.squeeze(feat, axis=-1)
            observations = FrozenDict(
                {k: v for k, v in observations.items()
                 if k not in ('pixels', 'pixel_features')}
            )
        else:
            x = self.encoder(observations['pixels'], training)
            # Detach the vision tower's output so gradients from the head do not
            # flow back into encoder params. The encoder still runs in forward
            # pass; bottleneck/Dense head below remains trainable. Cached-feature
            # mode above skips the encoder forward as well as the gradient.
            if self.freeze_vision_encoder:
                x = jax.lax.stop_gradient(x)

        if self.use_bottleneck:
            x = nn.Dense(self.latent_dim, kernel_init=xavier_init())(x)
            x = nn.LayerNorm()(x)
            x = nn.tanh(x)

        x = observations.copy(add_or_replace={'pixels': x})

        if 'base_action' in x and self.pop_base_actions:
           x = FrozenDict({k: v for k, v in x.items() if k != 'base_action'})
        elif 'base_action' in x:
            base_action_raw = observations['base_action']

            # chex.assert_rank(base_action_raw, 4)
            # chex.assert_equal(base_action_raw.shape[-1], 1)

            base_action = jnp.squeeze(base_action_raw, axis=-1)
            base_action = base_action.reshape(
                base_action.shape[0],
                -1
            ) #flatten time step and action dim
            x = x.copy(add_or_replace={'base_action': base_action})

        return x

    def __call__(self,
                 observations: Union[FrozenDict, Dict],
                 actions: Optional[jnp.ndarray] = None,
                 training: bool = False):
        x = self.encode(observations, training=training)
        if actions is None:
            return self.network(x, training=training)
        else:
            return self.network(x, actions, training=training)

    def encode_pixels(self, pixels: jnp.ndarray, training: bool = False):
        """Run only the vision encoder on raw pixels.

        Used at rollout time when ``freeze_vision_encoder=True`` to precompute
        encoder features and stash them in the replay buffer as
        ``observations['pixel_features']``. The cached-feature branch in
        ``encode`` then consumes that tensor and skips the encoder forward
        during training. Bottleneck/Dense head and the rest of
        ``PixelMultiplexer`` are not invoked here.
        """
        return self.encoder(pixels, training)

    def encode_only(self,
                    observations: Union[FrozenDict, Dict],
                    training: bool = False):
        """Public alias for the encode trunk.

        Pair with ``head_call``/``head_sample`` to amortize the encoder over
        many head-only forwards on the same observations (e.g. MIDAS's
        gradient-ascent refinement loop).
        """
        return self.encode(observations, training=training)

    def head_call(self,
                  encoded_observations: Union[FrozenDict, Dict],
                  actions: Optional[jnp.ndarray] = None,
                  training: bool = False):
        """Apply the inner network head to an already-encoded obs dict.

        Mirrors ``__call__`` except the encoder is skipped; the caller is
        responsible for having passed ``observations`` through ``encode_only``
        first.
        """
        if actions is None:
            return self.network(encoded_observations, training=training)
        else:
            return self.network(encoded_observations, actions, training=training)

    def head_sample(self,
                    encoded_observations: Union[FrozenDict, Dict],
                    rng,
                    training: bool = False):
        """Sample from the inner head given already-encoded observations."""
        return self.network.sample(encoded_observations, rng, training=training)

    def sample(self, observations, rng, training: bool = False):
        x = self.encode(observations, training=training)
        return self.network.sample(x, rng, training=training)

    def sample_with_logprob(self, observations, rng, training: bool = False):
        x = self.encode(observations, training=training)
        return self.network.sample_with_logprob(x, rng, training=training)

    def mode(self, observations):
        x = self.encode(observations, training=False)
        return self.network.mode(x)

    def compute_log_prob(self, observations, actions):
        x = self.encode(observations, training=False)
        return self.network.compute_log_prob(x, actions)

    def bc_loss(self, observations, targets, valid_mask, rng):
        x = self.encode(observations, training=False)
        return self.network.bc_loss(x, targets, valid_mask, rng)

    def td_loss(self, observations, actions, target_q, valid_mask, rng):
        x = self.encode(observations, training=False)
        return self.network.td_loss(x, actions, target_q, valid_mask, rng)
