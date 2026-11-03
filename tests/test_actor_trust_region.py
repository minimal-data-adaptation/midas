import jax.numpy as jnp
import numpy as np
import pytest

from midas.agents.midas.actor_updater import _apply_actor_trust_region


def test_actor_trust_region_is_noop_when_disabled():
    base = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    target = jnp.array([[[0.8, -0.8]]], dtype=jnp.float32)
    cap = jnp.array([0.1, 0.2], dtype=jnp.float32)

    actual, _, stats = _apply_actor_trust_region(target, base, cap, False)

    np.testing.assert_allclose(actual, target)
    assert float(stats["clip_fraction"]) == 0.0


def test_actor_trust_region_clips_when_enabled():
    base = jnp.zeros((1, 1, 2), dtype=jnp.float32)
    target = jnp.array([[[0.8, -0.8]]], dtype=jnp.float32)
    cap = jnp.array([0.1, 0.2], dtype=jnp.float32)

    actual, _, stats = _apply_actor_trust_region(target, base, cap, True)

    np.testing.assert_allclose(actual, [[[0.1, -0.2]]], rtol=1e-6, atol=1e-6)
    assert float(stats["clip_fraction"]) == 1.0


def test_enabled_actor_trust_region_requires_cap():
    target = jnp.zeros((1, 1, 1), dtype=jnp.float32)
    with pytest.raises(ValueError, match="trust-region cap"):
        _apply_actor_trust_region(target, target, None, True)
