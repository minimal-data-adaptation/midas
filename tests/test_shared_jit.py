import jax.numpy as jnp
import jax
import optax
from flax.core import freeze

from midas.agents.midas.actor_updater import compute_bc_loss_residual
from midas.agents.midas.shared_jit import _freeze_encoder_optimizer


def test_frozen_encoder_gets_zero_updates():
    params = freeze({"encoder": {"weight": jnp.ones((2,))}, "head": {"weight": jnp.ones((2,))}})
    optimizer = _freeze_encoder_optimizer(optax.sgd(0.1), True)
    state = optimizer.init(params)
    updates, _ = optimizer.update(params, state, params)
    assert jnp.all(updates["encoder"]["weight"] == 0)
    assert jnp.any(updates["head"]["weight"] != 0)


class _DeterministicDistribution:
    def sample(self, seed):
        del seed
        return jnp.zeros((2, 2))

    def mode(self):
        return jnp.zeros((2, 2))


def test_bc_regularizer_supports_success_mask():
    batch = {
        "actions": jnp.array([[[1.0], [1.0]], [[3.0], [3.0]]]),
        "success_flag": jnp.array([1.0, 0.0]),
    }
    loss, info = compute_bc_loss_residual(
        _DeterministicDistribution(), batch, 2, True, jax.random.PRNGKey(0)
    )
    assert float(loss) == 1.0
    assert float(info["bc/success_frac_in_batch"]) == 0.5
