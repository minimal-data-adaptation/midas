import flax.core
from flax.training.train_state import TrainState
import jax.numpy as jnp
import numpy as np
from openpi_client import msgpack_numpy
import optax
import pytest

from midas.agents.agent import Agent


class DummyAgent(Agent):
    pass


def _agent(value=1.0):
    agent = DummyAgent()
    agent._actor = TrainState.create(
        apply_fn=lambda *_: None,
        params=flax.core.freeze({"layer": {"kernel": jnp.full((2, 3), value)}}),
        tx=optax.sgd(0.1),
    )
    return agent


def test_actor_state_survives_wire_round_trip_and_swaps_atomically():
    source = _agent(2.0)
    target = _agent(1.0)
    wire_state = msgpack_numpy.unpackb(msgpack_numpy.packb(source.export_actor_state()))
    target.replace_actor_state(wire_state)
    np.testing.assert_array_equal(target._actor.params["layer"]["kernel"], np.full((2, 3), 2.0))


def test_actor_state_rejects_shape_mismatch_without_mutating():
    target = _agent(1.0)
    with pytest.raises(ValueError, match="shape"):
        target.replace_actor_state({"params": {"layer": {"kernel": np.zeros((3, 2), np.float32)}}})
    np.testing.assert_array_equal(target._actor.params["layer"]["kernel"], np.ones((2, 3)))
