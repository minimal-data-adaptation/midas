import asyncio

import numpy as np
import pytest

from midas.real.config import RealRunSpec
from midas.real.protocol import request, validate_inference_result
from training.serve_real import MockBasePolicy, RealPolicy


class FakeActor:
    def __init__(self):
        self.state = None

    def eval_actions(self, observation):
        del observation
        return np.full((1, 28), 0.25, np.float32)

    def replace_actor_state(self, state):
        self.state = state


class StatefulBasePolicy(MockBasePolicy):
    def __init__(self, spec, seed=0):
        super().__init__(spec)
        self.rng = np.random.default_rng(seed)

    def get_rng_state(self):
        return self.rng.bit_generator.state

    def set_rng_state(self, state):
        self.rng.bit_generator.state = state


def _spec():
    return RealRunSpec(
        resize_image=4,
        chunk_len=4,
        query_freq=2,
        use_vlm_embedding=False,
        midas_use_trust_region=False,
    )


def _observation():
    return {
        "state": np.zeros(14, np.float32),
        "images": {
            name: np.zeros((3, 4, 4), np.uint8) for name in ("top", "left_wrist", "right_wrist")
        },
    }


def test_real_policy_is_disarmed_until_valid_actor_update():
    spec = _spec()
    actor = FakeActor()
    policy = RealPolicy(spec, MockBasePolicy(spec), actor)
    with pytest.raises(RuntimeError, match="disarmed"):
        asyncio.run(policy.infer(_observation()))
    with pytest.raises(ValueError, match="signature"):
        asyncio.run(policy.update_actor_state({}, "wrong"))
    update = asyncio.run(policy.update_actor_state({"params": {}}, spec.actor_signature))
    assert update == {"actor_version": 1, "armed": True}
    result = asyncio.run(policy.infer(_observation()))
    envelope = {**request("unused"), **result}
    validate_inference_result(envelope, spec)
    assert result["actions"].shape == (2, 14)
    assert np.all(result["a_exec_norm"] == np.float32(0.25))


def test_protocol_rejects_out_of_range_normalized_actions():
    spec = _spec()
    envelope = request("unused")
    envelope.update(
        actions=np.zeros((2, 14)),
        base_action=np.zeros((4, 14)),
        a_exec_norm=np.full((2, 14), 1.1),
    )
    with pytest.raises(ValueError, match="outside"):
        validate_inference_result(envelope, spec)


def test_real_policy_round_trips_server_owned_base_rng():
    spec = _spec()
    base = StatefulBasePolicy(spec, seed=17)
    policy = RealPolicy(spec, base, FakeActor())

    saved = asyncio.run(policy.get_base_rng_state())["base_rng_state"]
    expected = base.rng.normal(size=8)
    base.rng = np.random.default_rng(999)
    result = asyncio.run(policy.set_base_rng_state(saved))
    assert result == {"base_rng_state_restored": True}
    np.testing.assert_array_equal(base.rng.normal(size=8), expected)
