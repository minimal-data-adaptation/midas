import random

import jax
import numpy as np

from midas.utils.reproducibility import (
    capture_environment_rng_state,
    capture_process_rng_state,
    capture_training_rng_state,
    decode_rng_state,
    encode_rng_state,
    restore_environment_rng_state,
    restore_process_rng_state,
    restore_training_rng_state,
    seed_environment,
    seed_process,
)


class _StatefulPolicy:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def get_rng_state(self):
        return self.rng.bit_generator.state

    def set_rng_state(self, state):
        self.rng.bit_generator.state = state

    def seed(self, seed):
        self.rng = np.random.default_rng(seed)


class _Environment:
    def __init__(self, seed):
        self._rng = np.random.default_rng(seed)

    @property
    def unwrapped(self):
        return self


class _Replay:
    def __init__(self, seed):
        self.rng = np.random.default_rng(seed)

    def get_rng_state(self):
        return self.rng.bit_generator.state

    def set_rng_state(self, state):
        self.rng.bit_generator.state = state


def test_process_rng_capture_restore_continues_streams():
    seed_process(17)
    state = capture_process_rng_state()
    expected = (random.random(), np.random.uniform())
    seed_process(999)
    restore_process_rng_state(state)
    assert (random.random(), np.random.uniform()) == expected


def test_environment_rng_capture_restore_and_reseed():
    env = _Environment(4)
    state = capture_environment_rng_state(env)
    expected = env._rng.normal(size=4)
    restore_environment_rng_state(env, state)
    np.testing.assert_array_equal(env._rng.normal(size=4), expected)

    seed_environment(env, 8)
    np.testing.assert_array_equal(
        env._rng.normal(size=4), np.random.default_rng(8).normal(size=4)
    )


def test_encoded_training_state_restores_all_private_streams():
    env = _Environment(1)
    policy = _StatefulPolicy(2)
    replay = _Replay(3)
    state = capture_training_rng_state(
        env=env, base_policy=policy, replay_buffer=replay
    )
    encoded = encode_rng_state(state)
    expected = (
        env._rng.integers(1000),
        policy.rng.integers(1000),
        replay.rng.integers(1000),
    )
    restore_training_rng_state(
        decode_rng_state(encoded),
        env=env,
        base_policy=policy,
        replay_buffer=replay,
    )
    actual = (
        env._rng.integers(1000),
        policy.rng.integers(1000),
        replay.rng.integers(1000),
    )
    assert actual == expected


def test_openpi_jax_policy_rng_round_trip():
    from openpi_client import msgpack_numpy
    from openpi.policies.policy import Policy

    policy = object.__new__(Policy)
    policy._is_pytorch_model = False
    policy._rng = jax.random.key(11)
    state = policy.get_rng_state()
    state = msgpack_numpy.unpackb(msgpack_numpy.packb(state))
    _, expected = jax.random.split(policy._rng)
    policy.seed(42)
    policy.set_rng_state(state)
    _, actual = jax.random.split(policy._rng)
    np.testing.assert_array_equal(jax.random.key_data(actual), jax.random.key_data(expected))


def test_openpi_torch_policy_rng_is_private_and_restorable():
    import torch
    from openpi_client import msgpack_numpy
    from openpi.policies.policy import Policy

    policy = object.__new__(Policy)
    policy._is_pytorch_model = True
    policy._torch_generator = torch.Generator(device="cpu")
    policy.seed(13)
    state = policy.get_rng_state()
    state = msgpack_numpy.unpackb(msgpack_numpy.packb(state))
    expected = torch.rand(5, generator=policy._torch_generator)
    torch.manual_seed(999)
    policy.set_rng_state(state)
    actual = torch.rand(5, generator=policy._torch_generator)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_midas_checkpoint_payload_contains_learner_rng():
    from midas.agents.midas.midas_learner import MidasLearner

    learner = object.__new__(MidasLearner)
    learner._rng = jax.random.PRNGKey(23)
    learner._critic = object()
    learner._target_critic_params = object()
    learner._actor = object()
    learner._residual_alpha = 0.5
    learner.algo = "midas"
    learner.predict_a_exec = True
    learner.actor_arch = "tanh_gaussian"
    learner.critic_arch = "mlp_ensemble"
    learner.mip_t_star = 0.9
    learner.mip_noise_std = 0.01
    learner.mip_use_film = False
    learner.mip_q_noise_scale = 1.0
    np.testing.assert_array_equal(
        learner._save_dict["rng"], learner._rng
    )


def test_midas_checkpoint_restores_rng_and_accepts_legacy_checkpoint(tmp_path):
    import jax.numpy as jnp
    from flax.training import checkpoints
    from midas.agents.midas.midas_learner import MidasLearner

    def learner_with_rng(seed):
        learner = object.__new__(MidasLearner)
        learner._rng = jax.random.PRNGKey(seed)
        learner._critic = jnp.array([1.0])
        learner._target_critic_params = jnp.array([2.0])
        learner._actor = jnp.array([3.0])
        learner._residual_alpha = 0.5
        learner.algo = "midas"
        learner.predict_a_exec = True
        learner.actor_arch = "tanh_gaussian"
        learner.critic_arch = "mlp_ensemble"
        learner.mip_t_star = 0.9
        learner.mip_noise_std = 0.01
        learner.mip_use_film = False
        learner.mip_q_noise_scale = 1.0
        return learner

    current_dir = tmp_path / "current"
    learner = learner_with_rng(23)
    checkpoints.save_checkpoint(current_dir, learner._save_dict, 1)
    learner._rng = jax.random.PRNGKey(99)
    learner.restore_checkpoint(current_dir)
    np.testing.assert_array_equal(learner._rng, jax.random.PRNGKey(23))

    legacy_dir = tmp_path / "legacy"
    legacy = dict(learner._save_dict)
    legacy.pop("rng")
    checkpoints.save_checkpoint(legacy_dir, legacy, 1)
    learner._rng = jax.random.PRNGKey(77)
    learner.restore_checkpoint(legacy_dir)
    np.testing.assert_array_equal(learner._rng, jax.random.PRNGKey(77))


def test_periodic_evaluation_does_not_advance_training_rngs(monkeypatch):
    from types import SimpleNamespace

    import training.train_utils_sim as train_utils

    env = _Environment(5)
    policy = _StatefulPolicy(6)
    seed_process(7)
    process_state = capture_process_rng_state()
    policy_state = policy.get_rng_state()

    def fake_eval(_agent, eval_env, _step, _variant, _logger, base, _vlm):
        return (
            random.random(),
            np.random.uniform(),
            int(eval_env._rng.integers(1_000_000)),
            int(base.rng.integers(1_000_000)),
        )

    monkeypatch.setattr(train_utils, "_perform_control_eval_residual", fake_eval)
    variant = SimpleNamespace(seed=11)
    first = train_utils.perform_control_eval_residual(None, env, 1, variant, None, policy)
    second = train_utils.perform_control_eval_residual(None, env, 1, variant, None, policy)
    assert first == second

    actual_process = (random.random(), np.random.uniform())
    restore_process_rng_state(process_state)
    expected_process = (random.random(), np.random.uniform())
    assert actual_process == expected_process

    actual_policy = policy.rng.integers(1_000_000)
    policy.set_rng_state(policy_state)
    expected_policy = policy.rng.integers(1_000_000)
    assert actual_policy == expected_policy
