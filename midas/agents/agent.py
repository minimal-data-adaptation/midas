import numpy as np
import jax
from flax import serialization
from flax.training import checkpoints
import orbax.checkpoint as ocp
from typing import Any
from flax.training.train_state import TrainState

from midas.agents.common import (
    eval_actions_jit,
    eval_log_prob_jit,
    eval_mse_jit,
    eval_reward_function_jit,
    sample_actions_jit,
    sample_actions_with_log_prob_jit,
    compute_log_prob_jit,
)
from midas.data.dataset import DatasetDict
from midas.types import PRNGKey


def get_batch_stats(actor):
    if hasattr(actor, "batch_stats"):
        return actor.batch_stats
    else:
        return None


class Agent(object):
    _actor: TrainState
    _critic: TrainState
    _rng: PRNGKey

    def eval_actions(self, observations: np.ndarray) -> np.ndarray:
        actions = eval_actions_jit(
            self._actor.apply_fn, self._actor.params, observations, get_batch_stats(self._actor)
        )
        return np.asarray(actions)

    def eval_log_probs(self, batch: DatasetDict) -> float:
        return eval_log_prob_jit(
            self._actor.apply_fn, self._actor.params, get_batch_stats(self._actor), batch
        )

    def eval_mse(self, batch: DatasetDict) -> float:
        return eval_mse_jit(
            self._actor.apply_fn, self._actor.params, get_batch_stats(self._actor), batch
        )

    def eval_reward_function(self, batch: DatasetDict) -> float:
        return eval_reward_function_jit(
            self._actor.apply_fn, self._actor.params, self._actor.batch_stats, batch
        )

    def sample_actions(self, observations: np.ndarray) -> np.ndarray:
        rng, actions = sample_actions_jit(
            self._rng,
            self._actor.apply_fn,
            self._actor.params,
            observations,
            get_batch_stats(self._actor),
        )

        self._rng = rng
        return np.asarray(actions)

    def sample_actions_with_log_prob(self, observations: np.ndarray):
        """Sample actions and return their log probabilities.

        Returns:
            (actions, log_probs) as numpy arrays.
        """
        rng, actions, log_probs = sample_actions_with_log_prob_jit(
            self._rng,
            self._actor.apply_fn,
            self._actor.params,
            observations,
            get_batch_stats(self._actor),
        )

        self._rng = rng
        return np.asarray(actions), np.asarray(log_probs)

    def compute_log_prob(self, observations: np.ndarray, actions: np.ndarray) -> np.ndarray:
        """Compute log probability of given actions under current policy.

        Clamps actions to (-1+eps, 1-eps) to avoid atanh(±1)=inf in TanhNormal.

        Args:
            observations: Observation dict.
            actions: Actions to evaluate, shape (B, action_dim_flat).

        Returns:
            log_probs as numpy array.
        """
        log_probs = compute_log_prob_jit(
            self._actor.apply_fn,
            self._actor.params,
            observations,
            actions,
            get_batch_stats(self._actor),
        )
        return np.asarray(log_probs)

    def export_actor_state(self) -> dict[str, Any]:
        """Return a host-resident actor state suitable for process transfer."""

        state = {"params": jax.device_get(serialization.to_state_dict(self._actor.params))}
        batch_stats = get_batch_stats(self._actor)
        if batch_stats is not None:
            state["batch_stats"] = jax.device_get(serialization.to_state_dict(batch_stats))
        return state

    def replace_actor_state(self, state: dict[str, Any]) -> None:
        """Atomically replace actor params and mutable statistics after validation."""

        if "params" not in state:
            raise ValueError("Actor state is missing 'params'")

        def _coerce_and_validate(name, current, incoming):
            # Msgpack reconstructs mapping containers as plain dicts. Restore
            # the exact Flax/FrozenDict container structure before comparing
            # and swapping the state.
            try:
                incoming = serialization.from_state_dict(
                    current, serialization.to_state_dict(incoming)
                )
            except Exception as error:
                raise ValueError(f"Actor {name} state structure does not match") from error
            if jax.tree_util.tree_structure(current) != jax.tree_util.tree_structure(incoming):
                raise ValueError(f"Actor {name} PyTree structure does not match")
            current_leaves = jax.tree_util.tree_leaves(current)
            incoming_leaves = jax.tree_util.tree_leaves(incoming)
            for index, (expected, actual) in enumerate(zip(current_leaves, incoming_leaves)):
                if np.shape(expected) != np.shape(actual):
                    raise ValueError(
                        f"Actor {name} leaf {index} shape {np.shape(actual)} does not match "
                        f"{np.shape(expected)}"
                    )
                if np.dtype(expected.dtype) != np.dtype(actual.dtype):
                    raise ValueError(
                        f"Actor {name} leaf {index} dtype {actual.dtype} does not match "
                        f"{expected.dtype}"
                    )
            return incoming

        params = _coerce_and_validate("params", self._actor.params, state["params"])
        replacements = {"params": params}
        current_batch_stats = get_batch_stats(self._actor)
        incoming_batch_stats = state.get("batch_stats")
        if current_batch_stats is None and incoming_batch_stats is not None:
            raise ValueError("Actor does not have batch_stats, but update supplied them")
        if current_batch_stats is not None:
            if incoming_batch_stats is None:
                raise ValueError("Actor update is missing required batch_stats")
            replacements["batch_stats"] = _coerce_and_validate(
                "batch_stats", current_batch_stats, incoming_batch_stats
            )
        self._actor = self._actor.replace(**replacements)

    @property
    def _save_dict(self):
        return None

    def _get_async_checkpointer(self):
        """Lazy-init a single AsyncCheckpointer reused across save calls.

        Holding one instance lets Orbax block on the previous save before
        starting the next one, which is what makes back-to-back saves safe
        without a separate wait call from the training loop.
        """
        if not hasattr(self, "_orbax_async_checkpointer"):
            self._orbax_async_checkpointer = ocp.AsyncCheckpointer(ocp.PyTreeCheckpointHandler())
        return self._orbax_async_checkpointer

    def save_checkpoint(self, dir, step, keep_every_n_steps):
        checkpoints.save_checkpoint(
            dir,
            self._save_dict,
            step,
            prefix="checkpoint",
            overwrite=False,
            keep_every_n_steps=keep_every_n_steps,
            orbax_checkpointer=self._get_async_checkpointer(),
        )

    def wait_for_checkpoints(self):
        """Block until any in-flight async checkpoint write has finished.

        Call before process exit / after the final save to ensure the last
        checkpoint is durably on disk.
        """
        if hasattr(self, "_orbax_async_checkpointer"):
            self._orbax_async_checkpointer.wait_until_finished()

    def restore_checkpoint(self, dir):
        raise NotImplementedError
