"""Utilities for deterministic seeding and resumable random-number streams."""

from __future__ import annotations

import base64
import copy
import pickle
import random
import sys
from typing import Any

import numpy as np


EVAL_SEED_OFFSET = 1_000_003


def seed_process(seed: int) -> None:
    """Seed process-global RNGs that are already available in this process."""

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.manual_seed(seed)


def capture_process_rng_state() -> dict[str, Any]:
    """Capture Python, NumPy, and initialized Torch global RNG state."""

    state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
    }
    torch = sys.modules.get("torch")
    if torch is not None:
        state["torch_cpu"] = torch.random.get_rng_state().cpu().numpy()
        if torch.cuda.is_initialized():
            state["torch_cuda"] = [value.cpu().numpy() for value in torch.cuda.get_rng_state_all()]
    return state


def restore_process_rng_state(state: dict[str, Any]) -> None:
    """Restore state returned by :func:`capture_process_rng_state`."""

    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch = sys.modules.get("torch")
    if torch is not None and "torch_cpu" in state:
        torch.random.set_rng_state(torch.as_tensor(state["torch_cpu"], dtype=torch.uint8))
        if "torch_cuda" in state and torch.cuda.is_initialized():
            torch.cuda.set_rng_state_all(
                [torch.as_tensor(value, dtype=torch.uint8) for value in state["torch_cuda"]]
            )


def capture_numpy_rng(rng: Any) -> dict[str, Any]:
    """Capture either a NumPy Generator or legacy RandomState."""

    if isinstance(rng, np.random.Generator):
        return {
            "kind": "generator",
            "bit_generator": type(rng.bit_generator).__name__,
            "state": copy.deepcopy(rng.bit_generator.state),
        }
    if isinstance(rng, np.random.RandomState):
        return {"kind": "random_state", "state": rng.get_state()}
    raise TypeError(f"Unsupported NumPy RNG type: {type(rng)!r}")


def restore_numpy_rng(current: Any, state: dict[str, Any]) -> Any:
    """Restore a NumPy RNG in place when possible and return the live RNG."""

    if state["kind"] == "generator":
        bit_generator_name = state["bit_generator"]
        if (
            not isinstance(current, np.random.Generator)
            or type(current.bit_generator).__name__ != bit_generator_name
        ):
            bit_generator_type = getattr(np.random, bit_generator_name)
            current = np.random.Generator(bit_generator_type())
        current.bit_generator.state = copy.deepcopy(state["state"])
        return current
    if state["kind"] == "random_state":
        if not isinstance(current, np.random.RandomState):
            current = np.random.RandomState()
        current.set_state(state["state"])
        return current
    raise ValueError(f"Unknown NumPy RNG state kind: {state.get('kind')!r}")


def _environment_root(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def capture_environment_rng_state(env: Any) -> dict[str, Any]:
    """Capture RNGs and reset cursors owned by a supported environment."""

    root = _environment_root(env)
    state: dict[str, Any] = {}
    for name, owner, attribute in (
        ("root_rng", root, "rng"),
        ("root_private_rng", root, "_rng"),
        ("root_np_random", root, "_np_random"),
        ("inner_rng", getattr(root, "env", None), "rng"),
    ):
        if owner is None:
            continue
        rng = getattr(owner, attribute, None)
        if isinstance(rng, (np.random.Generator, np.random.RandomState)):
            state[name] = capture_numpy_rng(rng)

    controller = getattr(root, "_eval_reset_controller", None)
    if controller is not None and hasattr(controller, "get_rng_state"):
        state["reset_controller"] = controller.get_rng_state()
    return state


def restore_environment_rng_state(env: Any, state: dict[str, Any]) -> None:
    """Restore environment state returned by :func:`capture_environment_rng_state`."""

    root = _environment_root(env)
    for name, owner, attribute in (
        ("root_rng", root, "rng"),
        ("root_private_rng", root, "_rng"),
        ("root_np_random", root, "_np_random"),
        ("inner_rng", getattr(root, "env", None), "rng"),
    ):
        if name not in state or owner is None:
            continue
        current = getattr(owner, attribute, None)
        setattr(owner, attribute, restore_numpy_rng(current, state[name]))

    controller = getattr(root, "_eval_reset_controller", None)
    if controller is not None and "reset_controller" in state:
        controller.set_rng_state(state["reset_controller"])


def seed_environment(env: Any, seed: int) -> None:
    """Reset an environment's RNG streams without consuming an episode reset."""

    seed = int(seed)
    root = _environment_root(env)
    seeded_local_rng = False
    for owner, attribute in (
        (root, "rng"),
        (root, "_rng"),
        (root, "_np_random"),
        (getattr(root, "env", None), "rng"),
    ):
        if owner is None:
            continue
        current = getattr(owner, attribute, None)
        if isinstance(current, np.random.Generator):
            replacement = np.random.default_rng(seed)
            current.bit_generator.state = replacement.bit_generator.state
            seeded_local_rng = True
        elif isinstance(current, np.random.RandomState):
            current.seed(seed)
            seeded_local_rng = True

    controller = getattr(root, "_eval_reset_controller", None)
    if controller is not None and hasattr(controller, "seed"):
        controller.seed(seed)

    # LIBERO owns no local RNG and implements seed() via process-global NumPy.
    if not seeded_local_rng and hasattr(env, "seed"):
        env.seed(seed)


def capture_component_rng_state(component: Any) -> Any | None:
    if component is not None and hasattr(component, "get_rng_state"):
        return component.get_rng_state()
    return None


def restore_component_rng_state(component: Any, state: Any | None) -> None:
    if state is not None and component is not None and hasattr(component, "set_rng_state"):
        component.set_rng_state(state)


def seed_component(component: Any, seed: int) -> None:
    if component is not None and hasattr(component, "seed"):
        component.seed(int(seed))


def encode_rng_state(state: Any) -> str:
    """Encode trusted local checkpoint state for storage in a JSON manifest."""

    return base64.b64encode(pickle.dumps(state, protocol=5)).decode("ascii")


def decode_rng_state(encoded: str) -> Any:
    """Decode state produced by :func:`encode_rng_state`."""

    return pickle.loads(base64.b64decode(encoded.encode("ascii")))


def capture_training_rng_state(
    *,
    env: Any,
    base_policy: Any = None,
    replay_buffer: Any = None,
    success_replay_buffer: Any = None,
) -> dict[str, Any]:
    """Capture stochastic state not contained in the learner checkpoint."""

    state = {
        "process": capture_process_rng_state(),
        "environment": capture_environment_rng_state(env),
    }
    policy_state = capture_component_rng_state(base_policy)
    if policy_state is not None:
        state["base_policy"] = policy_state
    if replay_buffer is not None:
        state["replay_buffer"] = replay_buffer.get_rng_state()
    if success_replay_buffer is not None:
        state["success_replay_buffer"] = success_replay_buffer.get_rng_state()
    return state


def restore_training_rng_state(
    state: dict[str, Any],
    *,
    env: Any,
    base_policy: Any = None,
    replay_buffer: Any = None,
    success_replay_buffer: Any = None,
) -> None:
    """Restore state returned by :func:`capture_training_rng_state`."""

    restore_environment_rng_state(env, state.get("environment", {}))
    restore_component_rng_state(base_policy, state.get("base_policy"))
    if replay_buffer is not None and "replay_buffer" in state:
        replay_buffer.set_rng_state(state["replay_buffer"])
    if success_replay_buffer is not None and "success_replay_buffer" in state:
        success_replay_buffer.set_rng_state(state["success_replay_buffer"])
    if "process" in state:
        # Restore process-global state last: environment restoration may invoke
        # library code that touches NumPy's legacy global generator.
        restore_process_rng_state(state["process"])


__all__ = [
    "EVAL_SEED_OFFSET",
    "capture_component_rng_state",
    "capture_environment_rng_state",
    "capture_process_rng_state",
    "capture_training_rng_state",
    "decode_rng_state",
    "encode_rng_state",
    "restore_component_rng_state",
    "restore_environment_rng_state",
    "restore_process_rng_state",
    "restore_training_rng_state",
    "seed_component",
    "seed_environment",
    "seed_process",
]
