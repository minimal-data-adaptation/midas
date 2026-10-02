"""Thread-safe client for real MIDAS inference and actor synchronization."""

from __future__ import annotations

import threading
import time
from typing import Any

import jax
import numpy as np
from openpi_client import msgpack_numpy
import websockets.sync.client

from midas.real.config import RealRunSpec
from midas.real.protocol import request, validate_envelope, validate_inference_result


def _host_tree(tree):
    tree = jax.device_get(tree)
    return jax.tree_util.tree_map(
        lambda value: (
            np.asarray(value) if isinstance(value, (jax.Array, np.ndarray, np.generic)) else value
        ),
        tree,
        is_leaf=lambda value: value is None,
    )


class RealPolicyClient:
    def __init__(
        self,
        spec: RealRunSpec,
        host: str = "localhost",
        port: int = 8000,
        *,
        api_key: str | None = None,
        connect_timeout: float = 300.0,
    ) -> None:
        self.spec = spec
        self.uri = host if host.startswith(("ws://", "wss://")) else f"ws://{host}:{port}"
        self.api_key = api_key
        self._lock = threading.Lock()
        self._connection = self._connect(connect_timeout)
        metadata = self.call("metadata")
        if metadata.get("actor_signature") != spec.actor_signature:
            self.close()
            raise ValueError("Server actor signature does not match trainer real_run_spec")
        if metadata.get("spec_hash") != spec.spec_hash:
            self.close()
            raise ValueError("Server real_run_spec does not match the client")

    def _connect(self, timeout: float):
        deadline = time.monotonic() + timeout
        headers = {"Authorization": f"Api-Key {self.api_key}"} if self.api_key else None
        last_error = None
        while time.monotonic() < deadline:
            try:
                return websockets.sync.client.connect(
                    self.uri, compression=None, max_size=None, additional_headers=headers
                )
            except (ConnectionRefusedError, OSError) as error:
                last_error = error
                time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
        raise TimeoutError(f"Could not connect to real policy server at {self.uri}") from last_error

    def call(self, method: str, **payload: Any) -> dict[str, Any]:
        message = request(method, **payload)
        with self._lock:
            self._connection.send(msgpack_numpy.packb(message))
            wire = self._connection.recv()
        if isinstance(wire, str):
            raise RuntimeError(wire)
        response = msgpack_numpy.unpackb(wire)
        validate_envelope(response)
        if response["request_id"] != message["request_id"]:
            raise RuntimeError("Mismatched WebSocket response request_id")
        if not response.get("ok", True):
            raise RuntimeError(response.get("error", "Unknown server error"))
        return response

    def infer(self, observation: dict) -> dict[str, Any]:
        response = self.call("infer", observation=observation)
        validate_inference_result(response, self.spec)
        return response

    def infer_base(self, observation: dict) -> dict[str, Any]:
        """Run only the frozen policy; safe for demo preprocessing while disarmed."""

        response = self.call("infer_base", observation=observation)
        base = np.asarray(response.get("base_action"))
        expected = (self.spec.chunk_len, self.spec.action_dim)
        if base.shape != expected or not np.all(np.isfinite(base)):
            raise ValueError(f"Server base_action shape/value mismatch: {base.shape} != {expected}")
        return response

    def update_actor_state(self, actor_state: dict[str, Any]) -> int:
        response = self.call(
            "update_actor_state",
            actor_signature=self.spec.actor_signature,
            actor_state=_host_tree(actor_state),
        )
        return int(response["actor_version"])

    def get_base_rng_state(self) -> Any | None:
        """Return the frozen policy's private sampling state from the server."""

        return self.call("get_base_rng_state").get("base_rng_state")

    def set_base_rng_state(self, state: Any) -> None:
        """Restore a base-policy state previously returned by the server."""

        self.call("set_base_rng_state", base_rng_state=_host_tree(state))

    def disarm(self) -> None:
        self.call("disarm")

    def close(self) -> None:
        connection = getattr(self, "_connection", None)
        if connection is not None:
            connection.close()
            self._connection = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


__all__ = ["RealPolicyClient"]
