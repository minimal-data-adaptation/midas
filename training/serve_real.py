"""Versioned, fail-closed Pi0.5 + MIDAS server for YAM."""

from __future__ import annotations

import argparse
import asyncio
import http
import logging
import os
import time
from typing import Any

import numpy as np
from openpi_client import msgpack_numpy
import websockets
import websockets.asyncio.server

from midas.real.config import REAL_PROTOCOL_VERSION, RealRunSpec
from midas.real.protocol import error_response, validate_envelope
from midas.utils.reproducibility import (
    capture_component_rng_state,
    restore_component_rng_state,
)


LOGGER = logging.getLogger(__name__)


def _vlm_feature(value, expected_dimension: int) -> np.ndarray | None:
    if value is None:
        return None
    if isinstance(value, (tuple, list)) and len(value) == 2:
        value = value[0]
    array = np.asarray(value, dtype=np.float32)
    if array.ndim >= 3 and array.shape[0] == 1:
        array = array[0]
    if array.ndim == 2:
        array = array.mean(axis=0)
    array = array.reshape(-1)
    if array.shape != (expected_dimension,):
        raise ValueError(f"VLM feature shape {array.shape} does not match ({expected_dimension},)")
    return array


class MockBasePolicy:
    """Zero-action base used only by the hardware-free process smoke test."""

    metadata = {"mock_base_policy": True}

    def __init__(self, spec: RealRunSpec):
        self.spec = spec

    def infer(self, observation, *, return_normalized=False, return_vlm_embedding=False):
        result = {
            "state": np.asarray(observation["state"], dtype=np.float32),
            "actions": np.zeros((self.spec.chunk_len, self.spec.action_dim), dtype=np.float32),
        }
        if return_vlm_embedding:
            result["vlm_embedding"] = np.zeros(self.spec.vlm_embedding_dim, dtype=np.float32)
        return result

    def apply_output_transforms(self, output):
        return {"actions": np.asarray(output["actions"], dtype=np.float32)}


class RealPolicy:
    def __init__(
        self,
        spec: RealRunSpec,
        base_policy,
        actor,
        *,
        residual_checkpoint_loaded: bool = False,
        allow_base_only: bool = False,
    ) -> None:
        self.spec = spec
        self.base = base_policy
        self.actor = actor
        self.actor_version = 0
        self.armed = bool(residual_checkpoint_loaded)
        self.allow_base_only = allow_base_only
        self._lock = asyncio.Lock()

    def metadata(self) -> dict[str, Any]:
        return {
            "ok": True,
            "protocol_version": REAL_PROTOCOL_VERSION,
            "actor_signature": self.spec.actor_signature,
            "spec_hash": self.spec.spec_hash,
            "actor_version": self.actor_version,
            "armed": self.armed,
            "allow_base_only": self.allow_base_only,
            "base_policy_metadata": getattr(self.base, "metadata", {}),
        }

    def _base(self, observation: dict) -> tuple[dict, np.ndarray, np.ndarray | None]:
        output = self.base.infer(
            observation,
            return_normalized=True,
            return_vlm_embedding=self.spec.use_vlm_embedding,
        )
        raw_actions = np.asarray(output["actions"], dtype=np.float32)
        if (
            raw_actions.ndim != 2
            or raw_actions.shape[0] < self.spec.chunk_len
            or raw_actions.shape[1] < 14
        ):
            raise ValueError(f"Base policy action shape is incompatible: {raw_actions.shape}")
        base = raw_actions[: self.spec.chunk_len, :14]
        if not np.all(np.isfinite(base)):
            raise ValueError("Base policy returned non-finite actions")
        feature = None
        if self.spec.use_vlm_embedding:
            feature = _vlm_feature(output.get("vlm_embedding"), self.spec.vlm_embedding_dim)
        return output, base, feature

    def infer_base(self, observation: dict) -> dict[str, Any]:
        _, base, feature = self._base(observation)
        result = {"base_action": base}
        if feature is not None:
            result["vlm_embedding"] = feature
        return result

    async def get_base_rng_state(self) -> dict[str, Any]:
        """Snapshot the server-owned frozen-policy sampling stream."""

        async with self._lock:
            return {"base_rng_state": capture_component_rng_state(self.base)}

    async def set_base_rng_state(self, state: Any) -> dict[str, Any]:
        """Restore the frozen-policy stream as part of trainer resume."""

        if state is None:
            raise ValueError("base_rng_state is required")
        async with self._lock:
            restore_component_rng_state(self.base, state)
        return {"base_rng_state_restored": True}

    def _actor_observation(
        self, observation: dict, base: np.ndarray, feature: np.ndarray | None
    ) -> dict[str, np.ndarray]:
        images = []
        for camera in self.spec.cameras:
            image = np.asarray(observation["images"][camera])
            if image.ndim == 3 and image.shape[0] == 3:
                image = np.transpose(image, (1, 2, 0))
            if image.shape != (self.spec.resize_image, self.spec.resize_image, 3):
                raise ValueError(f"Camera {camera} has incompatible shape {image.shape}")
            images.append(image)
        result = {
            "pixels": np.concatenate(images, axis=-1)[np.newaxis, ..., np.newaxis].astype(np.uint8),
            "state": np.asarray(observation["state"], np.float32)[np.newaxis, ..., np.newaxis],
            "base_action": base[np.newaxis, ..., np.newaxis],
        }
        if self.spec.use_vlm_embedding:
            if feature is None:
                raise ValueError(
                    "The actor requires a VLM feature, but the base policy returned none"
                )
            result["vlm_embedding"] = feature[np.newaxis, ..., np.newaxis]
        return result

    async def infer(self, observation: dict) -> dict[str, Any]:
        if not self.armed:
            if self.allow_base_only:
                base_output, base, feature = self._base(observation)
                physical = self.base.apply_output_transforms(base_output)["actions"]
                actions = np.asarray(physical, np.float32)[: self.spec.query_freq, :14]
                result = {
                    "actions": actions,
                    "base_action": base,
                    "a_exec_norm": base[: self.spec.query_freq],
                    "actor_version": self.actor_version,
                    "base_only": True,
                }
                if feature is not None:
                    result["vlm_embedding"] = feature
                return result
            raise RuntimeError(
                "Real policy server is disarmed; push a validated actor before inference"
            )

        base_output, base, feature = self._base(observation)
        actor_observation = self._actor_observation(observation, base, feature)
        async with self._lock:
            flat = np.asarray(self.actor.eval_actions(actor_observation))[0]
            actor_version = self.actor_version
        normalized = flat.reshape(self.spec.query_freq, self.spec.action_dim)
        if not np.all(np.isfinite(normalized)):
            raise ValueError("MIDAS actor returned non-finite normalized actions")
        if np.any(np.abs(normalized) > 1.0001):
            raise ValueError("MIDAS actor returned normalized actions outside [-1, 1]")
        normalized = np.clip(normalized, -1.0, 1.0).astype(np.float32)

        # OpenPI output transforms expect the base model's padded action width.
        raw_base = np.asarray(base_output["actions"])
        full = np.zeros_like(raw_base, dtype=np.float32)
        full[: self.spec.query_freq, : self.spec.action_dim] = normalized
        if raw_base.shape[0] > self.spec.query_freq:
            full[self.spec.query_freq :, : self.spec.action_dim] = normalized[-1]
        transformed = self.base.apply_output_transforms(
            {"state": base_output["state"], "actions": full}
        )
        actions = np.asarray(transformed["actions"], dtype=np.float32)[: self.spec.query_freq, :14]
        if actions.shape != (self.spec.query_freq, 14) or not np.all(np.isfinite(actions)):
            raise ValueError(f"Unsafe physical action result: shape={actions.shape}")
        result = {
            "actions": actions,
            "base_action": base,
            "a_exec_norm": normalized,
            "actor_version": actor_version,
            "base_only": False,
        }
        if feature is not None:
            result["vlm_embedding"] = feature
        return result

    async def update_actor_state(self, actor_state: dict, actor_signature: str) -> dict[str, Any]:
        if actor_signature != self.spec.actor_signature:
            raise ValueError("Actor update signature does not match server run spec")
        async with self._lock:
            self.actor.replace_actor_state(actor_state)
            self.actor_version += 1
            self.armed = True
            return {"actor_version": self.actor_version, "armed": True}

    async def disarm(self) -> dict[str, Any]:
        async with self._lock:
            self.armed = False
        return {"armed": False, "actor_version": self.actor_version}


class RealWebsocketServer:
    def __init__(self, policy: RealPolicy, host: str, port: int, api_key: str | None = None):
        self.policy = policy
        self.host = host
        self.port = port
        self.api_key = api_key

    async def _process_request(self, connection, request):
        if request.path == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        if self.api_key:
            expected = f"Api-Key {self.api_key}"
            if request.headers.get("Authorization") != expected:
                return connection.respond(http.HTTPStatus.UNAUTHORIZED, "Unauthorized\n")
        return None

    async def _handler(self, websocket) -> None:
        owned_actor_version = None
        try:
            async for wire in websocket:
                request_id = None
                started = time.monotonic()
                try:
                    if isinstance(wire, str):
                        raise ValueError("Real protocol requires binary msgpack messages")
                    message = msgpack_numpy.unpackb(wire)
                    request_id = message.get("request_id")
                    validate_envelope(message)
                    method = message.get("method")
                    if method == "metadata" or method == "health":
                        result = self.policy.metadata()
                    elif method == "infer_base":
                        result = self.policy.infer_base(message["observation"])
                    elif method == "infer":
                        result = await self.policy.infer(message["observation"])
                    elif method == "get_base_rng_state":
                        result = await self.policy.get_base_rng_state()
                    elif method == "set_base_rng_state":
                        result = await self.policy.set_base_rng_state(
                            message["base_rng_state"]
                        )
                    elif method == "update_actor_state":
                        result = await self.policy.update_actor_state(
                            message["actor_state"], message["actor_signature"]
                        )
                        owned_actor_version = result["actor_version"]
                    elif method == "disarm":
                        result = await self.policy.disarm()
                        owned_actor_version = None
                    else:
                        raise ValueError(f"Unknown real policy method: {method!r}")
                    response = {
                        "protocol_version": REAL_PROTOCOL_VERSION,
                        "request_id": request_id,
                        "ok": True,
                        **result,
                        "server_timing_ms": (time.monotonic() - started) * 1000.0,
                    }
                except Exception as error:
                    LOGGER.exception("Real policy request failed")
                    response = error_response(request_id, error)
                await websocket.send(msgpack_numpy.packb(response))
        finally:
            # A trainer connection owns the actor version it last pushed. If
            # that connection vanishes without an explicit disarm, fail closed
            # unless a newer trainer has already replaced its actor.
            if owned_actor_version is not None and self.policy.actor_version == owned_actor_version:
                await self.policy.disarm()

    def serve_forever(self) -> None:
        async def run():
            async with websockets.asyncio.server.serve(
                self._handler,
                self.host,
                self.port,
                compression=None,
                max_size=None,
                process_request=self._process_request,
            ) as server:
                await server.serve_forever()

        asyncio.run(run())


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Serve frozen Pi0.5 plus a synchronized MIDAS actor"
    )
    parser.add_argument("--run_spec", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--api_key", default="")
    parser.add_argument("--default_prompt", default=None)
    parser.add_argument("--residual_checkpoint", default="")
    parser.add_argument("--allow_base_only", action="store_true")
    parser.add_argument("--mock_base_policy", action="store_true")
    parser.add_argument("--mem_fraction", default=0.45, type=float)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", str(args.mem_fraction))
    logging.basicConfig(level=logging.INFO, force=True)
    spec = RealRunSpec.read(args.run_spec)
    from midas.real.learner import create_learner

    policy_seed = int(spec.policy_seed if spec.policy_seed is not None else 0)
    actor = create_learner(spec, seed=policy_seed)
    restored = False
    if args.residual_checkpoint:
        actor.restore_checkpoint(args.residual_checkpoint)
        restored = True
    if args.mock_base_policy:
        base = MockBasePolicy(spec)
    else:
        if not spec.pi_config or not spec.pi_checkpoint:
            raise ValueError("Real Pi server requires pi_config and pi_checkpoint in real_run_spec")
        from openpi.policies import policy_config
        from openpi.training import config

        base = policy_config.create_trained_policy(
            config.get_config(spec.pi_config),
            spec.pi_checkpoint,
            default_prompt=args.default_prompt,
            seed=policy_seed,
        )
    policy = RealPolicy(
        spec,
        base,
        actor,
        residual_checkpoint_loaded=restored,
        allow_base_only=args.allow_base_only,
    )
    RealWebsocketServer(policy, args.host, args.port, args.api_key or None).serve_forever()


if __name__ == "__main__":
    main()
