"""Versioned messages and validation for the MIDAS real-world server."""

from __future__ import annotations

import itertools
from typing import Any

import numpy as np

from midas.real.config import REAL_PROTOCOL_VERSION, RealRunSpec


_REQUEST_IDS = itertools.count(1)


def request(method: str, **payload: Any) -> dict[str, Any]:
    return {
        "protocol_version": REAL_PROTOCOL_VERSION,
        "request_id": next(_REQUEST_IDS),
        "method": method,
        **payload,
    }


def validate_envelope(message: dict[str, Any]) -> None:
    if int(message.get("protocol_version", -1)) != REAL_PROTOCOL_VERSION:
        raise ValueError(
            f"Protocol mismatch: peer={message.get('protocol_version')}, "
            f"local={REAL_PROTOCOL_VERSION}"
        )
    if "request_id" not in message:
        raise ValueError("Protocol message is missing request_id")


def validate_inference_result(result: dict[str, Any], spec: RealRunSpec) -> None:
    validate_envelope(result)
    for key, shape in (
        ("actions", (spec.query_freq, spec.action_dim)),
        ("base_action", (spec.chunk_len, spec.action_dim)),
        ("a_exec_norm", (spec.query_freq, spec.action_dim)),
    ):
        value = np.asarray(result.get(key))
        if value.shape != shape:
            raise ValueError(f"Server {key} shape {value.shape} does not match {shape}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"Server {key} contains non-finite values")
    normalized = np.asarray(result["a_exec_norm"])
    if not result.get("base_only", False) and np.any(np.abs(normalized) > 1.0001):
        raise ValueError("Server returned normalized actions outside [-1, 1]")


def error_response(request_id: Any, error: BaseException) -> dict[str, Any]:
    return {
        "protocol_version": REAL_PROTOCOL_VERSION,
        "request_id": request_id,
        "ok": False,
        "error": f"{type(error).__name__}: {error}",
    }


__all__ = ["error_response", "request", "validate_envelope", "validate_inference_result"]
