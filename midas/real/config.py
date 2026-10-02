"""Versioned trainer/server contract for real-world MIDAS runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from midas.utils.snapshots import write_json_manifest


REAL_RUN_SPEC_VERSION = 1
REAL_PROTOCOL_VERSION = 1
YAM_ACTION_DIM = 14
YAM_STATE_DIM = 14
YAM_CAMERAS = ("top", "left_wrist", "right_wrist")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class RealRunSpec:
    """Only the immutable fields required to mirror and validate an actor."""

    schema_version: int = REAL_RUN_SPEC_VERSION
    protocol_version: int = REAL_PROTOCOL_VERSION
    algo: str = "midas"
    robot: str = "yam"
    action_dim: int = YAM_ACTION_DIM
    state_dim: int = YAM_STATE_DIM
    cameras: tuple[str, ...] = YAM_CAMERAS
    resize_image: int = 224
    chunk_len: int = 60
    query_freq: int = 30
    predict_a_exec: bool = True
    use_vlm_embedding: bool = True
    vlm_embedding_dim: int = 2048
    midas_use_trust_region: bool = True
    midas_a_star_delta_clip_norm: tuple[float, ...] | None = None
    reward_type: str = "sparse"
    num_subtasks: int = 1
    # ``None`` denotes a legacy v1 run spec, whose base policy implicitly used
    # seed zero. New specs always record the experiment seed explicitly.
    policy_seed: int | None = None
    pi_config: str = ""
    pi_checkpoint: str = ""
    norm_stats_sha256: str = ""
    actor_kwargs: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.schema_version != REAL_RUN_SPEC_VERSION:
            raise ValueError(f"Unsupported real run spec version: {self.schema_version}")
        if self.protocol_version != REAL_PROTOCOL_VERSION:
            raise ValueError(f"Unsupported real protocol version: {self.protocol_version}")
        if self.algo != "midas":
            raise ValueError(f"Only algo='midas' is supported, got {self.algo!r}")
        if self.robot != "yam":
            raise ValueError(f"Only robot='yam' is supported, got {self.robot!r}")
        if self.action_dim != YAM_ACTION_DIM or self.state_dim != YAM_STATE_DIM:
            raise ValueError("YAM action and state dimensions must both be 14")
        if tuple(self.cameras) != YAM_CAMERAS:
            raise ValueError(f"YAM camera order must be {YAM_CAMERAS}, got {self.cameras}")
        if not 0 < self.query_freq <= self.chunk_len:
            raise ValueError("Expected 0 < query_freq <= chunk_len")
        if not self.predict_a_exec:
            raise ValueError("Real-world MIDAS requires predict_a_exec=True")
        if self.reward_type not in {"sparse", "dense"}:
            raise ValueError("reward_type must be 'sparse' or 'dense'")
        if self.reward_type == "dense" and self.num_subtasks < 1:
            raise ValueError("Dense reward requires num_subtasks >= 1")
        cap = self.midas_a_star_delta_clip_norm
        if self.midas_use_trust_region and (cap is None or len(cap) != self.action_dim):
            raise ValueError(
                "Enabled trust region requires one normalized radius per action dimension"
            )

    @property
    def actor_signature(self) -> str:
        fields = {
            "action_dim": self.action_dim,
            "state_dim": self.state_dim,
            "cameras": self.cameras,
            "resize_image": self.resize_image,
            "chunk_len": self.chunk_len,
            "query_freq": self.query_freq,
            "predict_a_exec": self.predict_a_exec,
            "use_vlm_embedding": self.use_vlm_embedding,
            "vlm_embedding_dim": self.vlm_embedding_dim,
            "midas_use_trust_region": self.midas_use_trust_region,
            "midas_a_star_delta_clip_norm": self.midas_a_star_delta_clip_norm,
            "actor_kwargs": self.actor_kwargs,
        }
        return _canonical_hash(fields)

    @property
    def spec_hash(self) -> str:
        fields = asdict(self)
        # Preserve hashes of run specs written before policy_seed was added.
        if self.policy_seed is None:
            fields.pop("policy_seed")
        return _canonical_hash(fields)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["actor_signature"] = self.actor_signature
        result["spec_hash"] = self.spec_hash
        return result

    def write(self, path: str | Path) -> None:
        write_json_manifest(self.to_dict(), path)

    @classmethod
    def read(cls, path: str | Path) -> "RealRunSpec":
        with open(path, encoding="utf-8") as handle:
            raw = json.load(handle)
        expected_actor = raw.pop("actor_signature", None)
        expected_spec = raw.pop("spec_hash", None)
        raw["cameras"] = tuple(raw.get("cameras", YAM_CAMERAS))
        cap = raw.get("midas_a_star_delta_clip_norm")
        raw["midas_a_star_delta_clip_norm"] = tuple(cap) if cap is not None else None
        spec = cls(**raw)
        if expected_actor is not None and expected_actor != spec.actor_signature:
            raise ValueError("real_run_spec actor signature hash does not match its contents")
        if expected_spec is not None and expected_spec != spec.spec_hash:
            raise ValueError("real_run_spec hash does not match its contents")
        return spec


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "REAL_PROTOCOL_VERSION",
    "REAL_RUN_SPEC_VERSION",
    "RealRunSpec",
    "YAM_ACTION_DIM",
    "YAM_CAMERAS",
    "YAM_STATE_DIM",
    "sha256_file",
]
