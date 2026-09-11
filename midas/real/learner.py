"""Construction of an identical MIDAS learner on trainer and server."""

from __future__ import annotations

import numpy as np

from midas.agents.midas.midas_learner import MidasLearner
from midas.real.config import RealRunSpec


def create_learner(spec: RealRunSpec, seed: int) -> MidasLearner:
    observations = {
        "pixels": np.zeros(
            (1, spec.resize_image, spec.resize_image, 3 * len(spec.cameras), 1), dtype=np.uint8
        ),
        "state": np.zeros((1, spec.state_dim, 1), dtype=np.float32),
        "base_action": np.zeros((1, spec.chunk_len, spec.action_dim, 1), dtype=np.float32),
    }
    if spec.use_vlm_embedding:
        observations["vlm_embedding"] = np.zeros((1, spec.vlm_embedding_dim, 1), dtype=np.float32)
    actions = np.zeros((1, spec.query_freq, spec.action_dim), dtype=np.float32)
    kwargs = dict(spec.actor_kwargs)
    kwargs.update(
        {
            "predict_a_exec": True,
            "use_vlm_embedding": spec.use_vlm_embedding,
            "midas_use_trust_region": spec.midas_use_trust_region,
            "midas_a_star_delta_clip_norm": spec.midas_a_star_delta_clip_norm,
        }
    )
    return MidasLearner(seed=seed, observations=observations, actions=actions, **kwargs)


__all__ = ["create_learner"]
