"""MIDAS: residual policy-agnostic reinforcement learning."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from midas.agents.midas.midas_learner import MidasLearner

__all__ = ["MidasLearner"]


def __getattr__(name: str):
    if name == "MidasLearner":
        from midas.agents.midas.midas_learner import MidasLearner

        return MidasLearner
    raise AttributeError(name)
