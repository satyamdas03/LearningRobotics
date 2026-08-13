"""Base predictor interface for PIBench."""
from __future__ import annotations

from abc import ABC, abstractmethod

from pibench.core.problem import Prediction, Problem


class Predictor(ABC):
    """A predictor answers a PIBench problem.

    Predictors can be:
      - analytical baselines (e.g., physics oracle)
      - neural network models
      - VLMs called via API
      - humans via CLI
    """

    name: str = "base"

    @abstractmethod
    def predict(self, problem: Problem) -> Prediction:
        """Answer the given problem."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
