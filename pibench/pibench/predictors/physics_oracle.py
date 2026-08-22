"""Physics oracle — runs the ground-truth computation and returns it.

This is the theoretical upper-bound baseline. It is allowed to inspect the
problem's internal scene and run the same physics the ground truth uses.
"""
from __future__ import annotations

from pibench.core.problem import Prediction, Problem
from pibench.predictors.base import Predictor


class PhysicsOraclePredictor(Predictor):
    """Perfect predictor that returns the problem's own ground truth."""

    name = "physics_oracle"

    def predict(self, problem: Problem) -> Prediction:
        gt = problem.ground_truth()
        return Prediction(answer=gt.answer, reasoning=gt.explanation, confidence=1.0)
