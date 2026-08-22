"""Random-guess predictor baseline."""
from __future__ import annotations

import numpy as np

from pibench.core.problem import AnswerType, Prediction, Problem
from pibench.predictors.base import Predictor


class RandomPredictor(Predictor):
    """Predicts a uniformly random valid answer."""

    name = "random"

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def predict(self, problem: Problem) -> Prediction:
        question = problem.question()
        confidence: float | None = None
        if question.answer_type == AnswerType.BOOLEAN:
            answer = bool(self.rng.integers(0, 2))
            confidence = 0.5
        elif question.answer_type == AnswerType.CHOICE and question.choices:
            answer = self.rng.choice(question.choices)
            confidence = 1.0 / len(question.choices)
        elif question.answer_type == AnswerType.NUMERIC:
            # Use a standard normal scaled to unit magnitude as a naive guess.
            answer = float(self.rng.standard_normal())
        elif question.answer_type == AnswerType.ORDER and question.choices:
            perm = self.rng.permutation(len(question.choices))
            answer = [question.choices[i] for i in perm]
        else:
            answer = None
        return Prediction(answer=answer, reasoning="random guess", confidence=confidence)
