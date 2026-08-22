"""Execution runner for PIBench suites."""
from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from pibench.core.problem import GroundTruth, Prediction, Problem
from pibench.core.registry import get_suite_of
from pibench.core.suite import Suite
from pibench.predictors.base import Predictor


class ProblemResult(BaseModel):
    suite: str
    problem: str
    seed: int
    question_text: str
    predicted: Any
    predicted_reasoning: str | None
    predicted_confidence: float | None = None
    ground_truth: Any
    gt_explanation: str | None
    latent_params: dict[str, Any]
    concepts: list[str] = Field(default_factory=list)
    score: float


class SuiteResult(BaseModel):
    suite: str
    results: list[ProblemResult] = Field(default_factory=list)

    @property
    def accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)


class RunResult(BaseModel):
    predictor: str
    suites: list[SuiteResult] = Field(default_factory=list)

    @property
    def overall_accuracy(self) -> float:
        all_results = [r for s in self.suites for r in s.results]
        if not all_results:
            return 0.0
        return sum(r.score for r in all_results) / len(all_results)


class Runner:
    """Run a predictor against one or more suites."""

    def __init__(self, predictor: Predictor) -> None:
        self.predictor = predictor

    def run(self, suites: list[Suite]) -> RunResult:
        suite_results: list[SuiteResult] = []
        for suite in suites:
            sr = SuiteResult(suite=suite.name)
            for problem in suite.problems():
                question = problem.question()
                prediction = self.predictor.predict(problem)
                gt = problem.ground_truth()
                score = problem.score(prediction)
                suite_name = get_suite_of(problem.__class__) or suite.name
                sr.results.append(
                    ProblemResult(
                        suite=suite_name,
                        problem=problem.__class__.__name__,
                        seed=problem.seed,
                        question_text=question.text,
                        predicted=prediction.answer,
                        predicted_reasoning=prediction.reasoning,
                        predicted_confidence=prediction.confidence,
                        ground_truth=gt.answer,
                        gt_explanation=gt.explanation,
                        latent_params=gt.latent_params,
                        concepts=problem.concept_tags(),
                        score=score,
                    )
                )
            suite_results.append(sr)
        return RunResult(predictor=self.predictor.name, suites=suite_results)
