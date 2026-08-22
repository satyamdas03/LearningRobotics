"""Tests for PIBench Phase 6 evaluation utilities."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.core.runner import ProblemResult, RunResult, SuiteResult
from pibench.core.suite import Suite
from pibench.evaluation.leaderboard import build_leaderboard, render_leaderboard_html
from pibench.evaluation.metrics import (
    accuracy_per_concept,
    brier_score,
    calibration_metrics,
    expected_calibration_error,
    negative_log_likelihood,
)
from pibench.harness import EvaluationHarness


@register_problem("test_suite")
class DummyProblem(Problem):
    """Minimal problem for testing evaluation plumbing."""

    def __init__(self, seed: int = 0) -> None:
        super().__init__(seed=seed)
        self._suite = "test_suite"

    def _build_scene(self) -> None:
        pass

    def question(self) -> Question:
        return Question(
            text="Is the sky blue?",
            answer_type=AnswerType.BOOLEAN,
        )

    def ground_truth(self) -> GroundTruth:
        return GroundTruth(
            answer=True,
            explanation="yes",
            latent_params={"seed": self.seed},
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if prediction.answer is True else 0.0

    def concept_tags(self) -> list[str]:
        return ["test_suite", "dummy_concept"]


def make_problem_result(score: float, confidence: float | None, concepts: list[str]) -> ProblemResult:
    return ProblemResult(
        suite="test_suite",
        problem="DummyProblem",
        seed=0,
        question_text="Is the sky blue?",
        predicted=True,
        predicted_reasoning=None,
        predicted_confidence=confidence,
        ground_truth=True,
        gt_explanation=None,
        latent_params={},
        concepts=concepts,
        score=score,
    )


def test_expected_calibration_error_perfectly_calibrated():
    # Confidence equals observed accuracy: ECE = 0.
    scores = [1.0] * 5 + [0.0] * 5
    confidences = [1.0] * 5 + [0.0] * 5
    assert expected_calibration_error(scores, confidences, n_bins=5) == pytest.approx(0.0, abs=1e-9)


def test_expected_calibration_error_miscalibrated():
    # 5 correct at confidence 0.9, 5 incorrect at confidence 0.9.
    scores = [1.0] * 5 + [0.0] * 5
    confidences = [0.9] * 10
    ece = expected_calibration_error(scores, confidences, n_bins=5)
    assert ece == pytest.approx(0.4, abs=1e-9)


def test_brier_score():
    scores = [1.0, 0.0, 1.0]
    confidences = [1.0, 0.0, 0.5]
    assert brier_score(scores, confidences) == pytest.approx((0 + 0 + 0.25) / 3)


def test_negative_log_likelihood():
    scores = [1.0, 0.0]
    confidences = [0.9, 0.2]
    nll = negative_log_likelihood(scores, confidences)
    expected = -0.5 * (np.log(0.9) + np.log(0.8))
    assert nll == pytest.approx(expected)


def test_negative_log_likelihood_invalid_confidence():
    assert negative_log_likelihood([1.0], [1.0]) is None
    assert negative_log_likelihood([1.0], [0.0]) is None


def test_accuracy_per_concept():
    results = [
        make_problem_result(1.0, None, ["a", "b"]),
        make_problem_result(0.0, None, ["a", "c"]),
        make_problem_result(1.0, None, ["b", "c"]),
    ]
    acc = accuracy_per_concept(results)
    assert acc["a"]["accuracy"] == 0.5
    assert acc["b"]["accuracy"] == 1.0
    assert acc["c"]["accuracy"] == 0.5


def test_calibration_metrics_handles_missing_confidence():
    results = [
        make_problem_result(1.0, None, []),
        make_problem_result(0.0, None, []),
    ]
    cal = calibration_metrics(results)
    assert cal.ece == 0.0
    assert cal.brier == 0.0
    assert cal.nll is None
    assert cal.n_with_confidence == 0


def test_calibration_metrics_with_confidences():
    results = [
        make_problem_result(1.0, 0.9, []),
        make_problem_result(0.0, 0.2, []),
    ]
    cal = calibration_metrics(results)
    assert cal.n_with_confidence == 2
    assert cal.ece == pytest.approx(0.15, abs=1e-9)


def test_build_leaderboard(tmp_path: Path):
    # Construct a tiny RunResult manually.
    result = RunResult(
        predictor="test_predictor",
        suites=[
            SuiteResult(
                suite="test_suite",
                results=[
                    ProblemResult(
                        suite="test_suite",
                        problem="DummyProblem",
                        seed=0,
                        question_text="q",
                        predicted=True,
                        predicted_reasoning=None,
                        predicted_confidence=0.9,
                        ground_truth=True,
                        gt_explanation=None,
                        latent_params={},
                        concepts=["test_suite"],
                        score=1.0,
                    )
                ],
            )
        ],
    )
    result_path = tmp_path / "results_test_predictor.json"
    result_path.write_text(json.dumps(result.model_dump()), encoding="utf-8")

    leaderboard = build_leaderboard(result_paths=[result_path])
    assert len(leaderboard.entries) == 1
    entry = leaderboard.entries[0]
    assert entry.predictor == "test_predictor"
    assert entry.overall_accuracy == 1.0
    assert entry.calibration.n_with_confidence == 1


def test_render_leaderboard_html():
    leaderboard = build_leaderboard(result_paths=[])
    html = render_leaderboard_html(leaderboard)
    assert "PIBench Leaderboard" in html
    assert "<table>" in html


def test_harness_evaluate_runs(tmp_path: Path):
    from pibench.predictors.physics_oracle import PhysicsOraclePredictor

    harness = EvaluationHarness(PhysicsOraclePredictor(), output_dir=tmp_path)
    suite = Suite("test_suite", seed=0, n_instances=3)
    metrics = harness.evaluate([suite])
    assert metrics["predictor"] == "physics_oracle"
    assert metrics["n_total"] == 3
    assert metrics["n_correct"] == 3.0
    assert (tmp_path / "results_physics_oracle.json").exists()
