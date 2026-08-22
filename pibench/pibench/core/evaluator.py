"""Evaluation metrics for PIBench results."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic import BaseModel

from pibench.core.runner import RunResult, SuiteResult
from pibench.evaluation.metrics import accuracy_per_concept, calibration_metrics


class SuiteMetrics(BaseModel):
    suite: str
    n_total: int
    n_correct: int
    accuracy: float


class Evaluator:
    """Compute aggregate metrics from a RunResult."""

    @staticmethod
    def metrics(run_result: RunResult) -> dict[str, Any]:
        suite_metrics: list[SuiteMetrics] = []
        per_problem: dict[str, dict[str, Any]] = defaultdict(
            lambda: {"total": 0, "correct": 0.0}
        )
        all_results = [r for s in run_result.suites for r in s.results]

        for suite_result in run_result.suites:
            total = len(suite_result.results)
            correct = sum(r.score for r in suite_result.results)
            suite_metrics.append(
                SuiteMetrics(
                    suite=suite_result.suite,
                    n_total=total,
                    n_correct=int(correct),
                    accuracy=correct / total if total else 0.0,
                )
            )
            for r in suite_result.results:
                per_problem[r.problem]["total"] += 1
                per_problem[r.problem]["correct"] += r.score

        per_problem_acc = {
            name: {"accuracy": stats["correct"] / stats["total"], **stats}
            for name, stats in per_problem.items()
        }

        per_concept = accuracy_per_concept(all_results)
        calibration = calibration_metrics(all_results)

        return {
            "predictor": run_result.predictor,
            "overall_accuracy": run_result.overall_accuracy,
            "n_total": sum(m.n_total for m in suite_metrics),
            "n_correct": sum(m.n_correct for m in suite_metrics),
            "suite_metrics": suite_metrics,
            "per_problem_accuracy": per_problem_acc,
            "per_concept_accuracy": per_concept,
            "calibration": calibration.model_dump(),
        }
