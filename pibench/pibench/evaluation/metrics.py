"""Calibration and concept-level metrics for PIBench.

This module extends the per-suite accuracy in :mod:`pibench.core.evaluator`
with:

* per-concept accuracy (concepts come from problem tags + latent parameter keys)
* confidence calibration: ECE, Brier score, and negative log-likelihood
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Any

import numpy as np
from pydantic import BaseModel

from pibench.core.runner import ProblemResult


class ConceptMetrics(BaseModel):
    """Accuracy aggregated by a single concept tag."""

    concept: str
    n_total: int
    n_correct: float
    accuracy: float


class CalibrationMetrics(BaseModel):
    """Confidence-calibration summary."""

    ece: float
    brier: float
    nll: float | None
    n_with_confidence: int
    n_total: int


def accuracy_per_concept(results: list[ProblemResult]) -> dict[str, dict[str, Any]]:
    """Return accuracy grouped by each problem's concept tags."""
    stats: dict[str, dict[str, Any]] = defaultdict(lambda: {"total": 0, "correct": 0.0})
    for r in results:
        for concept in r.concepts:
            stats[concept]["total"] += 1
            stats[concept]["correct"] += r.score

    return {
        concept: {
            "concept": concept,
            "n_total": data["total"],
            "n_correct": data["correct"],
            "accuracy": data["correct"] / data["total"] if data["total"] else 0.0,
        }
        for concept, data in sorted(stats.items())
    }


def expected_calibration_error(
    scores: list[float],
    confidences: list[float],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins.

    Each entry is a single prediction where ``score`` is 1.0 if correct and
    ``confidence`` is the model's reported confidence in [0, 1].
    """
    if not scores or not confidences:
        return 0.0

    scores = np.asarray(scores, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    if len(scores) != len(confidences):
        raise ValueError("scores and confidences must have the same length")

    # Clamp to valid probability range.
    confidences = np.clip(confidences, 0.0, 1.0)

    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(scores)
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        if high == 1.0:
            # Include the right edge in the last bin.
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        if not np.any(mask):
            continue
        bin_scores = scores[mask]
        bin_conf = confidences[mask]
        bin_accuracy = float(np.mean(bin_scores))
        bin_confidence = float(np.mean(bin_conf))
        bin_weight = float(np.sum(mask)) / n_total
        ece += bin_weight * abs(bin_accuracy - bin_confidence)
    return ece


def brier_score(scores: list[float], confidences: list[float]) -> float:
    """Compute the mean Brier score: (confidence - score)^2."""
    if not scores or not confidences:
        return 0.0
    scores = np.asarray(scores, dtype=float)
    confidences = np.asarray(confidences, dtype=float)
    return float(np.mean((confidences - scores) ** 2))


def negative_log_likelihood(scores: list[float], confidences: list[float]) -> float | None:
    """Compute average NLL treating ``confidence`` as P(correct).

    Returns ``None`` if any confidence is outside (0, 1).
    """
    if not scores or not confidences:
        return None
    confidences = np.asarray(confidences, dtype=float)
    scores = np.asarray(scores, dtype=float)
    if np.any(confidences <= 0.0) or np.any(confidences >= 1.0):
        return None
    # NLL = -mean(score * log(conf) + (1 - score) * log(1 - conf))
    eps = 1e-12
    nlls = -(
        scores * np.log(confidences + eps)
        + (1.0 - scores) * np.log(1.0 - confidences + eps)
    )
    return float(np.mean(nlls))


def calibration_metrics(results: list[ProblemResult], n_bins: int = 10) -> CalibrationMetrics:
    """Compute calibration metrics for all results that carry a confidence."""
    pairs = [
        (float(r.score), float(r.predicted_confidence))
        for r in results
        if r.predicted_confidence is not None
    ]
    if not pairs:
        return CalibrationMetrics(
            ece=0.0,
            brier=0.0,
            nll=None,
            n_with_confidence=0,
            n_total=len(results),
        )

    scores, confidences = zip(*pairs)
    return CalibrationMetrics(
        ece=expected_calibration_error(scores, confidences, n_bins=n_bins),
        brier=brier_score(scores, confidences),
        nll=negative_log_likelihood(scores, confidences),
        n_with_confidence=len(pairs),
        n_total=len(results),
    )
