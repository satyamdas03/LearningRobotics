"""Chapter 14 — Failure detection for validation results.

A small rule-based detector that turns ``ValidationResult`` (or any scalar
error/metric) into a structured failure report.  The retuner can use this
to decide whether to trigger system identification and which mismatch axis
to investigate first.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class FailureReport:
    """Structured description of a validation failure."""

    failed: bool
    task_id: str
    primary_axis: str | None
    residual_magnitude: float
    details: dict[str, Any]


class FailureDetector:
    """Decide whether a validation episode failed and why.

    Parameters
    ----------
    position_threshold
        Position error (m) above which a reach is considered failed.
    residual_threshold
        Mean acceleration residual (rad/s²) above which model mismatch is
        suspected.
    """

    def __init__(
        self,
        position_threshold: float = 0.05,
        residual_threshold: float = 5.0,
    ) -> None:
        self.position_threshold = float(position_threshold)
        self.residual_threshold = float(residual_threshold)

    def check_reach(
        self,
        task_id: str,
        q_desired: np.ndarray,
        q_actual: np.ndarray,
        residual: np.ndarray | None = None,
    ) -> FailureReport:
        """Check a joint-space reach against tolerance and residual."""
        position_error = float(np.linalg.norm(q_desired - q_actual))
        residual_magnitude = (
            float(np.linalg.norm(residual)) if residual is not None else 0.0
        )

        failed = position_error > self.position_threshold
        primary_axis = None
        if failed and residual_magnitude > self.residual_threshold:
            primary_axis = "dynamics_mismatch"
        elif failed:
            primary_axis = "tracking_error"
        elif residual_magnitude > self.residual_threshold:
            primary_axis = "dynamics_mismatch"
            failed = True

        return FailureReport(
            failed=failed,
            task_id=task_id,
            primary_axis=primary_axis,
            residual_magnitude=residual_magnitude,
            details={
                "position_error": position_error,
                "position_threshold": self.position_threshold,
                "residual_threshold": self.residual_threshold,
            },
        )

    def check_validation_result(self, result: Any) -> FailureReport:
        """Convenience wrapper around a PIBench ``ValidationResult``."""
        metrics = getattr(result, "metrics", {}) or {}
        q_desired = np.asarray(metrics.get("q_desired", []), dtype=float)
        q_actual = np.asarray(metrics.get("q_actual", []), dtype=float)
        residual = np.asarray(metrics.get("mean_residual", []), dtype=float)
        return self.check_reach(
            task_id=getattr(result, "task_id", "unknown"),
            q_desired=q_desired,
            q_actual=q_actual,
            residual=residual,
        )
