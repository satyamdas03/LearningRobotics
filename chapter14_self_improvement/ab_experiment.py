"""Chapter 14 — A/B validation of a retuned controller.

Runs a validation task with two controllers on the same injected mismatch and
compares success rate / tracking error.  The result is a data record that can be
saved to the skill library or fed back into the reasoning loop.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ABResult:
    """Outcome of an A/B controller comparison."""

    baseline_successes: int
    retuned_successes: int
    baseline_errors: list[float] = field(default_factory=list)
    retuned_errors: list[float] = field(default_factory=list)
    baseline_residuals: list[float] = field(default_factory=list)
    retuned_residuals: list[float] = field(default_factory=list)
    improved: bool = False
    p_value_approx: float | None = None


def _run_episode(
    arm_factory,
    controller,
    q_path: list[np.ndarray],
    dt: float,
    detector,
) -> tuple[bool, float, float]:
    """Run one reach episode and return (success, final_error, mean_residual)."""
    arm = arm_factory()
    if hasattr(controller, "reset"):
        controller.reset()
    residuals: list[float] = []
    state = arm.get_state()
    for q_target in q_path:
        tau = controller.compute(state.q, state.qdot, q_des=q_target, dt=dt)
        arm.send_torques(tau, dt=dt)
        next_state = arm.get_state()
        # Simple residual proxy: difference between desired and actual velocity change.
        residuals.append(float(np.linalg.norm(next_state.qdot - state.qdot)))
        state = next_state
    final_error = float(np.linalg.norm(q_path[-1] - state.q))
    report = detector.check_reach("episode", q_path[-1], state.q)
    return not report.failed, final_error, float(np.mean(residuals)) if residuals else 0.0


class ABExperiment:
    """Compare baseline vs retuned controller across randomized virtual arms."""

    def __init__(
        self,
        arm_factory,
        baseline_controller,
        retuned_controller,
        q_path: list[np.ndarray],
        detector,
        dt: float = 0.01,
    ) -> None:
        self.arm_factory = arm_factory
        self.baseline = baseline_controller
        self.retuned = retuned_controller
        self.q_path = q_path
        self.detector = detector
        self.dt = dt

    def run(self, n_trials: int = 20) -> ABResult:
        """Run ``n_trials`` A/B episodes and return comparison statistics."""
        baseline_successes = 0
        retuned_successes = 0
        baseline_errors: list[float] = []
        retuned_errors: list[float] = []
        baseline_residuals: list[float] = []
        retuned_residuals: list[float] = []

        for _ in range(n_trials):
            success_b, err_b, res_b = _run_episode(
                self.arm_factory, self.baseline, self.q_path, self.dt, self.detector
            )
            baseline_successes += int(success_b)
            baseline_errors.append(err_b)
            baseline_residuals.append(res_b)

            success_r, err_r, res_r = _run_episode(
                self.arm_factory, self.retuned, self.q_path, self.dt, self.detector
            )
            retuned_successes += int(success_r)
            retuned_errors.append(err_r)
            retuned_residuals.append(res_r)

        # Approximate paired-test: retuned better than baseline on more than
        # half the trials with a margin.
        improvements = sum(
            1 for b, r in zip(baseline_errors, retuned_errors) if r < b
        )
        n = len(baseline_errors)
        p_approx = improvements / n if n > 0 else 0.0

        return ABResult(
            baseline_successes=baseline_successes,
            retuned_successes=retuned_successes,
            baseline_errors=baseline_errors,
            retuned_errors=retuned_errors,
            baseline_residuals=baseline_residuals,
            retuned_residuals=retuned_residuals,
            improved=retuned_successes > baseline_successes
            or (retuned_successes == baseline_successes and p_approx > 0.6),
            p_value_approx=p_approx,
        )
