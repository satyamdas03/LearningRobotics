"""Chapter 14 — Orchestrator for the self-improving virtual real-sim-real loop.

The loop is:

1. Run a validation task on a virtual arm with injected mismatch.
2. Detect failure and characterize it (tracking error vs dynamics mismatch).
3. Calibrate online system ID on a reference trajectory.
4. Retune the controller with the estimated mismatch.
5. Run A/B validation to prove improvement.
6. Return a report and, if successful, a retuned controller ready for reuse.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class ImprovementReport:
    """Outcome of one self-improvement iteration."""

    initial_failure: dict[str, Any]
    mismatch_estimate: dict[str, Any]
    retuning_params: dict[str, Any]
    ab_result: dict[str, Any]
    success: bool
    log: list[str] = field(default_factory=list)


class SelfImprovementLoop:
    """Close the detect → identify → retune → validate loop."""

    def __init__(
        self,
        arm_factory,
        baseline_controller,
        retuner,
        detector,
        system_id,
        reference_q_path: list[np.ndarray],
        dt: float = 0.01,
        retune_kwargs: dict[str, Any] | None = None,
    ) -> None:
        self.arm_factory = arm_factory
        self.baseline_controller = baseline_controller
        self.retuner = retuner
        self.detector = detector
        self.system_id = system_id
        self.reference_q_path = reference_q_path
        self.dt = dt
        self.retune_kwargs = retune_kwargs or {}

    def improve(self, task_q_path: list[np.ndarray], n_ab_trials: int = 20) -> ImprovementReport:
        """Run one full self-improvement iteration.

        Parameters
        ----------
        task_q_path
            Joint-space waypoints that define the validation task.
        n_ab_trials
            Number of randomized arms to use in the A/B comparison.
        """
        from ab_experiment import ABExperiment, _run_episode

        log: list[str] = []

        # 1. Run baseline on a single mismatched arm to detect failure.
        arm = self.arm_factory()
        state = arm.get_state()
        for q_target in task_q_path:
            tau = self.baseline_controller.compute(
                state.q, state.qdot, q_des=q_target, dt=self.dt
            )
            arm.send_torques(tau, dt=self.dt)
            state = arm.get_state()

        initial_report = self.detector.check_reach(
            "initial", task_q_path[-1], state.q
        )
        log.append(
            f"Initial reach: error={initial_report.details['position_error']:.4f}, "
            f"axis={initial_report.primary_axis}, failed={initial_report.failed}"
        )

        # 2. Online system ID on the reference trajectory.
        estimate = self.system_id.calibrate_on_trajectory(
            self.arm_factory(),
            self.reference_q_path,
            self.baseline_controller,
            dt=self.dt,
        )
        log.append(
            f"System ID: n_samples={estimate.n_samples}, "
            f"mean_residual_norm={float(np.linalg.norm(estimate.mean_residual)):.4f}, "
            f"gear_ratio_mean={float(np.mean(estimate.estimated_gear_ratio)):.4f}"
        )

        # 3. Retune.
        retuning = self.retuner.retune(estimate, **self.retune_kwargs)
        log.append("Retuned controller ready.")

        # 4. A/B validation.
        experiment = ABExperiment(
            arm_factory=self.arm_factory,
            baseline_controller=self.baseline_controller,
            retuned_controller=retuning.controller,
            q_path=task_q_path,
            detector=self.detector,
            dt=self.dt,
        )
        ab_result = experiment.run(n_trials=n_ab_trials)
        log.append(
            f"A/B: baseline {ab_result.baseline_successes}/{n_ab_trials}, "
            f"retuned {ab_result.retuned_successes}/{n_ab_trials}, "
            f"improved={ab_result.improved}"
        )

        return ImprovementReport(
            initial_failure={
                "failed": initial_report.failed,
                "axis": initial_report.primary_axis,
                "position_error": initial_report.details["position_error"],
                "residual_magnitude": initial_report.residual_magnitude,
            },
            mismatch_estimate={
                "torque_offset": estimate.torque_offset.tolist(),
                "estimated_gear_ratio": estimate.estimated_gear_ratio.tolist(),
                "mean_residual": estimate.mean_residual.tolist(),
                "max_residual": estimate.max_residual.tolist(),
                "n_samples": estimate.n_samples,
            },
            retuning_params=retuning.params,
            ab_result={
                "baseline_successes": ab_result.baseline_successes,
                "retuned_successes": ab_result.retuned_successes,
                "baseline_mean_error": float(np.mean(ab_result.baseline_errors)),
                "retuned_mean_error": float(np.mean(ab_result.retuned_errors)),
                "improved": ab_result.improved,
                "p_value_approx": ab_result.p_value_approx,
            },
            success=ab_result.improved,
            log=log,
        )
