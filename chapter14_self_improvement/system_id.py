"""Chapter 14 — Online system identification for the virtual arm.

Wraps the Chapter 7/PIBench ``ResidualTracker`` and converts a trajectory of
residuals into a simple corrective model: an additive torque offset plus an
estimated gear-ratio perturbation.  The retuner can add the torque offset as a
feedforward term or scale its commanded torque to compensate for the gear error.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class MismatchEstimate:
    """Estimated mismatch parameters from a calibration episode."""

    torque_offset: np.ndarray
    estimated_gear_ratio: np.ndarray
    mean_residual: np.ndarray
    max_residual: np.ndarray
    n_samples: int


class OnlineSystemID:
    """Estimate simple mismatch terms from observed residuals.

    Parameters
    ----------
    nominal_xml_path
        Path to the model the controller thinks it is controlling.
    """

    def __init__(self, nominal_xml_path: str | Path | None = None) -> None:
        # Import here to keep the module usable without the full pibench tree.
        import sys

        _CAL_DIR = Path(__file__).parent.parent / "pibench" / "pibench" / "realrobot"
        if str(_CAL_DIR) not in sys.path:
            sys.path.insert(0, str(_CAL_DIR))
        from calibration import ResidualTracker

        self.tracker = ResidualTracker(nominal_xml_path=nominal_xml_path)
        self._history: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []

    def calibrate_on_trajectory(
        self,
        arm,
        q_path: list[np.ndarray],
        controller,
        dt: float = 0.01,
        settle_steps: int = 10,
    ) -> MismatchEstimate:
        """Run a calibration episode and return mismatch estimates."""
        self.tracker.reset()
        self._history = []

        for q_target in q_path:
            for _ in range(settle_steps):
                state = arm.get_state()
                tau = controller.compute(state.q, state.qdot, q_des=q_target, dt=dt)
                arm.send_torques(tau, dt=dt)
                next_state = arm.get_state()
                self.tracker.observe(state.q, state.qdot, tau, next_state.qdot, dt)
                self._history.append((state.q, state.qdot, tau, next_state.qdot))

        report = {
            "mean_residual": self.tracker.mean_residual().tolist(),
            "max_residual": (
                np.max(np.abs(np.array(self.tracker.residuals)), axis=0).tolist()
                if self.tracker.residuals
                else []
            ),
            "n_samples": len(self.tracker.residuals),
        }

        mean_residual = np.asarray(report["mean_residual"], dtype=float)
        max_residual = np.asarray(report["max_residual"], dtype=float)

        # Torque offset: the mean missing torque M(q_i) @ residual_i across the
        # calibration trajectory.  This correctly averages configuration-dependent
        # mass matrices and is the feedforward term that cancels a constant bias.
        if self._history and self.tracker.residuals:
            residuals_array = np.array(self.tracker.residuals)
            n = min(len(self._history), residuals_array.shape[0])
            torque_offsets = np.array([
                self.tracker.nominal.mass_matrix(q) @ residuals_array[i]
                for i, (q, _, _, _) in enumerate(self._history[:n])
            ])
            torque_offset = np.mean(torque_offsets, axis=0)
        else:
            torque_offset = np.zeros(self.tracker.nominal.model.nq)

        # Gear ratio estimate: compare actual acceleration magnitude to predicted
        # acceleration magnitude per joint.  Use the history recorded above.
        ratio = np.ones_like(mean_residual)
        if self._history:
            predicted = np.array([
                self.tracker.nominal.forward_dynamics(q, qdot, tau)
                for q, qdot, tau, _ in self._history
            ])
            residuals_array = np.array(self.tracker.residuals)
            n = min(predicted.shape[0], residuals_array.shape[0])
            if n > 0:
                actual = predicted[:n] + residuals_array[:n]
                p_mean = np.mean(np.abs(predicted[:n]), axis=0)
                a_mean = np.mean(np.abs(actual[:n]), axis=0)
                # Only estimate gear ratio for joints that were meaningfully
                # excited during calibration; otherwise leave ratio = 1.
                min_signal = 0.1
                mask = p_mean > min_signal
                ratio[mask] = np.clip(a_mean[mask] / p_mean[mask], 0.7, 1.3)

        return MismatchEstimate(
            torque_offset=torque_offset,
            estimated_gear_ratio=ratio,
            mean_residual=mean_residual,
            max_residual=max_residual,
            n_samples=report["n_samples"],
        )

    def reset(self) -> None:
        """Clear calibration history."""
        self.tracker.reset()
        self._history = []
