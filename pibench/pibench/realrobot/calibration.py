"""PIBench Phase 7 — Online sim-to-real mismatch calibration.

The tracker compares the acceleration predicted by a nominal MuJoCo model with
the acceleration observed on the real (or mocked) arm.  The mean residual can be
converted into an additive torque offset that a controller can feed forward.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Nominal dynamics lives in Chapter 6.
_DYN_DIR = Path(__file__).parent.parent.parent.parent / "chapter06_dynamics"
if str(_DYN_DIR) not in sys.path:
    sys.path.insert(0, str(_DYN_DIR))

from dynamics import ArmDynamics  # noqa: E402


class ResidualTracker:
    """Track the model-mismatch residual ``qddot_actual - qddot_predicted``."""

    def __init__(
        self,
        nominal_xml_path: str | Path | None = None,
    ) -> None:
        self.nominal = ArmDynamics(xml_path=nominal_xml_path)
        self._residuals: list[np.ndarray] = []
        self._prev_state: tuple[np.ndarray, np.ndarray] | None = None

    @property
    def residuals(self) -> list[np.ndarray]:
        return self._residuals

    def reset(self) -> None:
        """Clear recorded residuals."""
        self._residuals = []
        self._prev_state = None

    def observe(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        tau_commanded: np.ndarray,
        qdot_next: np.ndarray,
        dt: float,
    ) -> np.ndarray:
        """Record one residual from a real step.

        ``qdot_next`` is the velocity *after* the real arm stepped for ``dt``.
        """
        qddot_actual = (qdot_next - qdot) / dt
        qddot_predicted = self.nominal.forward_dynamics(q, qdot, tau_commanded)
        residual = qddot_actual - qddot_predicted
        self._residuals.append(residual)
        return residual

    def mean_residual(self) -> np.ndarray:
        """Return the mean acceleration residual across all observations."""
        if not self._residuals:
            return np.zeros(self.nominal.model.nq)
        return np.mean(np.array(self._residuals), axis=0)

    def torque_offset(self, q: np.ndarray) -> np.ndarray:
        """Convert the mean residual into an additive torque offset at ``q``."""
        M = self.nominal.mass_matrix(q)
        return M @ self.mean_residual()

    def run_calibration_episode(
        self,
        arm,
        q_path: list[np.ndarray],
        controller,
        dt: float = 0.01,
        settle_steps: int = 10,
    ) -> dict:
        """Execute a reference trajectory and record residuals.

        ``controller`` is called as ``controller(state, q_target) -> tau``.
        """
        self.reset()
        for q_target in q_path:
            for _ in range(settle_steps):
                state = arm.get_state()
                if hasattr(controller, "compute"):
                    tau = controller.compute(state.q, state.qdot, q_des=q_target, dt=dt)
                else:
                    tau = controller(state, q_target)
                arm.send_torques(tau, dt=dt)
                next_state = arm.get_state()
                self.observe(
                    state.q,
                    state.qdot,
                    tau,
                    next_state.qdot,
                    dt,
                )
        return {
            "mean_residual": self.mean_residual().tolist(),
            "max_residual": (
                np.max(np.abs(np.array(self._residuals)), axis=0).tolist()
                if self._residuals
                else []
            ),
            "n_samples": len(self._residuals),
        }
