"""PIBench Phase 7 — Real-robot validation harness."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

# Make the Chapter 6/7 control packages importable without installing them.
_REPO_ROOT = Path(__file__).parent.parent.parent.parent
_CH6 = _REPO_ROOT / "chapter06_dynamics"
_CH7 = _REPO_ROOT / "chapter07_control"
for _p in (_CH6, _CH7):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from control import JointSpacePIDController  # noqa: E402
from dynamics import ArmDynamics  # noqa: E402
from real_hardware import ArmState, MockRealArm, RealArm  # noqa: E402

from pibench.realrobot.protocol import ValidationResult, ValidationTask


class RealRobotValidationHarness:
    """Run PIBench-style predictions against a real (or mocked) robot arm.

    Phase 7 is intentionally minimal: it defines the protocol, executes simple
    action primitives, observes the physical outcome, and scores whether the
    prediction matches reality.  When hardware arrives, swap ``MockRealArm`` for
    a real adapter without changing the harness.
    """

    def __init__(
        self,
        arm: RealArm,
        dt: float = 0.01,
        controller_factory=None,
    ) -> None:
        self.arm = arm
        self.dt = dt
        self.controller_factory = controller_factory
        self.log: list[str] = []

    @staticmethod
    def _inertia_scaled_pid(omega: float = 6.0) -> JointSpacePIDController:
        """Return a stable joint-space PID tuned to the 6-DOF arm inertia."""
        dyn = ArmDynamics()
        M_diag = np.diag(dyn.mass_matrix(np.zeros(dyn.model.nq)))
        Kp = omega**2 * M_diag
        Kd = 2.0 * np.sqrt(Kp * M_diag)
        return JointSpacePIDController(
            dyn, Kp=Kp, Kd=Kd, gravity_comp=True, tau_max=dyn.model.actuator_ctrlrange[:, 1]
        )

    def _run_reach_q(self, task: ValidationTask) -> ValidationResult:
        params = task.action_params
        q_des = np.asarray(params.get("target_q"), dtype=float)
        duration = float(params.get("duration", 1.0))
        tolerance = float(params.get("tolerance", 0.05))

        if self.controller_factory is None:
            controller = self._inertia_scaled_pid(omega=6.0)
        else:
            controller = None  # type: ignore[assignment]

        n_steps = max(1, int(round(duration / self.dt)))
        for _ in range(n_steps):
            state = self.arm.get_state()
            if controller is not None:
                tau = controller.compute(state.q, state.qdot, q_des=q_des, dt=self.dt)
            else:
                tau = self.controller_factory(state, q_des)
            self.arm.send_torques(tau, dt=self.dt)

        final_state = self.arm.get_state()
        final_error = float(np.linalg.norm(q_des - final_state.q))
        actual = "reached" if final_error <= tolerance else "missed"
        match = self._answers_match(task.predicted_answer, actual)

        return ValidationResult(
            task_id=task.task_id,
            predicted=task.predicted_answer,
            actual=actual,
            match=match,
            log=[
                f"target_q={np.round(q_des, 4).tolist()}",
                f"final_q={np.round(final_state.q, 4).tolist()}",
                f"final_error={final_error:.4f}",
            ],
            metrics={"final_error": final_error, "tolerance": tolerance},
        )

    @staticmethod
    def _answers_match(predicted: str, actual: str) -> bool:
        pred_yes = predicted.lower() in {"yes", "true", "1", "reached"}
        actual_yes = actual.lower() in {"yes", "true", "1", "reached"}
        return pred_yes == actual_yes

    def run_task(self, task: ValidationTask) -> ValidationResult:
        """Execute a validation task and return the observed outcome."""
        if task.action_type == "reach_q":
            return self._run_reach_q(task)
        raise NotImplementedError(f"Action type '{task.action_type}' is not implemented")

    def run_tasks(self, tasks: list[ValidationTask]) -> list[ValidationResult]:
        """Run a batch of tasks and return their results."""
        return [self.run_task(t) for t in tasks]

    @staticmethod
    def create_mock_arm(
        xml_path: str | Path | None = None,
        dt: float = 0.01,
        torque_noise_std: float = 0.0,
        velocity_noise_std: float = 0.0,
    ) -> MockRealArm:
        """Convenience factory for the sim-to-real stand-in arm."""
        return MockRealArm(
            xml_path=xml_path,
            dt=dt,
            torque_noise_std=torque_noise_std,
            velocity_noise_std=velocity_noise_std,
        )
