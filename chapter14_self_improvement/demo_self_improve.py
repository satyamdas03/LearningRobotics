"""Chapter 14 — Demo: self-improving virtual real-sim-real loop.

This script creates a virtual arm with an unknown constant torque bias,
shows that the baseline controller fails to reach the target, runs online
system identification to estimate the bias, retunes the controller with a
canceling feedforward term, and validates the improvement with A/B testing.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make sibling chapters importable.
ROOT = Path(__file__).parent.parent
for sub in [
    "chapter06_dynamics",
    "chapter07_control",
    "pibench/pibench/realrobot",
    "chapter14_self_improvement",
]:
    d = ROOT / sub
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from control import ComputedTorqueController, JointSpacePIDController
from real_hardware import ArmState, MockRealArm
from failure_detector import FailureDetector
from retuner import Retuner
from self_improvement_loop import SelfImprovementLoop
from system_id import OnlineSystemID
from dynamics import ArmDynamics


class BiasedMockRealArm(MockRealArm):
    """Mock real arm with a constant per-joint torque bias.

    The controller is unaware of the bias.  It commands torques that would
    work on the nominal arm, but the bias pushes each joint away from the
    intended motion, producing a steady-state error that online system ID
    can learn and cancel.
    """

    def __init__(self, bias: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.bias = np.asarray(bias, dtype=float)

    def send_torques(self, tau: np.ndarray, dt: float | None = None) -> None:
        # Inject the unknown constant bias before the real arm sees the torque.
        biased_tau = np.asarray(tau, dtype=float) + self.bias
        return super().send_torques(biased_tau, dt=dt)


def _interpolate_path(waypoints: list[np.ndarray], steps: int) -> list[np.ndarray]:
    """Linear interpolation between waypoints for smooth motion."""
    if len(waypoints) < 2:
        return waypoints
    segments = len(waypoints) - 1
    per_segment = max(1, steps // segments)
    path: list[np.ndarray] = []
    for i in range(segments):
        a, b = waypoints[i], waypoints[i + 1]
        for t in range(per_segment + 1):
            s = t / per_segment
            path.append((1 - s) * a + s * b)
    return path


def make_reference_path(nq: int) -> list[np.ndarray]:
    """A slow, smooth sweep through the joint space for calibration."""
    waypoints = [
        np.zeros(nq),
        np.full(nq, 0.15),
        np.full(nq, -0.1),
        np.zeros(nq),
    ]
    return _interpolate_path(waypoints, steps=40)


def make_task_path(nq: int, dwell: int = 40) -> list[np.ndarray]:
    """The validation task: small reachable joint-space target with dwell."""
    waypoints = [
        np.zeros(nq),
        np.array([0.10, 0.06, 0.04, 0.02, -0.04, 0.0][:nq]),
    ]
    path = _interpolate_path(waypoints, steps=40)
    # Hold the final target so controllers can settle.
    path.extend([path[-1].copy() for _ in range(dwell)])
    return path


def main() -> int:
    xml_path = ROOT / "chapter01_foundation" / "simple_6dof_arm.xml"
    dt = 0.01
    dynamics = ArmDynamics(xml_path=xml_path)
    nq = dynamics.model.nq

    # Baseline controller: computed torque with feedback linearization.
    # A constant torque bias disturbs the arm; computed torque does not know
    # about the bias, so it leaves a steady-state tracking error.
    Kp = np.full(nq, 100.0)
    Kd = np.full(nq, 20.0)
    baseline = ComputedTorqueController(
        dynamics=dynamics,
        Kp=Kp,
        Kd=Kd,
        tau_max=40.0,
    )

    # Unknown constant torque bias applied by the virtual real arm.
    # Small biases on the major joints produce a measurable but stable
    # tracking error that online system ID can learn and cancel.
    true_bias = np.array([0.3, -0.2, 0.15, 0.0, 0.0, 0.0], dtype=float)

    def arm_factory() -> MockRealArm:
        return BiasedMockRealArm(
            bias=true_bias,
            xml_path=str(xml_path),
            dt=dt,
            control_mode="torque",
            gear_ratio=1.0,
            torque_noise_std=0.0,
            position_noise_std=0.0,
            velocity_noise_std=0.0,
        )

    detector = FailureDetector(position_threshold=0.05, residual_threshold=5.0)
    system_id = OnlineSystemID(nominal_xml_path=xml_path)
    retuner = Retuner(
        ComputedTorqueController,
        dynamics=dynamics,
        Kp=Kp,
        Kd=Kd,
        tau_max=40.0,
    )

    loop = SelfImprovementLoop(
        arm_factory=arm_factory,
        baseline_controller=baseline,
        retuner=retuner,
        detector=detector,
        system_id=system_id,
        reference_q_path=make_reference_path(nq),
        dt=dt,
        # Use the learned torque offset as feedforward; do not touch gear ratio.
        retune_kwargs={"disable_offset": False, "offset_gain": 1.0, "gear_compensation_gain": 0.0},
    )

    report = loop.improve(make_task_path(nq), n_ab_trials=10)

    estimated_bias = np.asarray(report.mismatch_estimate["torque_offset"], dtype=float)

    print("=" * 60)
    print("Self-Improvement Loop Report")
    print("=" * 60)
    for line in report.log:
        print(line)
    print("-" * 60)
    print(f"True bias:            {true_bias}")
    print(f"Estimated bias:       {estimated_bias}")
    print(f"Bias estimation error: {np.linalg.norm(true_bias - estimated_bias):.4f}")
    print(f"A/B improved:          {report.ab_result['improved']}")
    print(f"Baseline mean error:   {report.ab_result['baseline_mean_error']:.4f}")
    print(f"Retuned mean error:    {report.ab_result['retuned_mean_error']:.4f}")
    print(f"Baseline successes:    {report.ab_result['baseline_successes']}/10")
    print(f"Retuned successes:     {report.ab_result['retuned_successes']}/10")
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
