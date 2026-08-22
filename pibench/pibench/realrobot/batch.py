"""PIBench Phase 7 extension — batch virtual real-robot validation.

Run the same validation tasks across many randomized virtual arms and
controller variants, then report how prediction accuracy degrades as the
sim-to-sim gap grows.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from pibench.realrobot.harness import RealRobotValidationHarness
from pibench.realrobot.protocol import ValidationResult, ValidationTask


@dataclass
class BatchValidationReport:
    """Result of a batch validation sweep."""

    records: list[dict]
    summary: dict

    def accuracy_by_mismatch(self) -> dict[float, float]:
        """Return {mismatch_level: mean_accuracy}."""
        acc: dict[float, list[float]] = {}
        for r in self.records:
            acc.setdefault(r["mismatch_level"], []).append(r["accuracy"])
        return {level: float(np.mean(scores)) for level, scores in acc.items()}


class BatchValidator:
    """Run validation tasks across randomized virtual arms and controllers.

    Each arm is generated from the same base model but with an injected
    sim-to-sim gap controlled by ``mismatch_level``.  A level of ``0.0`` means
    the arm matches the controller's nominal model; higher levels inject more
    realistic actuator/sensor imperfections.
    """

    def __init__(
        self,
        tasks: list[ValidationTask],
        mismatch_levels: list[float] | None = None,
        seeds: list[int] | None = None,
        controller_names: list[str] | None = None,
        dt: float = 0.01,
    ) -> None:
        self.tasks = tasks
        self.mismatch_levels = mismatch_levels or [0.0, 0.2, 0.5, 1.0]
        self.seeds = seeds or [0, 1, 2]
        self.controller_names = controller_names or ["pid"]
        self.dt = dt

    @staticmethod
    def _make_randomized_arm(
        seed: int,
        mismatch_level: float,
        dt: float = 0.01,
    ) -> MockRealArm:
        """Create a virtual arm whose imperfections scale with ``mismatch_level``."""
        import sys
        from pathlib import Path

        ROOT = Path(__file__).parent.parent.parent.parent
        ch7 = ROOT / "chapter07_control"
        if str(ch7) not in sys.path:
            sys.path.insert(0, str(ch7))
        from real_hardware import MockRealArm

        rng = np.random.default_rng(seed)
        # Imperfections scale linearly with mismatch_level.
        nq = 6
        return MockRealArm(
            dt=dt,
            gear_ratio=1.0 + mismatch_level * rng.uniform(-0.15, 0.15, size=nq),
            torque_noise_std=mismatch_level * rng.uniform(0.0, 2.0),
            velocity_noise_std=mismatch_level * rng.uniform(0.0, 0.05),
            position_bias=mismatch_level * rng.uniform(-0.03, 0.03, size=nq),
            position_noise_std=mismatch_level * rng.uniform(0.0, 0.01),
            coulomb_friction=mismatch_level * rng.uniform(0.0, 1.0, size=nq),
            viscous_friction=mismatch_level * rng.uniform(0.0, 0.2, size=nq),
            feedback_delay_steps=int(round(mismatch_level * rng.uniform(0.0, 5.0))),
            torque_delay_steps=int(round(mismatch_level * rng.uniform(0.0, 3.0))),
            torque_lag_time_constant=mismatch_level * rng.uniform(0.0, 0.03),
            fixed_random_seed=seed,
        )

    def _make_controller(
        self,
        name: str,
        arm: MockRealArm,
    ) -> Callable[[Any, np.ndarray], np.ndarray]:
        """Return a controller closure for the given controller name."""
        if name == "pid":
            pid = RealRobotValidationHarness._inertia_scaled_pid(omega=6.0)
            return lambda state, q_des: pid.compute(
                state.q, state.qdot, q_des=q_des, dt=self.dt
            )
        if name == "computed_torque":
            import sys
            from pathlib import Path

            ROOT = Path(__file__).parent.parent.parent.parent
            ch6 = ROOT / "chapter06_dynamics"
            ch7 = ROOT / "chapter07_control"
            for p in (ch6, ch7):
                if str(p) not in sys.path:
                    sys.path.insert(0, str(p))
            from control import ComputedTorqueController
            from dynamics import ArmDynamics

            dyn = ArmDynamics()
            M_diag = np.diag(dyn.mass_matrix(np.zeros(dyn.model.nq)))
            Kp = 50.0 * np.ones(dyn.model.nq)
            Kd = 14.0 * np.ones(dyn.model.nq)
            ctrl = ComputedTorqueController(
                dyn, Kp=Kp, Kd=Kd, tau_max=dyn.model.actuator_ctrlrange[:, 1]
            )
            return lambda state, q_des: ctrl.compute(
                state.q, state.qdot, q_des=q_des, qdot_des=np.zeros_like(q_des)
            )
        raise ValueError(f"Unknown controller name: {name!r}")

    def run(self) -> BatchValidationReport:
        """Execute the full sweep and return a report."""
        records: list[dict] = []
        for mismatch_level in self.mismatch_levels:
            for seed in self.seeds:
                for controller_name in self.controller_names:
                    arm = self._make_randomized_arm(seed, mismatch_level, self.dt)
                    try:
                        controller = self._make_controller(controller_name, arm)
                        harness = RealRobotValidationHarness(
                            arm, dt=self.dt, controller_factory=controller
                        )
                        results = harness.run_tasks(copy.deepcopy(self.tasks))
                        accuracy = sum(r.match for r in results) / len(results)
                    finally:
                        arm.close()
                    records.append(
                        {
                            "mismatch_level": mismatch_level,
                            "seed": seed,
                            "controller": controller_name,
                            "accuracy": accuracy,
                            "n_tasks": len(results),
                            "n_matched": sum(r.match for r in results),
                        }
                    )

        summary = {
            "n_runs": len(records),
            "mean_accuracy_by_level": {
                level: float(np.mean([r["accuracy"] for r in records if r["mismatch_level"] == level]))
                for level in self.mismatch_levels
            },
        }
        return BatchValidationReport(records=records, summary=summary)
