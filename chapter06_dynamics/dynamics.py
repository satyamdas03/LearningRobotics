"""
Chapter 6 — Practical: Dynamics.

Implements the Euler-Lagrange / recursive Newton-Euler equations of motion for
the simple 6-DOF arm and validates them against MuJoCo's built-in algorithms.

Equations:
  tau = M(q) qddot + C(q, qdot) qdot + g(q)
  qddot = M(q)^{-1} (tau - C(q, qdot) qdot - g(q))
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


class ArmDynamics:
    """Compute mass matrix, Coriolis+gravity, forward dynamics, and inverse dynamics
    for the simple_6dof_arm.xml robot."""

    def __init__(self, xml_path: str | None = None) -> None:
        if xml_path is None:
            xml_path = str(
                Path(__file__).parent.parent
                / "chapter01_foundation"
                / "simple_6dof_arm.xml"
            )
        with open(xml_path, "r", encoding="utf-8") as f:
            self._xml = f.read()
        self.model = mujoco.MjModel.from_xml_string(self._xml)
        self.data = mujoco.MjData(self.model)

    def set_state(self, q: np.ndarray, qdot: np.ndarray | None = None) -> None:
        """Set configuration and velocity and run forward kinematics."""
        self.data.qpos[:] = q
        if qdot is not None:
            self.data.qvel[:] = qdot
        mujoco.mj_forward(self.model, self.data)

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        """Return the dense nq x nq mass matrix M(q)."""
        self.set_state(q)
        M = np.zeros((self.model.nq, self.model.nq))
        mujoco.mj_fullM(self.model, self.data, M)
        return M

    def coriolis_gravity(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Return C(q, qdot) qdot + g(q), the velocity-dependent and gravity terms.

        We use MuJoCo's inverse dynamics with the desired acceleration set to zero,
        which returns the total bias force (Coriolis + gravity) required to keep
        the joint accelerations at zero.
        """
        self.set_state(q, qdot)
        # mj_inverse with qacc = 0 gives qfrc_inverse = C(q,qdot) qdot + g(q).
        self.data.qacc[:] = 0.0
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse.copy()

    def gravity_term(self, q: np.ndarray) -> np.ndarray:
        """Return g(q) (gravity only), obtained with zero velocity."""
        return self.coriolis_gravity(q, np.zeros(self.model.nq))

    def coriolis_term(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Return C(q, qdot) qdot (velocity-dependent terms only)."""
        return self.coriolis_gravity(q, qdot) - self.gravity_term(q)

    def forward_dynamics(
        self, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray
    ) -> np.ndarray:
        """Compute qddot = M(q)^{-1} (tau - C(q,qdot) qdot - g(q))."""
        M = self.mass_matrix(q)
        bias = self.coriolis_gravity(q, qdot)
        return np.linalg.solve(M, tau - bias)

    def inverse_dynamics(
        self, q: np.ndarray, qdot: np.ndarray, qddot: np.ndarray
    ) -> np.ndarray:
        """Compute tau = M(q) qddot + C(q,qdot) qdot + g(q)."""
        self.set_state(q, qdot)
        self.data.qacc[:] = qddot
        mujoco.mj_inverse(self.model, self.data)
        return self.data.qfrc_inverse.copy()

    def step(
        self, q: np.ndarray, qdot: np.ndarray, tau: np.ndarray, dt: float | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Integrate one step with the given joint torques using Euler integration.

        Uses our own forward_dynamics to compute qddot, then updates q and qdot.
        """
        if dt is None:
            dt = float(self.model.opt.timestep)
        qddot = self.forward_dynamics(q, qdot, tau)
        q_new = q + dt * qdot
        qdot_new = qdot + dt * qddot
        return q_new, qdot_new


if __name__ == "__main__":
    dyn = ArmDynamics()
    q = np.zeros(dyn.model.nq)
    qdot = np.zeros(dyn.model.nq)

    M = dyn.mass_matrix(q)
    print("Mass matrix at home configuration:")
    print(M)
    print("Eigenvalues:", np.linalg.eigvalsh(M))

    bias = dyn.coriolis_gravity(q, qdot)
    print("\nGravity term at home configuration:", bias)

    tau = np.zeros(dyn.model.nq)
    qddot = dyn.forward_dynamics(q, qdot, tau)
    print("\nAcceleration under zero torque:", qddot)
