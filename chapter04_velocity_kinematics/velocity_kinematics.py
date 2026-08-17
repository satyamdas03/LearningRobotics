"""
Chapter 4 — Demo: Velocity Kinematics.

Shows how the Jacobian maps joint velocities to end-effector twists,
and how to solve the inverse problem to command a desired end-effector motion.
"""
from __future__ import annotations

import math

import numpy as np

from jacobian import ArmJacobian


def print_matrix(name: str, M: np.ndarray, fmt: str = "+.4f") -> None:
    """Pretty-print a matrix."""
    print(f"{name} =")
    for row in M:
        print("  [" + ", ".join(f"{v:{fmt}}" for v in row) + "]")
    print()


def demo_jacobian_agreement() -> None:
    print("=" * 70)
    print("DEMO: Numeric vs analytic Jacobian")
    print("=" * 70)

    jac = ArmJacobian()
    q = np.array([0.0, math.radians(20), -math.radians(15),
                  0.0, math.radians(10), math.radians(30)])

    J_num = jac.jac_numeric(q)
    J_ana = jac.jac_analytic(q)
    error = np.linalg.norm(J_num - J_ana)

    print(f"Configuration q (deg): {np.degrees(q)}")
    print(f"Numeric/analytic Jacobian error: {error:.2e}")
    print()


def demo_twist() -> None:
    print("=" * 70)
    print("DEMO: Joint velocities -> end-effector twist")
    print("=" * 70)

    jac = ArmJacobian()
    q = np.zeros(6)
    qdot = np.array([0.5, 0.0, 0.0, 0.0, 0.0, 0.0])  # waist spin only

    V = jac.twist(q, qdot)
    print(f"qdot = {qdot}")
    print(f"End-effector twist V = [v; w] = {V}")
    print(f"Linear speed: {np.linalg.norm(V[:3]):.4f} m/s")
    print(f"Angular speed: {np.linalg.norm(V[3:]):.4f} rad/s")
    print()


def demo_inverse_velocity() -> None:
    print("=" * 70)
    print("DEMO: Desired end-effector twist -> joint velocities")
    print("=" * 70)

    jac = ArmJacobian()
    q = np.zeros(6)

    # Command: move the end-effector 0.2 m/s in +X, 0.1 m/s in +Z, no rotation.
    V_desired = np.array([0.2, 0.0, 0.1, 0.0, 0.0, 0.0])

    qdot = jac.inverse_twist(q, V_desired, method="pinv")
    V_achieved = jac.twist(q, qdot)

    print(f"Desired twist V_des = {V_desired}")
    print(f"Solved qdot (rad/s) = {qdot}")
    print(f"Achieved twist V    = {V_achieved}")
    print(f"Tracking error: {np.linalg.norm(V_desired - V_achieved):.2e}")
    print()


def demo_null_space() -> None:
    print("=" * 70)
    print("DEMO: Null-space motion")
    print("=" * 70)

    jac = ArmJacobian()
    q = np.zeros(6)

    # Project a pure elbow-velocity command into the null space of J.
    qdot_0 = np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0])
    N = jac.null_space_projector(q)
    qdot_ns = N @ qdot_0

    V = jac.twist(q, qdot_ns)
    print(f"Raw joint velocity qdot_0   = {qdot_0}")
    print(f"Null-space velocity qdot_ns = {qdot_ns}")
    print(f"Resulting EE twist V     = {V}")
    print(f"EE twist norm: {np.linalg.norm(V):.2e}  (should be near zero)")
    print()


def demo_force_duality() -> None:
    print("=" * 70)
    print("DEMO: Static-force duality tau = J^T F")
    print("=" * 70)

    jac = ArmJacobian()
    q = np.zeros(6)

    # A 10 N force pushing the end-effector down (-Z), no moment.
    F = np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0])
    tau = jac.joint_torques_from_force(q, F)

    print(f"Applied wrench F = {F}")
    print(f"Joint torques tau  = {tau}")
    print(f"Torque on shoulder/elbow joints resists the downward force.")
    print()


def main() -> None:
    demo_jacobian_agreement()
    demo_twist()
    demo_inverse_velocity()
    demo_null_space()
    demo_force_duality()

    print("=" * 70)
    print("Chapter 4 practical complete.")
    print("Key takeaway: J(q) relates qdot to end-effector twist; its transpose")
    print("relates end-effector wrench to joint torques; its pseudoinverse")
    print("solves inverse velocity problems.")
    print("=" * 70)


if __name__ == "__main__":
    main()
