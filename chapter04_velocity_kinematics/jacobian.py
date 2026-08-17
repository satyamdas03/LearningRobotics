"""
Chapter 4 — Practical: Velocity Kinematics & Jacobians.

Demonstrates:
  * The Jacobian J(q) mapping joint velocities q̇ to end-effector twist V.
  * Numeric Jacobian via MuJoCo's built-in site Jacobian.
  * Analytic space Jacobian built from the screw axes of Chapter 3.
  * End-effector twist V = [v; ω] = J(q) q̇.
  * Inverse velocity: q̇ = J⁺(q) V_desired (damped pseudoinverse).
  * Static-force duality: τ = Jᵀ(q) F.
  * Null-space projection: (I - J⁺J) q̇_0 moves joints without moving the EE.
"""
from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np


def rotx(theta: float) -> np.ndarray:
    """SO(3) rotation about X."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def roty(theta: float) -> np.ndarray:
    """SO(3) rotation about Y."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rotz(theta: float) -> np.ndarray:
    """SO(3) rotation about Z."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def homogeneous_transform(R: np.ndarray, p: np.ndarray) -> np.ndarray:
    """Build 4x4 SE(3) matrix from rotation and translation."""
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = np.asarray(p, dtype=float)
    return T


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    """Convert a 3-vector to a 3x3 skew-symmetric matrix."""
    x, y, z = v
    return np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])


class ArmJacobian:
    """Jacobian engine for the simple_6dof_arm.xml robot."""

    # Joint axes in the zero-configuration (used to compute current axes).
    joint_axes = np.array([
        [0.0, 0.0, 1.0],    # joint 1: Z (waist)
        [0.0, 1.0, 0.0],    # joint 2: Y (shoulder)
        [0.0, 1.0, 0.0],    # joint 3: Y (elbow)
        [1.0, 0.0, 0.0],    # joint 4: X (wrist roll)
        [0.0, 1.0, 0.0],    # joint 5: Y (wrist pitch)
        [0.0, 0.0, 1.0],    # joint 6: Z (wrist yaw)
    ])

    # Link lengths between consecutive joints and base offset.
    base_offset = np.array([0.0, 0.0, 0.08])
    link_lengths = np.array([0.5, 0.5, 0.4, 0.15, 0.12])
    ee_offset = np.array([0.06, 0.0, 0.0])

    def __init__(self, xml_path: str | None = None) -> None:
        if xml_path is None:
            xml_path = str(
                Path(__file__).parent.parent
                / "chapter01_foundation"
                / "simple_6dof_arm.xml"
            )
        with open(xml_path, "r", encoding="utf-8") as f:
            self.model = mujoco.MjModel.from_xml_string(f.read())
        self.data = mujoco.MjData(self.model)
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee")

    def set_q(self, q: np.ndarray) -> None:
        """Set configuration and run forward kinematics."""
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

    def _forward_transforms(self, q: np.ndarray) -> list[np.ndarray]:
        """Return transforms T_{0,0}, T_{0,1}, ..., T_{0,6} for the current q.

        T_{0,i} is the pose of frame i after applying joints 1..i.
        Frame 0 is the base frame, frame 6 is the end-effector frame.
        """
        T = homogeneous_transform(np.eye(3), self.base_offset)
        transforms = [T.copy()]

        # Joint 1: rotate Z, translate up by link_lengths[0].
        T = T @ homogeneous_transform(rotz(q[0]), [0.0, 0.0, 0.0])
        T = T @ homogeneous_transform(np.eye(3), [0.0, 0.0, self.link_lengths[0]])
        transforms.append(T.copy())

        # Joint 2: rotate Y, translate along X.
        T = T @ homogeneous_transform(roty(q[1]), [0.0, 0.0, 0.0])
        T = T @ homogeneous_transform(np.eye(3), [self.link_lengths[1], 0.0, 0.0])
        transforms.append(T.copy())

        # Joint 3: rotate Y, translate along X.
        T = T @ homogeneous_transform(roty(q[2]), [0.0, 0.0, 0.0])
        T = T @ homogeneous_transform(np.eye(3), [self.link_lengths[2], 0.0, 0.0])
        transforms.append(T.copy())

        # Joint 4: rotate X, translate up.
        T = T @ homogeneous_transform(rotx(q[3]), [0.0, 0.0, 0.0])
        T = T @ homogeneous_transform(np.eye(3), [0.0, 0.0, self.link_lengths[3]])
        transforms.append(T.copy())

        # Joint 5: rotate Y, translate up.
        T = T @ homogeneous_transform(roty(q[4]), [0.0, 0.0, 0.0])
        T = T @ homogeneous_transform(np.eye(3), [0.0, 0.0, self.link_lengths[4]])
        transforms.append(T.copy())

        # Joint 6: rotate Z, translate to ee.
        T = T @ homogeneous_transform(rotz(q[5]), [0.0, 0.0, 0.0])
        T = T @ homogeneous_transform(np.eye(3), self.ee_offset)
        transforms.append(T.copy())

        return transforms

    def jac_numeric(self, q: np.ndarray) -> np.ndarray:
        """Return the 6x6 geometric Jacobian using MuJoCo's site Jacobian.

        Rows 0-2: linear velocity of the end-effector site (jacp).
        Rows 3-5: angular velocity of the end-effector site frame (jacr).
        """
        self.set_q(q)
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_id)
        return np.vstack([jacp, jacr])

    def jac_analytic(self, q: np.ndarray) -> np.ndarray:
        """Return the 6x6 geometric Jacobian from the transform chain.

        For each joint i, the column is:
            [omega_i(q) x (p_ee(q) - p_i(q)); omega_i(q)]
        where omega_i(q) is the joint axis in the current space frame and
        p_i(q) is the current position of joint i. This matches MuJoCo's
        site Jacobian convention (linear velocity of the EE site origin).
        """
        transforms = self._forward_transforms(q)
        p_ee = transforms[-1][:3, 3]
        J = np.zeros((6, 6))
        for i in range(6):
            T_i = transforms[i]
            omega = T_i[:3, :3] @ self.joint_axes[i]
            p_i = T_i[:3, 3]
            v = np.cross(omega, p_ee - p_i)
            J[:, i] = np.concatenate([v, omega])
        return J

    def joint_positions(self, q: np.ndarray) -> list[np.ndarray]:
        """Return the 3D positions of the 6 joint frames for configuration q."""
        transforms = self._forward_transforms(q)
        return [T[:3, 3] for T in transforms[:-1]]

    def end_effector_position(self, q: np.ndarray) -> np.ndarray:
        """Current end-effector position."""
        self.set_q(q)
        return self.data.site_xpos[self.ee_id].copy()

    def twist(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """End-effector twist V = [v; ω] = J(q) q̇."""
        return self.jac_numeric(q) @ qdot

    def inverse_twist(
        self,
        q: np.ndarray,
        V_desired: np.ndarray,
        method: str = "damped_pinv",
        damping: float = 0.01,
    ) -> np.ndarray:
        """Solve q̇ such that J(q) q̇ ≈ V_desired.

        method: 'pinv' for pure pseudoinverse, 'damped_pinv' for regularized.
        """
        J = self.jac_numeric(q)
        if method == "pinv":
            J_pinv = np.linalg.pinv(J)
        elif method == "damped_pinv":
            JtJ = J.T @ J
            reg = JtJ + damping**2 * np.eye(J.shape[1])
            J_pinv = np.linalg.solve(reg, J.T)
        else:
            raise ValueError(f"Unknown method '{method}'")
        return J_pinv @ V_desired

    def null_space_projector(self, q: np.ndarray) -> np.ndarray:
        """Return the null-space projector (I - J⁺J)."""
        J = self.jac_numeric(q)
        J_pinv = np.linalg.pinv(J)
        return np.eye(J.shape[1]) - J_pinv @ J

    def joint_torques_from_force(self, q: np.ndarray, F: np.ndarray) -> np.ndarray:
        """Static-force duality: τ = J(q)ᵀ F.

        F is a 6-vector [force; moment] expressed in the same frame as the Jacobian.
        """
        J = self.jac_numeric(q)
        return J.T @ F


def finite_difference_jacobian(
    jac: ArmJacobian, q: np.ndarray, h: float = 1e-6
) -> np.ndarray:
    """Numerically differentiate the end-effector pose to approximate J(q)."""
    J_fd = np.zeros((6, 6))
    f0 = jac.end_effector_position(q)
    # For angular velocity, finite-difference the rotation matrix is more involved,
    # so we compare the linear part against jacp and use small-angle approximation
    # for the angular part.
    R0 = jac._forward_transforms(q)[-1][:3, :3]
    for i in range(6):
        q_plus = q.copy()
        q_plus[i] += h
        f_plus = jac.end_effector_position(q_plus)
        J_fd[:3, i] = (f_plus - f0) / h

        R_plus = jac._forward_transforms(q_plus)[-1][:3, :3]
        dR = R_plus @ R0.T
        # Small-angle axis: axis*angle ≈ (dR[2,1]-dR[1,2], dR[0,2]-dR[2,0], dR[1,0]-dR[0,1]) / 2
        omega = np.array([
            dR[2, 1] - dR[1, 2],
            dR[0, 2] - dR[2, 0],
            dR[1, 0] - dR[0, 1],
        ]) / 2.0
        J_fd[3:, i] = omega / h
    return J_fd
