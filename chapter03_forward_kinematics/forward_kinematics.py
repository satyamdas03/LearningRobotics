"""
Chapter 3 — Practical: Forward Kinematics.

Demonstrates:
  * Product-of-Exponentials (PoE) forward kinematics for a 6-DOF serial arm
  * Geometric transform FK as a cross-check
  * Comparison against MuJoCo's built-in FK
  * Why analytical FK matters: you can predict the end-effector from joint angles
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


def skew_symmetric(w: np.ndarray) -> np.ndarray:
    """Convert a 3-vector to a 3x3 skew-symmetric matrix."""
    wx, wy, wz = w
    return np.array([[0, -wz, wy], [wz, 0, -wx], [-wy, wx, 0]])


def matrix_exponential_screw(S: np.ndarray, theta: float) -> np.ndarray:
    """Matrix exponential of a spatial twist S = [omega; v] for angle theta.

    Uses Rodrigues' formula for the rotation part and integrates the
    translation part accordingly.
    """
    omega = S[:3]
    v = S[3:]
    if np.allclose(omega, 0):
        # Pure translation.
        T = np.eye(4)
        T[:3, 3] = theta * v
        return T

    omega_hat = skew_symmetric(omega)
    omega_norm = np.linalg.norm(omega)
    I = np.eye(3)
    R = I + math.sin(omega_norm * theta) * omega_hat / omega_norm \
        + (1 - math.cos(omega_norm * theta)) * (omega_hat @ omega_hat) / (omega_norm ** 2)

    term1 = I * theta
    term2 = (1 - math.cos(omega_norm * theta)) * omega_hat / (omega_norm ** 2)
    term3 = (theta - math.sin(omega_norm * theta) / omega_norm) * (omega_hat @ omega_hat) / (omega_norm ** 3)
    p = (term1 + term2 + term3) @ v

    return homogeneous_transform(R, p)


class Arm6DOFFK:
    """Analytical forward kinematics for the simple_6dof_arm.xml robot.

    Joint layout (all revolute):
      1: waist   at (0, 0, 0.08), axis +Z
      2: shoulder at (0, 0, 0.58), axis +Y   (0.08 + 0.5)
      3: elbow    at (0.5, 0, 0.58), axis +Y
      4: wrist roll at (0.9, 0, 0.58), axis +X   (0.5 + 0.4)
      5: wrist pitch at (0.9, 0, 0.73), axis +Y   (0.58 + 0.15)
      6: wrist yaw  at (0.9, 0, 0.85), axis +Z   (0.73 + 0.12)
      ee: at (0.96, 0, 0.85)                (0.9 + 0.06)
    """

    # Joint positions in the zero-configuration (all q = 0).
    joint_positions = np.array([
        [0.0, 0.0, 0.08],   # joint 1
        [0.0, 0.0, 0.58],   # joint 2
        [0.5, 0.0, 0.58],   # joint 3
        [0.9, 0.0, 0.58],   # joint 4
        [0.9, 0.0, 0.73],   # joint 5
        [0.9, 0.0, 0.85],   # joint 6
    ])

    # Joint axes in the zero-configuration.
    joint_axes = np.array([
        [0.0, 0.0, 1.0],    # joint 1: Z
        [0.0, 1.0, 0.0],    # joint 2: Y
        [0.0, 1.0, 0.0],    # joint 3: Y
        [1.0, 0.0, 0.0],    # joint 4: X
        [0.0, 1.0, 0.0],    # joint 5: Y
        [0.0, 0.0, 1.0],    # joint 6: Z
    ])

    # Link lengths and base offset.
    base_offset = np.array([0.0, 0.0, 0.08])
    link_lengths = np.array([0.5, 0.5, 0.4, 0.15, 0.12])
    ee_offset = np.array([0.06, 0.0, 0.0])

    def __init__(self) -> None:
        # Home configuration transform T(0) = end-effector pose when q = 0.
        self.M = homogeneous_transform(np.eye(3), [0.96, 0.0, 0.85])
        # Spatial screw axes S_i = [omega_i; -omega_i x q_i].
        self.S = np.zeros((6, 6))
        for i in range(6):
            omega = self.joint_axes[i]
            q = self.joint_positions[i]
            self.S[:, i] = np.concatenate([omega, -np.cross(omega, q)])

    def poe_fk(self, q: np.ndarray) -> np.ndarray:
        """Product-of-Exponentials forward kinematics.

        T(q) = exp([S1]q1) @ exp([S2]q2) @ ... @ exp([S6]q6) @ M
        """
        T = np.eye(4)
        for i in range(6):
            T = T @ matrix_exponential_screw(self.S[:, i], q[i])
        T = T @ self.M
        return T

    def geometric_fk(self, q: np.ndarray) -> np.ndarray:
        """Geometric FK: multiply per-joint transforms from base to ee.

        This is the same math as PoE but written frame-by-frame.
        """
        # Base frame.
        T = homogeneous_transform(np.eye(3), self.base_offset)

        # Joint 1: rotate about Z, then translate up by link1 length.
        T = T @ homogeneous_transform(rotz(q[0]), [0.0, 0.0, 0.0])
        T = T @ homogeneous_transform(np.eye(3), [0.0, 0.0, self.link_lengths[0]])

        # Joint 2: rotate about Y, then translate along X by link2 length.
        T = T @ homogeneous_transform(roty(q[1]), [0.0, 0.0, 0.0])
        T = T @ homogeneous_transform(np.eye(3), [self.link_lengths[1], 0.0, 0.0])

        # Joint 3: rotate about Y, then translate along X by link3 length.
        T = T @ homogeneous_transform(roty(q[2]), [0.0, 0.0, 0.0])
        T = T @ homogeneous_transform(np.eye(3), [self.link_lengths[2], 0.0, 0.0])

        # Joint 4: rotate about X, then translate up by link4 length.
        T = T @ homogeneous_transform(rotx(q[3]), [0.0, 0.0, 0.0])
        T = T @ homogeneous_transform(np.eye(3), [0.0, 0.0, self.link_lengths[3]])

        # Joint 5: rotate about Y, then translate up by link5 length.
        T = T @ homogeneous_transform(roty(q[4]), [0.0, 0.0, 0.0])
        T = T @ homogeneous_transform(np.eye(3), [0.0, 0.0, self.link_lengths[4]])

        # Joint 6: rotate about Z, then translate to ee.
        T = T @ homogeneous_transform(rotz(q[5]), [0.0, 0.0, 0.0])
        T = T @ homogeneous_transform(np.eye(3), self.ee_offset)

        return T

    def end_effector_position(self, q: np.ndarray) -> np.ndarray:
        """Return just the (x, y, z) position of the end-effector."""
        return self.poe_fk(q)[:3, 3]


def mujoco_fk(xml_path: str, q: np.ndarray) -> np.ndarray:
    """Use MuJoCo to compute end-effector position for the 6-DOF arm."""
    with open(xml_path, "r", encoding="utf-8") as f:
        model = mujoco.MjModel.from_xml_string(f.read())
    data = mujoco.MjData(model)
    data.qpos[:] = q
    mujoco.mj_forward(model, data)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    return data.site_xpos[ee_id].copy()


def demo() -> None:
    print("=" * 70)
    print("DEMO: Forward Kinematics — analytical vs MuJoCo")
    print("=" * 70)

    xml_path = Path(__file__).parent.parent / "chapter01_foundation" / "simple_6dof_arm.xml"
    fk = Arm6DOFFK()

    configs = [
        ("all zeros", np.zeros(6)),
        ("waist 90°", np.array([math.radians(90), 0.0, 0.0, 0.0, 0.0, 0.0])),
        ("shoulder raised", np.array([0.0, math.radians(45), 0.0, 0.0, 0.0, 0.0])),
        ("elbow bent", np.array([0.0, 0.0, math.radians(45), 0.0, 0.0, 0.0])),
        ("wrist pitch", np.array([0.0, 0.0, 0.0, 0.0, math.radians(45), 0.0])),
        ("mixed", np.array([math.radians(30), math.radians(-20), math.radians(15),
                              math.radians(10), math.radians(-10), math.radians(25)])),
    ]

    print(f"{'Config':18s} | {'MuJoCo FK (x,y,z)':>30s} | {'PoE FK (x,y,z)':>30s} | {'Error':>12s}")
    print("-" * 100)
    for name, q in configs:
        mujoco_pos = mujoco_fk(str(xml_path), q)
        poe_pos = fk.end_effector_position(q)
        error = np.linalg.norm(mujoco_pos - poe_pos)
        print(f"{name:18s} | ({mujoco_pos[0]:+.4f}, {mujoco_pos[1]:+.4f}, {mujoco_pos[2]:+.4f}) | "
              f"({poe_pos[0]:+.4f}, {poe_pos[1]:+.4f}, {poe_pos[2]:+.4f}) | {error:.2e}")
    print()

    # Also verify geometric FK matches PoE FK for a random configuration.
    rng = np.random.default_rng(42)
    q_rand = rng.uniform(-math.pi / 3, math.pi / 3, size=6)
    T_poe = fk.poe_fk(q_rand)
    T_geo = fk.geometric_fk(q_rand)
    print(f"PoE vs geometric FK error for random q: {np.linalg.norm(T_poe - T_geo):.2e}")
    print()

    print("=" * 70)
    print("Chapter 3 practical complete.")
    print("Key takeaway: joint angles -> transform chain -> end-effector pose.")
    print("=" * 70)


if __name__ == "__main__":
    demo()
