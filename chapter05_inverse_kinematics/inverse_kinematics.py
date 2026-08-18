"""
Chapter 5 — Practical: Inverse Kinematics.

Demonstrates:
  * Numeric IK: q* = argmin || T(q) - T_desired || using Jacobian pseudoinverse.
  * Analytic IK for the planar 2R sub-problem of the 6-DOF arm.
  * Redundancy resolution via null-space optimization (joint-limit avoidance).
  * Reuse of the Chapter 4 geometric Jacobian.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


class InverseKinematics:
    """Numeric + analytic IK engine for the simple_6dof_arm.xml robot.

    This implementation mirrors the Chapter 4 Jacobian conventions:
      * MuJoCo's site Jacobian gives the linear velocity of the EE site origin
        and the angular velocity of the EE frame.
      * We integrate an end-effector pose error in SE(3) using a damped
        pseudoinverse of J(q).
    """

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
        self.ee_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "ee")
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 7)
        ]
        self.qpos_addrs = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]

    def set_q(self, q: np.ndarray) -> None:
        """Set configuration and run forward kinematics."""
        self.data.qpos[:] = q
        mujoco.mj_forward(self.model, self.data)

    def _current_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current (R_ee, p_ee) from MuJoCo."""
        site_xpos = self.data.site_xpos[self.ee_id]
        site_xmat = self.data.site_xmat[self.ee_id].reshape(3, 3)
        return site_xmat.copy(), site_xpos.copy()

    def _jacobian(self) -> np.ndarray:
        """Return 6x6 geometric Jacobian for the EE site."""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_id)
        return np.vstack([jacp, jacr])

    def _pose_error(
        self, R_desired: np.ndarray, p_desired: np.ndarray
    ) -> tuple[np.ndarray, float, float]:
        """Return 6D pose error vector plus position/rotation norms."""
        R_cur, p_cur = self._current_pose()
        pos_err = p_desired - p_cur
        # Orientation error: axis-angle of R_desired @ R_cur.T
        dR = R_desired @ R_cur.T
        trace = float(np.trace(dR))
        trace = np.clip(trace, -1.0, 3.0)
        angle = np.arccos((trace - 1.0) / 2.0)
        if angle < 1e-6:
            rot_err = np.zeros(3)
        else:
            s = np.sin(angle)
            rot_err = (angle / (2.0 * s)) * np.array([
                dR[2, 1] - dR[1, 2],
                dR[0, 2] - dR[2, 0],
                dR[1, 0] - dR[0, 1],
            ])
        err = np.concatenate([pos_err, rot_err])
        return err, float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))

    def ik_numeric(
        self,
        q0: np.ndarray,
        R_target: np.ndarray,
        p_target: np.ndarray,
        method: str = "damped_pinv",
        damping: float = 0.05,
        max_iters: int = 200,
        tol: float = 1e-5,
        step_scale: float = 0.5,
        position_only: bool = False,
        secondary_objective: callable | None = None,
        secondary_gain: float = 0.1,
    ) -> tuple[np.ndarray, dict]:
        """Solve IK numerically using the Jacobian pseudoinverse.

        Parameters
        ----------
        position_only:
            If True, only the 3D position error is minimized.  This creates a
            genuine null-space for a 6-DOF arm and lets redundancy resolution
            move joints while keeping the EE position fixed.
        secondary_objective:
            Function ``h(q) -> float`` to be maximized via null-space motion
            (e.g., joint-limit centering).
        secondary_gain:
            Small gain applied to the projected secondary gradient.  Using a
            separate gain keeps the primary task from being perturbed.

        Returns
        -------
        q, info
            ``info`` contains iterations, converged flag, final position error,
            final rotation error, and the final configuration.
        """
        q = np.asarray(q0, dtype=float).copy()
        for iteration in range(max_iters):
            self.set_q(q)
            err, pos_err, rot_err = self._pose_error(R_target, p_target)
            if position_only:
                err = err[:3]
                rot_err = 0.0

            if pos_err < tol and rot_err < tol:
                break

            J = self._jacobian()
            if position_only:
                J = J[:3, :]

            if method == "pinv":
                J_pinv = np.linalg.pinv(J)
            elif method == "damped_pinv":
                JtJ = J.T @ J
                reg = JtJ + damping**2 * np.eye(J.shape[1])
                J_pinv = np.linalg.solve(reg, J.T)
            else:
                raise ValueError(f"Unknown method '{method}'")

            qdot = step_scale * (J_pinv @ err)

            # Null-space redundancy resolution.  The projector is built from the
            # true Moore-Penrose pseudoinverse so that it is an actual projector
            # even when the primary update uses damping for robustness.
            if secondary_objective is not None:
                h = secondary_objective(q)
                grad_h = np.zeros(len(q))
                h_scale = 1e-3
                for i in range(len(q)):
                    q_perturb = q.copy()
                    q_perturb[i] += h_scale
                    grad_h[i] = (secondary_objective(q_perturb) - h) / h_scale
                J_pinv_true = np.linalg.pinv(J)
                N = np.eye(len(q)) - J_pinv_true @ J
                qdot = qdot + secondary_gain * (N @ grad_h)

            q = q + qdot
            # Respect joint limits.
            for i, jid in enumerate(self.joint_ids):
                lo = self.model.jnt_range[jid, 0]
                hi = self.model.jnt_range[jid, 1]
                q[i] = float(np.clip(q[i], lo, hi))

        self.set_q(q)
        final_err, pos_err, rot_err = self._pose_error(R_target, p_target)
        info = {
            "iterations": iteration + 1,
            "converged": bool(pos_err < tol and rot_err < tol),
            "position_error": pos_err,
            "rotation_error": rot_err,
            "final_q": q.copy(),
        }
        return q, info

    def ik_analytic_2r(
        self, target_xy: tuple[float, float], elbow_up: bool = True
    ) -> tuple[float, float]:
        """Analytic IK for the first two revolute joints treated as a 2R arm.

        Joint 1 rotates about Z, joint 2 about Y.  Their combined effect moves
        the end of link2 in the X-Z plane (strictly, after rotating the world
        by q1, link2 extends along X in the rotated frame).  Here we solve for
        q1 (waist) and q2 (shoulder) to place the point (L1 + L2 cos(q2),
        L2 sin(q2)) at the target (x, z) in the world X-Z plane.

        Link lengths are taken from the MuJoCo model: link1 is the capsule from
        z=0.08 to z=0.58 (length 0.5), link2 is from there along X with length
        0.5.
        """
        x, z = target_xy
        L1 = 0.5
        L2 = 0.5
        r2 = x * x + z * z
        # Elbow angle via law of cosines.
        cos_q2 = (r2 - L1 * L1 - L2 * L2) / (2.0 * L1 * L2)
        cos_q2 = float(np.clip(cos_q2, -1.0, 1.0))
        q2 = np.arccos(cos_q2)
        if not elbow_up:
            q2 = -q2
        # q1 = atan2(z, x) - atan2(L2 sin(q2), L1 + L2 cos(q2)).
        alpha = np.arctan2(z, x)
        beta = np.arctan2(L2 * np.sin(q2), L1 + L2 * np.cos(q2))
        q1 = alpha - beta
        return float(q1), float(q2)

    def current_end_effector_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return current (R, p) of the end-effector site."""
        self.set_q(self.data.qpos.copy())
        return self._current_pose()


def joint_limit_centering_objective(ik: InverseKinematics) -> callable:
    """Return a secondary objective that rewards staying near joint centers."""
    centers = np.zeros(6)
    widths = np.zeros(6)
    for i, jid in enumerate(ik.joint_ids):
        lo = ik.model.jnt_range[jid, 0]
        hi = ik.model.jnt_range[jid, 1]
        centers[i] = (lo + hi) / 2.0
        widths[i] = (hi - lo) / 2.0

    def h(q: np.ndarray) -> float:
        # Negative squared normalized distance from center; maximizing this
        # pushes joints toward the middle of their ranges.
        return -float(np.sum(((q - centers) / widths) ** 2))

    return h
