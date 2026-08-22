"""
Chapter 7 — Practical: Control.

Controllers for the simple_6dof_arm.xml robot.

Controllers use the dynamics abstraction from Chapter 6 (mass matrix, bias forces)
and the Jacobian abstraction from Chapter 4 (geometric Jacobian).  They return
joint torques that can be sent to either a MuJoCo-backed `MockRealArm` or a
future real `RealArm` implementation.
"""
from __future__ import annotations

import numpy as np

from utils import clip_vector, pose_error


class RobotController:
    """Base class for a joint-torque controller."""

    def compute(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        q_des: np.ndarray | None = None,
        qdot_des: np.ndarray | None = None,
        qddot_des: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Return the joint-torque command for the current state."""
        raise NotImplementedError


class GravityCompensationController(RobotController):
    """Return the torques needed to hold the arm stationary against gravity.

    tau = g(q)
    """

    def __init__(self, dynamics) -> None:
        self.dynamics = dynamics

    def compute(self, q, qdot=None, **kwargs):
        return self.dynamics.gravity_term(q)


class JointSpacePIDController(RobotController):
    """Independent-joint PID with optional gravity feedforward.

    tau = Kp*(q_des - q) + Ki*integral(err) + Kd*(qdot_des - qdot) + g(q)

    Includes per-joint torque saturation and a simple anti-windup that freezes
    the integrator while any component is saturated.
    """

    def __init__(
        self,
        dynamics,
        Kp: np.ndarray,
        Ki: np.ndarray | None = None,
        Kd: np.ndarray | None = None,
        gravity_comp: bool = True,
        tau_max: np.ndarray | float | None = None,
    ) -> None:
        self.dynamics = dynamics
        self.Kp = np.asarray(Kp, dtype=float)
        self.Ki = np.asarray(Ki, dtype=float) if Ki is not None else np.zeros_like(self.Kp)
        self.Kd = np.asarray(Kd, dtype=float) if Kd is not None else np.zeros_like(self.Kp)
        self.gravity_comp = gravity_comp
        self.tau_max = tau_max
        self._integral = np.zeros(self.Kp.shape)

    def reset(self) -> None:
        self._integral[:] = 0.0

    def compute(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        q_des: np.ndarray | None = None,
        qdot_des: np.ndarray | None = None,
        dt: float = 0.01,
        **kwargs,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        q_des = np.asarray(q_des, dtype=float) if q_des is not None else q
        qdot_des = np.asarray(qdot_des, dtype=float) if qdot_des is not None else np.zeros_like(q)

        err = q_des - q
        self._integral = self._integral + err * dt

        tau = self.Kp * err + self.Ki * self._integral + self.Kd * (qdot_des - qdot)
        if self.gravity_comp:
            tau = tau + self.dynamics.gravity_term(q)

        if self.tau_max is not None:
            tau_clipped = clip_vector(tau, self.tau_max)
            # Anti-windup: do not integrate while saturated.
            saturated = np.abs(tau) >= (np.asarray(self.tau_max) - 1e-12)
            self._integral = np.where(saturated, self._integral - err * dt, self._integral)
            tau = tau_clipped

        return tau


class ComputedTorqueController(RobotController):
    """Inverse-dynamics linearization with feedback.

    tau = M(q)*(qddot_des + Kp*e + Kd*edot) + C(q,qdot)*qdot + g(q)

    This cancels the nonlinear dynamics and leaves linear error dynamics
    governed by Kp and Kd.
    """

    def __init__(
        self,
        dynamics,
        Kp: np.ndarray,
        Kd: np.ndarray,
        tau_max: np.ndarray | float | None = None,
    ) -> None:
        self.dynamics = dynamics
        self.Kp = np.asarray(Kp, dtype=float)
        self.Kd = np.asarray(Kd, dtype=float)
        self.tau_max = tau_max

    def compute(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        q_des: np.ndarray | None = None,
        qdot_des: np.ndarray | None = None,
        qddot_des: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)
        q_des = np.asarray(q_des, dtype=float) if q_des is not None else q
        qdot_des = np.asarray(qdot_des, dtype=float) if qdot_des is not None else np.zeros_like(q)
        qddot_des = np.asarray(qddot_des, dtype=float) if qddot_des is not None else np.zeros_like(q)

        e = q_des - q
        edot = qdot_des - qdot
        qddot_fb = qddot_des + self.Kp * e + self.Kd * edot

        M = self.dynamics.mass_matrix(q)
        bias = self.dynamics.coriolis_gravity(q, qdot)
        tau = M @ qddot_fb + bias

        if self.tau_max is not None:
            tau = clip_vector(tau, self.tau_max)
        return tau


class TaskSpaceController(RobotController):
    """Task-space PD with gravity compensation via the Jacobian transpose.

    Given a desired end-effector pose (R_des, p_des), compute the task-space
    wrench [force; moment] and map it to joint torques with J^T.  This is the
    simplest operational-space controller and avoids explicit operational-space
    inertia inversion.

    tau = J(q)^T * (Kp_task * e + Kd_task * edot) + g(q)
    """

    def __init__(
        self,
        dynamics,
        jacobian,
        Kp: np.ndarray,
        Kd: np.ndarray,
        gravity_comp: bool = True,
        tau_max: np.ndarray | float | None = None,
    ) -> None:
        self.dynamics = dynamics
        self.jacobian = jacobian
        self.Kp = np.asarray(Kp, dtype=float)
        self.Kd = np.asarray(Kd, dtype=float)
        self.gravity_comp = gravity_comp
        self.tau_max = tau_max

    def compute(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        R_des: np.ndarray | None = None,
        p_des: np.ndarray | None = None,
        V_des: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)

        self.jacobian.set_q(q)
        R_cur, p_cur = self._current_pose(q)
        if R_des is None:
            R_des = R_cur
        if p_des is None:
            p_des = p_cur
        if V_des is None:
            V_des = np.zeros(6)

        err, _pos_err, _rot_err = pose_error(R_cur, p_cur, R_des, p_des)
        J = self.jacobian.jac_numeric(q)
        qdot_meas = qdot
        # Task-space velocity from current joint velocity.
        V_cur = J @ qdot_meas
        edot = V_des - V_cur

        wrench = self.Kp * err + self.Kd * edot
        tau = J.T @ wrench

        if self.gravity_comp:
            tau = tau + self.dynamics.gravity_term(q)

        if self.tau_max is not None:
            tau = clip_vector(tau, self.tau_max)
        return tau

    def _current_pose(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        self.jacobian.set_q(q)
        # Use MuJoCo site pose if available; otherwise fall back to transform chain.
        if hasattr(self.jacobian, "data") and hasattr(self.jacobian, "ee_id"):
            R = self.jacobian.data.site_xmat[self.jacobian.ee_id].reshape(3, 3).copy()
            p = self.jacobian.data.site_xpos[self.jacobian.ee_id].copy()
            return R, p
        # Fallback for plain matrix-only Jacobians.
        transforms = self.jacobian._forward_transforms(q)
        T = transforms[-1]
        return T[:3, :3].copy(), T[:3, 3].copy()


class OperationalSpaceController(RobotController):
    """Resolved-acceleration operational-space control.

    Maps a task-space PD command to joint accelerations via the Moore-Penrose
    pseudo-inverse of the Jacobian, then uses the Chapter 6 inverse-dynamics
    model to compute the corresponding joint torques.  A small null-space
    velocity damping term keeps redundant joints from drifting.

        xddot_des = Kp * e + Kd * edot
        qddot_des = J^+ * xddot_des + (I - J^+ J) * (-k_null * qdot)
        tau       = inverse_dynamics(q, qdot, qddot_des)
    """

    def __init__(
        self,
        dynamics,
        jacobian,
        Kp: np.ndarray,
        Kd: np.ndarray,
        k_null: float = 0.1,
        tau_max: np.ndarray | float | None = None,
    ) -> None:
        self.dynamics = dynamics
        self.jacobian = jacobian
        self.Kp = np.asarray(Kp, dtype=float)
        self.Kd = np.asarray(Kd, dtype=float)
        self.k_null = k_null
        self.tau_max = tau_max

    def compute(
        self,
        q: np.ndarray,
        qdot: np.ndarray,
        R_des: np.ndarray | None = None,
        p_des: np.ndarray | None = None,
        V_des: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        q = np.asarray(q, dtype=float)
        qdot = np.asarray(qdot, dtype=float)

        self.jacobian.set_q(q)
        R_cur, p_cur = self._current_pose()
        if R_des is None:
            R_des = R_cur
        if p_des is None:
            p_des = p_cur
        if V_des is None:
            V_des = np.zeros(6)

        err, _pos_err, _rot_err = pose_error(R_cur, p_cur, R_des, p_des)
        J = self.jacobian.jac_numeric(q)
        V_cur = J @ qdot
        edot = V_des - V_cur

        xddot_des = self.Kp * err + self.Kd * edot
        J_pinv = np.linalg.pinv(J)
        # Null-space damping: only damp the component of qdot that does not
        # contribute to the task, leaving the task-space command untouched.
        qddot_des = J_pinv @ xddot_des - self.k_null * (qdot - J_pinv @ J @ qdot)

        tau = self.dynamics.inverse_dynamics(q, qdot, qddot_des)

        if self.tau_max is not None:
            tau = clip_vector(tau, self.tau_max)
        return tau

    def _current_pose(self) -> tuple[np.ndarray, np.ndarray]:
        self.jacobian.set_q(self.jacobian.data.qpos)
        if hasattr(self.jacobian, "data") and hasattr(self.jacobian, "ee_id"):
            R = self.jacobian.data.site_xmat[self.jacobian.ee_id].reshape(3, 3).copy()
            p = self.jacobian.data.site_xpos[self.jacobian.ee_id].copy()
            return R, p
        transforms = self.jacobian._forward_transforms(self.jacobian.data.qpos)
        T = transforms[-1]
        return T[:3, :3].copy(), T[:3, 3].copy()


class UncertaintyAwareControlWrapper(RobotController):
    """Safety wrapper around any controller.

    Maintains a running estimate of the model-mismatch residual
    ``|| qddot_actual - qddot_predicted ||`` and clamps the commanded torque
    when the residual exceeds a threshold.  This is a minimal version of the
    "uncertainty-aware autonomous robot" idea from the manifesto.
    """

    def __init__(
        self,
        controller: RobotController,
        dynamics,
        residual_threshold: float = 5.0,
        clamp_ratio: float = 0.5,
        window_size: int = 5,
    ) -> None:
        self.controller = controller
        self.dynamics = dynamics
        self.residual_threshold = residual_threshold
        self.clamp_ratio = clamp_ratio
        self.window_size = window_size
        self._residuals: list[float] = []
        self._conservative_mode = False

    def compute(self, q, qdot, **kwargs):
        tau_cmd = self.controller.compute(q, qdot, **kwargs)

        # Predict the acceleration this torque would produce on the nominal model.
        qddot_pred = self.dynamics.forward_dynamics(q, qdot, tau_cmd)

        # If a previous actual acceleration is not supplied, use the prediction.
        qddot_actual = kwargs.get("qddot_actual", qddot_pred)
        residual = float(np.linalg.norm(qddot_actual - qddot_pred))
        self._residuals.append(residual)
        if len(self._residuals) > self.window_size:
            self._residuals.pop(0)

        mean_residual = float(np.mean(self._residuals)) if self._residuals else 0.0
        if mean_residual > self.residual_threshold:
            self._conservative_mode = True
        elif mean_residual < self.residual_threshold * 0.5:
            self._conservative_mode = False

        if self._conservative_mode:
            tau_cmd = tau_cmd * self.clamp_ratio

        return tau_cmd

    @property
    def conservative_mode(self) -> bool:
        return self._conservative_mode

    @property
    def latest_residual(self) -> float:
        return self._residuals[-1] if self._residuals else 0.0
