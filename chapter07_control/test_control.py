"""Tests for Chapter 7 control implementation."""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

# Add sibling chapter directories so we can reuse ArmDynamics and ArmJacobian.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "chapter06_dynamics"))
sys.path.insert(0, str(ROOT / "chapter04_velocity_kinematics"))
sys.path.insert(0, str(ROOT / "chapter07_control"))

from control import (
    ComputedTorqueController,
    GravityCompensationController,
    JointSpacePIDController,
    OperationalSpaceController,
    TaskSpaceController,
    UncertaintyAwareControlWrapper,
)
from dynamics import ArmDynamics
from jacobian import ArmJacobian
from real_hardware import MockRealArm
from utils import rotation_matrix


@pytest.fixture
def arm():
    """Provide an ArmDynamics instance with gravity and generous torque limits.

    The default actuator ctrlranges in ``simple_6dof_arm.xml`` are too small for
    the controllers to demonstrate linearization against gravity, so we widen
    them in the test fixture only.
    """
    dyn = ArmDynamics()
    dyn.model.opt.gravity[:] = np.array([0.0, 0.0, -9.81])
    dyn.model.actuator_ctrlrange[:, 0] = -200.0
    dyn.model.actuator_ctrlrange[:, 1] = 200.0
    return dyn


@pytest.fixture
def jac():
    """Provide an ArmJacobian instance."""
    return ArmJacobian()


def _inertia_scaled_gains(arm: ArmDynamics, omega: float) -> tuple[np.ndarray, np.ndarray]:
    """Return joint-space PD gains scaled by the home-configuration inertia."""
    M_diag = np.diag(arm.mass_matrix(np.zeros(arm.model.nq)))
    Kp = omega**2 * M_diag
    Kd = 2.0 * np.sqrt(Kp * M_diag)
    return Kp, Kd


def test_gravity_compensation_holds_static_pose(arm):
    """Gravity-compensation torque should equal the static gravity term."""
    q = np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1])
    ctrl = GravityCompensationController(arm)
    tau = ctrl.compute(q)
    g = arm.gravity_term(q)
    np.testing.assert_allclose(tau, g, atol=1e-10)


def test_pid_set_point_converges(arm):
    """A PID controller with gravity comp should drive the arm back to q_des."""
    q_des = np.zeros(arm.model.nq)
    q0 = np.array([0.2, -0.15, 0.15, 0.0, -0.05, 0.05])
    q = q0.copy()
    qdot = np.zeros(arm.model.nq)

    Kp, Kd = _inertia_scaled_gains(arm, omega=8.0)
    ctrl = JointSpacePIDController(arm, Kp=Kp, Kd=Kd, gravity_comp=True)

    dt = 0.01
    for _ in range(1500):
        tau = ctrl.compute(q, qdot, q_des=q_des, dt=dt)
        arm.set_state(q, qdot)
        arm.data.ctrl[:] = tau
        mujoco.mj_step(arm.model, arm.data)
        q = arm.data.qpos.copy()
        qdot = arm.data.qvel.copy()

    final_err = np.linalg.norm(q - q_des)
    assert final_err < 1e-2, f"final error {final_err}"


def test_computed_torque_tracks_sine(arm):
    """Computed torque should track a slow sinusoidal joint trajectory."""
    dt = 0.01
    n_steps = 500
    A = 0.05
    omega = 1.0

    q = np.zeros(arm.model.nq)
    qdot = np.zeros(arm.model.nq)

    Kp = np.full(arm.model.nq, 50.0)
    Kd = np.full(arm.model.nq, 14.0)
    ctrl = ComputedTorqueController(arm, Kp=Kp, Kd=Kd)

    errors = []
    for i in range(n_steps):
        t = i * dt
        q_des = A * np.sin(omega * t) * np.ones(arm.model.nq)
        qdot_des = A * omega * np.cos(omega * t) * np.ones(arm.model.nq)
        qddot_des = -A * omega**2 * np.sin(omega * t) * np.ones(arm.model.nq)

        tau = ctrl.compute(q, qdot, q_des=q_des, qdot_des=qdot_des, qddot_des=qddot_des)
        arm.set_state(q, qdot)
        arm.data.ctrl[:] = tau
        mujoco.mj_step(arm.model, arm.data)
        q = arm.data.qpos.copy()
        qdot = arm.data.qvel.copy()
        errors.append(np.linalg.norm(q - q_des))

    rmse = np.sqrt(np.mean(np.array(errors) ** 2))
    assert rmse < 0.03, f"computed-torque RMSE {rmse}"


def test_operational_space_reaches_desired_pose(arm, jac):
    """Operational-space resolved-acceleration control should reach a target pose."""
    q = np.zeros(arm.model.nq)
    qdot = np.zeros(arm.model.nq)

    jac.set_q(q)
    R_cur = jac.data.site_xmat[jac.ee_id].reshape(3, 3).copy()
    p_cur = jac.data.site_xpos[jac.ee_id].copy()

    p_des = p_cur + np.array([0.03, -0.03, 0.03])
    R_des = R_cur @ rotation_matrix(np.array([0, 1, 0]), 0.1)

    Kp = np.full(6, 50.0)
    Kd = np.full(6, 14.0)
    ctrl = OperationalSpaceController(arm, jac, Kp=Kp, Kd=Kd)

    for _ in range(600):
        tau = ctrl.compute(q, qdot, R_des=R_des, p_des=p_des)
        arm.set_state(q, qdot)
        arm.data.ctrl[:] = tau
        mujoco.mj_step(arm.model, arm.data)
        q = arm.data.qpos.copy()
        qdot = arm.data.qvel.copy()

    jac.set_q(q)
    p_final = jac.data.site_xpos[jac.ee_id].copy()
    R_final = jac.data.site_xmat[jac.ee_id].reshape(3, 3).copy()

    from utils import pose_error

    err, pos_err, rot_err = pose_error(R_final, p_final, R_des, p_des)
    assert pos_err < 0.02, f"position error {pos_err}"
    assert rot_err < 0.05, f"rotation error {rot_err}"


def test_task_space_torque_points_toward_error(arm, jac):
    """Task-space J^T PD should command a torque that pushes the end effector
    toward the position target.
    """
    q = np.zeros(arm.model.nq)
    qdot = np.zeros(arm.model.nq)
    jac.set_q(q)
    p_cur = jac.data.site_xpos[jac.ee_id].copy()
    p_des = p_cur + np.array([0.02, 0.0, 0.0])

    ctrl = TaskSpaceController(arm, jac, Kp=np.full(6, 10.0), Kd=np.full(6, 2.0), gravity_comp=False)
    tau = ctrl.compute(q, qdot, R_des=None, p_des=p_des)
    J = jac.jac_numeric(q)
    pos_err = p_des - p_cur
    # The resulting end-effector force should have a positive projection onto
    # the position error direction.
    ee_force = J @ tau
    assert np.dot(ee_force[:3], pos_err) > 0.0


def test_torque_saturation_respected(arm):
    """Controller output should never exceed supplied torque limits."""
    q = np.zeros(arm.model.nq)
    qdot = np.zeros(arm.model.nq)
    q_des = q + 0.5  # large error to drive big torques

    tau_max = np.full(arm.model.nq, 5.0)
    ctrl = JointSpacePIDController(arm, Kp=np.full(arm.model.nq, 200.0), tau_max=tau_max)
    tau = ctrl.compute(q, qdot, q_des=q_des, dt=0.01)
    assert np.all(np.abs(tau) <= tau_max + 1e-10)


def test_integral_anti_windup(arm):
    """With a large persistent error and small torque limit, integrator should not
    overshoot wildly once the error is removed.
    """
    ctrl = JointSpacePIDController(
        arm,
        Kp=np.full(arm.model.nq, 10.0),
        Ki=np.full(arm.model.nq, 20.0),
        Kd=np.full(arm.model.nq, 2.0),
        tau_max=np.full(arm.model.nq, 2.0),
    )
    q = np.zeros(arm.model.nq)
    qdot = np.zeros(arm.model.nq)

    # Saturate for many steps.
    for _ in range(200):
        tau = ctrl.compute(q, qdot, q_des=np.full(arm.model.nq, 1.0), dt=0.01)

    # Now remove error; controller should not command enormous overshoot.
    tau_after = ctrl.compute(q, qdot, q_des=np.zeros(arm.model.nq), dt=0.01)
    assert np.all(np.abs(tau_after) <= 2.0 + 1e-10)
    # Integrator should have been frozen while saturated.
    assert np.all(ctrl._integral == 0.0)


def test_uncertainty_wrapper_clamps_on_model_mismatch(arm):
    """Wrapper should enter conservative mode when supplied actual acceleration
    diverges from the nominal prediction.
    """
    base = JointSpacePIDController(
        arm,
        Kp=np.full(arm.model.nq, 100.0),
        Kd=np.full(arm.model.nq, 10.0),
        gravity_comp=False,
    )
    wrapper = UncertaintyAwareControlWrapper(
        base, arm, residual_threshold=0.5, clamp_ratio=0.5
    )

    q = np.zeros(arm.model.nq)
    qdot = np.zeros(arm.model.nq)
    q_des = np.full(arm.model.nq, 0.3)

    # First call: supply matching actual acceleration -> no clamp.
    tau_base = base.compute(q, qdot, q_des=q_des)
    tau_nominal = wrapper.compute(
        q, qdot, q_des=q_des, qddot_actual=arm.forward_dynamics(q, qdot, tau_base)
    )
    assert not wrapper.conservative_mode

    # Second call: supply a wildly different actual acceleration -> clamp.
    for _ in range(3):
        tau_conservative = wrapper.compute(q, qdot, q_des=q_des, qddot_actual=np.ones(arm.model.nq) * 20.0)
    assert wrapper.conservative_mode
    np.testing.assert_allclose(tau_conservative, tau_base * 0.5, atol=1e-6)


def test_mock_real_arm_roundtrip():
    """MockRealArm should accept torques and return consistent state after stepping."""
    arm = MockRealArm()
    tau = np.full(arm.nq(), 1.0)
    arm.send_torques(tau, dt=0.01)
    state = arm.get_state()
    assert state.q.shape == (arm.nq(),)
    assert state.qdot.shape == (arm.nq(),)
    # Sending torques should advance time.
    assert arm.data.time > 0.0
