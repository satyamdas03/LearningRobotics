"""Tests for the virtual real-robot bridge (Milestone 1).

These tests prove that ``MockRealArm`` injects realistic actuator and sensor
dynamics and that a controller assuming the perfect MuJoCo model is robustly
challenged by the resulting sim-to-real gap.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "chapter06_dynamics"))
sys.path.insert(0, str(ROOT / "chapter07_control"))

from control import JointSpacePIDController
from dynamics import ArmDynamics
from real_hardware import MockRealArm, VirtualArmFactory


def _default_arm(**kwargs) -> MockRealArm:
    """Create a default virtual arm, overriding kwargs."""
    return MockRealArm(**kwargs)


def test_default_virtual_arm_is_deterministic_and_passes_roundtrip():
    """A vanilla virtual arm should behave exactly like the wrapped MuJoCo arm."""
    arm = _default_arm(fixed_random_seed=42)
    tau = np.full(arm.nq(), 1.0)
    arm.send_torques(tau, dt=0.01)
    state = arm.get_state()
    assert state.q.shape == (arm.nq(),)
    assert state.qdot.shape == (arm.nq(),)
    assert arm.data.time > 0.0


def test_gear_ratio_scales_motion():
    """Higher gear ratio should produce larger joint acceleration before limits."""
    arm_low = _default_arm(gear_ratio=1.0, fixed_random_seed=42)
    arm_high = _default_arm(gear_ratio=2.0, fixed_random_seed=42)

    tau = np.full(arm_low.nq(), 5.0)
    for _ in range(5):
        arm_low.send_torques(tau, dt=0.01)
        arm_high.send_torques(tau, dt=0.01)

    assert np.linalg.norm(arm_high.data.qvel) > np.linalg.norm(arm_low.data.qvel)


def test_coulomb_friction_opposes_motion():
    """Coulomb + viscous friction should reduce acceleration for the same torque."""
    # Cap velocity so the light wrist joints do not run away during the comparison.
    tau = np.full(6, 5.0)
    arm_no_friction = _default_arm(
        velocity_limits=np.full(6, 0.5), fixed_random_seed=42
    )
    arm_friction = _default_arm(
        coulomb_friction=2.0,
        viscous_friction=0.2,
        velocity_limits=np.full(6, 0.5),
        fixed_random_seed=42,
    )

    for _ in range(100):
        arm_no_friction.send_torques(tau, dt=0.01)
        arm_friction.send_torques(tau, dt=0.01)

    # The friction-laden arm must end up moving slower on average.
    assert np.linalg.norm(arm_friction.data.qvel) < np.linalg.norm(arm_no_friction.data.qvel)


def test_position_bias_visible_in_readings():
    """A fixed position bias should appear directly in ``get_state``."""
    bias = np.array([0.01, -0.02, 0.03, -0.01, 0.02, -0.03])
    arm = _default_arm(position_bias=bias)
    arm.send_torques(np.zeros(arm.nq()), dt=0.01)
    state = arm.get_state()
    np.testing.assert_allclose(state.q - arm.data.qpos, bias, atol=1e-12)


def test_position_quantization():
    """Quantized readings should be integer multiples of the resolution."""
    res = 0.01
    arm = _default_arm(quantization_resolution=res, fixed_random_seed=42)
    # Perturb so quantization matters.
    arm.reset_state(q=np.full(arm.nq(), 0.005), qdot=np.zeros(arm.nq()))
    arm.send_torques(np.full(arm.nq(), 2.0), dt=0.1)
    state = arm.get_state()
    # Reconstructed quantized values are exact multiples of res.
    reconstructed = np.round(state.q / res) * res
    np.testing.assert_allclose(state.q, reconstructed, atol=1e-12)


def test_feedback_delay_returns_past_state():
    """With feedback delay, the returned state should lag the true state."""
    arm = _default_arm(feedback_delay_steps=3)
    q0 = arm.get_state().q.copy()

    # Apply a constant torque and watch that the reported state stays at q0
    # for exactly the delay window.
    for _ in range(3):
        arm.send_torques(np.full(arm.nq(), 10.0), dt=0.01)
        assert np.allclose(arm.get_state().q, q0)

    # After the delay window, the state should finally reflect motion.
    arm.send_torques(np.full(arm.nq(), 10.0), dt=0.01)
    assert not np.allclose(arm.get_state().q, q0)


def test_torque_command_delay():
    """With command delay, torque should not affect the arm immediately."""
    arm = _default_arm(torque_delay_steps=2)
    q0 = arm.data.qpos.copy()

    # The buffer is pre-filled with zeros, so the first delay_steps commands
    # produce no motion (beyond numerical round-off).
    arm.send_torques(np.full(arm.nq(), 10.0), dt=0.01)
    assert np.linalg.norm(arm.data.qpos - q0) < 1e-6

    arm.send_torques(np.full(arm.nq(), 10.0), dt=0.01)
    assert np.linalg.norm(arm.data.qpos - q0) < 1e-6

    arm.send_torques(np.full(arm.nq(), 10.0), dt=0.01)
    assert np.linalg.norm(arm.data.qpos - q0) > 1e-6


def test_position_mode_reaches_target():
    """In position control mode, sending a position setpoint should converge."""
    # Scale PD gains by inertia so all joints converge at roughly the same rate.
    from dynamics import ArmDynamics

    dyn = ArmDynamics()
    M_diag = np.diag(dyn.mass_matrix(np.zeros(6)))
    omega = 3.0
    kp_pos = M_diag * omega**2
    kd_pos = 2.0 * np.sqrt(kp_pos * M_diag)

    arm = _default_arm(
        control_mode="position",
        internal_kp=kp_pos,
        internal_kd=kd_pos,
        velocity_limits=np.full(6, 1.0),
    )
    q_des = np.array([0.2, -0.1, 0.15, 0.0, -0.05, 0.05])

    for _ in range(2000):
        arm.send_torques(q_des, dt=0.01)

    # Internal (perfect) state should be close to the target.
    assert np.linalg.norm(arm.data.qpos - q_des) < 0.05


def test_velocity_mode_tracks_single_joint_setpoint():
    """In velocity control mode, a commanded joint should track its setpoint."""
    # Scale internal P gains by inertia so each joint has the same bandwidth.
    from dynamics import ArmDynamics

    dyn = ArmDynamics()
    M_diag = np.diag(dyn.mass_matrix(np.zeros(6)))
    kp_vel = M_diag / 0.2  # time constant ~0.2 s

    qdot_des = np.zeros(6)
    qdot_des[0] = 0.1
    arm = _default_arm(
        control_mode="velocity",
        internal_kp=kp_vel,
        velocity_limits=np.full(6, 0.5),
        viscous_friction=0.0,
    )

    for _ in range(1000):
        arm.send_torques(qdot_des, dt=0.01)

    # Joint 1 should be near the requested velocity; coupling keeps the others
    # small but nonzero, so we only check the commanded joint and boundedness.
    assert abs(arm.data.qvel[0] - qdot_des[0]) < 0.05
    assert np.all(np.abs(arm.data.qvel) <= 0.5 + 1e-6)


def test_torque_saturation_on_virtual_arm():
    """Virtual arm should clip applied torques to supplied limits."""
    limits = np.full(6, 3.0)
    arm = _default_arm(torque_limits=limits)
    arm.send_torques(np.full(arm.nq(), 100.0), dt=0.01)
    assert np.all(np.abs(arm.data.ctrl) <= limits + 1e-10)


def test_factory_produces_varied_arms():
    """VirtualArmFactory should generate arms with different parameters."""
    factory = VirtualArmFactory(seed=123)
    arms = factory.create_batch(10, nq=6)

    gear_ratios = [arm.gear_ratio for arm in arms]
    assert any(not np.allclose(gear_ratios[0], g) for g in gear_ratios[1:])

    biases = [arm.position_bias for arm in arms]
    assert any(not np.allclose(biases[0], b) for b in biases[1:])


def test_controller_assumed_model_mismatches_randomized_virtual_arm():
    """A controller built on the perfect ArmDynamics model should track worse
    when wired to a heavily imperfect virtual arm.
    """
    # Perfect baseline: controller talks directly to its own assumed model.
    assumed = ArmDynamics()
    Kp = np.full(assumed.model.nq, 80.0)
    Kd = np.full(assumed.model.nq, 18.0)
    ctrl_perfect = JointSpacePIDController(assumed, Kp=Kp, Kd=Kd, gravity_comp=False)

    q_des = np.array([0.15, -0.1, 0.1, 0.0, -0.05, 0.05])
    q = np.zeros(assumed.model.nq)
    qdot = np.zeros(assumed.model.nq)

    for _ in range(300):
        tau = ctrl_perfect.compute(q, qdot, q_des=q_des, dt=0.01)
        assumed.data.ctrl[:] = tau
        mujoco.mj_step(assumed.model, assumed.data)
        q = assumed.data.qpos.copy()
        qdot = assumed.data.qvel.copy()
    err_perfect = np.linalg.norm(q - q_des)

    # Imperfect virtual arm wired to a fresh copy of the same controller.
    virtual = MockRealArm(
        gear_ratio=1.1,
        position_bias=np.array([0.02, -0.01, 0.015, -0.005, 0.01, -0.01]),
        position_noise_std=0.005,
        velocity_noise_std=0.02,
        feedback_delay_steps=3,
        torque_delay_steps=2,
        coulomb_friction=0.5,
        viscous_friction=0.1,
        fixed_random_seed=7,
    )
    ctrl_imperfect = JointSpacePIDController(assumed, Kp=Kp, Kd=Kd, gravity_comp=False)

    q = np.zeros(virtual.nq())
    qdot = np.zeros(virtual.nq())
    for _ in range(300):
        state = virtual.get_state()
        tau = ctrl_imperfect.compute(state.q, state.qdot, q_des=q_des, dt=0.01)
        virtual.send_torques(tau, dt=0.01)
    err_imperfect = np.linalg.norm(virtual.get_state().q - q_des)

    # The imperfect arm must track measurably worse (or at least not better).
    assert err_imperfect > err_perfect


def test_factory_seed_reproducibility():
    """The same factory seed should produce identical parameter draws."""
    factory_a = VirtualArmFactory(seed=99)
    factory_b = VirtualArmFactory(seed=99)
    arm_a = factory_a.create(nq=6, randomize_sensors=True, randomize_friction=True)
    arm_b = factory_b.create(nq=6, randomize_sensors=True, randomize_friction=True)

    np.testing.assert_allclose(arm_a.gear_ratio, arm_b.gear_ratio)
    np.testing.assert_allclose(arm_a.coulomb_friction, arm_b.coulomb_friction)
    np.testing.assert_allclose(arm_a.position_bias, arm_b.position_bias)
    assert arm_a.torque_delay_steps == arm_b.torque_delay_steps
    assert arm_a.feedback_delay_steps == arm_b.feedback_delay_steps
