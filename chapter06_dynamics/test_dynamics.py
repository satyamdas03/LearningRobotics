"""Tests for Chapter 6 dynamics implementation."""
from __future__ import annotations

import numpy as np
import pytest

from dynamics import ArmDynamics


@pytest.fixture
def dyn():
    return ArmDynamics()


def test_mass_matrix_symmetric_positive_definite(dyn):
    q = np.zeros(dyn.model.nq)
    M = dyn.mass_matrix(q)
    assert M.shape == (dyn.model.nq, dyn.model.nq)
    # Symmetry.
    np.testing.assert_allclose(M, M.T, atol=1e-12)
    # Positive definite: all eigenvalues positive.
    eigvals = np.linalg.eigvalsh(M)
    assert np.all(eigvals > 0), eigvals


def test_mass_matrix_matches_mujoco_inverse_dynamics(dyn):
    """If we command qddot = e_i and read the resulting tau, the i-th column of M
    should match the tau minus the bias force. This cross-checks M against
    MuJoCo's mj_inverse.
    """
    q = np.zeros(dyn.model.nq)
    qdot = np.zeros(dyn.model.nq)
    bias = dyn.coriolis_gravity(q, qdot)

    M = dyn.mass_matrix(q)
    for i in range(dyn.model.nq):
        qddot = np.zeros(dyn.model.nq)
        qddot[i] = 1.0
        tau = dyn.inverse_dynamics(q, qdot, qddot)
        np.testing.assert_allclose(tau - bias, M[:, i], atol=1e-3)


def test_inverse_dynamics_recover_static_torque(dyn):
    """Gravity compensation: with qdot=0, qddot=0, tau should equal g(q)."""
    q = np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1])
    tau = dyn.inverse_dynamics(q, np.zeros(dyn.model.nq), np.zeros(dyn.model.nq))
    g = dyn.gravity_term(q)
    np.testing.assert_allclose(tau, g, atol=1e-10)


def test_forward_dynamics_zero_torque_falls(dyn):
    """Under zero torque the arm should accelerate downward (gravity)."""
    q = np.zeros(dyn.model.nq)
    qdot = np.zeros(dyn.model.nq)
    tau = np.zeros(dyn.model.nq)
    qddot = dyn.forward_dynamics(q, qdot, tau)
    # Some joints should accelerate due to gravity (not exactly zero vector).
    assert np.linalg.norm(qddot) > 1e-4


def test_euler_step_matches_mujoco_single_step(dyn):
    """Our simple Euler integration should stay close to MuJoCo for one tiny step."""
    q = np.zeros(dyn.model.nq)
    qdot = np.zeros(dyn.model.nq)
    tau = np.zeros(dyn.model.nq)
    dt = float(dyn.model.opt.timestep)

    q_new_ours, qdot_new_ours = dyn.step(q, qdot, tau, dt)

    # MuJoCo's own step.
    dyn.set_state(q, qdot)
    dyn.data.ctrl[:] = 0.0  # no actuators, so torques are whatever we apply externally
    # mj_step with no actuation uses qfrc_applied; set it to tau.
    dyn.data.qfrc_applied[:] = tau
    mujoco = pytest.importorskip("mujoco")
    mujoco.mj_step(dyn.model, dyn.data)

    np.testing.assert_allclose(q_new_ours, dyn.data.qpos, atol=1e-3)
    np.testing.assert_allclose(qdot_new_ours, dyn.data.qvel, atol=1e-3)


def test_inverse_and_forward_dynamics_consistency(dyn):
    """Forward(q, qdot, inverse(q, qdot, qddot)) should recover qddot."""
    rng = np.random.default_rng(0)
    q = rng.uniform(-0.5, 0.5, size=dyn.model.nq)
    qdot = rng.uniform(-0.5, 0.5, size=dyn.model.nq)
    qddot = rng.uniform(-1.0, 1.0, size=dyn.model.nq)

    tau = dyn.inverse_dynamics(q, qdot, qddot)
    qddot_reconstructed = dyn.forward_dynamics(q, qdot, tau)
    np.testing.assert_allclose(qddot_reconstructed, qddot, atol=1e-3)
