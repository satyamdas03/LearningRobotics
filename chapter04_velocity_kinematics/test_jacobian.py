"""Tests for Chapter 4 velocity kinematics / Jacobian utilities."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from jacobian import ArmJacobian, finite_difference_jacobian


@pytest.fixture
def xml_path():
    return str(Path(__file__).parent.parent / "chapter01_foundation" / "simple_6dof_arm.xml")


@pytest.fixture
def jac(xml_path):
    return ArmJacobian(xml_path)


def test_analytic_matches_numeric_default(jac):
    q = np.zeros(6)
    J_num = jac.jac_numeric(q)
    J_ana = jac.jac_analytic(q)
    assert np.allclose(J_num, J_ana, atol=1e-10)


def test_analytic_matches_numeric_random_config(jac):
    rng = np.random.default_rng(0)
    q = rng.uniform(-math.pi / 3, math.pi / 3, size=6)
    J_num = jac.jac_numeric(q)
    J_ana = jac.jac_analytic(q)
    assert np.allclose(J_num, J_ana, atol=1e-10)


def test_twist_matches_finite_difference(jac):
    rng = np.random.default_rng(1)
    q = rng.uniform(-math.pi / 4, math.pi / 4, size=6)
    qdot = rng.uniform(-1.0, 1.0, size=6)

    V = jac.twist(q, qdot)

    # Finite-difference end-effector linear velocity over a small dt.
    dt = 1e-4
    p0 = jac.end_effector_position(q)
    q1 = q + qdot * dt
    p1 = jac.end_effector_position(q1)
    v_fd = (p1 - p0) / dt

    assert np.allclose(V[:3], v_fd, atol=1e-4)


def test_inverse_twist_recovers_qdot(jac):
    rng = np.random.default_rng(2)
    q = rng.uniform(-math.pi / 4, math.pi / 4, size=6)
    qdot_true = rng.uniform(-0.5, 0.5, size=6)
    V = jac.twist(q, qdot_true)

    qdot_solved = jac.inverse_twist(q, V, method="damped_pinv", damping=1e-4)
    assert np.allclose(qdot_solved, qdot_true, atol=1e-6)


def test_inverse_twist_tracks_desired_twist(jac):
    q = np.zeros(6)
    V_desired = np.array([0.1, 0.0, 0.05, 0.0, 0.0, 0.0])
    qdot = jac.inverse_twist(q, V_desired, method="pinv")
    V_achieved = jac.twist(q, qdot)
    assert np.linalg.norm(V_desired - V_achieved) < 1e-10


def test_null_space_projector(jac):
    q = np.zeros(6)
    N = jac.null_space_projector(q)
    J = jac.jac_numeric(q)

    # N should be idempotent and map any qdot into the null space of J.
    assert np.allclose(N @ N, N, atol=1e-10)
    qdot_test = np.array([0.2, -0.3, 0.5, -0.1, 0.1, -0.2])
    qdot_ns = N @ qdot_test
    assert np.allclose(J @ qdot_ns, 0.0, atol=1e-10)


def test_static_force_duality(jac):
    q = np.zeros(6)
    # Downward force only.
    F = np.array([0.0, 0.0, -10.0, 0.0, 0.0, 0.0])
    tau = jac.joint_torques_from_force(q, F)

    # Shoulder and elbow joints must develop positive torque to resist -Z force.
    assert tau[1] > 0.0
    assert tau[2] > 0.0


def test_waist_rotation_twist(jac):
    q = np.zeros(6)
    qdot = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    V = jac.twist(q, qdot)

    # Pure waist spin produces tangential velocity in +Y and angular velocity +Z.
    ee_pos = jac.end_effector_position(q)
    r = ee_pos - jac.joint_positions(q)[0]
    expected_v = np.cross([0.0, 0.0, 1.0], r) * qdot[0]
    assert np.allclose(V[:3], expected_v, atol=1e-10)
    assert np.allclose(V[3:], [0.0, 0.0, 1.0], atol=1e-10)
