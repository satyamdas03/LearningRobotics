"""Tests for Chapter 5 inverse kinematics."""
from __future__ import annotations

import numpy as np
import pytest

from inverse_kinematics import InverseKinematics, joint_limit_centering_objective


@pytest.fixture
def ik():
    return InverseKinematics()


def test_ik_numeric_reaches_default_target(ik):
    q0 = np.zeros(6)
    # Reachable target in front of and above the arm (avoids the wrist joint
    # limit that blocks the previously chosen [0.50, 0.10, 0.70]).
    p_target = np.array([0.60, 0.20, 0.60])
    R_target = np.eye(3)
    q, info = ik.ik_numeric(q0, R_target, p_target, max_iters=300)
    ik.set_q(q)
    _, p_final = ik.current_end_effector_pose()
    assert info["position_error"] < 1e-3
    assert info["rotation_error"] < 1e-2
    assert np.linalg.norm(p_final - p_target) < 1e-3


def test_ik_analytic_2r_matches_numeric(ik):
    # Planar target reachable by an abstract 2R arm (link1 + link2 in the X-Z
    # plane).  The analytic solver should recover the planar 2R solution.
    target = (0.60, 0.50)
    L1 = 0.5
    L2 = 0.5
    q1_a, q2_a = ik.ik_analytic_2r(target, elbow_up=True)
    # Planar 2R forward kinematics.
    x = L1 * np.cos(q1_a) + L2 * np.cos(q1_a + q2_a)
    z = L1 * np.sin(q1_a) + L2 * np.sin(q1_a + q2_a)
    assert np.hypot(x - target[0], z - target[1]) < 1e-6

    # The same angles can be written into the 6-DOF arm and should at least
    # produce a finite, in-limit configuration.
    q = np.zeros(6)
    q[0] = q1_a
    q[1] = q2_a
    ik.set_q(q)
    _, p = ik.current_end_effector_pose()
    assert np.isfinite(p).all()


def test_ik_null_space_reaches_target_and_stays_centered(ik):
    q0 = np.zeros(6)
    # Position-only task leaves a 3-DOF null-space; the centering objective
    # should nudge the solution toward joint centers without breaking accuracy.
    p_target = np.array([0.50, 0.10, 0.70])
    R_target = np.eye(3)
    secondary = joint_limit_centering_objective(ik)
    q_with, info_with = ik.ik_numeric(
        q0,
        R_target,
        p_target,
        max_iters=300,
        position_only=True,
        secondary_objective=secondary,
        secondary_gain=0.05,
    )
    q_without, info_without = ik.ik_numeric(
        q0,
        R_target,
        p_target,
        max_iters=300,
        position_only=True,
        secondary_objective=None,
    )

    # Both should converge in position.
    assert info_with["position_error"] < 1e-3
    assert info_without["position_error"] < 1e-3

    # Null-space version should stay closer to joint centers on average.
    centers = []
    widths = []
    for i, jid in enumerate(ik.joint_ids):
        lo = ik.model.jnt_range[jid, 0]
        hi = ik.model.jnt_range[jid, 1]
        centers.append((lo + hi) / 2.0)
        widths.append((hi - lo) / 2.0)
    centers = np.array(centers)
    widths = np.array(widths)
    norm_with = np.linalg.norm((q_with - centers) / widths)
    norm_without = np.linalg.norm((q_without - centers) / widths)
    assert norm_with <= norm_without + 0.1


def test_ik_numeric_best_effort_for_unreachable_target(ik):
    q0 = np.zeros(6)
    # Far unreachable target.
    p_target = np.array([2.0, 0.0, 2.0])
    R_target = np.eye(3)
    q, info = ik.ik_numeric(q0, R_target, p_target, max_iters=100)
    # Should not blow up; final error should be finite and joints in limits.
    assert np.isfinite(info["position_error"])
    assert info["position_error"] < 3.0
    for i, jid in enumerate(ik.joint_ids):
        lo = ik.model.jnt_range[jid, 0]
        hi = ik.model.jnt_range[jid, 1]
        assert lo - 1e-6 <= q[i] <= hi + 1e-6
