"""Tests for Chapter 9 — Trajectory generation."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "chapter09_trajectory_generation"))

from path_to_trajectory import path_to_trajectory, plan_to_timed_trajectory
from time_scaling import scurve_time_scaling, trapezoidal_time_scaling
from trajectory import cubic_interpolation, quintic_interpolation


def test_cubic_interpolation_hits_waypoints():
    """Cubic spline must pass through every waypoint."""
    waypoints = [
        np.zeros(6),
        np.array([0.2, -0.1, 0.15, 0.0, -0.05, 0.05]),
        np.array([0.4, -0.2, 0.0, 0.1, 0.0, 0.0]),
    ]
    times = [0.0, 1.0, 2.5]
    traj = cubic_interpolation(waypoints, times)

    for t, q in zip(times, waypoints):
        np.testing.assert_allclose(traj.evaluate(t), q, atol=1e-10)


def test_cubic_boundary_velocities():
    """Cubic spline start/end velocities should match the requested values."""
    waypoints = [np.zeros(6), np.full(6, 0.5)]
    times = [0.0, 1.0]
    v0 = np.full(6, 0.1)
    vf = np.full(6, -0.1)
    traj = cubic_interpolation(waypoints, times, start_velocity=v0, end_velocity=vf)

    np.testing.assert_allclose(traj.evaluate_velocity(0.0), v0, atol=1e-10)
    np.testing.assert_allclose(traj.evaluate_velocity(1.0), vf, atol=1e-10)


def test_quintic_interpolation_hits_waypoints_and_zero_acc():
    """Quintic spline hits waypoints with zero boundary acceleration by default."""
    waypoints = [np.zeros(6), np.full(6, 0.3), np.full(6, 0.6)]
    times = [0.0, 1.0, 2.0]
    traj = quintic_interpolation(waypoints, times)

    for t, q in zip(times, waypoints):
        np.testing.assert_allclose(traj.evaluate(t), q, atol=1e-10)
    np.testing.assert_allclose(traj.evaluate_acceleration(0.0), np.zeros(6), atol=1e-10)
    np.testing.assert_allclose(traj.evaluate_acceleration(2.0), np.zeros(6), atol=1e-10)


def test_trapezoidal_profile_is_unit_in_duration():
    """Trapezoidal profile should start at 0 and end at 1 over its duration."""
    profile = trapezoidal_time_scaling(max_velocity=1.0, max_acceleration=2.0)
    assert profile.s(0.0) == pytest.approx(0.0)
    assert profile.s(profile.t_total) == pytest.approx(1.0)
    assert profile.sdot(0.0) == pytest.approx(0.0, abs=1e-10)
    assert profile.sdot(profile.t_total) == pytest.approx(0.0, abs=1e-10)


def test_trapezoidal_respects_velocity_and_acceleration_bounds():
    """Trapezoidal profile should never exceed its declared bounds."""
    profile = trapezoidal_time_scaling(max_velocity=0.8, max_acceleration=1.5)
    _, s, sd, sdd = profile.sample(n=200)
    assert np.max(sd) <= 0.8 + 1e-9
    assert np.min(sd) >= -1e-9
    assert np.max(sdd) <= 1.5 + 1e-9
    assert np.min(sdd) >= -1.5 - 1e-9


def test_scurve_profile_is_unit_in_duration():
    """S-curve profile should start at 0 and end at 1 over its duration."""
    profile = scurve_time_scaling(
        max_velocity=1.0, max_acceleration=2.0, max_jerk=10.0
    )
    assert profile.s(0.0) == pytest.approx(0.0, abs=1e-6)
    assert profile.s(profile.t_total) == pytest.approx(1.0, abs=1e-6)
    assert profile.sdot(0.0) == pytest.approx(0.0, abs=1e-6)
    assert profile.sdot(profile.t_total) == pytest.approx(0.0, abs=1e-6)


def test_scurve_is_smoother_than_trapezoidal():
    """S-curve acceleration should be continuous, unlike trapezoidal."""
    trap = trapezoidal_time_scaling(max_velocity=1.0, max_acceleration=2.0)
    scurve = scurve_time_scaling(
        max_velocity=1.0, max_acceleration=2.0, max_jerk=10.0
    )

    n = 200
    ts_trap = np.linspace(0.0, trap.t_total, n)
    ts_scurve = np.linspace(0.0, scurve.t_total, n)
    sdd_trap = np.array([trap.sddot(t) for t in ts_trap])
    sdd_scurve = np.array([scurve.sddot(t) for t in ts_scurve])

    max_jump_trap = np.max(np.abs(np.diff(sdd_trap)))
    max_jump_scurve = np.max(np.abs(np.diff(sdd_scurve)))

    # The S-curve ramps acceleration continuously, so its sampled acceleration
    # jumps are bounded by jerk * sample_spacing (approximately).
    assert max_jump_scurve < max_jump_trap


def test_path_to_trajectory_hits_start_and_goal():
    """A trajectory built from a path should start at the first waypoint and
    end at the last.
    """
    path = [
        np.zeros(6),
        np.array([0.2, -0.1, 0.15, 0.0, -0.05, 0.05]),
        np.array([0.4, -0.2, 0.0, 0.1, 0.0, 0.0]),
    ]
    traj = path_to_trajectory(path, total_time=2.0)
    np.testing.assert_allclose(traj.evaluate(0.0), path[0], atol=1e-10)
    np.testing.assert_allclose(traj.evaluate(traj.t_total), path[-1], atol=1e-10)


def test_plan_to_timed_trajectory_monotonic_time():
    """The convenience wrapper should produce a finite-duration trajectory."""
    path = [np.zeros(6), np.full(6, 0.3)]
    traj = plan_to_timed_trajectory(
        path, max_joint_velocity=0.5, max_joint_acceleration=1.0
    )
    assert traj.t_total > 0.0
    ts, qs, _, _ = traj.sample(n=20)
    assert len(ts) == len(qs) == 20
    assert np.all(np.diff(ts) > 0.0)


def test_trajectory_sample_returns_consistent_shapes():
    """``Trajectory.sample`` should return arrays of matching length."""
    waypoints = [np.zeros(6), np.full(6, 0.5)]
    traj = cubic_interpolation(waypoints, [0.0, 1.0])
    ts, qs, qdots, qddots = traj.sample(n=10)
    assert len(ts) == len(qs) == len(qdots) == len(qddots) == 10
    assert all(q.shape == (6,) for q in qs)
