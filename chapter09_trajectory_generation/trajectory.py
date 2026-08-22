"""Chapter 9 — Trajectory generation: polynomial interpolation in joint space.

A trajectory is a time-parameterized path ``q(t)``.  This module builds piecewise
polynomial trajectories from C-space waypoints with boundary conditions on
position, velocity and acceleration.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    """A time-parameterized joint-space trajectory.

    Parameters
    ----------
    segments:
        List of ``(t_start, t_end, coeffs)`` where ``coeffs`` is an
        ``(n_q, order)`` array.  The polynomial is evaluated as
        ``q(t) = coeffs @ [1, dt, dt**2, ...]`` with ``dt = t - t_start``.
    t_total:
        Total duration in seconds.
    """

    segments: list[tuple[float, float, np.ndarray]]
    t_total: float

    def evaluate(self, t: float) -> np.ndarray:
        """Return ``q(t)`` (clamped to the trajectory bounds)."""
        t = float(np.clip(t, 0.0, self.t_total))
        for t_start, t_end, coeffs in self.segments:
            if t <= t_end or np.isclose(t, t_end):
                dt = t - t_start
                powers = np.vander([dt], N=coeffs.shape[1], increasing=True)[0]
                return coeffs @ powers
        # Fallback for the final endpoint.
        _, _, coeffs = self.segments[-1]
        return coeffs @ np.ones(coeffs.shape[1])

    def evaluate_velocity(self, t: float) -> np.ndarray:
        """Return analytical derivative ``qdot(t)``."""
        t = float(np.clip(t, 0.0, self.t_total))
        for t_start, t_end, coeffs in self.segments:
            if t <= t_end or np.isclose(t, t_end):
                dt = t - t_start
                deriv = np.polyder(np.poly1d([0.0] + coeffs[0].tolist()[::-1]))
                # Evaluate each joint's polynomial derivative.
                qdot = np.zeros(coeffs.shape[0])
                for i, c in enumerate(coeffs):
                    p = np.poly1d(c[::-1])
                    qdot[i] = np.polyder(p)(dt)
                return qdot
        return np.zeros(self.segments[-1][2].shape[0])

    def evaluate_acceleration(self, t: float) -> np.ndarray:
        """Return analytical second derivative ``qddot(t)``."""
        t = float(np.clip(t, 0.0, self.t_total))
        for t_start, t_end, coeffs in self.segments:
            if t <= t_end or np.isclose(t, t_end):
                dt = t - t_start
                qddot = np.zeros(coeffs.shape[0])
                for i, c in enumerate(coeffs):
                    p = np.poly1d(c[::-1])
                    qddot[i] = np.polyder(p, 2)(dt)
                return qddot
        return np.zeros(self.segments[-1][2].shape[0])

    def sample(self, n: int) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
        """Sample ``n`` evenly spaced points from the trajectory."""
        ts = np.linspace(0.0, self.t_total, n)
        qs = [self.evaluate(t) for t in ts]
        qdots = [self.evaluate_velocity(t) for t in ts]
        qddots = [self.evaluate_acceleration(t) for t in ts]
        return ts, qs, qdots, qddots


def _fit_cubic_segment(
    q0: np.ndarray,
    q1: np.ndarray,
    qdot0: np.ndarray,
    qdot1: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Return cubic polynomial coeffs ``[a0, a1, a2, a3]`` for each joint.

    ``q(s) = a0 + a1*s + a2*s**2 + a3*s**3`` with ``s`` in ``[0, dt]``.
    """
    a0 = q0
    a1 = qdot0
    a2 = 3.0 * (q1 - q0) / dt**2 - (2.0 * qdot0 + qdot1) / dt
    a3 = -2.0 * (q1 - q0) / dt**3 + (qdot0 + qdot1) / dt**2
    return np.stack([a0, a1, a2, a3], axis=1)


def _fit_quintic_segment(
    q0: np.ndarray,
    q1: np.ndarray,
    qdot0: np.ndarray,
    qdot1: np.ndarray,
    qddot0: np.ndarray,
    qddot1: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Return quintic polynomial coeffs ``[a0..a5]`` for each joint."""
    a0 = q0
    a1 = qdot0
    a2 = 0.5 * qddot0
    a3 = (
        20.0 * (q1 - q0)
        - (8.0 * qdot1 + 12.0 * qdot0) * dt
        - (3.0 * qddot0 - qddot1) * dt**2
    ) / (2.0 * dt**3)
    a4 = (
        -30.0 * (q1 - q0)
        + (14.0 * qdot1 + 16.0 * qdot0) * dt
        + (3.0 * qddot0 - 2.0 * qddot1) * dt**2
    ) / (2.0 * dt**4)
    a5 = (
        12.0 * (q1 - q0)
        - 6.0 * (qdot1 + qdot0) * dt
        + (qddot1 - qddot0) * dt**2
    ) / (2.0 * dt**5)
    return np.stack([a0, a1, a2, a3, a4, a5], axis=1)


def cubic_interpolation(
    waypoints: list[np.ndarray],
    waypoint_times: list[float] | np.ndarray,
    start_velocity: np.ndarray | None = None,
    end_velocity: np.ndarray | None = None,
) -> Trajectory:
    """Fit a C²-continuous cubic spline through ``waypoints``.

    Boundary velocities default to zero.  Intermediate waypoint velocities are
    chosen to match the central finite-difference slope, clamped by the
    average segment velocity to avoid overshoot.
    """
    n = len(waypoints)
    if n < 2:
        raise ValueError("At least two waypoints are required.")
    times = [float(t) for t in waypoint_times]
    if len(times) != n:
        raise ValueError("waypoint_times must match the number of waypoints.")

    nq = waypoints[0].shape[0]
    q = np.asarray(waypoints, dtype=float)
    t = np.asarray(times, dtype=float)

    # Boundary velocities.
    v0 = np.zeros(nq) if start_velocity is None else np.asarray(start_velocity, dtype=float)
    vf = np.zeros(nq) if end_velocity is None else np.asarray(end_velocity, dtype=float)

    # Intermediate velocities: finite-difference slopes, clamped to segment average.
    velocities: list[np.ndarray] = [v0]
    for i in range(1, n - 1):
        dt_prev = t[i] - t[i - 1]
        dt_next = t[i + 1] - t[i]
        slope = (q[i + 1] - q[i - 1]) / (dt_prev + dt_next)
        # Clamp to the slower of the two average segment velocities to reduce overshoot.
        avg_prev = (q[i] - q[i - 1]) / dt_prev if dt_prev > 1e-9 else np.zeros(nq)
        avg_next = (q[i + 1] - q[i]) / dt_next if dt_next > 1e-9 else np.zeros(nq)
        max_slope = np.minimum(np.abs(avg_prev), np.abs(avg_next))
        slope = np.clip(slope, -max_slope, max_slope)
        velocities.append(slope)
    velocities.append(vf)

    segments: list[tuple[float, float, np.ndarray]] = []
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        if dt <= 1e-9:
            continue
        coeffs = _fit_cubic_segment(q[i], q[i + 1], velocities[i], velocities[i + 1], dt)
        segments.append((t[i], t[i + 1], coeffs))

    return Trajectory(segments=segments, t_total=float(t[-1]))


def quintic_interpolation(
    waypoints: list[np.ndarray],
    waypoint_times: list[float] | np.ndarray,
    start_velocity: np.ndarray | None = None,
    end_velocity: np.ndarray | None = None,
    start_acceleration: np.ndarray | None = None,
    end_acceleration: np.ndarray | None = None,
) -> Trajectory:
    """Fit a C³-continuous quintic spline through ``waypoints``.

    All boundary velocities/accelerations default to zero.
    """
    n = len(waypoints)
    if n < 2:
        raise ValueError("At least two waypoints are required.")
    times = [float(t) for t in waypoint_times]
    if len(times) != n:
        raise ValueError("waypoint_times must match the number of waypoints.")

    nq = waypoints[0].shape[0]
    q = np.asarray(waypoints, dtype=float)
    t = np.asarray(times, dtype=float)

    v0 = np.zeros(nq) if start_velocity is None else np.asarray(start_velocity, dtype=float)
    vf = np.zeros(nq) if end_velocity is None else np.asarray(end_velocity, dtype=float)
    a0 = np.zeros(nq) if start_acceleration is None else np.asarray(start_acceleration, dtype=float)
    af = np.zeros(nq) if end_acceleration is None else np.asarray(end_acceleration, dtype=float)

    # Simple heuristic for intermediate v/a: finite differences, zero accel at peaks.
    velocities: list[np.ndarray] = [v0]
    accelerations: list[np.ndarray] = [a0]
    for i in range(1, n - 1):
        dt_prev = t[i] - t[i - 1]
        dt_next = t[i + 1] - t[i]
        slope = (q[i + 1] - q[i - 1]) / (dt_prev + dt_next)
        velocities.append(slope)
        # Acceleration estimate from three points.
        a = 2.0 * (
            (q[i + 1] - q[i]) / dt_next - (q[i] - q[i - 1]) / dt_prev
        ) / (dt_prev + dt_next)
        accelerations.append(a)
    velocities.append(vf)
    accelerations.append(af)

    segments: list[tuple[float, float, np.ndarray]] = []
    for i in range(n - 1):
        dt = t[i + 1] - t[i]
        if dt <= 1e-9:
            continue
        coeffs = _fit_quintic_segment(
            q[i],
            q[i + 1],
            velocities[i],
            velocities[i + 1],
            accelerations[i],
            accelerations[i + 1],
            dt,
        )
        segments.append((t[i], t[i + 1], coeffs))

    return Trajectory(segments=segments, t_total=float(t[-1]))
