"""Chapter 9 — Convert a Chapter 8 C-space path into a timed trajectory.

Given a path (list of configurations) and a total travel time, assign waypoint
times proportionally to C-space distance and fit a smooth polynomial
trajectory.  The result is ready to be tracked by a Chapter 7 controller.
"""
from __future__ import annotations

import numpy as np

from time_scaling import TimeScaling, trapezoidal_time_scaling
from trajectory import Trajectory, cubic_interpolation, quintic_interpolation


def path_to_waypoint_times(path: list[np.ndarray], total_time: float) -> list[float]:
    """Assign monotonically increasing waypoint times proportional to distance."""
    if len(path) < 2:
        raise ValueError("path must contain at least two configurations.")
    if total_time <= 0.0:
        raise ValueError("total_time must be positive.")

    distances = [0.0]
    for i in range(len(path) - 1):
        distances.append(distances[-1] + float(np.linalg.norm(path[i + 1] - path[i])))

    total_distance = distances[-1]
    if total_distance <= 1e-9:
        return [i * total_time / (len(path) - 1) for i in range(len(path))]

    return [total_time * d / total_distance for d in distances]


def path_to_trajectory(
    path: list[np.ndarray],
    total_time: float | None = None,
    scaling: TimeScaling | None = None,
    order: str = "cubic",
    start_velocity: np.ndarray | None = None,
    end_velocity: np.ndarray | None = None,
) -> Trajectory:
    """Convert a geometric path into a timed trajectory.

    Parameters
    ----------
    path:
        List of C-space configurations.
    total_time:
        Desired motion duration in seconds.  Ignored if ``scaling`` is given.
    scaling:
        Optional time scaling profile.  If provided, the path parameter is
        remapped through it so the trajectory respects velocity/acceleration
        bounds along the (unit) path.
    order:
        ``"cubic"`` or ``"quintic"`` interpolation between waypoints.
    start_velocity, end_velocity:
        Boundary velocities (default zero).

    Returns
    -------
    A ``Trajectory`` object that can be sampled at any ``t`` in ``[0, T]``.
    """
    if scaling is None:
        if total_time is None:
            raise ValueError("Either total_time or scaling must be provided.")
        waypoint_times = path_to_waypoint_times(path, total_time)
    else:
        # Build the trajectory in path-parameter space, then reparameterize by time.
        s_times, s_vals, _, _ = scaling.sample(n=max(50, len(path) * 10))
        # The sample covers [0, T]; we want waypoints spaced in s.
        s_waypoints = np.linspace(0.0, 1.0, len(path))
        waypoint_times = [float(np.interp(sw, s_vals, s_times)) for sw in s_waypoints]
        waypoint_times[-1] = float(scaling.t_total)

    if order == "cubic":
        return cubic_interpolation(path, waypoint_times, start_velocity, end_velocity)
    if order == "quintic":
        return quintic_interpolation(
            path, waypoint_times, start_velocity, end_velocity
        )
    raise ValueError(f"order must be 'cubic' or 'quintic', got {order!r}")


def plan_to_timed_trajectory(
    path: list[np.ndarray],
    max_joint_velocity: float,
    max_joint_acceleration: float,
    order: str = "cubic",
) -> Trajectory:
    """Convenience wrapper: plan + trapezoidal time scaling + polynomial fit.

    ``max_joint_velocity`` and ``max_joint_acceleration`` are interpreted as
    bounds on the normalized path parameter.  For a real arm you should scale
    them by the physical joint limits; here they give a consistent motion time.
    """
    scaling = trapezoidal_time_scaling(
        max_velocity=max_joint_velocity,
        max_acceleration=max_joint_acceleration,
    )
    return path_to_trajectory(path, scaling=scaling, order=order)
