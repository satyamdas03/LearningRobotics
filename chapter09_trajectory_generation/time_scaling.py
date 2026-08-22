"""Chapter 9 — Time scaling: trapezoidal and S-curve velocity profiles.

A time scaling maps a path parameter ``s`` in ``[0, 1]`` to time ``t`` in
``[0, T]`` such that the motion respects bounds on velocity, acceleration and
jerk.  The returned object gives ``s(t)``, ``sdot(t)`` and ``sddot(t)``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TimeScaling:
    """A time scaling profile ``s(t)`` for a unit path."""

    t_total: float
    max_sdot: float
    max_sddot: float
    max_sdddot: float | None = None

    def s(self, t: float) -> float:
        """Path parameter at time ``t``."""
        raise NotImplementedError

    def sdot(self, t: float) -> float:
        """First derivative of path parameter at time ``t``."""
        raise NotImplementedError

    def sddot(self, t: float) -> float:
        """Second derivative of path parameter at time ``t``."""
        raise NotImplementedError

    def sample(self, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Sample ``n`` points from the profile."""
        ts = np.linspace(0.0, self.t_total, n)
        s_vals = np.array([self.s(t) for t in ts])
        sd_vals = np.array([self.sdot(t) for t in ts])
        sdd_vals = np.array([self.sddot(t) for t in ts])
        return ts, s_vals, sd_vals, sdd_vals


def _phase(t: float, boundaries: list[float]) -> int:
    """Return the index of the phase containing ``t`` (clamped to last)."""
    for i, boundary in enumerate(boundaries):
        if t <= boundary:
            return i
    return len(boundaries)


def trapezoidal_time_scaling(max_velocity: float, max_acceleration: float) -> TimeScaling:
    """Return a trapezoidal profile: accelerate, cruise, decelerate.

    The total time is the minimum time to traverse a unit path while respecting
    ``|sdot| <= max_velocity`` and ``|sddot| <= max_acceleration``.
    """
    if max_velocity <= 0.0 or max_acceleration <= 0.0:
        raise ValueError("max_velocity and max_acceleration must be positive.")

    # Time to accelerate to max_velocity and distance covered during that.
    t_ramp = max_velocity / max_acceleration
    s_ramp = 0.5 * max_acceleration * t_ramp**2

    if 2.0 * s_ramp >= 1.0:
        # Triangular profile: never reaches cruise velocity.
        t_ramp = np.sqrt(1.0 / max_acceleration)
        max_velocity = max_acceleration * t_ramp
        t_total = 2.0 * t_ramp
        cruise_time = 0.0
    else:
        t_total = 2.0 * t_ramp + (1.0 - 2.0 * s_ramp) / max_velocity
        cruise_time = t_total - 2.0 * t_ramp

    boundaries = [t_ramp, t_ramp + cruise_time, t_total]

    class _Trapezoidal(TimeScaling):
        def s(self, t: float) -> float:
            t = float(np.clip(t, 0.0, t_total))
            phase = _phase(t, boundaries)
            if phase == 0:
                return 0.5 * max_acceleration * t**2
            if phase == 1:
                dt = t - t_ramp
                return s_ramp + max_velocity * dt
            dt = t_total - t
            return 1.0 - 0.5 * max_acceleration * dt**2

        def sdot(self, t: float) -> float:
            t = float(np.clip(t, 0.0, t_total))
            phase = _phase(t, boundaries)
            if phase == 0:
                return max_acceleration * t
            if phase == 1:
                return max_velocity
            return max_acceleration * (t_total - t)

        def sddot(self, t: float) -> float:
            t = float(np.clip(t, 0.0, t_total))
            phase = _phase(t, boundaries)
            if phase == 0:
                return max_acceleration
            if phase == 1:
                return 0.0
            return -max_acceleration

    return _Trapezoidal(
        t_total=t_total,
        max_sdot=max_velocity,
        max_sddot=max_acceleration,
    )


def scurve_time_scaling(
    max_velocity: float,
    max_acceleration: float,
    max_jerk: float,
) -> TimeScaling:
    """Return a 7-segment S-curve profile with bounded jerk.

    The profile consists of:

    1. jerk-up (+J)
    2. constant-acceleration (+A)
    3. jerk-down (-J) to zero acceleration
    4. constant-velocity (0)
    5. jerk-down (-J) to negative acceleration
    6. constant-deceleration (-A)
    7. jerk-up (+J) to zero acceleration

    If the unit path is short, the constant-acceleration and/or constant-velocity
    phases may be omitted.
    """
    if max_velocity <= 0.0 or max_acceleration <= 0.0 or max_jerk <= 0.0:
        raise ValueError("max_velocity, max_acceleration and max_jerk must be positive.")

    # Time to ramp acceleration from 0 to max with max_jerk.
    t_jerk = max_acceleration / max_jerk
    # Velocity reached after the first pair of jerk ramps (no constant accel).
    v_after_jerk_ramps = max_jerk * t_jerk**2

    if v_after_jerk_ramps >= max_velocity:
        # Pure S-curve: no constant-acceleration or constant-velocity phase.
        # Displacement of one jerk-up/jerk-down pair = J * t_jerk^3 / 3.
        # Total displacement for four ramps = 2 * J * t_jerk^3 / 3 = 1.
        t_jerk = np.cbrt(1.5 / max_jerk)
        t_const_accel = 0.0
        t_cruise = 0.0
        t_total = 4.0 * t_jerk
        max_accel_reached = max_jerk * t_jerk
        max_velocity_reached = max_jerk * t_jerk**2
    else:
        # Time needed at constant max acceleration to hit max_velocity.
        t_const_accel = (max_velocity - v_after_jerk_ramps) / max_acceleration
        if t_const_accel < 0.0:
            t_const_accel = 0.0

        # Displacement during the forward acceleration half (phases 1-3).
        s_accel_half = (
            # Phase 1: jerk-up displacement.
            max_jerk * t_jerk**3 / 6.0
            # Phase 2: constant-accel displacement.
            + v_after_jerk_ramps * t_const_accel
            + 0.5 * max_acceleration * t_const_accel**2
            # Phase 3: jerk-down displacement.
            + (v_after_jerk_ramps + max_acceleration * t_const_accel) * t_jerk
            - max_jerk * t_jerk**3 / 6.0
        )

        if 2.0 * s_accel_half >= 1.0:
            # No constant-velocity cruise; scale the accel phase to fit the path.
            scale = np.cbrt(1.0 / (2.0 * s_accel_half))
            t_jerk *= scale
            t_const_accel *= scale
            t_cruise = 0.0
            t_total = 4.0 * t_jerk + 2.0 * t_const_accel
            max_accel_reached = max_jerk * t_jerk
            max_velocity_reached = max_jerk * t_jerk**2 + max_accel_reached * t_const_accel
        else:
            t_cruise = (1.0 - 2.0 * s_accel_half) / max_velocity
            t_total = 4.0 * t_jerk + 2.0 * t_const_accel + t_cruise
            max_accel_reached = max_acceleration
            max_velocity_reached = max_velocity

    # Phase boundaries.
    t1 = t_jerk
    t2 = t_jerk + t_const_accel
    t3 = 2.0 * t_jerk + t_const_accel
    t4 = 2.0 * t_jerk + t_const_accel + t_cruise
    t5 = 3.0 * t_jerk + t_const_accel + t_cruise
    t6 = 3.0 * t_jerk + 2.0 * t_const_accel + t_cruise
    boundaries = [t1, t2, t3, t4, t5, t6, t_total]

    # Precompute boundary states for the forward half.
    v1 = 0.5 * max_jerk * t_jerk**2
    v2 = v1 + max_accel_reached * t_const_accel
    v3 = v2 + max_accel_reached * t_jerk - 0.5 * max_jerk * t_jerk**2
    s1 = max_jerk * t_jerk**3 / 6.0
    s2 = s1 + v1 * t_const_accel + 0.5 * max_accel_reached * t_const_accel**2
    s3 = s2 + v2 * t_jerk - max_jerk * t_jerk**3 / 6.0
    s4 = s3 + v3 * t_cruise

    class _SCurve(TimeScaling):
        def s(self, t: float) -> float:
            t = float(np.clip(t, 0.0, t_total))
            phase = _phase(t, boundaries)
            if phase == 0:
                return max_jerk * t**3 / 6.0
            if phase == 1:
                dt = t - t1
                return s1 + v1 * dt + 0.5 * max_accel_reached * dt**2
            if phase == 2:
                dt = t - t2
                return s2 + v2 * dt + 0.5 * max_accel_reached * dt**2 - max_jerk * dt**3 / 6.0
            if phase == 3:
                dt = t - t3
                return s3 + v3 * dt
            if phase == 4:
                dt = t - t4
                return s4 - max_jerk * dt**3 / 6.0
            if phase == 5:
                dt = t - t5
                return (1.0 - s1) - v1 * dt - 0.5 * max_accel_reached * dt**2
            # phase 6: symmetric to phase 0
            dt = t_total - t
            return 1.0 - max_jerk * dt**3 / 6.0

        def sdot(self, t: float) -> float:
            t = float(np.clip(t, 0.0, t_total))
            phase = _phase(t, boundaries)
            if phase == 0:
                return 0.5 * max_jerk * t**2
            if phase == 1:
                return v1 + max_accel_reached * (t - t1)
            if phase == 2:
                dt = t - t2
                return v2 + max_accel_reached * dt - 0.5 * max_jerk * dt**2
            if phase == 3:
                return v3
            if phase == 4:
                dt = t - t4
                return v3 - 0.5 * max_jerk * dt**2
            if phase == 5:
                return v1 + max_accel_reached * (t6 - t)
            # phase 6
            dt = t_total - t
            return 0.5 * max_jerk * dt**2

        def sddot(self, t: float) -> float:
            t = float(np.clip(t, 0.0, t_total))
            phase = _phase(t, boundaries)
            if phase == 0:
                return max_jerk * t
            if phase == 1:
                return max_accel_reached
            if phase == 2:
                return max_accel_reached - max_jerk * (t - t2)
            if phase == 3:
                return 0.0
            if phase == 4:
                return -max_jerk * (t - t4)
            if phase == 5:
                return -max_accel_reached
            # phase 6
            dt = t_total - t
            return -max_jerk * dt

    return _SCurve(
        t_total=t_total,
        max_sdot=max_velocity_reached,
        max_sddot=max_accel_reached,
        max_sdddot=max_jerk,
    )
