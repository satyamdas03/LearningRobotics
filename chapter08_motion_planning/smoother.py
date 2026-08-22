"""Chapter 8 — Motion Planning: path smoothing utilities."""
from __future__ import annotations

import numpy as np

from planners import CollisionChecker, q_distance


def shortcut_smooth(
    path: list[np.ndarray],
    checker: CollisionChecker,
    max_iters: int = 200,
    seed: int | None = 0,
) -> list[np.ndarray]:
    """Stochastic shortcut smoothing: try to replace sub-paths by straight segments."""
    if len(path) <= 2:
        return [np.asarray(p, dtype=float) for p in path]

    rng = np.random.default_rng(seed)
    smoothed = [np.asarray(p, dtype=float) for p in path]

    for _ in range(max_iters):
        n = len(smoothed)
        if n <= 2:
            break
        i = rng.integers(0, n - 1)
        j = rng.integers(i + 1, n)
        if checker.is_segment_free(smoothed[i], smoothed[j]):
            smoothed = smoothed[: i + 1] + [smoothed[j]] + smoothed[j + 1 :]

    return smoothed


def cubic_bspline_interpolate(
    path: list[np.ndarray], n_points: int = 100
) -> list[np.ndarray]:
    """Return a dense cubic B-spline interpolation of the input path."""
    if len(path) < 2:
        return [np.asarray(path[0], dtype=float)] if path else []

    points = np.array(path, dtype=float)
    diffs = np.sqrt(np.sum(np.diff(points, axis=0) ** 2, axis=1))
    s = np.concatenate([[0.0], np.cumsum(diffs)])
    if s[-1] < 1e-9:
        return [points[0]] * n_points
    s /= s[-1]

    # Uniform parameter values for output.
    t = np.linspace(0.0, 1.0, n_points)
    out: list[np.ndarray] = []
    for tt in t:
        # Find segment.
        idx = int(np.searchsorted(s, tt, side="right") - 1)
        idx = max(0, min(idx, len(s) - 2))
        u = (tt - s[idx]) / (s[idx + 1] - s[idx] + 1e-9)
        p0, p1, p2, p3 = _control_points(points, idx)
        out.append(_cubic_catmull_rom(p0, p1, p2, p3, u))
    return out


def _control_points(points: np.ndarray, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(points)
    p0 = points[max(0, idx - 1)]
    p1 = points[idx]
    p2 = points[min(n - 1, idx + 1)]
    p3 = points[min(n - 1, idx + 2)]
    return p0, p1, p2, p3


def _cubic_catmull_rom(
    p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, u: float
) -> np.ndarray:
    """Evaluate a Catmull-Rom spline segment at parameter u in [0,1]."""
    u2 = u * u
    u3 = u2 * u
    return (
        0.5
        * (
            (2.0 * p1)
            + (-p0 + p2) * u
            + (2.0 * p0 - 5.0 * p1 + 4.0 * p2 - p3) * u2
            + (-p0 + 3.0 * p1 - 3.0 * p2 + p3) * u3
        )
    )
