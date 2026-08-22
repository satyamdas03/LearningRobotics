"""Utility math helpers for Chapter 7 control."""
from __future__ import annotations

import numpy as np


def clip_vector(v: np.ndarray, limit: float | np.ndarray) -> np.ndarray:
    """Clip each component of v to [-limit, +limit].

    limit may be a scalar or a per-component array.
    """
    v = np.asarray(v, dtype=float)
    if np.isscalar(limit):
        return np.clip(v, -float(limit), float(limit))
    limit = np.asarray(limit, dtype=float)
    return np.clip(v, -limit, limit)


def pose_error(
    R_current: np.ndarray,
    p_current: np.ndarray,
    R_desired: np.ndarray,
    p_desired: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    """Return 6D pose error [pos_err; rot_err] plus position and rotation norms.

    Rotation error is the axis-angle of R_desired @ R_current.T, which is the
    minimal geodesic error in SO(3).
    """
    pos_err = p_desired - p_current
    dR = R_desired @ R_current.T
    trace = float(np.clip(np.trace(dR), -1.0, 3.0))
    angle = np.arccos((trace - 1.0) / 2.0)
    if angle < 1e-6:
        rot_err = np.zeros(3)
    else:
        s = np.sin(angle)
        rot_err = (angle / (2.0 * s)) * np.array([
            dR[2, 1] - dR[1, 2],
            dR[0, 2] - dR[2, 0],
            dR[1, 0] - dR[0, 1],
        ])
    err = np.concatenate([pos_err, rot_err])
    return err, float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))


def axis_angle_from_matrix(R: np.ndarray) -> tuple[np.ndarray, float]:
    """Recover (axis, angle) from a rotation matrix."""
    trace = float(np.clip(np.trace(R), -1.0, 3.0))
    angle = np.arccos((trace - 1.0) / 2.0)
    if angle < 1e-6:
        return np.array([1.0, 0.0, 0.0]), 0.0
    s = np.sin(angle)
    axis = (1.0 / (2.0 * s)) * np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ])
    return axis, angle


def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    """Return the 3x3 rotation matrix for a rotation of ``angle`` radians around
    ``axis`` using Rodrigues' formula.
    """
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    c = np.cos(angle)
    s = np.sin(angle)
    ux, uy, uz = axis
    return np.array([
        [c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s],
        [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s],
        [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)],
    ])
