"""Minimal controller for the perception demo.

The scene uses MuJoCo ``position`` actuators, so the high-level controller only
needs to output a target joint configuration.  A real robot would replace the
position actuators with a torque-tracking low-level loop, but the upstream
perception → planning → control interface stays the same.
"""
from __future__ import annotations

import numpy as np


class JointPositionController:
    """Hold a desired joint configuration and report the commanded one."""

    def __init__(self, q_des: np.ndarray) -> None:
        self.q_des = np.asarray(q_des, dtype=float)

    def compute(
        self,
        q: np.ndarray,
        qdot: np.ndarray | None = None,
        q_des: np.ndarray | None = None,
        **kwargs,
    ) -> np.ndarray:
        """Return the target joint position command for the actuators."""
        if q_des is not None:
            self.q_des = np.asarray(q_des, dtype=float)
        return self.q_des.copy()
