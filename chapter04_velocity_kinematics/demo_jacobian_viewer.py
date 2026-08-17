"""
Chapter 4 — Interactive demo: Jacobian-based velocity control in MuJoCo viewer.

The end-effector is commanded to track a moving target in space using the
pseudoinverse of the Jacobian. The viewer opens a 3D window so you can watch
the arm follow the trajectory in real time.

Usage:
    cd chapter04_velocity_kinematics
    . .venv\Scripts\Activate.ps1
    python demo_jacobian_viewer.py

Controls:
    Close the MuJoCo viewer window to stop.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import mujoco
import numpy as np

from jacobian import ArmJacobian


class VelocityTracker:
    """Simple inverse-velocity controller for the 6-DOF arm."""

    def __init__(self, xml_path: str) -> None:
        self.jac = ArmJacobian(xml_path)
        self.model = self.jac.model
        self.data = self.jac.data
        self.ee_id = self.jac.ee_id

        # Joint limits from the model (soft clamps).
        self.q_min = self.model.jnt_range[:, 0].copy()
        self.q_max = self.model.jnt_range[:, 1].copy()

        # Start at a reasonable configuration.
        self.q = np.array([
            0.0,
            math.radians(30),
            -math.radians(30),
            0.0,
            math.radians(45),
            0.0,
        ])
        self.apply_q()

    def apply_q(self) -> None:
        self.data.qpos[:] = self.q
        mujoco.mj_forward(self.model, self.data)

    def target_position(self, t: float) -> np.ndarray:
        """Return a target point that traces a circle in front of the arm."""
        center = np.array([0.55, 0.0, 0.85])
        radius = 0.15
        omega = 0.8  # rad/s
        x = center[0] + radius * math.cos(omega * t)
        y = center[1] + radius * math.sin(omega * t)
        z = center[2] + 0.05 * math.sin(2 * omega * t)
        return np.array([x, y, z])

    def step(self, dt: float) -> None:
        """One control step: drive EE toward the moving target via J⁺."""
        # Current end-effector position.
        p_ee = self.data.site_xpos[self.ee_id].copy()

        # Desired linear velocity = P-control toward target.
        t = time.monotonic()
        p_target = self.target_position(t)
        v_desired = 1.5 * (p_target - p_ee)
        # Clamp to avoid unrealistic speeds.
        v_norm = np.linalg.norm(v_desired)
        if v_norm > 0.5:
            v_desired = 0.5 * v_desired / v_norm

        # No desired angular velocity (keep current orientation).
        V_desired = np.concatenate([v_desired, np.zeros(3)])

        # Inverse velocity with light damping.
        qdot = self.jac.inverse_twist(self.q, V_desired, method="damped_pinv", damping=0.05)

        # Clamp joint velocities for smooth motion.
        qdot = np.clip(qdot, -2.0, 2.0)

        # Integrate.
        self.q = self.q + qdot * dt
        self.q = np.clip(self.q, self.q_min + 0.01, self.q_max - 0.01)

        self.apply_q()


def main() -> None:
    xml_path = str(
        Path(__file__).parent.parent / "chapter01_foundation" / "simple_6dof_arm.xml"
    )

    tracker = VelocityTracker(xml_path)
    dt = 1 / 60.0  # 60 Hz control loop.

    print("Opening MuJoCo viewer...")
    print("The end-effector will track a moving circular target using J⁺ velocity control.")
    print("Close the viewer window to stop.")

    with mujoco.viewer.launch_passive(tracker.model, tracker.data) as viewer:
        start_time = time.monotonic()
        while viewer.is_running():
            step_start = time.monotonic()

            tracker.step(dt)
            viewer.sync()

            # Throttle to ~60 Hz.
            elapsed = time.monotonic() - step_start
            sleep_time = max(0.0, dt - elapsed)
            time.sleep(sleep_time)

    elapsed = time.monotonic() - start_time
    print(f"Viewer closed. Demo ran for {elapsed:.1f} seconds.")


if __name__ == "__main__":
    main()
