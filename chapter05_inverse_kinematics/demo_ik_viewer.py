"""
Chapter 5 — Interactive demo: Jacobian-based inverse kinematics in MuJoCo viewer.

The 6-DOF arm tracks a moving target pose using the damped pseudoinverse.
Use the viewer to watch the EE converge.  The target cycles through a set of
reachable poses; after each pose is reached within tolerance, it moves to the
next one.
"""
from __future__ import annotations

import math
import time

import mujoco
import mujoco.viewer
import numpy as np

from inverse_kinematics import InverseKinematics, joint_limit_centering_objective


def _rot_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def main() -> None:
    ik = InverseKinematics()
    model = ik.model
    data = ik.data

    # A small red sphere to mark the target pose. It must live inside worldbody.
    target_body_xml = """
    <body name="target_marker" pos="0 0 0" mocap="true">
      <geom name="target_geom" type="sphere" size="0.035" rgba="1 0 0 0.6" />
    </body>
"""
    # We create a separate model that includes both the arm and the marker.
    xml_str = ik._xml
    xml_str = xml_str.replace("</worldbody>", target_body_xml + "    </worldbody>")
    marker_model = mujoco.MjModel.from_xml_string(xml_str)
    marker_data = mujoco.MjData(marker_model)

    # Sequence of reachable target poses (p_target, R_target).
    targets = [
        (np.array([0.50, 0.20, 0.70]), _rot_z(0.0) @ _rot_y(math.radians(30.0))),
        (np.array([0.40, -0.20, 0.60]), _rot_z(0.0) @ _rot_y(math.radians(10.0))),
        (np.array([0.30, 0.10, 0.80]), _rot_z(math.radians(45.0)) @ _rot_y(math.radians(-20.0))),
        (np.array([0.60, -0.10, 0.55]), _rot_z(math.radians(-30.0)) @ _rot_y(math.radians(40.0))),
    ]

    target_index = 0
    p_target, R_target = targets[target_index]
    marker_data.mocap_pos[0] = p_target
    q = np.zeros(6)
    secondary = joint_limit_centering_objective(ik)

    # Time between target switches.
    hold_time = 2.0
    last_switch = time.time()

    with mujoco.viewer.launch_passive(marker_model, marker_data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            # Run one IK step per viewer frame.
            # We reuse the same InverseKinematics object but need to sync its
            # internal data with the marker model's qpos for the arm joints.
            ik.set_q(q)
            q, info = ik.ik_numeric(
                q0=q,
                R_target=R_target,
                p_target=p_target,
                method="damped_pinv",
                damping=0.05,
                max_iters=5,
                tol=1e-4,
                step_scale=0.5,
                secondary_objective=secondary,
            )

            # Copy solved joints into the marker model's data.
            marker_data.qpos[:] = q
            mujoco.mj_forward(marker_model, marker_data)
            marker_data.mocap_pos[0] = p_target

            if info["converged"] and (time.time() - last_switch) > hold_time:
                target_index = (target_index + 1) % len(targets)
                p_target, R_target = targets[target_index]
                last_switch = time.time()

            # Update viewer overlay text with target index and error.
            viewer.user_scn.ngeom = 0
            # (viewer overlays can be added via custom render callbacks; keeping
            # it simple with console prints.)

            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
