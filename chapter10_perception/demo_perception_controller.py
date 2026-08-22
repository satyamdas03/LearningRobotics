"""Milestone 4 demo: detect a colored block in simulation, then plan a reach.

Pipeline:
1. Load the manipulation scene (arm + table + colored blocks).
2. Render an RGB image from a scene camera.
3. Use the ground-truth detector to locate the red block.
4. Use inverse kinematics to find a joint configuration that places the
   end-effector above the block.
5. Command the arm joints to that configuration and verify the end-effector
   reaches the intended target.
"""
from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

# Reuse Chapter 5 IK without installing anything.
HERE = Path(__file__).parent
ROOT = HERE.parent
for sub in ("chapter05_inverse_kinematics",):
    p = str(ROOT / sub)
    if p not in sys.path:
        sys.path.insert(0, p)

from inverse_kinematics import InverseKinematics  # noqa: E402
from renderer import MujocoRenderer  # noqa: E402
from perception import SceneObjectDetector  # noqa: E402
from controller import JointPositionController  # noqa: E402


def reach_red_block(
    xml_path: str | Path,
    use_position_actuators: bool = True,
    duration: float = 3.0,
    dt: float = 0.01,
) -> dict:
    """Run the full perception → planning → control demo and return diagnostics."""
    xml_path = Path(xml_path)
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    # Let objects settle on the table.
    for _ in range(300):
        mujoco.mj_step(model, data)

    renderer = MujocoRenderer(model, data, width=640, height=480)
    lookat = np.array([0.8, 0.0, 0.55])
    rgb = renderer.render_rgb(lookat=lookat, distance=1.2, azimuth=135.0, elevation=30.0)

    detector = SceneObjectDetector(model, data)
    red = detector.find_object_by_color((0.9, 0.1, 0.1), tolerance=0.3)
    if red is None:
        renderer.close()
        raise RuntimeError("Red block not detected")

    # Target: 10 cm above the block for a top-down reach.
    target_pos = red.position.copy()
    target_pos[2] += 0.10

    # Map the scene arm's qpos indices (free bodies come before the arm joints).
    arm_joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
        for i in range(1, 7)
    ]
    arm_qpos = [int(model.jnt_qposadr[jid]) for jid in arm_joint_ids]
    arm_qvel = [int(model.jnt_dofadr[jid]) for jid in arm_joint_ids]
    arm_act_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"act_joint{i}")
        for i in range(1, 7)
    ]

    # IK using a matching arm XML.
    arm_xml = HERE / "arm.xml"
    ik = InverseKinematics(str(arm_xml))
    q0 = np.zeros(ik.model.nq)
    q_des, info = ik.ik_numeric(
        q0=q0,
        R_target=np.eye(3),
        p_target=target_pos,
        position_only=True,
        max_iters=300,
        tol=1e-4,
    )
    if info["position_error"] > 0.05:
        renderer.close()
        raise RuntimeError(
            f"IK failed to converge: position error {info['position_error']:.4f} m"
        )

    # Command the arm toward the IK solution.
    controller = JointPositionController(q_des)
    data.qpos[arm_qpos] = q_des
    data.qvel[arm_qvel] = 0.0
    q_cmd = controller.compute(
        np.array([data.qpos[i] for i in arm_qpos]),
        np.array([data.qvel[i] for i in arm_qvel]),
    )
    if use_position_actuators:
        for idx, act_id in enumerate(arm_act_ids):
            data.ctrl[act_id] = q_cmd[idx]
    mujoco.mj_forward(model, data)

    # Run a short dynamic rollout if the scene has position actuators.
    if use_position_actuators:
        n_steps = max(1, int(round(duration / dt)))
        for _ in range(n_steps):
            mujoco.mj_step(model, data)

    final_ee = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")].copy()
    final_error = float(np.linalg.norm(target_pos - final_ee))

    renderer.close()
    return {
        "red_block_position": red.position.tolist(),
        "target_position": target_pos.tolist(),
        "final_ee_position": final_ee.tolist(),
        "final_error": final_error,
        "q_des": q_des.tolist(),
        "image_shape": rgb.shape,
        "ik_info": info,
    }


if __name__ == "__main__":
    here = Path(__file__).parent
    info = reach_red_block(here / "scene.xml")
    print("Red block at:", info["red_block_position"])
    print("Target EE at:", info["target_position"])
    print("Final EE at: ", info["final_ee_position"])
    print("Final error: ", f"{info['final_error']:.4f} m")
    print("Image shape: ", info["image_shape"])
