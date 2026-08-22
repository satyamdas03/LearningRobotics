"""
Chapter 7 — Interactive demo: controllers in the MuJoCo viewer.

Run with a controller selector:

    python demo_control_viewer.py --controller gravity
    python demo_control_viewer.py --controller pid
    python demo_control_viewer.py --controller computed_torque
    python demo_control_viewer.py --controller operational_space

The arm executes a simple reference trajectory so you can watch gravity
compensation, joint-space PID, computed-torque tracking, and operational-space
pose tracking side-by-side with the physics.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

# Reuse Chapter 6 dynamics and Chapter 4 Jacobian.
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "chapter06_dynamics"))
sys.path.insert(0, str(ROOT / "chapter04_velocity_kinematics"))
sys.path.insert(0, str(ROOT / "chapter07_control"))

from control import (
    ComputedTorqueController,
    GravityCompensationController,
    JointSpacePIDController,
    OperationalSpaceController,
)
from dynamics import ArmDynamics
from jacobian import ArmJacobian
from utils import rotation_matrix


def _gains_for_arm(arm: ArmDynamics, omega: float) -> tuple[np.ndarray, np.ndarray]:
    M_diag = np.diag(arm.mass_matrix(np.zeros(arm.model.nq)))
    Kp = omega**2 * M_diag
    Kd = 2.0 * np.sqrt(Kp * M_diag)
    return Kp, Kd


def main() -> None:
    parser = argparse.ArgumentParser(description="Chapter 7 control viewer demo")
    parser.add_argument(
        "--controller",
        choices=["gravity", "pid", "computed_torque", "operational_space"],
        default="operational_space",
    )
    args = parser.parse_args()

    arm = ArmDynamics()
    model = arm.model
    data = arm.data

    # Enable gravity and widen actuator limits for the demo.
    model.opt.gravity[:] = np.array([0.0, 0.0, -9.81])
    model.actuator_ctrlrange[:, 0] = -200.0
    model.actuator_ctrlrange[:, 1] = 200.0

    q = np.zeros(model.nq)
    qdot = np.zeros(model.nq)
    arm.set_state(q, qdot)

    controller_name = args.controller
    if controller_name == "gravity":
        controller = GravityCompensationController(arm)
        print("Demo: gravity compensation — arm should hold its home pose.")
    elif controller_name == "pid":
        Kp, Kd = _gains_for_arm(arm, omega=8.0)
        controller = JointSpacePIDController(arm, Kp=Kp, Kd=Kd, gravity_comp=True)
        print("Demo: joint-space PID setpoint tracking.")
    elif controller_name == "computed_torque":
        controller = ComputedTorqueController(
            arm, Kp=np.full(model.nq, 50.0), Kd=np.full(model.nq, 14.0)
        )
        print("Demo: computed-torque sinusoidal tracking.")
    elif controller_name == "operational_space":
        jac = ArmJacobian()
        controller = OperationalSpaceController(
            arm, jac, Kp=np.full(6, 50.0), Kd=np.full(6, 14.0)
        )
        print("Demo: operational-space pose tracking.")
    else:
        raise ValueError(f"Unknown controller: {controller_name}")

    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            step_start = time.time()
            t = time.time() - start

            # Compute the current command from the running simulation state.
            q = data.qpos.copy()
            qdot = data.qvel.copy()

            if controller_name == "gravity":
                tau = controller.compute(q)
            elif controller_name == "pid":
                q_des = 0.1 * np.sin(0.5 * t) * np.array([1.0, -1.0, 1.0, 0.0, 0.5, -0.5])
                tau = controller.compute(q, qdot, q_des=q_des, dt=model.opt.timestep)
            elif controller_name == "computed_torque":
                A, omega = 0.05, 1.0
                q_des = A * np.sin(omega * t) * np.ones(model.nq)
                qdot_des = A * omega * np.cos(omega * t) * np.ones(model.nq)
                qddot_des = -A * omega**2 * np.sin(omega * t) * np.ones(model.nq)
                tau = controller.compute(
                    q, qdot, q_des=q_des, qdot_des=qdot_des, qddot_des=qddot_des
                )
            else:  # operational_space
                jac = controller.jacobian
                jac.set_q(q)
                R_cur = jac.data.site_xmat[jac.ee_id].reshape(3, 3).copy()
                p_cur = jac.data.site_xpos[jac.ee_id].copy()
                # Move the target in a small circle around the home pose.
                p_des = p_cur + np.array([
                    0.05 * np.sin(0.5 * t),
                    0.05 * np.cos(0.5 * t),
                    0.03 * np.sin(0.25 * t),
                ])
                R_des = R_cur @ rotation_matrix(np.array([0, 0, 1]), 0.2 * np.sin(0.3 * t))
                tau = controller.compute(q, qdot, R_des=R_des, p_des=p_des)

            data.ctrl[:] = tau
            mujoco.mj_step(model, data)
            viewer.sync()

            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
