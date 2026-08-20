"""
Chapter 6 — Interactive demo: dynamics in the MuJoCo viewer.

The 6-DOF arm is shown in two modes that auto-toggle every few seconds:
  1. Gravity compensation: torques cancel gravity so the arm stays still.
  2. Free fall under gravity: zero torques, arm collapses downward.

The current mode and joint acceleration norm are printed to the console.
"""
from __future__ import annotations

import time

import mujoco
import mujoco.viewer
import numpy as np

from dynamics import ArmDynamics


def main() -> None:
    dyn = ArmDynamics()
    model = dyn.model
    data = dyn.data

    # Start in gravity-compensation mode.
    gravity_comp = True
    q = np.zeros(model.nq)
    qdot = np.zeros(model.nq)
    dyn.set_state(q, qdot)

    toggle_interval = 3.0
    last_toggle = time.time()

    print("Demo auto-toggles between gravity compensation and free fall every 3 seconds.")
    print("Close the viewer window to exit.")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            step_start = time.time()

            if time.time() - last_toggle > toggle_interval:
                gravity_comp = not gravity_comp
                last_toggle = time.time()
                mode = "GRAVITY COMP" if gravity_comp else "FREE FALL"
                print(f"Mode: {mode}")

            if gravity_comp:
                tau = dyn.gravity_term(data.qpos.copy())
            else:
                tau = np.zeros(model.nq)

            # Apply torques and step MuJoCo.
            data.qfrc_applied[:] = tau
            mujoco.mj_step(model, data)

            viewer.sync()
            elapsed = time.time() - step_start
            if elapsed < model.opt.timestep:
                time.sleep(model.opt.timestep - elapsed)


if __name__ == "__main__":
    main()
