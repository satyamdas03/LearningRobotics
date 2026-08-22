"""Chapter 9 — Trajectory generation: plan → trajectory → controller demo.

Plans a collision-free joint-space path with the Chapter 8 RRT* planner,
converts it to a timed quintic trajectory with trapezoidal time scaling, then
tracks it with a Chapter 7 joint-space PID controller in the MuJoCo viewer.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import mujoco
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "chapter08_motion_planning"))
sys.path.insert(0, str(ROOT / "chapter07_control"))
sys.path.insert(0, str(ROOT / "chapter06_dynamics"))
sys.path.insert(0, str(ROOT / "chapter09_trajectory_generation"))

from collision import ArmPlanningEnv, make_obstacle_xml
from control import JointSpacePIDController
from dynamics import ArmDynamics
from path_to_trajectory import plan_to_timed_trajectory
from planners import RRTStarPlanner, path_length
from smoother import shortcut_smooth


OBSTACLES = [
    {"type": "box", "pos": [0.78, 0.0, 0.40], "size": [0.05, 0.45, 0.45]},
    {"type": "sphere", "pos": [0.62, 0.35, 0.50], "size": [0.12]},
    {"type": "box", "pos": [0.58, -0.35, 0.35], "size": [0.18, 0.05, 0.35]},
]


def _find_collision_free_goal(env: ArmPlanningEnv, rng: np.random.Generator) -> np.ndarray:
    """Sample a goal config that is collision-free and visually different."""
    for _ in range(5000):
        q = rng.uniform(env.q_min, env.q_max)
        if not env.is_collision(q):
            if q[1] < -0.2 or q[2] > 0.3:
                return q
    raise RuntimeError("Could not find a collision-free goal configuration")


def _inertia_scaled_gains(arm: ArmDynamics, omega: float) -> tuple[np.ndarray, np.ndarray]:
    """Return joint-space PD gains scaled by the home-configuration inertia."""
    M_diag = np.diag(arm.mass_matrix(np.zeros(arm.model.nq)))
    Kp = omega**2 * M_diag
    Kd = 2.0 * np.sqrt(Kp * M_diag)
    return Kp, Kd


def main() -> None:
    rng = np.random.default_rng(7)
    env = ArmPlanningEnv(obstacles=OBSTACLES)
    start = np.zeros(env.model.nq)
    goal = _find_collision_free_goal(env, rng)

    print("Planning from:", np.round(start, 3))
    print("Planning to:  ", np.round(goal, 3))

    planner = RRTStarPlanner(step_size=0.12, goal_bias=0.15, max_iters=3000, seed=7)
    path = planner.plan(start, goal, env)
    if path is None:
        print("Planner failed to find a path.")
        return

    print(f"Raw path length: {path_length(path):.3f} rad (nodes={len(path)})")
    path = shortcut_smooth(path, env, max_iters=300, seed=7)
    print(f"After shortcut: {path_length(path):.3f} rad (nodes={len(path)})")

    # Build a timed trajectory with bounded joint velocity/acceleration.
    traj = plan_to_timed_trajectory(
        path,
        max_joint_velocity=0.8,
        max_joint_acceleration=2.0,
        order="quintic",
    )
    print(f"Trajectory duration: {traj.t_total:.3f} s")

    # Controller that tracks the trajectory on a clean dynamics model.
    dyn = ArmDynamics()
    # Use the planning XML (with obstacles) for visualization/physics.
    xml = make_obstacle_xml(OBSTACLES)
    dyn.model = mujoco.MjModel.from_xml_string(xml)
    dyn.data = mujoco.MjData(dyn.model)
    mujoco.mj_resetData(dyn.model, dyn.data)

    Kp, Kd = _inertia_scaled_gains(dyn, omega=8.0)
    ctrl = JointSpacePIDController(dyn, Kp=Kp, Kd=Kd, gravity_comp=True)

    env.set_state(start)
    with mujoco.viewer.launch_passive(dyn.model, dyn.data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t_elapsed = time.time() - t0
            if t_elapsed >= traj.t_total:
                # Pause at the goal, then restart.
                time.sleep(1.0)
                t0 = time.time()
                env.set_state(start)
                mujoco.mj_resetData(dyn.model, dyn.data)
                continue

            q_des = traj.evaluate(t_elapsed)
            qdot_des = traj.evaluate_velocity(t_elapsed)
            q = dyn.data.qpos.copy()
            qdot = dyn.data.qvel.copy()

            tau = ctrl.compute(q, qdot, q_des=q_des, qdot_des=qdot_des, dt=0.01)
            dyn.data.ctrl[:] = tau
            mujoco.mj_step(dyn.model, dyn.data)
            viewer.sync()
            time.sleep(dyn.model.opt.timestep)


if __name__ == "__main__":
    main()
