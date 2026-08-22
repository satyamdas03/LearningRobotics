"""Chapter 8 — Motion Planning: interactive viewer demo.

Plans a collision-free joint-space path for the 6-DOF arm around static
obstacles and plays the resulting trajectory in the MuJoCo passive viewer.
"""
from __future__ import annotations

import time

import mujoco
import numpy as np

from collision import ArmPlanningEnv, make_obstacle_xml
from planners import RRTStarPlanner, path_length
from smoother import cubic_bspline_interpolate, shortcut_smooth


OBSTACLES = [
    {"type": "box", "pos": [0.78, 0.0, 0.40], "size": [0.05, 0.45, 0.45]},
    {"type": "sphere", "pos": [0.62, 0.35, 0.50], "size": [0.12]},
    {"type": "box", "pos": [0.58, -0.35, 0.35], "size": [0.18, 0.05, 0.35]},
]


def _find_collision_free_goal(env: ArmPlanningEnv, rng: np.random.Generator) -> np.ndarray:
    """Sample a goal config that is collision-free and visually different from home."""
    for _ in range(5000):
        q = rng.uniform(env.q_min, env.q_max)
        if not env.is_collision(q):
            # Prefer configurations where the arm reaches out into the workspace.
            if q[1] < -0.2 or q[2] > 0.3:
                return q
    raise RuntimeError("Could not find a collision-free goal configuration")


def main() -> None:
    rng = np.random.default_rng(7)
    env = ArmPlanningEnv(obstacles=OBSTACLES)
    start = np.zeros(env.model.nq)
    goal = _find_collision_free_goal(env, rng)

    print("Planning from:", np.round(start, 3))
    print("Planning to:  ", np.round(goal, 3))

    planner = RRTStarPlanner(
        step_size=0.12, goal_bias=0.15, max_iters=3000, seed=7
    )
    path = planner.plan(start, goal, env)
    if path is None:
        print("Planner failed to find a path. Try increasing max_iters or relaxing obstacles.")
        return

    print(f"Raw path length: {path_length(path):.3f} rad (nodes={len(path)})")
    path = shortcut_smooth(path, env, max_iters=300, seed=7)
    print(f"After shortcut: {path_length(path):.3f} rad (nodes={len(path)})")

    dense = cubic_bspline_interpolate(path, n_points=400)
    print(f"Dense trajectory: {len(dense)} points")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        # Ensure obstacles render from the start.
        env.set_state(start)
        while viewer.is_running():
            for q in dense:
                env.set_state(q)
                viewer.sync()
                time.sleep(0.015)
            # Pause briefly at the goal, then loop.
            time.sleep(1.0)


if __name__ == "__main__":
    main()
