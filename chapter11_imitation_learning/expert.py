"""Chapter 11 — Expert demonstration generation for imitation learning.

An "expert" here is a planner + simulator that produces deterministic joint-space
trajectories for reach tasks on the 6-DOF arm from Chapter 10.  The resulting
demonstrations are stored as ``(state, action, next_state, goal)`` tuples that
can be used to train a behavior-cloning policy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from chapter05_inverse_kinematics.inverse_kinematics import InverseKinematics


DEFAULT_ARM_XML = Path(__file__).parent.parent / "chapter10_perception" / "arm.xml"


def _arm_qpos_addrs(model: mujoco.MjModel) -> np.ndarray:
    """Return the qpos addresses for the 6 arm joints, sorted by joint index."""
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
        for i in range(1, 7)
    ]
    return np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=int)


def _read_arm_q(data: mujoco.MjData, qpos_addrs: np.ndarray) -> np.ndarray:
    return np.array(data.qpos[qpos_addrs], dtype=float)


def _set_arm_q(data: mujoco.MjData, qpos_addrs: np.ndarray, q: np.ndarray) -> None:
    data.qpos[qpos_addrs] = q


def _ee_pose(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[np.ndarray, np.ndarray]:
    """Return current end-effector (R, p) from MuJoCo."""
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")
    R = data.site_xmat[ee_id].reshape(3, 3).copy()
    p = data.site_xpos[ee_id].copy()
    return R, p


def _cubic_interpolation_waypoints(
    q0: np.ndarray,
    q1: np.ndarray,
    duration: float,
    dt: float,
) -> np.ndarray:
    """Return ``T x nq`` waypoints following a zero-velocity cubic spline."""
    n_steps = max(2, int(np.round(duration / dt)) + 1)
    t = np.linspace(0.0, duration, n_steps)
    s = t / duration
    # Cubic Hermite with zero boundary velocities: h(s) = 3s^2 - 2s^3
    h = 3.0 * s**2 - 2.0 * s**3
    return q0 + (q1 - q0) * h[:, None]


def _solve_ik_for_position(
    xml_path: str,
    q0: np.ndarray,
    p_target: np.ndarray,
    R_target: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """Use Chapter 5 numeric IK to find a goal configuration for ``p_target``."""
    ik = InverseKinematics(xml_path)
    R_target = np.eye(3) if R_target is None else np.asarray(R_target, dtype=float)
    return ik.ik_numeric(
        q0=q0,
        R_target=R_target,
        p_target=p_target,
        position_only=True,
        max_iters=300,
        tol=1e-5,
        damping=0.05,
        step_scale=0.5,
    )


def record_reach_trajectory(
    xml_path: str | Path,
    start_q: np.ndarray,
    goal_ee_position: np.ndarray,
    duration: float = 2.0,
    dt: float = 0.01,
    R_target: np.ndarray | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Record one expert reach trajectory.

    Returns a dictionary with keys:
      - ``states``:      ``(T, 6)`` joint positions actually reached.
      - ``actions``:     ``(T, 6)`` target next-step joint positions.
      - ``next_states``: ``(T, 6)`` one-step-ahead joint positions.
      - ``ee_positions``: ``(T, 3)`` end-effector positions.
      - ``goal_ee``:     ``(3,)`` target end-effector position.
      - ``goal_q``:      ``(6,)`` IK solution used as joint target.
      - ``dt``:          controller timestep.
    """
    xml_path = str(xml_path)
    start_q = np.asarray(start_q, dtype=float)
    goal_ee_position = np.asarray(goal_ee_position, dtype=float)

    if R_target is None:
        R_target = np.eye(3)

    goal_q, ik_info = _solve_ik_for_position(xml_path, start_q, goal_ee_position, R_target)
    if not ik_info["converged"]:
        # Fall back to the best IK result even if tolerance was not met.
        pass

    waypoints = _cubic_interpolation_waypoints(start_q, goal_q, duration, dt)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    qpos_addrs = _arm_qpos_addrs(model)

    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    next_states: list[np.ndarray] = []
    ee_positions: list[np.ndarray] = []

    # The expert is treated as a perfect kinematic planner: we set the joint
    # positions directly and run forward kinematics.  This isolates the
    # imitation-learning problem from low-level actuator dynamics.
    for i in range(len(waypoints) - 1):
        q_current = waypoints[i]
        q_next_target = waypoints[i + 1]

        states.append(q_current.copy())
        actions.append(q_next_target.copy())
        next_states.append(q_next_target.copy())

        _set_arm_q(data, qpos_addrs, q_next_target)
        mujoco.mj_forward(model, data)
        _, p_ee = _ee_pose(model, data)
        ee_positions.append(p_ee)

    return {
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "next_states": np.asarray(next_states),
        "ee_positions": np.asarray(ee_positions),
        "goal_ee": goal_ee_position,
        "goal_q": goal_q,
        "dt": float(dt),
        "ik_info": ik_info,
    }


def generate_reach_dataset(
    xml_path: str | Path,
    n_demos: int = 10,
    duration: float = 2.0,
    dt: float = 0.01,
    seed: int | None = 0,
    start_q: np.ndarray | None = None,
) -> list[dict[str, Any]]:
    """Generate a dataset of ``n_demos`` reach demonstrations.

    Goal end-effector positions are produced by sampling random joint
    configurations, running forward kinematics, and then using those positions
    as reach targets.  This guarantees that every target is kinematically
    reachable from the arm model.
    """
    xml_path = str(xml_path)
    rng = np.random.default_rng(seed)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    qpos_addrs = _arm_qpos_addrs(model)

    if start_q is None:
        start_q = np.zeros(6)

    # Build random reachable EE goals by sampling joint configs and reading FK.
    goal_positions: list[np.ndarray] = []
    attempts = 0
    while len(goal_positions) < n_demos and attempts < n_demos * 20:
        attempts += 1
        q_sample = np.zeros(6)
        for i, jid in enumerate(
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{j}")
            for j in range(1, 7)
        ):
            lo, hi = model.jnt_range[jid]
            q_sample[i] = rng.uniform(max(lo, -3.0), min(hi, 3.0))

        _set_arm_q(data, qpos_addrs, q_sample)
        mujoco.mj_forward(model, data)
        _, p_goal = _ee_pose(model, data)

        # Keep goals that are reasonably above the ground and in front of arm.
        if p_goal[2] > 0.3 and p_goal[0] > 0.2 and np.linalg.norm(p_goal[:2]) > 0.15:
            goal_positions.append(p_goal.copy())

    demos: list[dict[str, Any]] = []
    for p_goal in goal_positions[:n_demos]:
        demo = record_reach_trajectory(
            xml_path=xml_path,
            start_q=start_q,
            goal_ee_position=p_goal,
            duration=duration,
            dt=dt,
            seed=seed,
        )
        demos.append(demo)

    return demos


def save_dataset(demos: list[dict[str, Any]], path: str | Path) -> None:
    """Save a list of demonstrations to a JSON file (arrays become lists)."""
    path = Path(path)
    serializable = []
    for demo in demos:
        entry = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in demo.items()}
        # Convert nested ik_info arrays too.
        if "ik_info" in entry:
            entry["ik_info"] = {
                k: (v.tolist() if isinstance(v, np.ndarray) else v)
                for k, v in demo["ik_info"].items()
            }
        serializable.append(entry)
    path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")


def load_dataset(path: str | Path) -> list[dict[str, Any]]:
    """Load demonstrations from JSON and restore NumPy arrays."""
    path = Path(path)
    raw = json.loads(path.read_text(encoding="utf-8"))
    demos: list[dict[str, Any]] = []
    for entry in raw:
        demo = {}
        for k, v in entry.items():
            if k == "ik_info":
                demo[k] = {
                    kk: (np.asarray(vv, dtype=float) if isinstance(vv, list) else vv)
                    for kk, vv in v.items()
                }
            elif isinstance(v, list):
                demo[k] = np.asarray(v, dtype=float)
            else:
                demo[k] = v
        demos.append(demo)
    return demos
