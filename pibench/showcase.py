"""Render a visual showcase of LearningRobotics Chapters 1-8 + PIBench Phases 0-7.

Outputs PNG thumbnails to output/showcase/ with auto-framed cameras.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "chapter05_inverse_kinematics"))
sys.path.insert(0, str(Path(__file__).parent.parent / "chapter06_dynamics"))
sys.path.insert(0, str(Path(__file__).parent.parent / "chapter04_velocity_kinematics"))
sys.path.insert(0, str(Path(__file__).parent.parent / "chapter07_control"))
sys.path.insert(0, str(Path(__file__).parent.parent / "chapter08_motion_planning"))
from inverse_kinematics import InverseKinematics  # noqa: E402
from dynamics import ArmDynamics  # noqa: E402
from jacobian import ArmJacobian  # noqa: E402
from control import (  # noqa: E402
    ComputedTorqueController,
    GravityCompensationController,
    JointSpacePIDController,
    OperationalSpaceController,
)
from collision import ArmPlanningEnv  # noqa: E402
from planners import RRTStarPlanner  # noqa: E402


OUTPUT_DIR = Path(__file__).parent / "output" / "showcase"


def _frame_camera(model: mujoco.MjModel, data: mujoco.MjData) -> mujoco.MjvCamera:
    """Place a camera so the whole scene fits in view."""
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING

    # Compute the world-space AABB of all non-plane geoms.
    centers = []
    sizes = []
    for i in range(model.ngeom):
        gtype = model.geom_type[i]
        if gtype == mujoco.mjtGeom.mjGEOM_PLANE:
            continue
        pos = model.geom_pos[i]
        size = model.geom_size[i]
        centers.append(np.array(pos))
        sizes.append(np.array(size))

    if centers:
        centers = np.array(centers)
        sizes = np.array(sizes)
        bbox_min = np.min(centers - sizes, axis=0)
        bbox_max = np.max(centers + sizes, axis=0)
        lookat = (bbox_min + bbox_max) / 2.0
        extent = np.max(bbox_max - bbox_min)
    else:
        lookat = np.zeros(3)
        extent = 2.0

    # Pick the body whose geoms are closest to the scene center to track.
    # Body 0 is the world body; skip it if there are other bodies.
    best_body = 0
    best_dist = float("inf")
    for body_id in range(model.nbody):
        if body_id == 0 and model.nbody > 1:
            continue
        # Body position from forward kinematics.
        body_pos = data.xpos[body_id].copy()
        d = float(np.linalg.norm(body_pos - lookat))
        if d < best_dist:
            best_dist = d
            best_body = body_id

    camera.trackbodyid = best_body
    camera.distance = max(extent * 1.8, 1.0)
    camera.azimuth = 120.0
    camera.elevation = -20.0
    camera.lookat[:] = lookat
    return camera


def _render_model(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    out: Path,
    height: int = 480,
    width: int = 640,
) -> Path:
    """Render a framed view of the given model/data and save it as PNG."""
    from PIL import Image

    mujoco.mj_forward(model, data)
    camera = _frame_camera(model, data)
    renderer = mujoco.Renderer(model, height=height, width=width)
    renderer.update_scene(data, camera=camera)
    img = renderer.render()
    Image.fromarray(img).save(out)
    return out


def render_arm_default() -> Path:
    """Render the simple_6dof_arm.xml in its default configuration."""
    xml_path = Path(__file__).parent.parent / "chapter01_foundation" / "simple_6dof_arm.xml"
    with open(xml_path, "r", encoding="utf-8") as f:
        model = mujoco.MjModel.from_xml_string(f.read())
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    out = OUTPUT_DIR / "arm_default.png"
    return _render_model(model, data, out)


def render_arm_ik() -> Path:
    """Render the 6-DOF arm at a pose reached by numeric IK."""
    ik = InverseKinematics()
    q0 = np.zeros(6)
    p_target = np.array([0.60, 0.20, 0.60])
    R_target = np.eye(3)
    q, _info = ik.ik_numeric(q0, R_target, p_target, max_iters=300)
    ik.set_q(q)
    out = OUTPUT_DIR / "arm_ik_solution.png"
    return _render_model(ik.model, ik.data, out)


def render_arm_dynamics() -> Path:
    """Render the 6-DOF arm under gravity compensation at a non-trivial pose."""
    dyn = ArmDynamics()
    q = np.array([0.2, -0.3, 0.5, 0.0, 0.2, -0.1])
    dyn.set_state(q, np.zeros(dyn.model.nq))
    out = OUTPUT_DIR / "arm_dynamics_pose.png"
    return _render_model(dyn.model, dyn.data, out)


def _make_arm() -> ArmDynamics:
    """Return an ArmDynamics instance with gravity and generous torque limits."""
    dyn = ArmDynamics()
    dyn.model.opt.gravity[:] = np.array([0.0, 0.0, -9.81])
    dyn.model.actuator_ctrlrange[:, 0] = -200.0
    dyn.model.actuator_ctrlrange[:, 1] = 200.0
    return dyn


def _inertia_scaled_gains(arm: ArmDynamics, omega: float) -> tuple[np.ndarray, np.ndarray]:
    """Return joint-space PD gains scaled by the home-configuration inertia."""
    M_diag = np.diag(arm.mass_matrix(np.zeros(arm.model.nq)))
    Kp = omega**2 * M_diag
    Kd = 2.0 * np.sqrt(Kp * M_diag)
    return Kp, Kd


def _ik_target() -> tuple[np.ndarray, np.ndarray]:
    """Solve for a joint configuration that reaches a visually interesting pose."""
    ik = InverseKinematics()
    p_target = np.array([0.55, 0.10, 0.45])
    R_target = np.eye(3)
    q0 = np.array([0.1, -0.2, 0.3, 0.0, 0.1, -0.1])
    q, _info = ik.ik_numeric(q0, R_target, p_target, max_iters=500)
    return q, ik


def render_arm_gravity_comp() -> Path:
    """Render the arm held in place by gravity-compensation torques."""
    dyn = _make_arm()
    q = np.array([0.3, -0.4, 0.5, 0.0, 0.2, -0.1])
    dyn.set_state(q, np.zeros(dyn.model.nq))
    dyn.data.ctrl[:] = GravityCompensationController(dyn).compute(q)
    mujoco.mj_step(dyn.model, dyn.data)
    out = OUTPUT_DIR / "arm_gravity_comp.png"
    return _render_model(dyn.model, dyn.data, out)


def _simulate_to_pose(
    dyn: ArmDynamics,
    controller,
    q: np.ndarray,
    qdot: np.ndarray,
    dt: float = 0.01,
    steps: int = 1200,
    **kwargs,
) -> None:
    """Run a controller for a fixed number of steps and leave dyn at the final state."""
    for _ in range(steps):
        tau = controller.compute(q, qdot, **kwargs)
        dyn.set_state(q, qdot)
        dyn.data.ctrl[:] = tau
        mujoco.mj_step(dyn.model, dyn.data)
        q = dyn.data.qpos.copy()
        qdot = dyn.data.qvel.copy()
    dyn.set_state(q, qdot)


def render_arm_pid() -> Path:
    """Render the arm at a set-point reached by joint-space PID + gravity."""
    dyn = _make_arm()
    q_des, _ik = _ik_target()
    q = np.zeros(dyn.model.nq)
    qdot = np.zeros(dyn.model.nq)
    Kp, Kd = _inertia_scaled_gains(dyn, omega=8.0)
    ctrl = JointSpacePIDController(dyn, Kp=Kp, Kd=Kd, gravity_comp=True)
    _simulate_to_pose(dyn, ctrl, q, qdot, q_des=q_des, dt=0.01, steps=1500)
    out = OUTPUT_DIR / "arm_pid.png"
    return _render_model(dyn.model, dyn.data, out)


def render_arm_computed_torque() -> Path:
    """Render the arm tracking a static pose via computed-torque linearization."""
    dyn = _make_arm()
    q_des, _ik = _ik_target()
    q = np.zeros(dyn.model.nq)
    qdot = np.zeros(dyn.model.nq)
    Kp = np.full(dyn.model.nq, 80.0)
    Kd = np.full(dyn.model.nq, 18.0)
    ctrl = ComputedTorqueController(dyn, Kp=Kp, Kd=Kd)
    _simulate_to_pose(dyn, ctrl, q, qdot, q_des=q_des, qdot_des=np.zeros(dyn.model.nq), dt=0.01, steps=1200)
    out = OUTPUT_DIR / "arm_computed_torque.png"
    return _render_model(dyn.model, dyn.data, out)


def render_arm_operational_space() -> Path:
    """Render the arm reaching a target pose via resolved-acceleration control."""
    dyn = _make_arm()
    jac = ArmJacobian()
    q0 = np.zeros(dyn.model.nq)
    qdot0 = np.zeros(dyn.model.nq)
    dyn.set_state(q0, qdot0)
    jac.set_q(q0)
    p_target = np.array([0.55, 0.10, 0.45])
    R_target = np.eye(3)
    Kp_task = np.array([60.0, 60.0, 60.0, 30.0, 30.0, 30.0])
    Kd_task = np.array([14.0, 14.0, 14.0, 8.0, 8.0, 8.0])
    ctrl = OperationalSpaceController(dyn, jac, Kp=Kp_task, Kd=Kd_task, k_null=0.2)
    _simulate_to_pose(dyn, ctrl, q0, qdot0, R_des=R_target, p_des=p_target, dt=0.01, steps=1200)
    out = OUTPUT_DIR / "arm_operational_space.png"
    return _render_model(dyn.model, dyn.data, out)


def render_arm_motion_planning() -> Path:
    """Render a collision-free RRT* goal configuration among obstacles."""
    obstacles = [
        {"name": "wall", "type": "box", "pos": [0.78, 0.0, 0.35], "size": [0.05, 0.25, 0.15]},
        {"name": "shelf", "type": "box", "pos": [0.78, 0.25, 0.5], "size": [0.05, 0.1, 0.05]},
        {"name": "floor_obstacle", "type": "box", "pos": [0.78, -0.25, 0.15], "size": [0.05, 0.1, 0.05]},
    ]
    env = ArmPlanningEnv(obstacles=obstacles)
    start = np.zeros(6)
    rng = np.random.default_rng(7)
    q_goal = env.sample_collision_free(rng)
    if q_goal is None:
        q_goal = np.array([0.5, -0.4, 0.7, 0.0, 0.0, 0.0])
    planner = RRTStarPlanner(max_iters=800, seed=7)
    path = planner.plan(start, q_goal, env)
    if path is not None:
        env.set_state(path[-1])
    else:
        env.set_state(q_goal)
    out = OUTPUT_DIR / "arm_motion_planning.png"
    return _render_model(env.model, env.data, out)


def render_scene(name: str) -> Path | None:
    """Render a PIBench scene with auto-framed camera."""
    out = OUTPUT_DIR / f"{name.lower()}_seed0.png"

    # Import the scene class directly so we can build it and render.
    from pibench.core.registry import list_problems

    problem_classes = {cls.__name__: cls for cls in list_problems()}
    if name not in problem_classes:
        print(f"Warning: unknown scene {name}")
        return None

    problem = problem_classes[name](seed=0)
    if not hasattr(problem, "model") or not hasattr(problem, "data"):
        print(f"Warning: {name} has no MuJoCo model/data to render")
        return None
    _render_model(problem.model, problem.data, out)
    return out


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Clear old showcase images.
    for p in OUTPUT_DIR.glob("*.png"):
        p.unlink()

    images: dict[str, Path] = {}
    images["Chapter 1 — 6-DOF arm default"] = render_arm_default()
    images["Chapter 5 — IK target solution"] = render_arm_ik()
    images["Chapter 6 — Dynamics pose"] = render_arm_dynamics()
    images["Chapter 7 — Gravity compensation"] = render_arm_gravity_comp()
    images["Chapter 7 — Joint-space PID"] = render_arm_pid()
    images["Chapter 7 — Computed torque"] = render_arm_computed_torque()
    images["Chapter 7 — Operational space"] = render_arm_operational_space()
    images["Chapter 8 — Motion planning"] = render_arm_motion_planning()

    scenes = [
        "TowerFall",
        "SlopeSlide",
        "SupportBalance",
        "ToppleDirection",
        "PendulumSwing",
        "CollisionBounce",
        "ProjectileHit",
        "PushTipVsSlide",
        "StackStability",
        "WedgeInsert",
        "FrictionPile",
        "SlipGrip",
        "DrawerPull",
        "DoorSwing",
        "RopeTension",
        "GearTurn",
        "ChainDrape",
        "MassOrder",
        "FrictionOrder",
        "CounterfactualMass",
        "CounterfactualFriction",
        "BalanceAfterMove",
    ]
    for scene in scenes:
        path = render_scene(scene)
        if path:
            images[f"PIBench — {scene}"] = path

    print("Showcase images written to:", OUTPUT_DIR)
    for label, path in images.items():
        print(f"  {label}: {path}")


if __name__ == "__main__":
    main()
