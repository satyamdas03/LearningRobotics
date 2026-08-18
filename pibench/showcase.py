"""Render a visual showcase of LearningRobotics Chapters 1-5 + PIBench Phases 0-4.

Outputs PNG thumbnails to output/showcase/ with auto-framed cameras.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "chapter05_inverse_kinematics"))
from inverse_kinematics import InverseKinematics  # noqa: E402


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
