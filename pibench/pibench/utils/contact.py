"""Helpers for contact-aware PIBench scenes.

Provides reusable MJCF fragments (prismatic pushers, mesh wedges) and contact
queries that work directly with MuJoCo's data structures.
"""
from __future__ import annotations

from typing import Callable, Iterable

import mujoco
import numpy as np


def mjcf_pusher(
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    rgba: tuple[float, float, float, float] = (0.2, 0.8, 0.4, 1.0),
    max_speed: float = 1.0,
    kv: float = 2000.0,
) -> tuple[str, str]:
    """Return a (body_string, actuator_string) for a prismatic pusher.

    The body has a single slide joint and a box geom. A velocity actuator is
    provided so the caller can drive the joint by setting ``data.ctrl`` to a
    constant speed.
    """
    ax = " ".join(str(v) for v in axis)
    body = f"""    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <joint name="{name}_joint" type="slide" axis="{ax}" pos="0 0 0" damping="0.01" />
      <geom name="{name}_geom" type="box" size="{size[0]} {size[1]} {size[2]}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" />
    </body>
"""
    actuator = f"""    <velocity joint="{name}_joint" kv="{kv}" ctrlrange="-{max_speed} {max_speed}" />
"""
    return body, actuator


def mjcf_wedge_mesh(
    name: str,
    pos: tuple[float, float, float],
    length: float,
    base_width: float,
    height: float,
    rgba: tuple[float, float, float, float] = (0.7, 0.5, 0.2, 1.0),
    friction: tuple[float, float, float] | None = None,
) -> tuple[str, str, str]:
    """Return (asset_string, body_string, actuator_string) for a triangular wedge.

    The wedge is a triangular prism whose cross-section is an isosceles
    triangle in the XY plane and is extruded along Z.  The tip points in +X and
    the base (widest part) points in -X.  As the wedge is pushed in +X, the tip
    enters a gap first and the base must fit for the wedge to pass through.

    Returns a mesh asset, the free-floating wedge body referencing the mesh, and
    a placeholder actuator string (empty here because the pusher provides the
    actuator).
    """
    l = length / 2.0
    h = height / 2.0
    w = base_width / 2.0

    # Six vertices: tip edge along Z, base rectangle.
    vertices = [
        (-l, 0.0, -h),   # tip lower
        (-l, 0.0, h),    # tip upper
        (l, -w, -h),     # base lower-left
        (l, w, -h),      # base lower-right
        (l, -w, h),      # base upper-left
        (l, w, h),       # base upper-right
    ]
    vertex_attr = " ".join(f"{v[0]} {v[1]} {v[2]}" for v in vertices)

    friction_attr = ""
    if friction is not None:
        friction_attr = f' friction="{friction[0]} {friction[1]} {friction[2]}"'

    asset = f"""    <mesh name="{name}_mesh" vertex="{vertex_attr}" />
"""
    body = f"""    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}">
      <freejoint />
      <geom name="{name}_geom" type="mesh" mesh="{name}_mesh" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"{friction_attr} />
    </body>
"""
    return asset, body, ""


def body_id(model: mujoco.MjModel, name: str) -> int:
    """Return the integer body id for a named body."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def body_in_contact(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    other_name: str | None = None,
) -> bool:
    """Return True if ``body_name`` is involved in any active contact.

    If ``other_name`` is given, only contacts between the two bodies count.
    """
    target = body_id(model, body_name)
    other = body_id(model, other_name) if other_name is not None else None
    for c in data.contact[: data.ncon]:
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        if b1 == target or b2 == target:
            if other is None or {b1, b2} == {target, other}:
                return True
    return False


def body_contact_force_norm(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str,
    other_name: str | None = None,
) -> float:
    """Return the largest normal contact force magnitude involving a body."""
    target = body_id(model, body_name)
    other = body_id(model, other_name) if other_name is not None else None
    max_force = 0.0
    force = np.zeros(6)
    for i, c in enumerate(data.contact[: data.ncon]):
        b1 = int(model.geom_bodyid[c.geom1])
        b2 = int(model.geom_bodyid[c.geom2])
        if b1 != target and b2 != target:
            continue
        if other is not None and {b1, b2} != {target, other}:
            continue
        mujoco.mj_contactForce(model, data, i, force)
        # Normal force is the first component in contact-local coordinates.
        max_force = max(max_force, float(abs(force[0])))
    return max_force


def body_z_axis_deviation(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> float:
    """Angle (radians) between a body's Z axis and the world Z axis.

    Useful for deciding whether a block has tipped.
    """
    bid = body_id(model, body_name)
    # xmat is a 3x3 rotation matrix stored flat as 9 floats, column-major.
    R = data.xmat[bid].reshape(3, 3)
    body_z = R[:, 2]
    # Deviation from world +Z: sin(theta) ~= sqrt(x^2 + y^2) for small angles.
    return float(np.linalg.norm(body_z[:2]))


def run_with_pusher(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pusher_joint_name: str,
    speed: float,
    steps: int,
    callbacks: Iterable[Callable[[mujoco.MjModel, mujoco.MjData, int], None]] | None = None,
) -> None:
    """Step a simulation while driving a pusher at constant speed.

    If a velocity actuator is attached to the named joint, its control signal is
    set to ``speed`` each step.  Otherwise the joint velocity is overwritten
    directly (kinematic drive).  ``callbacks`` are called every step with
    (model, data, step_index) so the caller can record state or detect contacts.
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, pusher_joint_name)
    callbacks = list(callbacks) if callbacks is not None else []

    # Try to find a velocity actuator that drives this joint.
    actuator_id: int | None = None
    trn_joint = mujoco.mjtTrn.mjTRN_JOINT
    for i in range(model.nu):
        if model.actuator_trnid[i, 1] == trn_joint and model.actuator_trnid[i, 0] == joint_id:
            actuator_id = i
            break

    if actuator_id is not None:
        for step in range(steps):
            data.ctrl[actuator_id] = speed
            mujoco.mj_step(model, data)
            for cb in callbacks:
                cb(model, data, step)
    else:
        qvel_addr = model.jnt_dofadr[joint_id]
        for step in range(steps):
            data.qvel[qvel_addr] = speed
            mujoco.mj_step(model, data)
            for cb in callbacks:
                cb(model, data, step)
