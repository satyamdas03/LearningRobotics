"""Helpers for articulated and deformable PIBench scenes.

Provides reusable MJCF fragments for prismatic joints, hinge joints, tendons,
and coarse deformable bodies approximated as capsule chains.
"""
from __future__ import annotations

import mujoco
import numpy as np


def mjcf_prismatic(
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    rgba: tuple[float, float, float, float] = (0.6, 0.4, 0.2, 1.0),
    mass: float | None = None,
    range_limits: tuple[float, float] = (-1.0, 1.0),
    friction: tuple[float, float, float] | None = None,
) -> str:
    """Return a body with a prismatic (slide) joint.

    The body is a box that slides along ``axis`` relative to its parent.
    Range limits keep the joint from moving outside ``range_limits`` (metres).
    """
    ax = " ".join(str(v) for v in axis)
    friction_attr = ""
    if friction is not None:
        friction_attr = f' friction="{friction[0]} {friction[1]} {friction[2]}"'
    inertial = ""
    if mass is not None:
        inertial = f'\n      <inertial pos="0 0 0" mass="{mass}" diaginertia="{mass/12*(size[1]**2+size[2]**2)} {mass/12*(size[0]**2+size[2]**2)} {mass/12*(size[0]**2+size[1]**2)}" />'
    return f"""    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}"{inertial}>
      <joint name="{name}_joint" type="slide" axis="{ax}" range="{range_limits[0]} {range_limits[1]}" damping="0.05" />
      <geom name="{name}_geom" type="box" size="{size[0]} {size[1]} {size[2]}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"{friction_attr} />
    </body>
"""


def mjcf_hinge(
    name: str,
    pos: tuple[float, float, float],
    size: tuple[float, float, float],
    axis: tuple[float, float, float] = (0.0, 1.0, 0.0),
    rgba: tuple[float, float, float, float] = (0.5, 0.5, 0.7, 1.0),
    mass: float | None = None,
    range_limits: tuple[float, float] = (-3.14, 3.14),
    friction: tuple[float, float, float] | None = None,
) -> str:
    """Return a body with a revolute (hinge) joint.

    The hinge axis passes through the body origin.  Range limits are given in
    radians.
    """
    ax = " ".join(str(v) for v in axis)
    friction_attr = ""
    if friction is not None:
        friction_attr = f' friction="{friction[0]} {friction[1]} {friction[2]}"'
    inertial = ""
    if mass is not None:
        inertial = f'\n      <inertial pos="0 0 0" mass="{mass}" diaginertia="{mass/12*(size[1]**2+size[2]**2)} {mass/12*(size[0]**2+size[2]**2)} {mass/12*(size[0]**2+size[1]**2)}" />'
    return f"""    <body name="{name}" pos="{pos[0]} {pos[1]} {pos[2]}"{inertial}>
      <joint name="{name}_joint" type="hinge" axis="{ax}" range="{range_limits[0]} {range_limits[1]}" damping="0.05" />
      <geom name="{name}_geom" type="box" size="{size[0]} {size[1]} {size[2]}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"{friction_attr} />
    </body>
"""


def mjcf_tendon(
    name: str,
    body_a: str,
    body_b: str,
    anchor_a: tuple[float, float, float] = (0.0, 0.0, 0.0),
    anchor_b: tuple[float, float, float] = (0.0, 0.0, 0.0),
    rest_length: float | None = None,
    stiffness: float = 1000.0,
    damping: float = 10.0,
    rgba: tuple[float, float, float, float] = (0.9, 0.7, 0.2, 0.8),
) -> tuple[str, str]:
    """Return (site_xml, tendon_xml) for a slack-taut tendon connecting two bodies.

    The tendon is a MuJoCo spatial tendon between two sites.  It has zero
    stiffness below the rest length (slack) and ``stiffness`` when stretched
    beyond it.  If ``rest_length`` is None the current distance between anchors
    is used, so the tendon starts taut.
    """
    site_a = f"{name}_site_a"
    site_b = f"{name}_site_b"
    site_xml = f"""    <site name="{site_a}" body="{body_a}" pos="{anchor_a[0]} {anchor_a[1]} {anchor_a[2]}" size="0.01" />
    <site name="{site_b}" body="{body_b}" pos="{anchor_b[0]} {anchor_b[1]} {anchor_b[2]}" size="0.01" />
"""
    # Use a large lower bound (0) and the rest length as the upper bound so the
    # tendon only resists extension beyond the rest length.
    length_attr = ""
    if rest_length is not None:
        length_attr = f' range="0 {rest_length}"'
    tendon_xml = f"""    <tendon>
      <spatial name="{name}" site="{site_a} {site_b}" stiffness="{stiffness}" damping="{damping}" rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"{length_attr} />
    </tendon>
"""
    return site_xml, tendon_xml


def mjcf_capsule_chain(
    name: str,
    start_pos: tuple[float, float, float],
    n_capsules: int,
    capsule_radius: float,
    capsule_half_len: float,
    spacing: float | None = None,
    axis: tuple[float, float, float] = (1.0, 0.0, 0.0),
    rgba: tuple[float, float, float, float] = (0.4, 0.6, 0.8, 1.0),
    mass_per_capsule: float = 0.05,
) -> str:
    """Return worldbody XML for a coarse deformable capsule chain.

    The chain is built from nested capsules connected by ball joints.  The
    first segment is fixed to the world; later segments hang freely.  This
    approximates a deformable rope/cable without using MuJoCo's composite
    bodies.  Capsules are laid out along ``axis`` starting from ``start_pos``.
    If ``spacing`` is None it is set to ``2*capsule_half_len`` so the capsules
    just meet at the joints.
    """
    if spacing is None:
        spacing = 2.0 * capsule_half_len
    ax = np.array(axis, dtype=float)
    ax = ax / np.linalg.norm(ax)
    s_pos = np.array(start_pos, dtype=float)
    L = 2.0 * capsule_half_len
    r = capsule_radius
    m = mass_per_capsule

    def indent(level: int) -> str:
        return " " * (4 + 2 * level)

    offset = spacing * ax
    offset_str = f"{offset[0]} {offset[1]} {offset[2]}"
    first_pos = " ".join(str(v) for v in start_pos)

    # Solid-cylinder inertia approximation for each capsule.
    i_x = 0.5 * m * r * r
    i_y = i_z = m / 12.0 * (3.0 * r * r + L * L)

    lines: list[str] = []
    # Root segment is static (no joint) so the chain has a fixed anchor.
    lines.append(f'{indent(0)}<body name="{name}_seg0" pos="{first_pos}">')
    lines.append(f'{indent(1)}<inertial pos="0 0 0" mass="{m}" diaginertia="{i_x} {i_y} {i_z}" />')
    lines.append(
        f'{indent(1)}<geom name="{name}_seg0_geom" type="capsule" '
        f'fromto="-{capsule_half_len} 0 0 {capsule_half_len} 0 0" size="{capsule_radius}" '
        f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" />'
    )

    # Nested capsules hanging from the previous one.
    for i in range(1, n_capsules):
        level = i
        lines.append(f'{indent(level)}<body name="{name}_seg{i}" pos="{offset_str}">')
        lines.append(f'{indent(level+1)}<joint name="{name}_joint{i}" type="ball" damping="0.01" />')
        lines.append(f'{indent(level+1)}<inertial pos="0 0 0" mass="{m}" diaginertia="{i_x} {i_y} {i_z}" />')
        lines.append(
            f'{indent(level+1)}<geom name="{name}_seg{i}_geom" type="capsule" '
            f'fromto="-{capsule_half_len} 0 0 {capsule_half_len} 0 0" size="{capsule_radius}" '
            f'rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}" />'
        )

    # Close all body tags in reverse order.
    for i in range(n_capsules - 1, -1, -1):
        lines.append(f'{indent(i)}</body>')

    return "\n".join(lines)


def body_id(model: mujoco.MjModel, name: str) -> int:
    """Return the integer body id for a named body."""
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def joint_position(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str) -> float:
    """Return the current scalar joint position (angle or displacement)."""
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    qpos_addr = model.jnt_qposadr[jid]
    return float(data.qpos[qpos_addr])


def body_displacement(
    model: mujoco.MjModel, data: mujoco.MjData, body_name: str, initial_pos: np.ndarray
) -> float:
    """Return the Euclidean distance a body has moved since ``initial_pos``."""
    bid = body_id(model, body_name)
    return float(np.linalg.norm(data.xpos[bid] - initial_pos))
