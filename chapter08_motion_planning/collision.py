"""Chapter 8 — Motion Planning: collision checking and environment setup."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


_DEFAULT_XML = Path(__file__).parent.parent / "chapter01_foundation" / "simple_6dof_arm.xml"


def _obstacle_mjcf(obstacles: list[dict]) -> str:
    """Return an MJCF snippet for static obstacle geoms."""
    lines = ["    <!-- Planning obstacles -->"]
    for i, obs in enumerate(obstacles):
        otype = obs["type"]
        name = obs.get("name", f"obs_{i}")
        pos = " ".join(str(float(v)) for v in obs["pos"])
        size = " ".join(str(float(v)) for v in obs["size"])
        euler = obs.get("euler")
        euler_attr = "" if euler is None else f' euler="{" ".join(str(float(v)) for v in euler)}"'
        lines.append(
            f'    <body name="{name}_body" pos="{pos}">'
            f'      <geom name="{name}" type="{otype}" size="{size}"{euler_attr} rgba="0.8 0.2 0.2 0.8"/>'
            f'    </body>'
        )
    return "\n".join(lines)


def make_obstacle_xml(
    obstacles: list[dict], base_xml_path: str | Path | None = None
) -> str:
    """Read the 6-DOF arm XML and inject static obstacle bodies."""
    base_path = Path(base_xml_path) if base_xml_path else _DEFAULT_XML
    xml = base_path.read_text(encoding="utf-8")
    snippet = _obstacle_mjcf(obstacles)
    # Insert before </worldbody>.
    xml = xml.replace("  </worldbody>", f"{snippet}\n  </worldbody>")
    return xml


class ArmPlanningEnv:
    """MuJoCo-backed planning environment for the 6-DOF arm with obstacles."""

    def __init__(
        self,
        obstacles: list[dict] | None = None,
        base_xml_path: str | Path | None = None,
    ) -> None:
        obstacles = obstacles or []
        self.obstacles = obstacles
        xml = make_obstacle_xml(obstacles, base_xml_path)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Joint qpos addresses and limits.
        self.joint_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 7)
        ]
        self.qpos_addrs = [self.model.jnt_qposadr[jid] for jid in self.joint_ids]
        limits = np.array(
            [self.model.jnt_range[jid] for jid in self.joint_ids], dtype=float
        )
        self.q_min = limits[:, 0]
        self.q_max = limits[:, 1]

        # Obstacle geom ids (used to flag collisions).
        self.obstacle_geom_ids: set[int] = set()
        for i, obs in enumerate(obstacles):
            name = obs.get("name", f"obs_{i}")
            gid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            if gid >= 0:
                self.obstacle_geom_ids.add(gid)

        # Arm geom ids: every geom that belongs to the arm (not world, not obstacle).
        self.arm_geom_ids: set[int] = set()
        world_body = 0
        for gid in range(self.model.ngeom):
            body_id = self.model.geom_bodyid[gid]
            if body_id == world_body:
                continue
            if gid in self.obstacle_geom_ids:
                continue
            self.arm_geom_ids.add(gid)

    def set_state(self, q: np.ndarray) -> None:
        """Set the arm configuration and run forward kinematics."""
        q = np.asarray(q, dtype=float)
        for addr, value in zip(self.qpos_addrs, q):
            self.data.qpos[addr] = value
        mujoco.mj_forward(self.model, self.data)

    def is_within_joint_limits(self, q: np.ndarray) -> bool:
        """Return True if q is inside every joint range."""
        q = np.asarray(q, dtype=float)
        return bool(np.all(q >= self.q_min - 1e-9) and np.all(q <= self.q_max + 1e-9))

    def is_collision(self, q: np.ndarray, penetration: float = 1e-4) -> bool:
        """Return True if the arm is in collision with any obstacle."""
        if not self.is_within_joint_limits(q):
            return True
        self.set_state(q)
        for c in self.data.contact:
            g1, g2 = c.geom1, c.geom2
            # Only count contacts that involve an obstacle and the arm.
            obs_touch = g1 in self.obstacle_geom_ids or g2 in self.obstacle_geom_ids
            if not obs_touch:
                continue
            # Negative distance means penetration.
            if c.dist < -penetration:
                return True
        return False

    def is_segment_free(
        self,
        q1: np.ndarray,
        q2: np.ndarray,
        n_checks: int = 20,
        penetration: float = 1e-4,
    ) -> bool:
        """Linearly interpolate between two configurations and check collisions."""
        q1 = np.asarray(q1, dtype=float)
        q2 = np.asarray(q2, dtype=float)
        if not self.is_within_joint_limits(q1) or not self.is_within_joint_limits(q2):
            return False
        for alpha in np.linspace(0.0, 1.0, n_checks + 2)[1:-1]:
            q = q1 + alpha * (q2 - q1)
            if self.is_collision(q, penetration=penetration):
                return False
        return True

    def sample_collision_free(self, rng: np.random.Generator) -> np.ndarray | None:
        """Sample a random collision-free configuration (or None if unlucky)."""
        for _ in range(1000):
            q = rng.uniform(self.q_min, self.q_max)
            if not self.is_collision(q):
                return q
        return None
