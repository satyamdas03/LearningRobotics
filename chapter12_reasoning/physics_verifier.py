"""Chapter 12 — Physics verifier: simulate a plan in MuJoCo and report success.

The verifier loads the manipulation scene, applies each plan step, steps the
physics forward briefly, and then checks whether the requested spatial relation
between target and reference objects holds.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from planner import Plan
from task_parser import SkillName, SpatialRelation, TaskSpec


@dataclass
class VerifyResult:
    """Outcome of simulating a plan in MuJoCo."""

    success: bool
    target_position: np.ndarray
    reference_position: np.ndarray | None
    distance: float
    message: str
    final_qpos: np.ndarray | None = None


def _body_position(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    return np.array(data.xpos[body_id], dtype=float)


def _set_body_position(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name: str,
    pos: np.ndarray,
) -> None:
    """Move a free-floating body to ``pos`` by writing its qpos slice."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    jnt_id = model.body_jntadr[body_id]
    qpos_adr = model.jnt_qposadr[jnt_id]
    # Free joint: qpos is [x, y, z, qw, qx, qy, qz].  Keep identity quaternion.
    data.qpos[qpos_adr : qpos_adr + 3] = pos
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = [1.0, 0.0, 0.0, 0.0]


def _relation_satisfied(
    target_pos: np.ndarray,
    ref_pos: np.ndarray,
    relation: SpatialRelation,
    threshold: float = 0.08,
) -> tuple[bool, float]:
    """Check whether ``target_pos`` satisfies ``relation`` relative to ``ref_pos``."""
    delta = target_pos - ref_pos
    if relation == SpatialRelation.LEFT_OF:
        return delta[1] > threshold, float(delta[1])
    if relation == SpatialRelation.RIGHT_OF:
        return delta[1] < -threshold, float(-delta[1])
    if relation == SpatialRelation.IN_FRONT_OF:
        return delta[0] > threshold, float(delta[0])
    if relation == SpatialRelation.BEHIND:
        return delta[0] < -threshold, float(-delta[0])
    if relation == SpatialRelation.ABOVE:
        return delta[2] > threshold, float(delta[2])
    if relation == SpatialRelation.BELOW:
        return delta[2] < -threshold, float(-delta[2])
    if relation == SpatialRelation.ON:
        # Object should be near in x/y and slightly above in z.
        horizontal = np.linalg.norm(delta[:2]) < threshold
        vertical = 0.0 < delta[2] < 0.05
        return horizontal and vertical, float(np.linalg.norm(delta))
    if relation == SpatialRelation.NEAR:
        dist = float(np.linalg.norm(delta))
        return dist < 0.15, dist
    return False, float(np.linalg.norm(delta))


class PhysicsVerifier:
    """Simulate a plan and evaluate it against a task specification."""

    def __init__(self, xml_path: str | Path, settle_steps: int = 20) -> None:
        self.xml_path = str(xml_path)
        self.settle_steps = settle_steps
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, self.data)

    def reset(self) -> None:
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)

    def verify(self, plan: Plan, task: TaskSpec) -> VerifyResult:
        """Apply ``plan`` to the scene and check whether ``task`` is satisfied."""
        self.reset()

        for step in plan.steps:
            if step.object is not None and step.target_position is not None:
                _set_body_position(self.model, self.data, step.object, step.target_position)
            mujoco.mj_forward(self.model, self.data)
            # Let contacts settle briefly.
            for _ in range(self.settle_steps):
                mujoco.mj_step(self.model, self.data)

        target_pos = _body_position(self.model, self.data, task.target_object)
        ref_pos: np.ndarray | None = None
        if task.reference_object is not None and task.relation is not None:
            ref_pos = _body_position(self.model, self.data, task.reference_object)
            satisfied, metric = _relation_satisfied(target_pos, ref_pos, task.relation)
            if satisfied:
                return VerifyResult(
                    success=True,
                    target_position=target_pos,
                    reference_position=ref_pos,
                    distance=metric,
                    message=f"Relation {task.relation.value} satisfied (metric={metric:.3f}).",
                    final_qpos=self.data.qpos.copy(),
                )
            return VerifyResult(
                success=False,
                target_position=target_pos,
                reference_position=ref_pos,
                distance=metric,
                message=(
                    f"Relation {task.relation.value} NOT satisfied "
                    f"(metric={metric:.3f})."
                ),
                final_qpos=self.data.qpos.copy(),
            )

        # No relation to check: success means the simulation did not crash.
        return VerifyResult(
            success=True,
            target_position=target_pos,
            reference_position=None,
            distance=0.0,
            message="No spatial relation requested; plan executed without error.",
            final_qpos=self.data.qpos.copy(),
        )
