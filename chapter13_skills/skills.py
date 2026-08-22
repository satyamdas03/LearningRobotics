"""Chapter 13 — Core reusable manipulation skills.

Each skill is a ``Skill`` object whose ``generate`` method produces a concrete
Chapter 12 ``Plan`` for a given ``SkillInstance``.  These skills intentionally
reuse the Chapter 12 rule planner and verifier machinery so that a task plan
is just a sequence of verified skill executions.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from chapter12_reasoning.planner import Plan, PlanStep
from chapter12_reasoning.task_parser import SpatialRelation, SkillName
from chapter13_skills.skill import Skill, SkillInstance


def _body_position(xml_path: str | Path, body_name: str) -> np.ndarray:
    """Read a body's current position from the MuJoCo scene."""
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    return np.array(data.xpos[body_id], dtype=float)


def _offset_for_relation(relation: SpatialRelation, offset: float) -> np.ndarray:
    delta = np.zeros(3)
    if relation == SpatialRelation.LEFT_OF:
        delta[1] = offset
    elif relation == SpatialRelation.RIGHT_OF:
        delta[1] = -offset
    elif relation == SpatialRelation.IN_FRONT_OF:
        delta[0] = offset
    elif relation == SpatialRelation.BEHIND:
        delta[0] = -offset
    elif relation == SpatialRelation.ABOVE:
        delta[2] = offset
    elif relation == SpatialRelation.BELOW:
        delta[2] = -offset
    return delta


def _reach_plan(instance: SkillInstance, xml_path: str | Path) -> Plan:
    """Reach the end effector to a target object's current position."""
    target_pos = _body_position(xml_path, instance.target_object)
    return Plan(
        steps=[
            PlanStep(
                skill=SkillName.REACH,
                object=instance.target_object,
                target_position=target_pos,
                description=f"Reach toward {instance.target_object}.",
            )
        ],
        reasoning=f"Target {instance.target_object} is at {target_pos.tolist()}.",
        source="skill",
    )


def _push_plan(instance: SkillInstance, xml_path: str | Path) -> Plan:
    """Push an object to a target position derived from a spatial relation."""
    if instance.reference_object is None or instance.relation is None:
        raise ValueError("push skill requires a reference_object and relation.")

    ref_pos = _body_position(xml_path, instance.reference_object)
    relation = SpatialRelation(instance.relation)
    target_pos = ref_pos + _offset_for_relation(relation, instance.offset)

    return Plan(
        steps=[
            PlanStep(
                skill=SkillName.PUSH,
                object=instance.target_object,
                target_position=target_pos,
                description=(
                    f"Push {instance.target_object} to the {relation.value} "
                    f"of {instance.reference_object}."
                ),
            )
        ],
        reasoning=(
            f"Reference {instance.reference_object} at {ref_pos.tolist()}; "
            f"target position {target_pos.tolist()}."
        ),
        source="skill",
    )


def _pick_plan(instance: SkillInstance, xml_path: str | Path) -> Plan:
    """Pick (pre-grasp + grasp) the target object.

    This is a coarse approximation: a real pick skill would include an approach
    pose, a grasp pose, and a lift pose.  Here we emit a reach to the object and
    a small upward offset to represent lifting it off the table.
    """
    target_pos = _body_position(xml_path, instance.target_object)
    lift_pos = target_pos + np.array([0.0, 0.0, 0.05])
    return Plan(
        steps=[
            PlanStep(
                skill=SkillName.PICK,
                object=instance.target_object,
                target_position=target_pos,
                description=f"Approach and grasp {instance.target_object}.",
            ),
            PlanStep(
                skill=SkillName.PICK,
                object=instance.target_object,
                target_position=lift_pos,
                description=f"Lift {instance.target_object}.",
            ),
        ],
        reasoning=f"Pick {instance.target_object} at {target_pos.tolist()} and lift.",
        source="skill",
    )


def _place_plan(instance: SkillInstance, xml_path: str | Path) -> Plan:
    """Place an object at a target position relative to a reference."""
    if instance.reference_object is None or instance.relation is None:
        raise ValueError("place skill requires a reference_object and relation.")

    ref_pos = _body_position(xml_path, instance.reference_object)
    relation = SpatialRelation(instance.relation)
    target_pos = ref_pos + _offset_for_relation(relation, instance.offset)

    return Plan(
        steps=[
            PlanStep(
                skill=SkillName.PLACE,
                object=instance.target_object,
                target_position=target_pos,
                description=(
                    f"Place {instance.target_object} {relation.value} "
                    f"{instance.reference_object}."
                ),
            )
        ],
        reasoning=(
            f"Reference {instance.reference_object} at {ref_pos.tolist()}; "
            f"place target {target_pos.tolist()}."
        ),
        source="skill",
    )


def _slide_plan(instance: SkillInstance, xml_path: str | Path) -> Plan:
    """Slide an object a fixed distance along a chosen axis.

    The axis and distance come from ``instance.extra``; defaults to +0.1 m in x.
    """
    axis = instance.extra.get("axis", "x")
    distance = float(instance.extra.get("distance", 0.1))
    start_pos = _body_position(xml_path, instance.target_object)
    delta = np.zeros(3)
    idx = {"x": 0, "y": 1, "z": 2}.get(axis, 0)
    delta[idx] = distance
    target_pos = start_pos + delta

    return Plan(
        steps=[
            PlanStep(
                skill=SkillName.PUSH,
                object=instance.target_object,
                target_position=target_pos,
                description=f"Slide {instance.target_object} along {axis} by {distance} m.",
            )
        ],
        reasoning=f"Slide {instance.target_object} from {start_pos.tolist()} to {target_pos.tolist()}.",
        source="skill",
    )


# Public skill registry.
REACH_SKILL = Skill(
    name="reach",
    skill_name=SkillName.REACH,
    description="Move the end effector to a target object or position.",
    required_objects=["target_object"],
    parameter_schema={"target_object": "string"},
    generate=_reach_plan,
)

PUSH_SKILL = Skill(
    name="push",
    skill_name=SkillName.PUSH,
    description="Push an object to a position satisfying a spatial relation to a reference.",
    required_objects=["target_object", "reference_object"],
    parameter_schema={
        "target_object": "string",
        "reference_object": "string",
        "relation": "string",
        "offset": "float (default 0.15)",
    },
    generate=_push_plan,
)

PICK_SKILL = Skill(
    name="pick",
    skill_name=SkillName.PICK,
    description="Approach, grasp, and lift a target object.",
    required_objects=["target_object"],
    parameter_schema={"target_object": "string"},
    generate=_pick_plan,
)

PLACE_SKILL = Skill(
    name="place",
    skill_name=SkillName.PLACE,
    description="Place a held object at a position relative to a reference object.",
    required_objects=["target_object", "reference_object"],
    parameter_schema={
        "target_object": "string",
        "reference_object": "string",
        "relation": "string",
        "offset": "float (default 0.15)",
    },
    generate=_place_plan,
)

SLIDE_SKILL = Skill(
    name="slide",
    skill_name=SkillName.PUSH,
    description="Slide an object a fixed distance along an axis.",
    required_objects=["target_object"],
    parameter_schema={
        "target_object": "string",
        "axis": "string (x/y/z, default x)",
        "distance": "float (default 0.1)",
    },
    generate=_slide_plan,
)


def make_default_library() -> "SkillLibrary":
    """Return a library with the five core manipulation skills registered."""
    from chapter13_skills.skill import SkillLibrary

    lib = SkillLibrary()
    for skill in (REACH_SKILL, PUSH_SKILL, PICK_SKILL, PLACE_SKILL, SLIDE_SKILL):
        lib.register(skill)
    return lib
