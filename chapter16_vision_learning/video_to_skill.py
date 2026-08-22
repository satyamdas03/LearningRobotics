"""Bridge from a video observation to a verified, reusable skill instance.

The pipeline is:

1. Sample frames from the video.
2. Run a vision parser (heuristic or VLM) to get a structured description.
3. Map the description to a ``SkillInstance`` from the Chapter 13 skill library.
4. Verify the instance with the Chapter 12 physics verifier via the composer.
5. Save the verified instance to the skill library so it can be replayed or shared.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from chapter12_reasoning.physics_verifier import PhysicsVerifier
from chapter12_reasoning.task_parser import SpatialRelation, TaskSpec
from chapter13_skills.composer import Composer
from chapter13_skills.skill import SkillInstance, SkillLibrary
from chapter13_skills.skills import (
    PUSH_SKILL,
    REACH_SKILL,
    PICK_SKILL,
    PLACE_SKILL,
    make_default_library,
)
from chapter16_vision_learning.synthetic_video import sample_frames
from chapter16_vision_learning.vision_parser import (
    AnthropicVisionParser,
    HeuristicVisionParser,
    VisionParseResult,
)


# Map parser output to the registered Chapter 13 skill templates.
_SKILL_REGISTRY = {
    "push": PUSH_SKILL,
    "reach": REACH_SKILL,
    "pick": PICK_SKILL,
    "place": PLACE_SKILL,
}


def _result_to_skill_instance(result: VisionParseResult) -> SkillInstance:
    """Convert a vision parse result into a concrete skill instance."""
    skill_template = _SKILL_REGISTRY.get(result.skill)
    if skill_template is None:
        raise ValueError(
            f"Parser returned unknown skill '{result.skill}'. "
            f"Known skills: {list(_SKILL_REGISTRY)}"
        )

    kwargs: dict[str, Any] = {"target_object": result.target_object}
    if result.reference_object is not None:
        kwargs["reference_object"] = result.reference_object
    if result.relation is not None:
        kwargs["relation"] = result.relation
        # For a "near" relation, use a small offset so the target sits close to
        # the reference instead of exactly on top of it.
        if result.relation == "near":
            kwargs["offset"] = 0.05
    return skill_template.instantiate(**kwargs)


def video_to_skill(
    video_path: str | Path,
    xml_path: str | Path | None = None,
    parser: HeuristicVisionParser | AnthropicVisionParser | None = None,
    library: SkillLibrary | None = None,
) -> tuple[SkillInstance, Any, bool, list[str]]:
    """Extract a verified skill instance from an ordinary video.

    Returns
    -------
    instance
        The ``SkillInstance`` inferred from the video.
    plan
        The composed plan returned by the Chapter 13 composer.
    success
        Whether the physics verifier accepted the plan.
    failures
        Verifier failure messages (empty on success).
    """
    video_path = Path(video_path)
    if xml_path is None:
        xml_path = Path(__file__).parent.parent / "chapter10_perception" / "scene.xml"
    xml_path = Path(xml_path)

    if parser is None:
        parser = HeuristicVisionParser()

    frames = sample_frames(video_path, n=4)
    result = parser.parse(frames)
    instance = _result_to_skill_instance(result)

    composer = Composer(xml_path, library=library)
    plan, success, failures = composer.compose([instance])

    if success:
        # Persist the learned instance so it can be replayed or shared.
        lib = library or make_default_library()
        lib.add_instance(instance)

    return instance, plan, success, failures


def replay_skill_in_sim(
    instance: SkillInstance,
    xml_path: str | Path | None = None,
    library: SkillLibrary | None = None,
) -> bool:
    """Re-run a learned skill instance in the MuJoCo physics verifier.

    This is a lightweight replay check: it rebuilds the plan and verifies that
    the resulting spatial relation still holds.
    """
    if xml_path is None:
        xml_path = Path(__file__).parent.parent / "chapter10_perception" / "scene.xml"
    xml_path = Path(xml_path)

    composer = Composer(xml_path, library=library)
    plan, success, failures = composer.compose([instance])
    return success


if __name__ == "__main__":
    from chapter16_vision_learning.synthetic_video import generate_push_video

    out = generate_push_video("output/push_video.mp4")
    instance, plan, success, failures = video_to_skill(out)
    print(f"Skill: {instance.skill_name} {instance.target_object} -> {instance.relation} {instance.reference_object}")
    print(f"Verified: {success}")
    if failures:
        print("Failures:", failures)
    print(f"Plan steps: {len(plan.steps)}")
