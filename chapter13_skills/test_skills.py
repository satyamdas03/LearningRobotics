"""Tests for Milestone 7 — reusable skill library and composition."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chapter12_reasoning.task_parser import SkillName, SpatialRelation
from chapter13_skills.composer import Composer
from chapter13_skills.skill import SkillInstance, SkillLibrary
from chapter13_skills.skills import (
    PICK_SKILL,
    PLACE_SKILL,
    PUSH_SKILL,
    REACH_SKILL,
    SLIDE_SKILL,
    make_default_library,
)

HERE = Path(__file__).parent
SCENE_XML = HERE.parent / "chapter10_perception" / "scene.xml"


def test_reach_skill_targets_red_block():
    """The reach skill should produce a plan step near the red block."""
    instance = REACH_SKILL.instantiate(target_object="red_block")
    plan = REACH_SKILL.generate(instance, SCENE_XML)

    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.skill == SkillName.REACH
    assert step.object == "red_block"
    assert step.target_position is not None
    # The red block is at y = +0.15 in the default scene.
    assert step.target_position[1] > 0.1


def test_push_skill_places_target_left_of_reference():
    """The push skill should place the target to the left of the reference."""
    instance = PUSH_SKILL.instantiate(
        target_object="red_block",
        reference_object="blue_block",
        relation=SpatialRelation.LEFT_OF.value,
        offset=0.15,
    )
    plan = PUSH_SKILL.generate(instance, SCENE_XML)

    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.skill == SkillName.PUSH
    assert step.object == "red_block"
    # Blue block is at y = -0.15; left-of means higher world-y.
    assert step.target_position[1] > -0.05


def test_place_skill_verifies_in_physics():
    """A place skill plan should be accepted by the physics verifier."""
    instance = PLACE_SKILL.instantiate(
        target_object="red_block",
        reference_object="blue_block",
        relation=SpatialRelation.RIGHT_OF.value,
        offset=0.15,
    )
    composer = Composer(SCENE_XML)
    plan, success, failures = composer.compose([instance])

    assert success
    assert not failures
    assert len(plan.steps) == 1
    assert plan.steps[0].skill == SkillName.PLACE


def test_slide_skill_moves_block_along_x():
    """The slide skill should shift the target object along the requested axis."""
    instance = SLIDE_SKILL.instantiate(
        target_object="red_block",
        extra={"axis": "x", "distance": 0.1},
    )
    plan = SLIDE_SKILL.generate(instance, SCENE_XML)

    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.object == "red_block"
    # The red block starts at x ≈ 0.8; sliding +0.1 in x should increase x.
    assert step.target_position[0] > 0.85


def test_pick_skill_emits_two_steps():
    """The pick skill should emit an approach and a lift step."""
    instance = PICK_SKILL.instantiate(target_object="blue_block")
    plan = PICK_SKILL.generate(instance, SCENE_XML)

    assert len(plan.steps) == 2
    assert all(step.skill == SkillName.PICK for step in plan.steps)
    # The lift step should be higher in z than the approach step.
    assert plan.steps[1].target_position[2] > plan.steps[0].target_position[2]


def test_composer_chains_push_and_slide():
    """The composer should chain a push and a slide into a single verified plan."""
    composer = Composer(SCENE_XML)
    instances = [
        PUSH_SKILL.instantiate(
            target_object="red_block",
            reference_object="blue_block",
            relation=SpatialRelation.LEFT_OF.value,
        ),
        SLIDE_SKILL.instantiate(
            target_object="red_block",
            extra={"axis": "x", "distance": -0.05},
        ),
    ]
    plan, success, failures = composer.compose(instances)

    assert success
    assert not failures
    assert len(plan.steps) == 2


def test_skill_library_json_roundtrip(tmp_path):
    """A skill library should save and reload skill instances intact."""
    lib = make_default_library()
    instance = REACH_SKILL.instantiate(target_object="blue_block")
    lib.add_instance(instance)

    path = tmp_path / "skills.json"
    lib.save_json(path)

    loaded = SkillLibrary()
    loaded.register(REACH_SKILL)
    loaded.load_json(path)

    assert len(loaded.instances) == 1
    restored = loaded.instances[0]
    assert restored.skill_name == "reach"
    assert restored.target_object == "blue_block"
