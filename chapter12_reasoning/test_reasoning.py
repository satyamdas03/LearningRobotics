"""Tests for Milestone 6 — foundation-model + physics verifier loop."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from chapter12_reasoning.planner import LLMPlanner, Plan, RulePlanner
from chapter12_reasoning.physics_verifier import PhysicsVerifier, VerifyResult
from chapter12_reasoning.reasoning_loop import PhysicsGroundedReasoningLoop
from chapter12_reasoning.task_parser import SkillName, SpatialRelation, TaskSpec, parse_task


HERE = Path(__file__).parent
SCENE_XML = HERE.parent / "chapter10_perception" / "scene.xml"


def test_parser_extracts_red_left_of_blue():
    """The parser should identify target, reference, skill, and relation."""
    task = parse_task("Push the red block left of the blue block.")
    assert task.skill == SkillName.PUSH
    assert task.target_object == "red_block"
    assert task.reference_object == "blue_block"
    assert task.relation == SpatialRelation.LEFT_OF


def test_parser_falls_back_to_reach():
    """An instruction without a relation or reference should default to reach."""
    task = parse_task("Reach toward the blue block.")
    assert task.skill == SkillName.REACH
    assert task.target_object == "blue_block"
    assert task.reference_object is None
    assert task.relation is None


def test_rule_planner_generates_push_step():
    """The deterministic planner should emit a push step with a sensible target."""
    planner = RulePlanner()
    task = parse_task("push the red block left of the blue block")
    plan = planner.plan(task, SCENE_XML)

    assert len(plan.steps) == 1
    step = plan.steps[0]
    assert step.skill == SkillName.PUSH
    assert step.object == "red_block"
    assert step.target_position is not None
    # The blue block is at y = -0.15; left-of means a higher world-y value.
    assert step.target_position[1] > -0.1


def test_verifier_accepts_valid_plan():
    """A plan that places red left of blue should be accepted."""
    planner = RulePlanner()
    task = parse_task("push the red block left of the blue block")
    plan = planner.plan(task, SCENE_XML)

    verifier = PhysicsVerifier(SCENE_XML)
    result = verifier.verify(plan, task)

    assert isinstance(result, VerifyResult)
    assert result.success
    assert result.reference_position is not None
    assert result.target_position[1] > result.reference_position[1]


def test_verifier_rejects_invalid_plan():
    """A plan that places red right of blue for a left-of task should fail."""
    task = parse_task("push the red block left of the blue block")
    # Build a deliberately wrong plan.
    from chapter12_reasoning.planner import PlanStep

    wrong_plan = Plan(
        steps=[
            PlanStep(
                skill=SkillName.PUSH,
                object="red_block",
                target_position=np.array([0.8, -0.3, 0.46]),
                description="wrong side",
            )
        ]
    )

    verifier = PhysicsVerifier(SCENE_XML)
    result = verifier.verify(wrong_plan, task)

    assert not result.success
    assert "NOT satisfied" in result.message


def test_retry_loop_succeeds_with_rule_planner():
    """The reasoning loop should succeed on a simple spatial task."""
    task = parse_task("push the red block left of the blue block")
    loop = PhysicsGroundedReasoningLoop(
        xml_path=SCENE_XML,
        planner=RulePlanner(),
        max_attempts=2,
    )
    result = loop.solve(task)

    assert result.success
    assert result.plan is not None
    assert result.verification is not None
    assert result.attempts >= 1


def test_llm_planner_fallback_without_api():
    """When no API key is available, LLMPlanner should fall back to rule planning."""
    planner = LLMPlanner()
    task = parse_task("push the red block left of the blue block")
    plan = planner.plan(task, SCENE_XML)

    assert len(plan.steps) >= 1
    assert plan.steps[0].object == "red_block"
