"""Tests for Milestone 9 — end-to-end north-star demo."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from chapter15_north_star.north_star import NorthStarDemo


REPO_ROOT = Path(__file__).parent.parent
SCENE_XML = REPO_ROOT / "chapter10_perception" / "scene.xml"


@pytest.fixture
def demo() -> NorthStarDemo:
    return NorthStarDemo(scene_xml=SCENE_XML, reach_tolerance=0.08, trajectory_duration=2.5)


def test_parse_reach_task(demo: NorthStarDemo) -> None:
    """A reach instruction should parse to a reach skill on the red block."""
    instance = demo._build_instance("reach the red block")
    assert instance.skill_name == "reach"
    assert instance.target_object == "red_block"


def test_parse_push_task(demo: NorthStarDemo) -> None:
    """A push instruction should parse to a push skill with a relation."""
    instance = demo._build_instance("push the red block left of the blue block")
    assert instance.skill_name == "push"
    assert instance.target_object == "red_block"
    assert instance.reference_object == "blue_block"
    assert instance.relation == "left_of"


def test_plan_verifies_for_simple_tasks(demo: NorthStarDemo) -> None:
    """The skill composer should verify canonical tasks in the scene."""
    report = demo.run("reach the red block")
    assert report.plan_verified
    assert report.plan_steps >= 1


def test_ik_reaches_target_position(demo: NorthStarDemo) -> None:
    """IK should find a configuration whose EE is near the target point."""
    p_target = np.array([0.8, 0.15, 0.46])  # red block position in scene.xml
    q, info = demo._solve_ik(p_target)
    assert info["position_error"] < 1e-3
    assert np.linalg.norm(demo.ik._current_pose()[1] - p_target) < 1e-3


def test_arm_reaches_goal(demo: NorthStarDemo) -> None:
    """The full pipeline should move the virtual arm within tolerance."""
    report = demo.run("reach the red block")
    assert report.arm_reached, f"final error {report.final_arm_error:.4f} exceeded tolerance"


def test_skill_saved_to_library(demo: NorthStarDemo, tmp_path: Path) -> None:
    """A successful run should serialize the executed skill instance."""
    library_path = tmp_path / "learned_skill.json"
    report = demo.run("reach the red block", library_save_path=library_path)
    assert report.skill_saved
    assert library_path.exists()
    payload = json.loads(library_path.read_text(encoding="utf-8"))
    assert payload["instances"]
    assert payload["instances"][0]["target_object"] == "red_block"
