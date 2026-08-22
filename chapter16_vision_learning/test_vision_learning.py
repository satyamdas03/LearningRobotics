"""End-to-end tests for Chapter 16 — vision-based skill learning.

These tests exercise the full pipeline: synthetic video generation, vision
parsing, skill-instance creation, physics verification, and replay.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from chapter16_vision_learning.synthetic_video import generate_push_video, sample_frames
from chapter16_vision_learning.video_to_skill import replay_skill_in_sim, video_to_skill
from chapter16_vision_learning.vision_parser import AnthropicVisionParser, HeuristicVisionParser


HERE = Path(__file__).parent
SCENE_XML = HERE.parent / "chapter10_perception" / "scene.xml"


def test_generate_push_video(tmp_path: Path) -> None:
    """The synthetic video generator must produce a readable MP4 file."""
    out = generate_push_video(tmp_path / "push.mp4", xml_path=SCENE_XML, n_frames=20, fps=10)
    assert out.exists()
    frames = sample_frames(out, n=3)
    assert len(frames) == 3
    assert all(f.size[0] == 640 and f.size[1] == 480 for f in frames)


def test_heuristic_parser_detects_push_and_near_relation(tmp_path: Path) -> None:
    """The heuristic parser should infer a push skill with red_block near blue_block."""
    out = generate_push_video(tmp_path / "push.mp4", xml_path=SCENE_XML, n_frames=30, fps=15)
    frames = sample_frames(out, n=4)
    result = HeuristicVisionParser().parse(frames)

    assert result.skill == "push"
    assert result.target_object == "red_block"
    assert result.reference_object == "blue_block"
    assert result.relation == "near"


def test_video_to_skill_verifies(tmp_path: Path) -> None:
    """The full video-to-skill pipeline must produce a verifier-accepted skill."""
    out = generate_push_video(tmp_path / "push.mp4", xml_path=SCENE_XML, n_frames=30, fps=15)
    instance, plan, success, failures = video_to_skill(
        out, xml_path=SCENE_XML, parser=HeuristicVisionParser()
    )

    assert instance.skill_name == "push"
    assert instance.target_object == "red_block"
    assert instance.reference_object == "blue_block"
    assert instance.relation == "near"
    assert success
    assert not failures
    assert len(plan.steps) >= 1


def test_replay_skill_in_sim(tmp_path: Path) -> None:
    """A learned skill instance should replay successfully in the verifier."""
    out = generate_push_video(tmp_path / "push.mp4", xml_path=SCENE_XML, n_frames=30, fps=15)
    instance, *_ = video_to_skill(out, xml_path=SCENE_XML, parser=HeuristicVisionParser())
    assert replay_skill_in_sim(instance, xml_path=SCENE_XML)


def test_anthropic_parser_falls_back_without_key(tmp_path: Path) -> None:
    """Without an API key the Anthropic parser must still return a valid result."""
    parser = AnthropicVisionParser()
    out = generate_push_video(tmp_path / "push.mp4", xml_path=SCENE_XML, n_frames=20, fps=10)
    frames = sample_frames(out, n=3)
    result = parser.parse(frames)

    assert result.skill == "push"
    assert result.target_object == "red_block"
    assert result.reference_object == "blue_block"
