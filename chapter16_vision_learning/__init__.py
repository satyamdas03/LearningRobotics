"""Chapter 16 — Vision-based learning from ordinary video.

Convert a passive video observation of a manipulation scene into a structured
``SkillInstance``, verify it with the Chapter 12 physics verifier, and replay it
in simulation.
"""

from chapter16_vision_learning.synthetic_video import generate_push_video, sample_frames
from chapter16_vision_learning.vision_parser import (
    HeuristicVisionParser,
    AnthropicVisionParser,
    VisionParseResult,
)
from chapter16_vision_learning.video_to_skill import video_to_skill, replay_skill_in_sim

__all__ = [
    "generate_push_video",
    "sample_frames",
    "HeuristicVisionParser",
    "AnthropicVisionParser",
    "VisionParseResult",
    "video_to_skill",
    "replay_skill_in_sim",
]
