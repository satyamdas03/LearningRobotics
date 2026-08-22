"""End-to-end demo: watch a synthetic manipulation video and learn a skill.

The demo:
1. Renders a short RGB video of the red block being pushed toward the blue block.
2. Parses the video with a vision parser (Claude vision if ``ANTHROPIC_API_KEY``
   is available, otherwise the deterministic heuristic parser).
3. Converts the parse result into a ``SkillInstance``.
4. Verifies the instance with the Chapter 12 physics verifier.
5. Replays the learned skill and saves it to a JSON skill library.
"""
from __future__ import annotations

import os
from pathlib import Path

from chapter16_vision_learning.synthetic_video import generate_push_video, sample_frames
from chapter16_vision_learning.vision_parser import AnthropicVisionParser, HeuristicVisionParser
from chapter16_vision_learning.video_to_skill import replay_skill_in_sim, video_to_skill
from chapter13_skills.skills import make_default_library


def main() -> int:
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    video_path = output_dir / "vision_push_demo.mp4"
    print(f"1. Generating synthetic video: {video_path}")
    generate_push_video(video_path)

    print("2. Sampling frames and parsing with vision backend...")
    parser = (
        AnthropicVisionParser()
        if os.environ.get("ANTHROPIC_API_KEY")
        else HeuristicVisionParser()
    )
    backend_name = "Claude vision" if os.environ.get("ANTHROPIC_API_KEY") else "heuristic"
    print(f"   Using {backend_name} parser")

    frames = sample_frames(video_path, n=4)
    parse_result = parser.parse(frames)
    print(
        f"   Parsed: {parse_result.skill} {parse_result.target_object} "
        f"-u003e {parse_result.relation} {parse_result.reference_object}"
    )
    print(f"   Description: {parse_result.description}")

    print("3. Converting to SkillInstance and verifying in physics...")
    instance, plan, success, failures = video_to_skill(
        video_path,
        parser=parser,
        library=make_default_library(),
    )
    print(f"   SkillInstance: {instance}")
    print(f"   Verifier success: {success}")
    if failures:
        print(f"   Failures: {failures}")

    print("4. Replaying the learned skill...")
    replay_ok = replay_skill_in_sim(instance)
    print(f"   Replay success: {replay_ok}")

    print("5. Saving learned skill to library...")
    lib = make_default_library()
    lib.add_instance(instance)
    lib_path = output_dir / "learned_vision_skill.json"
    lib.save_json(lib_path)
    print(f"   Library saved: {lib_path}")

    return 0 if (success and replay_ok) else 1


if __name__ == "__main__":
    raise SystemExit(main())
