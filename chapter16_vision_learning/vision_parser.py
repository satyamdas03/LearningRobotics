"""Vision parser: turn video frames into a structured manipulation description.

Two backends are provided:

* ``HeuristicVisionParser`` — fast, deterministic, no API key. It uses simple
  color segmentation to detect the red and blue blocks, infers which one moved,
  and emits a spatial relation such as ``near`` or ``left_of``.

* ``AnthropicVisionParser`` — optional Claude vision backend for real-world
  videos. Falls back to the heuristic parser when the API key is missing or the
  call fails.
"""
from __future__ import annotations

import base64
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass
class VisionParseResult:
    """Structured description extracted from a video."""

    skill: str
    target_object: str
    reference_object: str | None
    relation: str | None
    description: str


class HeuristicVisionParser:
    """Deterministic color-based vision parser for the synthetic demo scene."""

    def __init__(self, near_px_threshold: int = 80) -> None:
        self.near_px_threshold = near_px_threshold

    @staticmethod
    def _centroid(frames: list[Image.Image], color: str) -> np.ndarray | None:
        """Return the mean (x, y) pixel position of the requested color across frames."""
        positions: list[tuple[float, float]] = []
        for frame in frames:
            arr = np.asarray(frame)
            if color == "red":
                mask = (arr[:, :, 0] > 150) & (arr[:, :, 1] < 100) & (arr[:, :, 2] < 100)
            elif color == "blue":
                mask = (arr[:, :, 0] < 100) & (arr[:, :, 1] < 100) & (arr[:, :, 2] > 150)
            elif color == "green":
                mask = (arr[:, :, 1] > 150) & (arr[:, :, 0] < 100) & (arr[:, :, 2] < 100)
            else:
                continue
            coords = np.argwhere(mask)
            if len(coords):
                positions.append((float(coords[:, 1].mean()), float(coords[:, 0].mean())))
        if not positions:
            return None
        return np.array(positions, dtype=float)

    def parse(self, frames: list[Image.Image]) -> VisionParseResult:
        red_centroids = self._centroid(frames, "red")
        blue_centroids = self._centroid(frames, "blue")

        if red_centroids is None or blue_centroids is None:
            raise ValueError(
                "Heuristic parser could not detect both red and blue blocks in the video."
            )

        red_move = float(np.linalg.norm(red_centroids[-1] - red_centroids[0]))
        blue_move = float(np.linalg.norm(blue_centroids[-1] - blue_centroids[0]))

        if red_move >= blue_move:
            target_color, reference_color = "red", "blue"
            target_centroids, ref_centroids = red_centroids, blue_centroids
        else:
            target_color, reference_color = "blue", "red"
            target_centroids, ref_centroids = blue_centroids, red_centroids

        target_object = f"{target_color}_block"
        reference_object = f"{reference_color}_block"

        # Classify the final spatial relation in image space.
        delta = target_centroids[-1] - ref_centroids[-1]
        final_dist = float(np.linalg.norm(delta))
        if final_dist < self.near_px_threshold:
            relation = "near"
        elif abs(delta[0]) >= abs(delta[1]):
            relation = "right_of" if delta[0] > 0 else "left_of"
        else:
            relation = "below" if delta[1] > 0 else "above"

        # We call any non-trivial motion a "push" in this demo domain.
        skill = "push" if max(red_move, blue_move) > 5 else "reach"
        description = (
            f"The {target_color} block moves toward the {reference_color} block "
            f"and ends up {relation.replace('_', ' ')} it."
        )

        return VisionParseResult(
            skill=skill,
            target_object=target_object,
            reference_object=reference_object,
            relation=relation,
            description=description,
        )


class AnthropicVisionParser:
    """Optional Claude vision parser. Falls back to ``HeuristicVisionParser``."""

    def __init__(self, model: str = "claude-3-5-sonnet-20241022") -> None:
        self.model = model
        self._fallback = HeuristicVisionParser()
        self._client: Any | None = None
        try:
            from anthropic import Anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = Anthropic(api_key=api_key)
        except Exception:
            self._client = None

    def _encode_frame(self, frame: Image.Image) -> str:
        buf = io.BytesIO()
        frame.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("utf-8")

    def parse(self, frames: list[Image.Image]) -> VisionParseResult:
        if self._client is None:
            return self._fallback.parse(frames)

        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": (
                    "You are watching a short video of a simple robot manipulation scene. "
                    "Describe what happens using exactly this JSON format, with no extra text:\n"
                    "{\"skill\": \"push\" or \"reach\" or \"pick\" or \"place\", "
                    "\"target_object\": \"red_block\" or \"blue_block\" or similar, "
                    "\"reference_object\": \"red_block\" or \"blue_block\" or null, "
                    "\"relation\": \"near\" or \"left_of\" or \"right_of\" or \"above\" or \"below\" or null, "
                    "\"description\": \"...\"}"
                ),
            }
        ]
        for frame in frames:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": self._encode_frame(frame),
                    },
                }
            )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=128,
                temperature=0.0,
                messages=[{"role": "user", "content": content}],
            )
            raw = response.content[0].text.strip()
            # Extract JSON even if wrapped in markdown fences.
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()
            data = json.loads(raw)
            return VisionParseResult(
                skill=data.get("skill", "push"),
                target_object=data.get("target_object", "red_block"),
                reference_object=data.get("reference_object"),
                relation=data.get("relation"),
                description=data.get("description", ""),
            )
        except Exception as exc:
            print(f"Anthropic vision parser failed ({exc}); falling back to heuristic parser.")
            return self._fallback.parse(frames)


if __name__ == "__main__":
    from chapter16_vision_learning.synthetic_video import generate_push_video, sample_frames

    video_path = generate_push_video("output/push_video.mp4")
    frames = sample_frames(video_path, n=4)
    parser = HeuristicVisionParser()
    result = parser.parse(frames)
    print(result)
