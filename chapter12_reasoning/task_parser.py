"""Chapter 12 — Natural-language task parser for physical reasoning.

Converts free-form instructions like "push the red block left of the blue block"
into a structured ``TaskSpec`` that the rest of the reasoning loop can plan and
verify against a MuJoCo scene.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum


class SpatialRelation(str, Enum):
    """Supported spatial relations for manipulation tasks."""

    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    IN_FRONT_OF = "in_front_of"
    BEHIND = "behind"
    ON = "on"
    NEAR = "near"


class SkillName(str, Enum):
    """High-level manipulation skills the planner can emit."""

    REACH = "reach"
    PUSH = "push"
    PICK = "pick"
    PLACE = "place"


@dataclass
class TaskSpec:
    """Structured description of a manipulation instruction."""

    skill: SkillName
    target_object: str
    reference_object: str | None
    relation: SpatialRelation | None
    extra: dict[str, str] | None = None

    def requires_verification(self) -> bool:
        """A task is verifiable only if it specifies a target relation."""
        return self.relation is not None and self.reference_object is not None


def parse_task(text: str) -> TaskSpec:
    """Parse a short manipulation instruction into a ``TaskSpec``.

    The parser is intentionally small and rule-based.  It handles the canonical
    patterns tested in this milestone and falls back to a generic ``reach``
    skill when the relation cannot be determined.
    """
    text_lower = text.lower().strip()

    # Identify target object: first colored block mentioned.
    target_match = re.search(
        r"\b(red|blue|green|yellow|orange|purple)\s+(block|cube|object|body)\b",
        text_lower,
    )
    target_object = "red_block"
    if target_match:
        target_object = f"{target_match.group(1)}_block"

    # Identify reference object: second colored block mentioned.
    reference_object: str | None = None
    all_color_blocks = re.findall(
        r"\b(red|blue|green|yellow|orange|purple)\s+(block|cube|object|body)\b",
        text_lower,
    )
    if len(all_color_blocks) >= 2:
        reference_object = f"{all_color_blocks[1][0]}_block"

    # Determine skill from verb.
    skill = SkillName.REACH
    if "push" in text_lower:
        skill = SkillName.PUSH
    elif "pick" in text_lower or "grab" in text_lower or "lift" in text_lower:
        skill = SkillName.PICK
    elif "place" in text_lower or "put" in text_lower or "move" in text_lower:
        skill = SkillName.PLACE

    # Determine spatial relation.
    relation: SpatialRelation | None = None
    if re.search(r"\bleft\s+of\b", text_lower):
        relation = SpatialRelation.LEFT_OF
    elif re.search(r"\bright\s+of\b", text_lower):
        relation = SpatialRelation.RIGHT_OF
    elif re.search(r"\bin\s+front\s+of\b", text_lower):
        relation = SpatialRelation.IN_FRONT_OF
    elif re.search(r"\bbehind\b", text_lower):
        relation = SpatialRelation.BEHIND
    elif re.search(r"\bon\s+(top\s+of)?\b", text_lower):
        relation = SpatialRelation.ON
    elif re.search(r"\babove\b", text_lower):
        relation = SpatialRelation.ABOVE
    elif re.search(r"\bbelow\b", text_lower):
        relation = SpatialRelation.BELOW
    elif re.search(r"\bnear\b", text_lower) or re.search(r"\bnext\s+to\b", text_lower):
        relation = SpatialRelation.NEAR

    return TaskSpec(
        skill=skill,
        target_object=target_object,
        reference_object=reference_object,
        relation=relation,
    )
