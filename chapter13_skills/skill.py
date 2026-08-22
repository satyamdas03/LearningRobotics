"""Chapter 13 — Skill representation for reusable manipulation behaviors.

A ``Skill`` is a parameterized template that can be instantiated with concrete
object names and goal parameters to produce a ``Plan`` (Chapter 12).  The
library stores skills as JSON so they can be saved, loaded, and shared across
agents.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from chapter12_reasoning.planner import Plan, PlanStep
from chapter12_reasoning.task_parser import SkillName


@dataclass
class SkillInstance:
    """Concrete binding of a skill to objects and numeric parameters."""

    skill_name: str
    target_object: str
    reference_object: str | None = None
    relation: str | None = None
    offset: float = 0.15
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_name": self.skill_name,
            "target_object": self.target_object,
            "reference_object": self.reference_object,
            "relation": self.relation,
            "offset": self.offset,
            "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillInstance":
        return cls(
            skill_name=data["skill_name"],
            target_object=data["target_object"],
            reference_object=data.get("reference_object"),
            relation=data.get("relation"),
            offset=data.get("offset", 0.15),
            extra=data.get("extra", {}),
        )


@dataclass
class Skill:
    """A reusable manipulation skill template.

    ``generate`` is a callable that receives the skill instance and a scene XML
    path and returns a concrete ``Plan``.
    """

    name: str
    skill_name: SkillName
    description: str
    required_objects: list[str]
    parameter_schema: dict[str, Any]
    generate: Callable[[SkillInstance, str | Path], Plan]

    def instantiate(self, **kwargs: Any) -> SkillInstance:
        """Create a ``SkillInstance`` from this template."""
        return SkillInstance(skill_name=self.name, **kwargs)


class SkillLibrary:
    """JSON-backed collection of reusable skills and skill instances."""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}
        self._instances: list[SkillInstance] = []

    def register(self, skill: Skill) -> None:
        """Register a ``Skill`` template."""
        self._skills[skill.name] = skill

    def add_instance(self, instance: SkillInstance) -> None:
        """Record a concrete skill execution."""
        self._instances.append(instance)

    def get(self, name: str) -> Skill:
        """Return a registered skill by name."""
        if name not in self._skills:
            raise KeyError(f"Skill '{name}' not registered.")
        return self._skills[name]

    def list_skills(self) -> list[str]:
        """Return the names of all registered skills."""
        return list(self._skills.keys())

    def save_json(self, path: str | Path) -> None:
        """Save the registered skill *instances* to a JSON file."""
        path = Path(path)
        payload = {
            "skills": [name for name in self._skills],
            "instances": [inst.to_dict() for inst in self._instances],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def load_json(self, path: str | Path) -> None:
        """Load skill *instances* from a JSON file."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        self._instances = [SkillInstance.from_dict(item) for item in payload.get("instances", [])]

    @property
    def instances(self) -> list[SkillInstance]:
        return list(self._instances)
