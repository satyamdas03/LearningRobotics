"""Chapter 12 — LLM/VLM + rule-based planner for manipulation tasks.

A planner converts a ``TaskSpec`` into a concrete plan: a list of waypoints or
high-level skill steps.  If an Anthropic API key is available, the LLM planner
asks Claude for a JSON plan; otherwise it falls back to a deterministic geometric
planner built on the scene state.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from task_parser import SkillName, SpatialRelation, TaskSpec


@dataclass
class PlanStep:
    """One step of a manipulation plan."""

    skill: SkillName
    object: str | None = None
    target_position: np.ndarray | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill": self.skill.value,
            "object": self.object,
            "target_position": (
                self.target_position.tolist() if self.target_position is not None else None
            ),
            "description": self.description,
        }


@dataclass
class Plan:
    """A plan is an ordered list of ``PlanStep``s plus optional reasoning."""

    steps: list[PlanStep]
    reasoning: str = ""
    source: str = "unknown"


class RulePlanner:
    """Deterministic geometric planner for the benchmark tasks in this milestone.

    Given a task like "push red block left of blue block", it reads the current
    MuJoCo scene, computes a target position for the manipulated object, and
    emits a single ``push``/``place`` step that achieves the relation.
    """

    OFFSET = 0.15  # meters; how far to place/push the target from the reference.

    def plan(
        self,
        task: TaskSpec,
        xml_path: str | Path,
        prior_failures: list[str] | None = None,
    ) -> Plan:
        xml_path = str(xml_path)
        model = mujoco.MjModel.from_xml_path(xml_path)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        if task.reference_object is None or task.relation is None:
            # No relation: just emit a reach step to the target object's current position.
            target_pos = self._body_position(model, data, task.target_object)
            return Plan(
                steps=[
                    PlanStep(
                        skill=SkillName.REACH,
                        object=task.target_object,
                        target_position=target_pos,
                        description=f"Reach the {task.target_object}.",
                    )
                ],
                reasoning="No spatial relation requested; reaching the target object.",
                source="rule",
            )

        ref_pos = self._body_position(model, data, task.reference_object)
        target_pos = self._compute_target_position(ref_pos, task.relation)

        skill = task.skill if task.skill in {SkillName.PUSH, SkillName.PLACE} else SkillName.PUSH
        return Plan(
            steps=[
                PlanStep(
                    skill=skill,
                    object=task.target_object,
                    target_position=target_pos,
                    description=(
                        f"{skill.value} {task.target_object} to the "
                        f"{task.relation.value} of {task.reference_object}."
                    ),
                )
            ],
            reasoning=(
                f"Reference {task.reference_object} is at {ref_pos.tolist()}. "
                f"Target position for {task.relation.value} is {target_pos.tolist()}."
            ),
            source="rule",
        )

    def _body_position(
        self, model: mujoco.MjModel, data: mujoco.MjData, name: str
    ) -> np.ndarray:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return np.array(data.xpos[body_id], dtype=float)

    def _compute_target_position(
        self, ref_pos: np.ndarray, relation: SpatialRelation
    ) -> np.ndarray:
        delta = np.zeros(3)
        if relation == SpatialRelation.LEFT_OF:
            delta[1] = self.OFFSET
        elif relation == SpatialRelation.RIGHT_OF:
            delta[1] = -self.OFFSET
        elif relation == SpatialRelation.IN_FRONT_OF:
            delta[0] = self.OFFSET
        elif relation == SpatialRelation.BEHIND:
            delta[0] = -self.OFFSET
        elif relation == SpatialRelation.ABOVE:
            delta[2] = self.OFFSET
        elif relation == SpatialRelation.BELOW:
            delta[2] = -self.OFFSET
        elif relation in {SpatialRelation.ON, SpatialRelation.NEAR}:
            delta = np.zeros(3)
        return ref_pos + delta


class LLMPlanner:
    """Claude-powered planner with a deterministic fallback.

    When the Anthropic API is unavailable, it delegates to ``RulePlanner`` so
    tests and demos still work without an API key.
    """

    def __init__(self, model: str = "claude-3-5-sonnet-20241022") -> None:
        self.model = model
        self._rule = RulePlanner()
        self._client: Any | None = None

        try:
            from anthropic import Anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY")
            if api_key:
                self._client = Anthropic(api_key=api_key)
        except Exception:
            self._client = None

    def plan(self, task: TaskSpec, xml_path: str | Path, prior_failures: list[str] | None = None) -> Plan:
        """Return a plan, using Claude if available and otherwise the rule fallback."""
        if self._client is None:
            return self._rule.plan(task, xml_path)

        # Build a prompt that includes the task and any prior failure messages.
        failure_text = ""
        if prior_failures:
            failure_text = "\nPrior attempts failed for these reasons:\n" + "\n".join(
                f"- {f}" for f in prior_failures
            )

        prompt = (
            "You are a robot task planner. Given a manipulation instruction, output a short "
            "JSON list of plan steps. Each step must have keys: skill (one of reach/push/pick/place), "
            "object (the MuJoCo body name), target_position ([x, y, z] in meters), and description.\n\n"
            f"Instruction: {self._task_description(task)}\n"
            f"Available scene bodies: red_block, blue_block, table.\n"
            f"{failure_text}\n\n"
            "Plan (JSON):"
        )

        try:
            response = self._client.messages.create(
                model=self.model,
                max_tokens=256,
                temperature=0.0,
                messages=[{"role": "user", "content": prompt}],
            )
            raw_text = response.content[0].text
            # Try to extract a JSON array from the response.
            match = self._extract_json_array(raw_text)
            if match:
                steps = self._parse_steps(match)
                return Plan(
                    steps=steps,
                    reasoning=raw_text.strip(),
                    source="llm",
                )
        except Exception as exc:
            pass

        # Any parsing or API error falls back to the deterministic planner.
        return self._rule.plan(task, xml_path)

    def _task_description(self, task: TaskSpec) -> str:
        if task.reference_object and task.relation:
            return f"{task.skill.value} {task.target_object} {task.relation.value.replace('_', ' ')} {task.reference_object}"
        return f"{task.skill.value} {task.target_object}"

    def _extract_json_array(self, text: str) -> str | None:
        start = text.find("[")
        end = text.rfind("]")
        if start >= 0 and end > start:
            return text[start : end + 1]
        return None

    def _parse_steps(self, json_text: str) -> list[PlanStep]:
        raw = json.loads(json_text)
        steps: list[PlanStep] = []
        for item in raw:
            skill = SkillName(item.get("skill", "reach"))
            obj = item.get("object")
            pos = item.get("target_position")
            pos_arr = np.asarray(pos, dtype=float) if pos is not None else None
            steps.append(
                PlanStep(
                    skill=skill,
                    object=obj,
                    target_position=pos_arr,
                    description=item.get("description", ""),
                )
            )
        return steps
