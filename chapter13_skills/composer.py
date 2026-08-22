"""Chapter 13 — Skill composer that chains skill instances into verified plans.

The ``Composer`` turns an ordered list of ``SkillInstance`` objects into a
concrete trajectory by delegating each instance to the corresponding registered
``Skill.generate`` callable.  Each skill plan is verified by the Chapter 12
physics verifier so that failures stay local and can be reported back to a
reasoning loop for retry.
"""
from __future__ import annotations

from pathlib import Path

from chapter12_reasoning.physics_verifier import PhysicsVerifier
from chapter12_reasoning.planner import Plan
from chapter12_reasoning.task_parser import SpatialRelation, TaskSpec
from chapter13_skills.skill import SkillInstance, SkillLibrary


class Composer:
    """Compose and verify sequences of skill instances.

    Parameters
    ----------
    xml_path
        Path to the MuJoCo scene used for both planning and verification.
    library
        ``SkillLibrary`` containing the registered skills.  Defaults to the
        library produced by ``make_default_library``.
    """

    def __init__(
        self,
        xml_path: str | Path,
        library: SkillLibrary | None = None,
    ):
        self.xml_path = Path(xml_path)
        self.library = library or self._default_library()
        self.verifier = PhysicsVerifier(self.xml_path)

    @staticmethod
    def _default_library() -> SkillLibrary:
        from chapter13_skills.skills import make_default_library

        return make_default_library()

    @staticmethod
    def _task_from_instance(instance: SkillInstance) -> TaskSpec:
        relation = SpatialRelation(instance.relation) if instance.relation else None
        return TaskSpec(
            skill=None,
            target_object=instance.target_object,
            reference_object=instance.reference_object,
            relation=relation,
            extra=instance.extra,
        )

    def compose(self, instances: list[SkillInstance]) -> tuple[Plan, bool, list[str]]:
        """Compose a full plan and verify it step by step.

        Returns
        -------
        plan
            Concatenated plan of all accepted skill steps.
        success
            True if every skill plan passed verification.
        failures
            Descriptions of any failed steps.
        """
        all_steps: list = []
        failures: list[str] = []

        for idx, instance in enumerate(instances):
            skill = self.library.get(instance.skill_name)
            sub_plan = skill.generate(instance, self.xml_path)
            task = self._task_from_instance(instance)
            result = self.verifier.verify(sub_plan, task)
            if not result.success:
                failures.append(f"Skill {idx} ({instance.skill_name}): {result.message}")
                return Plan(steps=all_steps, reasoning="", source="composer"), False, failures
            all_steps.extend(sub_plan.steps)

        return (
            Plan(
                steps=all_steps,
                reasoning=f"Composed plan with {len(all_steps)} steps.",
                source="composer",
            ),
            True,
            failures,
        )
