"""Chapter 12 — Physics-grounded retry loop.

A planner proposes a manipulation plan; a MuJoCo verifier simulates it.  If the
plan fails, the failure message is fed back to the planner and a revised plan is
requested.  The loop continues until either a plan succeeds or the retry budget
is exhausted.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chapter12_reasoning.planner import LLMPlanner, Plan, RulePlanner
from chapter12_reasoning.physics_verifier import PhysicsVerifier, VerifyResult
from chapter12_reasoning.task_parser import TaskSpec


@dataclass
class ReasoningResult:
    """Final outcome of the reasoning loop."""

    success: bool
    plan: Plan | None
    verification: VerifyResult | None
    attempts: int
    feedback: list[str]


class PhysicsGroundedReasoningLoop:
    """Plan → simulate → feedback → retry loop."""

    def __init__(
        self,
        xml_path: str | Path,
        planner: LLMPlanner | RulePlanner | None = None,
        max_attempts: int = 3,
    ) -> None:
        self.xml_path = str(xml_path)
        self.planner = planner or LLMPlanner()
        self.verifier = PhysicsVerifier(self.xml_path)
        self.max_attempts = max_attempts

    def solve(self, task: TaskSpec) -> ReasoningResult:
        """Run the retry loop for ``task`` and return the final result."""
        feedback: list[str] = []
        last_plan: Plan | None = None
        last_result: VerifyResult | None = None

        for attempt in range(self.max_attempts):
            plan = self.planner.plan(task, self.xml_path, prior_failures=feedback)
            last_plan = plan

            result = self.verifier.verify(plan, task)
            last_result = result

            if result.success:
                return ReasoningResult(
                    success=True,
                    plan=plan,
                    verification=result,
                    attempts=attempt + 1,
                    feedback=feedback,
                )

            feedback.append(
                f"Attempt {attempt + 1}: {result.message} "
                f"Target at {result.target_position.tolist()}."
            )

        return ReasoningResult(
            success=False,
            plan=last_plan,
            verification=last_result,
            attempts=self.max_attempts,
            feedback=feedback,
        )
