"""Friction-pile problem.

Three blocks rest on a surface. Each has a different mass and coefficient of
static friction. The question: which block is hardest to start moving with a
horizontal push?

Physical concepts: static friction, normal force, Coulomb friction.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml, mjcf_box


@register_problem("contact")
class FrictionPile(Problem):
    """Which object in a pile is hardest to start moving?"""

    N_OBJECTS = 3
    LABELS = ["A", "B", "C"]

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.masses = [float(rng.uniform(0.4, 2.0)) for _ in range(self.N_OBJECTS)]
        self.mu_s = [float(rng.uniform(0.15, 0.85)) for _ in range(self.N_OBJECTS)]
        self.half_size = 0.08

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        worldbody_parts: list[str] = []
        spacing = 4.0 * self.half_size
        for i, (mass, mu) in enumerate(zip(self.masses, self.mu_s)):
            x = (i - 1) * spacing
            color = (0.85, 0.30, 0.25, 1.0) if i == 0 else (0.25, 0.55, 0.85, 1.0) if i == 1 else (0.30, 0.75, 0.35, 1.0)
            worldbody_parts.append(
                mjcf_box(
                    name=f"obj_{i}",
                    pos=(x, 0.0, self.half_size + 0.01),
                    size=(self.half_size, self.half_size, self.half_size),
                    rgba=color,
                    mass=mass,
                    friction=(mu, 0.01, 0.01),
                )
            )
        return build_xml("".join(worldbody_parts), timestep=0.002)

    def _compute_answer(self) -> str:
        if getattr(self, "_answer", None) is not None:
            return self._answer

        # The horizontal force needed to start sliding is approximately mu_s * N,
        # where N = m*g.  The object with the largest threshold is hardest to push.
        thresholds = [mu * mass for mu, mass in zip(self.mu_s, self.masses)]
        max_threshold = max(thresholds)
        # Break ties deterministically by choosing the leftmost maximum.
        best_idx = next(i for i, t in enumerate(thresholds) if abs(t - max_threshold) < 1e-9)
        self._answer = self.LABELS[best_idx]
        return self._answer

    def question(self) -> Question:
        lines = []
        for label, mass, mu in zip(self.LABELS, self.masses, self.mu_s):
            lines.append(f"Object {label}: mass {mass:.2f} kg, static friction {mu:.2f}")
        return Question(
            text=(
                "Three objects sit on a flat surface. A horizontal push is applied to each. "
                "Which object is hardest to start moving?\n"
                + "\n".join(lines)
            ),
            answer_type=AnswerType.CHOICE,
            choices=list(self.LABELS),
        )

    def ground_truth(self) -> GroundTruth:
        answer = self._compute_answer()
        return GroundTruth(
            answer=answer,
            explanation=(
                "The minimum push force to start sliding is approximately F = mu_s * m * g. "
                + "; ".join(
                    f"Object {label}: mu*m = {mu*mass:.3f}"
                    for label, mass, mu in zip(self.LABELS, self.masses, self.mu_s)
                )
                + f". Hardest to push: Object {answer}."
            ),
            latent_params={
                "masses": self.masses,
                "mu_s": self.mu_s,
                "thresholds": [mu * mass for mu, mass in zip(self.mu_s, self.masses)],
            },
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if prediction.answer == self.ground_truth().answer else 0.0
