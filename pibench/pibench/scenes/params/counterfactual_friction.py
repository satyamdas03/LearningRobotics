"""Counterfactual friction problem.

A block rests on an incline and currently does not slide. The question: if the
static-friction coefficient were zero, would the block slide?

Physical concepts: static friction, angle of repose, counterfactual reasoning.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml, mjcf_box, mjcf_incline


@register_problem("params")
class CounterfactualFriction(Problem):
    """If the block's friction were zero, would it slide down the incline?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.angle = float(rng.uniform(np.deg2rad(15), np.deg2rad(40)))
        self.mu_s = float(rng.uniform(0.3, 0.9))
        # Ensure the original block does NOT slide (mu_s > tan(angle)).
        while self.mu_s <= np.tan(self.angle):
            self.mu_s = float(rng.uniform(0.3, 1.2))
        self.block_size = (0.08, 0.08, 0.08)
        self.block_mass = 0.5
        self.slide_threshold = 0.05

        xml = self._make_xml(self.mu_s)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self, mu_s: float) -> str:
        incline_length = 1.0
        incline_thickness = 0.02
        incline_width = 0.4

        worldbody_parts: list[str] = [
            mjcf_incline(
                name="ramp",
                pos=(0.0, 0.0, 0.0),
                size=(incline_length, incline_width, incline_thickness),
                angle=self.angle,
                friction=(0.9, 0.01, 0.01),
            ),
        ]

        # Place block near the center of the ramp.
        ramp_x = -0.2
        ramp_z = incline_thickness + ramp_x * np.tan(self.angle) + self.block_size[2] + 0.01
        worldbody_parts.append(
            mjcf_box(
                name="block",
                pos=(ramp_x, 0.0, ramp_z),
                size=self.block_size,
                rgba=(0.7, 0.4, 0.2, 1.0),
                mass=self.block_mass,
                friction=(mu_s, 0.01, 0.01),
            )
        )

        return build_xml("".join(worldbody_parts), timestep=0.002)

    def _run_outcome(self, mu_s: float) -> str:
        cache_key = mu_s
        if hasattr(self, "_outcomes") and cache_key in self._outcomes:
            return self._outcomes[cache_key]

        xml = self._make_xml(mu_s)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Let it settle, then check whether it slides.
        steps = 600
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
        initial_pos = None
        slid = False
        for _ in range(steps):
            mujoco.mj_step(model, data)
            pos = data.xpos[body_id].copy()
            if initial_pos is None:
                initial_pos = pos
            if float(np.linalg.norm(pos - initial_pos)) > self.slide_threshold:
                slid = True
                break

        outcome = "yes" if slid else "no"
        if not hasattr(self, "_outcomes"):
            self._outcomes = {}
        self._outcomes[cache_key] = outcome
        return outcome

    def _counterfactual_params(self) -> list[str]:
        return ["mu_s"]

    def question(self) -> Question:
        return Question(
            text=(
                f"A {self.block_mass:.2f} kg block rests on a {np.rad2deg(self.angle):.1f}° incline. "
                "It is currently held in place by static friction. "
                "If the coefficient of static friction were zero, would the block slide?"
            ),
            answer_type=AnswerType.BOOLEAN,
        )

    def ground_truth(self) -> GroundTruth:
        original_outcome = self._run_outcome(self.mu_s)
        cf_outcome = self._run_outcome(0.0)

        return GroundTruth(
            answer=cf_outcome,
            explanation=(
                f"Original friction coefficient {self.mu_s:.2f} gives slide outcome: {original_outcome}. "
                "With zero friction, gravity's component along the ramp is unopposed, "
                f"so the block {'slides' if cf_outcome == 'yes' else 'does not slide'}."
            ),
            latent_params={
                "angle_deg": float(np.rad2deg(self.angle)),
                "mu_s": self.mu_s,
                "original_slides": original_outcome,
            },
        )

    def score(self, prediction: Prediction) -> float:
        gt = self.ground_truth().answer
        pred = str(prediction.answer).lower()
        return 1.0 if pred in (gt, "true", "1") else 0.0
