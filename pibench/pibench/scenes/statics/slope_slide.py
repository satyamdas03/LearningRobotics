"""Slope-slide problem.

A block sits on an inclined plane. Given the incline angle and the coefficient
of static friction, predict whether the block slides down.

Physical concepts: static friction, gravity component, angle of repose.
"""
from __future__ import annotations

import math

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml, mjcf_box, mjcf_incline


@register_problem("statics")
class SlopeSlide(Problem):
    """Will the block slide down the incline?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        # Incline angle in radians.
        self.angle = float(rng.uniform(np.deg2rad(15), np.deg2rad(60)))
        # Coefficient of static friction between block and incline.
        self.mu_s = float(rng.uniform(0.05, 0.95))

        # Block dimensions.
        self.block_size = (0.12, 0.12, 0.12)
        self.block_mass = float(rng.uniform(0.5, 1.5))

        # Incline dimensions: length, width, thickness.
        self.incline_length = 1.5
        self.incline_width = 0.6
        self.incline_thickness = 0.04

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        # Incline centered at origin, high end toward +X, low end toward -X.
        # The surface of the incline sits at z = 0 locally (because box half-size
        # in z is thickness). We position the incline so its bottom-front corner
        # is near z = 0 in world, and the block sits near the top.
        l = self.incline_length
        h = self.incline_thickness
        angle = self.angle

        # Incline body position: center of the box.
        # We want the low end (x = -l cosθ, z = -l sinθ) to touch the floor near z=0.
        center_x = l * math.cos(angle) * 0.5
        center_z = l * math.sin(angle) * 0.5 + h

        worldbody_parts = [
            mjcf_incline(
                name="slope",
                pos=(center_x, 0.0, center_z),
                size=(l, self.incline_width, h),
                angle=angle,
                rgba=(0.4, 0.4, 0.4, 1.0),
                friction=(self.mu_s, 0.02, 0.02),
            )
        ]

        # Place block near top of incline, slightly above surface.
        # Top of incline: x ≈ +l cosθ, z ≈ +l sinθ.
        block_x = (l - 0.3) * math.cos(angle)
        block_z = (l - 0.3) * math.sin(angle) + self.block_size[2] + h
        worldbody_parts.append(
            mjcf_box(
                name="block",
                pos=(block_x, 0.0, block_z),
                size=self.block_size,
                rgba=(0.85, 0.30, 0.25, 1.0),
                mass=self.block_mass,
                friction=(self.mu_s, 0.02, 0.02),
            )
        )

        return build_xml("".join(worldbody_parts), timestep=0.002)

    def _compute_answer(self) -> str:
        """Compare incline angle to angle of repose atan(mu_s).

        The block slides if the component of gravity along the ramp exceeds
        the maximum static friction force. This is a deterministic criterion
        and is the intended physical concept for this scene.
        """
        if getattr(self, "_answer", None) is not None:
            return self._answer
        angle_of_repose = math.atan(self.mu_s)
        answer = "yes" if self.angle > angle_of_repose else "no"
        self._answer = answer
        return answer

    def question(self) -> Question:
        return Question(
            text=f"A block rests on a {math.degrees(self.angle):.1f}° incline. "
                 f"The coefficient of static friction is {self.mu_s:.2f}. Does the block slide down?",
            answer_type=AnswerType.BOOLEAN,
        )

    def ground_truth(self) -> GroundTruth:
        answer = self._compute_answer()
        return GroundTruth(
            answer=answer,
            explanation=(
                f"Incline angle = {math.degrees(self.angle):.1f}°. "
                f"Angle of repose = {math.degrees(math.atan(self.mu_s)):.1f}°. "
                f"Block {'slides' if answer == 'yes' else 'does not slide'}."
            ),
            latent_params={
                "angle_deg": float(math.degrees(self.angle)),
                "mu_s": self.mu_s,
                "block_mass": self.block_mass,
            },
        )

    def score(self, prediction: Prediction) -> float:
        gt = self.ground_truth().answer
        pred = str(prediction.answer).lower()
        return 1.0 if pred in (gt, "true", "1") else 0.0
