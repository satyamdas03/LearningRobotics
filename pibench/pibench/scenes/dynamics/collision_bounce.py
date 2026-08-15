"""Collision-bounce problem.

A moving ball collides with a stationary ball on a frictionless surface.
Predict which ball moves faster after a perfectly elastic head-on collision.

Physical concepts: conservation of momentum, conservation of kinetic energy.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml, mjcf_sphere


@register_problem("dynamics")
class CollisionBounce(Problem):
    """After the collision, which ball moves faster?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.mass_a = float(rng.uniform(0.5, 2.0))
        self.mass_b = float(rng.uniform(0.5, 2.0))
        self.velocity_a = float(rng.uniform(1.0, 3.0))
        self.radius = 0.08

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        gap = 0.02
        start_x_a = -0.6
        start_x_b = start_x_a + 2 * self.radius + gap
        z = self.radius + 0.01

        worldbody_parts = [
            mjcf_sphere(
                name="ball_a",
                pos=(start_x_a, 0.0, z),
                radius=self.radius,
                rgba=(0.85, 0.30, 0.25, 1.0),
                mass=self.mass_a,
                friction=(0.0, 0.0, 0.0),
            ),
            mjcf_sphere(
                name="ball_b",
                pos=(start_x_b, 0.0, z),
                radius=self.radius,
                rgba=(0.25, 0.55, 0.85, 1.0),
                mass=self.mass_b,
                friction=(0.0, 0.0, 0.0),
            ),
        ]
        return build_xml("".join(worldbody_parts), timestep=0.001)

    def _analytic_answer(self) -> str:
        """1D elastic collision velocities.

        v1' = (m1 - m2)/(m1 + m2) * v1
        v2' = (2*m1)/(m1 + m2) * v1
        """
        if getattr(self, "_answer", None) is not None:
            return self._answer

        m1, m2, v1 = self.mass_a, self.mass_b, self.velocity_a
        v1_final = (m1 - m2) / (m1 + m2) * v1
        v2_final = (2 * m1) / (m1 + m2) * v1

        if abs(abs(v1_final) - abs(v2_final)) < 0.05:
            answer = "same"
        else:
            answer = "A" if abs(v1_final) > abs(v2_final) else "B"
        self._answer = answer
        return answer

    def _simulated_speeds(self) -> tuple[float, float]:
        """Run MuJoCo and measure post-collision speeds."""
        if getattr(self, "_simulated_speeds_cache", None) is not None:
            return self._simulated_speeds_cache

        # Set initial velocity of ball A.
        body_a_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball_a")
        qvel_addr = self.model.body_dofadr[body_a_id]
        self.data.qvel[qvel_addr] = self.velocity_a
        mujoco.mj_forward(self.model, self.data)

        body_b_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "ball_b")

        for _ in range(500):
            mujoco.mj_step(self.model, self.data)

        # Measure average speed over a short window after collision.
        speed_a = float(np.linalg.norm(self.data.qvel[qvel_addr:qvel_addr + 3]))
        qvel_b = self.model.body_dofadr[body_b_id]
        speed_b = float(np.linalg.norm(self.data.qvel[qvel_b:qvel_b + 3]))

        self._simulated_speeds_cache = (speed_a, speed_b)
        return self._simulated_speeds_cache

    def question(self) -> Question:
        return Question(
            text=(
                f"Ball A (mass {self.mass_a:.2f} kg) moves at {self.velocity_a:.2f} m/s "
                f"toward stationary Ball B (mass {self.mass_b:.2f} kg) on a frictionless surface. "
                "After a perfectly elastic head-on collision, which ball moves faster?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=["A", "B", "same"],
        )

    def ground_truth(self) -> GroundTruth:
        answer = self._analytic_answer()
        return GroundTruth(
            answer=answer,
            explanation=(
                f"Elastic 1D collision: v_A' = {(self.mass_a - self.mass_b)/(self.mass_a + self.mass_b)*self.velocity_a:.3f} m/s, "
                f"v_B' = {2*self.mass_a/(self.mass_a + self.mass_b)*self.velocity_a:.3f} m/s. "
                f"Faster ball: {answer}."
            ),
            latent_params={
                "mass_a": self.mass_a,
                "mass_b": self.mass_b,
                "velocity_a": self.velocity_a,
            },
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if prediction.answer == self.ground_truth().answer else 0.0
