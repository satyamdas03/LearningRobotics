"""Projectile-hit problem.

A ball is launched from ground level with a given speed and angle. Predict
how far from the launch point it lands.

Physical concepts: projectile motion, range equation, independence of mass.
"""
from __future__ import annotations

import math

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml, mjcf_sphere


@register_problem("dynamics")
class ProjectileHit(Problem):
    """How far from the launch point does the projectile land?"""

    GRAVITY = 9.81

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.speed = float(rng.uniform(4.0, 10.0))
        self.angle = float(rng.uniform(np.deg2rad(20), np.deg2rad(70)))
        self.radius = 0.06
        self.mass = float(rng.uniform(0.2, 1.0))

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        z = self.radius + 0.01
        worldbody_parts = [
            mjcf_sphere(
                name="projectile",
                pos=(0.0, 0.0, z),
                radius=self.radius,
                rgba=(0.85, 0.30, 0.25, 1.0),
                mass=self.mass,
                friction=(0.0, 0.0, 0.0),
            ),
        ]
        return build_xml("".join(worldbody_parts), timestep=0.001)

    def _analytic_range(self) -> float:
        """Projectile range on flat ground: R = v^2 sin(2θ) / g."""
        return (self.speed ** 2) * math.sin(2 * self.angle) / self.GRAVITY

    def _simulated_range(self) -> float:
        """Run MuJoCo and measure landing x position."""
        if getattr(self, "_simulated_range_cache", None) is not None:
            return self._simulated_range_cache

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "projectile")
        qvel_addr = self.model.body_dofadr[body_id]

        vx = self.speed * math.cos(self.angle)
        vz = self.speed * math.sin(self.angle)
        self.data.qvel[qvel_addr] = vx
        self.data.qvel[qvel_addr + 2] = vz
        mujoco.mj_forward(self.model, self.data)

        start_x = self.data.xpos[body_id][0]
        start_z = self.data.xpos[body_id][2]

        # Step until projectile lands (z returns to starting height or below).
        for _ in range(3000):
            mujoco.mj_step(self.model, self.data)
            z = self.data.xpos[body_id][2]
            if z < start_z - 0.01:
                break

        end_x = self.data.xpos[body_id][0]
        self._simulated_range_cache = float(end_x - start_x)
        return self._simulated_range_cache

    def question(self) -> Question:
        return Question(
            text=(
                f"A ball is launched from ground level at {self.speed:.2f} m/s "
                f"at an angle of {math.degrees(self.angle):.1f}° above the horizontal. "
                "How far from the launch point does it land (range in meters)?"
            ),
            answer_type=AnswerType.NUMERIC,
            units="m",
        )

    def ground_truth(self) -> GroundTruth:
        analytic = self._analytic_range()
        simulated = self._simulated_range()
        # Prefer analytic unless it is very far from simulated (indicates sim issue).
        if abs(analytic - simulated) > 0.5:
            answer = simulated
        else:
            answer = analytic
        return GroundTruth(
            answer=answer,
            explanation=(
                f"Analytic range = {analytic:.3f} m (v² sin(2θ)/g). "
                f"MuJoCo measured range = {simulated:.3f} m."
            ),
            latent_params={
                "speed": self.speed,
                "angle_deg": float(math.degrees(self.angle)),
                "mass": self.mass,
                "analytic_range": analytic,
                "simulated_range": simulated,
            },
        )

    def score(self, prediction: Prediction) -> float:
        gt = self.ground_truth().answer
        try:
            pred = float(prediction.answer)
        except (TypeError, ValueError):
            return 0.0
        # Tolerance: 10% of true range or 0.2 m, whichever is larger.
        tolerance = max(0.2, 0.1 * abs(gt))
        return 1.0 if abs(pred - gt) <= tolerance else 0.0
