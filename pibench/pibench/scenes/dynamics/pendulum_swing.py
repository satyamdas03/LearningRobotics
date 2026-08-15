"""Pendulum-swing problem.

Two simple pendulums are released from the same angle. One is longer and/or
heavier than the other. Predict which pendulum has the longer period.

Physical concepts: pendulum period ~ sqrt(L/g), independence from mass.
"""
from __future__ import annotations

import math

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml


@register_problem("dynamics")
class PendulumSwing(Problem):
    """Which pendulum has the longer period?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.length_a = float(rng.uniform(0.6, 1.0))
        self.length_b = float(rng.uniform(0.3, 0.5))
        self.mass_a = float(rng.uniform(0.3, 1.0))
        self.mass_b = float(rng.uniform(0.3, 1.0))
        self.release_angle = float(rng.uniform(np.deg2rad(10), np.deg2rad(45)))

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        # Build two pendulums side by side. Each is a single body hinged at the
        # top, with a capsule rod and a sphere bob whose mass is concentrated
        # near the tip via an inertial element.
        worldbody_parts = []
        for name, length, mass, offset_x, color in [
            ("pendulum_a", self.length_a, self.mass_a, -0.6, (0.85, 0.30, 0.25)),
            ("pendulum_b", self.length_b, self.mass_b, 0.6, (0.25, 0.55, 0.85)),
        ]:
            bob_radius = 0.05
            # Treat the pendulum as a point mass at the bob for inertia.
            i = 0.4 * mass * bob_radius ** 2
            worldbody_parts.append(
                f"""    <body name="{name}" pos="{offset_x} 0 {length}">
      <joint name="{name}_joint" type="hinge" axis="0 1 0" pos="0 0 0" damping="0.01" />
      <geom name="{name}_rod" type="capsule" fromto="0 0 0 0 0 {-length}" size="0.01" rgba="0.5 0.5 0.5 1" />
      <inertial pos="0 0 {-length}" mass="{mass}" diaginertia="{i} {i} {i}" />
      <geom name="{name}_bob" type="sphere" size="{bob_radius}" pos="0 0 {-length}" rgba="{color[0]} {color[1]} {color[2]} 1" />
    </body>
"""
            )
        return build_xml("".join(worldbody_parts), timestep=0.002)

    def _analytic_answer(self) -> str:
        """Longer pendulum has longer period (T ≈ 2π sqrt(L/g))."""
        if getattr(self, "_answer", None) is not None:
            return self._answer
        if abs(self.length_a - self.length_b) < 0.05:
            answer = "same"
        else:
            answer = "A" if self.length_a > self.length_b else "B"
        self._answer = answer
        return answer

    def _simulated_period(self, name: str) -> float:
        """Estimate period by measuring time between successive zero crossings."""
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, f"{name}_joint")
        qpos_addr = self.model.jnt_qposadr[joint_id]
        # Set initial angle and reset simulation state.
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[qpos_addr] = self.release_angle
        mujoco.mj_forward(self.model, self.data)

        crossings: list[float] = []
        prev_x = self.data.xpos[body_id][0]
        for _ in range(3000):
            mujoco.mj_step(self.model, self.data)
            x = self.data.xpos[body_id][0]
            if prev_x < 0 and x >= 0:
                crossings.append(self.data.time)
            prev_x = x

        if len(crossings) < 2:
            return float("inf")
        # Period is twice the time between consecutive same-direction crossings.
        periods = [2 * (crossings[i] - crossings[i - 1]) for i in range(1, len(crossings))]
        return float(np.mean(periods))

    def question(self) -> Question:
        return Question(
            text=(
                "Two pendulums are released from the same angle. "
                f"Pendulum A has length {self.length_a:.2f} m and mass {self.mass_a:.2f} kg; "
                f"Pendulum B has length {self.length_b:.2f} m and mass {self.mass_b:.2f} kg. "
                "Which pendulum has the longer period?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=["A", "B", "same"],
        )

    def ground_truth(self) -> GroundTruth:
        answer = self._analytic_answer()
        return GroundTruth(
            answer=answer,
            explanation=(
                f"Period is approximately 2π√(L/g), so it depends mainly on length, not mass. "
                f"Length A = {self.length_a:.2f} m, length B = {self.length_b:.2f} m. "
                f"Longer pendulum: {answer}."
            ),
            latent_params={
                "length_a": self.length_a,
                "length_b": self.length_b,
                "mass_a": self.mass_a,
                "mass_b": self.mass_b,
                "release_angle_deg": float(math.degrees(self.release_angle)),
            },
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if prediction.answer == self.ground_truth().answer else 0.0
