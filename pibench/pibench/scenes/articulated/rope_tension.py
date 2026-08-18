"""Rope-tension problem.

Two masses hang from a rope passing over a fixed pulley. The question asks which
mass descends once the system is released.

Physical concepts: tension, pulleys, constrained motion, gravity.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.articulated import body_id
from pibench.utils.mjcf import build_xml


@register_problem("articulated")
class RopeTension(Problem):
    """Which side of a pulley descends?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.mass_a = float(rng.uniform(0.8, 3.0))
        self.mass_b = float(rng.uniform(0.8, 3.0))
        self.pulley_height = 1.5
        self.half_span = 0.5
        self.box_size = (0.10, 0.10, 0.10)
        self.rest_length = 2.2
        self.stiffness = 2000.0
        self.damping = 20.0

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        s = self.box_size
        m_a = self.mass_a
        m_b = self.mass_b
        i = lambda m: f"{m/12*(s[1]**2+s[2]**2)} {m/12*(s[0]**2+s[2]**2)} {m/12*(s[0]**2+s[1]**2)}"

        worldbody_parts: list[str] = []

        # Static pulley.
        worldbody_parts.append(
            f"""    <body name="pulley" pos="0 0 {self.pulley_height}">
      <geom name="pulley_geom" type="cylinder" size="0.08 0.04" euler="0 1.57 0" rgba="0.5 0.5 0.5 1" />
      <site name="site_p" pos="0 0 0" />
    </body>
"""
        )

        # Hanging masses at equal initial heights.
        initial_z = self.pulley_height - 0.5
        worldbody_parts.append(
            f"""    <body name="mass_a" pos="-{self.half_span} 0 {initial_z}">
      <freejoint />
      <inertial pos="0 0 0" mass="{m_a}" diaginertia="{i(m_a)}" />
      <geom name="mass_a_geom" type="box" size="{s[0]} {s[1]} {s[2]}" rgba="0.8 0.3 0.2 1" />
      <site name="site_a" pos="0 0 {s[2]}" />
    </body>
"""
        )
        worldbody_parts.append(
            f"""    <body name="mass_b" pos="{self.half_span} 0 {initial_z}">
      <freejoint />
      <inertial pos="0 0 0" mass="{m_b}" diaginertia="{i(m_b)}" />
      <geom name="mass_b_geom" type="box" size="{s[0]} {s[1]} {s[2]}" rgba="0.2 0.3 0.8 1" />
      <site name="site_b" pos="0 0 {s[2]}" />
    </body>
"""
        )

        xml = build_xml("".join(worldbody_parts), timestep=0.002)
        tendon = f"""  <tendon>
    <spatial name="rope" stiffness="{self.stiffness}" damping="{self.damping}" range="0 {self.rest_length}" rgba="0.9 0.7 0.2 1">
      <site site="site_a" />
      <site site="site_p" />
      <site site="site_b" />
    </spatial>
  </tendon>
"""
        xml = xml.replace("</mujoco>", tendon + "</mujoco>")
        return xml

    def _run_outcome(self) -> str:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        a_id = body_id(self.model, "mass_a")
        b_id = body_id(self.model, "mass_b")

        # Let the system settle.
        steps = 2000
        z_a_samples: list[float] = []
        z_b_samples: list[float] = []
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            z_a_samples.append(float(self.data.xpos[a_id, 2]))
            z_b_samples.append(float(self.data.xpos[b_id, 2]))

        # Average the last few samples to suppress oscillation.
        avg_a = float(np.mean(z_a_samples[-100:]))
        avg_b = float(np.mean(z_b_samples[-100:]))

        diff = avg_b - avg_a
        if diff > 0.05:
            outcome = "A"
        elif diff < -0.05:
            outcome = "B"
        else:
            outcome = "same"

        self._outcome = outcome
        self._final_z_a = avg_a
        self._final_z_b = avg_b
        return outcome

    def question(self) -> Question:
        return Question(
            text=(
                f"Two masses hang from a rope over a pulley. Mass A is {self.mass_a:.2f} kg "
                f"and mass B is {self.mass_b:.2f} kg. Which mass descends?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=["A", "B", "same"],
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=outcome,
            explanation=(
                f"Mass A = {self.mass_a:.2f} kg, Mass B = {self.mass_b:.2f} kg. "
                f"Final heights: A={self._final_z_a:.3f} m, B={self._final_z_b:.3f} m. "
                f"Outcome: {outcome} descends."
            ),
            latent_params={
                "mass_a": self.mass_a,
                "mass_b": self.mass_b,
                "final_z_a": self._final_z_a,
                "final_z_b": self._final_z_b,
            },
        )

    def score(self, prediction: Prediction) -> float:
        gt = self.ground_truth().answer
        return 1.0 if prediction.answer == gt else 0.0
