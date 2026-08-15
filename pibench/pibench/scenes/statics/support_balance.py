"""Support-balance problem.

A uniform beam has a point mass placed somewhere along it. The task is to
predict where a pivot/support must be placed so the beam balances horizontally.

Physical concepts: center of mass, torque balance, static equilibrium.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml, mjcf_box


@register_problem("statics")
class SupportBalance(Problem):
    """Where must the support be placed so the beam balances?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        # Beam dimensions (half-size) and mass.
        self.beam_length = 2.0  # full length
        self.beam_width = 0.1
        self.beam_height = 0.05
        self.beam_mass = float(rng.uniform(0.5, 1.0))

        # Point mass placed along the beam at offset from center.
        self.mass_value = float(rng.uniform(0.3, 1.5))
        self.mass_offset = float(rng.uniform(-0.6, 0.6))

        # Candidate support positions (x coordinates in world = beam frame).
        self.candidates = ["left of center", "center", "right of center"]

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        beam_half_length = self.beam_length / 2
        beam_z = 0.3

        worldbody_parts = [
            # Beam with freejoint so it can rotate and translate.
            mjcf_box(
                name="beam",
                pos=(0.0, 0.0, beam_z),
                size=(beam_half_length, self.beam_width, self.beam_height),
                rgba=(0.6, 0.6, 0.6, 1.0),
                mass=self.beam_mass,
            ),
            # Point mass attached to the beam via an equality constraint (weld).
            mjcf_box(
                name="point_mass",
                pos=(self.mass_offset, 0.0, beam_z + self.beam_height + 0.06),
                size=(0.06, 0.06, 0.06),
                rgba=(0.85, 0.30, 0.25, 1.0),
                mass=self.mass_value,
            ),
        ]

        # Static support pillar at the correct balance point.
        support_x = self._balance_x()
        support_height = beam_z - self.beam_height
        worldbody_parts.append(
            f"""    <body name="support" pos="{support_x} 0 {support_height/2}">
      <geom name="support_geom" type="box" size="0.04 {self.beam_width} {support_height/2}" rgba="0.2 0.7 0.3 1" />
    </body>
"""
        )

        xml = build_xml("".join(worldbody_parts), timestep=0.002)
        # Weld the point mass to the beam at the offset.
        weld = f"""  <equality>
    <weld body1="beam" body2="point_mass" anchor="{self.mass_offset} 0 0" />
  </equality>
"""
        xml = xml.replace("</mujoco>", weld + "</mujoco>")
        return xml

    def _balance_x(self) -> float:
        """Analytic balance point: weighted average of beam and point-mass CoMs.

        Beam CoM is at x = 0 (center). Point mass CoM is at x = mass_offset.
        Balance point x = (m_beam * 0 + m_point * mass_offset) / (m_beam + m_point).
        """
        return (self.mass_value * self.mass_offset) / (self.beam_mass + self.mass_value)

    def _run_outcome(self) -> str:
        """Run simulation and check whether beam remains horizontal."""
        if getattr(self, "_simulated_outcome", None) is not None:
            return self._simulated_outcome

        for _ in range(300):
            mujoco.mj_step(self.model, self.data)

        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "beam")
        quat = self.data.xquat[body_id]
        # Convert quaternion to tilt angle about Y (roll would be about X; we care about pitch).
        # For small angles, sin(pitch/2) ≈ quat[2].
        tilt = 2 * np.arcsin(np.clip(quat[2], -1.0, 1.0))
        outcome = "balanced" if abs(tilt) < 0.05 else "not balanced"
        self._simulated_outcome = outcome
        return outcome

    def question(self) -> Question:
        side = "right" if self.mass_offset > 0 else "left"
        return Question(
            text=(
                f"A uniform beam has a {self.mass_value:.2f} kg mass placed "
                f"{abs(self.mass_offset):.2f} m to the {side} of center. "
                f"Where must the support be placed so the beam balances horizontally?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=self.candidates,
        )

    def ground_truth(self) -> GroundTruth:
        balance_x = self._balance_x()
        # Map balance_x to one of the three candidate labels.
        if balance_x < -0.05:
            answer = "left of center"
        elif balance_x > 0.05:
            answer = "right of center"
        else:
            answer = "center"

        simulated = self._run_outcome()
        return GroundTruth(
            answer=answer,
            explanation=(
                f"Beam mass = {self.beam_mass:.2f} kg at center; "
                f"point mass = {self.mass_value:.2f} kg at x = {self.mass_offset:+.2f} m. "
                f"Balance point x = {balance_x:+.3f} m. "
                f"Simulation outcome: {simulated}."
            ),
            latent_params={
                "beam_mass": self.beam_mass,
                "point_mass": self.mass_value,
                "mass_offset": self.mass_offset,
                "balance_x": balance_x,
                "simulated": simulated,
            },
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if prediction.answer == self.ground_truth().answer else 0.0
