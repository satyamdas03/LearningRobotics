"""Gear-turn problem.

Two circular gears are mounted on parallel revolute joints and placed in a
meshed configuration. The first gear is driven by a motor. The question asks
for the rotation direction of the second gear.

Physical concepts: external gear meshing, rotation direction, kinematic
constraints.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.articulated import joint_position
from pibench.utils.mjcf import build_xml


@register_problem("articulated")
class GearTurn(Problem):
    """Which way does the second meshed gear turn?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.radius_a = float(rng.uniform(0.12, 0.22))
        self.radius_b = float(rng.uniform(0.10, 0.20))
        self.mass_a = float(rng.uniform(0.3, 0.8))
        self.mass_b = float(rng.uniform(0.3, 0.8))
        self.applied_torque = float(rng.uniform(0.3, 1.5))
        self.center_distance = self.radius_a + self.radius_b + 0.01
        self.thickness = 0.04

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        r_a = self.radius_a
        r_b = self.radius_b
        t = self.thickness
        m_a = self.mass_a
        m_b = self.mass_b
        d = self.center_distance
        # Cylinder inertia about central Z axis: 0.5 * m * r^2; about X/Y: m/12 * (3r^2 + h^2).
        i_a = f"{0.5*m_a*r_a**2} {m_a/12*(3*r_a**2 + (2*t)**2)} {m_a/12*(3*r_a**2 + (2*t)**2)}"
        i_b = f"{0.5*m_b*r_b**2} {m_b/12*(3*r_b**2 + (2*t)**2)} {m_b/12*(3*r_b**2 + (2*t)**2)}"

        worldbody_parts: list[str] = []

        # Static supports for the gear axles.
        worldbody_parts.append(
            """    <body name="support_a" pos="0 0 0">
      <geom name="support_a_geom" type="cylinder" size="0.03 0.1" rgba="0.4 0.4 0.4 1" />
    </body>
"""
        )
        worldbody_parts.append(
            f"""    <body name="support_b" pos="{d} 0 0">
      <geom name="support_b_geom" type="cylinder" size="0.03 0.1" rgba="0.4 0.4 0.4 1" />
    </body>
"""
        )

        # Gear A driven about Z.
        worldbody_parts.append(
            f"""    <body name="gear_a" pos="0 0 0.05">
      <joint name="gear_a_joint" type="hinge" axis="0 0 1" range="-1e6 1e6" damping="0.01" />
      <inertial pos="0 0 0" mass="{m_a}" diaginertia="{i_a}" />
      <geom name="gear_a_geom" type="cylinder" size="{r_a} {t}" rgba="0.8 0.3 0.2 1" />
    </body>
"""
        )

        # Gear B is meshed with A.  MuJoCo does not have a native gear equality,
        # and contact between smooth cylinders does not reliably transmit torque,
        # so the direction is determined by the gear-meshing principle (external
        # meshing = opposite rotation) rather than by a hard constraint.
        worldbody_parts.append(
            f"""    <body name="gear_b" pos="{d} 0 0.05">
      <joint name="gear_b_joint" type="hinge" axis="0 0 1" range="-1e6 1e6" damping="0.01" />
      <inertial pos="0 0 0" mass="{m_b}" diaginertia="{i_b}" />
      <geom name="gear_b_geom" type="cylinder" size="{r_b} {t}" rgba="0.2 0.3 0.8 1" />
    </body>
"""
        )

        xml = build_xml("".join(worldbody_parts), timestep=0.002)

        actuator = f"""  <actuator>
    <motor joint="gear_a_joint" gear="1" ctrlrange="-{self.applied_torque} {self.applied_torque}" />
  </actuator>
"""
        xml = xml.replace("</mujoco>", actuator + "</mujoco>")
        return xml

    def _run_outcome(self) -> str:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        self.data.ctrl[0] = self.applied_torque
        q_a_initial = joint_position(self.model, self.data, "gear_a_joint")

        steps = 500
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

        q_a_final = joint_position(self.model, self.data, "gear_a_joint")
        delta_a = q_a_final - q_a_initial

        # A positive torque about Z drives gear A counter-clockwise.  Two
        # externally meshed gears rotate in opposite directions, so gear B turns
        # clockwise.  We record the driven gear's rotation as diagnostic data.
        outcome = "clockwise"
        self._outcome = outcome
        self._delta_a = float(np.rad2deg(delta_a))
        return outcome

    def question(self) -> Question:
        return Question(
            text=(
                f"Gear A (radius {self.radius_a:.2f} m) is driven with a torque and begins to "
                f"turn counter-clockwise. It meshes with gear B (radius {self.radius_b:.2f} m). "
                f"Which way does gear B turn?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=["clockwise", "counter-clockwise"],
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=outcome,
            explanation=(
                f"Gear A driven counter-clockwise meshes externally with gear B. "
                f"Gear A rotation: {self._delta_a:.1f}°. "
                f"External meshing produces opposite rotation, so gear B turns {outcome}."
            ),
            latent_params={
                "radius_a": self.radius_a,
                "radius_b": self.radius_b,
                "mass_a": self.mass_a,
                "mass_b": self.mass_b,
                "applied_torque": self.applied_torque,
                "gear_a_rotation_deg": self._delta_a,
            },
        )

    def score(self, prediction: Prediction) -> float:
        gt = self.ground_truth().answer
        return 1.0 if prediction.answer == gt else 0.0
