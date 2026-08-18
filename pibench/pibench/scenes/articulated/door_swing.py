"""Door-swing problem.

A door is mounted on a vertical hinge with static friction. A constant pushing
torque is applied via a motor actuator. The question: does the door swing open
or stay shut?

Physical concepts: revolute joints, hinge friction, torque, angular motion.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.articulated import joint_position
from pibench.utils.mjcf import build_xml


@register_problem("articulated")
class DoorSwing(Problem):
    """Does a pushed door swing open or stick at the hinge?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.door_mass = float(rng.uniform(3.0, 12.0))
        self.frictionloss = float(rng.uniform(1.0, 8.0))
        self.applied_torque = float(rng.uniform(0.5, 12.0))
        self.open_threshold = np.deg2rad(20.0)
        self.door_width = 0.45
        self.door_height = 1.0
        self.door_thickness = 0.04

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        w = self.door_width
        h = self.door_height
        t = self.door_thickness
        worldbody_parts: list[str] = []

        # Door frame (static).
        worldbody_parts.append(
            """    <body name=\"frame\" pos=\"0 0 0\">
      <geom name=\"frame_geom\" type=\"box\" size=\"0.05 0.05 1.1\" rgba=\"0.4 0.4 0.4 1\" />
    </body>
"""
        )

        # Door hinged at one edge about the world Z axis.
        worldbody_parts.append(
            f"""    <body name=\"door\" pos=\"{w} 0 {h}\">
      <joint name=\"door_joint\" type=\"hinge\" axis=\"0 0 1\" pos=\"{-w} 0 0\" range=\"-1.57 1.57\" frictionloss=\"{self.frictionloss}\" damping=\"0.05\" />
      <inertial pos=\"0 0 0\" mass=\"{self.door_mass}\" diaginertia=\"{self.door_mass/12*(h**2+t**2)} {self.door_mass/12*(w**2+t**2)} {self.door_mass/12*(w**2+h**2)}\" />
      <geom name=\"door_geom\" type=\"box\" size=\"{w} {t} {h}\" rgba=\"0.6 0.3 0.2 1\" />
    </body>
"""
        )

        xml = build_xml("".join(worldbody_parts), timestep=0.002)
        actuator = f"""  <actuator>
    <motor joint=\"door_joint\" gear=\"1\" ctrlrange=\"0 {self.applied_torque}\" />
  </actuator>
"""
        xml = xml.replace("</mujoco>", actuator + "</mujoco>")
        return xml

    def _run_outcome(self) -> str:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        self.data.ctrl[0] = self.applied_torque
        initial_q = joint_position(self.model, self.data, "door_joint")

        steps = 1000
        max_angle = abs(initial_q)
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            max_angle = max(max_angle, abs(joint_position(self.model, self.data, "door_joint")))

        opened = max_angle - abs(initial_q) > self.open_threshold
        outcome = "yes" if opened else "no"
        self._outcome = outcome
        self._max_angle = float(np.rad2deg(max_angle - abs(initial_q)))
        return outcome

    def question(self) -> Question:
        return Question(
            text=(
                f"A {self.door_mass:.2f} kg door is pushed with a constant torque of "
                f"{self.applied_torque:.2f} Nm. The hinge has {self.frictionloss:.2f} Nm "
                f"of static friction. Does the door swing open?"
            ),
            answer_type=AnswerType.BOOLEAN,
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=outcome,
            explanation=(
                f"Door mass {self.door_mass:.2f} kg, applied torque {self.applied_torque:.2f} Nm, "
                f"hinge friction {self.frictionloss:.2f} Nm. "
                f"Maximum angular displacement: {self._max_angle:.1f}°. "
                f"Door {'swings open' if outcome == 'yes' else 'stays shut'}."
            ),
            latent_params={
                "door_mass": self.door_mass,
                "applied_torque": self.applied_torque,
                "frictionloss": self.frictionloss,
                "max_angle_deg": self._max_angle,
            },
        )

    def score(self, prediction: Prediction) -> float:
        gt = self.ground_truth().answer
        pred = str(prediction.answer).lower()
        return 1.0 if pred in (gt, "true", "1") else 0.0
