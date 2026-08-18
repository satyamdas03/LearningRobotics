"""Drawer-pull problem.

A drawer is mounted on a prismatic slide joint with static friction. A constant
pulling force is applied via a motor actuator. The question: does the drawer
open or jam?

Physical concepts: prismatic joints, static friction, actuators, work.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.articulated import body_id, joint_position
from pibench.utils.mjcf import build_xml


@register_problem("articulated")
class DrawerPull(Problem):
    """Does a pulled drawer open or jam against static friction?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.drawer_mass = float(rng.uniform(0.5, 2.0))
        self.frictionloss = float(rng.uniform(2.0, 8.0))
        self.applied_force = float(rng.uniform(1.0, 10.0))
        self.open_threshold = 0.05
        self.drawer_size = (0.20, 0.15, 0.08)

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        dx, dy, dz = self.drawer_size
        worldbody_parts: list[str] = []

        # Cabinet frame: static walls that visually surround the drawer slot.
        wall_thickness = 0.02
        # Top and bottom plates.
        worldbody_parts.append(
            f"""    <body name=\"cabinet_top\" pos=\"0 0 {dz*2+wall_thickness}\">
      <geom name=\"cabinet_top_geom\" type=\"box\" size=\"{dx} {dy} {wall_thickness}\" rgba=\"0.5 0.5 0.5 1\" />
    </body>
"""
        )
        worldbody_parts.append(
            f"""    <body name=\"cabinet_bottom\" pos=\"0 0 {-wall_thickness}\">
      <geom name=\"cabinet_bottom_geom\" type=\"box\" size=\"{dx} {dy} {wall_thickness}\" rgba=\"0.5 0.5 0.5 1\" />
    </body>
"""
        )

        # Drawer on a prismatic joint along +X.
        worldbody_parts.append(
            f"""    <body name=\"drawer\" pos=\"0 0 {dz}\">
      <joint name=\"drawer_joint\" type=\"slide\" axis=\"1 0 0\" range=\"0 0.5\" frictionloss=\"{self.frictionloss}\" damping=\"0.01\" />
      <inertial pos=\"0 0 0\" mass=\"{self.drawer_mass}\" diaginertia=\"{self.drawer_mass/12*(dy**2+dz**2)} {self.drawer_mass/12*(dx**2+dz**2)} {self.drawer_mass/12*(dx**2+dy**2)}\" />
      <geom name=\"drawer_geom\" type=\"box\" size=\"{dx} {dy} {dz}\" rgba=\"0.7 0.4 0.2 1\" />
      <geom name=\"drawer_handle\" type=\"sphere\" size=\"0.03\" pos=\"{dx+0.05} 0 0\" rgba=\"0.3 0.3 0.3 1\" />
    </body>
"""
        )

        xml = build_xml("".join(worldbody_parts), timestep=0.002)
        actuator = f"""  <actuator>
    <motor joint=\"drawer_joint\" gear=\"1\" ctrlrange=\"0 {self.applied_force}\" />
  </actuator>
"""
        xml = xml.replace("</mujoco>", actuator + "</mujoco>")
        return xml

    def _run_outcome(self) -> str:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        # Apply a constant pulling force.
        self.data.ctrl[0] = self.applied_force
        initial_q = joint_position(self.model, self.data, "drawer_joint")

        steps = 1000
        max_q = initial_q
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            max_q = max(max_q, joint_position(self.model, self.data, "drawer_joint"))

        opened = max_q - initial_q > self.open_threshold
        outcome = "yes" if opened else "no"
        self._outcome = outcome
        self._max_displacement = float(max_q - initial_q)
        return outcome

    def question(self) -> Question:
        return Question(
            text=(
                f"A {self.drawer_mass:.2f} kg drawer is pulled with a constant force of "
                f"{self.applied_force:.2f} N. The slide joint has {self.frictionloss:.2f} N "
                f"of static friction. Does the drawer open?"
            ),
            answer_type=AnswerType.BOOLEAN,
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=outcome,
            explanation=(
                f"Drawer mass {self.drawer_mass:.2f} kg, applied force {self.applied_force:.2f} N, "
                f"static friction {self.frictionloss:.2f} N. "
                f"Maximum displacement: {self._max_displacement:.4f} m. "
                f"Drawer {'opens' if outcome == 'yes' else 'jams'}."
            ),
            latent_params={
                "drawer_mass": self.drawer_mass,
                "applied_force": self.applied_force,
                "frictionloss": self.frictionloss,
                "max_displacement": self._max_displacement,
            },
        )

    def score(self, prediction: Prediction) -> float:
        gt = self.ground_truth().answer
        pred = str(prediction.answer).lower()
        return 1.0 if pred in (gt, "true", "1") else 0.0
