"""Hanging-beam problem.

A uniform beam is hinged horizontally at one end. A heavy mass hangs from the
free end of the beam. The question: does the beam tip down under the load, or
stay approximately horizontal?

Physical concepts: torque balance, lever arm, hanging loads, static equilibrium.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml


@register_problem("statics")
class HangingBeam(Problem):
    """Does a hinged beam tip down when a mass hangs from its free end?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.beam_length = float(rng.uniform(0.40, 0.70))
        self.beam_mass = float(rng.uniform(0.3, 0.8))
        self.load_mass = float(rng.uniform(0.1, 1.2))
        # Beam thickness.
        self.beam_h = 0.04
        self.beam_w = 0.05
        # Hinge/pivot point height above floor.
        self.pivot_z = 1.0

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        # Pivot body is fixed in space; beam is attached by a hinge at its end.
        half_len = self.beam_length / 2.0
        beam_diaginertia = (
            self.beam_mass / 12.0 * (self.beam_w**2 + self.beam_h**2),
            self.beam_mass / 12.0 * (self.beam_length**2 + self.beam_h**2),
            self.beam_mass / 12.0 * (self.beam_length**2 + self.beam_w**2),
        )

        worldbody_parts = [
            f"""    <body name="pivot" pos="0 0 {self.pivot_z}">
      <geom name="pivot_geom" type="sphere" size="0.03" rgba="0.5 0.5 0.5 1" />
    </body>
""",
            f"""    <body name="beam" pos="{half_len} 0 {self.pivot_z}">
      <joint name="beam_hinge" type="hinge" axis="0 1 0" pos="{-half_len} 0 0" range="-1.57 1.57" damping="0.1" />
      <inertial pos="0 0 0" mass="{self.beam_mass}" diaginertia="{beam_diaginertia[0]} {beam_diaginertia[1]} {beam_diaginertia[2]}" />
      <geom name="beam_geom" type="box" size="{half_len} {self.beam_w/2.0} {self.beam_h/2.0}" rgba="0.8 0.5 0.3 1" />
      <body name="load" pos="{half_len} 0 {-self.beam_h/2.0 - 0.05}">
        <inertial pos="0 0 0" mass="{self.load_mass}" diaginertia="{0.4*self.load_mass*0.05**2} {0.4*self.load_mass*0.05**2} {0.4*self.load_mass*0.05**2}" />
        <geom name="load_geom" type="sphere" size="0.05" rgba="0.2 0.4 0.8 1" />
      </body>
    </body>
""",
        ]

        return build_xml("".join(worldbody_parts), timestep=0.002)

    def _run_outcome(self) -> str:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        # Settle the hinged beam.
        for _ in range(2000):
            mujoco.mj_step(self.model, self.data)

        hinge_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "beam_hinge")
        qpos_addr = self.model.jnt_qposadr[hinge_id]
        final_angle = float(self.data.qpos[qpos_addr])
        self._final_angle_deg = float(np.degrees(final_angle))

        # "Tips" if the free end rotates more than 15 degrees downward.
        outcome = "yes" if abs(self._final_angle_deg) > 15.0 else "no"
        self._outcome = outcome
        return outcome

    def question(self) -> Question:
        return Question(
            text=(
                f"A uniform {self.beam_mass:.2f} kg beam of length {self.beam_length:.2f} m is "
                f"hinged horizontally at one end. A {self.load_mass:.2f} kg mass hangs from the "
                "free end. Does the beam tip down under the load?"
            ),
            answer_type=AnswerType.BOOLEAN,
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=outcome,
            explanation=(
                f"Beam mass = {self.beam_mass:.2f} kg, load mass = {self.load_mass:.2f} kg, "
                f"beam length = {self.beam_length:.2f} m. Final beam angle: "
                f"{self._final_angle_deg:.1f}°. Outcome: {'tips' if outcome == 'yes' else 'stays horizontal'}."
            ),
            latent_params={
                "beam_length": self.beam_length,
                "beam_mass": self.beam_mass,
                "load_mass": self.load_mass,
                "final_angle_deg": self._final_angle_deg,
            },
        )

    def score(self, prediction: Prediction) -> float:
        gt = self.ground_truth().answer
        pred = str(prediction.answer).lower()
        return 1.0 if pred in (gt, "true", "1") else 0.0
