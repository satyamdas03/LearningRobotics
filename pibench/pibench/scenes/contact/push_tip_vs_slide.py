"""Push-tip-vs-slide problem.

A block stands on a flat surface and is pushed horizontally at a given height.
The question: does it tip over, or slide along the floor?

Physical concepts: moment balance, static/kinetic friction, line of action.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.contact import (
    body_z_axis_deviation,
    mjcf_pusher,
    run_with_pusher,
)
from pibench.utils.mjcf import build_xml, mjcf_box


@register_problem("contact")
class PushTipVsSlide(Problem):
    """When pushed at height h, does the block tip or slide?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        # Block half-sizes: width (x), depth (y), height (z).
        self.half_width = float(rng.uniform(0.08, 0.18))
        self.half_depth = float(rng.uniform(0.08, 0.18))
        self.half_height = float(rng.uniform(0.12, 0.30))
        self.block_mass = float(rng.uniform(0.5, 2.0))

        # Push height measured from the floor (base) up the side face.
        max_height = 2.0 * self.half_height
        self.push_height = float(rng.uniform(0.05, max_height - 0.02))

        # Pusher speed and friction.
        self.pusher_speed = 0.3
        self.floor_friction = (0.45, 0.02, 0.02)

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        block_z = self.half_height + 0.01
        pusher_half = (0.02, self.half_depth * 0.8, self.push_height * 0.25 + 0.01)
        pusher_x = -(self.half_width + pusher_half[0] + 0.01)
        pusher_z = self.push_height

        worldbody_parts = [
            mjcf_box(
                name="block",
                pos=(0.0, 0.0, block_z),
                size=(self.half_width, self.half_depth, self.half_height),
                rgba=(0.85, 0.30, 0.25, 1.0),
                mass=self.block_mass,
                friction=self.floor_friction,
            ),
        ]

        pusher_body, pusher_actuator = mjcf_pusher(
            name="pusher",
            pos=(pusher_x, 0.0, pusher_z),
            size=pusher_half,
            axis=(1.0, 0.0, 0.0),
            max_speed=1.0,
        )
        worldbody_parts.append(pusher_body)

        xml = build_xml("".join(worldbody_parts), timestep=0.002)
        xml = xml.replace("</mujoco>", f"  <actuator>\n{pusher_actuator}  </actuator>\n</mujoco>")
        return xml

    def _run_outcome(self) -> str:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        self._max_tilt = 0.0

        def _record(_model: mujoco.MjModel, data: mujoco.MjData, _step: int) -> None:
            self._max_tilt = max(self._max_tilt, body_z_axis_deviation(_model, data, "block"))

        run_with_pusher(
            self.model,
            self.data,
            pusher_joint_name="pusher_joint",
            speed=self.pusher_speed,
            steps=600,
            callbacks=[_record],
        )

        # A block is "tipped" if its Z axis deviates by more than ~12°.
        outcome = "tip" if self._max_tilt > 0.20 else "slide"
        self._outcome = outcome
        return outcome

    def question(self) -> Question:
        return Question(
            text=(
                f"A {self.block_mass:.2f} kg block "
                f"(width {2*self.half_width:.2f} m, height {2*self.half_height:.2f} m) "
                f"is pushed horizontally at height {self.push_height:.2f} m from the floor. "
                "Does it tip over or slide?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=["tip", "slide"],
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=outcome,
            explanation=(
                f"Block pushed at h={self.push_height:.2f} m on a {2*self.half_width:.2f} m wide base. "
                f"Maximum observed tilt: {np.rad2deg(self._max_tilt):.1f}°. "
                f"Outcome: {outcome}."
            ),
            latent_params={
                "block_mass": self.block_mass,
                "half_width": self.half_width,
                "half_height": self.half_height,
                "push_height": self.push_height,
                "max_tilt_deg": float(np.rad2deg(self._max_tilt)),
            },
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if prediction.answer == self.ground_truth().answer else 0.0
