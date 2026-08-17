"""Slip-grip problem.

A parallel-jaw gripper squeezes a block with a given normal force and then
tries to lift it. The question: does the block lift, or do the fingers slip
off and the block stays on the ground?

Physical concepts: Coulomb friction, grip force, weight, static equilibrium.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml, mjcf_box


@register_problem("contact")
class SlipGrip(Problem):
    """Will the gripper lift the block or let it slip?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.block_mass = float(rng.uniform(0.3, 2.0))
        self.mu_s = float(rng.uniform(0.15, 0.75))
        self.grip_force = float(rng.uniform(5.0, 25.0))
        self.block_size = (0.08, 0.08, 0.16)  # half-sizes

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        block_z = self.block_size[2] + 0.01
        worldbody_parts = [
            mjcf_box(
                name="block",
                pos=(0.0, 0.0, block_z),
                size=self.block_size,
                rgba=(0.85, 0.30, 0.25, 1.0),
                mass=self.block_mass,
                friction=(self.mu_s, 0.01, 0.01),
            ),
        ]

        # Static gripper fingers around the block (visual / collision representation).
        finger_thickness = 0.015
        finger_width = 0.04
        finger_height = self.block_size[2] * 1.5
        gap = self.block_size[1] + 0.005
        for side, y in [("left", gap), ("right", -gap)]:
            worldbody_parts.append(
                f"""    <body name="finger_{side}" pos="0 {y} {block_z}">
      <geom name="finger_{side}_geom" type="box" size="{finger_width} {finger_thickness} {finger_height}" rgba="0.3 0.3 0.3 1" />
    </body>
"""
            )

        return build_xml("".join(worldbody_parts), timestep=0.002)

    def _compute_answer(self) -> str:
        if getattr(self, "_answer", None) is not None:
            return self._answer
        # Two fingers share the load; total upward friction must exceed weight.
        total_friction_capacity = 2.0 * self.mu_s * self.grip_force
        weight = self.block_mass * 9.81
        self._answer = "lift" if total_friction_capacity >= weight else "slip"
        return self._answer

    def question(self) -> Question:
        return Question(
            text=(
                f"A parallel-jaw gripper squeezes a {self.block_mass:.2f} kg block with a normal "
                f"force of {self.grip_force:.2f} N per finger. The coefficient of static friction "
                f"between the fingers and the block is {self.mu_s:.2f}. Does the gripper lift the "
                "block, or do the fingers slip?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=["lift", "slip"],
        )

    def ground_truth(self) -> GroundTruth:
        answer = self._compute_answer()
        total_friction = 2.0 * self.mu_s * self.grip_force
        weight = self.block_mass * 9.81
        return GroundTruth(
            answer=answer,
            explanation=(
                f"Block weight = {weight:.2f} N. Total available friction = 2 * {self.mu_s:.2f} * "
                f"{self.grip_force:.2f} N = {total_friction:.2f} N. "
                f"{'Friction can support weight' if answer == 'lift' else 'Friction is insufficient'}: {answer}."
            ),
            latent_params={
                "block_mass": self.block_mass,
                "grip_force": self.grip_force,
                "mu_s": self.mu_s,
                "weight": weight,
                "total_friction_capacity": total_friction,
            },
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if prediction.answer == self.ground_truth().answer else 0.0
