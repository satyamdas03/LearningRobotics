"""Topple-direction problem.

A single block tower is built with its center of mass offset to one side.
The base platform is tilted. Predict which way the tower topples.

Physical concepts: center of mass, support polygon, toppling direction.
"""
from __future__ import annotations

import math

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml, mjcf_box


@register_problem("statics")
class ToppleDirection(Problem):
    """Which way does the off-center tower topple?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.tower_height = int(rng.integers(3, 6))
        self.base_size = (0.15, 0.15, 0.10)  # half-size
        self.block_mass = float(rng.uniform(0.5, 1.0))

        # CoM offset created by stacking blocks with increasing overhang to one side.
        self.overhang_direction = rng.choice([-1.0, 1.0])
        self.max_overhang = float(rng.uniform(0.03, 0.12))

        self.tilt_angle = float(rng.uniform(np.deg2rad(10), np.deg2rad(25)))
        # Tilt direction: positive = high end toward +X.
        self.tilt_direction = rng.choice([-1.0, 1.0])

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        worldbody_parts = []
        platform_length = 2.0
        platform_width = 0.8
        platform_thickness = 0.05

        # Platform tilted about Y axis.
        worldbody_parts.append(
            f"""    <body name="platform" pos="0 0 {platform_thickness}" euler="0 {self.tilt_direction * self.tilt_angle} 0">
      <geom name="platform_geom" type="box" size="{platform_length} {platform_width} {platform_thickness}" rgba="0.4 0.4 0.4 1" />
    </body>
"""
        )

        # Stack blocks with progressive overhang.
        z = platform_thickness
        center_x = 0.0
        for i in range(self.tower_height):
            overhang = (i / max(1, self.tower_height - 1)) * self.max_overhang * self.overhang_direction
            x = center_x + overhang
            z += self.base_size[2]
            worldbody_parts.append(
                mjcf_box(
                    name=f"block_{i}",
                    pos=(x, 0.0, z),
                    size=self.base_size,
                    rgba=(0.85, 0.55, 0.25, 1.0),
                    mass=self.block_mass,
                    friction=(0.8, 0.02, 0.02),
                )
            )
            z += self.base_size[2]

        return build_xml("".join(worldbody_parts), timestep=0.002)

    def _analytic_direction(self) -> str:
        """Predict topple direction from net CoM offset and platform tilt.

        The tower tends to fall toward the side where the combined CoM is
        further outside the support polygon. Overhang pushes CoM toward
        overhang_direction; tilt pushes the high side (opposite tilt_direction).
        """
        # Effective CoM offset: overhang + tilt effect (high side is -tilt_direction).
        tilt_effect = -self.tilt_direction * math.sin(self.tilt_angle) * self.tower_height * self.base_size[2] * 0.5
        net_offset = self.overhang_direction * self.max_overhang + tilt_effect
        if abs(net_offset) < 0.01:
            return "neither"
        return "right" if net_offset > 0 else "left"

    def _simulated_direction(self) -> str:
        """Run MuJoCo and measure top block displacement in X."""
        if getattr(self, "_simulated_direction_cache", None) is not None:
            return self._simulated_direction_cache

        top_name = f"block_{self.tower_height - 1}"
        body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, top_name)
        start = self.data.xpos[body_id].copy()

        for _ in range(300):
            mujoco.mj_step(self.model, self.data)

        end = self.data.xpos[body_id].copy()
        dx = end[0] - start[0]
        if abs(dx) < 0.03:
            direction = "neither"
        else:
            direction = "right" if dx > 0 else "left"
        self._simulated_direction_cache = direction
        return direction

    def question(self) -> Question:
        return Question(
            text=(
                "An off-center block tower stands on a tilted platform. "
                "Which way does the tower topple?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=["left", "right", "neither"],
        )

    def ground_truth(self) -> GroundTruth:
        analytic = self._analytic_direction()
        simulated = self._simulated_direction()
        answer = analytic if analytic == simulated else simulated
        return GroundTruth(
            answer=answer,
            explanation=(
                f"Tower has {self.tower_height} blocks with overhang toward "
                f"{'right' if self.overhang_direction > 0 else 'left'} and platform tilt toward "
                f"{'right' if self.tilt_direction > 0 else 'left'}. "
                f"Predicted: {analytic}; simulated: {simulated}."
            ),
            latent_params={
                "tower_height": self.tower_height,
                "overhang_direction": float(self.overhang_direction),
                "max_overhang": self.max_overhang,
                "tilt_angle_deg": float(math.degrees(self.tilt_angle)),
                "tilt_direction": float(self.tilt_direction),
            },
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if prediction.answer == self.ground_truth().answer else 0.0
