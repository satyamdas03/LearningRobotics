"""Stack-stability problem.

A stack of blocks is tapped from the side by a moving ball. The question: does
the stack remain standing after the tap?

Physical concepts: impulse transfer, support polygon, center of mass, stacking.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.contact import body_id
from pibench.utils.mjcf import build_xml, mjcf_box, mjcf_sphere


@register_problem("contact")
class StackStability(Problem):
    """Does a block stack survive a side tap?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.n_blocks = int(rng.integers(3, 6))
        self.block_size = (0.10, 0.10, 0.10)  # half-sizes
        self.block_mass = float(rng.uniform(0.3, 0.8))
        self.ball_mass = float(rng.uniform(0.2, 0.6))
        self.ball_radius = 0.05
        self.ball_speed = float(rng.uniform(0.6, 2.2))
        # Height at which the ball strikes the stack.
        self.impact_height = float(rng.uniform(0.10, max(0.15, self.n_blocks * 0.08)))

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        worldbody_parts: list[str] = []

        # Stack blocks vertically at the origin.
        z = 0.01
        for i in range(self.n_blocks):
            z += self.block_size[2]
            worldbody_parts.append(
                mjcf_box(
                    name=f"block_{i}",
                    pos=(0.0, 0.0, z),
                    size=self.block_size,
                    rgba=(0.85, 0.30, 0.25, 1.0),
                    mass=self.block_mass,
                    friction=(0.6, 0.02, 0.02),
                )
            )
            z += self.block_size[2]

        # Impacting ball approaches from the -X side.
        ball_x = -(self.block_size[0] + self.ball_radius + 0.30)
        ball_z = max(self.ball_radius + 0.01, self.impact_height)
        worldbody_parts.append(
            mjcf_sphere(
                name="ball",
                pos=(ball_x, 0.0, ball_z),
                radius=self.ball_radius,
                rgba=(0.25, 0.55, 0.85, 1.0),
                mass=self.ball_mass,
                friction=(0.3, 0.01, 0.01),
            )
        )

        return build_xml("".join(worldbody_parts), timestep=0.002)

    def _run_outcome(self) -> str:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        # Launch the ball toward +X.
        ball_bid = body_id(self.model, "ball")
        ball_qvel = self.model.body_dofadr[ball_bid]
        self.data.qvel[ball_qvel] = self.ball_speed
        mujoco.mj_forward(self.model, self.data)

        # Record initial positions of each block.
        block_bodies = [body_id(self.model, f"block_{i}") for i in range(self.n_blocks)]
        initial_positions = {bid: self.data.xpos[bid].copy() for bid in block_bodies}

        max_displacements: dict[int, float] = {bid: 0.0 for bid in block_bodies}
        for _ in range(1000):
            mujoco.mj_step(self.model, self.data)
            for bid in block_bodies:
                displacement = float(np.linalg.norm(self.data.xpos[bid] - initial_positions[bid]))
                max_displacements[bid] = max(max_displacements[bid], displacement)

        # A stack "falls" if any block moves more than a quarter of its height.
        threshold = 0.25 * (2.0 * self.block_size[2])
        collapsed = any(d > threshold for d in max_displacements.values())
        outcome = "no" if collapsed else "yes"
        self._outcome = outcome
        self._max_displacement = float(max(max_displacements.values()))
        return outcome

    def question(self) -> Question:
        return Question(
            text=(
                f"A stack of {self.n_blocks} identical blocks is tapped from the side by a "
                f"{self.ball_mass:.2f} kg ball moving at {self.ball_speed:.2f} m/s. "
                "Does the stack remain standing?"
            ),
            answer_type=AnswerType.BOOLEAN,
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=outcome,
            explanation=(
                f"Stack of {self.n_blocks} blocks hit at height {self.impact_height:.2f} m "
                f"by a ball moving at {self.ball_speed:.2f} m/s. "
                f"Maximum block displacement: {self._max_displacement:.3f} m. "
                f"Stack {'remains standing' if outcome == 'yes' else 'collapses'}."
            ),
            latent_params={
                "n_blocks": self.n_blocks,
                "block_mass": self.block_mass,
                "ball_mass": self.ball_mass,
                "ball_speed": self.ball_speed,
                "impact_height": self.impact_height,
                "max_displacement": self._max_displacement,
            },
        )

    def score(self, prediction: Prediction) -> float:
        gt = self.ground_truth().answer
        pred = str(prediction.answer).lower()
        return 1.0 if pred in (gt, "true", "1") else 0.0
