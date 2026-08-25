"""Rope-sag problem.

Two coarse deformable chains (ropes) hang from the same height, side by side.
Chain A has more capsules than chain B. After settling under gravity, which
chain's free end hangs lower?

Physical concepts: deformable-body approximation, sag, gravity, chain length.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.articulated import body_id, mjcf_capsule_chain
from pibench.utils.mjcf import build_xml


@register_problem("deformable")
class RopeSag(Problem):
    """Which of two hanging chains sags lower?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.n_capsules_a = int(rng.integers(8, 13))
        self.n_capsules_b = int(rng.integers(4, 8))
        # Randomly swap which side is longer.
        if rng.random() > 0.5:
            self.n_capsules_a, self.n_capsules_b = self.n_capsules_b, self.n_capsules_a

        self.capsule_radius = 0.03
        self.capsule_half_len = 0.10
        self.mass_per_capsule = 0.10
        self.anchor_height = 1.8
        self.spacing = 0.45

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        worldbody_parts: list[str] = []

        for label, n_capsules, sign in (("A", self.n_capsules_a, -1.0), ("B", self.n_capsules_b, 1.0)):
            x_anchor = sign * self.spacing
            start_pos = (x_anchor, 0.0, self.anchor_height)
            worldbody_parts.append(
                mjcf_capsule_chain(
                    name=f"rope_{label}",
                    start_pos=start_pos,
                    n_capsules=n_capsules,
                    capsule_radius=self.capsule_radius,
                    capsule_half_len=self.capsule_half_len,
                    spacing=2.0 * self.capsule_half_len,
                    mass_per_capsule=self.mass_per_capsule,
                    rgba=(0.2, 0.6, 0.8, 1.0),
                )
            )

        return build_xml("".join(worldbody_parts), timestep=0.001)

    def _free_end_height(self, label: str) -> float:
        n = getattr(self, f"n_capsules_{label.lower()}")
        last_id = body_id(self.model, f"rope_{label}_seg{n - 1}")
        # Height of the bottom of the last capsule.
        return float(self.data.xpos[last_id, 2] - self.capsule_radius)

    def _run_outcome(self) -> str:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        # Settle the chains.
        steps = 6000
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)

        height_a = self._free_end_height("A")
        height_b = self._free_end_height("B")
        self._height_a = height_a
        self._height_b = height_b

        # Lower free end means more sag.
        if abs(height_a - height_b) < 0.02:
            outcome = "same"
        elif height_a < height_b:
            outcome = "A"
        else:
            outcome = "B"
        self._outcome = outcome
        return outcome

    def question(self) -> Question:
        return Question(
            text=(
                f"Two deformable chains hang from the same height. Chain A has "
                f"{self.n_capsules_a} capsules and chain B has {self.n_capsules_b} capsules. "
                "After they settle under gravity, which chain's free end hangs lower?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=["A", "B", "same"],
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=outcome,
            explanation=(
                f"Chain A has {self.n_capsules_a} capsules and ends at {self._height_a:.3f} m. "
                f"Chain B has {self.n_capsules_b} capsules and ends at {self._height_b:.3f} m. "
                f"Lower end: {outcome}."
            ),
            latent_params={
                "n_capsules_a": self.n_capsules_a,
                "n_capsules_b": self.n_capsules_b,
                "capsule_radius": self.capsule_radius,
                "capsule_half_len": self.capsule_half_len,
                "mass_per_capsule": self.mass_per_capsule,
                "height_a": self._height_a,
                "height_b": self._height_b,
            },
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if str(prediction.answer).lower() == self.ground_truth().answer.lower() else 0.0
