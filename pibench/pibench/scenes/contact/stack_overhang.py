"""Stack-overhang problem.

Two vertical stacks of identical blocks stand side by side. Each stack has a
single top block shifted horizontally by a different overhang. The question:
which stack topples?

Physical concepts: center of mass, support polygon, stacking stability.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml, mjcf_box


@register_problem("contact")
class StackOverhang(Problem):
    """Which of two block stacks with different overhangs topples?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.n_blocks = int(rng.integers(3, 6))
        self.block_size = (0.06, 0.06, 0.06)  # half-sizes
        self.block_mass = float(rng.uniform(0.2, 0.5))
        # Two overhang amounts; one is always safe, the other is often unsafe.
        base_overhang = float(rng.uniform(0.005, 0.04))
        self.overhang_a = base_overhang
        self.overhang_b = base_overhang + float(rng.uniform(0.02, 0.05))
        # Swap sides randomly.
        if rng.random() > 0.5:
            self.overhang_a, self.overhang_b = self.overhang_b, self.overhang_a

        self.xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(self.xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        worldbody_parts: list[str] = []
        spacing = 0.35  # distance between stack centers.

        for stack, overhang, sign in (("A", self.overhang_a, -1.0), ("B", self.overhang_b, 1.0)):
            x_center = sign * spacing
            z = 0.01
            for i in range(self.n_blocks):
                z += self.block_size[2]
                # Only the top block is overhung.
                x_offset = overhang if i == self.n_blocks - 1 else 0.0
                worldbody_parts.append(
                    mjcf_box(
                        name=f"stack_{stack}_{i}",
                        pos=(x_center + x_offset, 0.0, z),
                        size=self.block_size,
                        rgba=(0.85, 0.30, 0.25, 1.0),
                        mass=self.block_mass,
                        friction=(0.5, 0.02, 0.02),
                    )
                )
                z += self.block_size[2]

        return build_xml("".join(worldbody_parts), timestep=0.002)

    def _run_outcome(self) -> str:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        # Settle under gravity.
        for _ in range(1200):
            mujoco.mj_step(self.model, self.data)

        def stack_toppled(stack: str) -> bool:
            for i in range(self.n_blocks):
                bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"stack_{stack}_{i}")
                # Compute tilt from the z-axis of the body's rotation matrix.
                z_axis = self.data.xmat[bid].reshape(3, 3)[:, 2]
                tilt_deg = float(np.degrees(np.arccos(np.clip(z_axis[2], -1.0, 1.0))))
                if tilt_deg > 10.0:
                    return True
            return False

        toppled_a = stack_toppled("A")
        toppled_b = stack_toppled("B")

        if toppled_a and toppled_b:
            outcome = "both"
        elif toppled_a:
            outcome = "A"
        elif toppled_b:
            outcome = "B"
        else:
            outcome = "neither"

        self._outcome = outcome
        return outcome

    def question(self) -> Question:
        return Question(
            text=(
                f"Two stacks of {self.n_blocks} identical blocks stand side by side. "
                f"The top block of stack A is overhung by {self.overhang_a:.3f} m, "
                f"and the top block of stack B is overhung by {self.overhang_b:.3f} m. "
                "Which stack topples?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=["A", "B", "both", "neither"],
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=outcome,
            explanation=(
                f"Stack A overhang = {self.overhang_a:.3f} m, stack B overhang = {self.overhang_b:.3f} m. "
                f"Outcome: {outcome} topple(s)."
            ),
            latent_params={
                "n_blocks": self.n_blocks,
                "block_mass": self.block_mass,
                "overhang_a": self.overhang_a,
                "overhang_b": self.overhang_b,
            },
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if str(prediction.answer).lower() == self.ground_truth().answer.lower() else 0.0
