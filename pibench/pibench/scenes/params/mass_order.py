"""Mass-ordering problem.

Three identical-looking blocks are pushed by identical brief forces on a
frictionless surface. The question: which block is the heaviest?

Physical concepts: Newton's second law (F = m a), mass affects acceleration,
parameter estimation from observed motion.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml, mjcf_box


@register_problem("params")
class MassOrder(Problem):
    """Which of three pushed blocks is the heaviest?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        # Three distinct masses, shuffled with a random assignment to A/B/C.
        masses = sorted([float(rng.uniform(0.5, 3.0)) for _ in range(3)])
        perm = rng.permutation(3)
        self.masses = [masses[int(i)] for i in perm]
        self.heaviest_label = ["A", "B", "C"][int(np.argmax(self.masses))]

        self.box_size = (0.08, 0.08, 0.08)
        self.push_force = 5.0
        self.push_duration = 0.3  # seconds
        self.start_x = -1.0
        self.spacing = 0.4

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        worldbody_parts: list[str] = []
        for i, mass in enumerate(self.masses):
            y = (i - 1) * self.spacing
            worldbody_parts.append(
                mjcf_box(
                    name=f"block_{i}",
                    pos=(self.start_x, y, self.box_size[2] + 0.01),
                    size=self.box_size,
                    rgba=(0.6, 0.6, 0.6, 1.0),
                    mass=mass,
                    friction=(0.0, 0.0, 0.0),  # frictionless surface
                )
            )
        return build_xml("".join(worldbody_parts), timestep=0.002)

    def _run_outcome(self) -> dict:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        # Apply constant horizontal force to each block for a fixed duration.
        steps = int(self.push_duration / self.model.opt.timestep)
        for body_idx in range(3):
            body_name = f"block_{body_idx}"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            for _ in range(steps):
                self.data.xfrc_applied[body_id, 0] = self.push_force
                mujoco.mj_step(self.model, self.data)

        # Measure final displacement along X.
        displacements = []
        for i in range(3):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"block_{i}")
            initial = np.array([self.start_x, (i - 1) * self.spacing, self.box_size[2] + 0.01])
            final = self.data.xpos[body_id].copy()
            displacements.append(float(final[0] - initial[0]))

        self._outcome = {
            "displacements": displacements,
            "heaviest_label": self.heaviest_label,
        }
        return self._outcome

    def question(self) -> Question:
        return Question(
            text=(
                "Three identical-looking blocks (A, B, C) are pushed by identical brief "
                "forces on a frictionless surface. Which block is the heaviest?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=["A", "B", "C"],
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=self.heaviest_label,
            explanation=(
                f"Masses: A={self.masses[0]:.2f} kg, B={self.masses[1]:.2f} kg, "
                f"C={self.masses[2]:.2f} kg. Displacements: "
                f"A={outcome['displacements'][0]:.3f} m, "
                f"B={outcome['displacements'][1]:.3f} m, "
                f"C={outcome['displacements'][2]:.3f} m. "
                "The heaviest block accelerates least and travels the shortest distance."
            ),
            latent_params={
                "mass_a": self.masses[0],
                "mass_b": self.masses[1],
                "mass_c": self.masses[2],
                "displacement_a": outcome["displacements"][0],
                "displacement_b": outcome["displacements"][1],
                "displacement_c": outcome["displacements"][2],
            },
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if prediction.answer == self.ground_truth().answer else 0.0
