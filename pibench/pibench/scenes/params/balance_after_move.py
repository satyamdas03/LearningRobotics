"""Balance-after-move problem.

A uniform beam rests on a support with a point mass placed somewhere along its
length. The question: if the point mass is moved a given distance to the right,
how far to the right must the support move so the beam remains horizontal?

Physical concepts: torque balance, center of mass, static equilibrium.
"""
from __future__ import annotations

import numpy as np

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem


@register_problem("params")
class BalanceAfterMove(Problem):
    """How far must the support move right to keep the beam balanced?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.beam_mass = float(rng.uniform(1.0, 3.0))
        self.beam_length = 2.0
        self.point_mass = float(rng.uniform(0.5, 2.0))
        # Initial point-mass position relative to beam center.
        self.mass_offset = float(rng.uniform(-0.4, 0.4))
        self.move_distance = float(rng.uniform(0.1, 0.3))

        # Compute initial and new balance positions analytically.
        # For a uniform beam, beam CoM is at x=0. Support position x_s solves:
        #   m_beam * (0 - x_s) + m_point * (offset - x_s) = 0
        #   => x_s = m_point * offset / (m_beam + m_point)
        self.initial_balance = self._balance_x(self.mass_offset)
        self.new_balance = self._balance_x(self.mass_offset + self.move_distance)
        self.required_support_shift = self.new_balance - self.initial_balance

    def _balance_x(self, mass_offset: float) -> float:
        return (self.point_mass * mass_offset) / (self.beam_mass + self.point_mass)

    def _counterfactual_params(self) -> list[str]:
        return ["mass_offset", "move_distance"]

    def question(self) -> Question:
        return Question(
            text=(
                f"A {self.beam_mass:.2f} kg uniform beam rests on a movable support, with a "
                f"{self.point_mass:.2f} kg point mass placed {self.mass_offset:.2f} m to the right of center. "
                f"If the point mass is moved {self.move_distance:.2f} m farther to the right, "
                "how far to the right must the support move to keep the beam horizontal? "
                "(Answer in meters, positive = right.)"
            ),
            answer_type=AnswerType.NUMERIC,
            units="m",
        )

    def ground_truth(self) -> GroundTruth:
        return GroundTruth(
            answer=float(self.required_support_shift),
            explanation=(
                f"Initial balance position: {self.initial_balance:.4f} m. "
                f"After moving the point mass to {self.mass_offset + self.move_distance:.2f} m, "
                f"the new balance position is {self.new_balance:.4f} m. "
                f"Support shift = {self.required_support_shift:.4f} m."
            ),
            latent_params={
                "beam_mass": self.beam_mass,
                "point_mass": self.point_mass,
                "mass_offset": self.mass_offset,
                "move_distance": self.move_distance,
                "initial_balance": self.initial_balance,
                "new_balance": self.new_balance,
            },
        )

    def score(self, prediction: Prediction) -> float:
        gt = float(self.ground_truth().answer)
        pred = float(prediction.answer)
        # Tolerance: 5% of the move distance, minimum 1 cm.
        tolerance = max(0.05 * abs(self.move_distance), 0.01)
        return 1.0 if abs(pred - gt) <= tolerance else 0.0
