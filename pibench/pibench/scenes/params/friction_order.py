"""Friction-ordering problem.

Three identical blocks sit on a slowly tilting platform. The blocks have different
static-friction coefficients. The question: which block is the most slippery
(lowest static friction)?

Physical concepts: static friction, angle of repose, parameter estimation from
observed sliding threshold.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml, mjcf_box


@register_problem("params")
class FrictionOrder(Problem):
    """Which of three blocks is the most slippery?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        # Three distinct friction coefficients, shuffled.
        mu_values = sorted([float(rng.uniform(0.1, 0.8)) for _ in range(3)])
        perm = rng.permutation(3)
        self.mu_values = [mu_values[int(i)] for i in perm]
        self.most_slippery_label = ["A", "B", "C"][int(np.argmin(self.mu_values))]

        self.box_size = (0.08, 0.08, 0.08)
        self.platform_length = 1.5
        self.platform_width = 1.2
        self.platform_thickness = 0.02
        self.max_tilt_deg = 55.0

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        worldbody_parts: list[str] = []

        # Tilting platform hinged about Y at its center.
        worldbody_parts.append(
            f"""    <body name="platform" pos="0 0 {self.platform_thickness}">
      <joint name="platform_tilt" type="hinge" axis="0 1 0" pos="0 0 0" range="-{np.deg2rad(self.max_tilt_deg)} {np.deg2rad(self.max_tilt_deg)}" damping="0.5" />
      <geom name="platform_geom" type="box" size="{self.platform_length} {self.platform_width} {self.platform_thickness}" rgba="0.4 0.4 0.4 1" />
    </body>
"""
        )

        for i, mu in enumerate(self.mu_values):
            y = (i - 1) * 0.3
            worldbody_parts.append(
                mjcf_box(
                    name=f"block_{i}",
                    pos=(0.0, y, self.platform_thickness + self.box_size[2] + 0.005),
                    size=self.box_size,
                    rgba=(0.7, 0.7, 0.7, 1.0),
                    mass=0.5,
                    friction=(mu, 0.01, 0.01),
                )
            )

        actuator = f"""  <actuator>
    <position joint="platform_tilt" kp="1000" kv="100" ctrlrange="-{np.deg2rad(self.max_tilt_deg)} {np.deg2rad(self.max_tilt_deg)}" />
  </actuator>
"""
        xml = build_xml("".join(worldbody_parts), timestep=0.002)
        xml = xml.replace("</mujoco>", actuator + "</mujoco>")
        return xml

    def _run_outcome(self) -> dict:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        # Settle briefly, then tilt to max angle.
        for _ in range(50):
            self.data.ctrl[0] = 0.0
            mujoco.mj_step(self.model, self.data)

        target_tilt = np.deg2rad(self.max_tilt_deg)
        self.data.ctrl[0] = target_tilt
        steps = 800
        sample_every = 10

        initial_positions = []
        for i in range(3):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"block_{i}")
            initial_positions.append(self.data.xpos[body_id].copy())

        max_displacements = [0.0, 0.0, 0.0]
        for step in range(steps):
            mujoco.mj_step(self.model, self.data)
            if step % sample_every == 0:
                for i in range(3):
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"block_{i}")
                    disp = float(np.linalg.norm(self.data.xpos[body_id] - initial_positions[i]))
                    max_displacements[i] = max(max_displacements[i], disp)

        # First block to move significantly is the most slippery.
        threshold = 0.05
        slipped = [disp > threshold for disp in max_displacements]
        if any(slipped):
            first_slip_idx = next(i for i, s in enumerate(slipped) if s)
        else:
            first_slip_idx = int(np.argmin(self.mu_values))  # fallback: lowest mu

        most_slippery_label = ["A", "B", "C"][first_slip_idx]
        self._outcome = {
            "max_displacements": max_displacements,
            "slipped": slipped,
            "most_slippery_label": most_slippery_label,
        }
        return self._outcome

    def question(self) -> Question:
        return Question(
            text=(
                "Three identical blocks (A, B, C) sit on a slowly tilting platform. "
                "The platform is tilted to the same final angle for all three blocks. "
                "Which block is the most slippery (has the lowest static-friction coefficient)?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=["A", "B", "C"],
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=outcome["most_slippery_label"],
            explanation=(
                f"Friction coefficients: A={self.mu_values[0]:.2f}, "
                f"B={self.mu_values[1]:.2f}, C={self.mu_values[2]:.2f}. "
                f"Max displacements: A={outcome['max_displacements'][0]:.3f} m, "
                f"B={outcome['max_displacements'][1]:.3f} m, "
                f"C={outcome['max_displacements'][2]:.3f} m. "
                "The most slippery block slides at the shallowest angle."
            ),
            latent_params={
                "mu_a": self.mu_values[0],
                "mu_b": self.mu_values[1],
                "mu_c": self.mu_values[2],
                "max_displacement_a": outcome["max_displacements"][0],
                "max_displacement_b": outcome["max_displacements"][1],
                "max_displacement_c": outcome["max_displacements"][2],
            },
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if prediction.answer == self.ground_truth().answer else 0.0
