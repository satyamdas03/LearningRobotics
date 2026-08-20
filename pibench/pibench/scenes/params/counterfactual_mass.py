"""Counterfactual mass problem.

A block tower stands on a tilting platform. The question: if the top block had
double the mass, would the tower topple?

Physical concepts: center of mass, stability margin, counterfactual reasoning,
simulation-based prediction.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml, mjcf_box


@register_problem("params")
class CounterfactualMass(Problem):
    """Would doubling the top-block mass make the tower topple?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.base_size = (0.12, 0.12, 0.06)
        self.height = int(rng.integers(3, 6))
        self.block_masses = [float(rng.uniform(0.2, 0.6)) for _ in range(self.height)]
        self.top_mass_multiplier = 2.0
        self.tilt_angle = float(rng.uniform(np.deg2rad(15), np.deg2rad(30)))
        self.topple_threshold = 0.12

        xml = self._make_xml(self.block_masses)
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self, masses: list[float]) -> str:
        worldbody_parts: list[str] = []

        # Tilting platform.
        worldbody_parts.append(
            f"""    <body name="platform" pos="0 0 {self.base_size[2]}">
      <joint name="platform_tilt" type="hinge" axis="0 1 0" pos="0 0 0" range="-{self.tilt_angle} {self.tilt_angle}" damping="0.5" />
      <geom name="platform_geom" type="box" size="1.5 0.8 {self.base_size[2]}" rgba="0.4 0.4 0.4 1" />
    </body>
"""
        )

        z = self.base_size[2]
        for i, mass in enumerate(masses):
            z += self.base_size[2]
            worldbody_parts.append(
                mjcf_box(
                    name=f"block_{i}",
                    pos=(0.0, 0.0, z),
                    size=self.base_size,
                    rgba=(0.85, 0.30, 0.25, 1.0),
                    mass=mass,
                    friction=(0.8, 0.02, 0.02),
                )
            )
            z += self.base_size[2]

        actuator = f"""  <actuator>
    <position joint="platform_tilt" kp="1000" kv="100" ctrlrange="-{self.tilt_angle} {self.tilt_angle}" />
  </actuator>
"""
        xml = build_xml("".join(worldbody_parts), timestep=0.002)
        xml = xml.replace("</mujoco>", actuator + "</mujoco>")
        return xml

    def _run_outcome(self, masses: list[float]) -> str:
        cache_key = tuple(masses)
        if not hasattr(self, "_outcomes"):
            self._outcomes = {}
        if cache_key in self._outcomes:
            return self._outcomes[cache_key]

        # Build a fresh model for this mass configuration so the counterfactual
        # does not mutate the original scene.
        xml = self._make_xml(masses)
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_resetData(model, data)

        # Settle and tilt.
        for _ in range(50):
            data.ctrl[0] = 0.0
            mujoco.mj_step(model, data)
        data.ctrl[0] = self.tilt_angle

        steps = 400
        initial_pos = None
        toppled = False
        for step in range(steps):
            mujoco.mj_step(model, data)
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block_0")
            pos = data.xpos[body_id].copy()
            if initial_pos is None:
                initial_pos = pos
            if float(np.linalg.norm(pos - initial_pos)) > self.topple_threshold:
                toppled = True
                break

        outcome = "yes" if toppled else "no"
        self._outcomes[cache_key] = outcome
        return outcome

    def _counterfactual_params(self) -> list[str]:
        return ["top_mass_multiplier"]

    def question(self) -> Question:
        return Question(
            text=(
                f"A tower of {self.height} blocks stands on a tilting platform. "
                "If the top block's mass were doubled, would the tower topple?"
            ),
            answer_type=AnswerType.BOOLEAN,
        )

    def ground_truth(self) -> GroundTruth:
        original_outcome = self._run_outcome(self.block_masses)
        counterfactual_masses = self.block_masses.copy()
        counterfactual_masses[-1] *= self.top_mass_multiplier
        cf_outcome = self._run_outcome(counterfactual_masses)

        return GroundTruth(
            answer=cf_outcome,
            explanation=(
                f"Original tower has topple outcome: {original_outcome}. "
                f"With top-block mass doubled from {self.block_masses[-1]:.2f} kg to "
                f"{counterfactual_masses[-1]:.2f} kg, the tower "
                f"{'topples' if cf_outcome == 'yes' else 'does not topple'}."
            ),
            latent_params={
                "n_blocks": self.height,
                "original_top_mass": self.block_masses[-1],
                "counterfactual_top_mass": counterfactual_masses[-1],
                "tilt_angle_deg": float(np.rad2deg(self.tilt_angle)),
                "original_topples": original_outcome,
            },
        )

    def score(self, prediction: Prediction) -> float:
        gt = self.ground_truth().answer
        pred = str(prediction.answer).lower()
        return 1.0 if pred in (gt, "true", "1") else 0.0
