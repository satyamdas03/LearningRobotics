"""Chain-drape problem.

A coarse deformable chain (capsules linked by ball joints) is draped over a
rectangular bar. After settling under gravity, the free end hangs at some height
above the floor. The question asks for that height.

Physical concepts: deformable bodies, gravity, contact, equilibrium shape.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.articulated import body_id, mjcf_capsule_chain
from pibench.utils.mjcf import build_xml


@register_problem("deformable")
class ChainDrape(Problem):
    """How far above the floor does the free end of a draped chain hang?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.bar_height = float(rng.uniform(0.8, 1.4))
        self.bar_width = 0.25
        self.n_capsules = int(rng.integers(6, 11))
        self.capsule_radius = 0.05
        self.capsule_half_len = 0.15
        self.spacing = 0.30
        self.mass_per_capsule = 0.20

        # Root is placed so the chain reaches past the bar toward +X.
        self.root_pos = (-0.8, 0.0, self.bar_height + 0.45)

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        worldbody_parts: list[str] = []

        # Rectangular bar over which the chain drapes.
        worldbody_parts.append(
            f"""    <body name="bar" pos="0 0 {self.bar_height}">
      <geom name="bar_geom" type="box" size="{self.bar_width} 1.0 0.1" rgba="0.5 0.5 0.5 1" />
    </body>
"""
        )

        # Coarse deformable capsule chain.
        worldbody_parts.append(
            mjcf_capsule_chain(
                name="chain",
                start_pos=self.root_pos,
                n_capsules=self.n_capsules,
                capsule_radius=self.capsule_radius,
                capsule_half_len=self.capsule_half_len,
                spacing=self.spacing,
                mass_per_capsule=self.mass_per_capsule,
                rgba=(0.4, 0.6, 0.8, 1.0),
            )
        )

        return build_xml("\n".join(worldbody_parts), timestep=0.001)

    def _run_outcome(self) -> float:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        last_id = body_id(self.model, f"chain_seg{self.n_capsules - 1}")

        # Settle the chain.
        steps = 8000
        z_samples: list[float] = []
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            # Height of the bottom of the last capsule.
            z_samples.append(float(self.data.xpos[last_id, 2] - self.capsule_radius))

        # Average the last few samples to suppress residual oscillation.
        free_end_height = float(np.mean(z_samples[-200:]))
        self._outcome = free_end_height
        return free_end_height

    def question(self) -> Question:
        return Question(
            text=(
                f"A deformable chain of {self.n_capsules} capsules is draped over a "
                f"rectangular bar at height {self.bar_height:.2f} m. After the chain settles, "
                f"how far above the floor does the free end hang?"
            ),
            answer_type=AnswerType.NUMERIC,
            units="m",
        )

    def ground_truth(self) -> GroundTruth:
        height = self._run_outcome()
        return GroundTruth(
            answer=height,
            explanation=(
                f"Chain with {self.n_capsules} capsules draped over a bar at "
                f"{self.bar_height:.2f} m. Simulated free-end height above floor: {height:.3f} m."
            ),
            latent_params={
                "bar_height": self.bar_height,
                "n_capsules": self.n_capsules,
                "capsule_radius": self.capsule_radius,
                "capsule_half_len": self.capsule_half_len,
                "spacing": self.spacing,
                "mass_per_capsule": self.mass_per_capsule,
            },
        )

    def score(self, prediction: Prediction) -> float:
        gt = self.ground_truth().answer
        try:
            pred = float(prediction.answer)
        except (TypeError, ValueError):
            return 0.0
        # Tolerance: within 5 cm or 20% relative.
        return 1.0 if abs(pred - gt) <= max(0.05, 0.2 * abs(gt)) else 0.0
