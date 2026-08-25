"""Peg-in-hole problem.

A cylindrical peg is lowered vertically into a hole in a fixed block. The
question: does the peg fit cleanly into the hole, or does it jam on the top
surface because the hole is too small?

Physical concepts: contact geometry, clearance, insertion, jamming.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml


@register_problem("contact")
class PegInHole(Problem):
    """Does a cylindrical peg fit into the hole or jam?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        self.peg_radius = float(rng.uniform(0.025, 0.045))
        self.peg_length = float(rng.uniform(0.25, 0.45))
        self.hole_depth = float(rng.uniform(0.20, 0.40))
        # Hole radius relative to peg radius. Make mixed outcomes.
        clearance = float(rng.uniform(-0.003, 0.008))
        self.hole_radius = self.peg_radius + clearance

        self.block_size = (0.30, 0.30, 0.10)  # half-sizes
        self.block_height = 2.0 * self.block_size[2]

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        half_x, half_y, half_z = self.block_size

        # Fixed base block with a hole approximated by four inward-facing walls.
        wall_thickness = 0.02
        hole_r = self.hole_radius
        # The four walls leave a square-ish hole of width 2*hole_r.
        walls = [
            # +Y wall
            f"""    <body name="wall_ny" pos="0 {hole_r + wall_thickness/2.0} {half_z}">
      <geom name="wall_ny_geom" type="box" size="{half_x} {wall_thickness/2.0} {half_z}" rgba="0.5 0.5 0.5 1" />
    </body>
""",
            # -Y wall
            f"""    <body name="wall_py" pos="0 {-hole_r - wall_thickness/2.0} {half_z}">
      <geom name="wall_py_geom" type="box" size="{half_x} {wall_thickness/2.0} {half_z}" rgba="0.5 0.5 0.5 1" />
    </body>
""",
            # +X wall
            f"""    <body name="wall_px" pos="{hole_r + wall_thickness/2.0} 0 {half_z}">
      <geom name="wall_px_geom" type="box" size="{wall_thickness/2.0} {hole_r} {half_z}" rgba="0.5 0.5 0.5 1" />
    </body>
""",
            # -X wall
            f"""    <body name="wall_nx" pos="{-hole_r - wall_thickness/2.0} 0 {half_z}">
      <geom name="wall_nx_geom" type="box" size="{wall_thickness/2.0} {hole_r} {half_z}" rgba="0.5 0.5 0.5 1" />
    </body>
""",
        ]

        # Peg is a free body placed just above the hole, given a small downward velocity.
        peg_mass = 0.3
        peg_i = 0.4 * peg_mass * self.peg_radius**2
        peg_inertia = f'{peg_i} {peg_i} {peg_i}'
        peg_z = self.block_height + self.peg_length / 2.0 + 0.05
        peg = f"""    <body name="peg" pos="0 0 {peg_z}">
      <freejoint />
      <inertial pos="0 0 0" mass="{peg_mass}" diaginertia="{peg_inertia}" />
      <geom name="peg_geom" type="cylinder" size="{self.peg_radius} {self.peg_length/2.0}" rgba="0.85 0.55 0.25 1" friction="0.4 0.01 0.01" />
    </body>
"""

        # Limiter plane at the bottom of the hole to stop the peg.
        floor_hole = f"""    <body name="hole_floor" pos="0 0 0.01">
      <geom name="hole_floor_geom" type="box" size="{hole_r} {hole_r} 0.01" rgba="0.3 0.3 0.3 1" />
    </body>
"""

        worldbody_parts = walls + [peg, floor_hole]
        return build_xml("".join(worldbody_parts), timestep=0.002)

    def _run_outcome(self) -> str:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        peg_bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "peg")
        peg_qvel = self.model.body_dofadr[peg_bid]
        # Start with a small downward velocity and tiny random perturbation.
        self.data.qvel[peg_qvel + 2] = -0.3
        self.data.qvel[peg_qvel + 3:peg_qvel + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

        for _ in range(1200):
            mujoco.mj_step(self.model, self.data)

        final_z = float(self.data.xpos[peg_bid, 2])
        # Peg is "in" if its center dropped below the original top surface of the block.
        self._final_z = final_z
        outcome = "fits" if final_z < self.block_height + 0.02 else "jams"
        self._outcome = outcome
        return outcome

    def question(self) -> Question:
        return Question(
            text=(
                f"A cylindrical peg of radius {self.peg_radius:.3f} m is lowered into a hole of "
                f"radius {self.hole_radius:.3f} m. Does the peg fit into the hole or jam?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=["fits", "jams"],
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=outcome,
            explanation=(
                f"Peg radius = {self.peg_radius:.3f} m, hole radius = {self.hole_radius:.3f} m. "
                f"Final peg center height: {self._final_z:.3f} m. Outcome: {outcome}."
            ),
            latent_params={
                "peg_radius": self.peg_radius,
                "hole_radius": self.hole_radius,
                "peg_length": self.peg_length,
                "hole_depth": self.hole_depth,
                "final_z": self._final_z,
            },
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if str(prediction.answer).lower() == self.ground_truth().answer.lower() else 0.0
