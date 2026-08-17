"""Wedge-insert problem.

A triangular wedge is pushed into a gap between two fixed blocks. The question:
will it fit through the gap, or jam?

Physical concepts: contact geometry, clearance, jamming, normal forces.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.contact import body_id, mjcf_pusher, mjcf_wedge_mesh, run_with_pusher
from pibench.utils.mjcf import build_xml


@register_problem("contact")
class WedgeInsert(Problem):
    """Does a wedge fit into the gap or jam?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        # Wedge dimensions.
        self.wedge_length = float(rng.uniform(0.25, 0.45))
        self.wedge_base_width = float(rng.uniform(0.08, 0.22))
        self.wedge_height = 0.15
        self.wedge_friction = (0.25, 0.01, 0.01)

        # Gap between the two fixed walls.
        self.gap_width = float(rng.uniform(0.08, 0.24))
        # Make the outcome mixed by sometimes giving the wedge a base wider than the gap.
        if rng.random() > 0.5:
            self.gap_width = max(self.gap_width, self.wedge_base_width + 0.02)
        else:
            self.gap_width = min(self.gap_width, self.wedge_base_width - 0.02)

        self.wall_thickness = 0.08
        self.wall_depth = 0.40
        self.wall_height = 0.30
        self.pusher_speed = 0.25

        xml = self._make_xml()
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(self) -> str:
        # Fixed vertical walls on either side of the gap (centered at y=0).
        wall_y = self.gap_width / 2.0 + self.wall_thickness / 2.0
        wall_z = self.wall_height / 2.0 + 0.01
        worldbody_parts = [
            f"""    <body name="wall_left" pos="0 {wall_y} {wall_z}">
      <geom name="wall_left_geom" type="box" size="{self.wall_depth/2.0} {self.wall_thickness/2.0} {self.wall_height/2.0}" rgba="0.4 0.4 0.4 1" />
    </body>
""",
            f"""    <body name="wall_right" pos="0 {-wall_y} {wall_z}">
      <geom name="wall_right_geom" type="box" size="{self.wall_depth/2.0} {self.wall_thickness/2.0} {self.wall_height/2.0}" rgba="0.4 0.4 0.4 1" />
    </body>
""",
        ]

        # Wedge starts to the left of the gap.  The mesh is centered so the tip
        # is at +X relative to the body origin.
        wedge_start_x = -0.55
        wedge_pos_z = self.wedge_height / 2.0 + 0.01
        asset, wedge_body, _ = mjcf_wedge_mesh(
            name="wedge",
            pos=(wedge_start_x, 0.0, wedge_pos_z),
            length=self.wedge_length,
            base_width=self.wedge_base_width,
            height=self.wedge_height,
            rgba=(0.85, 0.55, 0.25, 1.0),
            friction=self.wedge_friction,
        )
        worldbody_parts.append(wedge_body)

        # Pusher contacts the wide base of the wedge and drives it in +X.
        pusher_half_x = 0.02
        pusher_x = wedge_start_x - self.wedge_length / 2.0 - pusher_half_x - 0.01
        pusher_body, pusher_actuator = mjcf_pusher(
            name="pusher",
            pos=(pusher_x, 0.0, wedge_pos_z),
            size=(pusher_half_x, self.wedge_base_width * 0.4, self.wedge_height * 0.4),
            axis=(1.0, 0.0, 0.0),
            max_speed=1.0,
        )
        worldbody_parts.append(pusher_body)

        xml = build_xml("".join(worldbody_parts), timestep=0.002)
        # Inject the mesh asset before the worldbody.
        xml = xml.replace(
            '<compiler angle="radian" />',
            f'<compiler angle="radian" />\n  <asset>\n{asset}  </asset>',
        )
        xml = xml.replace("</mujoco>", f"  <actuator>\n{pusher_actuator}  </actuator>\n</mujoco>")
        return xml

    def _run_outcome(self) -> str:
        if getattr(self, "_outcome", None) is not None:
            return self._outcome

        wedge_bid = body_id(self.model, "wedge")
        start_x = float(self.data.xpos[wedge_bid][0])

        def _record(_model: mujoco.MjModel, data: mujoco.MjData, _step: int) -> None:
            pass

        run_with_pusher(
            self.model,
            self.data,
            pusher_joint_name="pusher_joint",
            speed=self.pusher_speed,
            steps=1200,
            callbacks=[_record],
        )

        final_x = float(self.data.xpos[wedge_bid][0])
        # The wedge is considered to have fit if its tip (center) has advanced
        # well past the gap center.
        outcome = "fits" if final_x > 0.10 else "jams"
        self._outcome = outcome
        self._wedge_delta_x = final_x - start_x
        return outcome

    def question(self) -> Question:
        return Question(
            text=(
                f"A wedge with base width {self.wedge_base_width:.2f} m is pushed into a "
                f"gap {self.gap_width:.2f} m wide. Does it fit through or jam?"
            ),
            answer_type=AnswerType.CHOICE,
            choices=["fits", "jams"],
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=outcome,
            explanation=(
                f"Wedge base width = {self.wedge_base_width:.3f} m, gap = {self.gap_width:.3f} m. "
                f"Wedge advanced {self._wedge_delta_x:.3f} m. "
                f"Outcome: {outcome}."
            ),
            latent_params={
                "wedge_base_width": self.wedge_base_width,
                "wedge_length": self.wedge_length,
                "gap_width": self.gap_width,
                "wedge_delta_x": self._wedge_delta_x,
            },
        )

    def score(self, prediction: Prediction) -> float:
        return 1.0 if prediction.answer == self.ground_truth().answer else 0.0
