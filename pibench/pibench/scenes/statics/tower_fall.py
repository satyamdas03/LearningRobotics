"""Tower-fall stability problem.

Two block towers are built on a shared platform. The platform is suddenly
tilted. The question: which tower falls first (or neither / both)?

Physical concepts: center of mass, support polygon, stability under tilt.
"""
from __future__ import annotations

import numpy as np

import mujoco

from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import register_problem
from pibench.utils.mjcf import build_xml, mjcf_box


@register_problem("statics")
class TowerFall(Problem):
    """Which tower falls when the platform tilts?"""

    def _build_scene(self) -> None:
        rng = np.random.default_rng(self.seed)

        # Platform dimensions.
        platform_length = float(rng.uniform(2.5, 4.0))
        platform_width = 1.5
        platform_thickness = 0.05

        # Tower A: narrow and tall (unstable).
        base_a = (0.12, 0.12, 0.12)
        height_a = int(rng.integers(4, 7))
        mass_a = float(rng.uniform(0.3, 0.7))
        offset_a_x = -platform_length * 0.25
        color_a = (0.85, 0.30, 0.25, 1.0)

        # Tower B: wide and short (stable).
        base_b = (0.22, 0.22, 0.10)
        height_b = int(rng.integers(2, 4))
        mass_b = float(rng.uniform(0.6, 1.0))
        offset_b_x = platform_length * 0.25
        color_b = (0.25, 0.55, 0.85, 1.0)

        # Tilt angle applied to the platform hinge.
        tilt_angle = float(rng.uniform(np.deg2rad(12), np.deg2rad(25)))

        self.tower_a_height = height_a
        self.tower_b_height = height_b
        self.base_a = base_a
        self.base_b = base_b
        self.mass_a = mass_a
        self.mass_b = mass_b
        self.tilt_angle = tilt_angle

        xml = self._make_xml(
            platform_length,
            platform_width,
            platform_thickness,
            offset_a_x,
            offset_b_x,
            base_a,
            height_a,
            mass_a,
            color_a,
            base_b,
            height_b,
            mass_b,
            color_b,
            tilt_angle,
        )
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

    def _make_xml(
        self,
        platform_length: float,
        platform_width: float,
        platform_thickness: float,
        offset_a_x: float,
        offset_b_x: float,
        base_a: tuple[float, float, float],
        height_a: int,
        mass_a: float,
        color_a: tuple[float, float, float, float],
        base_b: tuple[float, float, float],
        height_b: int,
        mass_b: float,
        color_b: tuple[float, float, float, float],
        tilt_angle: float,
    ) -> str:
        worldbody_parts: list[str] = []

        # Platform as a thin box hinged at its center about the Y-axis.
        worldbody_parts.append(
            f"""    <body name="platform" pos="0 0 {platform_thickness}">
      <joint name="platform_tilt" type="hinge" axis="0 1 0" pos="0 0 0" range="-{tilt_angle} {tilt_angle}" damping="0.5" />
      <geom name="platform_geom" type="box" size="{platform_length} {platform_width} {platform_thickness}" rgba="0.4 0.4 0.4 1" />
    </body>
"""
        )

        # Build towers as stacked free-floating boxes sitting on the platform.
        z = platform_thickness
        for i in range(height_a):
            half_size = base_a
            z += half_size[2]
            worldbody_parts.append(
                mjcf_box(
                    name=f"tower_a_block_{i}",
                    pos=(offset_a_x, 0.0, z),
                    size=half_size,
                    rgba=color_a,
                    mass=mass_a,
                    friction=(0.8, 0.02, 0.02),
                )
            )
            z += half_size[2]

        z = platform_thickness
        for i in range(height_b):
            half_size = base_b
            z += half_size[2]
            worldbody_parts.append(
                mjcf_box(
                    name=f"tower_b_block_{i}",
                    pos=(offset_b_x, 0.0, z),
                    size=half_size,
                    rgba=color_b,
                    mass=mass_b,
                    friction=(0.8, 0.02, 0.02),
                )
            )
            z += half_size[2]

        actuator = f"""  <actuator>
    <position joint="platform_tilt" kp="1000" kv="100" ctrlrange="-{tilt_angle} {tilt_angle}" />
  </actuator>
"""

        xml = build_xml("".join(worldbody_parts), timestep=0.002)
        xml = xml.replace("</mujoco>", actuator + "</mujoco>")
        return xml

    def _run_outcome(self) -> str:
        """Run the tilt and classify which tower falls."""
        # Hold initial pose steady for a few steps.
        for _ in range(50):
            self.data.ctrl[0] = 0.0
            mujoco.mj_step(self.model, self.data)

        # Command the platform to its maximum tilt angle.
        self.data.ctrl[0] = self.tilt_angle
        steps = 300
        sample_every = 5
        positions: dict[str, list[np.ndarray]] = {}

        for step in range(steps):
            mujoco.mj_step(self.model, self.data)
            if step % sample_every == 0:
                for name in ["tower_a_block_0", "tower_b_block_0"]:
                    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                    pos = self.data.xpos[body_id].copy()
                    positions.setdefault(name, []).append(pos)

        def _max_disp(name: str) -> float:
            if not positions.get(name):
                return 0.0
            initial = positions[name][0]
            return max(float(np.linalg.norm(p - initial)) for p in positions[name])

        disp_a = _max_disp("tower_a_block_0")
        disp_b = _max_disp("tower_b_block_0")

        threshold = 0.15
        fell_a = disp_a > threshold
        fell_b = disp_b > threshold

        if fell_a and fell_b:
            return "both"
        if fell_a:
            return "A"
        if fell_b:
            return "B"
        return "neither"

    def question(self) -> Question:
        return Question(
            text="Two block towers sit on a tilting platform. Tower A is narrow and tall; Tower B is wide and short. Which tower falls?",
            answer_type=AnswerType.CHOICE,
            choices=["A", "B", "both", "neither"],
        )

    def ground_truth(self) -> GroundTruth:
        outcome = self._run_outcome()
        return GroundTruth(
            answer=outcome,
            explanation=(
                f"Tower A has {self.tower_a_height} blocks of size {self.base_a}; "
                f"Tower B has {self.tower_b_height} blocks of size {self.base_b}. "
                f"Platform tilted to {np.rad2deg(self.tilt_angle):.1f}°. "
                f"Outcome: {outcome} tower(s) fell."
            ),
            latent_params={
                "tilt_angle_deg": float(np.rad2deg(self.tilt_angle)),
                "tower_a_blocks": self.tower_a_height,
                "tower_b_blocks": self.tower_b_height,
                "mass_a": self.mass_a,
                "mass_b": self.mass_b,
            },
        )

    def score(self, prediction: Prediction) -> float:
        gt = self.ground_truth().answer
        return 1.0 if prediction.answer == gt else 0.0
