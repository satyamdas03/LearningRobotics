"""End-to-end north-star demo: NL task → skill plan → trajectory → execute → validate → save."""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

# Ensure sibling chapter packages are importable when this file is run directly.
_REPO_ROOT = Path(__file__).parent.parent
for _ch in [
    "chapter05_inverse_kinematics",
    "chapter06_dynamics",
    "chapter07_control",
    "chapter08_motion_planning",
    "chapter09_trajectory_generation",
    "chapter12_reasoning",
    "chapter13_skills",
    "pibench",
]:
    _p = str(_REPO_ROOT / _ch)
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chapter05_inverse_kinematics.inverse_kinematics import InverseKinematics  # noqa: E402
from chapter06_dynamics.dynamics import ArmDynamics  # noqa: E402
from chapter07_control.control import JointSpacePIDController  # noqa: E402
from chapter07_control.real_hardware import MockRealArm  # noqa: E402
from chapter08_motion_planning.planners import linear_interpolate  # noqa: E402
from chapter09_trajectory_generation.path_to_trajectory import plan_to_timed_trajectory  # noqa: E402
from chapter12_reasoning.planner import Plan  # noqa: E402
from chapter12_reasoning.task_parser import SpatialRelation, parse_task  # noqa: E402
from chapter13_skills.composer import Composer  # noqa: E402
from chapter13_skills.skill import SkillInstance, SkillLibrary  # noqa: E402
from chapter13_skills.skills import make_default_library  # noqa: E402
from pibench.realrobot.calibration import ResidualTracker  # noqa: E402


@dataclass
class NorthStarReport:
    """Human-readable outcome of the north-star demo."""

    task_text: str
    parsed_skill: str
    target_object: str
    plan_verified: bool
    plan_steps: int
    final_arm_error: float
    arm_reached: bool
    mean_residual_norm: float
    skill_saved: bool
    library_path: str | None
    log: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_text": self.task_text,
            "parsed_skill": self.parsed_skill,
            "target_object": self.target_object,
            "plan_verified": self.plan_verified,
            "plan_steps": self.plan_steps,
            "final_arm_error": round(self.final_arm_error, 5),
            "arm_reached": self.arm_reached,
            "mean_residual_norm": round(self.mean_residual_norm, 5),
            "skill_saved": self.skill_saved,
            "library_path": self.library_path,
            "log": self.log,
        }


class NorthStarDemo:
    """One reproducible pass through the full autonomy loop.

    The demo intentionally separates the *manipulation scene* (used for skill
    planning and verification) from the *arm scene* (used for control and
    residual tracking).  This keeps the control model simple (motor actuators)
    while still exercising every layer of the stack.
    """

    def __init__(
        self,
        scene_xml: str | Path,
        arm_xml: str | Path | None = None,
        dt: float = 0.01,
        reach_tolerance: float = 0.05,
        trajectory_duration: float = 2.0,
    ) -> None:
        self.scene_xml = Path(scene_xml)
        self.arm_xml = Path(
            arm_xml
            if arm_xml is not None
            else Path(__file__).parent.parent
            / "chapter01_foundation"
            / "simple_6dof_arm.xml"
        )
        self.dt = dt
        self.reach_tolerance = reach_tolerance
        self.trajectory_duration = trajectory_duration

        self.ik = InverseKinematics(str(self.arm_xml))
        self.dynamics = ArmDynamics(xml_path=str(self.arm_xml))
        self.composer = Composer(self.scene_xml, library=make_default_library())

    def _build_instance(self, task_text: str) -> SkillInstance:
        """Convert a raw NL instruction into a concrete skill instance."""
        spec = parse_task(task_text)
        relation = spec.relation.value if spec.relation is not None else None
        return SkillInstance(
            skill_name=spec.skill.value,
            target_object=spec.target_object,
            reference_object=spec.reference_object,
            relation=relation,
            offset=0.15,
        )

    @staticmethod
    def _target_position_from_plan(plan: Plan) -> np.ndarray:
        """Return the first plan step that carries a target position."""
        for step in plan.steps:
            if step.target_position is not None:
                return np.asarray(step.target_position, dtype=float)
        raise ValueError("Plan contains no target position.")

    def _solve_ik(
        self,
        p_target: np.ndarray,
        q0: np.ndarray | None = None,
    ) -> tuple[np.ndarray, dict]:
        """Find a joint configuration that reaches ``p_target`` (position only)."""
        q0 = np.asarray(q0, dtype=float) if q0 is not None else np.zeros(self.dynamics.model.nq)
        # Orientation is intentionally free; the arm only needs to reach the point.
        R_target = np.eye(3)
        q, info = self.ik.ik_numeric(
            q0=q0,
            R_target=R_target,
            p_target=np.asarray(p_target, dtype=float),
            position_only=True,
            max_iters=300,
            tol=1e-4,
        )
        return q, info

    def _build_trajectory(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> Any:
        """Create a smooth timed trajectory between two configurations."""
        path = linear_interpolate(q_start, q_goal, step=0.1)
        if len(path) < 2:
            path = [q_start, q_goal]
        return plan_to_timed_trajectory(
            path,
            max_joint_velocity=1.0,
            max_joint_acceleration=2.0,
            order="cubic",
        )

    def _make_controller(self) -> JointSpacePIDController:
        """Return an inertia-scaled joint-space PID for the virtual arm."""
        M_diag = np.diag(self.dynamics.mass_matrix(np.zeros(self.dynamics.model.nq)))
        omega = 6.0
        Kp = omega**2 * M_diag
        Kd = 2.0 * np.sqrt(Kp * M_diag)
        return JointSpacePIDController(
            self.dynamics,
            Kp=Kp,
            Kd=Kd,
            gravity_comp=True,
            tau_max=self.dynamics.model.actuator_ctrlrange[:, 1],
        )

    def _execute_on_arm(
        self,
        q_start: np.ndarray,
        q_goal: np.ndarray,
    ) -> tuple[MockRealArm, list[float], ResidualTracker]:
        """Run the trajectory on MockRealArm and record residuals."""
        arm = MockRealArm(xml_path=str(self.arm_xml), dt=self.dt, control_mode="torque")
        arm.data.qpos[:] = q_start
        mujoco = __import__("mujoco")
        mujoco.mj_forward(arm.model, arm.data)

        controller = self._make_controller()
        controller.reset()
        tracker = ResidualTracker(nominal_xml_path=str(self.arm_xml))
        tracker.reset()

        trajectory = self._build_trajectory(q_start, q_goal)
        n_steps = max(1, int(round(trajectory.t_total / self.dt)))
        errors: list[float] = []

        for i in range(n_steps):
            t = i * self.dt
            q_des = trajectory.evaluate(t)
            qdot_des = trajectory.evaluate_velocity(t)
            state = arm.get_state()
            tau = controller.compute(state.q, state.qdot, q_des=q_des, qdot_des=qdot_des, dt=self.dt)
            # Record residual before stepping.
            next_state = None
            arm.send_torques(tau, dt=self.dt)
            next_state = arm.get_state()
            tracker.observe(state.q, state.qdot, tau, next_state.qdot, self.dt)
            errors.append(float(np.linalg.norm(q_des - next_state.q)))

        return arm, errors, tracker

    def run(
        self,
        task_text: str,
        library_save_path: str | Path | None = None,
    ) -> NorthStarReport:
        """Execute the full pipeline for ``task_text`` and return a report."""
        log: list[str] = []
        log.append(f"Task: {task_text!r}")

        # 1. Parse + plan + verify.
        instance = self._build_instance(task_text)
        log.append(
            f"Parsed: skill={instance.skill_name}, target={instance.target_object}, "
            f"reference={instance.reference_object}, relation={instance.relation}"
        )

        plan, verified, failures = self.composer.compose([instance])
        log.append(f"Plan verified: {verified} ({len(plan.steps)} steps)")
        if failures:
            log.extend(failures)

        # 2. Map verified plan to a joint-space goal.
        try:
            p_target = self._target_position_from_plan(plan)
            log.append(f"Target position: {np.round(p_target, 4).tolist()}")
        except ValueError as exc:
            log.append(str(exc))
            p_target = np.zeros(3)

        q_start = np.zeros(self.dynamics.model.nq)
        q_goal, ik_info = self._solve_ik(p_target, q0=q_start)
        log.append(
            f"IK: position_error={ik_info.get('position_error', -1):.5f}, "
            f"rotation_error={ik_info.get('rotation_error', -1):.5f}, "
            f"iterations={ik_info.get('iterations', -1)}"
        )

        # 3. Execute trajectory on virtual arm.
        arm, errors, tracker = self._execute_on_arm(q_start, q_goal)
        final_error = float(np.linalg.norm(arm.get_state().q - q_goal))
        arm_reached = final_error <= self.reach_tolerance
        mean_residual = tracker.mean_residual()
        mean_residual_norm = float(np.linalg.norm(mean_residual))
        log.append(f"Final arm error: {final_error:.5f} (reached={arm_reached})")
        log.append(f"Mean residual norm: {mean_residual_norm:.5f}")

        # 4. Save skill instance if the plan verified and the arm reached.
        skill_saved = False
        library_path = None
        if verified and arm_reached and library_save_path is not None:
            # Build a library with the default skill templates plus this instance.
            fresh = SkillLibrary()
            # Register skills so list is non-empty, but instances are what we save.
            default = make_default_library()
            for name in default.list_skills():
                fresh.register(default.get(name))
            fresh.add_instance(instance)
            fresh.save_json(library_save_path)
            skill_saved = True
            library_path = str(library_save_path)
            log.append(f"Skill library saved to {library_save_path}")

        return NorthStarReport(
            task_text=task_text,
            parsed_skill=instance.skill_name,
            target_object=instance.target_object,
            plan_verified=verified,
            plan_steps=len(plan.steps),
            final_arm_error=final_error,
            arm_reached=arm_reached,
            mean_residual_norm=mean_residual_norm,
            skill_saved=skill_saved,
            library_path=library_path,
            log=log,
        )
