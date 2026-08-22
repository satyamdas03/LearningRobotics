"""Chapter 14 — Recorded demo of the self-improving virtual real-sim-real loop.

Same logic as ``demo_self_improve.py``, but renders the baseline and retuned
validation runs to an MP4 so the learning signal is visible.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

# Make sibling chapters importable.
ROOT = Path(__file__).parent.parent
for sub in [
    "chapter06_dynamics",
    "chapter07_control",
    "pibench/pibench/realrobot",
    "chapter14_self_improvement",
]:
    d = ROOT / sub
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from control import ComputedTorqueController
from real_hardware import MockRealArm
from failure_detector import FailureDetector
from retuner import Retuner
from self_improvement_loop import SelfImprovementLoop
from system_id import OnlineSystemID
from dynamics import ArmDynamics

import mujoco
import imageio


class BiasedMockRealArm(MockRealArm):
    """Mock real arm with a constant per-joint torque bias."""

    def __init__(self, bias: np.ndarray, **kwargs):
        super().__init__(**kwargs)
        self.bias = np.asarray(bias, dtype=float)

    def send_torques(self, tau: np.ndarray, dt: float | None = None) -> None:
        biased_tau = np.asarray(tau, dtype=float) + self.bias
        return super().send_torques(biased_tau, dt=dt)


def _interpolate_path(waypoints: list[np.ndarray], steps: int) -> list[np.ndarray]:
    if len(waypoints) < 2:
        return waypoints
    segments = len(waypoints) - 1
    per_segment = max(1, steps // segments)
    path: list[np.ndarray] = []
    for i in range(segments):
        a, b = waypoints[i], waypoints[i + 1]
        for t in range(per_segment + 1):
            s = t / per_segment
            path.append((1 - s) * a + s * b)
    return path


def make_reference_path(nq: int) -> list[np.ndarray]:
    """A slow, smooth sweep through the joint space for calibration."""
    waypoints = [
        np.zeros(nq),
        np.full(nq, 0.15),
        np.full(nq, -0.1),
        np.zeros(nq),
    ]
    return _interpolate_path(waypoints, steps=40)


def make_task_path(nq: int, dwell: int = 40) -> list[np.ndarray]:
    waypoints = [
        np.zeros(nq),
        np.array([0.10, 0.06, 0.04, 0.02, -0.04, 0.0][:nq]),
    ]
    path = _interpolate_path(waypoints, steps=40)
    path.extend([path[-1].copy() for _ in range(dwell)])
    return path


def _record_episode(
    arm_factory,
    controller,
    q_path: list[np.ndarray],
    dt: float,
    width: int = 640,
    height: int = 480,
) -> tuple[list[np.ndarray], list[float]]:
    """Run one episode and return rendered frames + tracking errors."""
    arm = arm_factory()
    if hasattr(controller, "reset"):
        controller.reset()

    renderer = mujoco.Renderer(arm.model, height=height, width=width)
    # Frame the arm from a fixed, attractive angle so the viewer sees all joints.
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = arm.model.body("base").id
    camera.azimuth = 135.0
    camera.elevation = -20.0
    camera.distance = 2.5
    camera.lookat[:] = [0.0, 0.0, 0.6]

    frames: list[np.ndarray] = []
    errors: list[float] = []

    state = arm.get_state()
    for q_target in q_path:
        tau = controller.compute(state.q, state.qdot, q_des=q_target, dt=dt)
        arm.send_torques(tau, dt=dt)
        state = arm.get_state()
        errors.append(float(np.linalg.norm(q_target - state.q)))
        mujoco.mj_forward(arm.model, arm.data)
        renderer.update_scene(arm.data, camera=camera)
        frames.append(renderer.render().copy())

    return frames, errors


def _make_overlay_frames(
    frames: list[np.ndarray],
    errors: list[float],
    label: str,
    color: tuple[int, int, int] = (0, 255, 0),
) -> list[np.ndarray]:
    from PIL import Image, ImageDraw, ImageFont

    out: list[np.ndarray] = []
    for idx, frame in enumerate(frames):
        img = Image.fromarray(frame)
        draw = ImageDraw.Draw(img)
        try:
            font = ImageFont.truetype("arial.ttf", 20)
        except Exception:
            font = ImageFont.load_default()
        draw.text((10, 10), label, fill=color, font=font)
        draw.text((10, 35), f"frame={idx}", fill=(255, 255, 255), font=font)
        draw.text((10, 60), f"error={errors[idx]:.4f}", fill=(255, 255, 255), font=font)
        out.append(np.array(img))
    return out


def main() -> int:
    xml_path = ROOT / "chapter01_foundation" / "simple_6dof_arm.xml"
    dt = 0.01
    dynamics = ArmDynamics(xml_path=xml_path)
    nq = dynamics.model.nq

    Kp = np.full(nq, 100.0)
    Kd = np.full(nq, 20.0)
    baseline = ComputedTorqueController(
        dynamics=dynamics,
        Kp=Kp,
        Kd=Kd,
        tau_max=40.0,
    )

    true_bias = np.array([0.3, -0.2, 0.15, 0.0, 0.0, 0.0], dtype=float)

    def arm_factory() -> MockRealArm:
        return BiasedMockRealArm(
            bias=true_bias,
            xml_path=str(xml_path),
            dt=dt,
            control_mode="torque",
            gear_ratio=1.0,
            torque_noise_std=0.0,
            position_noise_std=0.0,
            velocity_noise_std=0.0,
        )

    detector = FailureDetector(position_threshold=0.05, residual_threshold=5.0)
    system_id = OnlineSystemID(nominal_xml_path=xml_path)
    retuner = Retuner(
        ComputedTorqueController,
        dynamics=dynamics,
        Kp=Kp,
        Kd=Kd,
        tau_max=40.0,
    )

    loop = SelfImprovementLoop(
        arm_factory=arm_factory,
        baseline_controller=baseline,
        retuner=retuner,
        detector=detector,
        system_id=system_id,
        reference_q_path=make_reference_path(nq),
        dt=dt,
        retune_kwargs={"disable_offset": False, "offset_gain": 1.0, "gear_compensation_gain": 0.0},
    )

    q_path = make_task_path(nq)
    report = loop.improve(q_path, n_ab_trials=10)

    # Record one baseline episode and one retuned episode for visual comparison.
    print("Recording baseline episode...")
    baseline_frames, baseline_errors = _record_episode(arm_factory, baseline, q_path, dt)
    baseline_frames = _make_overlay_frames(baseline_frames, baseline_errors, "BASELINE (uncompensated)", color=(255, 80, 80))

    retuned_controller = retuner.retune(
        type("E", (), report.mismatch_estimate)(),
        disable_offset=False,
        offset_gain=1.0,
        gear_compensation_gain=0.0,
    ).controller

    print("Recording retuned episode...")
    retuned_frames, retuned_errors = _record_episode(arm_factory, retuned_controller, q_path, dt)
    retuned_frames = _make_overlay_frames(retuned_frames, retuned_errors, "RETUNED (bias cancelled)", color=(80, 255, 80))

    # Build side-by-side comparison video at 30 fps.
    print("Writing video...")
    output_dir = ROOT / "output"
    output_dir.mkdir(exist_ok=True)
    video_path = output_dir / "self_improve_baseline_vs_retuned.mp4"

    combined: list[np.ndarray] = []
    for b, r in zip(baseline_frames, retuned_frames):
        h = max(b.shape[0], r.shape[0])
        # Pad if needed; they should be the same size.
        if b.shape[0] < h:
            b = np.pad(b, ((0, h - b.shape[0]), (0, 0), (0, 0)))
        if r.shape[0] < h:
            r = np.pad(r, ((0, h - r.shape[0]), (0, 0), (0, 0)))
        combined.append(np.concatenate([b, r], axis=1))

    imageio.mimsave(str(video_path), combined, fps=30, quality=8)
    print(f"Video saved to {video_path}")

    # Print numeric report.
    estimated_bias = np.asarray(report.mismatch_estimate["torque_offset"], dtype=float)
    print("=" * 60)
    print("Self-Improvement Loop Report")
    print("=" * 60)
    for line in report.log:
        print(line)
    print("-" * 60)
    print(f"True bias:            {true_bias}")
    print(f"Estimated bias:       {estimated_bias}")
    print(f"Bias estimation error: {np.linalg.norm(true_bias - estimated_bias):.4f}")
    print(f"A/B improved:          {report.ab_result['improved']}")
    print(f"Baseline mean error:   {report.ab_result['baseline_mean_error']:.4f}")
    print(f"Retuned mean error:    {report.ab_result['retuned_mean_error']:.4f}")
    print(f"Baseline successes:    {report.ab_result['baseline_successes']}/10")
    print(f"Retuned successes:     {report.ab_result['retuned_successes']}/10")
    return 0 if report.success else 1


if __name__ == "__main__":
    raise SystemExit(main())
