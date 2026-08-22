"""Chapter 14 — Tests for the self-improving virtual real-sim-real loop."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).parent.parent
for sub in ["chapter06_dynamics", "chapter07_control", "chapter14_self_improvement"]:
    d = ROOT / sub
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))

from chapter07_control.control import JointSpacePIDController
from chapter07_control.real_hardware import MockRealArm, VirtualArmFactory
from chapter14_self_improvement.ab_experiment import ABExperiment
from chapter14_self_improvement.failure_detector import FailureDetector
from chapter14_self_improvement.retuner import Retuner
from chapter14_self_improvement.self_improvement_loop import SelfImprovementLoop
from chapter14_self_improvement.system_id import OnlineSystemID
from dynamics import ArmDynamics


@pytest.fixture
def xml_path():
    return ROOT / "chapter01_foundation" / "simple_6dof_arm.xml"


@pytest.fixture
def dynamics(xml_path):
    return ArmDynamics(xml_path=xml_path)


@pytest.fixture
def nq(dynamics):
    return dynamics.model.nq


@pytest.fixture
def baseline_controller(dynamics, nq):
    return JointSpacePIDController(
        dynamics=dynamics,
        Kp=np.full(nq, 80.0),
        Kd=np.full(nq, 10.0),
        Ki=np.full(nq, 2.0),
        gravity_comp=True,
        tau_max=50.0,
    )


@pytest.fixture
def arm_factory(xml_path, nq):
    factory = VirtualArmFactory(xml_path=str(xml_path), base_dt=0.01, seed=7)

    def _make():
        return factory.create(
            nq=nq,
            control_mode="torque",
            randomize_gear=True,
            randomize_friction=False,
            randomize_sensors=False,
            randomize_delays=False,
            randomize_actuator_lag=False,
            gear_range=(0.9, 1.1),
        )

    return _make


def test_failure_detector_flags_large_position_error():
    det = FailureDetector(position_threshold=0.1, residual_threshold=5.0)
    q_des = np.zeros(6)
    q_actual = np.full(6, 0.2)
    report = det.check_reach("t", q_des, q_actual)
    assert report.failed
    assert report.primary_axis == "tracking_error"


def test_failure_detector_flags_dynamics_mismatch():
    det = FailureDetector(position_threshold=0.1, residual_threshold=1.0)
    report = det.check_reach(
        "t", np.zeros(6), np.zeros(6), residual=np.full(6, 2.0)
    )
    assert report.failed
    assert report.primary_axis == "dynamics_mismatch"


def test_system_id_returns_estimate(arm_factory, baseline_controller, nq):
    sid = OnlineSystemID()
    q_path = [np.zeros(nq), np.full(nq, 0.1), np.zeros(nq)]
    estimate = sid.calibrate_on_trajectory(
        arm_factory(), q_path, baseline_controller, dt=0.01
    )
    assert estimate.n_samples > 0
    assert estimate.torque_offset.shape == (nq,)
    assert estimate.estimated_gear_ratio.shape == (nq,)


def test_retuner_produces_compensated_controller(
    dynamics, baseline_controller, nq, arm_factory
):
    sid = OnlineSystemID()
    q_path = [np.zeros(nq), np.full(nq, 0.1), np.zeros(nq)]
    estimate = sid.calibrate_on_trajectory(
        arm_factory(), q_path, baseline_controller, dt=0.01
    )
    retuner = Retuner(
        JointSpacePIDController,
        dynamics=dynamics,
        Kp=np.full(nq, 80.0),
        Kd=np.full(nq, 10.0),
        Ki=np.full(nq, 2.0),
    )
    retuning = retuner.retune(estimate)
    q = np.zeros(nq)
    qdot = np.zeros(nq)
    tau = retuning.controller.compute(q, qdot, q_des=np.full(nq, 0.1), dt=0.01)
    assert tau.shape == (nq,)


def test_ab_experiment_runs(arm_factory, baseline_controller, nq):
    detector = FailureDetector(position_threshold=0.05, residual_threshold=5.0)
    retuner = Retuner(
        JointSpacePIDController,
        dynamics=ArmDynamics(xml_path=ROOT / "chapter01_foundation" / "simple_6dof_arm.xml"),
        Kp=np.full(nq, 80.0),
        Kd=np.full(nq, 10.0),
        Ki=np.full(nq, 2.0),
    )
    sid = OnlineSystemID()
    q_path = [np.zeros(nq), np.full(nq, 0.1), np.zeros(nq)]
    estimate = sid.calibrate_on_trajectory(
        arm_factory(), q_path, baseline_controller, dt=0.01
    )
    retuned = retuner.retune(estimate).controller
    experiment = ABExperiment(
        arm_factory=arm_factory,
        baseline_controller=baseline_controller,
        retuned_controller=retuned,
        q_path=q_path,
        detector=detector,
        dt=0.01,
    )
    result = experiment.run(n_trials=3)
    assert 0 <= result.baseline_successes <= 3
    assert 0 <= result.retuned_successes <= 3
    assert len(result.baseline_errors) == 3
    assert len(result.retuned_errors) == 3


def test_self_improvement_loop_runs(arm_factory, baseline_controller, nq, dynamics):
    detector = FailureDetector(position_threshold=0.05, residual_threshold=5.0)
    system_id = OnlineSystemID()
    retuner = Retuner(
        JointSpacePIDController,
        dynamics=dynamics,
        Kp=np.full(nq, 80.0),
        Kd=np.full(nq, 10.0),
        Ki=np.full(nq, 2.0),
    )
    loop = SelfImprovementLoop(
        arm_factory=arm_factory,
        baseline_controller=baseline_controller,
        retuner=retuner,
        detector=detector,
        system_id=system_id,
        reference_q_path=[np.zeros(nq), np.full(nq, 0.1), np.zeros(nq)],
        dt=0.01,
    )
    task_path = [np.zeros(nq), np.full(nq, 0.1)]
    report = loop.improve(task_path, n_ab_trials=3)
    assert isinstance(report.log, list)
    assert "Initial reach" in report.log[0]
    assert "A/B" in report.log[-1]
    assert "baseline_mean_error" in report.ab_result
    assert "retuned_mean_error" in report.ab_result
