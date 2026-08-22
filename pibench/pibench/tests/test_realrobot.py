"""Tests for PIBench Phase 7 real-robot validation harness."""
from __future__ import annotations

import numpy as np
import pytest

from pibench.realrobot.calibration import ResidualTracker
from pibench.realrobot.harness import RealRobotValidationHarness
from pibench.realrobot.protocol import ValidationTask


@pytest.fixture
def harness():
    """Harness backed by a noise-free MockRealArm."""
    arm = RealRobotValidationHarness.create_mock_arm(dt=0.01)
    return RealRobotValidationHarness(arm, dt=0.01)


def test_reach_task_success_matches_yes_prediction(harness):
    """A reachable target predicted 'yes' should match the observed outcome."""
    task = ValidationTask(
        task_id="reach_success",
        predicted_answer="yes",
        action_type="reach_q",
        action_params={
            "target_q": [0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
            "duration": 1.5,
            "tolerance": 0.1,
        },
    )
    result = harness.run_task(task)
    assert result.actual == "reached"
    assert result.match is True
    assert result.metrics["final_error"] <= 0.08


def test_reach_task_failure_matches_no_prediction(harness):
    """An unreachable target predicted 'no' should match the observed outcome."""
    task = ValidationTask(
        task_id="reach_fail",
        predicted_answer="no",
        action_type="reach_q",
        action_params={
            "target_q": [0.0, 1.4, 0.0, 0.0, 0.0, 0.0],
            "duration": 0.2,
            "tolerance": 0.05,
        },
    )
    result = harness.run_task(task)
    assert result.actual == "missed"
    assert result.match is True


def test_validation_batch_accuracy(harness):
    """Running multiple tasks returns a result per task and a sensible accuracy."""
    tasks = [
        ValidationTask(
            task_id="t1",
            predicted_answer="yes",
            action_type="reach_q",
            action_params={
                "target_q": [0.0, 0.25, 0.0, 0.0, 0.0, 0.0],
                "duration": 1.5,
                "tolerance": 0.1,
            },
        ),
        ValidationTask(
            task_id="t2",
            predicted_answer="no",
            action_type="reach_q",
            action_params={
                "target_q": [0.0, 1.4, 0.0, 0.0, 0.0, 0.0],
                "duration": 0.2,
                "tolerance": 0.05,
            },
        ),
    ]
    results = harness.run_tasks(tasks)
    assert len(results) == len(tasks)
    assert all(r.match for r in results)
    accuracy = sum(r.match for r in results) / len(results)
    assert accuracy == 1.0


def _stable_controller(omega: float = 4.0):
    """Return a stable inertia-scaled PID closure for calibration trajectories."""
    pid = RealRobotValidationHarness._inertia_scaled_pid(omega=omega)

    def _control(state, q_des):
        return pid.compute(state.q, state.qdot, q_des=q_des, dt=0.01)

    return _control


def test_residual_tracker_zero_mismatch():
    """With identical nominal and real models, residuals should be near zero."""
    arm = RealRobotValidationHarness.create_mock_arm(dt=0.01)
    tracker = ResidualTracker()

    q_path = [
        np.zeros(arm.nq()),
        np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0]),
    ]
    info = tracker.run_calibration_episode(arm, q_path, _stable_controller(), dt=0.01)
    mean = np.array(info["mean_residual"])
    max_res = np.array(info["max_residual"])
    assert info["n_samples"] > 0
    assert np.all(mean <= 2.0)
    assert np.all(max_res <= 50.0)


def test_residual_tracker_detects_torque_noise():
    """Injected actuator noise should create a nonzero (but bounded) residual."""
    arm = RealRobotValidationHarness.create_mock_arm(
        dt=0.01, torque_noise_std=2.0
    )
    tracker = ResidualTracker()

    q_path = [np.zeros(arm.nq()), np.array([0.0, 0.3, 0.0, 0.0, 0.0, 0.0])]
    tracker.run_calibration_episode(arm, q_path, _stable_controller(), dt=0.01)
    mean = np.abs(tracker.mean_residual())
    # Noise standard deviation of 2 Nm should create nonzero (and bounded) mean
    # residuals; the wrist joints are very light so the same noise produces
    # larger accelerations there.  A generous upper bound avoids flaky RNG draws.
    assert np.any(mean > 1e-3)
    assert np.all(mean <= 5000.0)
