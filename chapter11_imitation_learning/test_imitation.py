"""Tests for Milestone 5 — imitation learning / behavior cloning."""
from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np
import pytest

from behavior_cloning import (
    BCConfig,
    BCPolicy,
    prepare_bc_dataset,
    prepare_goal_residual_dataset,
    prepare_ik_dataset,
    rollout_goal_policy,
    rollout_policy,
)
from expert import DEFAULT_ARM_XML, generate_reach_dataset, record_reach_trajectory
from teleop import TeleopRecorder


HERE = Path(__file__).parent


def test_expert_reach_trajectory_records_and_converges():
    """A recorded reach trajectory should reach near the IK goal."""
    demo = record_reach_trajectory(
        xml_path=DEFAULT_ARM_XML,
        start_q=np.zeros(6),
        goal_ee_position=np.array([0.8, 0.0, 0.6]),
        duration=2.0,
        dt=0.01,
    )
    assert demo["states"].shape[0] > 0
    assert demo["actions"].shape == demo["states"].shape
    assert demo["next_states"].shape == demo["states"].shape

    # Final end-effector position should be close to the target.
    final_ee = demo["ee_positions"][-1]
    error = float(np.linalg.norm(final_ee - demo["goal_ee"]))
    assert error <= 0.05


def test_generate_dataset_has_multiple_demos():
    """generate_reach_dataset should produce the requested number of demos."""
    demos = generate_reach_dataset(
        xml_path=DEFAULT_ARM_XML,
        n_demos=4,
        duration=1.0,
        dt=0.02,
        seed=42,
    )
    assert len(demos) == 4
    for demo in demos:
        assert demo["states"].ndim == 2
        assert demo["states"].shape[1] == 6
        assert demo["states"].shape[0] > 1


def test_behavior_cloning_fits_expert_data():
    """A small MLP should overfit a small expert dataset (sanity check)."""
    demos = generate_reach_dataset(
        xml_path=DEFAULT_ARM_XML,
        n_demos=8,
        duration=1.0,
        dt=0.02,
        seed=0,
    )
    X, Y = prepare_bc_dataset(demos)
    assert X.shape[1] == 9
    assert Y.shape[1] == 6
    assert X.shape[0] == Y.shape[0]

    cfg = BCConfig(input_dim=9, hidden_dim=64, output_dim=6, epochs=600, learning_rate=2e-3, seed=1)
    policy = BCPolicy(config=cfg)
    history = policy.fit(X, Y, verbose=False)

    final_mse = history[-1]
    assert final_mse < 0.001
    assert history[0] > history[-1]


def test_goal_policy_reaches_near_goal():
    """A one-shot IK behavior-cloning policy should place the EE near the target."""
    demos = generate_reach_dataset(
        xml_path=DEFAULT_ARM_XML,
        n_demos=20,
        duration=1.0,
        dt=0.02,
        seed=4,
    )
    X, Y = prepare_ik_dataset(demos)

    cfg = BCConfig(input_dim=3, hidden_dim=96, output_dim=6, epochs=800, learning_rate=2e-3, seed=5)
    policy = BCPolicy(config=cfg)
    policy.fit(X, Y, verbose=False)

    # Test on a held-out demonstration: train on all but the first demo.
    held_out = demos[0]
    train_demos = demos[1:]
    X_train, Y_train = prepare_ik_dataset(train_demos)
    policy.fit(X_train, Y_train, verbose=False)

    goal = held_out["goal_ee"]
    result = rollout_goal_policy(xml_path=DEFAULT_ARM_XML, policy=policy, goal_ee_position=goal)
    final_ee = result["ee_position"]
    final_error = float(np.linalg.norm(final_ee - goal))
    assert final_error <= 0.15


def test_teleop_recorder_save_load_roundtrip(tmp_path: Path) -> None:
    """The teleop recorder should serialize and deserialize exactly."""
    recorder = TeleopRecorder(joint_count=6)
    for i in range(10):
        recorder.record(np.linspace(-0.1, 0.1, 6) * i, source="test")

    save_path = tmp_path / "teleop.json"
    recorder.save_json(save_path)

    loaded = TeleopRecorder.load_json(save_path)
    assert len(loaded.samples) == 10
    for original, sample in zip(recorder.samples, loaded.samples):
        np.testing.assert_allclose(original.q, sample.q, atol=1e-9)
        assert original.source == sample.source


def test_teleop_recorder_as_trajectory_format():
    """as_trajectory must return a dict compatible with expert demos."""
    recorder = TeleopRecorder()
    for i in range(5):
        recorder.record(np.zeros(6) + i * 0.01)

    traj = recorder.as_trajectory(dt=0.01)
    assert traj["states"].shape == (4, 6)
    assert traj["actions"].shape == (4, 6)
    np.testing.assert_allclose(traj["actions"], traj["next_states"])
