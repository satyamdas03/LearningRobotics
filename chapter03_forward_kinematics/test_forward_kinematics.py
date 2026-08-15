"""Tests for Chapter 3 forward kinematics."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from forward_kinematics import Arm6DOFFK, mujoco_fk


@pytest.fixture
def xml_path():
    return str(Path(__file__).parent.parent / "chapter01_foundation" / "simple_6dof_arm.xml")


def test_poe_matches_mujoco_default(xml_path):
    fk = Arm6DOFFK()
    q = np.zeros(6)
    assert np.allclose(fk.end_effector_position(q), mujoco_fk(xml_path, q))


def test_poe_matches_mujoco_random_configs(xml_path):
    fk = Arm6DOFFK()
    rng = np.random.default_rng(0)
    for _ in range(20):
        q = rng.uniform(-math.pi / 2, math.pi / 2, size=6)
        assert np.allclose(fk.end_effector_position(q), mujoco_fk(xml_path, q), atol=1e-10)


def test_geometric_matches_poe():
    fk = Arm6DOFFK()
    rng = np.random.default_rng(1)
    for _ in range(20):
        q = rng.uniform(-math.pi / 2, math.pi / 2, size=6)
        assert np.allclose(fk.poe_fk(q), fk.geometric_fk(q), atol=1e-10)


def test_waist_rotation():
    fk = Arm6DOFFK()
    q = np.array([math.radians(90), 0.0, 0.0, 0.0, 0.0, 0.0])
    pos = fk.end_effector_position(q)
    assert np.allclose(pos, [0.0, 0.96, 0.85], atol=1e-10)
