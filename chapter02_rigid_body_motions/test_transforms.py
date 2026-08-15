"""Tests for Chapter 2 rigid-body motion utilities."""
from __future__ import annotations

import math

import numpy as np

from transforms import (
    axis_angle,
    euler_xyz,
    homogeneous_transform,
    inverse_transform,
    rotx,
    roty,
    rotz,
    rotation_matrix_to_euler_xyz,
    transform_point,
)


def test_rotz_90():
    R = rotz(math.radians(90))
    assert np.allclose(R @ np.array([1.0, 0.0, 0.0]), [0.0, 1.0, 0.0], atol=1e-10)


def test_rotation_matrix_properties():
    R = euler_xyz(math.radians(10), math.radians(20), math.radians(30))
    assert np.allclose(R @ R.T, np.eye(3))
    assert abs(np.linalg.det(R) - 1.0) < 1e-10


def test_euler_round_trip():
    roll = math.radians(25)
    pitch = math.radians(40)
    yaw = math.radians(-55)
    R = euler_xyz(roll, pitch, yaw)
    recovered = rotation_matrix_to_euler_xyz(R)
    assert np.allclose(recovered, [roll, pitch, yaw])


def test_axis_angle_matches_rotz():
    R_axis = axis_angle([0, 0, 1], math.radians(90))
    R_z = rotz(math.radians(90))
    assert np.allclose(R_axis, R_z)


def test_homogeneous_transform_inverse():
    T = homogeneous_transform(rotx(math.radians(30)), [1.0, 2.0, 3.0])
    T_inv = inverse_transform(T)
    assert np.allclose(T @ T_inv, np.eye(4))


def test_transform_point():
    T = homogeneous_transform(rotz(math.radians(90)), [1.0, 0.0, 0.0])
    p = np.array([1.0, 0.0, 0.0])
    assert np.allclose(transform_point(T, p), [1.0, 1.0, 0.0], atol=1e-10)
