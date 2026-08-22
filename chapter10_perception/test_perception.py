"""Tests for Milestone 4 — simulated camera perception stack."""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

from demo_perception_controller import reach_red_block
from perception import SceneObjectDetector, build_camera_transform
from renderer import MujocoRenderer


HERE = Path(__file__).parent


@pytest.fixture
def model_data():
    """Load the manipulation scene and return (model, data)."""
    model = mujoco.MjModel.from_xml_path(str(HERE / "scene.xml"))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    return model, data


def test_renderer_produces_rgb_image(model_data):
    """The MuJoCo renderer should return a non-empty RGB array."""
    model, data = model_data
    mujoco.mj_forward(model, data)
    renderer = MujocoRenderer(model, data, width=320, height=240)
    rgb = renderer.render_rgb(
        lookat=np.array([0.8, 0.0, 0.55]),
        distance=1.2,
        azimuth=135.0,
        elevation=30.0,
    )
    renderer.close()
    assert rgb.shape == (240, 320, 3)
    assert rgb.dtype == np.uint8
    assert np.any(rgb > 0)


def test_detector_finds_red_and_blue_blocks(model_data):
    """The scene detector should return the two colored blocks."""
    model, data = model_data
    detector = SceneObjectDetector(model, data)
    objects = detector.detect_objects(exclude_arm=True)
    names = {o.name for o in objects}
    assert "red_block" in names
    assert "blue_block" in names

    red = detector.find_object_by_color((0.9, 0.1, 0.1), tolerance=0.3)
    blue = detector.find_object_by_color((0.1, 0.1, 0.9), tolerance=0.3)
    assert red is not None and red.name == "red_block"
    assert blue is not None and blue.name == "blue_block"


def test_color_detector_distinguishes_red_from_blue(model_data):
    """The red color hint should not accidentally pick the blue block."""
    model, data = model_data
    detector = SceneObjectDetector(model, data)
    red = detector.find_object_by_color((0.9, 0.1, 0.1), tolerance=0.3)
    assert red is not None
    assert red.name == "red_block"
    assert red.position[2] > 0.4  # sitting on the table


def test_camera_transform_is_se3():
    """build_camera_transform must return a valid 4x4 camera-to-world matrix."""
    T = build_camera_transform(
        lookat=np.array([0.8, 0.0, 0.55]),
        distance=1.0,
        azimuth_deg=90.0,
        elevation_deg=-20.0,
    )
    assert T.shape == (4, 4)
    R = T[:3, :3]
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-6)
    assert np.linalg.det(R) > 0


def test_camera_projection_matches_world_point(model_data):
    """Projecting a world point should put it inside the rendered image."""
    model, data = model_data
    detector = SceneObjectDetector(model, data)
    red = detector.find_object_by_color((0.9, 0.1, 0.1), tolerance=0.3)
    assert red is not None

    from perception import CameraParams

    T = build_camera_transform(
        lookat=np.array([0.8, 0.0, 0.55]),
        distance=1.2,
        azimuth_deg=135.0,
        elevation_deg=30.0,
    )
    cam = CameraParams(width=640, height=480, fovy=45.0)
    uv = cam.project_world_to_pixel(red.position, T)
    assert uv is not None
    u, v = uv
    assert 0 <= u < 640
    assert 0 <= v < 480


def test_perception_to_reach_pipeline():
    """The full perception → IK → control demo should reach near the red block."""
    info = reach_red_block(HERE / "scene.xml", duration=3.0, dt=0.01)
    assert info["final_error"] <= 0.08
