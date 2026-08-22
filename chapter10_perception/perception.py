"""Simple perception helpers for the simulated manipulation scene.

Because this is simulation, we can extract ground-truth object poses directly
from MuJoCo for debugging and closed-loop control.  A real robot would replace
these helpers with vision-model outputs, but the downstream controller interface
stays the same.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass
class DetectedObject:
    """A single perceived object in the scene."""

    name: str
    body_id: int
    position: np.ndarray
    color: tuple[float, float, float, float]
    size: tuple[float, float, float]
    geom_type: int


class SceneObjectDetector:
    """Extract object poses/boxes from a MuJoCo scene by body/geom metadata."""

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        self.model = model
        self.data = data

    def detect_objects(
        self,
        body_name_prefix: str = "",
        exclude_arm: bool = True,
    ) -> list[DetectedObject]:
        """Return all matching bodies with their first geom's color and size."""
        mujoco.mj_forward(self.model, self.data)
        objects: list[DetectedObject] = []
        for body_id in range(self.model.nbody):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, body_id)
            if exclude_arm and name in {"arm_base", "link1", "link2", "link3", "link4", "link5", "link6", "table"}:
                continue
            if body_name_prefix and not name.startswith(body_name_prefix):
                continue
            if self.model.body_parentid[body_id] != 0 and not body_name_prefix:
                # Skip child bodies of the arm unless explicitly requested.
                continue
            geom_id = self._first_geom_for_body(body_id)
            if geom_id < 0:
                continue
            geom_type = self.model.geom_type[geom_id]
            size = self.model.geom_size[geom_id].copy()
            rgba = self.model.geom_rgba[geom_id].copy()
            position = self.data.xpos[body_id].copy()
            objects.append(
                DetectedObject(
                    name=name,
                    body_id=body_id,
                    position=position,
                    color=tuple(rgba.tolist()),
                    size=tuple(size.tolist()),
                    geom_type=int(geom_type),
                )
            )
        return objects

    def find_object_by_color(
        self,
        color_hint: tuple[float, float, float],
        tolerance: float = 0.25,
        exclude_arm: bool = True,
    ) -> DetectedObject | None:
        """Return the object whose RGB color is closest to ``color_hint``."""
        objects = self.detect_objects(exclude_arm=exclude_arm)
        if not objects:
            return None
        best: DetectedObject | None = None
        best_score = float("inf")
        for obj in objects:
            rgb = np.array(obj.color[:3])
            score = float(np.linalg.norm(rgb - np.array(color_hint)))
            if score < best_score and score <= tolerance:
                best_score = score
                best = obj
        return best

    def _first_geom_for_body(self, body_id: int) -> int:
        """Return the first geom that belongs to ``body_id`` or -1."""
        # MuJoCo stores body->geom ranges in body_geomadr and body_geomnum.
        adr = self.model.body_geomadr[body_id]
        num = self.model.body_geomnum[body_id]
        if num > 0:
            return int(adr)
        return -1


@dataclass
class CameraParams:
    """Pinhole camera intrinsics + resolution."""

    width: int
    height: int
    fovy: float

    @property
    def focal_length_px(self) -> float:
        """Vertical focal length in pixels."""
        return self.height / (2.0 * np.tan(np.deg2rad(self.fovy) / 2.0))

    def project_world_to_pixel(
        self,
        world_point: np.ndarray,
        camera_transform: np.ndarray,
    ) -> tuple[int, int] | None:
        """Project a 3D world point into image UV coordinates.

        ``camera_transform`` is the 4x4 camera-to-world transform (i.e. column
        vectors are the camera axes in world coordinates, last column is origin).
        Returns ``(u, v)`` in pixel coordinates or ``None`` if behind the camera.
        """
        T_c_w = np.linalg.inv(camera_transform)
        point_cam = T_c_w @ np.append(world_point, 1.0)
        if point_cam[2] <= 0:
            return None
        f = self.focal_length_px
        u = int(self.width / 2 + f * point_cam[0] / point_cam[2])
        v = int(self.height / 2 - f * point_cam[1] / point_cam[2])  # y-up image convention
        return u, v


def build_camera_transform(
    lookat: np.ndarray,
    distance: float,
    azimuth_deg: float,
    elevation_deg: float,
) -> np.ndarray:
    """Build a camera-to-world SE(3) matrix.

    Returns the 4x4 matrix ``T_w_c`` whose columns are the camera axes expressed
    in world coordinates.  The camera frame is +X right, +Y up, +Z forward (into
    the scene), matching the projection convention used by ``CameraParams``.
    """
    az = np.deg2rad(azimuth_deg)
    el = np.deg2rad(elevation_deg)
    # Spherical offset from lookat to eye.
    offset = np.array([
        distance * np.cos(el) * np.cos(az),
        distance * np.cos(el) * np.sin(az),
        distance * np.sin(el),
    ])
    eye = lookat + offset

    # Camera Z points from eye toward lookat (into the scene).
    z_axis = -offset / (np.linalg.norm(offset) + 1e-12)
    world_up = np.array([0.0, 0.0, 1.0])
    y_axis = world_up - np.dot(world_up, z_axis) * z_axis
    y_axis /= np.linalg.norm(y_axis) + 1e-12
    x_axis = np.cross(y_axis, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12

    T = np.eye(4)
    T[:3, 0] = x_axis
    T[:3, 1] = y_axis
    T[:3, 2] = z_axis
    T[:3, 3] = eye
    return T
