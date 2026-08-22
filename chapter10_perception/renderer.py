"""MuJoCo RGB/depth renderer for the perception stack.

Thin wrapper around `mujoco.Renderer` so the rest of the pipeline can request
a camera view without knowing the MuJoCo rendering API.
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np


class MujocoRenderer:
    """Render RGB and depth images from a MuJoCo model + data pair."""

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        width: int = 640,
        height: int = 480,
    ) -> None:
        self.model = model
        self.data = data
        self.width = width
        self.height = height
        self._renderer = mujoco.Renderer(model, height=height, width=width)
        self._camera = mujoco.MjvCamera()

    def render_rgb(
        self,
        camera_name: str | None = None,
        lookat: np.ndarray | None = None,
        distance: float | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
    ) -> np.ndarray:
        """Return an RGB image as a uint8 HWC array."""
        self._update_camera(camera_name, lookat, distance, azimuth, elevation)
        self._renderer.update_scene(self.data, camera=self._camera)
        return self._renderer.render()

    def render_depth(
        self,
        camera_name: str | None = None,
        lookat: np.ndarray | None = None,
        distance: float | None = None,
        azimuth: float | None = None,
        elevation: float | None = None,
    ) -> np.ndarray:
        """Return a depth image as a float32 HW array (distance from camera)."""
        self._update_camera(camera_name, lookat, distance, azimuth, elevation)
        self._renderer.enable_depth_rendering()
        self._renderer.update_scene(self.data, camera=self._camera)
        depth = self._renderer.render()
        self._renderer.disable_depth_rendering()
        return depth

    def _update_camera(
        self,
        camera_name: str | None,
        lookat: np.ndarray | None,
        distance: float | None,
        azimuth: float | None,
        elevation: float | None,
    ) -> None:
        """Mutate the renderer's camera configuration if explicit params given."""
        if camera_name is not None:
            self._camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self._camera.fixedcamid = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name
            )
            if self._camera.fixedcamid < 0:
                raise ValueError(f"Camera '{camera_name}' not found in model")
        if lookat is not None:
            self._camera.lookat[:] = lookat
        if distance is not None:
            self._camera.distance = distance
        if azimuth is not None:
            self._camera.azimuth = azimuth
        if elevation is not None:
            self._camera.elevation = elevation

    @classmethod
    def from_xml(cls, xml_path: str | Path, **kwargs) -> "MujocoRenderer":
        """Convenience factory from an XML file path."""
        model = mujoco.MjModel.from_xml_path(str(xml_path))
        data = mujoco.MjData(model)
        return cls(model, data, **kwargs)

    def close(self) -> None:
        """Release underlying renderer resources."""
        self._renderer.close()
