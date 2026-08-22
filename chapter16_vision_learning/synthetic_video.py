"""Synthetic video generator for vision-based skill learning.

Records a MuJoCo manipulation scene to a ordinary RGB video file (no labels,
no simulation state) so that a vision parser must infer objects, motion, and
spatial relations from pixels alone.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import imageio
import mujoco
import numpy as np
from PIL import Image


def _body_qpos_adr(model: mujoco.MjModel, body_name: str) -> int:
    """Return the qpos address of the free joint attached to ``body_name``."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    jnt_id = model.body_jntadr[body_id]
    return int(model.jnt_qposadr[jnt_id])


def generate_push_video(
    output_path: str | Path,
    xml_path: str | Path | None = None,
    n_frames: int = 60,
    fps: int = 30,
    width: int = 640,
    height: int = 480,
) -> Path:
    """Render a video of the red block being pushed toward the blue block.

    The generated video is a plain RGB MP4 with no overlays, no labels, and no
    simulation metadata — exactly the kind of "ordinary video" a vision learner
    would receive.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if xml_path is None:
        xml_path = Path(__file__).parent.parent / "chapter10_perception" / "scene.xml"
    xml_path = Path(xml_path)

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    renderer = mujoco.Renderer(model, height=height, width=width)

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "table")
    camera.azimuth = 150.0
    camera.elevation = -30.0
    camera.distance = 1.6
    camera.lookat[:] = [0.8, 0.0, 0.5]

    red_adr = _body_qpos_adr(model, "red_block")
    blue_adr = _body_qpos_adr(model, "blue_block")

    # Ground-truth start/end positions for the red block.
    start_pos = np.array([0.8, 0.15, 0.46])
    end_pos = np.array([0.8, -0.11, 0.46])  # just next to the blue block

    # Keep the blue block fixed for this demo.
    blue_pos = np.array([0.8, -0.15, 0.46])

    writer = imageio.get_writer(output_path, fps=fps, codec="libx264", quality=8)

    for i in range(n_frames):
        alpha = i / max(n_frames - 1, 1)
        pos = start_pos + alpha * (end_pos - start_pos)

        data.qpos[red_adr : red_adr + 3] = pos
        data.qpos[red_adr + 3 : red_adr + 7] = [1.0, 0.0, 0.0, 0.0]
        data.qpos[blue_adr : blue_adr + 3] = blue_pos
        data.qpos[blue_adr + 3 : blue_adr + 7] = [1.0, 0.0, 0.0, 0.0]

        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        frame = renderer.render()
        writer.append_data(frame)

    writer.close()
    renderer.close()
    return output_path


def sample_frames(video_path: str | Path, n: int = 4) -> list[Image.Image]:
    """Sample ``n`` evenly-spaced frames from a video file as PIL images."""
    video_path = Path(video_path)
    reader = imageio.get_reader(video_path)
    meta = reader.get_meta_data()
    total = meta.get("n_frames") or reader.count_frames()
    indices = [int(i * (total - 1) / max(n - 1, 1)) for i in range(n)]
    frames = [Image.fromarray(reader.get_data(idx)) for idx in indices]
    reader.close()
    return frames


if __name__ == "__main__":
    out = generate_push_video("output/push_video.mp4")
    print(f"Synthetic push video written to {out}")
