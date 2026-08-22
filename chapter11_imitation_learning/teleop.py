"""Chapter 11 — Kinesthetic / keyboard teleoperation recorder.

This module records a human-provided joint-space trajectory.  The intended use is
to collect a few demonstration examples for imitation learning without writing
a full motion planner.  The recorder itself is input-agnostic: any source that
produces a 6-DOF joint vector at each timestep can feed it.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class TeleopSample:
    """One timestep of a teleoperated demonstration."""

    q: np.ndarray
    timestamp: float | None = None
    source: str | None = None  # e.g. "keyboard", "mouse", "planned"


class TeleopRecorder:
    """Accumulate joint-space samples and export them as a trajectory."""

    def __init__(self, joint_count: int = 6) -> None:
        self.joint_count = joint_count
        self.samples: list[TeleopSample] = []
        self.start_time: float | None = None

    def reset(self) -> None:
        """Clear all recorded samples."""
        self.samples.clear()
        self.start_time = None

    def record(self, q: np.ndarray, source: str | None = None) -> None:
        """Append a new joint configuration."""
        q = np.asarray(q, dtype=float)
        if q.shape != (self.joint_count,):
            raise ValueError(f"Expected shape ({self.joint_count},), got {q.shape}")

        if self.start_time is None:
            self.start_time = time.time()
        timestamp = time.time() - self.start_time
        self.samples.append(TeleopSample(q=q.copy(), timestamp=timestamp, source=source))

    def as_trajectory(self, dt: float = 0.01) -> dict[str, Any]:
        """Convert recorded samples into the same format as ``expert.py``."""
        if len(self.samples) < 2:
            raise ValueError("At least two samples are required to form a trajectory.")

        qs = np.stack([s.q for s in self.samples[:-1]])
        next_qs = np.stack([s.q for s in self.samples[1:]])
        # Fill a placeholder goal (last recorded configuration) so the format matches.
        goal_q = self.samples[-1].q
        timestamps = np.array([s.timestamp for s in self.samples[:-1]])
        if dt is None and len(timestamps) > 1:
            dt = float(np.mean(np.diff(timestamps)))
        return {
            "states": qs,
            "actions": next_qs,
            "next_states": next_qs,
            "timestamps": timestamps,
            "goal_q": goal_q,
            "dt": float(dt),
        }

    def save_json(self, path: str | Path) -> None:
        """Save samples to a JSON file."""
        path = Path(path)
        payload = {
            "joint_count": self.joint_count,
            "samples": [
                {
                    "q": s.q.tolist(),
                    "timestamp": s.timestamp,
                    "source": s.source,
                }
                for s in self.samples
            ],
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "TeleopRecorder":
        """Load samples from a JSON file."""
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        rec = cls(joint_count=payload.get("joint_count", 6))
        for item in payload["samples"]:
            rec.samples.append(
                TeleopSample(
                    q=np.asarray(item["q"], dtype=float),
                    timestamp=item.get("timestamp"),
                    source=item.get("source"),
                )
            )
        if rec.samples:
            rec.start_time = 0.0
        return rec


def keyboard_teleop_loop(
    xml_path: str | Path,
    recorder: TeleopRecorder,
    key_step: float = 0.05,
) -> None:
    """Open an interactive MuJoCo viewer and record keyboard-driven joint commands.

    Keys:
      - ``1``/``2``: decrease/increase joint 1
      - ``3``/``4``: decrease/increase joint 2
      - ``5``/``6``: decrease/increase joint 3
      - ``7``/``8``: decrease/increase joint 4
      - ``9``/``0``: decrease/increase joint 5
      - ``q``/``w``: decrease/increase joint 6
      - ``s``: save the recorded trajectory to ``teleop_demo.json``

    This is a convenience demo; the test suite exercises the recorder without
    opening a GUI.
    """
    import mujoco
    import mujoco.viewer

    xml_path = str(xml_path)
    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
        for i in range(1, 7)
    ]
    qpos_addrs = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=int)

    # Map keyboard keys to (joint_index, sign).
    key_map = {
        mujoco.mjtKeyboard.mjKEY_1: (0, -1),
        mujoco.mjtKeyboard.mjKEY_2: (0, +1),
        mujoco.mjtKeyboard.mjKEY_3: (1, -1),
        mujoco.mjtKeyboard.mjKEY_4: (1, +1),
        mujoco.mjtKeyboard.mjKEY_5: (2, -1),
        mujoco.mjtKeyboard.mjKEY_6: (2, +1),
        mujoco.mjtKeyboard.mjKEY_7: (3, -1),
        mujoco.mjtKeyboard.mjKEY_8: (3, +1),
        mujoco.mjtKeyboard.mjKEY_9: (4, -1),
        mujoco.mjtKeyboard.mjKEY_0: (4, +1),
        ord("q"): (5, -1),
        ord("w"): (5, +1),
        ord("s"): (-1, 0),  # save
    }

    q_target = np.zeros(6)

    def key_callback(keycode: int) -> None:
        nonlocal q_target
        if keycode in key_map:
            idx, sign = key_map[keycode]
            if idx < 0:
                recorder.save_json("teleop_demo.json")
                print("Saved teleop_demo.json")
                return
            q_target[idx] += sign * key_step
            lo = model.jnt_range[idx, 0]
            hi = model.jnt_range[idx, 1]
            q_target[idx] = float(np.clip(q_target[idx], lo, hi))

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            q = np.array(data.qpos[qpos_addrs], dtype=float)
            # Smoothly drive the arm toward the keyboard target.
            data.ctrl[:] = q_target
            recorder.record(q)
            mujoco.mj_step(model, data)
            viewer.sync()

    return recorder
