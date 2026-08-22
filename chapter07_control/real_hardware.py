"""Abstract and mock real-arm interfaces for Chapter 7 control.

The goal is to write control laws against ``RealArm`` so the same code can later
run on a Forte/AM-ARM without rewriting the controller.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


@dataclass
class ArmState:
    """Snapshot of a robot arm's state."""

    q: np.ndarray
    qdot: np.ndarray
    tau: np.ndarray | None = None
    timestamp: float = 0.0


class RealArm(ABC):
    """Abstract interface for a real robot arm."""

    @abstractmethod
    def get_state(self) -> ArmState:
        """Return the current arm state."""
        ...

    @abstractmethod
    def send_torques(self, tau: np.ndarray, dt: float | None = None) -> None:
        """Send joint torques and advance one control step."""
        ...

    @abstractmethod
    def nq(self) -> int:
        """Number of position DOF."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Release any hardware resources."""
        ...


class MockRealArm(RealArm):
    """Software-only arm that wraps a MuJoCo model.

    Can inject actuator noise, communication delay, and torque saturation so
    controllers can be tested under realistic imperfections before touching
    real hardware.
    """

    def __init__(
        self,
        xml_path: str | None = None,
        dt: float = 0.01,
        torque_noise_std: float = 0.0,
        velocity_noise_std: float = 0.0,
        torque_delay_steps: int = 0,
        torque_limits: np.ndarray | None = None,
    ) -> None:
        if xml_path is None:
            xml_path = str(
                Path(__file__).parent.parent
                / "chapter01_foundation"
                / "simple_6dof_arm.xml"
            )
        with open(xml_path, "r", encoding="utf-8") as f:
            self.model = mujoco.MjModel.from_xml_string(f.read())
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)
        self.dt = dt
        self.torque_noise_std = torque_noise_std
        self.velocity_noise_std = velocity_noise_std
        self.torque_delay_steps = max(0, int(torque_delay_steps))
        self.torque_limits = (
            np.asarray(torque_limits, dtype=float)
            if torque_limits is not None
            else self.model.actuator_ctrlrange[:, 1].copy()
        )
        self._delay_buffer: list[np.ndarray] = []

    def nq(self) -> int:
        return self.model.nq

    def get_state(self) -> ArmState:
        q = self.data.qpos.copy()
        qdot = self.data.qvel.copy()
        if self.velocity_noise_std > 0.0:
            qdot = qdot + np.random.normal(0.0, self.velocity_noise_std, size=qdot.shape)
        return ArmState(q=q, qdot=qdot, tau=self.data.ctrl.copy(), timestamp=self.data.time)

    def send_torques(self, tau: np.ndarray, dt: float | None = None) -> None:
        tau = np.asarray(tau, dtype=float)
        if tau.shape != (self.model.nu,):
            raise ValueError(f"Expected tau shape {(self.model.nu,)}, got {tau.shape}")

        # Saturate to actuator limits.
        tau = np.clip(tau, -self.torque_limits, self.torque_limits)

        # Add noise.
        if self.torque_noise_std > 0.0:
            tau = tau + np.random.normal(0.0, self.torque_noise_std, size=tau.shape)

        # Apply delay: store commands, send oldest if buffer full.
        self._delay_buffer.append(tau.copy())
        if len(self._delay_buffer) > self.torque_delay_steps + 1:
            self._delay_buffer.pop(0)
        applied = self._delay_buffer[0]

        self.data.ctrl[:] = applied
        step_dt = dt if dt is not None else self.dt
        n_steps = max(1, int(round(step_dt / self.model.opt.timestep)))
        for _ in range(n_steps):
            mujoco.mj_step(self.model, self.data)

    def close(self) -> None:
        pass


class ForteAMArmAdapter(RealArm):
    """Placeholder adapter for the Forte / AM-ARM real robot.

    This stub defines the interface shape but does not implement SDK calls.
    When hardware is available, fill in ``__init__``, ``get_state``, and
    ``send_torques`` without changing the controller code.
    """

    def __init__(self, *args, **kwargs) -> None:  # noqa: ARG002
        raise NotImplementedError(
            "Forte/AM-ARM SDK adapter is a stub. Implement once hardware is available."
        )

    def get_state(self) -> ArmState:
        raise NotImplementedError

    def send_torques(self, tau: np.ndarray, dt: float | None = None) -> None:
        raise NotImplementedError

    def nq(self) -> int:
        raise NotImplementedError

    def close(self) -> None:
        raise NotImplementedError
