"""Abstract and mock real-arm interfaces for Chapter 7 control.

The goal is to write control laws against ``RealArm`` so the same code can later
run on a Forte/AM-ARM without rewriting the controller.  ``MockRealArm`` is a
software-only stand-in that can inject realistic actuator and sensor dynamics
so controllers can be hardened before touching real hardware.
"""
from __future__ import annotations

import copy
import random
from abc import ABC, abstractmethod
from collections import deque
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

    In addition to the basic ``RealArm`` contract, this class can simulate a
    wide range of real-world imperfections:

    * **Control modes**: ``torque`` (raw), ``velocity`` (joint-level velocity
      servo) or ``position`` (joint-level position servo).
    * **Gear ratio**: scales commanded torque before it reaches the joints.
    * **Friction**: extra Coulomb + viscous friction beyond MuJoCo joint
      damping.
    * **Actuator lag**: first-order low-pass filter on the torque command.
    * **Command delay**: torque commands sit in a FIFO before being applied.
    * **Sensor noise / bias / drift / quantization** on position and velocity.
    * **Feedback delay**: the state returned by ``get_state`` is from N steps
      in the past.
    * **Limits**: independent torque, velocity and position saturation.

    All parameters can be set to zero / ``None`` so that the default instance
    behaves like the original unmodified MuJoCo model, keeping existing tests
    unchanged.
    """

    VALID_MODES = {"torque", "velocity", "position"}

    def __init__(
        self,
        xml_path: str | None = None,
        dt: float = 0.01,
        control_mode: str = "torque",
        gear_ratio: float | np.ndarray = 1.0,
        torque_noise_std: float = 0.0,
        velocity_noise_std: float = 0.0,
        position_noise_std: float = 0.0,
        position_bias: np.ndarray | float | None = None,
        position_drift_rate: float = 0.0,
        quantization_resolution: float | None = None,
        feedback_delay_steps: int = 0,
        torque_delay_steps: int = 0,
        torque_lag_time_constant: float = 0.0,
        coulomb_friction: np.ndarray | float | None = None,
        viscous_friction: np.ndarray | float | None = None,
        torque_limits: np.ndarray | float | None = None,
        velocity_limits: np.ndarray | float | None = None,
        position_limits: np.ndarray | float | None = None,
        internal_kp: np.ndarray | float = 20.0,
        internal_kd: np.ndarray | float = 10.0,
        fixed_random_seed: int | None = None,
    ) -> None:
        if xml_path is None:
            xml_path = str(
                Path(__file__).parent.parent
                / "chapter01_foundation"
                / "simple_6dof_arm.xml"
            )
        with open(xml_path, "r", encoding="utf-8") as f:
            self._xml_string = f.read()
        self.model = mujoco.MjModel.from_xml_string(self._xml_string)
        self.data = mujoco.MjData(self.model)
        mujoco.mj_resetData(self.model, self.data)

        if control_mode not in self.VALID_MODES:
            raise ValueError(f"control_mode must be one of {self.VALID_MODES}")
        self.control_mode = control_mode
        self.dt = float(dt)

        self._rng = np.random.default_rng(fixed_random_seed)
        self._fixed_random_seed = fixed_random_seed

        self.nq_ = int(self.model.nq)
        self.nu_ = int(self.model.nu)

        # Actuator scaling.
        self.gear_ratio = self._broadcast(gear_ratio, "gear_ratio", positive=True)

        # Sensor imperfections.
        self.torque_noise_std = float(torque_noise_std)
        self.velocity_noise_std = float(velocity_noise_std)
        self.position_noise_std = float(position_noise_std)
        self.position_bias = self._broadcast(position_bias, "position_bias", default=0.0)
        self.position_drift_rate = float(position_drift_rate)
        self.quantization_resolution = (
            float(quantization_resolution) if quantization_resolution is not None else None
        )

        # Delays.
        self.torque_delay_steps = max(0, int(torque_delay_steps))
        self.feedback_delay_steps = max(0, int(feedback_delay_steps))

        # Actuator lag (first-order, continuous-time time constant).
        self.torque_lag_time_constant = float(torque_lag_time_constant)
        self._filtered_torque = np.zeros(self.nu_)

        # Friction.
        self.coulomb_friction = self._broadcast(coulomb_friction, "coulomb_friction", default=0.0)
        self.viscous_friction = self._broadcast(viscous_friction, "viscous_friction", default=0.0)

        # Saturation limits.
        self.torque_limits = self._broadcast(
            torque_limits,
            "torque_limits",
            default=self.model.actuator_ctrlrange[:, 1].copy(),
            positive=True,
        )
        self.velocity_limits = self._broadcast(
            velocity_limits, "velocity_limits", default=None
        )
        self.position_limits = self._broadcast(
            position_limits, "position_limits", default=None
        )

        # Internal PD gains for velocity/position control modes.
        # Default internal servos are intentionally conservative so that
        # position/velocity modes are stable on the light wrist joints.
        self.internal_kp = self._broadcast(internal_kp, "internal_kp", positive=True)
        self.internal_kd = self._broadcast(internal_kd, "internal_kd", positive=True)

        # Buffers.  Pre-fill the torque-delay buffer with zeros so commands do not
        # reach the physics until ``torque_delay_steps`` have elapsed.
        self._torque_delay_buffer: deque[np.ndarray] = deque(
            [np.zeros(self.nu_) for _ in range(self.torque_delay_steps + 1)]
        )
        self._state_feedback_buffer: deque[ArmState] = deque()

        # Internal target when running in velocity/position mode.  ``send_torques``
        # receives the target in these modes.
        self._current_target = np.zeros(self.nu_)

    # ------------------------------------------------------------------ helpers

    def _broadcast(
        self,
        value: float | np.ndarray | None,
        name: str,
        *,
        default: float | np.ndarray | None = None,
        positive: bool = False,
    ) -> np.ndarray:
        """Broadcast a scalar/array parameter to ``(nq,)``."""
        if value is None:
            value = default
        if value is None:
            return None  # type: ignore[return-value]
        arr = np.asarray(value, dtype=float)
        if arr.shape in ((), (1,)):
            arr = np.full(self.nq_, float(arr))
        elif arr.shape != (self.nq_,):
            raise ValueError(
                f"{name} must be scalar or shape {(self.nq_,)}, got {arr.shape}"
            )
        if positive and np.any(arr <= 0.0):
            raise ValueError(f"{name} must be positive, got {arr}")
        return arr

    def _quantize(self, value: np.ndarray) -> np.ndarray:
        """Apply uniform quantization if requested."""
        if self.quantization_resolution is None or self.quantization_resolution <= 0.0:
            return value
        return np.round(value / self.quantization_resolution) * self.quantization_resolution

    def _add_sensor_noise(self, q: np.ndarray, qdot: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Inject noise, bias, drift and quantization into sensor readings."""
        # Random-walk drift on the position bias.
        if self.position_drift_rate > 0.0:
            self.position_bias += self._rng.normal(
                0.0, self.position_drift_rate, size=self.position_bias.shape
            ) * self.dt

        q_noisy = q.copy()
        qdot_noisy = qdot.copy()

        if self.position_noise_std > 0.0:
            q_noisy += self._rng.normal(0.0, self.position_noise_std, size=q.shape)
        if self.position_bias is not None:
            q_noisy += self.position_bias
        q_noisy = self._quantize(q_noisy)

        if self.velocity_noise_std > 0.0:
            qdot_noisy += self._rng.normal(0.0, self.velocity_noise_std, size=qdot.shape)
        qdot_noisy = self._quantize(qdot_noisy)

        return q_noisy, qdot_noisy

    def _compute_internal_torque(self, q: np.ndarray, qdot: np.ndarray) -> np.ndarray:
        """Convert a velocity/position setpoint into joint torque."""
        if self.control_mode == "torque":
            return self._current_target.copy()

        if self.control_mode == "velocity":
            qdot_des = self._current_target
            # Cancel the model's joint damping so the P controller only has to
            # provide acceleration / compensation for coupling.
            damping = self.model.dof_damping.copy()
            return self.internal_kp * (qdot_des - qdot) + damping * qdot_des

        # position mode
        q_des = self._current_target
        return self.internal_kp * (q_des - q) - self.internal_kd * qdot

    def _apply_actuator_dynamics(self, tau: np.ndarray) -> np.ndarray:
        """Apply delay, lag, gear ratio and friction to the torque command."""
        # Gear ratio scales the motor-side command to joint-side torque.
        tau = tau * self.gear_ratio

        # Optional first-order lag (exponential smoothing).
        if self.torque_lag_time_constant > 0.0:
            alpha = self.dt / (self.torque_lag_time_constant + self.dt)
            self._filtered_torque = (
                alpha * tau + (1.0 - alpha) * self._filtered_torque
            )
            tau = self._filtered_torque.copy()

        # Command delay FIFO.
        self._torque_delay_buffer.append(tau.copy())
        while len(self._torque_delay_buffer) > self.torque_delay_steps + 1:
            self._torque_delay_buffer.popleft()
        delayed_tau = self._torque_delay_buffer[0]

        # Add noise before saturation so the noise is also bounded naturally.
        if self.torque_noise_std > 0.0:
            delayed_tau = delayed_tau + self._rng.normal(
                0.0, self.torque_noise_std, size=delayed_tau.shape
            )

        # Torque saturation.
        delayed_tau = np.clip(delayed_tau, -self.torque_limits, self.torque_limits)

        # Friction is a disturbance applied in addition to the controller torque.
        # Coulomb friction is regularized with tanh so the discontinuity at
        # zero velocity does not cause numerical chattering on light joints.
        qdot = self.data.qvel.copy()
        v_reg = 0.001  # velocity half-band for Coulomb transition
        coulomb_force = self.coulomb_friction * np.tanh(qdot / v_reg)
        friction = coulomb_force + self.viscous_friction * qdot
        delayed_tau = delayed_tau - friction

        return delayed_tau

    # ------------------------------------------------------------------ public API

    def nq(self) -> int:
        return self.nq_

    def get_state(self) -> ArmState:
        q = self.data.qpos.copy()
        qdot = self.data.qvel.copy()

        q_noisy, qdot_noisy = self._add_sensor_noise(q, qdot)

        state = ArmState(
            q=q_noisy,
            qdot=qdot_noisy,
            tau=self.data.ctrl.copy(),
            timestamp=self.data.time,
        )

        # Feedback delay: return the oldest buffered reading.
        self._state_feedback_buffer.append(state)
        while len(self._state_feedback_buffer) > self.feedback_delay_steps + 1:
            self._state_feedback_buffer.popleft()
        return self._state_feedback_buffer[0]

    def send_torques(self, tau: np.ndarray, dt: float | None = None) -> None:
        tau = np.asarray(tau, dtype=float)
        if tau.shape != (self.nu_,):
            raise ValueError(f"Expected tau shape {(self.nu_,)}, got {tau.shape}")

        step_dt = float(dt) if dt is not None else self.dt
        n_steps = max(1, int(round(step_dt / self.model.opt.timestep)))

        self._current_target = tau.copy()

        for _ in range(n_steps):
            # Read perfect state for the internal controller / friction calc.
            q = self.data.qpos.copy()
            qdot = self.data.qvel.copy()

            # Velocity / position saturation before the controller sees the state.
            if self.velocity_limits is not None:
                qdot = np.clip(qdot, -self.velocity_limits, self.velocity_limits)
                self.data.qvel[:] = qdot
            if self.position_limits is not None:
                q = np.clip(q, -self.position_limits, self.position_limits)
                self.data.qpos[:] = q

            # Compute torque according to control mode.
            tau_joint = self._compute_internal_torque(q, qdot)
            tau_joint = self._apply_actuator_dynamics(tau_joint)

            self.data.ctrl[:] = tau_joint
            mujoco.mj_step(self.model, self.data)

    def close(self) -> None:
        pass

    # ------------------------------------------------------------------ introspection / reset

    def reset_state(self, q: np.ndarray | None = None, qdot: np.ndarray | None = None) -> None:
        """Reset the arm to a given state and clear dynamics buffers."""
        mujoco.mj_resetData(self.model, self.data)
        if q is not None:
            self.data.qpos[:] = np.asarray(q, dtype=float)
        if qdot is not None:
            self.data.qvel[:] = np.asarray(qdot, dtype=float)
        self._filtered_torque = np.zeros(self.nu_)
        self._torque_delay_buffer.clear()
        self._torque_delay_buffer.extend(
            [np.zeros(self.nu_) for _ in range(self.torque_delay_steps + 1)]
        )
        self._state_feedback_buffer.clear()
        self.position_bias = np.zeros(self.nq_) if self.position_bias is not None else None


class VirtualArmFactory:
    """Generate domain-randomized ``MockRealArm`` instances.

    The factory draws each imperfection parameter from a bounded distribution.
    Controllers that are trained/tuned on instances produced by this factory
    learn to tolerate the sim-to-real gap without needing access to physical
    hardware.
    """

    def __init__(
        self,
        xml_path: str | None = None,
        base_dt: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.xml_path = xml_path
        self.base_dt = base_dt
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

    def _sample_log_uniform(self, low: float, high: float) -> float:
        """Sample a positive scalar on a log-uniform grid."""
        log_low, log_high = np.log(low), np.log(high)
        return float(np.exp(self._np_rng.uniform(log_low, log_high)))

    def _sample_array(
        self,
        nq: int,
        low: float,
        high: float,
        relative: float | None = None,
        per_joint_variation: float = 0.0,
    ) -> np.ndarray:
        """Sample an (nq,) array.  If ``relative`` is set, base is ``relative``."""
        base = relative if relative is not None else self._np_rng.uniform(low, high)
        if per_joint_variation > 0.0:
            return base * (1.0 + self._np_rng.normal(0.0, per_joint_variation, size=nq))
        return np.full(nq, base)

    def create(
        self,
        nq: int | None = None,
        control_mode: str = "torque",
        randomize_gear: bool = True,
        randomize_friction: bool = True,
        randomize_sensors: bool = True,
        randomize_delays: bool = True,
        randomize_actuator_lag: bool = True,
        gear_range: tuple[float, float] = (0.8, 1.25),
        coulomb_friction_range: tuple[float, float] = (0.0, 2.0),
        viscous_friction_range: tuple[float, float] = (0.0, 0.5),
        position_noise_std_range: tuple[float, float] = (0.0, 0.005),
        velocity_noise_std_range: tuple[float, float] = (0.0, 0.02),
        position_bias_range: tuple[float, float] = (-0.02, 0.02),
        position_drift_rate_range: tuple[float, float] = (0.0, 0.001),
        quantization_resolution_range: tuple[float, float] = (0.0, 0.002),
        torque_delay_range: tuple[int, int] = (0, 3),
        feedback_delay_range: tuple[int, int] = (0, 5),
        torque_lag_range: tuple[float, float] = (0.0, 0.05),
        fixed_random_seed: int | None = None,
    ) -> MockRealArm:
        """Return a randomized virtual arm."""
        # Need nq to sample arrays; peek at the model if not provided.
        if nq is None:
            xml = self.xml_path or str(
                Path(__file__).parent.parent
                / "chapter01_foundation"
                / "simple_6dof_arm.xml"
            )
            with open(xml, "r", encoding="utf-8") as f:
                model = mujoco.MjModel.from_xml_string(f.read())
            nq = model.nq

        kwargs: dict = {
            "xml_path": self.xml_path,
            "dt": self.base_dt,
            "control_mode": control_mode,
            "fixed_random_seed": fixed_random_seed,
        }

        if randomize_gear:
            kwargs["gear_ratio"] = self._sample_array(nq, *gear_range)
        if randomize_friction:
            kwargs["coulomb_friction"] = self._sample_array(nq, *coulomb_friction_range)
            kwargs["viscous_friction"] = self._sample_array(nq, *viscous_friction_range)
        if randomize_sensors:
            kwargs["position_noise_std"] = self._np_rng.uniform(*position_noise_std_range)
            kwargs["velocity_noise_std"] = self._np_rng.uniform(*velocity_noise_std_range)
            kwargs["position_bias"] = self._np_rng.uniform(*position_bias_range, size=nq)
            kwargs["position_drift_rate"] = self._np_rng.uniform(*position_drift_rate_range)
            qres = self._np_rng.uniform(*quantization_resolution_range)
            kwargs["quantization_resolution"] = qres if qres > 1e-9 else None
        if randomize_delays:
            kwargs["torque_delay_steps"] = self._rng.randint(*torque_delay_range)
            kwargs["feedback_delay_steps"] = self._rng.randint(*feedback_delay_range)
        if randomize_actuator_lag:
            lag = self._np_rng.uniform(*torque_lag_range)
            kwargs["torque_lag_time_constant"] = lag if lag > 1e-6 else 0.0

        return MockRealArm(**kwargs)

    def create_batch(
        self,
        count: int,
        nq: int | None = None,
        **kwargs,
    ) -> list[MockRealArm]:
        """Create ``count`` independent randomized virtual arms."""
        return [self.create(nq=nq, **kwargs) for _ in range(count)]


class ForteAMArmAdapter(RealArm):
    """Placeholder adapter for the Forte / AM-ARM real robot.

    This stub defines the interface shape but does not implement SDK calls.
    When hardware is available, fill in ``__init__``, ``get_state`` and
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
