"""Chapter 11 — Behavior cloning baseline for imitation learning.

A tiny NumPy-only MLP is trained to map ``(current_joint_angles, goal_ee_pos)``
to the expert's next-step joint command.  Keeping the implementation dependency-free
lets the project stay on MuJoCo + NumPy while still demonstrating the core
supervised-learning idea of behavior cloning.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def _relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(float)


@dataclass
class BCConfig:
    """Hyperparameters for the behavior-cloning MLP."""

    input_dim: int = 9  # 6 joints + 3 goal position
    hidden_dim: int = 64
    output_dim: int = 6  # predicted next joint positions
    learning_rate: float = 1e-3
    epochs: int = 400
    batch_size: int | None = None  # None = full-batch
    seed: int = 0


class BCPolicy:
    """MLP policy trained with MSE on expert (state, goal) -> action pairs."""

    def __init__(self, config: BCConfig | None = None) -> None:
        self.config = config or BCConfig()
        cfg = self.config
        rng = np.random.default_rng(cfg.seed)

        # Heuristic Xavier-like initialization.
        self.W1 = rng.standard_normal((cfg.input_dim, cfg.hidden_dim)) * np.sqrt(
            2.0 / cfg.input_dim
        )
        self.b1 = np.zeros(cfg.hidden_dim)
        self.W2 = rng.standard_normal((cfg.hidden_dim, cfg.output_dim)) * np.sqrt(
            2.0 / cfg.hidden_dim
        )
        self.b2 = np.zeros(cfg.output_dim)

        # Adam state.
        self._m: dict[str, np.ndarray] = {}
        self._v: dict[str, np.ndarray] = {}
        self._t = 0

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Forward pass returning the predicted action."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        z1 = x @ self.W1 + self.b1
        a1 = _relu(z1)
        return a1 @ self.W2 + self.b2

    def _forward_cache(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        z1 = x @ self.W1 + self.b1
        a1 = _relu(z1)
        z2 = a1 @ self.W2 + self.b2
        return z1, a1, z2

    def _adam_step(self, name: str, grad: np.ndarray, lr: float, b1: float, b2: float, eps: float) -> None:
        param = getattr(self, name)
        if name not in self._m:
            self._m[name] = np.zeros_like(param)
            self._v[name] = np.zeros_like(param)
        self._m[name] = b1 * self._m[name] + (1 - b1) * grad
        self._v[name] = b2 * self._v[name] + (1 - b2) * (grad * grad)
        m_hat = self._m[name] / (1 - b1**self._t)
        v_hat = self._v[name] / (1 - b2**self._t)
        setattr(self, name, param - lr * m_hat / (np.sqrt(v_hat) + eps))

    def fit(self, X: np.ndarray, Y: np.ndarray, verbose: bool = False) -> list[float]:
        """Train the MLP with Adam on the provided (state+goal, action) data.

        Returns the per-epoch mean-squared-error history.
        """
        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)
        cfg = self.config
        lr = cfg.learning_rate
        history: list[float] = []

        n = X.shape[0]
        batch_size = n if cfg.batch_size is None else min(cfg.batch_size, n)
        n_batches = max(1, n // batch_size)

        rng = np.random.default_rng(cfg.seed + 1)

        for epoch in range(cfg.epochs):
            # Shuffle each epoch.
            perm = rng.permutation(n)
            X_shuffled = X[perm]
            Y_shuffled = Y[perm]

            epoch_loss = 0.0
            for b in range(n_batches):
                start = b * batch_size
                end = start + batch_size
                xb = X_shuffled[start:end]
                yb = Y_shuffled[start:end]

                self._t += 1
                z1, a1, pred = self._forward_cache(xb)
                error = pred - yb
                loss = float(np.mean(error**2))
                epoch_loss += loss

                # Backprop.
                d_z2 = (2.0 / xb.shape[0]) * error
                d_W2 = a1.T @ d_z2
                d_b2 = np.sum(d_z2, axis=0)
                d_a1 = d_z2 @ self.W2.T
                d_z1 = d_a1 * _relu_derivative(z1)
                d_W1 = xb.T @ d_z1
                d_b1 = np.sum(d_z1, axis=0)

                for name, grad in [
                    ("W1", d_W1),
                    ("b1", d_b1),
                    ("W2", d_W2),
                    ("b2", d_b2),
                ]:
                    self._adam_step(name, grad, lr, b1=0.9, b2=0.999, eps=1e-8)

            history.append(epoch_loss / n_batches)
            if verbose and (epoch + 1) % 50 == 0:
                print(f"Epoch {epoch + 1}/{cfg.epochs}, MSE: {history[-1]:.6f}")

        return history

    def save(self, path: str | Path) -> None:
        """Persist policy weights and config to an NPZ file."""
        path = Path(path)
        np.savez(
            path,
            W1=self.W1,
            b1=self.b1,
            W2=self.W2,
            b2=self.b2,
            input_dim=self.config.input_dim,
            hidden_dim=self.config.hidden_dim,
            output_dim=self.config.output_dim,
            learning_rate=self.config.learning_rate,
            epochs=self.config.epochs,
            seed=self.config.seed,
        )

    @classmethod
    def load(cls, path: str | Path) -> "BCPolicy":
        """Load a saved policy."""
        path = Path(path)
        data = np.load(path, allow_pickle=False)
        cfg = BCConfig(
            input_dim=int(data["input_dim"]),
            hidden_dim=int(data["hidden_dim"]),
            output_dim=int(data["output_dim"]),
            learning_rate=float(data["learning_rate"]),
            epochs=int(data["epochs"]),
            seed=int(data["seed"]),
        )
        policy = cls(config=cfg)
        policy.W1 = data["W1"]
        policy.b1 = data["b1"]
        policy.W2 = data["W2"]
        policy.b2 = data["b2"]
        return policy


def prepare_bc_dataset(demos: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert a list of expert demonstrations into (X, Y) training arrays.

    Each sample is:
      X = [current_joint_angles, goal_ee_position]
      Y = expert_path_residual = next_q - q

    This preserves the original trajectory shape.  For a stable one-shot
    reaching baseline see ``prepare_ik_dataset``.
    """
    Xs: list[np.ndarray] = []
    Ys: list[np.ndarray] = []
    for demo in demos:
        states = np.asarray(demo["states"])
        next_states = np.asarray(demo["next_states"])
        goal = np.asarray(demo["goal_ee"])
        n = states.shape[0]
        goal_repeated = np.tile(goal, (n, 1))
        Xs.append(np.concatenate([states, goal_repeated], axis=1))
        Ys.append(next_states - states)
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def prepare_goal_residual_dataset(demos: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert demonstrations into (state+goal_ee) -> residual-to-goal-angles data.

    Each sample is:
      X = [current_joint_angles, goal_ee_position]
      Y = goal_q - current_q
    """
    Xs: list[np.ndarray] = []
    Ys: list[np.ndarray] = []
    for demo in demos:
        states = np.asarray(demo["states"])
        goal_q = np.asarray(demo["goal_q"])
        goal_ee = np.asarray(demo["goal_ee"])
        n = states.shape[0]
        goal_ee_repeated = np.tile(goal_ee, (n, 1))
        Xs.append(np.concatenate([states, goal_ee_repeated], axis=1))
        Ys.append(np.tile(goal_q, (n, 1)) - states)
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def prepare_ik_dataset(demos: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    """Convert demonstrations into one-shot inverse-kinematics training data.

    Each sample is:
      X = goal_ee_position
      Y = goal_joint_configuration

    The learned policy is the simplest possible reaching baseline: given a
    desired end-effector location, output a joint configuration that reaches it.
    """
    Xs: list[np.ndarray] = []
    Ys: list[np.ndarray] = []
    for demo in demos:
        goal_ee = np.asarray(demo["goal_ee"])
        goal_q = np.asarray(demo["goal_q"])
        n = np.asarray(demo["states"]).shape[0]
        Xs.append(np.tile(goal_ee, (n, 1)))
        Ys.append(np.tile(goal_q, (n, 1)))
    return np.concatenate(Xs, axis=0), np.concatenate(Ys, axis=0)


def rollout_policy(
    xml_path: str | Path,
    policy: BCPolicy,
    start_q: np.ndarray,
    goal_ee_position: np.ndarray,
    duration: float = 2.0,
    dt: float = 0.01,
) -> dict[str, Any]:
    """Run a trained BC policy open-loop on the MuJoCo arm.

    At each timestep the policy receives ``(q_t, goal)`` and outputs the target
    joint command ``q_{t+1}``.
    """
    import mujoco

    xml_path = str(xml_path)
    start_q = np.asarray(start_q, dtype=float)
    goal_ee_position = np.asarray(goal_ee_position, dtype=float)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
        for i in range(1, 7)
    ]
    qpos_addrs = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=int)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")

    data.qpos[qpos_addrs] = start_q
    mujoco.mj_forward(model, data)

    n_steps = max(2, int(np.round(duration / dt)))
    states: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    ee_positions: list[np.ndarray] = []

    # For the Milestone 5 baseline we replay the learned residual displacements
    # kinematically.  The policy predicts ``q_next - q`` and we add it to the
    # current configuration.  This isolates imitation-learning generalization
    # from low-level actuator-tracking errors.
    for _ in range(n_steps):
        q = np.array(data.qpos[qpos_addrs], dtype=float)
        x = np.concatenate([q, goal_ee_position])
        delta = policy.predict(x).ravel()
        q_next = np.clip(q + delta, model.jnt_range[:, 0], model.jnt_range[:, 1])

        states.append(q)
        actions.append(delta)

        data.qpos[qpos_addrs] = q_next
        mujoco.mj_forward(model, data)

        ee_positions.append(data.site_xpos[ee_id].copy())

    return {
        "states": np.asarray(states),
        "actions": np.asarray(actions),
        "ee_positions": np.asarray(ee_positions),
        "goal_ee": goal_ee_position,
        "dt": float(dt),
    }


def rollout_goal_policy(
    xml_path: str | Path,
    policy: BCPolicy,
    goal_ee_position: np.ndarray,
) -> dict[str, Any]:
    """Execute a one-shot inverse-kinematics policy and return the resulting pose.

    The policy predicts a goal joint configuration directly from the desired
    end-effector location.  The arm is placed at that configuration
    kinematically so the test can verify the predicted configuration places the
    end effector near the target.
    """
    import mujoco

    xml_path = str(xml_path)
    goal_ee_position = np.asarray(goal_ee_position, dtype=float)

    model = mujoco.MjModel.from_xml_path(xml_path)
    data = mujoco.MjData(model)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
        for i in range(1, 7)
    ]
    qpos_addrs = np.array([model.jnt_qposadr[jid] for jid in joint_ids], dtype=int)
    ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "ee")

    x = goal_ee_position.reshape(1, -1)
    goal_q = policy.predict(x).ravel()
    goal_q = np.clip(goal_q, model.jnt_range[:, 0], model.jnt_range[:, 1])

    data.qpos[qpos_addrs] = goal_q
    mujoco.mj_forward(model, data)
    final_ee = data.site_xpos[ee_id].copy()

    return {
        "goal_q": goal_q,
        "ee_position": final_ee,
        "goal_ee": goal_ee_position,
    }
