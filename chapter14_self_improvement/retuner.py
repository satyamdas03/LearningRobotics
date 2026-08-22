"""Chapter 14 — Adaptive controller retuning.

Given a ``MismatchEstimate`` from ``OnlineSystemID``, produce an updated
controller (or controller parameters) that compensates for the observed
mismatch.  This is intentionally simple: feedforward torque offset and a gear-
ratio gain scaling.  Future extensions can optimize gains with random search,
Bayesian optimization, or a learned residual policy.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class Retuning:
    """Output of the retuner: a callable controller + its parameters."""

    controller: Any
    params: dict[str, Any]


class Retuner:
    """Build a retuned controller from a mismatch estimate.

    Parameters
    ----------
    base_controller_class
        Class of the controller to retune (e.g. ``JointSpacePIDController``).
    base_controller_kwargs
        Keyword arguments used to instantiate the nominal controller.
    """

    def __init__(
        self,
        base_controller_class: type,
        **base_controller_kwargs: Any,
    ) -> None:
        self.controller_class = base_controller_class
        self.base_kwargs = base_controller_kwargs

    def retune(
        self,
        estimate: Any,
        *,
        gear_compensation_gain: float = 0.5,
        offset_gain: float = 1.0,
        min_gear: float = 0.7,
        max_gear: float = 1.3,
        disable_offset: bool = False,
    ) -> Retuning:
        """Return a controller that applies learned compensations.

        The returned controller wraps the base controller and:

        1. Subtracts ``estimate.torque_offset`` as feedforward (the offset is
           the missing torque observed in the residuals, so we cancel it).
        2. Gently scales torque to compensate for estimated gear perturbation,
           but only when the estimate is confident and within bounds.
        """
        torque_offset = np.asarray(getattr(estimate, "torque_offset", 0.0), dtype=float)
        raw_gear_ratio = np.asarray(getattr(estimate, "estimated_gear_ratio", 1.0), dtype=float)

        # Clamp to a physically plausible range and avoid division by zero.
        safe_gear = np.clip(np.where(np.abs(raw_gear_ratio) > 1e-6, raw_gear_ratio, 1.0), min_gear, max_gear)

        # Blend between identity and inverse gear so we do not overcompensate.
        inv_gear = 1.0 + gear_compensation_gain * (1.0 / safe_gear - 1.0)

        base = self.controller_class(**self.base_kwargs)

        class CompensatedController:
            """Wraps the base controller with mismatch compensation."""

            def __init__(self, base_controller, offset, gear_inv, offset_gain):
                self.base = base_controller
                self.offset = offset
                self.gear_inv = gear_inv
                self.offset_gain = offset_gain

            def compute(self, q, qdot, q_des=None, qdot_des=None, **kwargs):
                tau = self.base.compute(q, qdot, q_des=q_des, qdot_des=qdot_des, **kwargs)
                offset = np.zeros_like(tau) if disable_offset else self.offset
                return (tau - self.offset_gain * offset) * self.gear_inv

            def reset(self):
                if hasattr(self.base, "reset"):
                    self.base.reset()

        compensated = CompensatedController(
            base,
            torque_offset,
            inv_gear,
            offset_gain,
        )

        return Retuning(
            controller=compensated,
            params={
                "torque_offset": torque_offset.tolist(),
                "estimated_gear_ratio": raw_gear_ratio.tolist(),
                "effective_gear_scale": inv_gear.tolist(),
                "gear_compensation_gain": gear_compensation_gain,
                "offset_gain": offset_gain,
                "disable_offset": disable_offset,
            },
        )
