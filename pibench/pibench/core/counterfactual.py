"""Counterfactual simulation support for PIBench.

A counterfactual takes an existing Problem instance, rebuilds the same scene from
its seed, overrides one or more named latent parameters, and returns the new
problem instance. This lets predictors and evaluators answer "what if?" queries
such as "if this object had double mass, would the tower still stand?".
"""
from __future__ import annotations

from typing import Any

from pibench.core.problem import Problem


class CounterfactualBuilder:
    """Clone a problem by seed and override parameters before scene construction."""

    def __init__(self, problem: Problem) -> None:
        self._cls = problem.__class__
        self._seed = problem.seed
        self._overrides: dict[str, Any] = {}

    def with_param(self, name: str, value: Any) -> "CounterfactualBuilder":
        """Request a single parameter override; chainable."""
        self._overrides[name] = value
        return self

    def run(self) -> Problem:
        """Build and return the counterfactual problem instance.

        We instantiate the class without calling its normal constructor, then
        apply the overrides before ``_build_scene()`` runs. This avoids
        deep-copying MuJoCo ``MjModel`` / ``MjData`` objects, which are not
        trivially copyable.
        """
        instance = self._cls.__new__(self._cls)
        instance.seed = self._seed
        instance._scene_built = False

        # Apply overrides.
        supported = self._cls._counterfactual_params(instance)
        for name, value in self._overrides.items():
            if name not in supported:
                raise ValueError(
                    f"Parameter '{name}' is not supported for counterfactuals by "
                    f"{self._cls.__name__}. Supported: {supported}"
                )
            setattr(instance, name, value)

        instance._build_scene()
        instance._scene_built = True
        return instance


def counterfactual(problem: Problem, **overrides: Any) -> Problem:
    """Convenience function: build a counterfactual with the given overrides."""
    builder = CounterfactualBuilder(problem)
    for name, value in overrides.items():
        builder.with_param(name, value)
    return builder.run()
