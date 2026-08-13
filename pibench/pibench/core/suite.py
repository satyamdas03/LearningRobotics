"""Suite definition for PIBench."""
from __future__ import annotations

from pydantic import BaseModel

from pibench.core.problem import Problem
from pibench.core.registry import list_problems


class Suite:
    """A collection of problem classes that can be instantiated with seeds."""

    def __init__(self, name: str, seed: int = 0, n_instances: int = 10) -> None:
        self.name = name
        self.seed = seed
        self.n_instances = n_instances
        self.classes = list_problems(name)
        if not self.classes:
            raise ValueError(f"No problems registered for suite '{name}'")

    def problems(self) -> list[Problem]:
        """Instantiate all problems in the suite with deterministic seeds."""
        instances: list[Problem] = []
        for idx, cls in enumerate(self.classes):
            for instance_idx in range(self.n_instances):
                seed = self.seed + idx * 997 + instance_idx
                instances.append(cls(seed=seed))
        return instances

    def __len__(self) -> int:
        return len(self.classes) * self.n_instances

    def __repr__(self) -> str:
        return f"Suite(name={self.name!r}, n_problems={len(self)})"


class SuiteSummary(BaseModel):
    name: str
    n_problems: int
    problem_names: list[str]
