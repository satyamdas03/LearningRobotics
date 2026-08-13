"""Global registry of PIBench problem classes grouped by suite."""
from __future__ import annotations

from typing import Callable, Type

from pibench.core.problem import Problem

_REGISTRY: dict[str, list[Type[Problem]]] = {}


def register_problem(suite: str) -> Callable[[Type[Problem]], Type[Problem]]:
    """Decorator that registers a Problem subclass under a suite.

    Example::

        @register_problem("statics")
        class TowerFall(Problem):
            ...
    """

    def _decorator(cls: Type[Problem]) -> Type[Problem]:
        if not issubclass(cls, Problem):
            raise TypeError(f"{cls.__name__} must inherit from Problem")
        _REGISTRY.setdefault(suite, [])
        if cls not in _REGISTRY[suite]:
            _REGISTRY[suite].append(cls)
        cls._suite = suite  # type: ignore[attr-defined]
        return cls

    return _decorator


def list_suites() -> list[str]:
    """Return all registered suite names."""
    return sorted(_REGISTRY.keys())


def list_problems(suite: str | None = None) -> list[Type[Problem]]:
    """Return problem classes, optionally filtered by suite."""
    if suite is None:
        return [cls for classes in _REGISTRY.values() for cls in classes]
    return list(_REGISTRY.get(suite, []))


def get_suite_of(cls: Type[Problem]) -> str | None:
    """Return the suite name a Problem class was registered under."""
    return getattr(cls, "_suite", None)
