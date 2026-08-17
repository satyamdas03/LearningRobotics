"""PIBench scene collection."""
# Import suites so the registry is populated.
from pibench.scenes import contact, dynamics, statics

__all__ = ["statics", "dynamics", "contact"]
