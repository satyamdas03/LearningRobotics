"""PIBench scene collection."""
# Import suites so the registry is populated.
from pibench.scenes import articulated, contact, deformable, dynamics, statics

__all__ = ["statics", "dynamics", "contact", "articulated", "deformable"]
