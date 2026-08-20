"""PIBench predictor implementations."""
from pibench.predictors.base import Predictor
from pibench.predictors.random_predictor import RandomPredictor
from pibench.predictors.physics_oracle import PhysicsOraclePredictor

try:
    from pibench.predictors.llm_predictor import LLMPredictor
except ImportError:
    LLMPredictor = None  # type: ignore[misc, assignment]

__all__ = ["Predictor", "RandomPredictor", "PhysicsOraclePredictor", "LLMPredictor"]
