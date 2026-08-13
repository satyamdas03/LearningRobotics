"""PIBench — Physical Intuition Benchmark.

A lightweight MuJoCo-based benchmark for evaluating physical common-sense
reasoning in models, policies, and humans.
"""

__version__ = "0.1.0"
__all__ = ["Problem", "Suite", "Runner", "Evaluator", "Predictor", "Prediction"]

from pibench.core.problem import Problem, Question, AnswerType, GroundTruth, Prediction
from pibench.core.suite import Suite
from pibench.core.runner import Runner
from pibench.core.evaluator import Evaluator
from pibench.predictors.base import Predictor

# Import scene packages so they self-register with the global suite registry.
from pibench import scenes  # noqa: F401
