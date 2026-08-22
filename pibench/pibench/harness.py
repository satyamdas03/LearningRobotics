"""High-level model-agnostic evaluation harness for PIBench.

``EvaluationHarness`` wraps a :class:`pibench.core.runner.Runner` together with
result serialization and leaderboard generation so that new predictors can be
evaluated with a single call.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from pibench.core.evaluator import Evaluator
from pibench.core.runner import Runner, RunResult
from pibench.core.suite import Suite
from pibench.evaluation.leaderboard import Leaderboard, build_leaderboard, write_leaderboard
from pibench.predictors.base import Predictor


def _serialize(obj: Any) -> Any:
    """Recursively turn Pydantic models and other objects into plain JSON types."""
    if isinstance(obj, BaseModel):
        return _serialize(obj.model_dump())
    if isinstance(obj, list):
        return [_serialize(item) for item in obj]
    if isinstance(obj, dict):
        return {str(k): _serialize(v) for k, v in obj.items()}
    return obj


class EvaluationHarness:
    """Run a predictor across suites, persist results, and produce metrics."""

    def __init__(
        self,
        predictor: Predictor,
        output_dir: Path | str = "output",
    ) -> None:
        self.predictor = predictor
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, suites: list[Suite]) -> RunResult:
        """Run the predictor and return the raw ``RunResult``."""
        runner = Runner(self.predictor)
        return runner.run(suites)

    def evaluate(self, suites: list[Suite]) -> dict[str, Any]:
        """Run the predictor, save detailed results, and return metrics."""
        result = self.run(suites)

        result_path = self.output_dir / f"results_{self.predictor.name}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)

        metrics = Evaluator.metrics(result)
        metrics_path = self.output_dir / f"metrics_{self.predictor.name}.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(_serialize(metrics), f, indent=2)

        return metrics

    @staticmethod
    def build_leaderboard(
        output_dir: Path | str = "output",
        results_glob: str = "results_*.json",
    ) -> Leaderboard:
        """Build a leaderboard from result files in ``output_dir``."""
        output_dir = Path(output_dir)
        return build_leaderboard(
            result_paths=sorted(output_dir.glob(results_glob)),
        )

    @staticmethod
    def write_leaderboard(
        leaderboard: Leaderboard,
        output_dir: Path | str = "output",
    ) -> tuple[Path, Path]:
        """Write ``leaderboard.json`` and ``leaderboard.html``."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        return write_leaderboard(
            leaderboard,
            json_path=output_dir / "leaderboard.json",
            html_path=output_dir / "leaderboard.html",
        )
