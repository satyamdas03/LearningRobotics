"""Convenience script: run every suite with every baseline predictor."""
from __future__ import annotations

import json
from pathlib import Path

from pibench.core.evaluator import Evaluator
from pibench.core.registry import list_suites
from pibench.core.runner import Runner
from pibench.core.suite import Suite
from pibench.predictors.physics_oracle import PhysicsOraclePredictor
from pibench.predictors.random_predictor import RandomPredictor


def main() -> int:
    suites = [Suite(name, seed=0, n_instances=10) for name in list_suites()]
    predictors = [PhysicsOraclePredictor(), RandomPredictor(seed=7)]

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_metrics = []
    for predictor in predictors:
        runner = Runner(predictor)
        result = runner.run(suites)
        metrics = Evaluator.metrics(result)
        all_metrics.append(metrics)

        result_path = output_dir / f"results_{predictor.name}.json"
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result.model_dump(), f, indent=2)
        print(f"Wrote {result_path}")

    summary_path = output_dir / "summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"Wrote {summary_path}")

    print("\nSummary:")
    for m in all_metrics:
        print(f"  {m['predictor']:20s}: {m['overall_accuracy']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
