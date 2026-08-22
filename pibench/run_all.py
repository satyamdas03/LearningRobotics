"""Convenience script: run every suite with every baseline predictor.

Also builds a static leaderboard from the saved results.
"""
from __future__ import annotations

from pathlib import Path

from pibench.core.registry import list_suites
from pibench.core.suite import Suite
from pibench.harness import EvaluationHarness
from pibench.predictors.physics_oracle import PhysicsOraclePredictor
from pibench.predictors.random_predictor import RandomPredictor


def main() -> int:
    suites = [Suite(name, seed=0, n_instances=10) for name in list_suites()]
    predictors = [PhysicsOraclePredictor(), RandomPredictor(seed=7)]

    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    for predictor in predictors:
        harness = EvaluationHarness(predictor, output_dir=output_dir)
        metrics = harness.evaluate(suites)
        print(
            f"{metrics['predictor']:20s}: {metrics['overall_accuracy']:.1%} "
            f"({metrics['n_correct']:.0f}/{metrics['n_total']})"
        )

    leaderboard = EvaluationHarness.build_leaderboard(output_dir=output_dir)
    json_path, html_path = EvaluationHarness.write_leaderboard(leaderboard, output_dir=output_dir)
    print(f"\nLeaderboard: {json_path}")
    print(f"HTML leaderboard: {html_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
