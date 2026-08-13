"""Core engine tests."""
import numpy as np

from pibench.core.evaluator import Evaluator
from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import list_problems, list_suites
from pibench.core.runner import Runner
from pibench.core.suite import Suite
from pibench.predictors.physics_oracle import PhysicsOraclePredictor
from pibench.predictors.random_predictor import RandomPredictor
from pibench.scenes.statics import TowerFall


def test_registry_populated():
    assert "statics" in list_suites()
    names = [cls.__name__ for cls in list_problems("statics")]
    assert "TowerFall" in names


def test_tower_fall_question():
    problem = TowerFall(seed=0)
    q = problem.question()
    assert q.answer_type == AnswerType.CHOICE
    assert set(q.choices) == {"A", "B", "both", "neither"}


def test_tower_fall_ground_truth_runs():
    problem = TowerFall(seed=0)
    gt = problem.ground_truth()
    assert gt.answer in {"A", "B", "both", "neither"}
    assert "tilt_angle_deg" in gt.latent_params


def test_physics_oracle_perfect():
    problem = TowerFall(seed=1)
    oracle = PhysicsOraclePredictor()
    pred = oracle.predict(problem)
    assert problem.score(pred) == 1.0


def test_suite_instantiates():
    suite = Suite("statics", seed=0, n_instances=3)
    problems = suite.problems()
    assert len(problems) == 3
    assert all(isinstance(p, TowerFall) for p in problems)


def test_runner_and_evaluator():
    suite = Suite("statics", seed=0, n_instances=5)
    runner = Runner(PhysicsOraclePredictor())
    result = runner.run([suite])
    metrics = Evaluator.metrics(result)
    assert metrics["overall_accuracy"] == 1.0
    assert result.predictor == "physics_oracle"


def test_random_baseline_scores_less_than_oracle():
    suite = Suite("statics", seed=0, n_instances=20)
    oracle_result = Runner(PhysicsOraclePredictor()).run([suite])
    random_result = Runner(RandomPredictor(seed=7)).run([suite])
    oracle_acc = Evaluator.metrics(oracle_result)["overall_accuracy"]
    random_acc = Evaluator.metrics(random_result)["overall_accuracy"]
    assert oracle_acc == 1.0
    assert random_acc < 1.0
