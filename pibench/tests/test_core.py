"""Core engine tests."""
from __future__ import annotations

import numpy as np

from pibench.core.evaluator import Evaluator
from pibench.core.problem import AnswerType, GroundTruth, Prediction, Problem, Question
from pibench.core.registry import list_problems, list_suites
from pibench.core.runner import Runner
from pibench.core.suite import Suite
from pibench.predictors.physics_oracle import PhysicsOraclePredictor
from pibench.predictors.random_predictor import RandomPredictor
from pibench.scenes.contact import (
    FrictionPile,
    PushTipVsSlide,
    SlipGrip,
    StackStability,
    WedgeInsert,
)
from pibench.scenes.dynamics import CollisionBounce, PendulumSwing, ProjectileHit
from pibench.scenes.statics import SlopeSlide, SupportBalance, ToppleDirection, TowerFall


STATICS_CLASSES = [TowerFall, SlopeSlide, SupportBalance, ToppleDirection]
DYNAMICS_CLASSES = [PendulumSwing, CollisionBounce, ProjectileHit]
CONTACT_CLASSES = [PushTipVsSlide, StackStability, WedgeInsert, FrictionPile, SlipGrip]
ALL_CLASSES = STATICS_CLASSES + DYNAMICS_CLASSES + CONTACT_CLASSES


def test_registry_populated():
    assert "statics" in list_suites()
    assert "dynamics" in list_suites()
    assert "contact" in list_suites()
    for suite, classes in [
        ("statics", STATICS_CLASSES),
        ("dynamics", DYNAMICS_CLASSES),
        ("contact", CONTACT_CLASSES),
    ]:
        names = [cls.__name__ for cls in list_problems(suite)]
        for cls in classes:
            assert cls.__name__ in names


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


def test_slope_slide_question():
    problem = SlopeSlide(seed=0)
    q = problem.question()
    assert q.answer_type == AnswerType.BOOLEAN
    assert q.text.lower().startswith("a block rests")


def test_slope_slide_ground_truth_runs():
    problem = SlopeSlide(seed=0)
    gt = problem.ground_truth()
    assert gt.answer in {"yes", "no"}
    assert "angle_deg" in gt.latent_params
    assert "mu_s" in gt.latent_params


def test_support_balance_question():
    problem = SupportBalance(seed=0)
    q = problem.question()
    assert q.answer_type == AnswerType.CHOICE
    assert set(q.choices) == {"left of center", "center", "right of center"}


def test_support_balance_ground_truth_runs():
    problem = SupportBalance(seed=0)
    gt = problem.ground_truth()
    assert gt.answer in {"left of center", "center", "right of center"}
    assert "balance_x" in gt.latent_params


def test_topple_direction_question():
    problem = ToppleDirection(seed=0)
    q = problem.question()
    assert q.answer_type == AnswerType.CHOICE
    assert set(q.choices) == {"left", "right", "neither"}


def test_topple_direction_ground_truth_runs():
    problem = ToppleDirection(seed=0)
    gt = problem.ground_truth()
    assert gt.answer in {"left", "right", "neither"}
    assert "tilt_angle_deg" in gt.latent_params


def test_physics_oracle_perfect_on_each_statics_scene():
    for cls in STATICS_CLASSES:
        problem = cls(seed=1)
        oracle = PhysicsOraclePredictor()
        pred = oracle.predict(problem)
        assert problem.score(pred) == 1.0, f"physics oracle failed on {cls.__name__}"


def test_pendulum_swing_question():
    problem = PendulumSwing(seed=0)
    q = problem.question()
    assert q.answer_type == AnswerType.CHOICE
    assert set(q.choices) == {"A", "B", "same"}


def test_pendulum_swing_ground_truth_runs():
    problem = PendulumSwing(seed=0)
    gt = problem.ground_truth()
    assert gt.answer in {"A", "B", "same"}
    assert "length_a" in gt.latent_params


def test_collision_bounce_question():
    problem = CollisionBounce(seed=0)
    q = problem.question()
    assert q.answer_type == AnswerType.CHOICE
    assert set(q.choices) == {"A", "B", "same"}


def test_collision_bounce_ground_truth_runs():
    problem = CollisionBounce(seed=0)
    gt = problem.ground_truth()
    assert gt.answer in {"A", "B", "same"}
    assert "mass_a" in gt.latent_params
    assert "mass_b" in gt.latent_params


def test_projectile_hit_question():
    problem = ProjectileHit(seed=0)
    q = problem.question()
    assert q.answer_type == AnswerType.NUMERIC
    assert q.units == "m"


def test_projectile_hit_ground_truth_runs():
    problem = ProjectileHit(seed=0)
    gt = problem.ground_truth()
    assert isinstance(gt.answer, float)
    assert gt.answer > 0
    assert "speed" in gt.latent_params


def test_physics_oracle_perfect_on_each_dynamics_scene():
    for cls in DYNAMICS_CLASSES:
        problem = cls(seed=1)
        oracle = PhysicsOraclePredictor()
        pred = oracle.predict(problem)
        assert problem.score(pred) == 1.0, f"physics oracle failed on {cls.__name__}"


def test_push_tip_vs_slide_question():
    problem = PushTipVsSlide(seed=0)
    q = problem.question()
    assert q.answer_type == AnswerType.CHOICE
    assert set(q.choices) == {"tip", "slide"}


def test_push_tip_vs_slide_ground_truth_runs():
    problem = PushTipVsSlide(seed=0)
    gt = problem.ground_truth()
    assert gt.answer in {"tip", "slide"}
    assert "max_tilt_deg" in gt.latent_params


def test_stack_stability_question():
    problem = StackStability(seed=0)
    q = problem.question()
    assert q.answer_type == AnswerType.BOOLEAN


def test_stack_stability_ground_truth_runs():
    problem = StackStability(seed=0)
    gt = problem.ground_truth()
    assert gt.answer in {"yes", "no"}
    assert "n_blocks" in gt.latent_params


def test_wedge_insert_question():
    problem = WedgeInsert(seed=0)
    q = problem.question()
    assert q.answer_type == AnswerType.CHOICE
    assert set(q.choices) == {"fits", "jams"}


def test_wedge_insert_ground_truth_runs():
    problem = WedgeInsert(seed=0)
    gt = problem.ground_truth()
    assert gt.answer in {"fits", "jams"}
    assert "wedge_base_width" in gt.latent_params
    assert "gap_width" in gt.latent_params


def test_friction_pile_question():
    problem = FrictionPile(seed=0)
    q = problem.question()
    assert q.answer_type == AnswerType.CHOICE
    assert set(q.choices) == {"A", "B", "C"}


def test_friction_pile_ground_truth_runs():
    problem = FrictionPile(seed=0)
    gt = problem.ground_truth()
    assert gt.answer in {"A", "B", "C"}
    assert "thresholds" in gt.latent_params


def test_slip_grip_question():
    problem = SlipGrip(seed=0)
    q = problem.question()
    assert q.answer_type == AnswerType.CHOICE
    assert set(q.choices) == {"lift", "slip"}


def test_slip_grip_ground_truth_runs():
    problem = SlipGrip(seed=0)
    gt = problem.ground_truth()
    assert gt.answer in {"lift", "slip"}
    assert "total_friction_capacity" in gt.latent_params


def test_physics_oracle_perfect_on_each_contact_scene():
    for cls in CONTACT_CLASSES:
        problem = cls(seed=1)
        oracle = PhysicsOraclePredictor()
        pred = oracle.predict(problem)
        assert problem.score(pred) == 1.0, f"physics oracle failed on {cls.__name__}"


def test_suite_instantiates():
    suite = Suite("statics", seed=0, n_instances=2)
    problems = suite.problems()
    assert len(problems) == len(STATICS_CLASSES) * 2
    assert all(type(p) in STATICS_CLASSES for p in problems)

    dynamics_suite = Suite("dynamics", seed=0, n_instances=2)
    dyn_problems = dynamics_suite.problems()
    assert len(dyn_problems) == len(DYNAMICS_CLASSES) * 2
    assert all(type(p) in DYNAMICS_CLASSES for p in dyn_problems)

    contact_suite = Suite("contact", seed=0, n_instances=2)
    contact_problems = contact_suite.problems()
    assert len(contact_problems) == len(CONTACT_CLASSES) * 2
    assert all(type(p) in CONTACT_CLASSES for p in contact_problems)


def test_runner_and_evaluator():
    statics_suite = Suite("statics", seed=0, n_instances=3)
    dynamics_suite = Suite("dynamics", seed=0, n_instances=3)
    contact_suite = Suite("contact", seed=0, n_instances=3)
    runner = Runner(PhysicsOraclePredictor())
    result = runner.run([statics_suite, dynamics_suite, contact_suite])
    metrics = Evaluator.metrics(result)
    assert metrics["overall_accuracy"] == 1.0
    assert result.predictor == "physics_oracle"


def test_random_baseline_scores_less_than_oracle():
    statics_suite = Suite("statics", seed=0, n_instances=10)
    dynamics_suite = Suite("dynamics", seed=0, n_instances=10)
    contact_suite = Suite("contact", seed=0, n_instances=10)
    suites = [statics_suite, dynamics_suite, contact_suite]
    oracle_result = Runner(PhysicsOraclePredictor()).run(suites)
    random_result = Runner(RandomPredictor(seed=7)).run(suites)
    oracle_acc = Evaluator.metrics(oracle_result)["overall_accuracy"]
    random_acc = Evaluator.metrics(random_result)["overall_accuracy"]
    assert oracle_acc == 1.0
    assert random_acc < 1.0
