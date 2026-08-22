"""Tests for Chapter 8 motion planning."""
from __future__ import annotations

import numpy as np
import pytest

from collision import ArmPlanningEnv
from planners import (
    PRMPlanner,
    PotentialFieldPlanner,
    RRTPlanner,
    RRTStarPlanner,
    path_length,
)
from smoother import shortcut_smooth


@pytest.fixture
def env_obstacles():
    """Planning environment with a few static obstacles."""
    obstacles = [
        # A vertical wall in front of the home configuration.
        {"type": "box", "pos": [0.78, 0.0, 0.40], "size": [0.05, 0.45, 0.45]},
        # A sphere guarding the upper-right workspace.
        {"type": "sphere", "pos": [0.62, 0.35, 0.50], "size": [0.12]},
        # A thin bar guarding the lower-right workspace.
        {"type": "box", "pos": [0.58, -0.35, 0.35], "size": [0.18, 0.05, 0.35]},
    ]
    return ArmPlanningEnv(obstacles=obstacles)


@pytest.fixture
def env_empty():
    """Obstacle-free environment for validating planner correctness."""
    return ArmPlanningEnv(obstacles=[])


@pytest.fixture
def start_goal_empty(env_empty):
    """Return a collision-free start/goal pair in the empty environment."""
    start = np.zeros(env_empty.model.nq)
    goal = np.array([0.0, -0.6, 0.0, 0.0, 0.0, 0.0])
    assert not env_empty.is_collision(goal)
    return start, goal


def test_joint_limit_penalty(env_obstacles):
    """Configurations outside joint limits count as collision."""
    q = np.zeros(env_obstacles.model.nq)
    assert not env_obstacles.is_collision(q)
    q_bad = q.copy()
    q_bad[1] = env_obstacles.q_max[1] + 0.5
    assert env_obstacles.is_collision(q_bad)


def test_env_detects_obstacle_collision(env_obstacles):
    """The collision checker should detect an arm placed inside an obstacle."""
    rng = np.random.default_rng(99)
    found = False
    for _ in range(2000):
        q = rng.uniform(env_obstacles.q_min, env_obstacles.q_max)
        if env_obstacles.is_collision(q):
            found = True
            break
    assert found


def test_segment_free_between_known_free_configs(env_obstacles):
    """A short straight segment between two nearby free configs should be free."""
    start = np.zeros(env_obstacles.model.nq)
    goal = np.array([0.0, -0.05, 0.0, 0.0, 0.0, 0.0])
    assert env_obstacles.is_segment_free(start, goal, n_checks=15)


def test_segment_blocked_by_obstacle(env_obstacles):
    """A long straight segment through the obstacle field should be blocked."""
    start = np.zeros(env_obstacles.model.nq)
    goal = np.array([0.0, -0.6, 0.0, 0.0, 0.0, 0.0])
    assert not env_obstacles.is_collision(goal)
    assert not env_obstacles.is_segment_free(start, goal, n_checks=30)


def _assert_valid_path(env, path, start, goal):
    assert path is not None
    assert len(path) >= 2
    np.testing.assert_allclose(path[0], start, atol=1e-6)
    np.testing.assert_allclose(path[-1], goal, atol=1e-6)
    for i in range(len(path) - 1):
        assert env.is_segment_free(path[i], path[i + 1], n_checks=10)


def test_rrt_plans_collision_free_path(env_empty, start_goal_empty):
    start, goal = start_goal_empty
    planner = RRTPlanner(step_size=0.15, goal_bias=0.1, max_iters=2000, seed=1)
    path = planner.plan(start, goal, env_empty)
    _assert_valid_path(env_empty, path, start, goal)


def test_rrt_star_plans_collision_free_path(env_empty, start_goal_empty):
    start, goal = start_goal_empty
    planner = RRTStarPlanner(
        step_size=0.15, goal_bias=0.1, max_iters=3000, seed=2
    )
    path = planner.plan(start, goal, env_empty)
    _assert_valid_path(env_empty, path, start, goal)


def test_prm_plans_collision_free_path(env_empty, start_goal_empty):
    start, goal = start_goal_empty
    planner = PRMPlanner(
        n_nodes=200, k_neighbors=8, max_edge_length=1.2, max_iters=3000, seed=3
    )
    path = planner.plan(start, goal, env_empty)
    _assert_valid_path(env_empty, path, start, goal)


def test_shortcut_smoothing_reduces_length(env_empty, start_goal_empty):
    start, goal = start_goal_empty
    planner = RRTPlanner(step_size=0.15, goal_bias=0.1, max_iters=3000, seed=4)
    path = planner.plan(start, goal, env_empty)
    _assert_valid_path(env_empty, path, start, goal)
    before = path_length(path)
    smoothed = shortcut_smooth(path, env_empty, max_iters=200, seed=4)
    after = path_length(smoothed)
    _assert_valid_path(env_empty, smoothed, start, goal)
    assert after <= before + 1e-6


def test_rrt_star_shorter_than_rrt(env_empty, start_goal_empty):
    """On this easy problem, RRT* should produce a path no longer than RRT."""
    start, goal = start_goal_empty
    rrt = RRTPlanner(step_size=0.15, goal_bias=0.1, max_iters=3000, seed=5)
    rrt_star = RRTStarPlanner(
        step_size=0.15, goal_bias=0.1, max_iters=5000, seed=5
    )
    path_rrt = rrt.plan(start, goal, env_empty)
    path_star = rrt_star.plan(start, goal, env_empty)
    _assert_valid_path(env_empty, path_rrt, start, goal)
    _assert_valid_path(env_empty, path_star, start, goal)
    assert path_length(path_star) <= path_length(path_rrt) + 1e-6


def test_potential_field_may_reach_goal(env_empty, start_goal_empty):
    """Potential field is not guaranteed; if it succeeds, the path is valid."""
    start, goal = start_goal_empty
    planner = PotentialFieldPlanner(
        step_size=0.05, max_iters=3000, seed=6, attractive_gain=2.0
    )
    path = planner.plan(start, goal, env_empty)
    if path is not None:
        _assert_valid_path(env_empty, path, start, goal)
