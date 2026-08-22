"""Chapter 8 — Motion Planning: sampling-based and potential-field planners."""
from __future__ import annotations

import heapq
from typing import Protocol

import numpy as np


class CollisionChecker(Protocol):
    """Minimal interface expected by planners."""

    def is_collision(self, q: np.ndarray, penetration: float = 1e-4) -> bool: ...

    def is_segment_free(
        self, q1: np.ndarray, q2: np.ndarray, n_checks: int = 20
    ) -> bool: ...

    def sample_collision_free(self, rng: np.random.Generator) -> np.ndarray | None: ...


def q_distance(q1: np.ndarray, q2: np.ndarray) -> float:
    """Weighted Euclidean distance in C-space (heavier joints move bigger links)."""
    weights = np.array([1.0, 1.2, 1.0, 0.6, 0.4, 0.2], dtype=float)
    return float(np.linalg.norm(weights * (q1 - q2)))


def linear_interpolate(q1: np.ndarray, q2: np.ndarray, step: float) -> list[np.ndarray]:
    """Return intermediate points from q1 to q2 spaced by at most `step`."""
    dist = q_distance(q1, q2)
    if dist <= 1e-9:
        return [q1]
    n = max(1, int(np.ceil(dist / step)))
    return [q1 + (i / n) * (q2 - q1) for i in range(n + 1)]


def path_length(path: list[np.ndarray]) -> float:
    """Total C-space distance along a path."""
    if len(path) < 2:
        return 0.0
    total = 0.0
    for i in range(len(path) - 1):
        total += q_distance(path[i], path[i + 1])
    return total


class Planner:
    """Base class for a C-space planner."""

    def __init__(self, max_iters: int = 2000, seed: int | None = 0) -> None:
        self.max_iters = max_iters
        self.rng = np.random.default_rng(seed)

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        checker: CollisionChecker,
    ) -> list[np.ndarray] | None:
        raise NotImplementedError


class PRMPlanner(Planner):
    """Probabilistic Roadmap planner with A* search on a k-NN graph."""

    def __init__(
        self,
        n_nodes: int = 500,
        k_neighbors: int = 10,
        max_edge_length: float = 1.5,
        max_iters: int = 2000,
        seed: int | None = 0,
    ) -> None:
        super().__init__(max_iters=max_iters, seed=seed)
        self.n_nodes = n_nodes
        self.k_neighbors = k_neighbors
        self.max_edge_length = max_edge_length

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        checker: CollisionChecker,
    ) -> list[np.ndarray] | None:
        nodes: list[np.ndarray] = [np.asarray(start, dtype=float), np.asarray(goal, dtype=float)]
        edges: list[list[tuple[int, float]]] = [[], []]

        # Sample collision-free nodes.
        attempts = 0
        while len(nodes) < self.n_nodes + 2 and attempts < self.n_nodes * 20:
            attempts += 1
            q = checker.sample_collision_free(self.rng)
            if q is None:
                continue
            nodes.append(q)
            edges.append([])

        n = len(nodes)

        def _nearest_indices(query: np.ndarray, k: int, exclude: int) -> list[int]:
            ds = [(i, q_distance(query, nodes[i])) for i in range(n) if i != exclude]
            ds.sort(key=lambda x: x[1])
            return [i for i, _ in ds[:k]]

        # Connect k-NN edges.
        for i in range(n):
            for j in _nearest_indices(nodes[i], self.k_neighbors, i):
                d = q_distance(nodes[i], nodes[j])
                if d > self.max_edge_length:
                    continue
                if checker.is_segment_free(nodes[i], nodes[j]):
                    edges[i].append((j, d))
                    edges[j].append((i, d))

        # A* from start (0) to goal (1).
        start_idx, goal_idx = 0, 1
        open_set: list[tuple[float, int]] = [(0.0, start_idx)]
        g_score: dict[int, float] = {start_idx: 0.0}
        came_from: dict[int, int] = {}
        visited: set[int] = set()

        while open_set:
            _, current = heapq.heappop(open_set)
            if current in visited:
                continue
            visited.add(current)
            if current == goal_idx:
                path = [nodes[goal_idx]]
                while current in came_from:
                    current = came_from[current]
                    path.append(nodes[current])
                path.reverse()
                return path
            for nb, cost in edges[current]:
                if nb in visited:
                    continue
                tentative = g_score[current] + cost
                if nb not in g_score or tentative < g_score[nb]:
                    g_score[nb] = tentative
                    came_from[nb] = current
                    f = tentative + q_distance(nodes[nb], nodes[goal_idx])
                    heapq.heappush(open_set, (f, nb))

        return None


class RRTPlanner(Planner):
    """Single-tree RRT in joint space."""

    def __init__(
        self,
        step_size: float = 0.15,
        goal_bias: float = 0.1,
        max_iters: int = 2000,
        seed: int | None = 0,
    ) -> None:
        super().__init__(max_iters=max_iters, seed=seed)
        self.step_size = step_size
        self.goal_bias = goal_bias

    def _steer(self, q_near: np.ndarray, q_rand: np.ndarray) -> np.ndarray:
        d = q_distance(q_near, q_rand)
        if d <= self.step_size:
            return q_rand
        return q_near + (self.step_size / d) * (q_rand - q_near)

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        checker: CollisionChecker,
    ) -> list[np.ndarray] | None:
        nodes: list[np.ndarray] = [np.asarray(start, dtype=float)]
        parents: list[int] = [-1]
        goal = np.asarray(goal, dtype=float)

        for _ in range(self.max_iters):
            if self.rng.random() < self.goal_bias:
                q_rand = goal
            else:
                q_free = checker.sample_collision_free(self.rng)
                if q_free is None:
                    continue
                q_rand = q_free

            # Nearest node.
            nearest = int(np.argmin([q_distance(n, q_rand) for n in nodes]))
            q_new = self._steer(nodes[nearest], q_rand)
            if checker.is_segment_free(nodes[nearest], q_new, n_checks=10):
                nodes.append(q_new)
                parents.append(nearest)
                if q_distance(q_new, goal) <= self.step_size and checker.is_segment_free(
                    q_new, goal, n_checks=10
                ):
                    nodes.append(goal)
                    parents.append(len(nodes) - 2)
                    # Reconstruct path.
                    path = [nodes[-1]]
                    idx = len(nodes) - 1
                    while idx >= 0:
                        idx = parents[idx]
                        if idx < 0:
                            break
                        path.append(nodes[idx])
                    path.reverse()
                    return path
        return None


class RRTStarPlanner(RRTPlanner):
    """RRT* with near-neighbor rewiring in joint space."""

    def __init__(
        self,
        step_size: float = 0.15,
        goal_bias: float = 0.1,
        max_iters: int = 2000,
        rewire_radius: float | None = None,
        seed: int | None = 0,
    ) -> None:
        super().__init__(
            step_size=step_size, goal_bias=goal_bias, max_iters=max_iters, seed=seed
        )
        self.rewire_radius = rewire_radius

    def _radius(self, n: int) -> float:
        if self.rewire_radius is not None:
            return self.rewire_radius
        # Diminishing radius heuristic.
        return min(1.5, self.step_size * 3.0 * (np.log(n + 1) / (n + 1)) ** (1.0 / 6.0))

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        checker: CollisionChecker,
    ) -> list[np.ndarray] | None:
        nodes: list[np.ndarray] = [np.asarray(start, dtype=float)]
        parents: list[int] = [-1]
        costs: list[float] = [0.0]
        goal = np.asarray(goal, dtype=float)

        for _ in range(self.max_iters):
            if self.rng.random() < self.goal_bias:
                q_rand = goal
            else:
                q_free = checker.sample_collision_free(self.rng)
                if q_free is None:
                    continue
                q_rand = q_free

            ds = [q_distance(n, q_rand) for n in nodes]
            nearest = int(np.argmin(ds))
            q_new = self._steer(nodes[nearest], q_rand)
            if not checker.is_segment_free(nodes[nearest], q_new, n_checks=10):
                continue

            new_cost = costs[nearest] + q_distance(nodes[nearest], q_new)

            # Choose best parent among neighbors within radius.
            radius = self._radius(len(nodes))
            neighbors = [i for i, d in enumerate(ds) if d <= radius]
            best_parent = nearest
            best_cost = new_cost
            for i in neighbors:
                c = costs[i] + q_distance(nodes[i], q_new)
                if c < best_cost and checker.is_segment_free(
                    nodes[i], q_new, n_checks=10
                ):
                    best_parent = i
                    best_cost = c

            new_idx = len(nodes)
            nodes.append(q_new)
            parents.append(best_parent)
            costs.append(best_cost)

            # Rewire neighbors through new node.
            for i in neighbors:
                if i == best_parent:
                    continue
                c = costs[new_idx] + q_distance(q_new, nodes[i])
                if c < costs[i] and checker.is_segment_free(
                    q_new, nodes[i], n_checks=10
                ):
                    parents[i] = new_idx
                    # Update costs for subtree rooted at i (simple recursive update).
                    _update_costs(i, parents, costs, nodes)

            if q_distance(q_new, goal) <= self.step_size and checker.is_segment_free(
                q_new, goal, n_checks=10
            ):
                nodes.append(goal)
                parents.append(new_idx)
                costs.append(costs[new_idx] + q_distance(q_new, goal))
                path = [nodes[-1]]
                idx = len(nodes) - 1
                while idx >= 0:
                    idx = parents[idx]
                    if idx < 0:
                        break
                    path.append(nodes[idx])
                path.reverse()
                return path

        return None


def _update_costs(root: int, parents: list[int], costs: list[float], nodes: list[np.ndarray]) -> None:
    """Recursively update path costs after a rewire."""
    children = [i for i, p in enumerate(parents) if p == root and i != root]
    for child in children:
        costs[child] = costs[root] + q_distance(nodes[root], nodes[child])
        _update_costs(child, parents, costs, nodes)


class PotentialFieldPlanner(Planner):
    """Simple artificial-potential-field gradient descent (may get stuck)."""

    def __init__(
        self,
        step_size: float = 0.05,
        max_iters: int = 2000,
        seed: int | None = 0,
        attractive_gain: float = 1.0,
        repulsive_gain: float = 0.5,
        obstacle_influence: float = 0.6,
    ) -> None:
        super().__init__(max_iters=max_iters, seed=seed)
        self.step_size = step_size
        self.attractive_gain = attractive_gain
        self.repulsive_gain = repulsive_gain
        self.obstacle_influence = obstacle_influence

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        checker: CollisionChecker,
    ) -> list[np.ndarray] | None:
        q = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        path = [q.copy()]
        for _ in range(self.max_iters):
            grad = self.attractive_gain * (goal - q)
            # Simple obstacle repulsion: finite-difference probe at each joint.
            for j in range(len(q)):
                dq = np.zeros_like(q)
                dq[j] = self.obstacle_influence
                q_plus = q + dq
                q_minus = q - dq
                # Penalize moving closer to obstacles.
                if checker.is_collision(q_plus):
                    grad[j] += self.repulsive_gain / self.obstacle_influence
                if checker.is_collision(q_minus):
                    grad[j] -= self.repulsive_gain / self.obstacle_influence

            step = self.step_size * grad / (np.linalg.norm(grad) + 1e-9)
            q = q + step
            path.append(q.copy())
            if q_distance(q, goal) < self.step_size:
                path.append(goal)
                return path
        return None
