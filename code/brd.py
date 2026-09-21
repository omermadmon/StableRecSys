from __future__ import annotations

import itertools
from dataclasses import dataclass

import numpy as np

from game import StableMechanismPublishersGame


def create_directions_set(
    k: int,
    rng: np.random.Generator | None = None,
    max_directions: int = 200,
) -> np.ndarray:
    """Create normalized grid directions, sampled when 3^k is too large."""
    total_nonzero = (3**k) - 1
    if total_nonzero <= max_directions:
        directions = itertools.product([-1, 0, 1], repeat=k)
        return np.array(
            [np.array(d) / np.linalg.norm(d) for d in directions if np.linalg.norm(d) > 0]
        )

    if rng is None:
        rng = np.random.default_rng(0)
    directions = []
    for dim in range(k):
        unit = np.zeros(k)
        unit[dim] = 1.0
        directions.append(unit)
        directions.append(-unit)
    while len(directions) < max_directions:
        candidate = rng.choice([-1.0, 0.0, 1.0], size=k)
        norm = np.linalg.norm(candidate)
        if norm > 0:
            directions.append(candidate / norm)
    return np.array(directions)


def find_best_step_strategy_discrete(
    game: StableMechanismPublishersGame,
    i: int,
    directions: np.ndarray,
    steps: list[float] | np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray | None, float]:
    shuffled = np.array(directions, copy=True)
    rng.shuffle(shuffled)
    max_util = -np.inf
    strategy = None
    for direction, step_size in itertools.product(shuffled, steps):
        candidate = game.x[i] + step_size * direction
        if np.all((0 <= candidate) & (candidate <= 1)):
            candidate_util = game.calc_u_deviation(candidate, i)
            if candidate_util > max_util:
                strategy = candidate
                max_util = candidate_util
    return strategy, float(max_util)


def max_discrete_improvement(
    game: StableMechanismPublishersGame,
    directions: np.ndarray,
    steps: list[float] | np.ndarray,
    rng: np.random.Generator,
) -> float:
    improvements = []
    for i in game.N:
        _, best_u = find_best_step_strategy_discrete(game, i, directions, steps, rng)
        improvements.append(best_u - game.get_u(i))
    return float(max(improvements))


@dataclass(frozen=True)
class BRDResult:
    publishers_welfare: float
    users_welfare: float
    updates: int
    sweeps: float
    converged: bool
    max_improvement: float


def discrete_better_response_dynamics(
    game: StableMechanismPublishersGame,
    steps: list[float] | np.ndarray,
    max_rounds: int,
    eps: float,
    rng: np.random.Generator,
) -> BRDResult:
    """Discrete better-response dynamics."""
    game.initialize()
    directions = create_directions_set(game.k, rng)
    updates = 0

    for _ in range(max_rounds):
        found_improving_player = False
        for i in rng.permutation(game.N):
            candidate, candidate_u = find_best_step_strategy_discrete(
                game, i, directions, steps, rng
            )
            if candidate is not None and game.get_u(i) + eps < candidate_u:
                game.update_x_i(candidate, i)
                updates += 1
                found_improving_player = True
                break
        if not found_improving_player:
            return BRDResult(
                game.get_publishers_welfare(),
                game.get_users_welfare(),
                updates,
                updates / game.n,
                True,
                max_discrete_improvement(game, directions, steps, rng),
            )

    return BRDResult(
        game.get_publishers_welfare(),
        game.get_users_welfare(),
        updates,
        updates / game.n,
        False,
        max_discrete_improvement(game, directions, steps, rng),
    )
