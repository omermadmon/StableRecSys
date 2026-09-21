from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from brd import create_directions_set, max_discrete_improvement
from game import StableMechanismPublishersGame

try:
    from scipy.optimize import minimize
except ModuleNotFoundError:  # pragma: no cover - exercised only without scipy.
    minimize = None


@dataclass
class RegretMinimizingPlayer:
    """LRL-OFTRL player adaptation."""

    d: int
    eta: float

    def __post_init__(self) -> None:
        self.U_tilde = np.zeros(self.d + 1)
        self.u_tilde = np.zeros(self.d + 1)
        self.lam = None
        self.y = None
        self.x = None

    def act(self, rng: np.random.Generator) -> np.ndarray:
        initial_guess = None
        if self.lam is not None:
            initial_guess = np.array([self.lam, *self.y])
        self.lam, self.y = solve_opt_prob(
            self.u_tilde, self.U_tilde, self.eta, rng, initial_guess
        )
        self.x = self.y / self.lam
        return self.x

    def update(self, utility_gradient: np.ndarray) -> None:
        self.u_tilde = np.array([-np.dot(self.x, utility_gradient), *utility_gradient])
        self.U_tilde += self.u_tilde


def solve_opt_prob(
    u_tilde: np.ndarray,
    U_tilde: np.ndarray,
    eta: float,
    rng: np.random.Generator,
    initial_guess: np.ndarray | None = None,
) -> tuple[float, np.ndarray]:
    d = len(u_tilde) - 1
    tmp = U_tilde + u_tilde

    if initial_guess is None:
        initial_guess = rng.uniform(low=1e-5, high=1.0, size=d + 1)
    initial_guess = project_lam_y(initial_guess)

    if minimize is not None:
        return solve_opt_prob_slsqp(tmp, eta, initial_guess)

    z = initial_guess

    for step_num in range(1, 501):
        grad = eta * tmp + 1 / z
        step = 0.05 / np.sqrt(step_num)
        z = project_lam_y(z + step * grad)
    return float(z[0]), z[1:]


def project_lam_y(lam_y: np.ndarray) -> np.ndarray:
    """Enforce the lifted feasibility constraints: eps <= y_i <= lambda <= 1."""
    z = np.asarray(lam_y, dtype=float).copy()
    z = np.clip(z, 1e-5, 1.0)
    z[1:] = np.minimum(z[1:], z[0])
    return z


def solve_opt_prob_slsqp(
    tmp: np.ndarray, eta: float, initial_guess: np.ndarray
) -> tuple[float, np.ndarray]:
    """Solve the log-regularized OFTRL update using SLSQP."""

    def objective(lam_y: np.ndarray) -> float:
        return float(-(eta * np.dot(tmp, lam_y) + np.sum(np.log(lam_y))))

    def objective_grad(lam_y: np.ndarray) -> np.ndarray:
        return -tmp * eta - 1 / lam_y

    d = len(initial_guess) - 1
    bounds = [(1e-5, 1.0)] * (d + 1)
    constraints = [
        {"type": "ineq", "fun": lambda lam_y, idx=i: lam_y[0] - lam_y[idx]}
        for i in range(1, d + 1)
    ]
    result = minimize(
        fun=objective,
        x0=initial_guess,
        bounds=bounds,
        method="SLSQP",
        constraints=constraints,
        jac=objective_grad,
        options={"ftol": 1e-9, "maxiter": 200},
    )
    if not result.success:
        raise ValueError(f"LRL-OFTRL optimization failed: {result.message}")
    return float(result.x[0]), result.x[1:]


@dataclass(frozen=True)
class NRDResult:
    publishers_welfare_last: float
    users_welfare_last: float
    publishers_welfare_average: float
    users_welfare_average: float
    convergence_round_average: int | None
    convergence_round_last: int | None
    converged_average: bool
    converged_last: bool
    max_improvement_average: float
    max_improvement_last: float


def no_regret_dynamics(
    game: StableMechanismPublishersGame,
    max_rounds: int,
    eta: float,
    eps: float,
    check_every: int,
    steps: list[float] | np.ndarray,
    rng: np.random.Generator,
) -> NRDResult:
    """No-regret dynamics using analytic gradients from the game."""
    players = [RegretMinimizingPlayer(d=game.k, eta=eta) for _ in game.N]
    directions = create_directions_set(game.k, rng)
    running_sum = np.zeros_like(game.x)
    convergence_round_average = None
    convergence_round_last = None

    for round_num in range(1, max_rounds + 1):
        x_new = np.array([player.act(rng) for player in players])
        game.update_x(x_new)
        running_sum += x_new

        gradients = game.get_grad_all()
        for i in game.N:
            players[i].update(gradients[i])

        if round_num % check_every == 0:
            avg_x = running_sum / round_num
            avg_game = game.copy().update_x(avg_x)
            if convergence_round_average is None:
                avg_improvement = max_discrete_improvement(avg_game, directions, steps, rng)
                if avg_improvement <= eps:
                    convergence_round_average = round_num
            if convergence_round_last is None:
                last_improvement = max_discrete_improvement(game, directions, steps, rng)
                if last_improvement <= eps:
                    convergence_round_last = round_num

    average_x = running_sum / max_rounds
    average_game = game.copy().update_x(average_x)
    avg_improvement = max_discrete_improvement(average_game, directions, steps, rng)
    last_improvement = max_discrete_improvement(game, directions, steps, rng)

    return NRDResult(
        game.get_publishers_welfare(),
        game.get_users_welfare(),
        average_game.get_publishers_welfare(),
        average_game.get_users_welfare(),
        convergence_round_average,
        convergence_round_last,
        convergence_round_average is not None,
        convergence_round_last is not None,
        avg_improvement,
        last_improvement,
    )
