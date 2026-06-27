from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from activations import Activation
from mechanisms import (
    publisher_welfare,
    squared_distance,
    stable_relative_exposure_all,
    user_welfare,
)


@dataclass
class StableMechanismPublishersGame:
    """Continuous publisher game for the characterized stable mechanism class."""

    k: int
    n: int
    s: int
    lam: float
    x_0: np.ndarray
    y_values: np.ndarray
    activation: Activation
    beta: float = 1.0
    finite_diff_eps: float = 1e-6

    def __post_init__(self) -> None:
        self.x_0 = np.asarray(self.x_0, dtype=float)
        self.y_values = np.asarray(self.y_values, dtype=float)
        if self.x_0.shape != (self.n, self.k):
            raise ValueError(f"x_0.shape={self.x_0.shape}, expected {(self.n, self.k)}")
        if self.y_values.shape != (self.s, self.k):
            raise ValueError(
                f"y_values.shape={self.y_values.shape}, expected {(self.s, self.k)}"
            )
        if self.lam < 0:
            raise ValueError(f"lambda must be nonnegative, got {self.lam}")
        self.N = list(range(self.n))
        self.x = self.x_0.copy()

    def initialize(self) -> "StableMechanismPublishersGame":
        return self.update_x(self.x_0)

    def copy(self) -> "StableMechanismPublishersGame":
        copied = StableMechanismPublishersGame(
            self.k,
            self.n,
            self.s,
            self.lam,
            self.x_0.copy(),
            self.y_values.copy(),
            self.activation,
            self.beta,
            self.finite_diff_eps,
        )
        copied.update_x(self.x)
        return copied

    def update_x(self, new_x: np.ndarray) -> "StableMechanismPublishersGame":
        new_x = np.asarray(new_x, dtype=float)
        if new_x.shape != (self.n, self.k):
            raise ValueError(f"new_x.shape={new_x.shape}, expected {(self.n, self.k)}")
        self.x = new_x.copy()
        return self

    def update_x_i(self, new_x_i: np.ndarray, i: int) -> "StableMechanismPublishersGame":
        new_x_i = np.asarray(new_x_i, dtype=float)
        if new_x_i.shape != (self.k,):
            raise ValueError(f"new_x_i.shape={new_x_i.shape}, expected {(self.k,)}")
        self.x[i] = new_x_i.copy()
        return self

    def calc_r_all(self, x: np.ndarray | None = None) -> np.ndarray:
        profile = self.x if x is None else np.asarray(x, dtype=float)
        return stable_relative_exposure_all(
            profile, self.y_values, self.activation, self.beta
        )

    def calc_u_all(self, x: np.ndarray | None = None) -> np.ndarray:
        profile = self.x if x is None else np.asarray(x, dtype=float)
        exposures = self.calc_r_all(profile)
        exposure_terms = np.mean(exposures, axis=0)
        costs = self.lam * squared_distance(profile, self.x_0)
        return exposure_terms - costs

    def get_u(self, i: int) -> float:
        return float(self.calc_u_all(self.x)[i])

    def calc_u_deviation(self, xi: np.ndarray, i: int) -> float:
        profile = self.x.copy()
        profile[i] = np.asarray(xi, dtype=float)
        return float(self.calc_u_all(profile)[i])

    def get_publishers_welfare(self) -> float:
        return publisher_welfare(self.x, self.x_0, self.lam)

    def get_users_welfare(self) -> float:
        return user_welfare(self.x, self.y_values, self.activation, self.beta)

    def calc_grad(self, i: int) -> np.ndarray:
        """Analytic gradient of publisher i's utility under the stable rule."""
        xi = self.x[i]
        q = 1.0 - squared_distance(xi, self.y_values)
        h_prime = self.activation.derivative(q)
        relevance_grad = -2.0 * (xi - self.y_values) / self.k
        exposure_grad = (self.beta / self.n) * np.mean(
            h_prime[:, np.newaxis] * relevance_grad, axis=0
        )
        cost_grad = 2.0 * self.lam * (xi - self.x_0[i]) / self.k
        return exposure_grad - cost_grad

    def calc_grad_finite_difference(self, i: int) -> np.ndarray:
        """Central finite-difference gradient for debugging analytic gradients."""
        grad = np.zeros(self.k)
        base = self.x[i].copy()
        h = self.finite_diff_eps
        for dim in range(self.k):
            plus = base.copy()
            minus = base.copy()
            plus[dim] = min(1.0, plus[dim] + h)
            minus[dim] = max(0.0, minus[dim] - h)
            if plus[dim] == minus[dim]:
                grad[dim] = 0.0
            else:
                grad[dim] = (
                    self.calc_u_deviation(plus, i) - self.calc_u_deviation(minus, i)
                ) / (plus[dim] - minus[dim])
        return grad

    def get_grad_all(self) -> np.ndarray:
        return np.array([self.calc_grad(i) for i in self.N])


def generate_uniform_instance(
    k: int,
    n: int,
    s: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    return rng.random((n, k)), rng.random((s, k))
