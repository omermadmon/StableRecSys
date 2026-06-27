from __future__ import annotations

import numpy as np

from activations import Activation


def squared_distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Normalized squared Euclidean distance on [0,1]^k."""
    diff = np.asarray(a) - np.asarray(b)
    return np.mean(diff * diff, axis=-1)


def relevance(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return 1.0 - squared_distance(x, y)


def stable_relative_exposure(
    x: np.ndarray,
    y: np.ndarray,
    activation: Activation,
    beta: float = 1.0,
) -> np.ndarray:
    """Exposure vector for a single information need."""
    x = np.asarray(x, dtype=float)
    n = x.shape[0]
    if n < 2:
        raise ValueError("At least two publishers are required")
    if not (0 <= beta <= 1):
        raise ValueError(f"beta must lie in [0, 1], got {beta}")

    h_vals = activation.value(relevance(x, y))
    avg_others = (np.sum(h_vals) - h_vals) / (n - 1)
    return np.full(n, 1.0 / n) + (beta / n) * (h_vals - avg_others)


def stable_relative_exposure_all(
    x: np.ndarray,
    y_values: np.ndarray,
    activation: Activation,
    beta: float = 1.0,
) -> np.ndarray:
    """Exposure matrix with shape (s, n)."""
    return np.array(
        [stable_relative_exposure(x, y, activation, beta) for y in np.asarray(y_values)]
    )


def publisher_welfare(x: np.ndarray, x_0: np.ndarray, lam: float) -> float:
    return float(1.0 - lam * np.sum(squared_distance(x, x_0)))


def user_welfare(
    x: np.ndarray,
    y_values: np.ndarray,
    activation: Activation,
    beta: float = 1.0,
) -> float:
    exposures = stable_relative_exposure_all(x, y_values, activation, beta)
    rel = np.array([relevance(x, y) for y in np.asarray(y_values)])
    return float(np.mean(np.sum(exposures * rel, axis=1)))


def exposure_feasibility_violation(exposures: np.ndarray) -> float:
    exposures = np.asarray(exposures)
    simplex_violation = np.max(np.abs(np.sum(exposures, axis=-1) - 1.0))
    lower_violation = max(0.0, float(-np.min(exposures)))
    upper_violation = max(0.0, float(np.max(exposures) - 1.0))
    return max(float(simplex_violation), lower_violation, upper_violation)
