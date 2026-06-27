from __future__ import annotations

from dataclasses import dataclass

import numpy as np


ArrayLike = np.ndarray | float


class Activation:
    """Base callable for relevance-side activations h:[0,1]->[0,1]."""

    name: str

    def value(self, q: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def derivative(self, q: ArrayLike) -> ArrayLike:
        raise NotImplementedError

    def __call__(self, q: ArrayLike) -> ArrayLike:
        return self.value(q)


@dataclass(frozen=True)
class ShiftedLinear(Activation):
    b: float = 1.00001
    name: str = "shifted_linear"

    def __post_init__(self) -> None:
        if self.b <= 1:
            raise ValueError(f"b must be greater than 1, got {self.b}")

    def value(self, q: ArrayLike) -> ArrayLike:
        q = np.asarray(q)
        return (self.b - 1 + q) / self.b

    def derivative(self, q: ArrayLike) -> ArrayLike:
        return np.ones_like(np.asarray(q), dtype=float) / self.b


@dataclass(frozen=True)
class RootPower(Activation):
    a: float = 0.5
    name: str = "root_power"

    def __post_init__(self) -> None:
        if not (0 <= self.a < 1):
            raise ValueError(f"a must lie in [0, 1), got {self.a}")

    @property
    def exponent(self) -> float:
        return 1.0 - self.a

    def value(self, q: ArrayLike) -> ArrayLike:
        q = np.asarray(q)
        return np.power(np.clip(q, 0.0, 1.0), self.exponent)

    def derivative(self, q: ArrayLike) -> ArrayLike:
        q = np.asarray(q, dtype=float)
        if self.exponent == 1:
            return np.ones_like(q)
        return self.exponent * np.power(np.maximum(q, 1e-12), self.exponent - 1)


@dataclass(frozen=True)
class ShiftedLog(Activation):
    c: float = 2.00001
    name: str = "shifted_log"

    def __post_init__(self) -> None:
        if self.c <= 2:
            raise ValueError(f"c must be greater than 2, got {self.c}")

    def value(self, q: ArrayLike) -> ArrayLike:
        q = np.asarray(q)
        return np.log(self.c - 1 + q) / np.log(self.c)

    def derivative(self, q: ArrayLike) -> ArrayLike:
        q = np.asarray(q, dtype=float)
        return 1 / ((self.c - 1 + q) * np.log(self.c))


@dataclass(frozen=True)
class Saturation(Activation):
    tau: float = 1.0
    name: str = "saturation"

    def __post_init__(self) -> None:
        if self.tau <= 0:
            raise ValueError(f"tau must be positive, got {self.tau}")

    def value(self, q: ArrayLike) -> ArrayLike:
        q = np.asarray(q)
        return (1 - np.exp(-self.tau * q)) / (1 - np.exp(-self.tau))

    def derivative(self, q: ArrayLike) -> ArrayLike:
        q = np.asarray(q, dtype=float)
        return self.tau * np.exp(-self.tau * q) / (1 - np.exp(-self.tau))


def default_activations() -> list[Activation]:
    return [ShiftedLinear(), RootPower(), ShiftedLog()]
