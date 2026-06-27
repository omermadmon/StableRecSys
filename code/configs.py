from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


DEFAULT_STEPS = sorted(
    np.round(np.arange(1.0, 0.49, -0.1), 1).tolist()
    + [0.5**j for j in range(2, 7)],
    reverse=True,
)


@dataclass(frozen=True)
class SimulationConfig:
    k: int = 3
    n: int = 3
    s: int = 3
    lam: float = 0.5
    beta: float = 1.0
    eps: float = 1e-4
    eta: float = 0.5
    brd_max_rounds: int = 1000
    nrd_max_rounds: int = 5000
    nrd_check_every: int = 25
    steps: list[float] = field(default_factory=lambda: list(DEFAULT_STEPS))
    seed: int = 0
