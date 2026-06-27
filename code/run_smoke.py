from __future__ import annotations

import argparse

import numpy as np

from activations import RootPower, ShiftedLinear, ShiftedLog
from brd import discrete_better_response_dynamics
from configs import SimulationConfig
from game import StableMechanismPublishersGame, generate_uniform_instance
from nrd import no_regret_dynamics


def build_game(config: SimulationConfig, activation, rng: np.random.Generator):
    x_0, y_values = generate_uniform_instance(config.k, config.n, config.s, rng)
    return StableMechanismPublishersGame(
        config.k,
        config.n,
        config.s,
        config.lam,
        x_0,
        y_values,
        activation,
        config.beta,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrd-rounds", type=int, default=100)
    parser.add_argument("--brd-rounds", type=int, default=100)
    args = parser.parse_args()

    config = SimulationConfig(nrd_max_rounds=args.nrd_rounds, brd_max_rounds=args.brd_rounds)
    activations = [ShiftedLinear(), RootPower(), ShiftedLog()]
    for idx, activation in enumerate(activations):
        rng = np.random.default_rng(config.seed + idx)
        brd_game = build_game(config, activation, rng)
        brd = discrete_better_response_dynamics(
            brd_game, config.steps, config.brd_max_rounds, config.eps, rng
        )

        nrd_game = build_game(config, activation, rng)
        nrd = no_regret_dynamics(
            nrd_game,
            config.nrd_max_rounds,
            config.eta,
            config.eps,
            config.nrd_check_every,
            config.steps,
            rng,
        )

        print(
            activation.name,
            {
                "brd_converged": brd.converged,
                "brd_updates": brd.updates,
                "brd_creator_welfare": round(brd.publishers_welfare, 6),
                "brd_users_welfare": round(brd.users_welfare, 6),
                "nrd_avg_converged": nrd.converged_average,
                "nrd_avg_round": nrd.convergence_round_average,
                "nrd_avg_creator_welfare": round(nrd.publishers_welfare_average, 6),
                "nrd_avg_users_welfare": round(nrd.users_welfare_average, 6),
            },
        )


if __name__ == "__main__":
    main()
