from __future__ import annotations

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from activations import Activation, RootPower, ShiftedLinear, ShiftedLog
from brd import discrete_better_response_dynamics
from configs import SimulationConfig
from game import StableMechanismPublishersGame, generate_uniform_instance
from mechanisms import exposure_feasibility_violation
from nrd import no_regret_dynamics


OUTPUT_ROOT = Path("data")


@dataclass(frozen=True)
class ActivationSpec:
    family: str
    param_name: str
    param_value: float

    def build(self) -> Activation:
        if self.family == "shifted_linear":
            return ShiftedLinear(b=self.param_value)
        if self.family == "root_power":
            return RootPower(a=self.param_value)
        if self.family == "shifted_log":
            return ShiftedLog(c=self.param_value)
        raise ValueError(f"Unknown activation family: {self.family}")


def default_activation_specs() -> list[ActivationSpec]:
    return [
        ActivationSpec("shifted_linear", "b", 1.00001),
        ActivationSpec("root_power", "a", 0.5),
        ActivationSpec("shifted_log", "c", 2.00001),
    ]


@dataclass(frozen=True)
class SuiteJob:
    research_question: str
    experiment_name: str
    sweep_parameter: str
    sweep_value: float
    dynamics: str
    activation_family: str
    activation_param_name: str
    activation_param_value: float
    lam: float
    n: int
    s: int
    k: int
    beta: float
    seed: int
    brd_rounds: int
    nrd_rounds: int
    nrd_check_every: int
    eta: float
    eps: float


@dataclass(frozen=True)
class SuiteRow:
    research_question: str
    experiment_name: str
    sweep_parameter: str
    sweep_value: float
    dynamics: str
    activation_family: str
    activation_param_name: str
    activation_param_value: float
    lam: float
    n: int
    s: int
    k: int
    beta: float
    seed: int
    converged: bool
    convergence_step: int | None
    publisher_welfare: float
    user_welfare: float
    max_improvement: float
    last_converged: bool | None
    last_convergence_step: int | None
    last_publisher_welfare: float | None
    last_user_welfare: float | None
    last_max_improvement: float | None
    exposure_violation_initial: float
    exposure_violation_terminal: float


def activation_from_job(job: SuiteJob):
    return ActivationSpec(
        job.activation_family,
        job.activation_param_name,
        job.activation_param_value,
    ).build()


def build_game(job: SuiteJob, x_0: np.ndarray, y_values: np.ndarray, activation):
    return StableMechanismPublishersGame(
        job.k,
        job.n,
        job.s,
        job.lam,
        x_0,
        y_values,
        activation,
        job.beta,
    )


def run_suite_job(job: SuiteJob) -> SuiteRow:
    activation = activation_from_job(job)
    instance_rng = np.random.default_rng(job.seed)
    dynamics_rng = np.random.default_rng(job.seed + 1_000_003)
    x_0, y_values = generate_uniform_instance(job.k, job.n, job.s, instance_rng)

    initial_game = build_game(job, x_0, y_values, activation)
    exposure_initial = exposure_feasibility_violation(initial_game.calc_r_all())

    game = build_game(job, x_0, y_values, activation)
    if job.dynamics == "brd":
        config = SimulationConfig(
            k=job.k,
            n=job.n,
            s=job.s,
            lam=job.lam,
            beta=job.beta,
            eps=job.eps,
            brd_max_rounds=job.brd_rounds,
            seed=job.seed,
        )
        result = discrete_better_response_dynamics(
            game, config.steps, job.brd_rounds, job.eps, dynamics_rng
        )
        return SuiteRow(
            job.research_question,
            job.experiment_name,
            job.sweep_parameter,
            job.sweep_value,
            job.dynamics,
            job.activation_family,
            job.activation_param_name,
            job.activation_param_value,
            job.lam,
            job.n,
            job.s,
            job.k,
            job.beta,
            job.seed,
            result.converged,
            result.updates,
            result.publishers_welfare,
            result.users_welfare,
            result.max_improvement,
            None,
            None,
            None,
            None,
            None,
            exposure_initial,
            exposure_feasibility_violation(game.calc_r_all()),
        )

    if job.dynamics == "nrd":
        config = SimulationConfig(
            k=job.k,
            n=job.n,
            s=job.s,
            lam=job.lam,
            beta=job.beta,
            eps=job.eps,
            eta=job.eta,
            nrd_max_rounds=job.nrd_rounds,
            nrd_check_every=job.nrd_check_every,
            seed=job.seed,
        )
        result = no_regret_dynamics(
            game,
            job.nrd_rounds,
            job.eta,
            job.eps,
            job.nrd_check_every,
            config.steps,
            dynamics_rng,
        )
        return SuiteRow(
            job.research_question,
            job.experiment_name,
            job.sweep_parameter,
            job.sweep_value,
            job.dynamics,
            job.activation_family,
            job.activation_param_name,
            job.activation_param_value,
            job.lam,
            job.n,
            job.s,
            job.k,
            job.beta,
            job.seed,
            result.converged_average,
            result.convergence_round_average,
            result.publishers_welfare_average,
            result.users_welfare_average,
            result.max_improvement_average,
            result.converged_last,
            result.convergence_round_last,
            result.publishers_welfare_last,
            result.users_welfare_last,
            result.max_improvement_last,
            exposure_initial,
            exposure_feasibility_violation(game.calc_r_all()),
        )

    raise ValueError(f"Unknown dynamics: {job.dynamics}")


def append_row(path: Path, row: SuiteRow) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    row_dict = asdict(row)
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row_dict.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)


def completed_keys(path: Path) -> set[tuple]:
    if not path.exists():
        return set()
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        return {
            (
                row["experiment_name"],
                row["sweep_parameter"],
                float(row["sweep_value"]),
                row["dynamics"],
                row["activation_family"],
                float(row["activation_param_value"]),
                int(float(row["seed"])),
            )
            for row in reader
        }


def job_key(job: SuiteJob) -> tuple:
    return (
        job.experiment_name,
        job.sweep_parameter,
        float(job.sweep_value),
        job.dynamics,
        job.activation_family,
        float(job.activation_param_value),
        int(job.seed),
    )


def run_jobs(jobs: list[SuiteJob], output_path: Path, max_workers: int) -> Path:
    done = completed_keys(output_path)
    pending = [job for job in jobs if job_key(job) not in done]
    total = len(jobs)
    completed = total - len(pending)
    print(f"starting {len(pending)} pending jobs ({completed}/{total} already complete)")

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_job = {executor.submit(run_suite_job, job): job for job in pending}
        for future in as_completed(future_to_job):
            row = future.result()
            append_row(output_path, row)
            completed += 1
            if completed % 50 == 0 or completed == total:
                print(f"completed {completed}/{total}")
    return output_path


def base_job(
    research_question: str,
    experiment_name: str,
    sweep_parameter: str,
    sweep_value: float,
    dynamics: str,
    activation_spec: ActivationSpec,
    seed: int,
    *,
    lam: float = 0.5,
    n: int = 3,
    s: int = 3,
    k: int = 3,
    beta: float = 1.0,
    brd_rounds: int = 1000,
    nrd_rounds: int = 1000,
    nrd_check_every: int = 100,
    eta: float = 0.5,
    eps: float = 1e-4,
) -> SuiteJob:
    return SuiteJob(
        research_question,
        experiment_name,
        sweep_parameter,
        sweep_value,
        dynamics,
        activation_spec.family,
        activation_spec.param_name,
        activation_spec.param_value,
        lam,
        n,
        s,
        k,
        beta,
        seed,
        brd_rounds,
        nrd_rounds,
        nrd_check_every,
        eta,
        eps,
    )


def between_family_jobs(
    dynamics: str,
    seeds: int,
    brd_rounds: int,
    nrd_rounds: int,
    nrd_check_every: int,
) -> list[SuiteJob]:
    rq = "How do ecosystem parameters affect welfare and convergence across stable activation families?"
    activations = default_activation_specs()
    jobs: list[SuiteJob] = []
    sweeps = {
        "lambda": np.round(np.arange(0.0, 1.01, 0.1), 1).tolist(),
        "n": [2, 3, 5, 10, 20],
        "s": [1, 2, 3, 5, 10, 20],
        "k": [1, 2, 3, 5, 10, 20],
    }
    for parameter, values in sweeps.items():
        for value in values:
            for activation in activations:
                for seed in range(seeds):
                    kwargs = {"lam": 0.5, "n": 3, "s": 3, "k": 3}
                    if parameter == "lambda":
                        kwargs["lam"] = float(value)
                    elif parameter == "n":
                        kwargs["n"] = int(value)
                    elif parameter == "s":
                        kwargs["s"] = int(value)
                    elif parameter == "k":
                        kwargs["k"] = int(value)
                    jobs.append(
                        base_job(
                            rq,
                            f"between_family_{parameter}",
                            parameter,
                            float(value),
                            dynamics,
                            activation,
                            seed,
                            brd_rounds=brd_rounds,
                            nrd_rounds=nrd_rounds,
                            nrd_check_every=nrd_check_every,
                            **kwargs,
                        )
                    )
    return jobs


def hyperparameter_jobs(
    dynamics: str,
    seeds: int,
    brd_rounds: int,
    nrd_rounds: int,
    nrd_check_every: int,
) -> list[SuiteJob]:
    rq = "How do within-family activation hyperparameters tune the welfare-convergence tradeoff?"
    specs = [
        *[ActivationSpec("root_power", "a", float(a)) for a in np.round(np.arange(0.0, 1.0, 0.1), 1)],
        *[ActivationSpec("shifted_linear", "b", float(b)) for b in [1.00001, 1.1, 1.5, 2, 3, 5, 7, 10]],
        *[ActivationSpec("shifted_log", "c", float(c)) for c in [2.00001, 2.1, 2.5, 3, 4, 5, 7, 10]],
    ]
    jobs: list[SuiteJob] = []
    for spec in specs:
        for seed in range(seeds):
            jobs.append(
                base_job(
                    rq,
                    f"hyperparameter_{spec.family}",
                    spec.param_name,
                    spec.param_value,
                    dynamics,
                    spec,
                    seed,
                    brd_rounds=brd_rounds,
                    nrd_rounds=nrd_rounds,
                    nrd_check_every=nrd_check_every,
                )
            )
    return jobs


def convergence_comparison_jobs(
    seeds: int,
    brd_rounds: int,
    nrd_rounds: int,
    nrd_check_every: int,
) -> list[SuiteJob]:
    rq = "Are activations that converge quickly under BRD also fast under NRD?"
    jobs: list[SuiteJob] = []
    specs = default_activation_specs()
    for dynamics in ["brd", "nrd"]:
        for spec in specs:
            for seed in range(seeds):
                jobs.append(
                    base_job(
                        rq,
                        "convergence_comparison_default",
                        "activation",
                        spec.param_value,
                        dynamics,
                        spec,
                        seed,
                        brd_rounds=brd_rounds,
                        nrd_rounds=nrd_rounds,
                        nrd_check_every=nrd_check_every,
                    )
                )
    return jobs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--suite",
        choices=["between", "hyperparameters", "convergence"],
        required=True,
    )
    parser.add_argument("--dynamics", choices=["brd", "nrd"], default="brd")
    parser.add_argument("--seeds", type=int, default=100)
    parser.add_argument("--brd-rounds", type=int, default=1000)
    parser.add_argument("--nrd-rounds", type=int, default=5000)
    parser.add_argument("--nrd-check-every", type=int, default=100)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.suite == "between":
        jobs = between_family_jobs(
            args.dynamics,
            args.seeds,
            args.brd_rounds,
            args.nrd_rounds,
            args.nrd_check_every,
        )
    elif args.suite == "hyperparameters":
        jobs = hyperparameter_jobs(
            args.dynamics,
            args.seeds,
            args.brd_rounds,
            args.nrd_rounds,
            args.nrd_check_every,
        )
    else:
        jobs = convergence_comparison_jobs(
            args.seeds,
            args.brd_rounds,
            args.nrd_rounds,
            args.nrd_check_every,
        )

    run_jobs(jobs, args.output, args.workers)
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
