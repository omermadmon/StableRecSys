from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


LABELS = {
    "shifted_linear": "Linear",
    "root_power": "Root",
    "shifted_log": "Logarithmic",
}
COLORS = {
    "shifted_linear": "#ff7f0e",
    "root_power": "#1f77b4",
    "shifted_log": "#2ca02c",
}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def mean_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    if len(arr) <= 1:
        return mean, mean, mean
    se = float(np.std(arr, ddof=1) / np.sqrt(len(arr)))
    return mean, mean - 1.96 * se, mean + 1.96 * se


def group_summary(rows, x_key: str, metric: str):
    grouped = defaultdict(list)
    for row in rows:
        if row[metric] == "":
            continue
        key = (row["activation_family"], float(row[x_key]))
        grouped[key].append(float(row[metric]))
    return {key: mean_ci(vals) for key, vals in grouped.items()}


def convergence_column(target: str) -> str:
    return "last_converged" if target == "last" else "converged"


def convergence_step_column(target: str) -> str:
    return "last_convergence_step" if target == "last" else "convergence_step"


def improvement_column(dynamics: str, target: str) -> str:
    if dynamics != "nrd":
        return "max_improvement"
    return "last_max_improvement" if target == "last" else "max_improvement"


def assert_all_converged(rows: list[dict[str, str]], context: str, target: str) -> None:
    col = convergence_column(target)
    failed = [row for row in rows if row.get(col) != "True"]
    if failed:
        raise ValueError(
            f"{context}: {len(failed)}/{len(rows)} rows did not converge. "
            "Increase the horizon, use --drop-nonconverged for convergence-rate plots, "
            "or omit --require-converged to plot terminal improvement diagnostics."
        )


def drop_nonconverged(
    rows: list[dict[str, str]],
    context: str,
    target: str,
    require_converged: bool,
    drop_nonconverged_rows: bool,
) -> list[dict[str, str]]:
    col = convergence_column(target)
    failed = [row for row in rows if row.get(col) != "True"]
    if not failed:
        return rows
    if require_converged:
        assert_all_converged(rows, context, target)
    if drop_nonconverged_rows:
        pct = 100 * len(failed) / len(rows)
        print(f"WARNING: {context}: dropping {len(failed)}/{len(rows)} non-converged rows ({pct:.1f}%).")
        return [row for row in rows if row.get(col) == "True"]
    return rows


def convergence_metric(dynamics: str, use_convergence_steps: bool, target: str) -> tuple[str, str]:
    if dynamics != "nrd" or use_convergence_steps:
        label = "Rounds to convergence"
        if dynamics == "nrd":
            label += f" ({target} profile)"
        return convergence_step_column(target), label
    return (
        improvement_column(dynamics, target) if dynamics == "nrd" else "convergence_step",
        f"Terminal improvement ({target})" if dynamics == "nrd" else "BRD updates",
    )


def plot_between(
    input_path: Path,
    output_dir: Path,
    dynamics: str,
    require_converged: bool,
    drop_nonconverged_rows: bool,
    nrd_target: str,
) -> None:
    rows = [row for row in read_rows(input_path) if row["dynamics"] == dynamics]
    target = nrd_target if dynamics == "nrd" else "average"
    use_convergence_steps = dynamics != "nrd" or require_converged or drop_nonconverged_rows
    rate_rows = drop_nonconverged(
        rows,
        f"between-family {dynamics}",
        target,
        require_converged,
        drop_nonconverged_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rate_metric = convergence_metric(dynamics, use_convergence_steps, target)
    metrics = [
        ("publisher_welfare", "Creator welfare", rows),
        ("user_welfare", "User welfare", rows),
        (*rate_metric, rate_rows),
    ]
    for parameter in ["lambda", "n", "s", "k"]:
        subset = [row for row in rows if row["sweep_parameter"] == parameter]
        if not subset:
            continue
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
        for ax, (metric, title, metric_rows) in zip(axes, metrics):
            metric_subset = [row for row in metric_rows if row["sweep_parameter"] == parameter]
            summary = group_summary(metric_subset, "sweep_value", metric)
            for family in ["shifted_linear", "root_power", "shifted_log"]:
                xs = sorted({x for fam, x in summary if fam == family})
                if not xs:
                    continue
                means = [summary[(family, x)][0] for x in xs]
                lows = [summary[(family, x)][1] for x in xs]
                highs = [summary[(family, x)][2] for x in xs]
                yerr = [np.array(means) - np.array(lows), np.array(highs) - np.array(means)]
                ax.errorbar(xs, means, yerr=yerr, marker="o", linewidth=2, capsize=3,
                            label=LABELS[family], color=COLORS[family])
            ax.set_title(title)
            ax.set_xlabel(r"$\lambda$" if parameter == "lambda" else parameter)
            ax.grid(alpha=0.25)
        axes[0].legend(frameon=False)
        fig.tight_layout()
        out = output_dir / f"between_{parameter}_{dynamics}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"wrote {out}")


def plot_hyperparameters(
    input_path: Path,
    output_dir: Path,
    dynamics: str,
    require_converged: bool,
    drop_nonconverged_rows: bool,
    nrd_target: str,
) -> None:
    rows = [row for row in read_rows(input_path) if row["dynamics"] == dynamics]
    target = nrd_target if dynamics == "nrd" else "average"
    use_convergence_steps = dynamics != "nrd" or require_converged or drop_nonconverged_rows
    rate_rows = drop_nonconverged(
        rows,
        f"hyperparameter {dynamics}",
        target,
        require_converged,
        drop_nonconverged_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    rate_metric = convergence_metric(dynamics, use_convergence_steps, target)
    metrics = [
        ("publisher_welfare", "Creator welfare", rows),
        ("user_welfare", "User welfare", rows),
        (*rate_metric, rate_rows),
    ]
    for family in ["root_power", "shifted_linear", "shifted_log"]:
        subset = [row for row in rows if row["activation_family"] == family]
        if not subset:
            continue
        x_label = subset[0]["activation_param_name"]
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), sharex=True)
        for ax, (metric, title, metric_rows) in zip(axes, metrics):
            grouped = defaultdict(list)
            for row in metric_rows:
                if row["activation_family"] != family:
                    continue
                if row[metric] != "":
                    grouped[float(row["activation_param_value"])].append(float(row[metric]))
            xs = sorted(grouped)
            means, lows, highs = [], [], []
            for x in xs:
                mean, low, high = mean_ci(grouped[x])
                means.append(mean)
                lows.append(low)
                highs.append(high)
            yerr = [np.array(means) - np.array(lows), np.array(highs) - np.array(means)]
            ax.errorbar(xs, means, yerr=yerr, marker="o", linewidth=2, capsize=3,
                        color=COLORS[family])
            ax.set_title(title)
            ax.set_xlabel(x_label)
            ax.grid(alpha=0.25)
        fig.tight_layout()
        out = output_dir / f"hyperparameter_{family}_{dynamics}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"wrote {out}")


def plot_convergence_comparison(
    input_path: Path,
    output_dir: Path,
    require_converged: bool,
    drop_nonconverged_rows: bool,
    nrd_target: str,
) -> None:
    rows = read_rows(input_path)
    brd_rows = [row for row in rows if row["dynamics"] == "brd"]
    nrd_rows = [row for row in rows if row["dynamics"] == "nrd"]
    brd_rows = drop_nonconverged(
        brd_rows, "BRD convergence comparison", "average", require_converged, drop_nonconverged_rows
    )
    nrd_rows = drop_nonconverged(
        nrd_rows, "NRD convergence comparison", nrd_target, require_converged, drop_nonconverged_rows
    )
    rows = brd_rows + nrd_rows
    output_dir.mkdir(parents=True, exist_ok=True)
    grouped = defaultdict(list)
    for row in rows:
        if row["dynamics"] == "brd":
            metric = "convergence_step"
        elif require_converged or drop_nonconverged_rows:
            metric = convergence_step_column(nrd_target)
        else:
            metric = improvement_column("nrd", nrd_target)
        if row[metric] == "":
            continue
        grouped[(row["activation_family"], row["dynamics"])].append(float(row[metric]))

    families = ["shifted_linear", "root_power", "shifted_log"]
    brd_means = [mean_ci(grouped[(fam, "brd")])[0] for fam in families]
    nrd_means = [mean_ci(grouped[(fam, "nrd")])[0] if grouped[(fam, "nrd")] else np.nan for fam in families]

    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.scatter(brd_means, nrd_means, s=90, color=[COLORS[f] for f in families])
    for fam, x, y in zip(families, brd_means, nrd_means):
        ax.annotate(LABELS[fam], (x, y), textcoords="offset points", xytext=(6, 5))
    ax.set_xlabel("Mean BRD updates")
    ax.set_ylabel(
        f"Mean NRD rounds to convergence ({nrd_target} profile)"
        if require_converged or drop_nonconverged_rows
        else f"Mean NRD terminal improvement ({nrd_target})"
    )
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out = output_dir / "convergence_brd_vs_nrd.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")


def plot_metric_lines(
    ax,
    rows: list[dict[str, str]],
    x_key: str,
    metric: str,
    families: list[str],
) -> None:
    summary = group_summary(rows, x_key, metric)
    for family in families:
        xs = sorted({x for fam, x in summary if fam == family})
        if not xs:
            continue
        means = [summary[(family, x)][0] for x in xs]
        lows = [summary[(family, x)][1] for x in xs]
        highs = [summary[(family, x)][2] for x in xs]
        yerr = [np.array(means) - np.array(lows), np.array(highs) - np.array(means)]
        ax.errorbar(
            xs,
            means,
            yerr=yerr,
            marker="o",
            linewidth=2,
            capsize=3,
            label=LABELS[family],
            color=COLORS[family],
        )


def sweep_x_label(parameter: str) -> str:
    return r"$\lambda$" if parameter == "lambda" else parameter


def apply_sweep_ticks(ax, parameter: str) -> None:
    if parameter in {"n", "s", "k"}:
        ax.set_xticks([5, 10, 15, 20])


def plot_combined_sweep(
    brd_rows: list[dict[str, str]],
    nrd_rows: list[dict[str, str]],
    output_dir: Path,
    parameter: str,
    title: str,
) -> None:
    families = ["shifted_linear", "root_power", "shifted_log"]
    brd_subset = [row for row in brd_rows if row["sweep_parameter"] == parameter]
    nrd_subset = [row for row in nrd_rows if row["sweep_parameter"] == parameter]
    panels = [
        ("publisher_welfare", "Creator welfare", brd_subset),
        ("user_welfare", "User welfare", brd_subset),
        ("convergence_step", "BRD rounds", brd_subset),
        ("convergence_step", "NRD rounds", nrd_subset),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(12.8, 2.85), sharex=True)
    for ax, (metric, panel_title, rows) in zip(axes, panels):
        plot_metric_lines(ax, rows, "sweep_value", metric, families)
        ax.set_title(panel_title, fontsize=9)
        ax.set_xlabel(sweep_x_label(parameter))
        apply_sweep_ticks(ax, parameter)
        ax.tick_params(axis="both", labelsize=8)
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(w_pad=0.9)
    out = output_dir / f"sweep_{parameter}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
    print(f"wrote {out}")
    plt.close(fig)


def plot_paper_ready_between(
    brd_input: Path,
    nrd_input: Path,
    output_dir: Path,
) -> None:
    brd_rows = [row for row in read_rows(brd_input) if row["dynamics"] == "brd"]
    nrd_rows = [row for row in read_rows(nrd_input) if row["dynamics"] == "nrd"]
    assert_all_converged(brd_rows, "paper-ready BRD figures", "average")
    nrd_rate_rows = drop_nonconverged(
        nrd_rows,
        "paper-ready NRD average-profile figures",
        "average",
        require_converged=False,
        drop_nonconverged_rows=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    for parameter, title in [
        ("lambda", "Effect of penalty factor"),
        ("n", "Effect of number of publishers"),
        ("s", "Effect of demand support size"),
        ("k", "Effect of embedding dimension"),
    ]:
        plot_combined_sweep(brd_rows, nrd_rate_rows, output_dir, parameter, title)


def plot_paper_ready_hyperparameters(
    brd_input: Path,
    nrd_input: Path,
    output_dir: Path,
) -> None:
    brd_rows = [row for row in read_rows(brd_input) if row["dynamics"] == "brd"]
    nrd_rows = [row for row in read_rows(nrd_input) if row["dynamics"] == "nrd"]
    assert_all_converged(brd_rows, "paper-ready BRD hyperparameter figures", "average")
    nrd_rate_rows = drop_nonconverged(
        nrd_rows,
        "paper-ready NRD average-profile hyperparameter figures",
        "average",
        require_converged=False,
        drop_nonconverged_rows=True,
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for family in ["shifted_linear", "root_power", "shifted_log"]:
        family_brd = [row for row in brd_rows if row["activation_family"] == family]
        family_nrd = [row for row in nrd_rate_rows if row["activation_family"] == family]
        parameter = family_brd[0]["activation_param_name"]
        panels = [
            ("publisher_welfare", "Creator welfare", family_brd),
            ("user_welfare", "User welfare", family_brd),
            ("convergence_step", "BRD rounds", family_brd),
            ("convergence_step", "NRD rounds", family_nrd),
        ]
        fig, axes = plt.subplots(1, 4, figsize=(12.8, 2.85), sharex=True)
        for ax, (metric, panel_title, rows) in zip(axes, panels):
            grouped = defaultdict(list)
            for row in rows:
                if row[metric] != "":
                    grouped[float(row["activation_param_value"])].append(float(row[metric]))
            xs = sorted(grouped)
            means, lows, highs = [], [], []
            for x in xs:
                mean, low, high = mean_ci(grouped[x])
                means.append(mean)
                lows.append(low)
                highs.append(high)
            yerr = [np.array(means) - np.array(lows), np.array(highs) - np.array(means)]
            ax.errorbar(xs, means, yerr=yerr, marker="o", linewidth=2, capsize=3, color=COLORS[family])
            ax.set_title(panel_title, fontsize=9)
            ax.set_xlabel(parameter)
            ax.tick_params(axis="both", labelsize=8)
            ax.grid(alpha=0.25)
        fig.tight_layout(w_pad=0.9)
        output_name = LABELS[family].lower()
        out = output_dir / f"hyperparameter_{output_name}.pdf"
        fig.savefig(out, bbox_inches="tight")
        fig.savefig(out.with_suffix(".png"), dpi=300, bbox_inches="tight")
        print(f"wrote {out}")
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path)
    parser.add_argument("--brd-input", type=Path)
    parser.add_argument("--nrd-input", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--kind",
        choices=["between", "hyperparameters", "convergence", "paper-ready-between", "paper-ready-hyperparameters"],
        required=True,
    )
    parser.add_argument("--dynamics", choices=["brd", "nrd"], default="brd")
    parser.add_argument(
        "--require-converged",
        action="store_true",
        help="Fail if any row did not converge and plot convergence_step as the rate metric.",
    )
    parser.add_argument(
        "--drop-nonconverged",
        action="store_true",
        help="Drop non-converged rows from convergence-rate plots and print a warning.",
    )
    parser.add_argument(
        "--nrd-target",
        choices=["average", "last"],
        default="average",
        help="For NRD, use average-profile or last-iterate convergence diagnostics.",
    )
    args = parser.parse_args()

    if args.kind == "paper-ready-between":
        if args.brd_input is None or args.nrd_input is None:
            parser.error("paper-ready-between requires --brd-input and --nrd-input")
        plot_paper_ready_between(args.brd_input, args.nrd_input, args.output_dir)
    elif args.kind == "paper-ready-hyperparameters":
        if args.brd_input is None or args.nrd_input is None:
            parser.error("paper-ready-hyperparameters requires --brd-input and --nrd-input")
        plot_paper_ready_hyperparameters(args.brd_input, args.nrd_input, args.output_dir)
    elif args.input is None:
        parser.error("--input is required for this plot kind")
    elif args.kind == "between":
        plot_between(
            args.input,
            args.output_dir,
            args.dynamics,
            args.require_converged,
            args.drop_nonconverged,
            args.nrd_target,
        )
    elif args.kind == "hyperparameters":
        plot_hyperparameters(
            args.input,
            args.output_dir,
            args.dynamics,
            args.require_converged,
            args.drop_nonconverged,
            args.nrd_target,
        )
    else:
        plot_convergence_comparison(
            args.input,
            args.output_dir,
            args.require_converged,
            args.drop_nonconverged,
            args.nrd_target,
        )


if __name__ == "__main__":
    main()
