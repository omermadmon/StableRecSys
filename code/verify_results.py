from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def group_counts(rows: list[dict[str, str]]) -> dict[tuple, int]:
    counts = defaultdict(int)
    for row in rows:
        key = (
            row.get("experiment_name", ""),
            row.get("sweep_parameter", ""),
            row.get("sweep_value", ""),
            row.get("dynamics", ""),
            row.get("activation_family", ""),
            row.get("activation_param_value", ""),
        )
        counts[key] += 1
    return dict(counts)


def convergence_column(target: str) -> str:
    return "last_converged" if target == "last" else "converged"


def verify(path: Path, min_seeds: int | None, require_converged: bool, target: str) -> None:
    rows = read_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")

    counts = group_counts(rows)
    min_count = min(counts.values())
    max_count = max(counts.values())
    failed_groups = {key: count for key, count in counts.items() if min_seeds and count < min_seeds}
    conv_col = convergence_column(target)
    failed_convergence = [row for row in rows if row.get(conv_col) != "True"]
    exposure_columns = [
        col
        for col in ["exposure_violation_terminal", "exposure_violation_brd_terminal", "exposure_violation_nrd_last"]
        if col in rows[0]
    ]
    max_exposure_violation = 0.0
    for col in exposure_columns:
        max_exposure_violation = max(max_exposure_violation, max(float(row[col]) for row in rows))

    print(f"file: {path}")
    print(f"rows: {len(rows)}")
    print(f"groups: {len(counts)}")
    print(f"samples per group: min={min_count}, max={max_count}")
    print(f"max exposure violation: {max_exposure_violation:.3e}")
    failed_pct = 100 * len(failed_convergence) / len(rows)
    print(f"convergence target: {target}")
    print(f"converged rows: {len(rows) - len(failed_convergence)}/{len(rows)}")
    if failed_convergence:
        print(f"WARNING: {len(failed_convergence)} rows did not converge ({failed_pct:.1f}%).")

    if failed_groups:
        examples = list(failed_groups.items())[:5]
        raise ValueError(f"{len(failed_groups)} groups have fewer than {min_seeds} rows. Examples: {examples}")
    if require_converged and failed_convergence:
        raise ValueError(f"{len(failed_convergence)} rows did not converge for target={target}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--min-seeds", type=int)
    parser.add_argument("--require-converged", action="store_true")
    parser.add_argument("--target", choices=["average", "last"], default="average")
    args = parser.parse_args()
    verify(args.input, args.min_seeds, args.require_converged, args.target)


if __name__ == "__main__":
    main()
