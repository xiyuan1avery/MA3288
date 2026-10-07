#!/usr/bin/env python3
"""Audit formulas, invariants, and repeatability across three saved runs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


EPSILONS = (0.00, 0.02, 0.05, 0.06, 0.09, 0.12, 0.14, 0.19, 0.235)
ALGORITHMS = (
    "UCB1",
    "epsilon-UCB1",
    "alpha-UCB1-1.1",
    "alpha-UCB1-2",
)
HORIZONS = (
    1000, 2000, 5000, 10000, 20000,
    50000, 100000, 200000, 500000, 1000000,
)
GAPS = np.array([0.00, 0.05, 0.13, 0.18, 0.23])
K = 5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dirs", nargs=3, type=Path, required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def close(a: float, b: float, tolerance: float = 1e-8) -> bool:
    return abs(a - b) <= tolerance * max(1.0, abs(a), abs(b))


def expected_coefficient(epsilon: float, algorithm: str) -> float:
    bad = GAPS[GAPS > epsilon]
    if not bad.size:
        return 0.0
    if algorithm == "UCB1":
        return float(4.0 * np.sum(1.0 / bad))
    base = float(np.sum(bad / (bad + epsilon / 2.0) ** 2))
    if algorithm == "epsilon-UCB1":
        return 4.0 * base
    if algorithm == "alpha-UCB1-1.1":
        return 8.0 * 1.1 * base
    if algorithm == "alpha-UCB1-2":
        return 8.0 * 2.0 * base
    raise KeyError(algorithm)


def audit_run(run_dir: Path) -> tuple[list[str], dict[str, object]]:
    failures: list[str] = []
    config = json.loads((run_dir / "config.json").read_text())
    replications = int(config["replications"])
    raw_path = run_dir / "data" / "replicate_regret.csv"
    diagnostic_path = (
        run_dir / "data" / "certificate_diagnostics.csv"
    )
    raw = read_csv(raw_path)
    diagnostics = read_csv(diagnostic_path)
    performance = read_csv(run_dir / "all_algorithms_performance.csv")

    expected_rows = (
        replications * len(EPSILONS) * len(ALGORITHMS) * len(HORIZONS)
    )
    if len(raw) != expected_rows:
        failures.append(
            f"raw row count {len(raw)} != {expected_rows}"
        )
    if len(diagnostics) != replications * len(EPSILONS):
        failures.append("diagnostic row count is incorrect")

    regrets: dict[tuple[int, str, int, float], float] = {}
    for row in raw:
        key = (
            int(row["replication"]),
            row["algorithm"],
            int(row["horizon"]),
            round(float(row["epsilon"]), 8),
        )
        value = float(row["regret"])
        if value < -1e-10 or not np.isfinite(value):
            failures.append(f"invalid regret at {key}: {value}")
        regrets[key] = value

    for rep in range(replications):
        for algorithm in ALGORITHMS:
            for horizon in HORIZONS:
                sequence = [
                    regrets[(rep, algorithm, horizon, round(e, 8))]
                    for e in EPSILONS
                ]
                if algorithm == "UCB1":
                    for left, right in zip(sequence, sequence[1:]):
                        if right > left + 1e-8:
                            failures.append(
                                "UCB1 pathwise regret is not monotone "
                                f"for rep={rep}, horizon={horizon}"
                            )
                            break
                if abs(sequence[-1]) > 1e-10:
                    failures.append(
                        "epsilon=0.235 regret is not identically zero "
                        f"for rep={rep}, algorithm={algorithm}, "
                        f"horizon={horizon}"
                    )

    for row in performance:
        epsilon = float(row["epsilon"])
        algorithm = row["algorithm"]
        reported = float(row["theory_reference_coefficient"])
        expected = expected_coefficient(epsilon, algorithm)
        if not close(reported, expected):
            failures.append(
                f"theory coefficient mismatch: eps={epsilon}, "
                f"algorithm={algorithm}, {reported} != {expected}"
            )
        expected_good = int(np.sum(GAPS <= epsilon))
        if int(row["good_arm_count"]) != expected_good:
            failures.append(
                f"good-arm count mismatch at epsilon={epsilon}"
            )
        status = row["theory_reference_status"]
        if algorithm.startswith("alpha") and status != "proved":
            failures.append("alpha-UCB1 coefficient is not labelled proved")

    for row in diagnostics:
        epsilon = float(row["epsilon"])
        found = bool(int(row["certificate_found"]))
        time = int(row["certificate_time"])
        arm = int(row["committed_arm"])
        good = bool(int(row["committed_is_good"]))
        if found:
            if not (K <= time < HORIZONS[-1]):
                failures.append(
                    f"invalid certificate time: {time}"
                )
            if arm not in range(1, K + 1):
                failures.append(f"invalid committed arm: {arm}")
            expected_good = GAPS[arm - 1] <= epsilon
            if good != expected_good:
                failures.append(
                    "committed_is_good disagrees with hard threshold"
                )
        elif time != -1 or arm != -1 or good:
            failures.append(
                "no-certificate diagnostic contains commitment data"
            )

    summary = {
        "run": run_dir.name,
        "seed": config["seed"],
        "replications": replications,
        "raw_rows": len(raw),
        "diagnostic_rows": len(diagnostics),
        "status": "PASS" if not failures else "FAIL",
    }
    return failures, summary


def collect_repeatability(
    run_dirs: list[Path],
) -> list[dict[str, object]]:
    by_key: dict[tuple[float, str], list[tuple[str, float, float]]] = {}
    for run_dir in run_dirs:
        rows = read_csv(run_dir / "all_algorithms_performance.csv")
        for row in rows:
            key = (float(row["epsilon"]), row["algorithm"])
            by_key.setdefault(key, []).append(
                (
                    run_dir.name,
                    float(row["mean_regret_n1e6"]),
                    float(row["empirical_log_coefficient"]),
                )
            )

    output: list[dict[str, object]] = []
    for (epsilon, algorithm), values in sorted(by_key.items()):
        regrets = np.array([value[1] for value in values])
        coefficients = np.array([value[2] for value in values])
        mean_regret = float(regrets.mean())
        mean_coefficient = float(coefficients.mean())
        output.append(
            {
                "epsilon": epsilon,
                "algorithm": algorithm,
                "run_1_mean_regret_n1e6": regrets[0],
                "run_2_mean_regret_n1e6": regrets[1],
                "run_3_mean_regret_n1e6": regrets[2],
                "mean_across_runs_regret_n1e6": mean_regret,
                "relative_spread_regret": (
                    float((regrets.max() - regrets.min()) / mean_regret)
                    if abs(mean_regret) > 1e-12
                    else 0.0
                ),
                "run_1_empirical_coefficient": coefficients[0],
                "run_2_empirical_coefficient": coefficients[1],
                "run_3_empirical_coefficient": coefficients[2],
                "mean_across_runs_empirical_coefficient": mean_coefficient,
                "relative_spread_coefficient": (
                    float(
                        (coefficients.max() - coefficients.min())
                        / abs(mean_coefficient)
                    )
                    if abs(mean_coefficient) > 1e-12
                    else 0.0
                ),
            }
        )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_failures: dict[str, list[str]] = {}
    run_summaries: list[dict[str, object]] = []
    for run_dir in args.run_dirs:
        failures, summary = audit_run(run_dir)
        all_failures[run_dir.name] = failures
        run_summaries.append(summary)

    repeatability = collect_repeatability(args.run_dirs)
    write_csv(
        args.output_dir / "repeatability_audit.csv",
        repeatability,
    )
    write_csv(
        args.output_dir / "run_invariant_audit.csv",
        run_summaries,
    )

    primary = [
        row for row in repeatability
        if close(float(row["epsilon"]), 0.09)
    ]
    maximum_primary_regret_spread = max(
        float(row["relative_spread_regret"]) for row in primary
    )
    report_lines = [
        "# Revised experiment audit",
        "",
        "## Formula and invariant checks",
        "",
    ]
    for summary in run_summaries:
        report_lines.append(
            f"- {summary['run']}: {summary['status']} "
            f"(seed {summary['seed']}, "
            f"{summary['replications']} replications)."
        )
        for failure in all_failures[summary["run"]]:
            report_lines.append(f"  - {failure}")
    report_lines.extend(
        [
            "",
            "Checks include row completeness, nonnegative regret, "
            "pathwise UCB1 monotonicity in epsilon, exact zero regret "
            "at epsilon=0.235, hard-threshold good-set membership, "
            "certificate diagnostic consistency, analytic coefficient "
            "recalculation, and proved labelling of the alpha-UCB1 "
            "coefficient.",
            "",
            "## Repeatability",
            "",
            "Three runs use distinct base seeds and common random numbers "
            "within each run.",
            (
                "Maximum relative spread of mean R(10^6) among the four "
                "algorithms at epsilon=0.09: "
                f"{100 * maximum_primary_regret_spread:.2f}%."
            ),
            "",
            "See repeatability_audit.csv for every epsilon and algorithm.",
            "",
        ]
    )
    (args.output_dir / "AUDIT.md").write_text(
        "\n".join(report_lines)
    )

    any_failure = any(all_failures.values())
    print("Audit status:", "FAIL" if any_failure else "PASS")
    print(
        "Maximum primary regret spread:",
        f"{100 * maximum_primary_regret_spread:.2f}%",
    )
    if any_failure:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
