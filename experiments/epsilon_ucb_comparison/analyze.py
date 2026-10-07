#!/usr/bin/env python3
"""Analyse the revised Week 7 epsilon sweep without changing old outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


EPSILONS = (0.00, 0.02, 0.05, 0.06, 0.09, 0.12, 0.14, 0.19, 0.235)
ALGORITHMS = (
    "UCB1",
    "epsilon-UCB1",
    "alpha-UCB1-1.1",
    "alpha-UCB1-2",
)
HORIZONS = np.array(
    [1000, 2000, 5000, 10000, 20000, 50000,
     100000, 200000, 500000, 1000000],
    dtype=int,
)
GAPS = np.array([0.00, 0.05, 0.13, 0.18, 0.23])
MAIN_EPSILON = 0.09
TAIL_POINTS = 5
BOOTSTRAP_DRAWS = 1000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--make-figure", action="store_true")
    return parser.parse_args()


def load_regret(path: Path) -> tuple[np.ndarray, int]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")

    n_replications = max(int(row["replication"]) for row in rows) + 1
    data = np.full(
        (n_replications, len(EPSILONS), len(ALGORITHMS), len(HORIZONS)),
        np.nan,
    )
    epsilon_index = {round(value, 8): i for i, value in enumerate(EPSILONS)}
    algorithm_index = {value: i for i, value in enumerate(ALGORITHMS)}
    horizon_index = {int(value): i for i, value in enumerate(HORIZONS)}

    for row in rows:
        data[
            int(row["replication"]),
            epsilon_index[round(float(row["epsilon"]), 8)],
            algorithm_index[row["algorithm"]],
            horizon_index[int(row["horizon"])],
        ] = float(row["regret"])

    if np.isnan(data).any():
        raise ValueError("Regret CSV is incomplete.")
    return data, n_replications


def normal_ci(samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = samples.mean(axis=0)
    if samples.shape[0] < 2:
        return mean, mean
    half_width = (
        1.96 * samples.std(axis=0, ddof=1) / np.sqrt(samples.shape[0])
    )
    return mean - half_width, mean + half_width


def fitted_coefficient(curve: np.ndarray) -> float:
    x = np.log(HORIZONS[-TAIL_POINTS:].astype(float))
    return float(np.polyfit(x, curve[-TAIL_POINTS:], deg=1)[0])


def theoretical_reference(
    epsilon: float, algorithm: str
) -> tuple[float, str]:
    selected_gaps = GAPS[GAPS > epsilon]
    if algorithm == "UCB1":
        coefficient = (
            4.0 * np.sum(1.0 / selected_gaps)
            if selected_gaps.size
            else 0.0
        )
        return float(coefficient), "exact asymptotic"

    base = (
        np.sum(selected_gaps / (selected_gaps + epsilon / 2.0) ** 2)
        if selected_gaps.size
        else 0.0
    )
    if algorithm == "epsilon-UCB1":
        return float(4.0 * base), "proved upper coefficient"
    if algorithm == "alpha-UCB1-1.1":
        return float(8.0 * 1.1 * base), "proposed target, not proved"
    if algorithm == "alpha-UCB1-2":
        return float(8.0 * 2.0 * base), "proposed target, not proved"
    raise KeyError(algorithm)


def bootstrap_intervals(
    data: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed + 7417)
    n_replications = data.shape[0]
    coefficient_draws = np.full(
        (BOOTSTRAP_DRAWS, len(EPSILONS), len(ALGORITHMS)),
        np.nan,
    )
    reduction_draws = np.full_like(coefficient_draws, np.nan)

    for draw in range(BOOTSTRAP_DRAWS):
        indices = rng.integers(0, n_replications, size=n_replications)
        boot_mean = data[indices].mean(axis=0)
        for e in range(len(EPSILONS)):
            baseline = boot_mean[e, 0, -1]
            for a in range(len(ALGORITHMS)):
                coefficient_draws[draw, e, a] = fitted_coefficient(
                    boot_mean[e, a]
                )
                if baseline > 0:
                    reduction_draws[draw, e, a] = (
                        100.0
                        * (baseline - boot_mean[e, a, -1])
                        / baseline
                    )

    coefficient_ci = np.percentile(
        coefficient_draws, [2.5, 97.5], axis=0
    )
    reduction_ci = np.full(
        (2, len(EPSILONS), len(ALGORITHMS)), np.nan
    )
    for e in range(len(EPSILONS)):
        for a in range(len(ALGORITHMS)):
            finite = reduction_draws[:, e, a]
            finite = finite[np.isfinite(finite)]
            if finite.size:
                reduction_ci[:, e, a] = np.percentile(
                    finite, [2.5, 97.5]
                )
    return coefficient_draws, coefficient_ci, reduction_draws, reduction_ci


def build_performance_rows(
    data: np.ndarray,
    mean: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    coefficient_ci: np.ndarray,
    reduction_ci: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for e, epsilon in enumerate(EPSILONS):
        baseline = mean[e, 0, -1]
        baseline_coefficient = fitted_coefficient(mean[e, 0])
        for a, algorithm in enumerate(ALGORITHMS):
            coefficient = fitted_coefficient(mean[e, a])
            reduction = (
                100.0 * (baseline - mean[e, a, -1]) / baseline
                if baseline > 0
                else np.nan
            )
            theory, status = theoretical_reference(epsilon, algorithm)
            rows.append(
                {
                    "epsilon": epsilon,
                    "good_arm_count": int(np.sum(GAPS <= epsilon)),
                    "bad_arm_count": int(np.sum(GAPS > epsilon)),
                    "algorithm": algorithm,
                    "mean_regret_n1e6": mean[e, a, -1],
                    "regret_ci95_low": low[e, a, -1],
                    "regret_ci95_high": high[e, a, -1],
                    "reduction_vs_ucb1_percent": reduction,
                    "reduction_ci95_low": reduction_ci[0, e, a],
                    "reduction_ci95_high": reduction_ci[1, e, a],
                    "empirical_log_coefficient": coefficient,
                    "coefficient_ci95_low": coefficient_ci[0, e, a],
                    "coefficient_ci95_high": coefficient_ci[1, e, a],
                    "coefficient_ratio_vs_ucb1": (
                        coefficient / baseline_coefficient
                        if abs(baseline_coefficient) > 1e-12
                        else np.nan
                    ),
                    "theory_reference_coefficient": theory,
                    "theory_reference_status": status,
                    "replications": data.shape[0],
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_regret_summary(
    path: Path,
    mean: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> None:
    rows: list[dict[str, object]] = []
    for e, epsilon in enumerate(EPSILONS):
        for a, algorithm in enumerate(ALGORITHMS):
            for h, horizon in enumerate(HORIZONS):
                rows.append(
                    {
                        "epsilon": epsilon,
                        "algorithm": algorithm,
                        "horizon": int(horizon),
                        "mean_regret": mean[e, a, h],
                        "ci95_low": low[e, a, h],
                        "ci95_high": high[e, a, h],
                    }
                )
    write_csv(path, rows)


def diagnostic_summary(path: Path) -> list[dict[str, object]]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped: dict[float, list[dict[str, str]]] = {
        epsilon: [] for epsilon in EPSILONS
    }
    for row in rows:
        grouped[round(float(row["epsilon"]), 8)].append(row)

    summary: list[dict[str, object]] = []
    for epsilon in EPSILONS:
        group = grouped[round(epsilon, 8)]
        found = np.array(
            [int(row["certificate_found"]) for row in group], dtype=bool
        )
        times = np.array(
            [int(row["certificate_time"]) for row in group], dtype=int
        )
        arms = np.array(
            [int(row["committed_arm"]) for row in group], dtype=int
        )
        good = np.array(
            [int(row["committed_is_good"]) for row in group], dtype=bool
        )
        conditional_arms = arms[found]
        summary.append(
            {
                "epsilon": epsilon,
                "replications": len(group),
                "certificate_probability": float(found.mean()),
                "mean_certificate_time_conditional": (
                    float(times[found].mean()) if found.any() else np.nan
                ),
                "median_certificate_time_conditional": (
                    float(np.median(times[found])) if found.any() else np.nan
                ),
                "p_arm1_conditional": (
                    float(np.mean(conditional_arms == 1))
                    if found.any()
                    else np.nan
                ),
                "p_arm2_conditional": (
                    float(np.mean(conditional_arms == 2))
                    if found.any()
                    else np.nan
                ),
                "p_bad_arm_conditional": (
                    float(np.mean(~good[found]))
                    if found.any()
                    else np.nan
                ),
                "post_certificate_zero_regret_rate_conditional": (
                    float(np.mean(good[found]))
                    if found.any()
                    else np.nan
                ),
            }
        )
    return summary


def write_tex_table(
    path: Path, rows: list[dict[str, object]]
) -> None:
    selected = [
        row
        for row in rows
        if row["algorithm"] in ("UCB1", "epsilon-UCB1")
    ]
    lines = [
        r"\begin{tabular}{rrlrrrr}",
        r"\toprule",
        r"$\varepsilon$ & $|\mathcal G_\varepsilon|$ & Algorithm & "
        r"$R(10^6)$ & Reduction & $\widehat C$ & $C_{\rm ref}$ \\",
        r"\midrule",
    ]
    for index, row in enumerate(selected):
        if index > 0 and index % 2 == 0:
            lines.append(r"\midrule")
        reduction = row["reduction_vs_ucb1_percent"]
        reduction_text = (
            "--" if not np.isfinite(float(reduction))
            else f"{float(reduction):.1f}\\%"
        )
        algorithm = (
            "UCB1"
            if row["algorithm"] == "UCB1"
            else r"$\varepsilon$-UCB1"
        )
        lines.append(
            f"{float(row['epsilon']):.3f} & "
            f"{int(row['good_arm_count'])} & {algorithm} & "
            f"{float(row['mean_regret_n1e6']):.1f} & "
            f"{reduction_text} & "
            f"{float(row['empirical_log_coefficient']):.2f} & "
            f"{float(row['theory_reference_coefficient']):.2f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def draw_main_figure(
    output_dir: Path,
    mean: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
) -> None:
    styles = {
        "UCB1": dict(color="#111111", marker="o", linestyle="-"),
        "epsilon-UCB1": dict(color="#0072B2", marker="s", linestyle="-"),
        "alpha-UCB1-1.1": dict(
            color="#009E73", marker="^", linestyle="--"
        ),
        "alpha-UCB1-2": dict(
            color="#D55E00", marker="D", linestyle=":"
        ),
    }
    labels = {
        "UCB1": "UCB1",
        "epsilon-UCB1": r"$\varepsilon$-UCB1",
        "alpha-UCB1-1.1": r"$\alpha$-UCB1 ($\alpha=1.1$)",
        "alpha-UCB1-2": r"$\alpha$-UCB1 ($\alpha=2$)",
    }
    epsilon_index = EPSILONS.index(MAIN_EPSILON)
    early = HORIZONS <= 50000

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.labelsize": 11,
            "axes.titlesize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 160,
        }
    )
    fig, (full_axis, early_axis) = plt.subplots(
        1,
        2,
        figsize=(10.4, 4.4),
        gridspec_kw={"width_ratios": [2.15, 1.0]},
    )

    for axis, mask in (
        (full_axis, np.ones(len(HORIZONS), dtype=bool)),
        (early_axis, early),
    ):
        for a, algorithm in enumerate(ALGORITHMS):
            style = styles[algorithm]
            axis.plot(
                HORIZONS[mask],
                mean[epsilon_index, a, mask],
                label=labels[algorithm],
                linewidth=2.0,
                markersize=5.2,
                markeredgewidth=0.8,
                **style,
            )
            axis.fill_between(
                HORIZONS[mask],
                low[epsilon_index, a, mask],
                high[epsilon_index, a, mask],
                color=style["color"],
                alpha=0.10,
                linewidth=0,
            )
        axis.grid(alpha=0.22, linewidth=0.7)
        axis.set_xlabel(r"Horizon $n$")

    full_axis.set_title("Full horizon")
    full_axis.set_xlim(0, HORIZONS[-1] * 1.02)
    full_axis.set_ylim(
        0, float(np.max(high[epsilon_index])) * 1.08
    )
    formatter = ScalarFormatter(useMathText=True)
    formatter.set_powerlimits((5, 5))
    full_axis.xaxis.set_major_formatter(formatter)
    full_axis.set_ylabel(r"Expected $\varepsilon$-lenient regret")

    early_axis.set_title(r"Early horizon ($n\leq 50{,}000$)")
    early_axis.set_xlim(0, 52000)
    early_axis.set_ylim(
        0, float(np.max(high[epsilon_index, :, early])) * 1.08
    )
    early_axis.ticklabel_format(axis="x", style="plain", useOffset=False)

    handles, legend_labels = full_axis.get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.01),
        ncol=4,
        frameon=False,
    )
    fig.suptitle(
        r"$\varepsilon=0.09,\quad "
        r"\mathcal{G}_\varepsilon=\{1,2\}$",
        y=0.90,
        fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.84))
    fig.savefig(
        output_dir / "regret_vs_horizon_epsilon_0p09.png",
        dpi=300,
        bbox_inches="tight",
    )
    fig.savefig(
        output_dir / "regret_vs_horizon_epsilon_0p09.pdf",
        bbox_inches="tight",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "data").mkdir(parents=True, exist_ok=True)

    data, n_replications = load_regret(args.input)
    mean = data.mean(axis=0)
    low, high = normal_ci(data)
    _, coefficient_ci, _, reduction_ci = bootstrap_intervals(
        data, args.seed
    )
    performance_rows = build_performance_rows(
        data, mean, low, high, coefficient_ci, reduction_ci
    )

    write_regret_summary(
        args.output_dir / "data" / "regret_summary.csv",
        mean,
        low,
        high,
    )
    write_csv(
        args.output_dir / "all_algorithms_performance.csv",
        performance_rows,
    )
    main_table_rows = [
        row
        for row in performance_rows
        if row["algorithm"] in ("UCB1", "epsilon-UCB1")
    ]
    write_csv(
        args.output_dir / "epsilon_sweep_table.csv",
        main_table_rows,
    )
    write_tex_table(
        args.output_dir / "epsilon_sweep_table.tex",
        performance_rows,
    )
    diagnostic_rows = diagnostic_summary(args.diagnostics)
    write_csv(
        args.output_dir / "data" / "certificate_diagnostics_summary.csv",
        diagnostic_rows,
    )

    if args.make_figure:
        draw_main_figure(args.output_dir, mean, low, high)

    config = {
        "replications": n_replications,
        "seed": args.seed,
        "reward_model": "independent unit-variance Gaussian rewards",
        "means": [1.00, 0.95, 0.87, 0.82, 0.77],
        "gaps": GAPS.tolist(),
        "epsilons": list(EPSILONS),
        "horizons": HORIZONS.tolist(),
        "main_epsilon": MAIN_EPSILON,
        "main_good_set": [1, 2],
        "alpha_values": [1.1, 2.0],
        "confidence_exponent_q": 4,
        "coefficient_fit": (
            "OLS slope in R(n)=beta_0+C log(n), using the last five horizons"
        ),
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "theory_status": {
            "UCB1": "exact asymptotic coefficient",
            "epsilon-UCB1": "proved upper coefficient",
            "alpha-UCB1": "proposed target, not proved",
        },
    }
    (args.output_dir / "config.json").write_text(
        json.dumps(config, indent=2) + "\n"
    )
    print(f"Analysed {n_replications} replications.")
    print(f"Wrote outputs to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
