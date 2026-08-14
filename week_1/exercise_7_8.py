"""Empirical comparison of ETC and UCB(delta) for Exercise 7.8.

Run from this folder with:
    python3 exercise_7_8.py
"""

import math
import random

import matplotlib.pyplot as plt

from ucb_algorithms import GaussianBandit, ucb_delta


HORIZON = 1_000
N_SIMULATIONS = 500
UCB_DELTA = 1 / HORIZON
FIXED_EXPLORATION_COUNTS = (25, 50, 75, 100)


def normal_cdf(x):
    """Return the CDF of a standard normal random variable at x."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def etc_expected_regret(horizon, gap, exploration_count):
    """Return the exact expected pseudo-regret of two-armed ETC.

    ETC samples each arm ``exploration_count`` times, then commits to the
    arm with the larger empirical mean.  The arms have means 0 and -gap,
    and Gaussian rewards of variance one.
    """
    if not 1 <= exploration_count <= horizon // 2:
        raise ValueError("exploration_count must be between 1 and horizon // 2.")

    probability_wrong = normal_cdf(-gap * math.sqrt(exploration_count / 2.0))
    return gap * (
        exploration_count
        + (horizon - 2 * exploration_count) * probability_wrong
    )


def optimal_etc_m(horizon, gap):
    """Return the integer exploration count minimizing ETC expected regret."""
    return min(
        range(1, horizon // 2 + 1),
        key=lambda m: etc_expected_regret(horizon, gap, m),
    )


def estimate_ucb_regret(horizon, gap, simulations, delta, seed):
    """Estimate UCB(delta) pseudo-regret and its standard error."""
    master_rng = random.Random(seed)
    regrets = []

    for _ in range(simulations):
        bandit_rng = random.Random(master_rng.randrange(2**63))
        tie_rng = random.Random(master_rng.randrange(2**63))
        bandit = GaussianBandit([0.0, -gap], rng=bandit_rng)
        regrets.append(ucb_delta(bandit, horizon, delta, rng=tie_rng))

    mean_regret = sum(regrets) / simulations
    sample_variance = sum((regret - mean_regret) ** 2 for regret in regrets)
    sample_variance /= simulations - 1
    standard_error = math.sqrt(sample_variance / simulations)
    return mean_regret, standard_error


def reproduce_figure_7_1():
    """Compare ETC variants with UCB(delta) as the gap changes."""
    gaps = [value / 100 for value in range(1, 100)]
    etc_regrets = {m: [] for m in FIXED_EXPLORATION_COUNTS}
    optimal_etc_regrets = []
    ucb_regrets = []

    for index, gap in enumerate(gaps):
        for m in FIXED_EXPLORATION_COUNTS:
            etc_regrets[m].append(etc_expected_regret(HORIZON, gap, m))

        best_m = optimal_etc_m(HORIZON, gap)
        optimal_etc_regrets.append(etc_expected_regret(HORIZON, gap, best_m))

        mean_regret, _ = estimate_ucb_regret(
            HORIZON,
            gap,
            N_SIMULATIONS,
            UCB_DELTA,
            seed=20260814 + index,
        )
        ucb_regrets.append(mean_regret)

    plt.figure(figsize=(8, 5))
    for m in FIXED_EXPLORATION_COUNTS:
        plt.plot(gaps, etc_regrets[m], linewidth=2, label=f"ETC (m = {m})")

    plt.plot(gaps, optimal_etc_regrets, color="gray", linewidth=2, label="ETC (optimal m)")
    plt.plot(gaps, ucb_regrets, color="black", linewidth=2.5, label="UCB(delta)")

    plt.xlabel("Gap Δ")
    plt.ylabel("Estimated expected pseudo-regret")
    plt.title("ETC and UCB(delta) Across Gaussian Bandit Instances")
    plt.legend()
    plt.tight_layout()
    plt.savefig("exercise_7_8_figure_7_1.png", dpi=300)
    plt.show()


def delta_sensitivity_experiment():
    """Measure the practical effect of the confidence parameter delta.

    The bandit instance is fixed at means [0, -0.1].  Smaller delta produces
    a wider confidence bonus, so it generally encourages more exploration.
    """
    confidence_levels = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1]
    means = []
    standard_errors = []

    for index, delta in enumerate(confidence_levels):
        mean_regret, standard_error = estimate_ucb_regret(
            HORIZON,
            gap=0.1,
            simulations=N_SIMULATIONS,
            delta=delta,
            seed=20260900 + index,
        )
        means.append(mean_regret)
        standard_errors.append(standard_error)
        print(
            f"delta = {delta:>8g} | average regret = {mean_regret:6.3f} "
            f"| standard error = {standard_error:5.3f}"
        )

    plt.figure(figsize=(8, 5))
    plt.errorbar(
        confidence_levels,
        means,
        yerr=standard_errors,
        marker="o",
        capsize=4,
        linewidth=2,
    )
    plt.xscale("log")
    plt.xlabel("Confidence parameter δ (log scale)")
    plt.ylabel("Estimated expected pseudo-regret")
    plt.title("Practical Effect of δ for UCB(delta)")
    plt.tight_layout()
    plt.savefig("exercise_7_8_delta_sensitivity.png", dpi=300)
    plt.show()


def main():
    reproduce_figure_7_1()
    print("\nDelta sensitivity experiment (means = [0.0, -0.1]):")
    delta_sensitivity_experiment()


if __name__ == "__main__":
    main()
