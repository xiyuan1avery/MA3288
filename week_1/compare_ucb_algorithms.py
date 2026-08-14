"""Compare sequential UCB(delta) and phased UCB on a Gaussian bandit."""

import math
import random

import matplotlib.pyplot as plt

from ucb_algorithms import GaussianBandit, phased_ucb, ucb_delta


MEANS = [0.0, -0.1]
HORIZONS = range(100, 2001, 100)
# Increase this after the first successful run if narrower error bars are needed.
N_SIMULATIONS = 500
ALPHA = 2.0


def mean(values):
    return sum(values) / len(values)


def standard_error(values):
    average = mean(values)
    variance = sum((value - average) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance / len(values))


def delta_for_horizon(horizon):
    """Use a horizon-dependent confidence level for both policies."""
    return 1.0 / horizon


def estimate_regrets(horizon, seed):
    """Estimate pseudo-regret for both policies using matched random seeds."""
    sequential_regrets = []
    phased_regrets = []
    delta = delta_for_horizon(horizon)

    for simulation in range(N_SIMULATIONS):
        sequential_rng = random.Random(seed + 2 * simulation)
        phased_rng = random.Random(seed + 2 * simulation + 1)

        sequential_bandit = GaussianBandit(MEANS, rng=sequential_rng)
        phased_bandit = GaussianBandit(MEANS, rng=phased_rng)

        sequential_regrets.append(
            ucb_delta(sequential_bandit, horizon, delta, rng=sequential_rng)
        )
        phased_regrets.append(
            phased_ucb(
                phased_bandit,
                horizon,
                delta,
                alpha=ALPHA,
                rng=phased_rng,
            )
        )

    return sequential_regrets, phased_regrets


def main():
    sequential_means = []
    sequential_errors = []
    phased_means = []
    phased_errors = []

    print("Horizon | UCB(delta) mean +/- SE | Phased UCB mean +/- SE")
    print("---------------------------------------------------------------")

    for horizon in HORIZONS:
        sequential, phased = estimate_regrets(horizon, seed=20260814 + horizon)

        sequential_mean = mean(sequential)
        sequential_error = standard_error(sequential)
        phased_mean = mean(phased)
        phased_error = standard_error(phased)

        sequential_means.append(sequential_mean)
        sequential_errors.append(sequential_error)
        phased_means.append(phased_mean)
        phased_errors.append(phased_error)

        print(
            f"{horizon:7d} | "
            f"{sequential_mean:7.3f} +/- {sequential_error:6.3f} | "
            f"{phased_mean:7.3f} +/- {phased_error:6.3f}"
        )

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        list(HORIZONS),
        sequential_means,
        yerr=sequential_errors,
        fmt="-o",
        capsize=4,
        label="UCB(delta)",
    )
    plt.errorbar(
        list(HORIZONS),
        phased_means,
        yerr=phased_errors,
        fmt="-s",
        capsize=4,
        label="Phased UCB (alpha = 2)",
    )

    plt.xlabel("Horizon n")
    plt.ylabel("Estimated expected pseudo-regret")
    plt.title("Sequential UCB(delta) vs Phased UCB on a Gaussian Bandit")
    plt.legend()
    plt.tight_layout()
    plt.savefig("ucb_vs_phased_ucb.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
