import math
import random

import matplotlib.pyplot as plt

from bernoulli_bandit import BernoulliBandit
from follow_the_leader_1 import FollowTheLeader


MEANS = [0.5, 0.6]
N_SIMULATIONS = 1000
HORIZONS = range(100, 1001, 100)


def mean(values):
    return sum(values) / len(values)


def standard_error(values):
    """
    Return the standard error of the sample mean.
    """
    sample_mean = mean(values)

    sample_variance = sum(
        (value - sample_mean) ** 2
        for value in values
    ) / (len(values) - 1)

    return math.sqrt(sample_variance / len(values))


def estimate_regret(horizon):
    """
    Run 1,000 simulations and estimate expected regret at one horizon.
    """
    regrets = []

    for _ in range(N_SIMULATIONS):
        bandit = BernoulliBandit(MEANS)
        FollowTheLeader(bandit, horizon)

        regrets.append(bandit.regret())

    return mean(regrets), standard_error(regrets)


def main():
    random.seed(20260811)

    average_regrets = []
    error_bars = []

    print("Horizon | Average regret | Standard error")
    print("------------------------------------------")

    for horizon in HORIZONS:
        average_regret, error_bar = estimate_regret(horizon)

        average_regrets.append(average_regret)
        error_bars.append(error_bar)

        print(
            f"{horizon:7d} | "
            f"{average_regret:14.3f} | "
            f"{error_bar:14.3f}"
        )

    plt.figure(figsize=(8, 5))

    plt.errorbar(
        list(HORIZONS),
        average_regrets,
        yerr=error_bars,
        fmt="-o",
        color="black",
        capsize=4,
        label="Follow-the-Leader",
    )

    plt.xlabel("Horizon n")
    plt.ylabel("Estimated expected regret")
    plt.title(
        "Follow-the-Leader: Expected Regret by Horizon\n"
        "Bernoulli bandit with means [0.5, 0.6]"
    )

    plt.legend()
    plt.tight_layout()
    plt.savefig("ftl_expected_regret_vs_horizon.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()