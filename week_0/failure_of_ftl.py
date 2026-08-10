import random

import matplotlib.pyplot as plt

from bernoulli_bandit import BernoulliBandit
from follow_the_leader_1 import FollowTheLeader


N_ROUNDS = 100
N_SIMULATIONS = 1000
MEANS = [0.5, 0.6]
OPTIMAL_MEAN = max(MEANS)


def run_simulations():
    random_regrets = []

    for _ in range(N_SIMULATIONS):
        bandit = BernoulliBandit(MEANS)

        total_reward = FollowTheLeader(bandit, N_ROUNDS)

        # Random regret: n * mu_star - S_n
        regret = N_ROUNDS * OPTIMAL_MEAN - total_reward
        random_regrets.append(regret)

    return random_regrets


def plot_histogram(random_regrets):
    plt.figure(figsize=(8, 5))

    plt.hist(
        random_regrets,
        bins=30,
        edgecolor="black",
        color="steelblue",
    )

    plt.xlabel("Random regret")
    plt.ylabel("Number of simulations")
    plt.title(
        "Follow-the-Leader Random Regret\n"
        "Bernoulli bandit: means = [0.5, 0.6]"
    )

    plt.tight_layout()
    plt.savefig("ftl_random_regret_histogram.png", dpi=300)
    plt.show()


def main():
    random_regrets = run_simulations()

    mean_regret = sum(random_regrets) / N_SIMULATIONS
    high_regret_rate = sum(regret >= 10 for regret in random_regrets) / N_SIMULATIONS

    print(f"Mean random regret: {mean_regret:.2f}")
    print(f"Proportion with regret >= 10: {high_regret_rate:.3f}")

    print(f"Number of simulations: {N_SIMULATIONS}")
    print(f"Mean random regret: {sum(random_regrets) / N_SIMULATIONS:.2f}")
    print(f"Minimum random regret: {min(random_regrets):.2f}")
    print(f"Maximum random regret: {max(random_regrets):.2f}")

    plot_histogram(random_regrets)


if __name__ == "__main__":
    main()