"""Basic checks for the UCB policies."""

import random

from ucb_algorithms import GaussianBandit, phased_ucb, ucb_delta


def run_tests():
    horizon = 50
    means = [0.0, -0.1]

    sequential_bandit = GaussianBandit(means, rng=random.Random(1))
    sequential_regret = ucb_delta(
        sequential_bandit,
        horizon,
        delta=1.0 / horizon,
        rng=random.Random(2),
    )

    assert len(sequential_bandit.actions) == horizon
    assert sequential_bandit.actions[:2] == [0, 1]
    assert 0.0 <= sequential_regret <= horizon * 0.1

    phased_bandit = GaussianBandit(means, rng=random.Random(3))
    phased_regret = phased_ucb(
        phased_bandit,
        horizon,
        delta=1.0 / horizon,
        alpha=2.0,
        rng=random.Random(4),
    )

    assert len(phased_bandit.actions) == horizon
    assert phased_bandit.actions[:2] == [0, 1]
    assert 0.0 <= phased_regret <= horizon * 0.1

    print("All UCB tests passed.")


if __name__ == "__main__":
    run_tests()
