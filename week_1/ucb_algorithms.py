"""UCB(delta) and phased UCB policies for Gaussian multi-armed bandits."""

import math
import random


class GaussianBandit:
    """Gaussian bandit with independent unit-variance rewards."""

    def __init__(self, means, rng=None):
        if len(means) < 2:
            raise ValueError("means must contain at least two arms.")

        self.means = tuple(float(mean) for mean in means)
        self.rng = rng if rng is not None else random.Random()
        self.best_mean = max(self.means)
        self.pseudo_regret = 0.0
        self.actions = []

    def K(self):
        """Return the number of arms."""
        return len(self.means)

    def pull(self, arm):
        """Pull an arm, update pseudo-regret, and return a Gaussian reward."""
        if not 0 <= arm < self.K():
            raise IndexError("arm index is out of range.")

        self.actions.append(arm)
        self.pseudo_regret += self.best_mean - self.means[arm]

        return self.rng.gauss(self.means[arm], 1.0)

    def regret(self):
        """Return cumulative pseudo-regret."""
        return self.pseudo_regret


def ucb_index(empirical_mean, count, delta):
    """Compute the UCB(delta) index from Equation (7.2)."""
    if count == 0:
        return math.inf

    return empirical_mean + math.sqrt(2.0 * math.log(1.0 / delta) / count)


def choose_max_index(indices, rng):
    """Choose randomly among arms tied for the largest index."""
    best_index = max(indices)
    leaders = [arm for arm, index in enumerate(indices) if index == best_index]
    return rng.choice(leaders)


def validate_inputs(bandit, horizon, delta):
    if horizon < 0:
        raise ValueError("horizon must be non-negative.")

    if not 0.0 < delta < 1.0:
        raise ValueError("delta must lie in (0, 1).")

    if horizon < bandit.K():
        raise ValueError("horizon must be at least the number of arms.")


def initialise(bandit):
    """Play every arm once and return counts and reward sums."""
    counts = [0] * bandit.K()
    reward_sums = [0.0] * bandit.K()

    for arm in range(bandit.K()):
        reward_sums[arm] += bandit.pull(arm)
        counts[arm] += 1

    return counts, reward_sums


def ucb_delta(bandit, horizon, delta, rng=None):
    """Run sequential UCB(delta) for a fixed horizon."""
    validate_inputs(bandit, horizon, delta)
    rng = rng if rng is not None else random.Random()
    counts, reward_sums = initialise(bandit)

    while len(bandit.actions) < horizon:
        empirical_means = [reward_sums[arm] / counts[arm] for arm in range(bandit.K())]
        indices = [
            ucb_index(empirical_means[arm], counts[arm], delta)
            for arm in range(bandit.K())
        ]
        arm = choose_max_index(indices, rng)

        reward_sums[arm] += bandit.pull(arm)
        counts[arm] += 1

    return bandit.regret()


def phased_ucb(bandit, horizon, delta, alpha=2.0, rng=None):
    """Run Algorithm 5, selecting an arm until its count grows by alpha."""
    validate_inputs(bandit, horizon, delta)

    if alpha <= 1.0:
        raise ValueError("alpha must be greater than 1.")

    rng = rng if rng is not None else random.Random()
    counts, reward_sums = initialise(bandit)

    while len(bandit.actions) < horizon:
        empirical_means = [reward_sums[arm] / counts[arm] for arm in range(bandit.K())]
        indices = [
            ucb_index(empirical_means[arm], counts[arm], delta)
            for arm in range(bandit.K())
        ]
        arm = choose_max_index(indices, rng)
        target_count = math.ceil(alpha * counts[arm])

        while counts[arm] < target_count and len(bandit.actions) < horizon:
            reward_sums[arm] += bandit.pull(arm)
            counts[arm] += 1

    return bandit.regret()
