import random


class BernoulliBandit:
    """A K-armed Bernoulli bandit environment."""

    def __init__(self, means):
        if len(means) < 2:
            raise ValueError("means must contain at least two arms.")

        if any(not 0.0 <= mean <= 1.0 for mean in means):
            raise ValueError("each mean must lie in [0, 1].")

        self.means = tuple(float(mean) for mean in means)
        self.best_mean = max(self.means)
        self.cumulative_regret = 0.0

    def K(self):
        """Return the number of arms."""
        return len(self.means)

    def pull(self, a):
        """Pull arm a and return a Bernoulli reward."""
        if not isinstance(a, int):
            raise TypeError("arm index must be an integer.")

        if not 0 <= a < self.K():
            raise IndexError("arm index is out of range.")

        reward = int(random.random() < self.means[a])
        self.cumulative_regret += self.best_mean - self.means[a]

        return reward

    def regret(self):
        """Return the cumulative expected regret."""
        return self.cumulative_regret