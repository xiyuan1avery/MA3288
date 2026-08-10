from bernoulli_bandit import BernoulliBandit


bandit = BernoulliBandit([0.2, 0.8, 1.0])

assert bandit.K() == 3

assert bandit.pull(2) == 1
assert bandit.regret() == 0.0

assert bandit.pull(0) in (0, 1)
assert bandit.regret() == 0.8

print("All tests passed.")