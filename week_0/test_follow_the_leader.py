from bernoulli_bandit import BernoulliBandit
from follow_the_leader import FollowTheLeader

bandit = BernoulliBandit([0.2, 0.5, 0.8])

FollowTheLeader(bandit, 100)

print(f"Cumulative regret: {bandit.regret():.2f}")
