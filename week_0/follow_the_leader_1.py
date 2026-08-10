import random


def FollowTheLeader(bandit, n):
    if n < 0:
        raise ValueError("n must be non-negative.")

    K = bandit.K()
    pull_counts = [0] * K
    reward_sums = [0] * K
    total_reward = 0

    for a in range(min(K, n)):
        reward = bandit.pull(a)
        pull_counts[a] += 1
        reward_sums[a] += reward
        total_reward += reward

    for _ in range(K, n):
        average_rewards = [
            reward_sums[a] / pull_counts[a]
            for a in range(K)
        ]

        best_average = max(average_rewards)

        leaders = [
            a for a in range(K)
            if average_rewards[a] == best_average
        ]

        chosen_arm = random.choice(leaders)
        reward = bandit.pull(chosen_arm)

        pull_counts[chosen_arm] += 1
        reward_sums[chosen_arm] += reward
        total_reward += reward

    return total_reward