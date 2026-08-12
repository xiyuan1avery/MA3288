import math
import random

import matplotlib.pyplot as plt


N_FIGURE_6_1 = 1000
N_FIGURE_6_2_3 = 2000
DELTA_FIGURE_6_2_3 = 0.1
N_SIMULATIONS = 100_000


def normal_cdf(x):
    """Standard normal cumulative distribution function."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def probability_of_wrong_commitment(m, delta):
    """
    Probability that ETC commits to the suboptimal arm after m samples per arm.
    """
    return normal_cdf(-delta * math.sqrt(m / 2.0))


def expected_regret(n, delta, m):
    """
    Exact expected pseudo-regret of two-armed ETC.

    ETC pulls each arm m times, then commits for the remaining n - 2m rounds.
    """
    if not 1 <= m <= n // 2:
        raise ValueError("m must satisfy 1 <= m <= n // 2.")

    error_probability = probability_of_wrong_commitment(m, delta)

    return delta * (m + (n - 2 * m) * error_probability)


def regret_standard_deviation(n, delta, m):
    """
    Exact standard deviation of the pseudo-regret.
    """
    if not 1 <= m <= n // 2:
        raise ValueError("m must satisfy 1 <= m <= n // 2.")

    error_probability = probability_of_wrong_commitment(m, delta)

    return (
        delta
        * (n - 2 * m)
        * math.sqrt(error_probability * (1.0 - error_probability))
    )


def optimal_m(n, delta):
    """
    Return the integer m that exactly minimizes expected regret.
    """
    if n < 2:
        raise ValueError("n must be at least 2.")

    if delta <= 0:
        raise ValueError("delta must be positive.")

    candidate_ms = range(1, n // 2 + 1)

    return min(
        candidate_ms,
        key=lambda m: expected_regret(n, delta, m),
    )


def theoretical_m(n, delta):
    """
    The Chapter 6 theoretical tuning rule from Equation (6.5).

    This is used to reproduce Figure 6.1.
    """
    raw_value = (4.0 / delta**2) * math.log(n * delta**2 / 4.0)
    m = max(1, math.ceil(raw_value))

    return min(m, n // 2)


def regret_upper_bound(n, delta):
    """
    The instance-dependent upper bound from Equation (6.6).
    """
    logarithm = max(0.0, math.log(n * delta**2 / 4.0))

    return min(
        n * delta,
        delta + (4.0 / delta) * (1.0 + logarithm),
    )


def simulate_average_regret(n, delta, m, simulations, rng):
    """
    Monte Carlo estimate of expected pseudo-regret.

    Each simulation only needs to sample whether ETC commits to the
    wrong arm, because this fully determines the pseudo-regret.
    """
    error_probability = probability_of_wrong_commitment(m, delta)
    total_regret = 0.0

    for _ in range(simulations):
        committed_to_wrong_arm = rng.random() < error_probability

        regret = delta * m

        if committed_to_wrong_arm:
            regret += delta * (n - 2 * m)

        total_regret += regret

    return total_regret / simulations


def plot_figure_6_1():
    """
    Reproduce Figure 6.1:
    expected regret and theoretical upper bound versus Delta.
    """
    rng = random.Random(20260812)

    deltas = [i / 100 for i in range(1, 101)]
    estimated_regrets = []
    upper_bounds = []

    for delta in deltas:
        m = theoretical_m(N_FIGURE_6_1, delta)

        estimated_regret = simulate_average_regret(
            N_FIGURE_6_1,
            delta,
            m,
            N_SIMULATIONS,
            rng,
        )

        estimated_regrets.append(estimated_regret)
        upper_bounds.append(regret_upper_bound(N_FIGURE_6_1, delta))

    plt.figure(figsize=(8, 5))

    plt.plot(
        deltas,
        estimated_regrets,
        linewidth=2,
        label="ETC with theoretical m",
    )

    plt.plot(
        deltas,
        upper_bounds,
        linewidth=2,
        linestyle="--",
        label="Theoretical upper bound",
    )

    plt.xlabel("Gap Δ")
    plt.ylabel("Expected regret")
    plt.title("ETC Expected Regret and Theoretical Upper Bound")
    plt.legend()
    plt.tight_layout()
    plt.savefig("etc_figure_6_1.png", dpi=300)
    plt.show()


def plot_expected_regret_vs_m():
    """
    Reproduce Figure 6.2:
    expected regret versus m for n = 2000 and Delta = 0.1.
    """
    ms = list(range(20, 401, 10))

    regrets = [
        expected_regret(N_FIGURE_6_2_3, DELTA_FIGURE_6_2_3, m)
        for m in ms
    ]

    best_m = optimal_m(N_FIGURE_6_2_3, DELTA_FIGURE_6_2_3)
    best_regret = expected_regret(
        N_FIGURE_6_2_3,
        DELTA_FIGURE_6_2_3,
        best_m,
    )

    print(f"Exact optimal m: {best_m}")
    print(f"Expected regret at optimal m: {best_regret:.3f}")

    plt.figure(figsize=(8, 5))

    plt.plot(ms, regrets, color="black", linewidth=2, label="ETC")
    plt.scatter([best_m], [best_regret], color="red", zorder=3, label="Exact optimum")

    plt.xlabel("Exploration parameter m")
    plt.ylabel("Expected regret")
    plt.title("ETC Expected Regret as a Function of m")
    plt.legend()
    plt.tight_layout()
    plt.savefig("etc_expected_regret_vs_m.png", dpi=300)
    plt.show()


def plot_standard_deviation_vs_m():
    """
    Reproduce Figure 6.3:
    standard deviation of pseudo-regret versus m.
    """
    ms = list(range(20, 401, 10))

    standard_deviations = [
        regret_standard_deviation(
            N_FIGURE_6_2_3,
            DELTA_FIGURE_6_2_3,
            m,
        )
        for m in ms
    ]

    plt.figure(figsize=(8, 5))

    plt.plot(ms, standard_deviations, color="black", linewidth=2, label="ETC")

    plt.xlabel("Exploration parameter m")
    plt.ylabel("Standard deviation of pseudo-regret")
    plt.title("ETC Regret Standard Deviation as a Function of m")
    plt.legend()
    plt.tight_layout()
    plt.savefig("etc_regret_standard_deviation_vs_m.png", dpi=300)
    plt.show()

def plot_pseudo_regret_distribution():
    """
    Visualise the distribution of pseudo-regret at the exact optimal m.
    """
    n = N_FIGURE_6_2_3
    delta = DELTA_FIGURE_6_2_3
    m = optimal_m(n, delta)

    p_wrong = probability_of_wrong_commitment(m, delta)

    correct_commit_regret = m * delta
    wrong_commit_regret = (n - m) * delta

    rng = random.Random(20260812)

    regrets = []

    for _ in range(N_SIMULATIONS):
        if rng.random() < p_wrong:
            regrets.append(wrong_commit_regret)
        else:
            regrets.append(correct_commit_regret)

    print(f"Distribution plot uses m = {m}")
    print(f"P(correct commitment) = {1 - p_wrong:.4f}")
    print(f"P(wrong commitment) = {p_wrong:.4f}")
    print(f"Pseudo-regret after correct commitment = {correct_commit_regret:.1f}")
    print(f"Pseudo-regret after wrong commitment = {wrong_commit_regret:.1f}")

    plt.figure(figsize=(8, 5))

    plt.hist(
        regrets,
        bins=50,
        edgecolor="black",
        color="steelblue",
    )

    plt.axvline(
        sum(regrets) / len(regrets),
        color="red",
        linestyle="--",
        label="Sample mean",
    )

    plt.xlabel("Pseudo-regret")
    plt.ylabel("Number of simulations")
    plt.title("Distribution of ETC Pseudo-Regret at the Exact Optimal m")
    plt.legend()
    plt.tight_layout()
    plt.savefig("etc_pseudo_regret_distribution.png", dpi=300)
    plt.show()

def main():
    plot_figure_6_1()
    plot_expected_regret_vs_m()
    plot_standard_deviation_vs_m()
    plot_pseudo_regret_distribution()


if __name__ == "__main__":
    main()