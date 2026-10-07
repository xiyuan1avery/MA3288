#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

constexpr int K = 5;
constexpr double Q = 4.0;

const std::array<double, K> MEANS{1.00, 0.95, 0.87, 0.82, 0.77};
const std::array<double, K> GAPS{0.00, 0.05, 0.13, 0.18, 0.23};
const std::array<double, 9> EPSILONS{
    0.00, 0.02, 0.05, 0.06, 0.09, 0.12, 0.14, 0.19, 0.235
};
const std::array<double, 2> ALPHAS{1.1, 2.0};
const std::array<int, 10> HORIZONS{
    1000, 2000, 5000, 10000, 20000,
    50000, 100000, 200000, 500000, 1000000
};

constexpr int N_EPS = static_cast<int>(EPSILONS.size());
constexpr int N_ALG = 4;
constexpr int N_H = static_cast<int>(HORIZONS.size());

std::uint64_t splitmix64(std::uint64_t x) {
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30U)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27U)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31U);
}

struct RewardStreams {
    RewardStreams(int replication, std::uint64_t base_seed) {
        for (int arm = 0; arm < K; ++arm) {
            const auto seed = splitmix64(
                base_seed
                ^ (static_cast<std::uint64_t>(replication + 1) << 16U)
                ^ static_cast<std::uint64_t>(arm + 1));
            rng_[arm].seed(seed);
            prefix_[arm].push_back(0.0);
        }
    }

    double interval_sum(int arm, int start, int length) {
        if (arm < 0 || arm >= K || start < 0 || length < 0) {
            throw std::invalid_argument("invalid reward-stream interval");
        }
        ensure(arm, start + length);
        return prefix_[arm][start + length] - prefix_[arm][start];
    }

private:
    void ensure(int arm, int count) {
        auto& prefix = prefix_[arm];
        auto& rng = rng_[arm];
        while (static_cast<int>(prefix.size()) <= count) {
            prefix.push_back(
                prefix.back() + normal_[arm](rng) + MEANS[arm]);
        }
    }

    std::array<std::mt19937_64, K> rng_;
    std::array<std::vector<double>, K> prefix_;
    std::array<std::normal_distribution<double>, K> normal_;
};

double lenient_loss(int arm, double epsilon) {
    return GAPS[arm] > epsilon ? GAPS[arm] : 0.0;
}

bool is_good(int arm, double epsilon) {
    return GAPS[arm] <= epsilon;
}

int argmax(const std::array<double, K>& values) {
    int best = 0;
    for (int arm = 1; arm < K; ++arm) {
        if (values[arm] > values[best]) {
            best = arm;
        }
    }
    return best;
}

std::array<bool, K> certified_set(
    const std::array<double, K>& lower,
    const std::array<double, K>& upper,
    double epsilon) {
    std::array<bool, K> certified{};
    for (int arm = 0; arm < K; ++arm) {
        double largest_other_upper = -std::numeric_limits<double>::infinity();
        for (int challenger = 0; challenger < K; ++challenger) {
            if (challenger != arm) {
                largest_other_upper =
                    std::max(largest_other_upper, upper[challenger]);
            }
        }
        certified[arm] = lower[arm] >= largest_other_upper - epsilon;
    }
    return certified;
}

int select_certified(
    const std::array<bool, K>& certified,
    const std::array<double, K>& lower) {
    int selected = -1;
    for (int arm = 0; arm < K; ++arm) {
        if (certified[arm]
            && (selected < 0 || lower[arm] > lower[selected])) {
            selected = arm;
        }
    }
    return selected;
}

std::array<double, N_EPS> run_ucb1(
    RewardStreams& streams,
    int horizon) {
    std::array<int, K> counts{};
    std::array<double, K> sums{};
    std::array<double, N_EPS> regrets{};
    const double log_horizon = std::log(static_cast<double>(horizon));

    for (int time = 0; time < horizon; ++time) {
        int selected = time < K ? time : -1;
        if (selected < 0) {
            std::array<double, K> indices{};
            for (int arm = 0; arm < K; ++arm) {
                indices[arm] =
                    sums[arm] / counts[arm]
                    + std::sqrt(4.0 * log_horizon / counts[arm]);
            }
            selected = argmax(indices);
        }

        const double reward =
            streams.interval_sum(selected, counts[selected], 1);
        ++counts[selected];
        sums[selected] += reward;
        for (int e = 0; e < N_EPS; ++e) {
            regrets[e] += lenient_loss(selected, EPSILONS[e]);
        }
    }
    return regrets;
}

struct EpsilonOutcome {
    double regret = 0.0;
    bool certificate_found = false;
    int certificate_time = -1;
    int committed_arm = -1;
};

EpsilonOutcome run_epsilon_ucb1(
    RewardStreams& streams,
    int horizon,
    double epsilon) {
    std::array<int, K> counts{};
    std::array<double, K> sums{};
    EpsilonOutcome outcome;
    const double log_horizon = std::log(static_cast<double>(horizon));

    for (int time = 0; time < horizon; ++time) {
        int selected = time < K ? time : -1;
        if (selected < 0) {
            std::array<double, K> upper{};
            std::array<double, K> lower{};
            for (int arm = 0; arm < K; ++arm) {
                const double empirical_mean = sums[arm] / counts[arm];
                const double radius =
                    std::sqrt(4.0 * log_horizon / counts[arm]);
                upper[arm] = empirical_mean + radius;
                lower[arm] = empirical_mean - radius;
            }

            const auto certified =
                certified_set(lower, upper, epsilon);
            selected = select_certified(certified, lower);
            if (selected >= 0) {
                outcome.certificate_found = true;
                outcome.certificate_time = time;
                outcome.committed_arm = selected;
                outcome.regret +=
                    static_cast<double>(horizon - time)
                    * lenient_loss(selected, epsilon);
                return outcome;
            }
            selected = argmax(upper);
        }

        const double reward =
            streams.interval_sum(selected, counts[selected], 1);
        ++counts[selected];
        sums[selected] += reward;
        outcome.regret += lenient_loss(selected, epsilon);
    }
    return outcome;
}

std::array<double, N_H> run_alpha_ucb1(
    RewardStreams& streams,
    double epsilon,
    double alpha) {
    std::array<int, K> counts{};
    std::array<double, K> sums{};
    std::array<double, N_H> recorded{};
    double regret = 0.0;
    int time = 0;
    int next_horizon = 0;
    const int max_horizon = HORIZONS.back();

    auto record_initial_pull = [&](int arm) {
        const int start_time = time;
        const int end_time = time + 1;
        while (next_horizon < N_H
               && HORIZONS[next_horizon] <= end_time) {
            recorded[next_horizon] =
                regret
                + static_cast<double>(
                    HORIZONS[next_horizon] - start_time)
                    * lenient_loss(arm, epsilon);
            ++next_horizon;
        }
        regret += lenient_loss(arm, epsilon);
        ++time;
    };

    for (int arm = 0; arm < K && time < max_horizon; ++arm) {
        sums[arm] = streams.interval_sum(arm, 0, 1);
        counts[arm] = 1;
        record_initial_pull(arm);
    }

    while (time < max_horizon) {
        const int epoch_horizon = std::max(
            time + 1,
            static_cast<int>(std::ceil(alpha * time)));
        const double log_epoch =
            std::log(static_cast<double>(epoch_horizon));

        std::array<double, K> upper{};
        std::array<double, K> lower{};
        for (int arm = 0; arm < K; ++arm) {
            const double empirical_mean = sums[arm] / counts[arm];
            const double radius =
                std::sqrt(2.0 * Q * log_epoch / counts[arm]);
            upper[arm] = empirical_mean + radius;
            lower[arm] = empirical_mean - radius;
        }

        const auto certified = certified_set(lower, upper, epsilon);
        int selected = select_certified(certified, lower);
        if (selected < 0) {
            selected = argmax(upper);
        }

        int target_count =
            static_cast<int>(std::ceil(alpha * counts[selected]));
        target_count = std::max(target_count, counts[selected] + 1);
        int batch_size = target_count - counts[selected];
        batch_size = std::min(batch_size, max_horizon - time);

        const int start_time = time;
        const int end_time = time + batch_size;
        const double loss = lenient_loss(selected, epsilon);
        while (next_horizon < N_H
               && HORIZONS[next_horizon] <= end_time) {
            recorded[next_horizon] =
                regret
                + static_cast<double>(
                    HORIZONS[next_horizon] - start_time) * loss;
            ++next_horizon;
        }

        sums[selected] += streams.interval_sum(
            selected, counts[selected], batch_size);
        counts[selected] += batch_size;
        regret += static_cast<double>(batch_size) * loss;
        time = end_time;
    }

    return recorded;
}

struct Diagnostic {
    bool certificate_found = false;
    int certificate_time = -1;
    int committed_arm = -1;
};

struct Results {
    explicit Results(int replications)
        : regret(static_cast<std::size_t>(
              replications * N_EPS * N_ALG * N_H), 0.0),
          diagnostic(static_cast<std::size_t>(
              replications * N_EPS)) {}

    double& at(int rep, int epsilon, int algorithm, int horizon) {
        const std::size_t index =
            (((static_cast<std::size_t>(rep) * N_EPS + epsilon)
              * N_ALG + algorithm) * N_H + horizon);
        return regret[index];
    }

    Diagnostic& diag(int rep, int epsilon) {
        return diagnostic[
            static_cast<std::size_t>(rep) * N_EPS + epsilon];
    }

    std::vector<double> regret;
    std::vector<Diagnostic> diagnostic;
};

void simulate_replication(
    int replication,
    std::uint64_t base_seed,
    Results& results) {
    RewardStreams streams(replication, base_seed);

    for (int h = 0; h < N_H; ++h) {
        const int horizon = HORIZONS[h];
        const auto ucb_regrets = run_ucb1(streams, horizon);
        for (int e = 0; e < N_EPS; ++e) {
            results.at(replication, e, 0, h) = ucb_regrets[e];
            const auto epsilon_outcome =
                run_epsilon_ucb1(streams, horizon, EPSILONS[e]);
            results.at(replication, e, 1, h) =
                epsilon_outcome.regret;
            if (h == N_H - 1) {
                auto& diagnostic = results.diag(replication, e);
                diagnostic.certificate_found =
                    epsilon_outcome.certificate_found;
                diagnostic.certificate_time =
                    epsilon_outcome.certificate_time;
                diagnostic.committed_arm =
                    epsilon_outcome.committed_arm;
            }
        }
    }

    for (int e = 0; e < N_EPS; ++e) {
        for (int a = 0; a < static_cast<int>(ALPHAS.size()); ++a) {
            const auto regrets = run_alpha_ucb1(
                streams, EPSILONS[e], ALPHAS[a]);
            for (int h = 0; h < N_H; ++h) {
                results.at(replication, e, 2 + a, h) = regrets[h];
            }
        }
    }
}

void write_regret_csv(
    const std::string& output_path,
    int replications,
    Results& results) {
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error(
            "cannot open regret output: " + output_path);
    }
    const std::array<std::string, N_ALG> algorithms{
        "UCB1",
        "epsilon-UCB1",
        "alpha-UCB1-1.1",
        "alpha-UCB1-2"
    };

    output << "replication,epsilon,algorithm,horizon,regret\n";
    output << std::setprecision(12);
    for (int rep = 0; rep < replications; ++rep) {
        for (int e = 0; e < N_EPS; ++e) {
            for (int algorithm = 0; algorithm < N_ALG; ++algorithm) {
                for (int h = 0; h < N_H; ++h) {
                    output
                        << rep << ','
                        << EPSILONS[e] << ','
                        << algorithms[algorithm] << ','
                        << HORIZONS[h] << ','
                        << results.at(rep, e, algorithm, h)
                        << '\n';
                }
            }
        }
    }
}

void write_diagnostic_csv(
    const std::string& output_path,
    int replications,
    Results& results) {
    std::ofstream output(output_path);
    if (!output) {
        throw std::runtime_error(
            "cannot open diagnostic output: " + output_path);
    }
    output
        << "replication,epsilon,horizon,certificate_found,"
        << "certificate_time,committed_arm,committed_is_good,"
        << "post_certificate_zero_regret\n";
    output << std::setprecision(12);
    for (int rep = 0; rep < replications; ++rep) {
        for (int e = 0; e < N_EPS; ++e) {
            const auto& diagnostic = results.diag(rep, e);
            const int arm = diagnostic.committed_arm;
            const bool good =
                diagnostic.certificate_found
                && is_good(arm, EPSILONS[e]);
            output
                << rep << ','
                << EPSILONS[e] << ','
                << HORIZONS.back() << ','
                << static_cast<int>(diagnostic.certificate_found) << ','
                << diagnostic.certificate_time << ','
                << (arm < 0 ? -1 : arm + 1) << ','
                << static_cast<int>(good) << ','
                << static_cast<int>(good)
                << '\n';
        }
    }
}

}  // namespace

int main(int argc, char** argv) {
    int replications = 500;
    int thread_count = 4;
    std::uint64_t base_seed = 20261008ULL;
    std::string regret_output =
        "results/data/replicate_regret.csv";
    std::string diagnostic_output =
        "results/data/certificate_diagnostics.csv";

    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--replications" && index + 1 < argc) {
            replications = std::stoi(argv[++index]);
        } else if (argument == "--threads" && index + 1 < argc) {
            thread_count = std::stoi(argv[++index]);
        } else if (argument == "--seed" && index + 1 < argc) {
            base_seed = std::stoull(argv[++index]);
        } else if (argument == "--output" && index + 1 < argc) {
            regret_output = argv[++index];
        } else if (
            argument == "--diagnostics" && index + 1 < argc) {
            diagnostic_output = argv[++index];
        } else {
            std::cerr << "Unknown or incomplete argument: "
                      << argument << '\n';
            return 2;
        }
    }

    if (replications <= 0 || thread_count <= 0) {
        std::cerr << "replications and threads must be positive\n";
        return 2;
    }

    Results results(replications);
    std::atomic<int> next_replication{0};
    std::atomic<int> finished{0};
    const int workers = std::min(thread_count, replications);
    std::vector<std::thread> threads;
    threads.reserve(workers);

    for (int worker = 0; worker < workers; ++worker) {
        threads.emplace_back([&]() {
            while (true) {
                const int replication =
                    next_replication.fetch_add(1);
                if (replication >= replications) {
                    break;
                }
                simulate_replication(
                    replication, base_seed, results);
                const int done = finished.fetch_add(1) + 1;
                if (done == replications || done % 10 == 0) {
                    std::cerr
                        << "\rCompleted " << done
                        << '/' << replications << std::flush;
                }
            }
        });
    }

    for (auto& worker : threads) {
        worker.join();
    }
    std::cerr << '\n';

    try {
        write_regret_csv(regret_output, replications, results);
        write_diagnostic_csv(
            diagnostic_output, replications, results);
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }

    std::cout
        << "Wrote " << regret_output << '\n'
        << "Wrote " << diagnostic_output << '\n';
    return 0;
}
