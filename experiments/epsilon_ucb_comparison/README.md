# Reproducible epsilon-UCB experiment

This folder reproduces the finite-horizon comparison used in the Week 8
report. It is self-contained: the C++ program simulates the policies, the
Python scripts analyse the raw data and draw the figure, and the shell scripts
run either a quick pipeline check or the complete three-seed experiment.

## Research question

For the same five-arm Gaussian instance, compare the expected
epsilon-lenient regret of:

1. UCB1;
2. fixed-horizon epsilon-UCB1 with a sound epsilon-good certificate;
3. anytime alpha-UCB1 with count batching, for alpha in {1.1, 2}.

The loss at time `t` is

```text
Delta[A_t] * 1{Delta[A_t] > epsilon}.
```

Equality is good: an arm with `Delta_i = epsilon` incurs zero lenient regret.

## Experimental setup

- Independent rewards: `X_i,s ~ Normal(mu_i, 1)`.
- Means: `(1.00, 0.95, 0.87, 0.82, 0.77)`.
- Gaps: `(0, 0.05, 0.13, 0.18, 0.23)`.
- Epsilon grid: `0, 0.02, 0.05, 0.06, 0.09, 0.12, 0.14, 0.19, 0.235`.
- Horizons: `1,000, 2,000, 5,000, ..., 1,000,000`.
- Main case: `epsilon = 0.09`, so arms 1 and 2 are good.
- Full run: three seeds, 500 Monte Carlo replications per seed.
- Within each replication, every policy uses the same arm-wise reward
  streams (common random numbers).

The boundary cases are deliberate. At `epsilon = 0`, the loss is classical
regret. At `epsilon = 0.235`, every arm is good and lenient regret is exactly
zero.

## Algorithms implemented

UCB1 and fixed-horizon epsilon-UCB1 use

```text
r_i = sqrt(4 log(n) / T_i),
U_i = mean_i + r_i,
L_i = mean_i - r_i.
```

The fixed-horizon epsilon-UCB1 policy follows UCB1 until the certified set is
nonempty:

```text
C_epsilon = {i : L_i >= max_{j != i} U_j - epsilon}.
```

It then commits to the certified arm with the largest lower confidence bound.

Alpha-UCB1 is anytime. At an epoch starting at time `t_r`, it uses
`h_r = ceil(alpha * t_r)` and radius

```text
sqrt(2 q log(h_r) / T_i),   q = 4.
```

The selected arm is pulled until its count reaches
`ceil(alpha * T_i)`. Certificates are checked again at the next epoch and are
therefore revocable.

## Requirements

- A C++17 compiler with POSIX threads (`clang++` or `g++`).
- Python 3.9 or later.
- Python packages in `requirements.txt`.

Create an optional virtual environment and install the Python dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

## Reproduce the experiment

Run from this folder.

Quick end-to-end check (5 replications, one seed):

```bash
./run_experiment.sh quick
```

Complete reported experiment (3 x 500 replications):

```bash
./run_experiment.sh full
```

The full command uses seeds `20261008`, `20261009`, and `20261010`, then runs
the independent audit. It can take several minutes because every policy is
evaluated up to horizon `10^6`.

To customise one run:

```bash
./run_one.sh RUN_NAME SEED REPLICATIONS THREADS MAKE_FIGURE
# Example:
./run_one.sh trial 12345 100 8 yes
```

Set `CXX` or `PYTHON` to override the default `c++` compiler or `python3`
interpreter.

## Outputs

Each run is written to `results/<run-name>/`:

- `data/replicate_regret.csv`: replication-level regret;
- `data/certificate_diagnostics.csv`: certificate time and committed arm;
- `data/regret_summary.csv`: means and 95% normal confidence intervals;
- `all_algorithms_performance.csv`: final-horizon and coefficient summary;
- `epsilon_sweep_table.csv` and `.tex`: compact UCB1 comparison table;
- `regret_vs_horizon_epsilon_0p09.png` and `.pdf`: main figure when
  `MAKE_FIGURE=yes`.

The complete run also writes `results/audit/`, containing invariant checks and
between-seed repeatability diagnostics.

`results/reference/` contains the compact reference outputs used for the
report. Raw Monte Carlo CSV files and build artifacts are intentionally not
versioned; the commands above regenerate them.

## Result summary

Across the three 500-replication runs, at `epsilon = 0.09` and `n = 10^6`:

- UCB1 mean regret: `877.10`;
- epsilon-UCB1 mean regret: `596.69`;
- reduction: `31.97%`;
- maximum between-run relative spread among the four algorithms: `1.30%`.

The audit passed for all three runs. It checks row completeness, nonnegative
regret, hard-threshold membership, exact zero loss at `epsilon = 0.235`,
certificate consistency, and independent recalculation of every theoretical
reference coefficient.

## Interpretation of coefficients

- UCB1: `4 sum_{i in B_epsilon} 1 / Delta_i` is labelled an exact asymptotic
  coefficient.
- Fixed-horizon epsilon-UCB1:
  `4 sum_{i in B_epsilon} Delta_i / (Delta_i + epsilon/2)^2` is labelled a
  proved upper coefficient.
- Alpha-UCB1 coefficient status: proved.

The empirical coefficient is an OLS slope in
`R(n) = beta_0 + C log(n)` over the final five horizons. It is a finite-horizon
diagnostic, not a proof of an asymptotic equality.
