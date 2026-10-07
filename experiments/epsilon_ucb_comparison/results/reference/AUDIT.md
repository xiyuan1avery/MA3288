# Revised experiment audit

## Formula and invariant checks

- run_1: PASS (seed 20261008, 500 replications).
- run_2: PASS (seed 20261009, 500 replications).
- run_3: PASS (seed 20261010, 500 replications).

Checks include row completeness, nonnegative regret, pathwise UCB1 monotonicity in epsilon, exact zero regret at epsilon=0.235, hard-threshold good-set membership, certificate diagnostic consistency, analytic coefficient recalculation, and proved labelling of the alpha-UCB1 coefficient.

## Repeatability

Three runs use distinct base seeds and common random numbers within each run.
Maximum relative spread of mean R(10^6) among the four algorithms at epsilon=0.09: 1.30%.

See repeatability_audit.csv for every epsilon and algorithm.
