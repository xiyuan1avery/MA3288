## Exercise 7.5(d) — Empirical Comparison

The experiment compares sequential UCB(\(\delta\)) and phased UCB with \(\alpha=2\) on a two-armed Gaussian bandit with means \(0\) and \(-0.1\). Both algorithms use the same confidence level, \(\delta=1/n\), and are evaluated using estimated expected pseudo-regret over repeated simulations.

Both curves increase with the horizon, but phased UCB is consistently higher. The gap is initially small: at \(n=100\), both algorithms have estimated regret close to \(4.3\). As the horizon increases, the difference becomes substantial. At \(n=1000\), sequential UCB has regret approximately \(26.5\), whereas phased UCB has regret approximately \(29.6\). At \(n=2000\), the corresponding regrets are approximately \(40.5\) and \(49.7\), so phased UCB incurs about \(9.2\) additional regret.

The empirical difference is explained by the update frequency. Sequential UCB recomputes all confidence bounds after every reward, allowing it to correct a suboptimal choice immediately when new evidence arrives. Phased UCB computes indices only at phase boundaries. Once it selects an arm, it keeps pulling that arm until its number of samples doubles. If the selected arm is suboptimal, the algorithm may continue pulling it for many additional rounds before re-evaluating the decision. This delay produces the higher regret observed for phased UCB.

The error bars are small relative to the separation between the curves at larger horizons. Therefore, the observed advantage of sequential UCB is not plausibly explained by Monte Carlo uncertainty alone. Phased UCB reduces the number of decision updates, which may be computationally useful in some settings, but this experiment shows the cost of batching: slower adaptation and higher regret.

## Exercise 7.8: Empirical Comparison of ETC and UCB(delta)

### (b) Reproducing Figure 7.1

I compared UCB(delta) with Explore-Then-Commit (ETC) on two-armed Gaussian bandits with means `0` and `-Delta`, using horizon `n = 1000`. The UCB confidence parameter was set to `delta = 1 / n = 0.001`. The ETC algorithms used fixed exploration counts `m = 25, 50, 75, 100`, as well as the instance-specific optimal value of `m`.

The resulting figure is saved as `exercise_7_8_figure_7_1.png`. UCB(delta) has lower estimated regret than every fixed-`m` ETC method over most of the range of gaps. This illustrates that UCB adapts to the observed data instead of requiring a single exploration parameter that is appropriate for every bandit instance. The grey “optimal m” ETC curve is an oracle benchmark: it assumes that the gap Delta is known in advance, whereas UCB does not use this information.

### (c) Shape of the ETC curves

For ETC with exploration parameter `m`, the expected pseudo-regret is

R_n(m, Delta) = Delta [m + (n - 2m) Phi(-Delta sqrt(m / 2))],

where `Phi` is the standard normal cumulative distribution function.

For small gaps, the two arms are difficult to distinguish. The probability of committing to the wrong arm is close to one half, while the loss from choosing the wrong arm increases with Delta. Therefore, expected regret initially increases as Delta grows.

For intermediate gaps, the probability of a wrong commitment falls rapidly because the exploration samples become more informative. This reduction in the probability of error can dominate the increase in the per-round loss Delta, producing the dip in the ETC curve. For example, the curve for `m = 50` rises to approximately 39 near `Delta = 0.18`, then decreases to about 26 near `Delta = 0.45`.

For large gaps, committing to the wrong arm becomes very unlikely. The main contribution to regret is then the unavoidable exploration cost: ETC pulls the suboptimal arm `m` times during exploration. Hence,

R_n(m, Delta) approximately equals m Delta,

which is linear in Delta. This explains why the fixed-`m` ETC curves eventually become straight lines, with steeper slopes for larger values of `m`. In the figure, the large-gap regrets are approximately 25, 49, 74, and 99 for `m = 25, 50, 75, and 100`, respectively, consistent with the linear term `m Delta`.

### (d) Effect of the confidence parameter delta

I fixed the bandit means at `[0, -0.1]`, used horizon `n = 1000`, and ran 500 simulations for each value of delta. The plot `exercise_7_8_delta_sensitivity.png` reports the estimated expected pseudo-regret with standard-error bars.

| delta | Average pseudo-regret | Standard error |
|---:|---:|---:|
| 0.000001 | 31.331 | 0.473 |
| 0.00001 | 30.831 | 0.538 |
| 0.0001 | 29.103 | 0.588 |
| 0.001 | 25.532 | 0.687 |
| 0.01 | 24.333 | 0.865 |
| 0.05 | 20.835 | 1.051 |
| 0.1 | 24.689 | 1.298 |
| 0.2 | 26.622 | 1.531 |
| 0.5 | 32.357 | 1.898 |

### (e) Interpretation of the delta experiment

The results show a U-shaped relationship between delta and regret. The lowest estimated regret was obtained at `delta = 0.05`, with average pseudo-regret `20.835 ± 1.051` standard errors.

When delta is extremely small, the confidence bonus
`sqrt(2 log(1 / delta) / T_i)` is large. UCB is then overly cautious and continues exploring both arms too often. This explains the relatively high regret of `31.331` at `delta = 10^-6`.

When delta is too large, the confidence bonus becomes too small. The algorithm may stop exploring too soon after noisy early observations and pull the suboptimal arm too frequently. Regret increases from `20.835` at `delta = 0.05` to `32.357` at `delta = 0.5`. The larger error bars at large delta also indicate greater run-to-run variability, consistent with occasional early incorrect decisions.

Thus, delta controls the exploration-exploitation trade-off. Very small delta produces excessive exploration, while very large delta produces insufficient exploration. For this particular horizon and bandit gap, a moderate value around `0.05` performed best. This empirical optimum is instance-dependent and should not be treated as a universal choice.