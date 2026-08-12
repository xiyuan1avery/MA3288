## Exercise 4.6 — Bernoulli Bandit Environment

Implemented a K-armed Bernoulli bandit environment.

- `K()` returns the number of arms.
- `pull(a)` samples a binary reward from arm `a`.
- `regret()` returns cumulative expected regret.

## Exercise 4.7 — Follow-the-Leader Implementation

Implemented the Follow-the-Leader algorithm for the Bernoulli bandit environment.

### Algorithm

1. Pull each arm once.
2. Calculate the observed average reward for each arm.
3. Select an arm with the largest observed average reward.
4. Break ties randomly.
5. Update the selected arm's reward total and pull count.

## Exercise 4.10 — Failure of Follow-the-Leader I

Across 1,000 simulations, the mean random regret was 3.79. Random regret ranged from -13 to 23, and 22.4% of simulations had regret of at least 10. The positive mean indicates that Follow-the-Leader is suboptimal on average, while the broad range shows substantial run-to-run variability.

The histogram is right-skewed: a considerable fraction of runs has relatively large positive regret. This is consistent with the failure event in which arm 1 initially returns 1 and arm 2 initially returns 0. Its probability is \(0.5(1 - 0.6) = 0.20\). In that situation, the inferior arm has the larger observed average after initial exploration, and Follow-the-Leader can continue selecting it while failing to re-explore the optimal arm.

The observed proportion of high-regret runs, 22.4%, is close to this 20% failure probability. This supports the conclusion that early random observations are a primary cause of the right tail. When this failure occurs, the expected regret from the remaining 98 rounds is approximately \(98(0.6 - 0.5) = 9.8\), with realised regret varying further because rewards are random.

Negative regret values do not mean that the algorithm is better than the optimal arm in expectation. They occur when the realised total reward \(S_{100}\) happens to exceed its expected optimal-arm benchmark, \(100 \times 0.6 = 60\). Overall, the experiment shows that Follow-the-Leader lacks sufficient exploration: temporary early noise can lead to persistent selection of the suboptimal arm.

## Exercise 4.11 — Failure of Follow-the-Leader II

The estimated expected regret increases steadily as the horizon grows. It rises from 3.60 at \(n = 100\) to 36.54 at \(n = 1000\). The overall increase is approximately linear: over the interval from 100 to 1000 rounds, the estimated regret increases by about 32.94, corresponding to an average slope of approximately 0.0366 regret per additional round.

The error bars show the standard errors of the estimated means across 1,000 simulations. They increase with the horizon, from 0.142 at \(n = 100\) to 1.510 at \(n = 1000\). This increasing uncertainty is consistent with the fact that early random outcomes can lead to very different long-run paths: some runs identify the better arm, whereas others continue selecting the inferior arm for many rounds. Nevertheless, the error bars are small relative to the overall upward trend, so the conclusion that regret grows with the horizon is well supported by the simulations.

Follow-the-Leader is not a good algorithm for this stochastic bandit problem. A desirable bandit algorithm should have sublinear expected regret, so that average regret per round approaches zero as the horizon grows. In contrast, the approximately linear growth observed here indicates that Follow-the-Leader continues to make a non-negligible number of suboptimal selections. Its weakness is insufficient exploration: after unlucky initial observations, it may favour the arm with mean 0.5 and fail to collect enough new evidence about the optimal arm with mean 0.6. Thus, early sampling noise can have a persistent effect on later decisions.

## Exercise 6.9 — Empirical Study of Explore-Then-Commit

### Part (e): Interpretation of the curves

For a two-armed Gaussian bandit with means \(0\) and \(-\Delta\), ETC explores each arm \(m\) times and then commits to the arm with the larger empirical mean. The probability of committing to the inferior arm is

\[
p_m = \Phi\left(-\Delta\sqrt{m/2}\right).
\]

Consequently, the expected pseudo-regret is

\[
\mathbb{E}[\bar{R}_n]
=
\Delta\left[m + (n - 2m)p_m\right].
\]

Figure 6.1 uses \(n=1000\) and varies \(\Delta\). The ETC curve is non-monotone. For very small gaps, the regret is small because selecting the inferior arm has only a small per-round cost. As \(\Delta\) increases, selecting the wrong arm becomes more costly, causing regret to increase. For larger gaps, the two arms become easier to distinguish during exploration, so the probability of a wrong commitment decreases rapidly and expected regret falls again. In the output, ETC reaches its largest expected regret at an intermediate gap of approximately \(0.16\), where it is about \(48\). The theoretical upper bound has the same broad shape but is conservative; for example, at \(\Delta = 0.10\), the estimated ETC regret is approximately \(39\), whereas the upper bound is approximately \(77\).

For \(n=2000\) and \(\Delta=0.1\), the expected-regret curve is U-shaped. At small values of \(m\), ETC does not collect enough evidence to distinguish the two arms reliably. For example, when \(m=20\), the probability of committing to the inferior arm is approximately \(0.376\), and expected regret is \(75.679\). Increasing \(m\) reduces this error probability and therefore reduces expected regret.

However, after the optimum, further exploration is wasteful because ETC deliberately pulls the inferior arm \(m\) times. The exact minimizing value is \(m=246\), with expected regret \(44.762\). At this point, the wrong-commitment probability is approximately \(0.134\). When \(m=400\), the error probability is lower, approximately \(0.079\), but expected regret increases to \(49.438\) because the additional exploration cost dominates the reduction in commitment errors.

The standard deviation decreases monotonically over the displayed range, from approximately \(94.934\) at \(m=20\) to \(32.303\) at \(m=400\). Larger values of \(m\) make the commitment decision more reliable, reducing the probability of the high-regret failure outcome. This reduction in variability does not imply that large \(m\) is optimal: expected regret begins increasing after \(m=246\) because ETC continues to explore the inferior arm unnecessarily.

These observations agree with the theoretical ETC trade-off. Small \(m\) gives insufficient exploration and a large probability of committing incorrectly; large \(m\) reduces that probability but incurs excessive deterministic exploration regret.

### Part (f): Is standard deviation an adequate distributional summary?

The standard deviation is not, by itself, an adequate summary of the distribution of \(\bar{R}_n\). For ETC, the pseudo-regret has a discrete two-outcome distribution. Once exploration is complete, ETC either commits to the optimal arm or commits to the inferior arm.

At the exact optimal value \(m=246\), the pseudo-regret equals \(24.6\) when ETC commits correctly and equals \(175.4\) when ETC commits incorrectly. The correct-commitment outcome occurs with probability approximately \(0.866\), while the high-regret outcome occurs with probability approximately \(0.134\). Thus, the distribution consists of two separated spikes rather than a symmetric bell-shaped distribution.

The mean pseudo-regret is \(44.762\) and the standard deviation is \(51.322\). Reporting only these two values hides the most important feature of the distribution: most simulations have regret \(24.6\), but a relatively small probability of failure produces regret \(175.4\). The standard deviation detects that the distribution is variable, but it does not identify the bimodal structure, the probability of the failure event, or the magnitude of the failure loss.

Therefore, ETC pseudo-regret should be summarised using a histogram or probability mass plot together with the two possible regret values and the probability of wrong commitment. The mean and standard deviation remain useful, but they are insufficient on their own.

The histogram confirms that the pseudo-regret distribution is highly non-normal and bimodal. In approximately 86.6% of simulations, ETC commits to the optimal arm after exploration and the pseudo-regret is 24.6. In the remaining approximately 13.4% of simulations, ETC commits to the inferior arm and the pseudo-regret jumps to 175.4.

The sample mean is approximately 44.8, shown by the dashed red line. However, the histogram contains essentially no observations near this mean: the outcomes are concentrated at two distant values instead. Therefore, the mean and standard deviation do not describe a typical individual run. The standard deviation quantifies dispersion, but it conceals the discrete mixture structure and the probability of catastrophic commitment failure.

For this ETC problem, a more informative summary reports the two regret values together with their probabilities, or displays a histogram such as this one. In particular, the relevant risk statement is that ETC has approximately a 13.4% probability of producing pseudo-regret 175.4, despite producing the lower regret 24.6 in most runs.