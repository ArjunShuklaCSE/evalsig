# The statistics behind evalsig

This page explains each method evalsig uses, why it was chosen over the alternatives, and how it was checked. Code references point to [`src/evalsig/stats.py`](../src/evalsig/stats.py) and [`src/evalsig/core.py`](../src/evalsig/core.py).

## Contents

- [The setting](#the-setting)
- [One run, binary metric: Wilson score interval](#one-run-binary-metric-wilson-score-interval)
- [One run, continuous metric: bootstrap percentile interval](#one-run-continuous-metric-bootstrap-percentile-interval)
- [Two runs: why everything is paired](#two-runs-why-everything-is-paired)
- [Two runs: paired bootstrap interval for the difference](#two-runs-paired-bootstrap-interval-for-the-difference)
- [Two runs: sign-flip permutation test](#two-runs-sign-flip-permutation-test)
- [Binary metrics: McNemar's exact test](#binary-metrics-mcnemars-exact-test)
- [Grouped rows: cluster resampling](#grouped-rows-cluster-resampling)
- [Several metrics: Holm-Bonferroni](#several-metrics-holm-bonferroni)
- [Minimum detectable effect](#minimum-detectable-effect)
- [The verdict](#the-verdict)
- [Reproducibility](#reproducibility)
- [How the methods were validated](#how-the-methods-were-validated)
- [Known limitations](#known-limitations)
- [References](#references)

## The setting

An eval produces one score per test example: a pass/fail, a recall, a judge score, a latency. Two runs of the eval (baseline and candidate) score the *same* examples. The question is whether the candidate's mean score differs from the baseline's by more than the luck of which examples happen to be in the test set.

Each method below treats the test examples as a sample from a larger population of questions you care about. A confidence interval answers "what range of true differences is consistent with this sample?" A p-value answers "if the two systems were truly equal on that population, how surprising would a difference this large be?"

evalsig decides whether a metric is **binary** (every value is 0 or 1, including `true`/`false`) or **continuous** (anything else). `--metric-type name=continuous` overrides the detection.

## One run, binary metric: Wilson score interval

For a pass rate of *k* passes out of *n*, evalsig reports the Wilson score interval:

$$\frac{\hat p + \frac{z^2}{2n}}{1 + \frac{z^2}{n}} \pm \frac{z}{1 + \frac{z^2}{n}}\sqrt{\frac{\hat p(1-\hat p)}{n} + \frac{z^2}{4n^2}}$$

where $\hat p = k/n$ and $z = 1.96$ for 95%.

**Why Wilson.** The textbook interval $\hat p \pm z\sqrt{\hat p(1-\hat p)/n}$ (the Wald interval) has poor coverage exactly where evals live: small *n* and pass rates near 0 or 1. At 0 passes it collapses to [0, 0], which claims certainty from no evidence. Brown, Cai and DasGupta (2001) compare the alternatives and recommend Wilson for its coverage close to nominal and its simplicity. The Clopper–Pearson "exact" interval is a reasonable alternative, but it is conservative (it covers more than 95%), which makes it wider than necessary.

evalsig uses scipy's implementation (`binomtest(...).proportion_ci(method="wilson")`), which the tests check against hand-computed values.

## One run, continuous metric: bootstrap percentile interval

For a continuous metric evalsig resamples the *n* examples with replacement 10,000 times, recomputes the mean each time, and reports the 2.5th and 97.5th percentiles of those means (Efron and Tibshirani, 1993).

**Why the bootstrap.** Eval scores are rarely normal: recall per question is often 0, 0.5 or 1, judge scores pile up at the top of the scale, and latency is skewed. The bootstrap makes no assumption about the shape of the distribution.

**Why percentile rather than BCa.** The bias-corrected and accelerated (BCa) bootstrap has better theoretical accuracy, but it is harder to explain and to check, and with the sample sizes typical of evals (50 to a few thousand) the difference is small. The percentile interval is slightly too narrow at small *n* (see [Known limitations](#known-limitations)). evalsig warns when there are fewer than 20 examples or groups.

## Two runs: why everything is paired

Because both runs score the same examples, evalsig works on the per-example **difference** $d_i = \text{candidate}_i - \text{baseline}_i$, not on two separate means.

This matters a lot. The variance of the difference in means is

$$\operatorname{Var}(\bar c - \bar b) = \frac{\operatorname{Var}(c) + \operatorname{Var}(b) - 2\operatorname{Cov}(b, c)}{n}$$

and in evals the covariance is large and positive: a hard question is hard for both systems. Pairing removes that shared difficulty. Comparing two independent intervals ignores the covariance term, so it is far less sensitive. A common mistake is to check whether two separate 95% intervals overlap, which is even more conservative than an unpaired test (Schenker and Gentleman, 2001). Miller (2024) makes the same argument for language-model evals specifically.

## Two runs: paired bootstrap interval for the difference

evalsig resamples example ids with replacement (10,000 times by default), takes the mean of $d_i$ over each resample, and reports the percentile interval of those means. Resampling *ids*, not baseline and candidate rows separately, is what keeps the pairing.

## Two runs: sign-flip permutation test

The p-value comes from a paired permutation test (Fisher, 1935; Good, 2005). If the two systems are truly equivalent, then for each example it is arbitrary which score is labelled "baseline" and which "candidate". Swapping the labels flips the sign of $d_i$. The test compares the observed $|\sum_i d_i|$ against its distribution over random sign flips:

$$p = \Pr\left(\left|\sum_i \varepsilon_i d_i\right| \ge \left|\sum_i d_i\right|\right), \quad \varepsilon_i \in \{-1, +1\}$$

- Examples with $d_i = 0$ cannot change the statistic and are dropped.
- If the remaining *k* examples have $2^k$ sign patterns or fewer than the resample count, evalsig enumerates all of them and the p-value is **exact**.
- Otherwise it draws random sign patterns and reports $(\text{hits} + 1)/(\text{resamples} + 1)$. The +1 counts the observed data as one of the permutations, which keeps the test valid and means the p-value is never exactly 0 (Phipson and Smyth, 2010).

**Why a permutation test rather than a bootstrap p-value or a t-test.** Under the null hypothesis that the labels are exchangeable, the permutation test holds its false-positive rate at α by construction, at any sample size and for any distribution of scores. A paired t-test relies on approximate normality of the mean, which fails for small *n* and lopsided binary data. A p-value read off the bootstrap distribution is less accurate in the tails.

## Binary metrics: McNemar's exact test

For a binary metric, evalsig also reports the discordant counts. **Fixed** counts examples that went from fail to pass; **broke** counts examples that went from pass to fail. It adds McNemar's exact test (McNemar, 1947): a two-sided binomial test of *fixed* successes in *fixed + broke* trials at probability 0.5.

On binary data the sign-flip test above *is* McNemar's exact test. The only nonzero differences are +1 (fixed) and −1 (broke), so the sign-flip distribution of their sum is the distribution of $2X - k$ with $X \sim \text{Binomial}(k, 0.5)$. The tests check that the two p-values agree. McNemar's p is shown because many readers know it, and because "fixed 15, broke 3" is the most useful line in the report: it shows the amount of churn, not just the net change.

The reported McNemar p is uncorrected. The Holm-corrected sign-flip p is the one that decides the verdict. Fagerland, Lydersen and Laake (2013) show that the mid-p variant of McNemar has better power than the exact conditional test. evalsig keeps the exact version so that it matches the permutation test.

## Grouped rows: cluster resampling

Sometimes several rows belong to the same question: five samples per prompt, or one row per turn of a conversation. These rows are correlated, and treating them as independent understates the uncertainty. Add a `group` column and evalsig resamples and sign-flips **whole groups**:

- **Point estimate:** still the mean over all rows, so it matches the number your eval reports.
- **Bootstrap:** draws groups with replacement and computes the row-weighted mean of each resample: $\sum_{g} S_g / \sum_{g} n_g$, where $S_g$ is a group's sum and $n_g$ its row count (Field and Welsh, 2007).
- **Permutation test:** flips the sign of a whole group's sum at once. This is valid because, under the null hypothesis, the labels are exchangeable within each group.
- **Single-run binary metrics with groups** use the cluster bootstrap instead of Wilson, because Wilson assumes independent rows.
- **McNemar is not reported** with groups for the same reason.

In the validation simulation (50 groups of 4 strongly correlated rows), the cluster interval covered the true difference about 94% of the time, while treating rows as independent covered less than 80%. Miller (2024) gives the same recommendation (clustered standard errors) for evals with several samples per question.

## Several metrics: Holm-Bonferroni

Testing three metrics at α = 0.05 gives a much higher than 5% chance that at least one is called significant by luck. When more than one metric is compared, evalsig adjusts the p-values with Holm's step-down method (Holm, 1979):

1. Sort the *m* p-values from smallest to largest.
2. Multiply the *i*-th smallest by $(m - i + 1)$.
3. Make the results non-decreasing in that order, and cap them at 1.

Holm controls the family-wise error rate at α under any dependence between the metrics, and it is never less powerful than plain Bonferroni. The output says when it was applied: the p column is labelled "p (Holm)".

**The intervals are not adjusted.** A metric can therefore show an interval that excludes 0 together with a "no detectable difference" verdict. When this happens, evalsig adds a note saying the metric was significant on its own but not after the correction. The alternative, widening every interval to the Bonferroni level, would make the intervals harder to compare with other tools.

## Minimum detectable effect

For every comparison evalsig reports the smallest true difference the test set could reliably detect:

$$\text{MDE} = (z_{1-\alpha/2} + z_{\text{power}}) \times \text{SE} \approx 2.8 \times \text{SE} \quad (\alpha = 0.05,\ \text{power} = 0.8)$$

SE is the standard deviation of the bootstrap distribution of the mean difference, which already accounts for pairing and, with groups, clustering. This is the standard normal-approximation power formula (Cohen, 1988).

**How to read it.** With "no detectable difference" and an MDE of ±0.06, the data cannot tell a true change of, say, +0.03 from no change. Your test set is too small to answer questions at that resolution. In the validation simulation, a true difference equal to the reported MDE was detected 78.8% of the time, against the design value of 80%.

The MDE is computed after the fact from the observed SE, so it describes the test set you have, not a planned study. It assumes the SE under a real effect is about the same as the observed one.

## The verdict

Each metric gets one of three verdicts, decided by the (Holm-adjusted when needed) permutation p-value and the sign of the difference:

| Condition | Verdict |
|---|---|
| p < α and the difference goes the good way | **better** |
| p < α and the difference goes the bad way | **worse** |
| otherwise | **no detectable difference** |

"The good way" is up, unless the metric is listed with `--lower-is-better` (latency, cost, error rate).

evalsig never says "the same", "equal" or "no effect". Failing to detect a difference is not evidence that there is none (Altman and Bland, 1995). That is why the "no detectable difference" verdict always comes with the MDE. Showing that two systems are equivalent needs a different test, such as two one-sided tests (TOST) against a margin you choose in advance (Schuirmann, 1987). evalsig v0.1 does not offer that test.

If the interval and the test disagree at the margin (one excludes zero, the other does not), evalsig keeps the verdict from the test and adds a "borderline" note.

## Reproducibility

- **Seeding:** every random draw comes from `numpy.random.default_rng(seed)`, and the default seed is 0. The same input and seed give byte-identical output.
- **Independent metrics:** each metric gets a fresh generator from the same seed, so adding or removing a metric does not change another metric's numbers.
- **Exact cases:** exact enumeration (small *k*) and Wilson involve no randomness at all.
- **Memory:** large inputs are resampled in batches. The tests check that batching does not change the results.

## How the methods were validated

The test suite ([`tests/`](../tests)) checks every statistic against known values or scipy, and runs simulations with fixed seeds:

| Simulation (400 datasets each, n = 100) | Target | Measured |
|---|---:|---:|
| No true difference, binary metric: interval contains 0 | 95% | 97.0% |
| No true difference, binary metric: false "better"/"worse" | 5% | 2.0% |
| No true difference, continuous metric: interval contains 0 | 95% | 95.8% |
| No true difference, continuous metric: false "better"/"worse" | 5% | 3.5% |
| True difference equal to the reported MDE: detected | 80% | 78.8% |
| Clustered rows, 50 groups of 4: cluster interval contains 0 | 95% | ≈94% |
| Same data, rows treated as independent: interval contains 0 | (too low) | <80% |

The binary test is slightly conservative because a binary sum can only take a few values, so the exact p-value moves in steps. Property-based tests (Hypothesis) check invariants on random data: swapping baseline and candidate mirrors the interval and leaves the p-value unchanged, adding a constant to both runs changes nothing, and Holm never lowers a p-value or changes their order.

## Known limitations

- **Few groups or examples:** the percentile bootstrap is a little too narrow with few independent units. In 1,500 simulations with 25 groups, the 95% cluster interval covered about 92% of the time. With 50 or more groups it covered about 94%. The permutation test, which decides the verdict, kept its 5% false-positive rate in all of these runs. evalsig warns below 20 examples or groups.
- **The test set is treated as a random sample.** If your examples were hand-picked, the intervals describe uncertainty about *that kind* of example, not about all inputs.
- **No correction for selection across experiments.** If you tried 20 prompts and compare the best one against the baseline, the winner's p-value is optimistic. Holm only corrects across the metrics in one `compare` call. Confirm on a fresh test set.
- **Judge noise is part of the data.** When an LLM judge scores examples, its randomness and bias end up inside the per-example scores. evalsig measures sampling uncertainty over examples, not the judge's error.
- **Equivalence is not tested** (see [The verdict](#the-verdict)).

## References

- Altman, D. G., and Bland, J. M. (1995). Absence of evidence is not evidence of absence. *BMJ*, 311(7003), 485.
- Brown, L. D., Cai, T. T., and DasGupta, A. (2001). Interval estimation for a binomial proportion. *Statistical Science*, 16(2), 101–133.
- Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences* (2nd ed.). Lawrence Erlbaum.
- Efron, B., and Tibshirani, R. J. (1993). *An Introduction to the Bootstrap*. Chapman & Hall.
- Fagerland, M. W., Lydersen, S., and Laake, P. (2013). The McNemar test for binary matched-pairs data: mid-p and asymptotic are better than exact conditional. *BMC Medical Research Methodology*, 13, 91.
- Field, C. A., and Welsh, A. H. (2007). Bootstrapping clustered data. *Journal of the Royal Statistical Society: Series B*, 69(3), 369–390.
- Fisher, R. A. (1935). *The Design of Experiments*. Oliver & Boyd.
- Good, P. (2005). *Permutation, Parametric, and Bootstrap Tests of Hypotheses* (3rd ed.). Springer.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. *Psychometrika*, 12(2), 153–157.
- Miller, E. (2024). Adding error bars to evals: a statistical approach to language model evaluations. arXiv:2411.00640.
- Phipson, B., and Smyth, G. K. (2010). Permutation p-values should never be zero. *Statistical Applications in Genetics and Molecular Biology*, 9(1), Article 39.
- Schenker, N., and Gentleman, J. F. (2001). On judging the significance of differences by examining the overlap between confidence intervals. *The American Statistician*, 55(3), 182–186.
- Schuirmann, D. J. (1987). A comparison of the two one-sided tests procedure and the power approach for assessing the equivalence of average bioavailability. *Journal of Pharmacokinetics and Biopharmaceutics*, 15(6), 657–680.
