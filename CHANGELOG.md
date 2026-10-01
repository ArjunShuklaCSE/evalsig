# Changelog

## 0.1.0

First release.

- `evalsig compare`: paired bootstrap confidence interval for the difference between two runs, a sign-flip permutation p-value, McNemar's exact test with fixed/broke counts for pass/fail metrics, Holm-Bonferroni correction across metrics, and the minimum detectable effect.
- `evalsig summary`: Wilson intervals for pass rates and bootstrap intervals for continuous metrics.
- JSONL and CSV input joined on `id`, automatic metric detection, a `group` column for cluster resampling, and clear errors for mismatched ids, missing values and empty files.
- `--lower-is-better` for latency and cost, `--fail-on worse|not-better` exit codes for CI, and table, JSON and markdown output.
- A composite GitHub Action that posts one pull request comment and updates it on later runs.
- Calibration simulations, property-based tests and a statistics write-up with references.
