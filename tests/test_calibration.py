"""Simulation checks that the intervals and tests behave as advertised.

Every simulation is seeded, so these tests are deterministic. Tolerances allow for
Monte Carlo error over the number of simulated datasets (SD of a 95% coverage rate over
400 datasets is about 0.011) plus the known slight under-coverage of percentile
bootstrap intervals at moderate n.
"""

import numpy as np
import pytest

from evalsig import compare_arrays

SIMS = 400
N = 100
RESAMPLES = 1000


def run(base: np.ndarray, cand: np.ndarray, groups: list[str] | None = None):  # type: ignore[no-untyped-def]
    return compare_arrays({"m": base}, {"m": cand}, groups=groups, resamples=RESAMPLES).metrics[0]


def binary_null(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Both runs pass each example with that example's own difficulty-driven rate.
    p = rng.beta(2, 2, size=N)
    return (rng.random(N) < p).astype(float), (rng.random(N) < p).astype(float)


def continuous_null(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Skewed scores in [0, 1] with a shared per-example component.
    shared = rng.beta(2, 5, size=N)
    return (
        np.clip(shared + rng.normal(0, 0.1, N), 0, 1),
        np.clip(shared + rng.normal(0, 0.1, N), 0, 1),
    )


@pytest.mark.parametrize("make", [binary_null, continuous_null])
def test_no_true_difference_ci_covers_zero_and_test_holds_size(make) -> None:  # type: ignore[no-untyped-def]
    rng = np.random.default_rng(10)
    covered = rejected = 0
    for _ in range(SIMS):
        m = run(*make(rng))
        covered += m.ci_low <= 0 <= m.ci_high
        rejected += m.verdict != "no detectable difference"
    assert 0.92 <= covered / SIMS <= 0.98
    assert 0.02 <= rejected / SIMS <= 0.08


def test_real_difference_is_detected() -> None:
    rng = np.random.default_rng(11)
    detected = 0
    for _ in range(SIMS // 4):
        base, cand = continuous_null(rng)
        m = run(base, np.clip(cand + 0.06, 0, 1))
        detected += m.verdict == "better"
    assert detected / (SIMS // 4) >= 0.95


def test_true_effect_equal_to_mde_is_detected_about_80_percent() -> None:
    # Calibrate the effect to the MDE reported on a pilot dataset, then check power.
    rng = np.random.default_rng(12)
    pilot = run(*continuous_null(rng))
    effect = pilot.mde
    detected = 0
    for _ in range(SIMS):
        base, cand = continuous_null(rng)
        detected += run(base, cand + effect).verdict == "better"
    assert 0.72 <= detected / SIMS <= 0.88


def test_cluster_bootstrap_keeps_coverage_with_correlated_groups() -> None:
    # 50 questions x 4 samples each. Within a question the run difference is shared,
    # so rows are strongly correlated. Treating rows as independent would undercover.
    # (With ~25 groups the percentile bootstrap covers ~92%: the usual small-sample
    # shortfall, documented in docs/statistics.md. The sign-flip test is exact.)
    rng = np.random.default_rng(13)
    n_rows = 200
    groups = [str(i // 4) for i in range(n_rows)]
    covered_grouped = covered_naive = rejected_grouped = 0
    for _ in range(SIMS):
        question_effect = np.repeat(rng.normal(0, 0.2, n_rows // 4), 4)
        base = rng.normal(0, 0.05, n_rows)
        cand = base + question_effect + rng.normal(0, 0.05, n_rows)
        grouped = run(base, cand, groups)
        naive = run(base, cand)
        covered_grouped += grouped.ci_low <= 0 <= grouped.ci_high
        covered_naive += naive.ci_low <= 0 <= naive.ci_high
        rejected_grouped += grouped.verdict != "no detectable difference"
    assert 0.92 <= covered_grouped / SIMS <= 0.98
    assert 0.02 <= rejected_grouped / SIMS <= 0.08
    assert covered_naive / SIMS < 0.80
