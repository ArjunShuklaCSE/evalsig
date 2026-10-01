"""Each statistic checked against hand-computed values or scipy."""

import numpy as np
import pytest
from scipy import stats as sps

from evalsig import stats


def rng() -> np.random.Generator:
    return np.random.default_rng(123)


# --- Wilson ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [
        (5, 10, 0.236593, 0.763407),  # textbook value
        (0, 10, 0.0, 0.277533),  # z^2 / (n + z^2)
        (10, 10, 0.722467, 1.0),
    ],
)
def test_wilson_known_values(k: int, n: int, low: float, high: float) -> None:
    assert stats.wilson_interval(k, n, 0.05) == pytest.approx((low, high), abs=1e-6)


def test_wilson_matches_formula() -> None:
    k, n, z = 81, 151, sps.norm.ppf(0.975)
    p = k / n
    center = (p + z**2 / (2 * n)) / (1 + z**2 / n)
    half = z / (1 + z**2 / n) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    assert stats.wilson_interval(k, n, 0.05) == pytest.approx((center - half, center + half))


# --- Bootstrap ------------------------------------------------------------------------


def test_bootstrap_means_center_and_spread() -> None:
    x = rng().normal(size=200)
    sums, counts = stats.cluster_sums(x, np.arange(200))
    boot = stats.bootstrap_means(sums, counts, 20_000, rng())
    plug_in_se = x.std(ddof=0) / np.sqrt(len(x))
    assert boot.mean() == pytest.approx(x.mean(), abs=0.005)
    assert boot.std() == pytest.approx(plug_in_se, rel=0.03)


def test_bootstrap_batches_do_not_change_result(monkeypatch: pytest.MonkeyPatch) -> None:
    x = rng().normal(size=50)
    sums, counts = stats.cluster_sums(x, np.arange(50))
    whole = stats.bootstrap_means(sums, counts, 300, rng())
    monkeypatch.setattr(stats, "_BATCH_ELEMENTS", 50 * 7)  # forces 43 batches
    batched = stats.bootstrap_means(sums, counts, 300, rng())
    # Different batch shapes draw the same stream of integers in the same order.
    np.testing.assert_array_equal(whole, batched)


def test_cluster_bootstrap_resamples_whole_groups() -> None:
    # Two groups with constant values: every resample mean is a mix of whole groups.
    values = np.array([0.0, 0.0, 0.0, 1.0])
    sums, counts = stats.cluster_sums(values, np.array([0, 0, 0, 1]))
    boot = stats.bootstrap_means(sums, counts, 1000, rng())
    assert set(np.round(boot, 9)) <= {0.0, 0.25, 1.0}


def test_percentile_interval() -> None:
    samples = np.arange(1001, dtype=float)
    assert stats.percentile_interval(samples, 0.05) == (25.0, 975.0)


# --- Sign-flip permutation ------------------------------------------------------------


def test_sign_flip_exact_matches_scipy_permutation_test() -> None:
    diffs = rng().normal(0.3, 1, size=12)
    expected = sps.permutation_test(
        (diffs,),
        np.mean,
        permutation_type="samples",
        n_resamples=np.inf,
        alternative="two-sided",
    ).pvalue
    got = stats.sign_flip_pvalue(diffs, 10_000, rng())  # 2**12 <= 10_000: exact
    assert got == pytest.approx(expected)


@pytest.mark.parametrize(("fixed", "broke"), [(9, 3), (5, 0), (0, 4), (6, 6), (1, 0), (7, 2)])
def test_sign_flip_on_binary_equals_exact_mcnemar(fixed: int, broke: int) -> None:
    diffs = np.array([1.0] * fixed + [-1.0] * broke + [0.0] * 40)
    exact = sps.binomtest(fixed, fixed + broke, 0.5).pvalue
    assert stats.sign_flip_pvalue(diffs, 10_000, rng()) == pytest.approx(exact)
    assert stats.mcnemar_exact(fixed, broke) == pytest.approx(exact)


def test_sign_flip_monte_carlo_approximates_exact() -> None:
    diffs = np.array([1.0] * 24 + [-1.0] * 10)  # 2**34 patterns: Monte Carlo branch
    exact = sps.binomtest(24, 34, 0.5).pvalue
    assert stats.sign_flip_pvalue(diffs, 200_000, rng()) == pytest.approx(exact, abs=0.003)


def test_sign_flip_monte_carlo_never_zero() -> None:
    diffs = np.ones(60)
    assert stats.sign_flip_pvalue(diffs, 1000, rng()) == pytest.approx(1 / 1001)


def test_sign_flip_all_zero_is_one() -> None:
    assert stats.sign_flip_pvalue(np.zeros(10), 1000, rng()) == 1.0


def test_mcnemar_no_discordant_pairs() -> None:
    assert stats.mcnemar_exact(0, 0) == 1.0


# --- Holm and MDE ---------------------------------------------------------------------


def test_holm_known_values() -> None:
    # Sorted: 0.01*3 = 0.03, 0.03*2 = 0.06, 0.04*1 = 0.04 -> raised to 0.06 for monotonicity.
    assert stats.holm([0.01, 0.04, 0.03]) == pytest.approx([0.03, 0.06, 0.06])


def test_holm_caps_at_one_and_single_is_unchanged() -> None:
    assert stats.holm([0.5, 0.9]) == [1.0, 1.0]
    assert stats.holm([0.2]) == [0.2]


def test_mde_is_about_2_8_standard_errors() -> None:
    expected = (sps.norm.ppf(0.975) + sps.norm.ppf(0.8)) * 0.02
    assert stats.minimum_detectable_effect(0.02, 0.05) == pytest.approx(expected)
    assert stats.minimum_detectable_effect(1.0, 0.05) == pytest.approx(2.8, abs=0.01)
