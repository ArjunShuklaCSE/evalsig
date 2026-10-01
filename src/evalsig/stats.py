"""Statistical primitives.

Everything works on *cluster sums*: per-group sums of a per-row quantity, plus the
number of rows in each group. Ungrouped data is the special case of one row per
group, so the plain and the cluster versions of each method share one code path.

All randomness comes from the ``numpy.random.Generator`` passed in, so results are
reproducible from a seed.
"""

from __future__ import annotations

from collections.abc import Sequence
from statistics import NormalDist

import numpy as np
from numpy.typing import NDArray
from scipy.stats import binomtest

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.intp]

# Upper bound on array elements per resampling batch, to keep memory flat for large n.
_BATCH_ELEMENTS = 2_000_000


def cluster_sums(values: FloatArray, codes: IntArray) -> tuple[FloatArray, FloatArray]:
    """Per-group sum of ``values`` and per-group row count. ``codes`` are 0..G-1."""
    sums = np.bincount(codes, weights=values).astype(np.float64)
    counts = np.bincount(codes).astype(np.float64)
    return sums, counts


def wilson_interval(successes: int, n: int, alpha: float) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (no continuity correction)."""
    ci = binomtest(successes, n).proportion_ci(confidence_level=1 - alpha, method="wilson")
    return float(ci.low), float(ci.high)


def bootstrap_means(
    sums: FloatArray, counts: FloatArray, resamples: int, rng: np.random.Generator
) -> FloatArray:
    """Row-level mean for each bootstrap resample of groups drawn with replacement.

    With one row per group this is the ordinary nonparametric bootstrap of the mean.
    """
    n_groups = len(sums)
    batch = max(1, _BATCH_ELEMENTS // n_groups)
    means = np.empty(resamples, dtype=np.float64)
    for start in range(0, resamples, batch):
        stop = min(start + batch, resamples)
        idx = rng.integers(0, n_groups, size=(stop - start, n_groups))
        means[start:stop] = sums[idx].sum(axis=1) / counts[idx].sum(axis=1)
    return means


def percentile_interval(samples: FloatArray, alpha: float) -> tuple[float, float]:
    """Equal-tailed percentile interval of bootstrap samples."""
    low, high = np.quantile(samples, [alpha / 2, 1 - alpha / 2])
    return float(low), float(high)


def sign_flip_pvalue(sums: FloatArray, resamples: int, rng: np.random.Generator) -> float:
    """Two-sided paired sign-flip permutation test that the mean difference is zero.

    ``sums`` are per-group sums of paired differences (candidate - baseline). Under the
    null hypothesis the two runs are exchangeable within each group, so each group's sum
    is equally likely to have either sign. The statistic is |sum of signed group sums|.

    Groups whose sum is exactly zero cannot change the statistic and are dropped. If the
    remaining k groups allow 2**k <= ``resamples`` sign patterns, all are enumerated and
    the p-value is exact. Otherwise it is a Monte Carlo estimate with the (hits + 1) /
    (resamples + 1) correction, so it is never zero.
    """
    nonzero = sums[sums != 0]
    k = len(nonzero)
    if k == 0:
        return 1.0
    observed = abs(float(nonzero.sum()))
    # Sums of the same numbers in a different order can differ in the last bits.
    tolerance = 1e-9 * float(np.abs(nonzero).sum())

    if 2**k <= resamples:
        bits = (np.arange(2**k)[:, None] >> np.arange(k)) & 1
        stats = np.abs((bits * 2 - 1) @ nonzero)
        return float(np.mean(stats >= observed - tolerance))

    hits = 0
    batch = max(1, _BATCH_ELEMENTS // k)
    for start in range(0, resamples, batch):
        size = min(batch, resamples - start)
        signs = rng.integers(0, 2, size=(size, k)) * 2 - 1
        hits += int(np.count_nonzero(np.abs(signs @ nonzero) >= observed - tolerance))
    return (hits + 1) / (resamples + 1)


def mcnemar_exact(fixed: int, broke: int) -> float:
    """Exact two-sided McNemar test: binomial test of the discordant pairs at p = 0.5."""
    if fixed + broke == 0:
        return 1.0
    return float(binomtest(fixed, fixed + broke, 0.5).pvalue)


def holm(pvalues: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, returned in the input order."""
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running_max = 0.0
    for rank, i in enumerate(order):
        running_max = max(running_max, min(1.0, (m - rank) * pvalues[i]))
        adjusted[i] = running_max
    return adjusted


def minimum_detectable_effect(se: float, alpha: float, power: float = 0.8) -> float:
    """Smallest true difference a two-sided test at ``alpha`` detects with ``power``.

    (z_{1-alpha/2} + z_{power}) * SE, which is about 2.8 * SE at alpha 0.05, 80% power.
    """
    z = NormalDist().inv_cdf
    return (z(1 - alpha / 2) + z(power)) * se
