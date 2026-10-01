"""Paired comparison of two runs and summary of one run, from aligned per-row arrays."""

from __future__ import annotations

import json
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from . import stats

Kind = Literal["binary", "continuous"]

BETTER = "better"
WORSE = "worse"
NO_DIFFERENCE = "no detectable difference"

# Below this many examples (or groups) bootstrap intervals are noticeably too narrow.
SMALL_SAMPLE = 20
MIN_RESAMPLES = 100


class EvalsigError(ValueError):
    """Invalid input. The message is written to be shown to the user as-is."""


@dataclass(frozen=True)
class MetricComparison:
    metric: str
    kind: Kind
    higher_is_better: bool
    n: int
    n_groups: int | None
    baseline: float
    candidate: float
    diff: float
    ci_low: float
    ci_high: float
    p_value: float
    p_adjusted: float
    mde: float
    verdict: str
    fixed: int | None
    broke: int | None
    mcnemar_p: float | None
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Comparison:
    metrics: tuple[MetricComparison, ...]
    alpha: float
    resamples: int
    seed: int
    correction: Literal["holm", "none"]
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


@dataclass(frozen=True)
class MetricSummary:
    metric: str
    kind: Kind
    n: int
    n_groups: int | None
    mean: float
    ci_low: float
    ci_high: float
    method: Literal["wilson", "bootstrap", "cluster bootstrap"]
    notes: tuple[str, ...]


@dataclass(frozen=True)
class Summary:
    metrics: tuple[MetricSummary, ...]
    alpha: float
    resamples: int
    seed: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def detect_kind(values: ArrayLike) -> Kind:
    """``binary`` if every value is 0 or 1 (booleans included), else ``continuous``."""
    return (
        "binary"
        if bool(np.isin(np.asarray(values, dtype=np.float64), (0.0, 1.0)).all())
        else "continuous"
    )


def compare_arrays(
    baseline: Mapping[str, ArrayLike],
    candidate: Mapping[str, ArrayLike],
    kinds: Mapping[str, Kind] | None = None,
    groups: Sequence[str] | None = None,
    lower_is_better: Collection[str] = (),
    *,
    alpha: float = 0.05,
    resamples: int = 10_000,
    seed: int = 0,
) -> Comparison:
    """Compare two runs whose arrays are already aligned row by row.

    Each metric gets a fresh generator seeded with ``seed``, so a metric's result does
    not depend on which other metrics are compared alongside it. Metrics named in
    ``lower_is_better`` (latency, cost, error rate) are "better" when they go down.
    """
    _check_settings(alpha, resamples)
    if not baseline:
        raise EvalsigError("No metrics to compare.")
    _check_known("lower_is_better", lower_is_better, baseline)
    _check_known("metric type", kinds or {}, baseline)
    if list(baseline) != list(candidate):
        raise EvalsigError(
            f"Baseline metrics {list(baseline)} do not match candidate metrics {list(candidate)}."
        )
    kinds = kinds or {}
    n = len(np.asarray(next(iter(baseline.values()))))
    codes, n_groups = _group_codes(groups, n)

    raw = [
        _compare_metric(
            name,
            baseline[name],
            candidate[name],
            kinds.get(name),
            codes,
            n_groups,
            alpha,
            resamples,
            seed,
        )
        for name in baseline
    ]
    multiple = len(raw) > 1
    adjusted = stats.holm([r.p_value for r in raw]) if multiple else [raw[0].p_value]
    metrics = tuple(
        _finish(r, p, alpha, len(raw), higher_is_better=r.metric not in lower_is_better)
        for r, p in zip(raw, adjusted, strict=True)
    )
    return Comparison(
        metrics=metrics,
        alpha=alpha,
        resamples=resamples,
        seed=seed,
        correction="holm" if multiple else "none",
    )


def summarize_arrays(
    values: Mapping[str, ArrayLike],
    kinds: Mapping[str, Kind] | None = None,
    groups: Sequence[str] | None = None,
    *,
    alpha: float = 0.05,
    resamples: int = 10_000,
    seed: int = 0,
) -> Summary:
    """Mean of each metric for a single run, with a confidence interval."""
    _check_settings(alpha, resamples)
    if not values:
        raise EvalsigError("No metrics to summarize.")
    _check_known("metric type", kinds or {}, values)
    kinds = kinds or {}
    n = len(np.asarray(next(iter(values.values()))))
    codes, n_groups = _group_codes(groups, n)
    metrics = tuple(
        _summarize_metric(
            name, values[name], kinds.get(name), codes, n_groups, alpha, resamples, seed
        )
        for name in values
    )
    return Summary(metrics=metrics, alpha=alpha, resamples=resamples, seed=seed)


def _compare_metric(
    name: str,
    baseline: ArrayLike,
    candidate: ArrayLike,
    kind: Kind | None,
    codes: stats.IntArray,
    n_groups: int | None,
    alpha: float,
    resamples: int,
    seed: int,
) -> MetricComparison:
    b = _as_float(name, baseline, len(codes))
    c = _as_float(name, candidate, len(codes))
    kind = _resolve_kind(name, np.concatenate([b, c]), kind)
    diffs = c - b
    sums, counts = stats.cluster_sums(diffs, codes)

    rng = np.random.default_rng(seed)
    boot = stats.bootstrap_means(sums, counts, resamples, rng)
    ci_low, ci_high = stats.percentile_interval(boot, alpha)
    p_value = stats.sign_flip_pvalue(sums, resamples, rng)
    mde = stats.minimum_detectable_effect(float(boot.std(ddof=1)), alpha)

    notes: list[str] = []
    fixed = broke = None
    mcnemar_p = None
    if kind == "binary":
        fixed = int(np.count_nonzero((b == 0) & (c == 1)))
        broke = int(np.count_nonzero((b == 1) & (c == 0)))
        if n_groups is None:
            mcnemar_p = stats.mcnemar_exact(fixed, broke)
        else:
            notes.append(
                "McNemar's test not reported: it assumes independent rows, but rows are grouped."
            )
    if not diffs.any():
        notes.append("The two runs are identical on this metric.")
    notes.extend(_small_sample_notes(len(sums), n_groups))

    return MetricComparison(
        metric=name,
        kind=kind,
        higher_is_better=True,
        n=len(b),
        n_groups=n_groups,
        baseline=float(b.mean()),
        candidate=float(c.mean()),
        diff=float(diffs.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        p_value=p_value,
        p_adjusted=p_value,
        mde=mde,
        verdict=NO_DIFFERENCE,
        fixed=fixed,
        broke=broke,
        mcnemar_p=mcnemar_p,
        notes=tuple(notes),
    )


def _summarize_metric(
    name: str,
    values: ArrayLike,
    kind: Kind | None,
    codes: stats.IntArray,
    n_groups: int | None,
    alpha: float,
    resamples: int,
    seed: int,
) -> MetricSummary:
    v = _as_float(name, values, len(codes))
    kind = _resolve_kind(name, v, kind)
    method: Literal["wilson", "bootstrap", "cluster bootstrap"]
    if kind == "binary" and n_groups is None:
        method = "wilson"
        ci_low, ci_high = stats.wilson_interval(int(v.sum()), len(v), alpha)
    else:
        method = "bootstrap" if n_groups is None else "cluster bootstrap"
        sums, counts = stats.cluster_sums(v, codes)
        boot = stats.bootstrap_means(sums, counts, resamples, np.random.default_rng(seed))
        ci_low, ci_high = stats.percentile_interval(boot, alpha)
    # Wilson is fine for small n; the bootstrap is not.
    notes = () if method == "wilson" else tuple(_small_sample_notes(int(codes.max()) + 1, n_groups))
    return MetricSummary(
        metric=name,
        kind=kind,
        n=len(v),
        n_groups=n_groups,
        mean=float(v.mean()),
        ci_low=ci_low,
        ci_high=ci_high,
        method=method,
        notes=notes,
    )


def _finish(
    r: MetricComparison, p_adjusted: float, alpha: float, n_metrics: int, *, higher_is_better: bool
) -> MetricComparison:
    """Set the adjusted p-value and verdict, and explain when interval and verdict disagree."""
    verdict = _verdict(r.diff, p_adjusted, alpha, higher_is_better=higher_is_better)
    excludes_zero = r.ci_low > 0 or r.ci_high < 0
    notes = list(r.notes)
    if excludes_zero and verdict == NO_DIFFERENCE:
        if r.p_value < alpha <= p_adjusted:
            notes.append(
                f"Significant on its own (p = {r.p_value:.3f}) but not after correcting for "
                f"{n_metrics} metrics. The interval shown is not corrected."
            )
        else:
            notes.append("Borderline: the interval excludes 0 but the permutation test does not.")
    elif not excludes_zero and verdict != NO_DIFFERENCE:
        notes.append("Borderline: the permutation test is significant but the interval touches 0.")
    return replace(
        r,
        higher_is_better=higher_is_better,
        p_adjusted=p_adjusted,
        verdict=verdict,
        notes=tuple(notes),
    )


def _verdict(diff: float, p_adjusted: float, alpha: float, *, higher_is_better: bool) -> str:
    if p_adjusted < alpha and diff != 0:
        return BETTER if (diff > 0) == higher_is_better else WORSE
    return NO_DIFFERENCE


def _check_known(what: str, names: Iterable[str], metrics: Mapping[str, object]) -> None:
    unknown = [name for name in names if name not in metrics]
    if unknown:
        raise EvalsigError(f"{what}: unknown metric(s) {', '.join(unknown)}.")


def _check_settings(alpha: float, resamples: int) -> None:
    if not 0 < alpha < 1:
        raise EvalsigError(f"alpha must be between 0 and 1, got {alpha}.")
    if resamples < MIN_RESAMPLES:
        raise EvalsigError(f"resamples must be at least {MIN_RESAMPLES}, got {resamples}.")


def _group_codes(groups: Sequence[str] | None, n: int) -> tuple[stats.IntArray, int | None]:
    """Map group labels to 0..G-1. Without groups, every row is its own group."""
    if n == 0:
        raise EvalsigError("No examples: the input is empty.")
    if groups is None:
        return np.arange(n, dtype=np.intp), None
    if len(groups) != n:
        raise EvalsigError(f"Got {len(groups)} group labels for {n} rows.")
    _, codes = np.unique(np.asarray(groups, dtype=str), return_inverse=True)
    codes = codes.ravel().astype(np.intp)
    return codes, int(codes.max()) + 1


def _as_float(name: str, values: ArrayLike, n: int) -> stats.FloatArray:
    try:
        array = np.asarray(values, dtype=np.float64)
    except (TypeError, ValueError):
        raise EvalsigError(f"Metric '{name}' has values that are not numbers.") from None
    if array.ndim != 1 or len(array) != n:
        raise EvalsigError(f"Metric '{name}' has {array.size} values, expected {n}.")
    bad = int(np.count_nonzero(~np.isfinite(array)))
    if bad:
        raise EvalsigError(f"Metric '{name}' has {bad} missing or non-finite values (NaN or inf).")
    return array


def _resolve_kind(name: str, values: stats.FloatArray, kind: Kind | None) -> Kind:
    detected = detect_kind(values)
    if kind is None:
        return detected
    if kind not in ("binary", "continuous"):
        raise EvalsigError(f"Metric type must be 'binary' or 'continuous', got '{kind}'.")
    if kind == "binary" and detected != "binary":
        raise EvalsigError(
            f"Metric '{name}' was declared binary but has values other than 0 and 1."
        )
    return kind


def _small_sample_notes(n_units: int, n_groups: int | None) -> list[str]:
    if n_units >= SMALL_SAMPLE:
        return []
    units = "groups" if n_groups is not None else "examples"
    return [f"Only {n_units} {units}: intervals are less reliable below {SMALL_SAMPLE}."]
