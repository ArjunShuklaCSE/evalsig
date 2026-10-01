"""Property-based checks on invariants that must hold for any data."""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from evalsig import compare_arrays, stats

values = st.floats(min_value=-100, max_value=100, allow_nan=False)
RESAMPLES = 200


@st.composite
def paired_runs(draw: st.DrawFn) -> tuple[list[float], list[float]]:
    n = draw(st.integers(min_value=1, max_value=40))
    return draw(st.lists(values, min_size=n, max_size=n)), draw(
        st.lists(values, min_size=n, max_size=n)
    )


@settings(max_examples=60, deadline=None)
@given(paired_runs())
def test_swapping_runs_mirrors_the_result(runs: tuple[list[float], list[float]]) -> None:
    base, cand = runs
    fwd = compare_arrays({"m": base}, {"m": cand}, resamples=RESAMPLES).metrics[0]
    rev = compare_arrays({"m": cand}, {"m": base}, resamples=RESAMPLES).metrics[0]
    tol = 1e-9 * (1 + max(map(abs, base + cand)))
    assert abs(fwd.diff + rev.diff) <= tol
    assert abs(fwd.ci_low + rev.ci_high) <= tol
    assert abs(fwd.ci_high + rev.ci_low) <= tol
    assert fwd.p_value == rev.p_value
    assert {fwd.verdict, rev.verdict} in ({"no detectable difference"}, {"better", "worse"})


@settings(max_examples=60, deadline=None)
@given(paired_runs(), st.floats(min_value=-10, max_value=10))
def test_shifting_both_runs_changes_nothing(
    runs: tuple[list[float], list[float]], shift: float
) -> None:
    base, cand = runs
    a = compare_arrays({"m": base}, {"m": cand}, resamples=RESAMPLES).metrics[0]
    b = compare_arrays(
        {"m": [x + shift for x in base]}, {"m": [x + shift for x in cand]}, resamples=RESAMPLES
    ).metrics[0]
    assert np.allclose([a.diff, a.ci_low, a.ci_high], [b.diff, b.ci_low, b.ci_high], atol=1e-6)


@settings(max_examples=60, deadline=None)
@given(paired_runs())
def test_result_ranges(runs: tuple[list[float], list[float]]) -> None:
    base, cand = runs
    m = compare_arrays({"m": base}, {"m": cand}, resamples=RESAMPLES).metrics[0]
    assert 0 < m.p_value <= 1
    assert m.ci_low <= m.ci_high
    assert m.mde >= 0


@given(st.lists(st.floats(min_value=0, max_value=1), min_size=1, max_size=20))
def test_holm_never_lowers_and_keeps_order(pvalues: list[float]) -> None:
    adjusted = stats.holm(pvalues)
    assert all(p <= a <= 1 for p, a in zip(pvalues, adjusted, strict=True))
    order = sorted(range(len(pvalues)), key=lambda i: pvalues[i])
    assert all(adjusted[i] <= adjusted[j] for i, j in zip(order, order[1:], strict=False))


@given(st.integers(min_value=1, max_value=500), st.data())
def test_wilson_contains_observed_rate(n: int, data: st.DataObject) -> None:
    k = data.draw(st.integers(min_value=0, max_value=n))
    low, high = stats.wilson_interval(k, n, 0.05)
    assert 0 <= low <= k / n <= high <= 1
