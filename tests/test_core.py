"""compare_arrays / summarize_arrays: verdicts, notes, edge cases, serialization."""

import json

import numpy as np
import pytest

from evalsig import EvalsigError, compare_arrays, summarize_arrays
from evalsig.core import NO_DIFFERENCE, detect_kind


def binary_runs(
    fixed: int, broke: int, both_pass: int, both_fail: int
) -> tuple[list[int], list[int]]:
    base = [0] * fixed + [1] * broke + [1] * both_pass + [0] * both_fail
    cand = [1] * fixed + [0] * broke + [1] * both_pass + [0] * both_fail
    return base, cand


def test_detect_kind() -> None:
    assert detect_kind([0, 1, 1]) == "binary"
    assert detect_kind([True, False]) == "binary"
    assert detect_kind([0.0, 1.0]) == "binary"
    assert detect_kind([0, 0.5, 1]) == "continuous"


def test_clear_improvement_is_better() -> None:
    base, cand = binary_runs(fixed=30, broke=2, both_pass=60, both_fail=8)
    m = compare_arrays({"acc": base}, {"acc": cand}).metrics[0]
    assert m.kind == "binary"
    assert (m.fixed, m.broke) == (30, 2)
    assert m.diff == pytest.approx(0.28)
    assert m.ci_low > 0
    assert m.verdict == "better"
    assert m.mcnemar_p is not None and m.mcnemar_p < 1e-5


def test_clear_regression_is_worse() -> None:
    base, cand = binary_runs(fixed=2, broke=30, both_pass=60, both_fail=8)
    assert compare_arrays({"acc": base}, {"acc": cand}).metrics[0].verdict == "worse"


def test_small_difference_is_not_called() -> None:
    base, cand = binary_runs(fixed=9, broke=3, both_pass=120, both_fail=19)
    m = compare_arrays({"acc": base}, {"acc": cand}).metrics[0]
    assert m.verdict == NO_DIFFERENCE
    assert m.ci_low < 0 < m.ci_high
    assert m.mde > abs(m.diff)


def test_identical_runs() -> None:
    values = [0.2, 0.9, 0.4] * 10
    m = compare_arrays({"score": values}, {"score": values}).metrics[0]
    assert (m.diff, m.ci_low, m.ci_high, m.p_value, m.mde) == (0, 0, 0, 1, 0)
    assert m.verdict == NO_DIFFERENCE
    assert "identical" in m.notes[0]


@pytest.mark.parametrize("value", [0, 1])
def test_all_pass_or_all_fail(value: int) -> None:
    m = compare_arrays({"acc": [value] * 30}, {"acc": [value] * 30}).metrics[0]
    assert m.verdict == NO_DIFFERENCE
    assert (m.fixed, m.broke, m.mcnemar_p) == (0, 0, 1.0)
    s = summarize_arrays({"acc": [value] * 30}).metrics[0]
    assert s.mean == value
    assert s.ci_low <= value <= s.ci_high


def test_single_example() -> None:
    m = compare_arrays({"acc": [0]}, {"acc": [1]}).metrics[0]
    assert m.diff == 1
    assert m.p_value == 1.0  # one sign flip pattern each way: no evidence
    assert m.verdict == NO_DIFFERENCE
    assert any("Only 1 examples" in note for note in m.notes)
    s = summarize_arrays({"acc": [1]}).metrics[0]
    assert s.method == "wilson"
    assert s.ci_high == 1.0


def test_holm_is_applied_to_several_metrics() -> None:
    rng = np.random.default_rng(0)
    base = {"a": rng.normal(size=80), "b": rng.normal(size=80)}
    cand = {"a": base["a"] + 0.25 + rng.normal(0, 0.5, 80), "b": base["b"] + rng.normal(0, 1, 80)}
    result = compare_arrays(base, cand)
    assert result.correction == "holm"
    a, b = result.metrics
    assert a.p_adjusted == pytest.approx(min(1.0, 2 * a.p_value))
    assert b.p_adjusted >= b.p_value
    assert compare_arrays({"a": base["a"]}, {"a": cand["a"]}).correction == "none"


def test_metric_result_does_not_depend_on_other_metrics() -> None:
    rng = np.random.default_rng(1)
    base = {"a": rng.normal(size=50), "b": rng.normal(size=50)}
    cand = {"a": rng.normal(size=50), "b": rng.normal(size=50)}
    alone = compare_arrays({"a": base["a"]}, {"a": cand["a"]}).metrics[0]
    together = compare_arrays(base, cand).metrics[0]
    assert (alone.ci_low, alone.ci_high, alone.p_value) == (
        together.ci_low,
        together.ci_high,
        together.p_value,
    )


def test_same_seed_same_output_different_seed_different_output() -> None:
    rng = np.random.default_rng(2)
    base, cand = {"x": rng.normal(size=60)}, {"x": rng.normal(size=60)}
    assert compare_arrays(base, cand).to_json() == compare_arrays(base, cand).to_json()
    assert (
        compare_arrays(base, cand, seed=1).metrics[0].ci_low
        != compare_arrays(base, cand).metrics[0].ci_low
    )


def test_grouped_comparison_skips_mcnemar_and_counts_groups() -> None:
    base, cand = binary_runs(fixed=20, broke=5, both_pass=40, both_fail=15)
    groups = [str(i // 5) for i in range(80)]
    m = compare_arrays({"acc": base}, {"acc": cand}, groups=groups).metrics[0]
    assert m.n == 80 and m.n_groups == 16
    assert m.mcnemar_p is None
    assert any("grouped" in note for note in m.notes)
    assert any("Only 16 groups" in note for note in m.notes)


def test_kind_override() -> None:
    m = compare_arrays({"s": [0, 1, 1]}, {"s": [1, 1, 1]}, kinds={"s": "continuous"}).metrics[0]
    assert m.kind == "continuous" and m.fixed is None
    with pytest.raises(EvalsigError, match="declared binary"):
        compare_arrays({"s": [0.5]}, {"s": [1]}, kinds={"s": "binary"})


def test_summary_methods() -> None:
    rng = np.random.default_rng(3)
    s = summarize_arrays({"acc": [1, 0, 1, 1] * 25, "score": rng.random(100)})
    acc, score = s.metrics
    assert (acc.method, acc.mean) == ("wilson", 0.75)
    assert score.method == "bootstrap"
    assert score.ci_low < score.mean < score.ci_high
    grouped = summarize_arrays({"acc": [1, 0, 1, 1] * 25}, groups=[str(i // 4) for i in range(100)])
    assert grouped.metrics[0].method == "cluster bootstrap"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"baseline": {"a": [1.0, np.nan]}, "candidate": {"a": [1.0, 1.0]}}, "NaN"),
        ({"baseline": {"a": [1.0, 2.0]}, "candidate": {"a": [1.0]}}, "expected 2"),
        ({"baseline": {"a": []}, "candidate": {"a": []}}, "empty"),
        ({"baseline": {}, "candidate": {}}, "No metrics"),
        ({"baseline": {"a": [1]}, "candidate": {"b": [1]}}, "do not match"),
        ({"baseline": {"a": ["x"]}, "candidate": {"a": [1]}}, "not numbers"),
        ({"baseline": {"a": [1]}, "candidate": {"a": [1]}, "alpha": 1.5}, "alpha"),
        ({"baseline": {"a": [1]}, "candidate": {"a": [1]}, "resamples": 5}, "resamples"),
        ({"baseline": {"a": [1]}, "candidate": {"a": [1]}, "groups": ["g", "h"]}, "group labels"),
    ],
)
def test_bad_input_raises_clear_error(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(EvalsigError, match=message):
        compare_arrays(**kwargs)  # type: ignore[arg-type]


def test_to_dict_and_json() -> None:
    result = compare_arrays({"acc": [0, 1, 1]}, {"acc": [1, 1, 1]})
    data = json.loads(result.to_json())
    assert data == json.loads(json.dumps(result.to_dict()))
    assert data["metrics"][0]["metric"] == "acc"
    assert data["seed"] == 0 and data["resamples"] == 10_000


def test_note_when_holm_correction_changes_the_verdict() -> None:
    base, cand = binary_runs(fixed=12, broke=3, both_pass=70, both_fail=66)
    null = [0.0, 1.0] * 75 + [0.5]
    alone = compare_arrays({"acc": base}, {"acc": cand}).metrics[0]
    assert alone.verdict == "better" and alone.ci_low > 0
    together = compare_arrays(
        {"acc": base, "b": null, "c": null}, {"acc": cand, "b": null[::-1], "c": null[::-1]}
    ).metrics[0]
    assert together.verdict == NO_DIFFERENCE and together.ci_low > 0
    assert any("not after correcting for 3 metrics" in note for note in together.notes)
