"""Loading files and turning rows into a comparison: joins, detection, input errors."""

import json
from pathlib import Path

import pytest

from evalsig import EvalsigError, compare, load_rows, summarize


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> Path:
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def rows(
    scores: list[object], ids: list[object] | None = None, **extra: list[object]
) -> list[dict[str, object]]:
    ids = ids if ids is not None else list(range(len(scores)))
    out = [{"id": i, "score": s} for i, s in zip(ids, scores, strict=True)]
    for key, values in extra.items():
        for row, value in zip(out, values, strict=True):
            row[key] = value
    return out


# --- Files ----------------------------------------------------------------------------


def test_jsonl_and_csv_load_to_the_same_comparison(tmp_path: Path) -> None:
    base = [{"id": f"q{i}", "pass": i % 2, "f1": i / 10} for i in range(10)]
    cand = [{"id": f"q{i}", "pass": 1, "f1": i / 9} for i in range(10)]
    write_jsonl(tmp_path / "b.jsonl", base)
    (tmp_path / "c.csv").write_text(
        "id,pass,f1\n" + "".join(f"q{i},true,{i / 9}\n" for i in range(10)), encoding="utf-8"
    )
    from_files = compare(load_rows(tmp_path / "b.jsonl"), load_rows(tmp_path / "c.csv"))
    from_rows = compare(base, cand)
    assert from_files.to_json() == from_rows.to_json()
    assert [m.metric for m in from_rows.metrics] == ["pass", "f1"]
    assert [m.kind for m in from_rows.metrics] == ["binary", "continuous"]


def test_load_handles_bom_and_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "r.jsonl"
    path.write_text('﻿{"id": 1, "x": 1}\n\n{"id": 2, "x": 0}\n', encoding="utf-8")
    assert len(load_rows(path)) == 2


@pytest.mark.parametrize(
    ("name", "content", "message"),
    [
        ("empty.jsonl", "", "empty"),
        ("blank.jsonl", "\n\n", "empty"),
        ("header_only.csv", "id,score\n", "empty"),
        ("bad.jsonl", '{"id": 1}\n{oops\n', "line 2: not valid JSON"),
        ("list.jsonl", "[1, 2]\n", "line 1: expected a JSON object"),
        ("run.txt", "id,score\n", "unsupported file type"),
    ],
)
def test_bad_files(tmp_path: Path, name: str, content: str, message: str) -> None:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    with pytest.raises(EvalsigError, match=message):
        load_rows(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(EvalsigError, match="File not found"):
        load_rows(tmp_path / "nope.jsonl")


def test_binary_file(tmp_path: Path) -> None:
    path = tmp_path / "r.csv"
    path.write_bytes(b"\xff\xfe\x00\x81")
    with pytest.raises(EvalsigError, match="UTF-8"):
        load_rows(path)


# --- Joining on id --------------------------------------------------------------------


def test_rows_are_joined_by_id_not_position() -> None:
    base = rows([0, 1, 1], ids=["a", "b", "c"])
    cand = rows([1, 1, 0], ids=["c", "a", "b"])  # c=1, a=1, b=0
    m = compare(base, cand).metrics[0]
    assert (m.fixed, m.broke) == (1, 1)  # a fixed, b broke, c unchanged


def test_mismatched_ids_list_what_is_missing() -> None:
    base = rows([1, 1, 1, 1], ids=["a", "b", "c", "d"])
    cand = rows([1, 1, 1], ids=["a", "b", "e"])
    with pytest.raises(EvalsigError) as error:
        compare(base, cand)
    message = str(error.value)
    assert "Missing from candidate (2): c, d" in message
    assert "Missing from baseline (1): e" in message
    assert "--allow-partial" in message


def test_allow_partial_compares_shared_ids_with_warning() -> None:
    base = rows([0, 0, 1, 1], ids=["a", "b", "c", "d"])
    cand = rows([1, 1, 1], ids=["a", "b", "e"])
    result = compare(base, cand, allow_partial=True)
    assert result.metrics[0].n == 2
    assert "only the 2 shared ids" in result.warnings[0]


def test_no_shared_ids() -> None:
    with pytest.raises(EvalsigError, match="no ids in common"):
        compare(rows([1], ids=["a"]), rows([1], ids=["b"]), allow_partial=True)


def test_long_id_lists_are_truncated() -> None:
    with pytest.raises(EvalsigError, match="and 15 more"):
        compare(rows([1] * 30), rows([1] * 5))


@pytest.mark.parametrize(
    ("base", "message"),
    [
        (rows([1, 1], ids=["a", "a"]), "duplicate ids: a"),
        ([{"score": 1}], "row 1 has no 'id'"),
        ([{"id": "", "score": 1}], "row 1 has no 'id'"),
        ([], "has no rows"),
    ],
)
def test_bad_ids(base: list[dict[str, object]], message: str) -> None:
    with pytest.raises(EvalsigError, match=message):
        compare(base, rows([1, 1], ids=["a", "b"]))


# --- Metric columns -------------------------------------------------------------------


def test_metric_detection_skips_text_columns() -> None:
    base = rows([0.5, 0.7], output=["foo", "bar"], latency=[100, 120])
    cand = rows([0.6, 0.8], output=["baz", "qux"], latency=[90, 95])
    assert [m.metric for m in compare(base, cand).metrics] == ["score", "latency"]


def test_metrics_in_only_one_run_are_skipped_with_warning() -> None:
    base = rows([1, 0], extra=[1, 1])
    result = compare(base, rows([1, 1]))
    assert [m.metric for m in result.metrics] == ["score"]
    assert "only one run: extra" in result.warnings[0]


def test_explicit_metric_selection_and_errors() -> None:
    base = rows([1, 0], latency=[1, 2])
    cand = rows([1, 1], latency=[1, 1])
    assert [m.metric for m in compare(base, cand, ["latency"]).metrics] == ["latency"]
    with pytest.raises(EvalsigError, match="has no column 'nope'"):
        compare(base, cand, ["nope"])


def test_nan_and_missing_values_name_the_ids() -> None:
    base = rows([1.0, None, float("nan"), 0.5], ids=["a", "b", "c", "d"])
    with pytest.raises(EvalsigError, match="missing, NaN or infinite for ids: b, c"):
        compare(base, rows([1, 1, 1, 1], ids=["a", "b", "c", "d"]))


def test_csv_blank_cells_are_missing(tmp_path: Path) -> None:
    (tmp_path / "r.csv").write_text("id,score\na,1\nb,\n", encoding="utf-8")
    with pytest.raises(EvalsigError, match="ids: b"):
        summarize(load_rows(tmp_path / "r.csv"), ["score"])


def test_non_numeric_value_in_named_metric() -> None:
    with pytest.raises(EvalsigError, match="not a number for ids: 1"):
        compare(rows([1, "high"]), rows([1, 1]), ["score"])


def test_no_numeric_columns() -> None:
    with pytest.raises(EvalsigError, match="No numeric metric"):
        compare([{"id": 1, "out": "x"}], [{"id": 1, "out": "y"}])


def test_metric_type_and_direction_options() -> None:
    base = rows([0, 1, 1, 0] * 10, latency=[200.0] * 40)
    cand = rows([1, 1, 1, 0] * 10, latency=[150.0 + i % 3 for i in range(40)])
    result = compare(base, cand, metric_types={"score": "continuous"}, lower_is_better=["latency"])
    score, latency = result.metrics
    assert score.kind == "continuous"
    assert latency.verdict == "better" and not latency.higher_is_better
    with pytest.raises(EvalsigError, match="lower_is_better: unknown metric"):
        compare(base, cand, lower_is_better=["cost"])
    with pytest.raises(EvalsigError, match="metric type: unknown metric"):
        compare(base, cand, metric_types={"cost": "binary"})


# --- Groups ---------------------------------------------------------------------------


def test_group_column_enables_cluster_resampling() -> None:
    groups = [f"q{i // 5}" for i in range(50)]
    result = compare(rows([0] * 50, group=groups), rows([1] * 50, group=groups))
    m = result.metrics[0]
    assert (m.n, m.n_groups) == (50, 10)
    assert summarize(rows([1, 0] * 25, group=groups)).metrics[0].method == "cluster bootstrap"


def test_group_from_either_run_and_disagreement() -> None:
    base = rows([0, 1, 1, 0])
    cand = rows([1, 1, 1, 0], group=["x", "x", "y", "y"])
    assert compare(base, cand).metrics[0].n_groups == 2
    with pytest.raises(EvalsigError, match="different groups to ids: 3"):
        compare(rows([0, 1, 1, 0], group=["x", "x", "y", "z"]), cand)
    with pytest.raises(EvalsigError, match="missing a 'group' value for ids: 1"):
        compare(rows([0, 1], group=["x", None]), rows([0, 1]))


def test_summarize_rows() -> None:
    s = summarize(rows([1, 1, 0, 1]))
    assert s.metrics[0].mean == 0.75
    with pytest.raises(EvalsigError, match="No numeric"):
        summarize([{"id": 1, "out": "x"}])
