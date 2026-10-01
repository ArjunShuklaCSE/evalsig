"""From results files and rows to aligned metric arrays, with clear input errors.

A row is one test example: a mapping with an ``id``, one or more metric columns, and
optionally a ``group`` (several rows that belong to the same question).
"""

from __future__ import annotations

import csv
import io
import json
import math
from collections.abc import Collection, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

from .core import (
    Comparison,
    EvalsigError,
    Kind,
    Summary,
    compare_arrays,
    summarize_arrays,
)

Row = Mapping[str, Any]

ID = "id"
GROUP = "group"
# Error messages list at most this many ids.
MAX_LISTED = 10
_MISSING_STRINGS = {"", "na", "n/a", "nan", "null", "none"}


def load_rows(path: str | Path) -> list[dict[str, Any]]:
    """Read a ``.jsonl`` (one JSON object per line) or ``.csv`` results file."""
    path = Path(path)
    if not path.is_file():
        raise EvalsigError(f"File not found: {path}")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise EvalsigError(f"{path} is not a UTF-8 text file.") from None

    suffix = path.suffix.lower()
    if suffix in (".jsonl", ".ndjson"):
        rows = _parse_jsonl(text, path)
    elif suffix == ".csv":
        rows = [dict(row) for row in csv.DictReader(io.StringIO(text))]
    else:
        raise EvalsigError(f"{path}: unsupported file type '{suffix}'. Use .jsonl or .csv.")
    if not rows:
        raise EvalsigError(f"{path} is empty: it has no result rows.")
    return rows


def compare(
    baseline: Sequence[Row],
    candidate: Sequence[Row],
    metrics: Sequence[str] | None = None,
    *,
    metric_types: Mapping[str, Kind] | None = None,
    lower_is_better: Collection[str] = (),
    allow_partial: bool = False,
    alpha: float = 0.05,
    resamples: int = 10_000,
    seed: int = 0,
) -> Comparison:
    """Compare a candidate run against a baseline run on the same test examples.

    Rows are joined on ``id``. If the ids differ, this raises unless ``allow_partial``
    is set, in which case only shared ids are compared and a warning is recorded.
    ``metrics`` defaults to every numeric column present in both runs.
    """
    base = _index(baseline, "baseline")
    cand = _index(candidate, "candidate")
    warnings: list[str] = []

    only_base = [i for i in base if i not in cand]
    only_cand = [i for i in cand if i not in base]
    shared = [i for i in base if i in cand]
    if only_base or only_cand:
        detail = _id_mismatch(only_base, only_cand)
        if not shared:
            raise EvalsigError(f"The two runs have no ids in common. {detail}")
        if not allow_partial:
            raise EvalsigError(
                f"The two runs do not contain the same ids. {detail} "
                "Use --allow-partial (allow_partial=True) to compare only the shared ids."
            )
        warnings.append(f"Comparing only the {len(shared)} shared ids. {detail}")

    base_rows = [base[i] for i in shared]
    cand_rows = [cand[i] for i in shared]
    if metrics:
        names = list(dict.fromkeys(metrics))
    else:
        base_metrics = _numeric_columns(base_rows)
        cand_metrics = _numeric_columns(cand_rows)
        names = [m for m in base_metrics if m in cand_metrics]
        one_sided = [m for m in base_metrics + cand_metrics if m not in names]
        if one_sided:
            warnings.append(f"Skipped metrics found in only one run: {', '.join(one_sided)}.")
        if not names:
            raise EvalsigError("No numeric metric columns are shared by the two runs.")

    result = compare_arrays(
        {m: _column(base_rows, shared, m, "baseline") for m in names},
        {m: _column(cand_rows, shared, m, "candidate") for m in names},
        kinds=metric_types,
        groups=_groups(base_rows, cand_rows, shared),
        lower_is_better=lower_is_better,
        alpha=alpha,
        resamples=resamples,
        seed=seed,
    )
    return replace(result, warnings=tuple(warnings))


def summarize(
    rows: Sequence[Row],
    metrics: Sequence[str] | None = None,
    *,
    metric_types: Mapping[str, Kind] | None = None,
    alpha: float = 0.05,
    resamples: int = 10_000,
    seed: int = 0,
) -> Summary:
    """Mean of each metric in one run, with a 95% (1 - alpha) confidence interval."""
    index = _index(rows, "run")
    ids = list(index)
    run_rows = list(index.values())
    names = list(dict.fromkeys(metrics)) if metrics else _numeric_columns(run_rows)
    if not names:
        raise EvalsigError("No numeric metric columns found.")
    return summarize_arrays(
        {m: _column(run_rows, ids, m, "run") for m in names},
        kinds=metric_types,
        groups=_groups(run_rows, None, ids),
        alpha=alpha,
        resamples=resamples,
        seed=seed,
    )


def _parse_jsonl(text: str, path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise EvalsigError(f"{path}, line {number}: not valid JSON ({error.msg}).") from None
        if not isinstance(row, dict):
            raise EvalsigError(f"{path}, line {number}: expected a JSON object.")
        rows.append(row)
    return rows


def _index(rows: Sequence[Row], label: str) -> dict[str, Row]:
    """Rows keyed by id, in file order. Ids must be present and unique."""
    if not rows:
        raise EvalsigError(f"The {label} has no rows.")
    index: dict[str, Row] = {}
    duplicates: list[str] = []
    for number, row in enumerate(rows, start=1):
        raw = row.get(ID)
        if raw is None or str(raw).strip() == "":
            raise EvalsigError(f"The {label}'s row {number} has no '{ID}' value.")
        key = str(raw)
        if key in index:
            duplicates.append(key)
        index[key] = row
    if duplicates:
        raise EvalsigError(
            f"The {label} has duplicate ids: {_listing(duplicates)}. Each row needs a unique "
            f"'{ID}'; use a '{GROUP}' column to mark rows that belong to the same question."
        )
    return index


def _to_number(value: Any) -> float | None:
    """A float, NaN for a missing value, or None if the value is not a number."""
    if value is None:
        return math.nan
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if text in ("true", "false"):
            return float(text == "true")
        if text in _MISSING_STRINGS:
            return math.nan
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _numeric_columns(rows: Sequence[Row]) -> list[str]:
    """Columns (other than id and group) whose values are all numbers or missing."""
    columns = dict.fromkeys(key for row in rows for key in row if key not in (ID, GROUP))
    numeric = []
    for column in columns:
        values = [_to_number(row.get(column)) for row in rows]
        if all(v is not None for v in values) and any(
            v is not None and not math.isnan(v) for v in values
        ):
            numeric.append(column)
    return numeric


def _column(rows: Sequence[Row], ids: Sequence[str], name: str, label: str) -> list[float]:
    if not any(name in row for row in rows):
        raise EvalsigError(f"The {label} has no column '{name}'.")
    values: list[float] = []
    not_numbers: list[str] = []
    missing: list[str] = []
    for row_id, row in zip(ids, rows, strict=True):
        value = _to_number(row.get(name))
        if value is None:
            not_numbers.append(row_id)
        elif not math.isfinite(value):
            missing.append(row_id)
        values.append(math.nan if value is None else value)
    if not_numbers:
        raise EvalsigError(
            f"Metric '{name}' in the {label} is not a number for ids: {_listing(not_numbers)}."
        )
    if missing:
        raise EvalsigError(
            f"Metric '{name}' in the {label} is missing, NaN or infinite for ids: "
            f"{_listing(missing)}. Fix or remove those rows."
        )
    return values


def _groups(
    base_rows: Sequence[Row], cand_rows: Sequence[Row] | None, ids: Sequence[str]
) -> list[str] | None:
    """Group labels, taken from whichever run has a group column. Both must agree."""
    runs = [rows for rows in (base_rows, cand_rows) if rows and any(GROUP in r for r in rows)]
    if not runs:
        return None
    labels = []
    for rows in runs:
        run_labels = [row.get(GROUP) for row in rows]
        blank = [i for i, g in zip(ids, run_labels, strict=True) if g is None or str(g) == ""]
        if blank:
            raise EvalsigError(f"Rows are missing a '{GROUP}' value for ids: {_listing(blank)}.")
        labels.append([str(g) for g in run_labels])
    if len(labels) == 2:
        differ = [i for i, a, b in zip(ids, *labels, strict=True) if a != b]
        if differ:
            raise EvalsigError(
                f"Baseline and candidate assign different groups to ids: {_listing(differ)}."
            )
    return labels[0]


def _id_mismatch(only_base: Sequence[str], only_cand: Sequence[str]) -> str:
    parts = []
    if only_base:
        parts.append(f"Missing from candidate ({len(only_base)}): {_listing(only_base)}.")
    if only_cand:
        parts.append(f"Missing from baseline ({len(only_cand)}): {_listing(only_cand)}.")
    return " ".join(parts)


def _listing(ids: Sequence[str]) -> str:
    shown = ", ".join(ids[:MAX_LISTED])
    more = len(ids) - MAX_LISTED
    return f"{shown} and {more} more" if more > 0 else shown
