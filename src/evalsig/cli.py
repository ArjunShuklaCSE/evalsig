"""Command-line interface: ``evalsig summary`` and ``evalsig compare``.

Exit codes: 0 normally, 1 when the ``--fail-on`` condition is met, 2 for invalid input.
"""

from __future__ import annotations

import sys
from enum import Enum
from pathlib import Path
from typing import Annotated, NoReturn, cast

import typer
from rich.console import Console
from rich.text import Text

from . import __version__, report
from .core import BETTER, WORSE, EvalsigError, Kind
from .data import compare as compare_rows
from .data import load_rows
from .data import summarize as summarize_rows

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Is the new version of your AI system really better, or is the difference noise?",
)
# Off a terminal (CI logs, pipes) rich assumes 80 columns, which wraps the table.
out = Console(highlight=False, soft_wrap=True, width=None if sys.stdout.isatty() else 120)
err = Console(stderr=True, highlight=False, soft_wrap=True)


class Format(str, Enum):
    table = "table"
    json = "json"
    markdown = "markdown"


class FailOn(str, Enum):
    worse = "worse"
    not_better = "not-better"


MetricOption = Annotated[
    list[str] | None,
    typer.Option(
        "--metric",
        "-m",
        help="Metric column to analyse. Repeat for several. Default: every numeric column.",
    ),
]
MetricTypeOption = Annotated[
    list[str] | None,
    typer.Option(
        "--metric-type", help="Override type detection, e.g. score=continuous. Repeatable."
    ),
]
AlphaOption = Annotated[
    float, typer.Option(help="Significance level. Intervals are (1 - alpha) confidence.")
]
ResamplesOption = Annotated[int, typer.Option(help="Bootstrap and permutation resamples.")]
SeedOption = Annotated[int, typer.Option(help="Random seed. Same input + seed = same output.")]
FormatOption = Annotated[Format, typer.Option("--format", "-f", help="Output format.")]


def _version(value: bool) -> None:
    if value:
        out.print(f"evalsig {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: Annotated[
        bool, typer.Option("--version", callback=_version, is_eager=True, help="Show version.")
    ] = False,
) -> None:
    """Is the new version of your AI system really better, or is the difference noise?"""


@app.command()
def compare(
    baseline: Annotated[Path, typer.Argument(help="Results of the current version (.jsonl/.csv).")],
    candidate: Annotated[Path, typer.Argument(help="Results of the new version, same examples.")],
    metric: MetricOption = None,
    metric_type: MetricTypeOption = None,
    lower_is_better: Annotated[
        list[str] | None,
        typer.Option("--lower-is-better", help="Metric where lower is better (latency, cost)."),
    ] = None,
    alpha: AlphaOption = 0.05,
    resamples: ResamplesOption = 10_000,
    seed: SeedOption = 0,
    output_format: FormatOption = Format.table,
    fail_on: Annotated[
        FailOn | None,
        typer.Option(help="Exit with code 1 if any metric is worse, or is not better. For CI."),
    ] = None,
    allow_partial: Annotated[
        bool, typer.Option(help="Compare only ids present in both runs, with a warning.")
    ] = False,
) -> None:
    """Compare a candidate run against a baseline run on the same test examples."""
    try:
        result = compare_rows(
            load_rows(baseline),
            load_rows(candidate),
            metric,
            metric_types=_parse_types(metric_type),
            lower_is_better=lower_is_better or (),
            allow_partial=allow_partial,
            alpha=alpha,
            resamples=resamples,
            seed=seed,
        )
    except EvalsigError as error:
        _fail(error)

    if output_format is Format.json:
        sys.stdout.write(result.to_json() + "\n")
    elif output_format is Format.markdown:
        sys.stdout.write(report.comparison_markdown(result))
    else:
        for warning in result.warnings:
            err.print(Text.assemble(("warning: ", "yellow"), warning))
        out.print(report.comparison_table(result))

    verdicts = [m.verdict for m in result.metrics]
    if (fail_on is FailOn.worse and WORSE in verdicts) or (
        fail_on is FailOn.not_better and any(v != BETTER for v in verdicts)
    ):
        raise typer.Exit(1)


@app.command()
def summary(
    run: Annotated[Path, typer.Argument(help="Results file (.jsonl/.csv).")],
    metric: MetricOption = None,
    metric_type: MetricTypeOption = None,
    alpha: AlphaOption = 0.05,
    resamples: ResamplesOption = 10_000,
    seed: SeedOption = 0,
    output_format: FormatOption = Format.table,
) -> None:
    """Mean of each metric in one run, with a confidence interval."""
    try:
        result = summarize_rows(
            load_rows(run),
            metric,
            metric_types=_parse_types(metric_type),
            alpha=alpha,
            resamples=resamples,
            seed=seed,
        )
    except EvalsigError as error:
        _fail(error)

    if output_format is Format.json:
        sys.stdout.write(result.to_json() + "\n")
    elif output_format is Format.markdown:
        sys.stdout.write(report.summary_markdown(result))
    else:
        out.print(report.summary_table(result))


def _parse_types(items: list[str] | None) -> dict[str, Kind]:
    types: dict[str, Kind] = {}
    for item in items or []:
        name, _, kind = item.rpartition("=")
        if not name or kind not in ("binary", "continuous"):
            raise typer.BadParameter(
                f"'{item}': expected NAME=binary or NAME=continuous.", param_hint="--metric-type"
            )
        types[name] = cast(Kind, kind)
    return types


def _fail(error: EvalsigError) -> NoReturn:
    err.print(Text.assemble(("error: ", "red"), str(error)))
    raise typer.Exit(2)
