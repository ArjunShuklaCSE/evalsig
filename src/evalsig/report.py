"""Turning results into plain-English sentences, a terminal table, and markdown."""

from __future__ import annotations

from rich.console import Group
from rich.table import Table
from rich.text import Text

from .core import NO_DIFFERENCE, Comparison, MetricComparison, Summary

# Hidden marker the GitHub Action uses to find and update its own PR comment.
MARKER = "<!-- evalsig -->"


def decimals_for(*values: float) -> int:
    """Decimal places that suit the scale of a metric (0.81 -> 3, 1234 -> 0)."""
    scale = max(abs(v) for v in values)
    if scale < 10:
        return 3
    if scale < 100:
        return 2
    return 1 if scale < 1000 else 0


def number(value: float, decimals: int, signed: bool = False) -> str:
    if round(value, decimals) == 0:
        value = 0.0  # avoid "-0.000"
    return f"{value:+,.{decimals}f}" if signed else f"{value:,.{decimals}f}"


def pvalue(p: float) -> str:
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def interval(low: float, high: float, decimals: int, signed: bool = False) -> str:
    return f"[{number(low, decimals, signed)}, {number(high, decimals, signed)}]"


def ci_label(alpha: float) -> str:
    return f"{(1 - alpha) * 100:g}% CI"


def size(n: int, n_groups: int | None) -> str:
    return f"n={n}" if n_groups is None else f"n={n} rows in {n_groups} groups"


def verdict_sentence(m: MetricComparison) -> str:
    d = decimals_for(m.baseline, m.candidate)
    direction = "" if m.higher_is_better else "; lower is better for this metric"
    if m.verdict == NO_DIFFERENCE:
        return (
            f"{m.verdict} ({size(m.n, m.n_groups)}{direction}). Smallest change this test set "
            f"can reliably detect: about ±{number(m.mde, d)}."
        )
    return f"{m.verdict} ({size(m.n, m.n_groups)}, p {_p_phrase(m.p_adjusted)}{direction})."


def detail_lines(m: MetricComparison) -> list[str]:
    lines = []
    if m.fixed is not None and m.broke is not None:
        line = f"fixed {m.fixed}, broke {m.broke}"
        if m.mcnemar_p is not None:
            line += f" (McNemar exact p {_p_phrase(m.mcnemar_p)}, uncorrected)"
        lines.append(line + ".")
    lines.extend(m.notes)
    return lines


def method_line(result: Comparison) -> str:
    line = (
        f"{ci_label(result.alpha)}: paired bootstrap, {result.resamples:,} resamples, "
        f"seed {result.seed}. p: paired sign-flip permutation test"
    )
    if result.correction == "holm":
        line += f", Holm-adjusted across {len(result.metrics)} metrics"
    return line + f". Verdicts use alpha {result.alpha:g}."


def comparison_table(result: Comparison) -> Group:
    holm = result.correction == "holm"
    table = Table(show_edge=False, pad_edge=False, box=None, header_style="bold")
    for name in ("metric", "baseline", "candidate", "diff", ci_label(result.alpha)):
        table.add_column(name, justify="left" if name == "metric" else "right")
    table.add_column("p (Holm)" if holm else "p", justify="right")
    table.add_column("verdict")
    for m in result.metrics:
        d = decimals_for(m.baseline, m.candidate)
        style = {"better": "green", "worse": "red"}.get(m.verdict, "")
        table.add_row(
            Text(m.metric),
            number(m.baseline, d),
            number(m.candidate, d),
            number(m.diff, d, signed=True),
            interval(m.ci_low, m.ci_high, d, signed=True),
            pvalue(m.p_adjusted),
            Text(m.verdict, style=style),
        )

    lines: list[Text] = [Text("")]
    for m in result.metrics:
        lines.append(Text.assemble((m.metric, "bold"), ": ", verdict_sentence(m)))
        lines.extend(Text(f"  {line}", style="dim") for line in detail_lines(m))
    lines.append(Text(""))
    lines.append(Text(method_line(result), style="dim"))
    return Group(table, *lines)


def comparison_markdown(result: Comparison) -> str:
    holm = result.correction == "holm"
    out = [
        MARKER,
        "### evalsig: candidate vs baseline",
        "",
        f"| Metric | Baseline | Candidate | Diff | {ci_label(result.alpha)} | "
        f"{'p (Holm)' if holm else 'p'} | Verdict |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for m in result.metrics:
        d = decimals_for(m.baseline, m.candidate)
        verdict = m.verdict if m.verdict == NO_DIFFERENCE else f"**{m.verdict}**"
        out.append(
            f"| {_md_escape(m.metric)} | {number(m.baseline, d)} | {number(m.candidate, d)} | "
            f"{number(m.diff, d, signed=True)} | {interval(m.ci_low, m.ci_high, d, signed=True)} | "
            f"{pvalue(m.p_adjusted)} | {verdict} |"
        )
    out.append("")
    for m in result.metrics:
        out.append(f"- **{_md_escape(m.metric)}**: {verdict_sentence(m)}")
        out.extend(f"  - {line}" for line in detail_lines(m))
    out.extend(f"\n> **Warning:** {w}" for w in result.warnings)
    out.extend(["", f"<sub>{method_line(result)}</sub>", ""])
    return "\n".join(out)


def summary_table(result: Summary) -> Group:
    table = Table(show_edge=False, pad_edge=False, box=None, header_style="bold")
    for name in ("metric", "n", "mean", ci_label(result.alpha), "method"):
        table.add_column(name, justify="left" if name in ("metric", "method") else "right")
    notes: list[Text] = []
    for m in result.metrics:
        d = decimals_for(m.mean, m.ci_low, m.ci_high)
        table.add_row(
            Text(m.metric),
            size(m.n, m.n_groups).removeprefix("n="),
            number(m.mean, d),
            interval(m.ci_low, m.ci_high, d),
            m.method,
        )
        notes.extend(Text(f"{m.metric}: {note}", style="dim") for note in m.notes)
    return Group(table, *notes)


def summary_markdown(result: Summary) -> str:
    out = [
        MARKER,
        "### evalsig: run summary",
        "",
        f"| Metric | n | Mean | {ci_label(result.alpha)} | Method |",
        "|---|---:|---:|---:|---|",
    ]
    for m in result.metrics:
        d = decimals_for(m.mean, m.ci_low, m.ci_high)
        out.append(
            f"| {_md_escape(m.metric)} | {size(m.n, m.n_groups).removeprefix('n=')} | "
            f"{number(m.mean, d)} | {interval(m.ci_low, m.ci_high, d)} | {m.method} |"
        )
    notes = [f"- **{_md_escape(m.metric)}**: {note}" for m in result.metrics for note in m.notes]
    if notes:
        out.extend(["", *notes])
    return "\n".join(out) + "\n"


def _p_phrase(p: float) -> str:
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def _md_escape(text: str) -> str:
    return text.replace("|", "\\|")
