"""The CLI end to end: formats, exit codes, and clean error messages (never a traceback)."""

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalsig import __version__
from evalsig.cli import app
from evalsig.report import MARKER

EXAMPLES = Path(__file__).parent.parent / "examples"
BASE = str(EXAMPLES / "baseline.jsonl")
CAND = str(EXAMPLES / "candidate.jsonl")
runner = CliRunner()


def run(*args: str) -> tuple[int, str]:
    result = runner.invoke(app, list(args))
    assert "Traceback" not in result.output
    assert result.exception is None or isinstance(result.exception, SystemExit)
    return result.exit_code, result.output


def write(tmp_path: Path, name: str, rows: list[dict[str, object]]) -> str:
    path = tmp_path / name
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return str(path)


def test_compare_table() -> None:
    code, output = run("compare", BASE, CAND, "--lower-is-better", "latency_ms")
    assert code == 0
    assert "p (Holm)" in output
    assert "correct: better (n=151" in output
    assert "fixed 15, broke 3" in output
    assert "recall@10: no detectable difference (n=151). Smallest change" in output
    assert "latency_ms: worse" in output


def test_compare_json_is_deterministic_and_seeded() -> None:
    _, first = run("compare", BASE, CAND, "-f", "json")
    _, second = run("compare", BASE, CAND, "-f", "json")
    _, reseeded = run("compare", BASE, CAND, "-f", "json", "--seed", "1")
    assert first == second
    assert json.loads(first)["metrics"] != json.loads(reseeded)["metrics"]


def test_compare_markdown() -> None:
    code, output = run("compare", BASE, CAND, "-m", "correct", "--format", "markdown")
    assert code == 0
    assert output.startswith(MARKER)
    assert "| correct | 0.556 | 0.636 | +0.079 |" in output
    assert "**better**" in output
    assert "Holm" not in output  # one metric: no correction


@pytest.mark.parametrize(
    ("args", "code"),
    [
        (["--fail-on", "worse"], 0),  # without --lower-is-better, higher latency reads as better
        (["--fail-on", "worse", "--lower-is-better", "latency_ms"], 1),
        (["--fail-on", "not-better", "-m", "correct"], 0),
        (["--fail-on", "not-better"], 1),  # recall@10 is not detectably better
        (["--fail-on", "worse", "-m", "recall@10"], 0),
    ],
)
def test_fail_on_exit_codes(args: list[str], code: int) -> None:
    assert run("compare", BASE, CAND, *args)[0] == code


def test_summary_formats() -> None:
    code, output = run("summary", BASE)
    assert code == 0
    assert "wilson" in output and "bootstrap" in output
    _, as_json = run("summary", BASE, "--format", "json")
    assert [m["metric"] for m in json.loads(as_json)["metrics"]] == [
        "correct",
        "recall@10",
        "latency_ms",
    ]
    assert run("summary", BASE, "-f", "markdown")[1].startswith(MARKER)


def test_version() -> None:
    assert run("--version") == (0, f"evalsig {__version__}\n")


def test_edge_case_runs(tmp_path: Path) -> None:
    one = write(tmp_path, "one.jsonl", [{"id": 1, "ok": 1}])
    code, output = run("compare", one, one)
    assert code == 0 and "identical" in output and "Only 1 examples" in output

    all_pass = write(tmp_path, "pass.jsonl", [{"id": i, "ok": 1} for i in range(40)])
    all_fail = write(tmp_path, "fail.jsonl", [{"id": i, "ok": 0} for i in range(40)])
    code, output = run("compare", all_fail, all_pass)
    assert code == 0 and "ok: better" in output and "fixed 40, broke 0" in output
    assert run("summary", all_pass)[0] == 0


@pytest.mark.parametrize(
    ("baseline_rows", "args", "message"),
    [
        (
            [{"id": 1, "ok": None}, {"id": 2, "ok": 1}],
            ["--allow-partial", "-m", "ok"],
            "missing, NaN or infinite for ids: 1",
        ),
        ([{"id": 9, "ok": 1}], [], "Missing from candidate (1): 9"),
        ([{"id": 1, "ok": 1}], ["--metric-type", "ok=fancy"], "expected NAME=binary"),
        ([{"id": 1, "ok": 1}], ["--alpha", "2"], "alpha must be between 0 and 1"),
        ([{"id": 1, "ok": 1}], ["-m", "nope"], "has no column 'nope'"),
    ],
)
def test_input_errors_exit_2_with_a_message(
    tmp_path: Path, baseline_rows: list[dict[str, object]], args: list[str], message: str
) -> None:
    base = write(tmp_path, "b.jsonl", baseline_rows)
    cand = write(tmp_path, "c.jsonl", [{"id": 1, "ok": 1}])
    code, output = run("compare", base, cand, *args)
    assert code == 2
    assert message in " ".join(output.split())


def test_missing_and_empty_files(tmp_path: Path) -> None:
    (tmp_path / "empty.jsonl").write_text("", encoding="utf-8")
    code, output = run("summary", str(tmp_path / "empty.jsonl"))
    assert code == 2 and "is empty" in output
    code, output = run("compare", str(tmp_path / "nope.csv"), BASE)
    assert code == 2 and "File not found" in output


def test_allow_partial(tmp_path: Path) -> None:
    base = write(tmp_path, "b.jsonl", [{"id": i, "ok": i % 2} for i in range(30)])
    cand = write(tmp_path, "c.jsonl", [{"id": i, "ok": 1} for i in range(25)])
    assert run("compare", base, cand)[0] == 2
    code, output = run("compare", base, cand, "--allow-partial")
    assert code == 0 and "only the 25 shared ids" in output
    _, markdown = run("compare", base, cand, "--allow-partial", "-f", "markdown")
    assert "> **Warning:**" in markdown
