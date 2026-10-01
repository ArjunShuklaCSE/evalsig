"""The README's example outputs must match what the tool prints, and the Action must
look for the same marker the markdown report writes."""

import shlex
from pathlib import Path

import pytest
from typer.testing import CliRunner

from evalsig.cli import app
from evalsig.report import MARKER

ROOT = Path(__file__).parent.parent


def readme_examples() -> list[tuple[str, list[str]]]:
    """(command, expected output lines) for every README code block that starts with '$ evalsig'."""
    blocks = (ROOT / "README.md").read_text(encoding="utf-8").split("```")[1::2]
    examples = []
    for block in blocks:
        lines = block.strip("\n").splitlines()
        if lines and lines[0].startswith("$ evalsig "):
            examples.append((lines[0][2:], [line.rstrip() for line in lines[1:] if line.strip()]))
    return examples


@pytest.mark.parametrize(("command", "expected"), readme_examples())
def test_readme_output_is_real(
    command: str, expected: list[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    result = CliRunner().invoke(app, shlex.split(command)[1:])
    actual = [line.rstrip() for line in result.output.splitlines()]
    missing = [line for line in expected if line not in actual]
    assert not missing, f"README lines not in the output of '{command}':\n" + "\n".join(missing)


def test_readme_has_examples() -> None:
    assert len(readme_examples()) == 3


def test_action_uses_the_report_marker() -> None:
    assert MARKER in (ROOT / "action.yml").read_text(encoding="utf-8")
