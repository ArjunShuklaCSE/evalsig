"""Record the real CLI output for the README hero image.

Run from the repository root:  uv run python docs/images/src/terminal.py
Writes docs/images/src/terminal.svg (rich's own terminal export). terminal.html wraps it
so a headless browser can render docs/images/terminal.png.
"""

from pathlib import Path

from rich.console import Console
from rich.terminal_theme import TerminalTheme
from rich.text import Text

from evalsig import compare, load_rows, report

ROOT = Path(__file__).resolve().parents[3]
COMMAND = "evalsig compare baseline.jsonl candidate.jsonl --lower-is-better latency_ms"

THEME = TerminalTheme(
    (13, 15, 23),  # background: the brand surface
    (220, 218, 232),  # foreground
    [
        (13, 15, 23),
        (248, 113, 113),
        (45, 212, 191),
        (250, 204, 21),
        (139, 92, 246),
        (196, 181, 253),
        (94, 234, 212),
        (166, 171, 193),
    ],
    [
        (107, 112, 133),
        (252, 165, 165),
        (143, 221, 197),
        (253, 224, 71),
        (167, 139, 250),
        (221, 214, 254),
        (153, 246, 228),
        (240, 238, 248),
    ],
)


def main() -> None:
    result = compare(
        load_rows(ROOT / "examples" / "baseline.jsonl"),
        load_rows(ROOT / "examples" / "candidate.jsonl"),
        lower_is_better=["latency_ms"],
    )
    console = Console(record=True, width=118, highlight=False)
    console.print(Text.assemble(("$ ", "bright_black"), (COMMAND, "bold")))
    console.print()
    console.print(report.comparison_table(result))
    svg = console.export_svg(title="evalsig compare", theme=THEME)
    out = ROOT / "docs" / "images" / "src" / "terminal.svg"
    out.write_text(svg, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
