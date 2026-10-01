# Contributing

Thanks for helping. evalsig is small on purpose, and its value is that its numbers are right, so changes are judged first on correctness and second on simplicity.

## Setup

You need Python 3.10+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/ArjunShuklaCSE/evalsig && cd evalsig
uv sync
```

## Checks

CI runs all of these on Python 3.10 to 3.13. Run them before opening a pull request:

```bash
uv run pytest -q
uv run ruff check . && uv run ruff format --check .
uv run mypy
```

The test suite has four layers:

- **`tests/test_stats.py`** checks each statistic against hand-computed values or scipy.
- **`tests/test_calibration.py`** runs seeded simulations. It checks that intervals cover at their stated rate, that false positives stay near alpha, and that a true effect equal to the reported MDE is detected about 80% of the time.
- **`tests/test_properties.py`** uses Hypothesis to check invariants such as symmetry and shift invariance.
- **`tests/test_data.py`, `tests/test_cli.py` and `tests/test_readme.py`** cover input handling, the CLI, and that the README's example output is real.

## Changing the statistics

- **Explain the method.** Say which method you are adding or changing and why, with a reference, in the pull request and in [docs/statistics.md](docs/statistics.md).
- **Add a calibration check** to `tests/test_calibration.py` with a fixed seed. Choose its tolerance from the Monte Carlo error, not from what happens to pass.
- **Keep the output reproducible.** The same input and seed must give byte-identical output, so new randomness must come from the generator passed in.
- **If you're unsure a method is correct, say so in the pull request.** An honest "I'm not sure about X" is more useful than a confident guess.

## Style

- **Code:** clear code over clever code. Match the surrounding style. Public API functions have docstrings, and comments explain *why*.
- **Messages:** user-facing messages are full sentences that say what to fix.
- **Verdicts:** never say "same" or "no effect".

## Updating the README figures

The figures in `docs/images/` are generated from real simulations and real example data:

```bash
uv run docs/images/src/figures.py               # calibration.png, power.png, fomc-rag.png
uv run python docs/images/src/terminal.py       # terminal.svg, the recorded CLI output
```

`banner.html`, `social.html` and `terminal.html` are rendered to PNG with a headless browser at the sizes noted in each file.
