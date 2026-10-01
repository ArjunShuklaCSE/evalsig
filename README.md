# evalsig

**Tells you whether a change to your AI system made it really better, or whether the difference in your eval score could just be luck.**

[![ci](https://github.com/ArjunShuklaCSE/evalsig/actions/workflows/ci.yml/badge.svg)](https://github.com/ArjunShuklaCSE/evalsig/actions/workflows/ci.yml)
![python](https://img.shields.io/badge/python-3.10%E2%80%933.13-blue)
![license](https://img.shields.io/badge/license-MIT-green)

You change a prompt, swap a model or tune retrieval, re-run your eval, and the score goes from 0.81 to 0.85. Is that a real improvement, or would a different handful of test questions have flipped it? evalsig reads the per-example results of both runs and answers with a difference, a confidence interval and a plain-English verdict: **better**, **worse**, or **no detectable difference**. When the answer is "no detectable difference", it also tells you the smallest change your test set could have detected.

It works with the results of any eval tool (promptfoo, Inspect, DeepEval or your own scripts). It does not run evals or call a model. It is a small statistics layer that depends only on numpy and scipy, plus typer and rich for the command line.

---

## Contents

- [Quickstart](#quickstart)
- [Example output](#example-output)
- [How to read the result](#how-to-read-the-result)
- [A real example: a gain that did not replicate](#a-real-example-a-gain-that-did-not-replicate)
- [Input format](#input-format)
- [Command line](#command-line)
- [Python API](#python-api)
- [GitHub Action](#github-action)
- [How it works](#how-it-works)
- [Limitations](#limitations)
- [Development](#development)

---

## Quickstart

```bash
pip install git+https://github.com/ArjunShuklaCSE/evalsig
evalsig compare baseline.jsonl candidate.jsonl
```

Each file has one row per test example, with an `id` and one or more scores:

```json
{"id": "q001", "correct": 0, "recall@10": 1.0, "latency_ms": 666}
{"id": "q002", "correct": 1, "recall@10": 0.0, "latency_ms": 907}
```

To try it without your own data, clone the repository and run the example (Python 3.10+):

```bash
git clone https://github.com/ArjunShuklaCSE/evalsig && cd evalsig
pip install .
evalsig compare examples/baseline.jsonl examples/candidate.jsonl --lower-is-better latency_ms
```

The package is not on PyPI yet. The first tagged release will publish it as `evalsig-stats` (the name `evalsig` is taken there). The import name and the command are `evalsig` either way.

## Example output

The example data is a synthetic RAG system on 151 questions, before and after adding a reranker ([`examples/make_data.py`](examples/make_data.py)):

```
$ evalsig compare examples/baseline.jsonl examples/candidate.jsonl --lower-is-better latency_ms
metric      baseline  candidate    diff            95% CI  p (Holm)  verdict
correct        0.556      0.636  +0.079  [+0.026, +0.132]     0.019  better
recall@10      0.636      0.689  +0.053  [-0.030, +0.136]     0.261  no detectable difference
latency_ms       864      1,007    +142      [+131, +154]    <0.001  worse

correct: better (n=151, p = 0.019).
  fixed 15, broke 3 (McNemar exact p = 0.008, uncorrected).
recall@10: no detectable difference (n=151). Smallest change this test set can reliably detect: about ±0.120.
latency_ms: worse (n=151, p < 0.001; lower is better for this metric).

95% CI: paired bootstrap, 10,000 resamples, seed 0. p: paired sign-flip permutation test, Holm-adjusted across 3 metrics. Verdicts use alpha 0.05.
```

`--format markdown` prints the same result as a markdown table (this is what the GitHub Action posts), and `--format json` gives every number for scripts.

## How to read the result

- **baseline / candidate**: the average score of each run over the examples both runs share.
- **diff**: candidate minus baseline. For `correct`, the candidate answered 7.9 percentage points more questions correctly.
- **95% CI**: the range of true differences that fit the data. If you ran both systems on many more questions like these, the difference would very likely land in this range. For `recall@10` the range runs from −0.030 to +0.136. It includes zero, so the data cannot rule out "no change".
- **p**: how surprising a difference this large would be if the two runs were truly equal. Small means surprising. With several metrics this p-value is corrected (Holm) so that checking more metrics does not raise the odds of a lucky "win".
- **verdict**:
  - **better** or **worse**: the difference is larger than luck explains (p below 0.05), in that direction.
  - **no detectable difference**: this test set cannot tell. It does *not* mean the systems are the same. That is why evalsig never says "same" or "no effect".
- **Smallest change this test set can reliably detect**: if the true difference were smaller than this, you would usually miss it. For `recall@10`, ±0.120 means 151 questions cannot settle whether retrieval improved by 0.05. You need more questions, or you have to accept that you don't know.
- **fixed / broke** (pass/fail metrics): how many examples went from fail to pass and from pass to fail. A net gain of +12 can hide a lot of churn, and it is worth reading the examples that broke.
- **Notes**: evalsig adds a line when something needs attention: identical runs, fewer than 20 examples, a result that is significant only before the multiple-metric correction, or a borderline case where the interval and the test disagree.

## A real example: a gain that did not replicate

[fomc-rag](https://github.com/ArjunShuklaCSE/fomc-rag) is a retrieval system over Federal Reserve documents, tuned on a dev split and checked once on a held-out test split. One tuning step replaced BM25 retrieval (E8) with a weighted hybrid of BM25 and dense retrieval (E10). The per-question results are in [`examples/fomc-rag/`](examples/fomc-rag).

On the dev questions the hybrid looked like a clear win:

```
$ evalsig compare examples/fomc-rag/dev_E8.jsonl examples/fomc-rag/dev_E10.jsonl -m recall@10
metric     baseline  candidate    diff            95% CI      p  verdict
recall@10     0.333      0.439  +0.106  [+0.038, +0.182]  0.008  better
```

On the test questions it vanished:

```
$ evalsig compare examples/fomc-rag/test_E8.jsonl examples/fomc-rag/test_E10.jsonl -m recall@10
metric     baseline  candidate    diff            95% CI      p  verdict
recall@10     0.385      0.369  -0.015  [-0.092, +0.062]  1.000  no detectable difference

recall@10: no detectable difference (n=65). Smallest change this test set can reliably detect: about ±0.114.
```

Two lessons, both visible in the output:

1. **A significant result on the data you tuned on is not a result.** The hybrid weight was chosen from four candidates using these same 66 dev questions, so part of the +0.106 was the search fitting noise. evalsig cannot see that selection step (see [Limitations](#limitations)), which is why a held-out split matters.
2. **The test split is too small to detect the gain claimed on dev.** With 65 questions the smallest reliably detectable change is ±0.114, about the size of the whole dev effect. "No detectable difference" here means "this test set can't tell", not "fusion does nothing".

These numbers match the paired bootstrap intervals fomc-rag reports, which its own code computed separately.

## Input format

- **Files:** `.jsonl` (one JSON object per line) or `.csv` with a header row.
- **`id` (required):** identifies the example. The two runs are joined on `id`, not on row order. Ids must be unique within a file.
- **Metrics:** every other column whose values are all numbers is a metric. Text columns such as the model output are ignored, and `--metric` picks specific columns.
  - **Binary** metrics have only 0/1 or `true`/`false` values. They get Wilson intervals and McNemar's test.
  - **Continuous** metrics have any other numbers (scores, recall, latency).
  - Detection is automatic. `--metric-type score=continuous` overrides it.
- **`group` (optional):** marks rows that belong to the same question, for example 5 samples per prompt. Rows in a group are correlated, so evalsig resamples whole groups (a cluster bootstrap) instead of single rows.
- **Mismatched ids** stop the comparison with an error that lists the missing ids. `--allow-partial` compares only the shared ids and prints a warning.
- **Missing values:** an empty cell, `null` or NaN in a metric stops with an error naming the affected ids. evalsig does not silently drop examples, because dropping the failures would flatter a run.

## Command line

```
evalsig summary RUN [options]
evalsig compare BASELINE CANDIDATE [options]
```

| Option | Default | Meaning |
|---|---|---|
| `-m`, `--metric NAME` | all numeric columns | Metric to analyse. Repeat for several. |
| `--metric-type NAME=TYPE` | auto | Force `binary` or `continuous`. |
| `--lower-is-better NAME` | none | Metric where a decrease is an improvement (latency, cost, error rate). `compare` only. |
| `--alpha` | `0.05` | Significance level. Intervals are (1 − alpha) confidence. |
| `--resamples` | `10000` | Bootstrap and permutation resamples. |
| `--seed` | `0` | Random seed. Same input and seed give exactly the same output. |
| `-f`, `--format` | `table` | `table`, `json` or `markdown`. |
| `--fail-on` | none | `worse`: exit 1 if any metric is worse. `not-better`: exit 1 unless every metric is better. `compare` only. |
| `--allow-partial` | off | Compare only the ids present in both runs. `compare` only. |

Exit codes: **0** success, **1** the `--fail-on` condition was met, **2** invalid input (the message says what to fix).

`evalsig summary` reports each metric's mean for one run with a 95% interval: Wilson for pass rates, a bootstrap for everything else.

## Python API

```python
from evalsig import compare, load_rows, summarize

result = compare(
    load_rows("baseline.jsonl"),  # or any list of dicts
    load_rows("candidate.jsonl"),
    metrics=["correct", "latency_ms"],
    lower_is_better=["latency_ms"],
)
for m in result.metrics:
    print(m.metric, m.diff, (m.ci_low, m.ci_high), m.p_adjusted, m.verdict, m.mde)

result.to_dict()  # plain dict
result.to_json()  # JSON string
```

`compare` returns a frozen `Comparison` dataclass holding one `MetricComparison` per metric. Invalid input raises `evalsig.EvalsigError`, whose message is written to be shown to a user. If your scores are already aligned arrays, `compare_arrays` and `summarize_arrays` skip the row handling.

## GitHub Action

Run your eval in CI and let evalsig comment on the pull request:

```yaml
name: eval
on: pull_request

permissions:
  contents: read
  pull-requests: write   # to post the comment

jobs:
  eval:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v5
      - run: python run_my_evals.py --out results/candidate.jsonl   # your eval
      - uses: ArjunShuklaCSE/evalsig@main
        with:
          baseline: results/baseline.jsonl     # committed results of the main branch
          candidate: results/candidate.jsonl
          metrics: correct recall@10 latency_ms
          lower-is-better: latency_ms
          fail-on: worse                        # optional: fail the check on a regression
```

- **One comment per pull request.** The action posts one comment with the markdown table. Later runs update that comment instead of adding new ones; it finds its own comment by a hidden `<!-- evalsig -->` marker.
- **Job summary.** The report is also written to the job summary.
- **Inputs:** `baseline`, `candidate` (required), `metrics`, `lower-is-better`, `fail-on`, `alpha`, `allow-partial`, `comment` (default `true`) and `github-token`.
- **Outputs:** `failed` (`true` when the fail-on condition was met) and `report` (path to the markdown file).
- **Runners:** Linux and macOS.
- **Fork pull requests** get a read-only token. Posting the comment then fails without failing the job, and the check itself still runs.

## How it works

Each method, why it was chosen and the references are in **[docs/statistics.md](docs/statistics.md)**. In short:

| Question | Method |
|---|---|
| Pass rate of one run | Wilson score interval |
| Mean of one run | Bootstrap percentile interval |
| Difference between runs | Paired bootstrap over example ids (10,000 resamples) |
| Is the difference real? | Paired sign-flip permutation test (exact when small) |
| Pass/fail churn | Fixed and broke counts with McNemar's exact test |
| Several metrics at once | Holm–Bonferroni correction |
| Is my test set big enough? | Minimum detectable effect, (z₀.₉₇₅ + z₀.₈) × SE ≈ 2.8 × SE |
| Several rows per question | Cluster bootstrap and cluster sign-flips over `group` |

Everything is paired: both runs are scored on the same examples, so evalsig analyses the per-example difference. That removes the variation in question difficulty, which is usually the largest source of noise. Comparing two separate intervals ignores this and misses real effects.

The statistics are checked by simulation as well as by unit tests. When there is truly no difference, the 95% interval contains zero 95.8% of the time (continuous metrics) and the false-positive rate stays under 5%. A true effect equal to the reported minimum detectable effect is detected 78.8% of the time, against the 80% the formula promises. The tables are in [docs/statistics.md](docs/statistics.md#how-the-methods-were-validated).

## Limitations

- **It cannot see how you chose the candidate.** If you tried twenty prompts and compare the best one, its p-value is too optimistic. Confirm on data you did not tune on, as in the [example above](#a-real-example-a-gain-that-did-not-replicate).
- **The test set stands in for the inputs you care about.** The intervals describe uncertainty over examples *like these*. A test set of easy questions says little about hard ones.
- **Small samples:** below about 20 examples (or groups) the bootstrap intervals are too narrow, and evalsig says so. With 25 groups the 95% interval covered about 92% of the time in simulation. The permutation test, which decides the verdict, stays exact.
- **Judge noise:** if an LLM judge produced the scores, its randomness and bias are inside the data. evalsig measures sampling uncertainty, not judge error. Re-running the judge and averaging, or using a `group` per question, helps.
- **No equivalence test:** "no detectable difference" is never a claim that the systems are equal. Proving equivalence needs a test against a margin you set in advance (TOST), which v0.1 does not include.
- **Higher is better by default.** Pass `--lower-is-better` for latency, cost and error rates, or the verdicts are reversed.

## Development

```bash
uv sync
uv run pytest -q          # unit, property-based and calibration tests
uv run ruff check . && uv run ruff format --check .
uv run mypy               # strict, on src/
```

[`examples/reproduce.sh`](examples/reproduce.sh) regenerates the example data and prints every output shown in this README. A test checks that the README still matches the tool's real output.

**Releases.** Pushing a `v*` tag runs the tests, builds the package, publishes it to PyPI through trusted publishing (no token stored in the repository) and creates a GitHub release with notes from [CHANGELOG.md](CHANGELOG.md).

## License

MIT. The fomc-rag example data comes from [fomc-rag](https://github.com/ArjunShuklaCSE/fomc-rag) (MIT).
