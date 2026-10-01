# /// script
# requires-python = ">=3.10"
# dependencies = ["matplotlib>=3.8", "numpy", "scipy", "typer", "rich"]
# ///
"""Render the README figures from real simulations and real example data.

Run from the repository root:  uv run docs/images/src/figures.py
Writes docs/images/{calibration,power,fomc-rag}.png. Seeds are fixed, so the figures
are reproducible.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))
from evalsig import compare, compare_arrays, load_rows, summarize_arrays  # noqa: E402

OUT = ROOT / "docs" / "images"
RESAMPLES = 1000

# Colors: the brand surface, text tokens, and a palette checked with a CVD validator.
SURFACE = "#0d0f17"
GRID = "#262a3b"
TEXT = "#f0eef8"
TEXT_2 = "#a6abc1"
MUTED = "#6b7085"
VIOLET = "#8b5cf6"
TEAL = "#0d9488"

plt.rcParams.update(
    {
        "font.family": ["Segoe UI", "DejaVu Sans"],
        "font.size": 13,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": GRID,
        "axes.labelcolor": TEXT_2,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 1,
        "xtick.color": TEXT_2,
        "ytick.color": TEXT_2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "legend.frameon": False,
        "legend.labelcolor": TEXT_2,
        "savefig.facecolor": SURFACE,
    }
)


def title(fig: plt.Figure, text: str, subtitle: str) -> None:
    fig.text(0.035, 0.94, text, color=TEXT, fontsize=19, fontweight="semibold", va="top")
    fig.text(0.035, 0.865, subtitle, color=TEXT_2, fontsize=13, va="top")


def save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=200)
    plt.close(fig)
    print("wrote", OUT / name)


def scores(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Skewed per-question scores in [0, 1] with shared difficulty (as in the tests)."""
    shared = rng.beta(2, 5, size=n)
    return (
        np.clip(shared + rng.normal(0, 0.1, n), 0, 1),
        np.clip(shared + rng.normal(0, 0.1, n), 0, 1),
    )


def passes(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    p = rng.beta(2, 2, size=n)
    return (rng.random(n) < p).astype(float), (rng.random(n) < p).astype(float)


def calibration() -> None:
    """Coverage of the 95% interval when there is no true difference, by test-set size."""
    sizes = [20, 50, 100, 200, 400]
    sims = 400
    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.14)
    for make, color, label in ((scores, VIOLET, "continuous scores"), (passes, TEAL, "pass/fail")):
        rng = np.random.default_rng(20)
        coverage = []
        for n in sizes:
            hits = 0
            for _ in range(sims):
                base, cand = make(rng, n)
                m = compare_arrays({"m": base}, {"m": cand}, resamples=RESAMPLES).metrics[0]
                hits += m.ci_low <= 0 <= m.ci_high
            coverage.append(hits / sims)
        ax.plot(
            sizes,
            coverage,
            color=color,
            lw=2.5,
            marker="o",
            ms=8,
            mec=SURFACE,
            mew=2,
            label=label,
            zorder=3,
        )
        print(make.__name__, dict(zip(sizes, coverage, strict=True)))
    band = 1.96 * np.sqrt(0.95 * 0.05 / sims)
    ax.axhspan(0.95 - band, 0.95 + band, color=TEXT_2, alpha=0.08, lw=0)
    ax.axhline(0.95, color=TEXT_2, lw=1.2, ls=(0, (4, 4)))
    ax.text(
        sizes[-1],
        0.95 + band + 0.004,
        "target 95%  (shaded: simulation noise)",
        color=TEXT_2,
        ha="right",
        va="bottom",
        fontsize=11,
    )
    ax.set_xscale("log")
    ax.set_xticks(sizes, [str(s) for s in sizes])
    ax.minorticks_off()
    ax.set_ylim(0.88, 1.0)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("examples in the test set")
    ax.legend(loc="lower right")
    title(
        fig,
        "The 95% interval is right about 95% of the time",
        f"Share of {sims} simulated test sets per point where the interval"
        " contains the true difference (zero)",
    )
    save(fig, "calibration.png")


def power() -> None:
    """Detection rate by true effect: paired test vs. checking whether two CIs overlap."""
    n, sims = 100, 250
    effects = np.array([0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.06])
    paired, overlap, mdes = [], [], []
    rng = np.random.default_rng(21)
    for effect in effects:
        hit_paired = hit_overlap = 0
        for _ in range(sims):
            base, cand = scores(rng, n)
            cand = cand + effect
            m = compare_arrays({"m": base}, {"m": cand}, resamples=RESAMPLES).metrics[0]
            hit_paired += m.verdict == "better"
            mdes.append(m.mde)
            b, c = summarize_arrays({"b": base, "c": cand}, resamples=RESAMPLES).metrics
            hit_overlap += c.ci_low > b.ci_high
        paired.append(hit_paired / sims)
        overlap.append(hit_overlap / sims)
    mde = float(np.median(mdes))
    print("paired", paired, "overlap", overlap, "median mde", mde)

    fig, ax = plt.subplots(figsize=(10, 5.2))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.74, bottom=0.14)
    ax.axvline(mde, color=TEXT_2, lw=1.2, ls=(0, (4, 4)))
    ax.text(
        mde - 0.0012,
        0.97,
        f"reported minimum\ndetectable effect ≈ {mde:.3f}",
        color=TEXT_2,
        fontsize=11,
        ha="right",
        va="top",
    )
    ax.plot(
        effects,
        paired,
        color=VIOLET,
        lw=2.5,
        marker="o",
        ms=8,
        mec=SURFACE,
        mew=2,
        label="evalsig paired test",
        zorder=3,
    )
    ax.plot(
        effects,
        overlap,
        color=TEAL,
        lw=2.5,
        marker="o",
        ms=8,
        mec=SURFACE,
        mew=2,
        label="check: do the two 95% intervals overlap?",
        zorder=3,
    )
    ax.set_ylim(-0.02, 1.02)
    ax.yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")
    ax.set_xlabel("true improvement in mean score")
    ax.legend(
        loc="center right",
        bbox_to_anchor=(1.0, 0.42),
        frameon=True,
        facecolor=SURFACE,
        edgecolor=GRID,
        framealpha=1,
        borderpad=0.8,
    )
    title(
        fig,
        "Pairing finds real improvements that interval overlap misses",
        f"Share of {sims} simulated runs (n = {n}) called better,"
        " by the size of the true improvement",
    )
    save(fig, "power.png")


def fomc_rag() -> None:
    """The fomc-rag hybrid-retrieval change on dev and test, as interval rows."""
    data = ROOT / "examples" / "fomc-rag"
    rows = []
    for split, color in (("dev", VIOLET), ("test", TEAL)):
        result = compare(
            load_rows(data / f"{split}_E8.jsonl"),
            load_rows(data / f"{split}_E10.jsonl"),
            ["recall@10"],
        )
        rows.append((split, color, result.metrics[0]))

    fig, ax = plt.subplots(figsize=(11, 4.2))
    fig.subplots_adjust(left=0.23, right=0.74, top=0.68, bottom=0.17)
    ax.axvline(0, color=TEXT_2, lw=1.2)
    for y, (split, color, m) in zip((1, 0), rows, strict=True):
        ax.plot([m.ci_low, m.ci_high], [y, y], color=color, lw=4, solid_capstyle="round")
        ax.plot(m.diff, y, "o", color=color, ms=11, mec=SURFACE, mew=2.5, zorder=3)
        label = f"{split} split, {m.n} questions"
        ax.text(
            -0.03,
            y,
            label,
            color=TEXT,
            ha="right",
            va="center",
            fontsize=13,
            transform=ax.get_yaxis_transform(),
        )
        verdict = f"{m.diff:+.3f}  {m.verdict}"
        ax.text(
            1.03,
            y,
            verdict,
            color=TEXT_2,
            ha="left",
            va="center",
            fontsize=12,
            transform=ax.get_yaxis_transform(),
        )
    ax.set_xlim(-0.12, 0.2)
    ax.set_ylim(-0.6, 1.6)
    ax.set_yticks([])
    ax.grid(axis="y", visible=False)
    ax.xaxis.set_major_formatter(lambda v, _: f"{v:+.2f}" if v else "0")
    ax.set_xlabel("change in Recall@10 (candidate − baseline), with 95% interval")
    title(
        fig,
        "A dev-set win that did not replicate",
        "fomc-rag, BM25 → hybrid retrieval. The tuned gain on dev is inside the noise on test.",
    )
    save(fig, "fomc-rag.png")


if __name__ == "__main__":
    for name in sys.argv[1:] or ["fomc_rag", "power", "calibration"]:
        globals()[name]()
