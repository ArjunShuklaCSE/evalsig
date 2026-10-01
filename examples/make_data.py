"""Write the synthetic example used in the README: a RAG system before and after a change.

151 questions, three metrics per question:
  correct     1 if the final answer was judged correct, else 0
  recall@10   share of the gold passages found in the top 10 (0, 0.5 or 1)
  latency_ms  end-to-end time for the question

The candidate adds a reranker: answers improve, retrieval moves a little (not enough
to be sure), and every question gets slower.

Run from the repository root:  python examples/make_data.py
"""

import json
from pathlib import Path

import numpy as np

N = 151
HERE = Path(__file__).parent


def main() -> None:
    rng = np.random.default_rng(7)
    difficulty = rng.beta(2, 1.2, size=N)  # per-question chance of success
    two_passages = rng.random(N) < 0.3  # these questions need two gold passages

    def recall(boost: float) -> np.ndarray:
        found = rng.random((N, 2)) < np.clip(difficulty + boost, 0, 1)[:, None]
        return np.where(two_passages, found.mean(axis=1), found[:, 0])

    base_correct = rng.random(N) < difficulty * 0.85
    cand_correct = base_correct | (rng.random(N) < 0.25)  # fixes some failures...
    cand_correct &= rng.random(N) > 0.03  # ...and breaks a few successes
    base_latency = rng.lognormal(np.log(820), 0.25, N)
    cand_latency = base_latency + rng.normal(140, 60, N)  # the reranker costs time

    runs = {
        "baseline": (base_correct, recall(0.0), base_latency),
        "candidate": (cand_correct, recall(0.04), cand_latency),
    }
    for name, (correct, rec, latency) in runs.items():
        with open(HERE / f"{name}.jsonl", "w", encoding="utf-8", newline="\n") as file:
            for i in range(N):
                row = {
                    "id": f"q{i + 1:03d}",
                    "correct": int(correct[i]),
                    "recall@10": float(rec[i]),
                    "latency_ms": round(float(latency[i])),
                }
                file.write(json.dumps(row) + "\n")
    print(f"wrote {HERE / 'baseline.jsonl'} and {HERE / 'candidate.jsonl'} ({N} rows each)")


if __name__ == "__main__":
    main()
