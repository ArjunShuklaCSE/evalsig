#!/usr/bin/env sh
# Regenerates the example data and prints every output shown in the README.
set -e
cd "$(dirname "$0")/.."
python examples/make_data.py
evalsig compare examples/baseline.jsonl examples/candidate.jsonl --lower-is-better latency_ms
evalsig summary examples/baseline.jsonl
evalsig compare examples/fomc-rag/dev_E8.jsonl examples/fomc-rag/dev_E10.jsonl -m recall@10
evalsig compare examples/fomc-rag/test_E8.jsonl examples/fomc-rag/test_E10.jsonl -m recall@10
