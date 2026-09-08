---
title: production-rag
emoji: 🔎
colorFrom: indigo
colorTo: gray
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
license: mit
short_description: Measured hybrid retrieval over DuckDB, dbt and Dagster docs
---

# production-rag

Hybrid BM25 + dense retrieval over the official **DuckDB, dbt and Dagster**
documentation, with cited answers and a refusal path — where every arm was
**measured against a 184-question evaluation set** rather than asserted.

Source, evaluation and the findings that disappointed:
<https://github.com/Prithv122/production-rag>

## What this Space is running

- **Index:** [`Prithv122/production-rag-index`](https://huggingface.co/datasets/Prithv122/production-rag-index)
  — the `heading` chunking strategy, downloaded rather than rebuilt, so the demo
  searches exactly the index the published numbers came from.
- **Retrieval:** the same `production_rag.pipeline` the evaluation harness uses.
  The default arm, `hybrid_score_weighted`, is the one that won the grid — not
  the most impressive-sounding one.
- **Generation:** optional. Retrieval needs no secret at all. If the Space owner
  has not set an `OPENROUTER_API_KEY` secret, the app ranks and shows passages
  and says so, rather than failing.

## Configuration

| Secret / variable | Required | What it does |
|---|---|---|
| `OPENROUTER_API_KEY` | no | Enables answer generation. Without it the Space is retrieval-only. |
| `INDEX_REPO` | no | Override the dataset repo the index is pulled from. |

## Notes on the demo

The cross-encoder rerank checkbox states its real cost in its own label: on this
hardware it buys roughly +0.004 nDCG for about seven seconds of latency. It is
off by default for that reason.

Try the last example question. It is deliberately unanswerable, and a correct
system refuses it.
