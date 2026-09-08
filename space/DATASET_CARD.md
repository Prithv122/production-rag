---
license: apache-2.0
language:
  - en
tags:
  - retrieval
  - rag
  - documentation
pretty_name: production-rag heading index
---

# production-rag — prebuilt `heading` index

Retrieval artefacts for the [production-rag](https://github.com/Prithv122/production-rag)
Space. This repository holds **no original content**: it is a chunked, indexed
derivative of three public documentation repositories, pinned at exact commits.

It exists so the Space searches *the same index the published evaluation numbers
came from* instead of a rebuild, and so a free CPU Space does not spend ten
minutes re-encoding 24k chunks on every restart.

## Contents

```
heading/chunks.jsonl      chunk text, offsets, headings and source URLs
heading/bm25/             sparse weight matrix + vocabulary (scipy)
heading/dense/            float32 vectors, BAAI/bge-small-en-v1.5, L2-normalised
```

`heading` is one of three chunking strategies the project measures; it won on
nDCG@10. The chunker splits on markdown headings, keeps code fences whole, and
records the character span each chunk occupies in its source document.

## Sources and licences

| Tool | Repository | Commit | Licence |
|---|---|---|---|
| DuckDB | `duckdb/duckdb-web` (`docs/current` only) | `6f6cd1659f0e2ddd1965b1d3f1833e7fc512e7ac` | MIT |
| dbt | `dbt-labs/docs.getdbt.com` (`website/docs`) | `cd0e5b0e2b77302807f7dd126faac871e728edab` | Apache-2.0 |
| Dagster | `dagster-io/dagster` (`docs/docs`) | `76eed340c6b84517d91a86163c461c022ebef8d8` | Apache-2.0 |

Each chunk carries its `tool`, its source `doc_id` and a link back to the
canonical published page. DuckDB's superseded version trees (`docs/0.10` …
`docs/lts`) are deliberately **not** included — they are near-duplicates of
`current` and measurably wreck retrieval precision.

## Rebuilding this yourself

```bash
git clone https://github.com/Prithv122/production-rag
cd production-rag
uv run production-rag ingest                        # clones at the pinned SHAs
uv run --extra embed production-rag index --strategy heading
```

The result is byte-identical given the same commits and model.
