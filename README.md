# production-rag

> Hybrid BM25 + dense retrieval over DuckDB, dbt and Dagster documentation with cross-encoder reranking and cited answers - where retrieval quality is measured, not asserted.

[![CI](https://github.com/Prithv122/production-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/Prithv122/production-rag/actions/workflows/ci.yml)

**Live demo:** _link, or "not deployed — runs locally"_
**Stack:** _..._

---

## 1. The problem

_What business or practical problem does this solve, and for whom? Two or three sentences. No "I built this to learn X" — frame it as work._

## 2. The data

| | |
|---|---|
| Source | _..._ |
| Size | _rows / files / GB_ |
| Licence | _..._ |
| Refresh | _one-off / scheduled_ |

_Note here if the data is synthetic or a small sample — and say so again next to any metric derived from it._

## 3. Architecture

```mermaid
flowchart LR
    A[Source] --> B[Processing]
    B --> C[Store]
    C --> D[Interface]
```

## 4. Key decisions & tradeoffs

| Decision | Chose | Over | Why |
|---|---|---|---|
| _..._ | _..._ | _..._ | _..._ |

_The section interviewers actually read. At least three real decisions, each with the alternative you rejected._

## 5. Results

| Metric | Value | Baseline | Notes |
|---|---|---|---|
| _..._ | _..._ | _..._ | _..._ |

_Every number here must be reproducible from code in this repo. State the dataset and conditions._

## 6. How to run

```bash
git clone https://github.com/Prithv122/production-rag.git
cd production-rag
uv sync
cp .env.example .env   # then fill in values
uv run pytest
```

_Then the actual entry point. Every prerequisite — services, env vars, dataset downloads — must be listed. Verify these steps work from a clean clone before shipping._

## 7. What I'd change at 100× scale

_Where this breaks first, and what you'd do about it. Be specific: which component, which bottleneck, which replacement._

---

## References

_Any reference implementation or paper consulted. Tier 3 projects must credit these._
