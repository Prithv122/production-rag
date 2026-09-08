# production-rag — G3

**Tier:** 3 ⭐ · **Category:** G — NLP & AI engineering · **Wave:** 3

Root rules in `../CLAUDE.md` apply. This file is project-specific only — keep it under 40 lines.

## What this is

A RAG system over the official DuckDB, dbt and Dagster docs where **retrieval quality is
measured rather than asserted**. BM25, dense and hybrid arms are evaluated against the
same ground truth across three chunking strategies; the generation model is a swappable
component behind an `LLMProvider` interface, chosen by evidence rather than by hype.

Corpus was picked because projects 22 (dbt/Dagster) and 23 (DuckDB) mean the author can
personally judge whether a retrieved passage is actually relevant. Three overlapping
technical vocabularies ("dependencies", "assets", "models") create real retrieval
ambiguity — which is what makes hybrid + reranking meaningful rather than decorative.

## Stack

Python 3.12 · scipy sparse (own BM25) · sentence-transformers (bge-small-en-v1.5, exact
numpy search — no ANN at this corpus size) · cross-encoder reranker
(ms-marco-MiniLM-L-6-v2) · OpenRouter's OpenAI-compatible endpoint over stdlib `urllib`
(no client library — Ollama's endpoint is a different shape anyway), Ollama fallback
(`qwen2.5:7b-instruct-q3_K_M`) · Pydantic answer contract · Gradio on HF Spaces.

## Acceptance criteria

- [ ] Real corpus, licence recorded (MIT / Apache-2.0 / Apache-2.0), pinned commit SHAs
- [ ] ≥2 chunking strategies measured against each other, not asserted
- [ ] Hybrid BM25 + vector retrieval with a reranking stage
- [ ] Query rewriting as a *measured arm*, reported per question category
- [ ] Citations back to source spans; refusal path when retrieval is weak
- [ ] Eval numbers in the README: recall@k, nDCG, faithfulness, citation correctness
- [ ] Deployed to HF Spaces and linkable
- [ ] Graceful degradation to Ollama when the API is unavailable
- [ ] Ship gate passes (`/ship`)

## Project-specific notes

- **Env:** `OPENROUTER_API_KEY` is the only required secret, and only for generation.
  Every retrieval number in the README reproduces with no key and no network.
- **DuckDB docs: use `docs/current/` only.** The repo also ships `0.10`–`1.3` and `lts`
  (2,073 further files of near-duplicate prose) which poison retrieval precision. There is
  a deliberate ablation quantifying this — do not "fix" it by ingesting everything.
- **Keep the fast suite fast.** Tests inject a fake embedder through the interface so CI
  never downloads torch weights; real-model tests are `@pytest.mark.slow`. Project 20
  shipped a 209s suite and had to disclose it in its own README — don't repeat that.
- **Never commit the generation cache's raw API keys**; cache keys are prompt hashes only.
- **Not an agent.** No LLM-in-a-loop, no LangGraph, no tool orchestration — that is
  project 43 (`43-agent-system`). Query rewriting is a single measured step, not a loop.
- **LLM-dependent steps must stay replayable.** Query rewrites and generations both go
  through the on-disk cache so `eval --replay` reproduces every published number with no
  key and no network. Anything that can't be replayed can't be in the README.
