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
(`qwen2.5:7b-instruct-q3_K_M`) · dataclass answer contract over the existing JSON-repair
path (**not** Pydantic — deviation reasoned in NOTES.md) · Gradio (written and verified
locally; HF Spaces hosting blocked, see above).

## Acceptance criteria

- [x] Real corpus, licence recorded (MIT / Apache-2.0 / Apache-2.0), pinned commit SHAs
- [x] ≥2 chunking strategies measured against each other, not asserted
- [x] Hybrid BM25 + vector retrieval with a reranking stage
- [x] Query rewriting as a *measured arm*, reported per question category
- [x] Citations back to source spans; refusal path when retrieval is weak
- [x] Eval numbers in the README: recall@k, nDCG, citation validity, grounding, refusal
      (faithfulness is **not** claimed — see README §5 on why no LLM judge was appointed)
- [ ] **Deployed to HF Spaces and linkable — BLOCKED, deferred.** App written, verified
      locally end-to-end, index published as a HF dataset. `hf repos create --type space
      --space-sdk gradio` returns **402**: HF restricts Gradio Spaces on free `cpu-basic`
      to PRO. Not buying PRO, and not substituting a static client-side rewrite, which
      would stop the demo running the measured code. Blocker documented in README §6.
- [x] Graceful degradation to Ollama when the API is unavailable — and, after session 3,
      *disabled inside `answer-eval`*, because in a comparison between models it silently
      substitutes one for another. See NOTES.md.
- [ ] Ship gate passes (`/ship`)

## Project-specific notes

- **Env:** `OPENROUTER_API_KEY` is the only required secret, and only for generation.
  Every retrieval number in the README reproduces with no key and no network.
- **The OpenRouter account cannot reach hosted models.** `:free` ids return 429
  `free-models-per-day` — the cap is **50 requests/day and account-wide**, so swapping to a
  different free model is not a workaround. Paid ids return 402 (no credits ever purchased).
  The published answer table therefore runs on three **local Ollama** arms, labelled as such
  and never presented as the originally planned hosted comparison. Do not quietly re-label
  them, and do not mix results from the two provider families in one table.
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
