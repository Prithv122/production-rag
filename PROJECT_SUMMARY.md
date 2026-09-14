# Project 24 — `production-rag`: complete project summary

> **Scope of this document.** A single narrative record of what was built, what was measured,
> what broke, and what is still open. Every number here traces to a committed artefact in this
> repository; where something is unverified or contested, it says so rather than rounding up.
> Where this document corrects an earlier claim of mine, the correction is marked **⚠︎**.

---

## 1. Identity

| | |
|---|---|
| Project | 24 — `production-rag` |
| Category / tier | G3 — Production RAG · **Tier 3 ⭐** |
| Corpus | DuckDB (MIT), dbt (Apache-2.0), Dagster (Apache-2.0) documentation |
| Winning chunking strategy | `heading` |
| Repository | <https://github.com/Prithv122/production-rag> |
| Index dataset | <https://huggingface.co/datasets/Prithv122/production-rag-index> |
| Deployment | Google Cloud Run, `asia-south1` |
| Public URL | <https://production-rag-385330416945.asia-south1.run.app> |
| Status | **Shipped. Deployed and live, with two open defects on the deployed revision** (§12) |

---

## 2. What the system is

A RAG system built so that **retrieval quality is measured rather than asserted**. Every stage
is separable and separately scored: ingest → chunk → BM25 + dense index → fusion → rerank →
query rewrite → generation → citations/refusal, with an evaluation harness wrapped around it.

The corpus was chosen because projects 22 (dbt/Dagster) and 23 (DuckDB) mean the author can
personally judge whether a retrieved passage is genuinely relevant — an evaluation nobody can
read is ceremonial. The three tools also share colliding vocabulary ("dependencies", "models",
"assets"), which is what makes hybrid retrieval worth measuring rather than decorative.

Ingested by sparse git clone at **pinned commit SHAs** (`6f6cd165` / `cd0e5b0e` / `76eed340`),
not an HTML crawl, so any published number can be rebuilt from the bytes it came from.
**2,155 documents · 12.77 M characters.**

---

## 3. Retrieval results

Three chunking strategies: `fixed` (14,660 chunks), `heading` (22,789), `heading_ctx` (24,120).
184 questions, k=10, pool 50, scored against `(doc_id, start, end)` character-span labels so one
labelling effort scores all three strategies comparably.

**recall@5 / nDCG@10 on `heading`:**

| Arm | recall@5 | nDCG@10 | cost |
|---|---:|---:|---|
| `bm25` | 0.695 | 0.631 | 0.37 ms |
| `dense` | 0.555 | 0.499 | 1.07 ms |
| `hybrid` (RRF) | 0.688 | 0.602 | ~7 s |
| `hybrid_score` | 0.703 | 0.628 | ~9 s |
| **`hybrid_score_weighted`** | **0.706** | **0.658** | ~9 s |
| `hybrid_rerank` | 0.713 | 0.662 | ~781 s |
| `rerank_rewrite` (expand) | **0.730** | **0.665** | ~271 s |
| `rerank_rewrite_only` (replace) | 0.685 | 0.622 | ~226 s |

### The findings worth keeping

1. **Unweighted RRF hybrid loses to BM25 alone** — 0.602 vs 0.631 nDCG. The default hybrid
   configuration most tutorials ship is worse than the free lexical baseline on this corpus.
2. **Score fusion beats rank fusion**, and weighting lexical ×2 makes `hybrid_score_weighted`
   the only hybrid that clearly beats BM25 on both metrics.
3. **The cross-encoder buys +0.004 nDCG for 87× the latency** over that weighted fusion
   (0.662 vs 0.658; 781 s vs 9 s). It is the largest single gain in the project over *RRF*
   (+6 nDCG) and still the wrong first move.
4. **`heading_ctx` is worse than `heading`** — the breadcrumb prefix, the entire reason that
   arm exists, hurt every metric.
5. **The pooled mean hides a reversal:** BM25 takes exact_term 0.933 → 0.616 conceptual; score
   fusion takes conceptual 0.640 → 0.904 exact_term. Nothing beats BM25 on exact terminology.
6. **`cross_tool` is 0.000 for every arm** (n=4). Verified *not* a labelling bug — issuing the
   gold quote itself as a query returns its chunk at rank 1. The questions are simply beyond
   the system.

### Query rewriting, and a prediction that was half wrong

The pre-registered prediction was that `replace` would damage exact-terminology retrieval.
**It did the opposite**: replace produced the project's best exact_term score (**0.913**),
because the prompt forbids paraphrasing identifiers, so the rewrite keeps the rare token and
drops "how do I". The damage landed on **conceptual questions instead (−5.4 points)**. The
conclusion is "expand, don't replace" — recorded as prediction-vs-outcome rather than quietly
rewritten.

### The caveat that qualifies all of it

174 of 184 questions were written by an LLM *from a passage it had just read*, so questions
reuse vocabulary the passage contains. **That structurally inflates the BM25 column.** The
`conceptual` split, where the generator was told to avoid distinctive terms, is the more
trustworthy comparison; the hand-written cross-tool questions are the least contaminated.

---

## 4. Ground truth — **⚠︎ partially verified, not verified**

> Correcting a claim of "Ground truth: verified". **60 of 184 questions were human-reviewed.**

- **60 reviewed, 60 accepted, 0 edited, 0 rejected → correction rate 0/60.**
- Re-cutting the grid on those 60: every metric falls 5–9 points and **no conclusion changes**.
- With zero corrections the drop *cannot* be a correction — it is sampling. `production-rag band`
  resamples and shows **a 60-question subset moves recall@5 by ±0.10 at 95%**, wider than most
  gaps in the tables. **The arm ordering is the result; the third decimal is decoration.**
- The reviewed 60 were an ordered **prefix**, not a random sample, and sit at or below the
  band's lower edge. `verify` now walks pending questions in seeded-shuffled order.
- **Still open:** 124 unverified, and 0/60 cannot distinguish "the set is clean" from "the
  reviewer accepted too readily" — accept is the cheapest key in that loop.

Screening before the human pass is real, though: of 237 proposals, 43 were rejected for a quote
that does not exist in the source document and 20 for being too short.

---

## 5. Answer quality — three **local** models

> **⚠︎ This is not the hosted comparison that was planned, and is never presented as one.**
> A credit-less OpenRouter account is capped at **50 free-model requests per day, account-wide**
> (429 `free-models-per-day`, verified against several different free models — switching model
> is not a workaround), and paid ids return **402**. The hosted comparison is *unavailable*, not
> completed. The local comparison is the reproducible free substitute: no key, no quota.

60 questions (all 6 unanswerable + all 4 cross_tool + a seeded proportional sample; 21 verified).
Retrieval **frozen** — every arm fed the identical `hybrid_score_weighted`/`heading` top-10 — so
no row difference can be a retrieval difference. Fallback **disabled**, `fell_back` reported.

| Arm | grounded | uncited | refusal recall | false refusal | citation validity | median s |
|---|---:|---:|---:|---:|---:|---:|
| `qwen2.5:7b-q3_K_M` | 0.341 | 0.390 | 1.000 | 0.241 | 1.000 | 114.9 |
| **`qwen2.5-coder:7b-q3_K_M`** | **0.478** | **0.174** | 1.000 | **0.148** | 1.000 | 113.3 |
| `llama3.2:3b` | 0.333 | 0.222 | 0.833 | 0.167 | 1.000 | **60.5** |

**No LLM judge was appointed.** Every metric is arithmetic over something that either happened
or did not, because a judge would make the headline depend on a model a reader cannot pin,
cache or re-run.

1. **Instruction tuning was a controlled test and it won.** The two qwen arms share
   architecture, parameter count, quantisation, prompt and context — only tuning differs.
   Code-tuned grounds +13.7 points, and **0.786 vs 0.500 on exact_term**.
2. **The largest effect in the section was my prompt, not any model.** The first honest run
   scored **0.977 uncited**: the rules demanded `[n]` markers while the output example showed
   none, and an example is a stronger instruction than a rule. Fixing it → 0.174–0.390, a
   bigger move than the entire best-to-worst model gap. The *first* repair backfired — a
   realistic example (`on_schema_change` / `append_new_columns`) was plagiarised verbatim by
   the 3B model into a Dagster answer — so the example is now neutral placeholder text.
3. **141 citation markers written, 141 resolved, zero fabricated.** Reported as the null result
   it is: "none observed at n=141", not "these models do not fabricate".
4. **`llama3.2:3b` at under half the parameters** grounds level with the 7B general model at
   roughly half the latency, but missed one of six unanswerable questions and failed to emit
   parseable JSON three times as often.

---

## 6. The integrity failure, and what was built because of it

**A published table was one model wearing four hats.** The first answer grid produced four
near-identical rows — same 0.185 refusal rate, the same 10 of 54 questions refused by all four.
Every OpenRouter call had failed and `FallbackProvider` had silently answered with local Ollama.

Three faults stacked:

1. **429 account-wide quota** made every hosted arm unreachable.
2. **`nemotron-ultra` had never executed once** — its id was `nvidia/nemotron-3-ultra-550b:free`,
   missing the `-a55b` suffix, so every call was a 400. Session 2 had described its behaviour in
   a code comment on the basis of zero executions.
3. **The fallback hid both.** Graceful degradation is a feature of the demo and a falsification
   inside a comparison.

**It also retracted a published attribution.** Auditing the *previous* session's committed
cache: of 337 rewrite calls credited to a 120B model, **279 (83%) were answered by local
qwen2.5**. The retrieval comparison survives — every arm consumed the same committed rewrites,
which replay byte-identically — but the attribution was wrong, and so was a pooled "median 65 s"
that averaged two models and described neither (real: 34.8 s for 58 nemotron calls, 53.9 s for
279 local ones). Both corrected in the README with the split.

**Three structural fixes, not a footnote:**

- `answer-eval` builds providers with `fallback=False`.
- `ArmSummary.fell_back` reports, per arm, the share of rows answered by a different model.
- `production-rag cache audit` compares requested vs answering model across the committed
  bundles and **exits non-zero** on disagreement. It exits 1 on this repository *on purpose* —
  those 279 entries are required to replay published numbers, so they stay and the command
  tells the truth about them.

Both fields had been on disk, in git, for a session and a half with nothing reading them.

---

## 7. Other real bugs found and handled

- **Silent encoder truncation.** Session 1 assumed zero; actually **1.08% / 0.38% / 0.66%** of
  chunks exceed bge-small's 512-token window. Cause is **not** code fences — it is markdown
  table rules, which tokenise at ~1.00 chars/token against a corpus mean of 3.6. Fix deferred
  with the reason stated (it invalidates all three dense indexes *and* the cross-encoder cache);
  a parametrised slow test now bounds the rate so it cannot grow unnoticed.
- **Multi-span gold fallback.** The gold fallback was decided per *question* instead of per
  *evidence span*, silently halving the gold set for multi-span questions. Fixed, regression
  test added.
- **Model self-labelling.** The generating model's own literal/conceptual label disagreed with
  the corpus-computed category on **72/184 (39%)** — which is the argument for computing it.
- **Clean-clone gap.** `answer-eval --replay` needs `indexes/heading/chunks.jsonl`, which is
  gitignored. Now documented as two required rebuild steps (commit `a3bbe62`).

---

## 8. Replay

Every published number reproduces offline. The LLM and cross-encoder caches ship as sorted
JSONL bundles; `--replay` turns a cache miss into an error rather than a live call. **Verified**
by rebuilding a fresh cache from the committed bundle alone, with `OPENROUTER_API_KEY=""` and
`OLLAMA_HOST` pointed at a dead port — identical figures. The bundle was pruned to exactly what
published numbers depend on (517 entries; the 251 from the discarded first grid dropped).

---

## 9. Deployment

**Hugging Face Spaces was blocked** — `402 Payment Required` for a Gradio Space on free
`cpu-basic`; only static Spaces are free. A client-side static rewrite was rejected because it
would stop the hosted demo running the code the numbers were measured on. Retargeted at
**Google Cloud Run**, which runs the same package and the same `space/app.py` — a deployment
change, not an application rewrite.

**The index is baked into the image at build time**, still from the pinned HF dataset. A cold
start that *downloads* the 61 MB index measured **256 s**, against Cloud Run's 240 s startup
probe — fetching at boot would have been a coin flip on the revision ever going healthy.

### Infrastructure

| | |
|---|---|
| GCP project | `production-rag-2026` · region `asia-south1` · billing enabled |
| Artifact Registry | `asia-south1-docker.pkg.dev/production-rag-2026/production-rag/production-rag:latest` |
| Deployed digest | `sha256:d6f2fc227912b5695b2410185b0e5474675fd6d2a5e44c5fe503111f0c748c81` |
| Revision | `production-rag-00001-r47`, 100% traffic |
| Config | 2 GiB · 2 vCPU · **concurrency 4** · min 0 / max 2 · timeout 300 s · public |
| Secret | `openrouter-api-key` via Secret Manager → `OPENROUTER_API_KEY` |
| Runtime SA | `385330416945-compute@developer.gserviceaccount.com`, granted `roles/secretmanager.secretAccessor` |

> **⚠︎ Deliberately omitted:** the billing account identifier. It is not a credential, but it is
> an account identifier with no reason to sit in a public portfolio repository.

**The API key was never** committed, placed in the image, written to `.env`, passed on a command
line, or given to the coding agent. It reaches the container only through Secret Manager.

### Measurements — local and production kept separate

These measure **different things** and are not a like-for-like comparison. The local figure is
*process start → first HTTP 200* (boot + model load + index load). The Cloud Run figures are
*HTTP response latency* against the deployed service, where the instance may already have passed
its startup probe, and where Cloud Run applies a startup CPU boost.

| Local container (Docker, 2 GiB / 2 CPU, image present) | Measured |
|---|---:|
| Container start → first HTTP 200 | **62 s** |
| Warm page response | 0.008 s |
| Warm retrieval query | < 1 s |
| Generation via local Ollama (`ollama-qwen`) | 19–39 s |

| Cloud Run (`asia-south1`) | Measured |
|---|---:|
| Cold HTTP response | **24.9 s** |
| Warm HTTP response | **0.21 s** (independently re-measured: 0.219 s) |

| Image | |
|---|---:|
| Size, uncompressed | 2.78 GB |
| — dependency layer | 1.61 GB (CPU torch, transformers, gradio 6, scipy) |
| — baked index / encoder | 61 MB / 129 MB |
| torch build | `2.14.0+cpu`, `torch.version.cuda is None`, no `nvidia-*` packages |

> **⚠︎ An earlier reading of 691 MB was wrong** — taken from `docker images` while layers were
> still committing. The CPU-only pin did work; the size claim did not.

---

## 10. Production verification — what was actually checked

Verified against the **live public URL** (not the local container):

| Check | Result |
|---|---|
| Service returns HTTP 200 | ✅ |
| Gradio UI renders **warm** | ✅ 22,789 chunks, `generation nemotron-super` (secret bound) |
| Retrieval | ✅ scores **byte-identical to local** (2.9899 / 2.2363 / 1.9199 …) |
| Gradio UI on a **cold** load | ❌ see §12.1 |
| Generation + citations | ❌ see §12.2 |

---

## 11. Engineering lessons

1. RRF is not automatically better than BM25 — measure the default before shipping it.
2. Score-weighted fusion beat rank fusion; one weight beat a cross-encoder at 87× the latency.
3. Heading-aware chunking beat the heading-*context* variant, contradicting the hypothesis.
4. BM25 owns exact identifiers; hybrid helps conceptual queries. The pooled mean hides this.
5. Query expansion helped; replacement helped exact-term and hurt conceptual.
6. **The question-generation methodology itself biases the evaluation** — say so, and report the
   least-contaminated split separately.
7. **A silent fallback can invalidate every model-comparison claim you publish.** Audit the
   provenance of results, not just their values.
8. **A prompt's example outranks its rules.** The single largest metric movement in the
   generation section came from fixing an example, not from changing models.
9. A 60-question subset carries ±0.10 at 95%. Publish the ordering, not the third decimal.
10. Clean-clone testing found a real reproducibility gap that all the tests missed.
11. Production behaviour differed from local assumptions — so it was measured, not estimated.
12. Retract and correct beats preserving a better-looking number.

---

## 12. Open defects and limitations

### 12.1 Cold page loads return 429 on static assets — **fix committed, needs redeploy**

The deployed revision runs `--concurrency 4`. A Gradio page fires **~65 parallel static-asset
requests** on load, so while only one instance is up during a cold start, roughly half the
bundle returns **HTTP 429** and the page hangs on "Loading…" forever. Warm loads are fine.

This was caused by a configuration recommendation of mine that conflated **HTTP concurrency**
with **compute concurrency**. Fixed by widening the edge (`--concurrency 80`) and narrowing
compute in-process (`demo.queue(default_concurrency_limit=2)`), which is the correct layer.

### 12.2 Generation returns `Refused (unparseable)` — **fix committed, needs redeploy**

Every generation request against the deployed revision refused with reason `unparseable`
(observed at 2.2 s, 4.5 s, 11.8 s). Root cause, reproduced locally: under
`response_format: json_object`, `nemotron-3-super` can satisfy the constraint with the **empty
object `{}`** — it spends its budget reasoning and emits nothing. That parses cleanly and then
`payload["answer"]` raises `KeyError`. **Raising `max_tokens` does not help**, because nothing is
being truncated (verified at 700 and 2100). It is intermittent and prompt-specific — 2 of 3
sampled questions answered normally at the same settings.

Fixed by detecting a vacuous-but-valid payload and retrying **once with the format constraint
dropped**, letting `extract_json` repair the model's natural output. Validated against the exact
question that previously returned `{}`: it now answers with three valid citations.

> **⚠︎ This corrects an earlier "Citations ✅ / Generation ✅" claim about the deployed service.**
> Citations and refusal are verified on the *local container* and in the eval harness; on the
> *deployed revision* generation is currently broken.

### 12.3 Hosted model comparison — external, unresolved

OpenRouter's free tier is **50 requests/day, account-wide**, and resets daily at 00:00 UTC (it
had reset and was working again when this was written). Paid models require purchased credits.
The project therefore does **not** claim a completed hosted multi-model comparison, and no
merged table mixing provider families was ever published.

### 12.4 Evaluation limits

124 of 184 questions unverified; n=6 unanswerable and n=4 cross_tool are too small to support
strong claims; `grounded` measures citation targeting, not correctness.

---

## 13. Status

**Shipped and deployed.** Code complete, **354 fast tests + 18 slow passing**, ruff clean, CI
green, evaluation audited and replayable, secrets clean, public URL live.

Two defects are open on the deployed revision (§12.1, §12.2). **Both fixes are committed and
tested; neither is live until the image is rebuilt and redeployed** —
`docker build -f space/Dockerfile -t ... .` then `gcloud run deploy` per
[`space/DEPLOY.md`](space/DEPLOY.md), adding `--concurrency 80`.
