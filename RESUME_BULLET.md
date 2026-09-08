# Resume Bullets — production-rag

Form: **action → technical specifics → measured outcome.** Numbers or it doesn't go on the resume.

> Retrieval and generation bullets are final.

---

## Bullets

- Built a retrieval evaluation harness over 2,155 pinned documentation pages (12.8M chars,
  DuckDB/dbt/Dagster) scoring 7 retrieval arms × 3 chunking strategies against 184
  span-labelled questions, and found the default hybrid configuration — reciprocal-rank
  fusion of BM25 and dense vectors — scored **below BM25 alone** (0.602 vs 0.631 nDCG@10);
  weighting the lexical arm and switching to score fusion recovered it to **0.658**.

- Quantified the cross-encoder reranking tradeoff instead of assuming it: +6 nDCG points over
  RRF hybrid, but only **+0.004 nDCG for 87× the latency** (781s vs 9s) over a weighted score
  fusion — so the fusion tuning ships and the reranker is documented as the last-half-point
  option.

- Designed ground truth as `(doc_id, character span)` labels rather than chunk ids, so a
  single labelling effort scores all three chunking strategies comparably; derived per-strategy
  gold sets by span overlap, with 99.4–100% evidence coverage reported alongside every metric.

- Ran query rewriting as a measured arm with a pre-registered prediction, and reported the
  half that was wrong: retrieving with the rewrite *instead of* the original raised
  exact-terminology recall to **0.913** (predicted to fall) while costing conceptual questions
  **5.4 points** — conclusion "expand, don't replace", not "rewriting improves retrieval".

- Wrote Okapi BM25 in-repo over a precomputed scipy sparse weight matrix (**0.37 ms** mean
  query over 22,789 chunks) and exact numpy cosine search (**1.07 ms**), rejecting FAISS/HNSW
  with a stated scale threshold rather than by default.

- Made every published number reproducible offline: LLM and cross-encoder responses ship as
  sorted JSONL bundles and `eval --replay` turns a cache miss into an error, verified with an
  empty API key and an unreachable local model server.

- Scored generated answers **without an LLM judge** — citation validity, grounding against
  existing span labels, and refusal recall paired with false-refusal rate — then used it as a
  controlled test: `qwen2.5-coder:7b` vs `qwen2.5:7b` at identical size, quantisation, prompt
  and frozen retrieved context grounded **0.478 vs 0.341** overall and **0.786 vs 0.500** on
  exact-terminology questions, isolating instruction tuning as the cause.

- Traced a 0.977 uncited-answer rate to my own prompt rather than the models — the rules
  demanded `[n]` citations while the output example showed none — and cut it to **0.174–0.390**
  with a one-line change, a larger effect than the entire best-to-worst model gap.

- Caught and retracted a published attribution by auditing the committed response cache:
  **279 of 337 calls (83%)** credited to a 120B hosted model had silently been answered by a
  local 7B via the provider fallback chain; added `fell_back` reporting, disabled fallback
  inside the evaluation, and shipped a `cache audit` command that exits non-zero on any
  requested-vs-answered model mismatch.

- Put an error bar on the project's own tables: bootstrap resampling showed a 60-question
  subset moves recall@5 by **±0.10 at 95%**, wider than the gaps between most arms — which
  reframed the result as the arm *ordering* rather than the third decimal, and caught that a
  partially-verified question set was an ordered prefix rather than a random sample.

## Which roles this supports

- [ ] Data Scientist / ML
- [x] AI Engineer (LLM/NLP/CV)
- [ ] Data Engineer
- [x] Data Analyst / Python Developer

## Keywords this project earns

_Only list what you actually used and could be questioned on._

RAG · hybrid retrieval · BM25 (Okapi, implemented) · dense retrieval · sentence-transformers ·
bge-small-en-v1.5 · cross-encoder reranking · reciprocal rank fusion · score fusion ·
query rewriting · retrieval evaluation (recall@k, nDCG, MRR, hit-rate) · ground-truth labelling ·
chunking strategies · scipy sparse · numpy · OpenRouter · Ollama · provider abstraction ·
fallback and graceful degradation · response caching and offline replay · citation grounding ·
refusal/abstention evaluation · prompt engineering (measured) · bootstrap confidence intervals ·
experiment provenance auditing · Gradio · Hugging Face Hub · pytest · ruff · uv · CI

---

### Bad vs good

❌ "Built a machine learning model to predict customer churn using Python."
✅ "Built a churn classifier on 240k accounts (LightGBM, 1:40 class imbalance) with isotonic calibration and cost-sensitive thresholding, lifting precision@10% from 0.31 to 0.58 over the business's existing rules baseline."

The second one is answerable in an interview. The first invites the question you can't answer.
