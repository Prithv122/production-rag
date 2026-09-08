# Interview Prep — production-rag

**Seven questions, seven answers.** An unanswered question means this project is not shipped.

If you can't answer one, you don't understand that part of your own project yet — go back and understand it. This file is the difference between a portfolio that survives a technical screen and one that collapses in it.

> Answers cover retrieval and generation, both measured. Where a number is thin — n=6
> unanswerable questions, 60 of 184 verified — the answer says so rather than rounding it up.

---

### Q1. Walk me through the architecture in 90 seconds.

_A:_ Three documentation repos — DuckDB, dbt, Dagster — are sparse-cloned at pinned commit
SHAs and normalised to 2,155 markdown documents, 12.8 M characters. Pinning the SHA is what
makes a published number reproducible: a branch moves, a commit doesn't.

Those documents are chunked three different ways — fixed windows, one chunk per markdown
section, and sections plus a heading breadcrumb — producing 14.7k / 22.8k / 24.1k chunks.
Each set is indexed twice: BM25 over a precomputed scipy sparse weight matrix I wrote in the
repo, and dense vectors from bge-small searched exactly with numpy. No ANN — at 23k × 384
that's a 35 MB matrix and one matrix-vector product, so FAISS would add a dependency, a
tuning knob and approximation error to speed up a 1 ms operation.

At query time the two rankings are fused — reciprocal-rank or normalised-score, weighted or
not — into a candidate pool of 50, optionally reranked by a cross-encoder, and optionally
preceded by an LLM query rewrite. Seven arms in total, all data-driven from one `ArmSpec`
table so the CLI, the eval harness and the demo run the same code path.

The whole thing is wrapped in an evaluation harness scoring recall@k, nDCG, hit-rate and MRR
against 184 questions labelled with document character spans, broken down by question
category. That harness is the point of the project; the answer generator is a swappable
component on the end of it.

### Q2. Why did you choose score fusion over RRF?

_A:_ Because I measured both instead of picking one, and RRF lost — which was not what I
expected.

The reason to expect a difference is that the two arms are not on the same scale. BM25 has a
true zero floor: a chunk sharing no query term scores exactly 0 and is genuinely *absent*.
Cosine similarity ranks the entire corpus every time — there is no absent, only less similar.
RRF handles that by discarding scores and using rank alone, which is elegant and throws away
the information that one candidate beat its runner-up by a mile.

Measured on 184 questions: unweighted RRF hybrid scored 0.602 nDCG@10, *below* BM25 alone at
0.631. Score fusion got 0.628 — better, still not beating the lexical arm. The fix was
weighting: the dense arm is 14 points of recall worse here, and unweighted fusion gives it
exactly as many votes as the strong arm, so it dilutes rather than adds. Weighting lexical
×2 with score fusion gives 0.658 nDCG and 0.706 recall@5 — the only hybrid configuration that
clearly beats BM25 on both metrics.

The transferable point is that "hybrid beats single-arm retrieval" is a claim about a corpus,
not a law, and the default configuration most tutorials ship loses on this one.

### Q3. What's the weakest part of this, and what would break first under load?

_A:_ The weakest part is the evaluation set, and I would say that before an interviewer got
to it. 174 of 184 questions were written by an LLM from a passage it had just read, so the
questions reuse vocabulary the passage contains — which structurally flatters lexical
retrieval and inflates the BM25 column specifically. I built three defences (the claimed
quote must locate verbatim in the real document, which rejected 18% of proposals; the
question category is computed from corpus document frequency rather than taken from the
model's self-label, which disagreed 39% of the time; and a human verification pass), but the
first two only remove obvious failures and the third has covered 60 of 184 questions with
**zero corrections** — which is a fact about the screening in front of it as much as about
the questions, and cannot distinguish a clean set from a reviewer who accepted too readily.

The re-cut on those 60 is in the README and it changes no conclusion, but it is the second
thing I'd flag: every metric drops 5–9 points on the verified subset, and resampling says
that is what a 60-question slice does. A 60-question subset moves recall@5 by ±0.10 at 95%.
So the arm *ordering* is the result; the third decimal place in those tables is decoration,
and I built `production-rag band` to be able to say that with a number rather than a
feeling.

Under load, exact search breaks first, and not where it looks. At 100× (~2.4 M chunks) the
vector matrix is ~3.5 GB — still loadable, but a query moves 3.5 GB through memory, so
latency goes from sub-millisecond to hundreds of milliseconds and *memory bandwidth*, not
compute, is the wall. That's when HNSW earns its approximation error — and the honest way to
introduce it is to measure recall against the exact baseline this repo already has, because
the exact result is the ground truth ANN gets compared to.

Second to break is BM25's rebuild model: 1.1 M nonzeros becomes ~110 M, and the whole-index
rebuild on every ingest stops being viable. Incremental indexing means per-segment IDF, which
means scores are no longer comparable across segments — a real correctness problem, not a
performance one.

### Q4. How do you know it works? What did you measure, and against what baseline?

_A:_ Every arm is measured against BM25 alone, which is the baseline that matters — it's what
you get for free, with no model, no GPU and 0.37 ms per query.

184 questions, k=10, pool of 50, scored on recall@1/5/10, hit-rate@5, nDCG@10 and MRR, per
question category and never pooled into a single headline. Best configuration:
`rerank_rewrite` at 0.730 recall@5 / 0.665 nDCG@10 against BM25's 0.695 / 0.631.

The number I'd actually lead with is a cost one. The cross-encoder gives the single largest
gain in the project — +6 nDCG points over RRF hybrid — and it takes 110× longer. But a
weighted score fusion, which is one weight and one fusion choice, reaches 0.658 nDCG in 9
seconds against the reranker's 0.662 in 781 seconds. That's **0.004 nDCG for 87× the
latency.** On this corpus at this scale, tuning the fusion is the better engineering decision
and the cross-encoder is what you add afterwards for the last half-point.

Ground truth is labelled as `(doc_id, start, end)` character spans rather than chunk ids,
which is what makes the three-strategy comparison possible at all — the strategies share no
chunk ids, so a chunk-level label would belong to exactly one of them. And every published
number replays offline: the LLM and cross-encoder responses ship in the repository as sorted
JSONL bundles, and `eval --replay` turns a cache miss into an error rather than a live call.
I verified that with an empty API key and a dead Ollama port.

### Q5. Your eval questions were generated by an LLM from the passages you're scoring against. Isn't that circular?

_A:_ Partly, yes, and the honest answer is to say where it bites rather than to claim it
doesn't.

It is circular in one specific direction: the model writes questions using vocabulary it has
just read, so the question and its gold passage share tokens more often than a real user's
question would. That inflates lexical retrieval. It is *not* circular in the sense that would
invalidate the comparison — every arm is scored against the same labels, so a bias that lifts
BM25 lifts BM25's contribution to every hybrid arm too.

Three things bound it. First, the claimed evidence quote has to locate verbatim in the real
document; 43 of 237 proposals (18%) claimed a quote that doesn't exist and were rejected
rather than becoming mislabelled gold. Second, the category is computed from corpus document
frequency — does the question share a token with its own evidence appearing in ≤50 of 23,000
chunks? — rather than from the model's own label, which disagreed with the computed one on 39%
of questions. That matters because the `conceptual` bucket, where the generator was
instructed to avoid the passage's distinctive terms, is by construction the least contaminated
part of the set, and it's where the arms reorder: BM25 leads exact-terminology 0.933 to 0.904,
score fusion leads conceptual 0.640 to 0.616.

Third, the 10 hand-written questions — cross-tool and unanswerable — have no generator bias at
all. They're also where everything fails: 0.000 recall on cross-tool for every arm. I checked
that isn't a labelling bug by issuing the gold quote itself as a query, which returns its
chunk at rank 1. The questions are simply beyond the system. n=4, so it's a signal to write
more of them.

What would actually fix it is questions sampled from real user logs. I don't have those, and
saying "the eval set is the bottleneck" is more useful than pretending 184 synthetic questions
characterise a production system.

### Q6. How do you score the *answers*, given there's no gold answer text — and why not an LLM judge?

_A:_ I deliberately didn't appoint one. An LLM judge makes your headline number depend on a
model you can't pin, can't cache honestly across versions, and a reader can't re-run — and
this project's entire claim is that its numbers reproduce offline for someone who doesn't
trust me. So every answer metric is arithmetic over something that either happened or didn't:

- **Citation validity** — the model is handed N numbered passages, so a `[n]` outside `1..N`
  is a fabricated source, catchable by arithmetic. 141 markers written across three models,
  141 resolved, zero fabricated. That's a null result and I report it as one: "no fabrication
  observed at n=141", not "these models don't fabricate".
- **Grounded** — the answer cited at least one chunk containing a gold evidence span. Reuses
  the retrieval labels, so it's free over the whole set. It is *not* "the answer is correct";
  a model can cite the right passage and summarise it wrongly, and the README says so.
- **Refusal recall paired with false refusal**, never one alone — a model that refuses
  everything scores 1.000 on the first.

The result I'd lead with is a controlled one. `qwen2.5:7b` and `qwen2.5-coder:7b` are the same
architecture, same parameter count, same q3_K_M quantisation, same prompt, same frozen
retrieved context — the only variable is instruction tuning. The code-tuned model grounds
**0.478 vs 0.341**, and on exact-terminology questions **0.786 vs 0.500**. On API
documentation, which 7B you pick matters more than the prompt engineering on top of it.

Two honesty points I'd raise before being asked. First, **the biggest effect in that whole
section was my bug, not a model**: the first run scored 0.977 uncited, because my prompt's
rules said "cite with `[n]`" while the output example directly below showed an answer with no
markers in it. An example is a stronger instruction than a rule. Fixing the example moved
uncited to 0.174–0.390 — further than the gap between the best and worst model.

Second, **this is a local three-model comparison, not the hosted one I planned.** A
credit-less OpenRouter account is capped at 50 free-model requests a day, account-wide, and
paid models 402. I say that rather than quietly relabelling local models as the hosted table.

### Q7. Tell me about a bug you shipped and how you caught it.

_A:_ I published a table that was one model wearing four hats.

The first answer eval compared four models and produced four near-identical rows — same
refusal rate to three decimals, the same 10 of 54 questions refused by all four. A 120B, a
550B, a hosted Gemini and a 7B local quant do not agree perfectly. It was one model: every
OpenRouter call had failed and `FallbackProvider` had silently answered with local Ollama.

Three faults stacked. The account can't reach hosted models (429 free-tier daily cap, 402 for
paid). One arm had a typo'd model id — `nemotron-3-ultra-550b:free` instead of
`...-550b-a55b:free` — so it had *never once executed*, including in the previous session
where I'd described its behaviour in a code comment. And the fallback made both invisible by
design: graceful degradation is a feature of the demo and a falsification inside a comparison.

Then it got worse, in the useful way. I wrote an audit over the committed response cache and
pointed it at the *previous* session's bundle: of 337 rewrite calls I'd attributed to a 120B
model, **279 (83%) had been answered by local qwen**. The retrieval comparison survived —
every arm consumed the same committed rewrites and they replay byte-identically — but the
attribution in my README was wrong, and so was a latency figure that averaged two models and
described neither.

What I actually take from it: every one of those faults was *already recorded in an artefact
I had committed to git*. The cache stored the requested model and the answering model side by
side, for a session and a half, and nothing read them. A resilience feature and an integrity
check want opposite things from the same event, and I'd built only the resilience half. The
fixes are structural — `answer-eval` disables fallback, the summary reports a per-arm
`fell_back` rate, and `cache audit` exits non-zero on any requested-vs-answered mismatch.

---

## 30-second pitch

A RAG system over DuckDB, dbt and Dagster documentation — 2,155 real documents at pinned
commit SHAs — built so that retrieval quality is measured rather than asserted. Seven
retrieval arms across three chunking strategies, scored against 184 questions labelled with
document character spans so the same labels work for every strategy.

The result that made it worth building: the default hybrid configuration — reciprocal-rank
fusion of BM25 and dense — scored *below* BM25 alone, 0.602 nDCG against 0.631. Weighting the
lexical arm and switching to score fusion fixed it, at 0.658. And a cross-encoder reranker,
the expensive stage everyone adds, bought 0.004 nDCG over that for 87× the latency.

The lesson I took from it is that the most sophisticated-looking configuration was not the
best one, and the only reason I know that is that I built the measurement before I built the
opinion.
