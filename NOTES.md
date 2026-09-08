# Build notes

What broke, what was tried, and why one option beat the other. Newest section last.

---

## Corpus selection

**Why these three tools.** DuckDB, dbt and Dagster documentation, because projects 22
(dbt + Dagster) and 23 (DuckDB) in this portfolio were built with them. That matters for
a retrieval project specifically: evaluating retrieval means judging whether a returned
passage actually answers the question, and that judgement is worthless from someone who
cannot read the passage. A corpus chosen for convenience — arXiv, Wikipedia — makes the
evaluation ceremonial.

The second reason is that the three vocabularies *overlap and collide*. "Dependencies"
means DAG edges in dbt and asset dependencies in Dagster. "Models" means a dbt SQL file,
not an ML model. "Assets" is a Dagster primitive and a dbt-adjacent concept. A query like
"how are dependencies declared?" is genuinely ambiguous across the corpus, which is what
makes hybrid retrieval and reranking worth measuring rather than decorative.

### DuckDB: `docs/current` only, not `docs/`

`duckdb/duckdb-web` ships every documentation version simultaneously:

| Path | Files | Size |
|---|---:|---:|
| `docs/0.10` | 296 | 2.4 MB |
| `docs/1.0` | 315 | 2.5 MB |
| `docs/1.1` | 346 | 3.0 MB |
| `docs/1.2` | 361 | 3.2 MB |
| `docs/1.3` | 370 | 3.7 MB |
| `docs/lts` | 385 | 3.9 MB |
| **`docs/current`** | **434** | **4.4 MB** |

Ingesting all 2,507 files would put six near-identical copies of most pages into the
index. Two consequences, and the second is worse than the first:

1. Retrieval precision collapses for reasons that have nothing to do with the ranker —
   the top 10 fills with the same passage at six versions.
2. **Ground truth stops being well-defined.** "Which chunk answers this question?" has six
   equally correct answers, so recall@k becomes a measurement of arbitrary tiebreaking.

So the shipped corpus is `docs/current` (430 documents after normalisation). The other
trees are reachable behind `production-rag ingest --include-version-archive` purely to
quantify the damage — the ablation is reported in the README rather than the decision
being asserted.

### Ingest by git, not by crawler

Sparse partial clone (`--filter=blob:none` + `sparse-checkout`) at a pinned commit SHA,
rather than scraping the rendered docs sites. Markdown source has no nav chrome, no cookie
banner and no duplicated sidebar; the LICENSE travels with the repo so the corpus licence
is verifiable rather than claimed; and the whole Dagster monorepo costs seconds instead of
gigabytes. Most importantly the corpus is addressable: "rebuild from SHA `76eed340`"
reproduces exactly the corpus a published number came from. "I scraped it in September"
does not.

Licences confirmed against the GitHub licence API at pin time, not assumed:
`duckdb/duckdb-web` MIT, `dbt-labs/docs.getdbt.com` Apache-2.0, `dagster-io/dagster`
Apache-2.0.

---

## Normalisation: two silent corruption bugs

Both were found by scanning the *whole corpus* for leftover markup rather than eyeballing
three files. Neither raised an error; both would have quietly degraded retrieval.

### 1. Single-pass tag stripping left markup in 135 documents

The dbt docs contain tags whose attributes embed a placeholder in angle brackets:

```
<File name='models/<filename>.yml'>
```

`re.sub` scans left to right. The outer `<File ...>` cannot match, because the attribute
pattern `[^<>]*` will not cross the inner `<`. So the inner `<filename>` is removed
instead, and the pass ends — leaving `<File name='models/.yml'>` in the text. A second
pass would match it, but there was no second pass.

**Fix:** iterate the substitution to a fixed point (bounded at 5 passes). Leftover
JSX-tag documents went 135 → 1.

**The survivor, deliberately not fixed.** One document contains
`<DetailsToggle alt_header="...(dev project <> prod project)?">` — a literal `<>` inside a
quoted attribute value. No regex handles that without becoming an MDX parser. That is
1 document in 2,155 (0.05%), and the correct engineering call is to quantify the residual
and stop, not to escalate a regex arms race for one page.

### 2. A greedy `\s*` relocated body text across newlines

The Docusaurus admonition pattern was `^:::[a-zA-Z]*\s*(.*)$`. In Python, `\s` matches
newlines. On a bare closing `:::`, the greedy `\s*` therefore consumed the blank line
after it and `(.*)` captured *the following paragraph*, which was then substituted in as
the directive's title. Body text moved silently; nothing errored.

**Fix:** `[ \t]*` instead of `\s*`, so the pattern cannot leave its line. Both bugs now
have named regression tests.

### Code fences are split out before any tag stripping

Not a bug that shipped, but the reason it did not. This is technical documentation: the
answer is frequently *inside* a code block, and code legitimately contains `<` and `>`.
Running a JSX tag stripper over `SELECT * FROM t WHERE a < b` or `std::vector<int>` eats
them. The normaliser splits on fences first and only rewrites prose segments.

---

## Chunking: two budget bugs, both found by measuring

Three strategies (`fixed`, `heading`, `heading_ctx`) share one packer so the evaluation
can choose between them instead of the author guessing.

### 1. The overlap tail was not charged against the budget

The tail of chunk *n* is prepended to chunk *n+1*, but only the *blocks* were counted when
deciding whether the next chunk was full. So chunks were built to `max_chars` and then had
up to `overlap_chars` added afterwards. **49% of `fixed` chunks exceeded the budget** and
ran past the encoder's context window — the overflow is truncated by the encoder, silently.

**Fix:** seed the next chunk's size with `len(tail)`. Over-budget went 49% → 9.1%.

### 2. An unwrapped paragraph could not be split at all

`_hard_split` divided oversized blocks on line boundaries. Markdown routinely stores an
entire paragraph on one physical line, and such a block has no line boundary to split on —
so it was emitted whole. A test with a single 10,000-character paragraph produced **one
9,999-character chunk against a 400-character budget**.

**Fix:** a line longer than the budget is split again on whitespace. Every chunk across the
real corpus now lands within `max_chars + overlap`.

This one is worth noting as a process point: it was caught by a *test I wrote to assert
the first fix*, not by inspection. The first fix was real but partial, and the failing
assertion is what exposed the second, deeper cause.

### The overlap tail starts at a word boundary

A raw `text[-200:]` slice starts the next chunk mid-word (`"n the query results..."`),
which is both unreadable in a citation and noise for the encoder. Advancing to the next
whitespace costs at most one word.

---

## BM25: written out rather than imported

Two reasons. The portfolio rule for a flagship project is that at least one core component
has to be legibly the author's own work. The practical reason: `rank_bm25` scores by
looping in Python over every document for every query, and this project runs a
chunking × arm grid over ~23k chunks — that is thousands of full-corpus scans.

Precomputing the whole weight matrix at build time turns a query into one CSC column
gather plus a row sum, because the document-dependent half of each term's weight does not
involve the query:

> **22,789 chunks · 41,599 vocabulary terms · 1.1 M nonzeros · 4.4 MB · 2.5 s to build ·
> 0.37 ms mean query latency**

### The tokenizer emits identifiers whole *and* split

`on_schema_change` becomes `["on_schema_change", "on", "schema", "change"]`. Keeping only
the whole identifier means the query "schema change" misses the page that only ever writes
`on_schema_change`; keeping only the pieces destroys the exact-match recall that BM25
exists to provide. Emitting both keeps each.

### No stopword list, deliberately

The obvious stoplist is `in`, `as`, `is`, `all`, `having`, `order`, `by` — and every one of
those is a SQL keyword carrying real meaning in this corpus. "order by" is a query someone
actually types. IDF already discounts corpus-frequent terms, and it does so from *this*
corpus rather than from a generic English list.

### Scoring is cross-checked against the formula

A hand-written implementation has nothing else to catch a flipped IDF or a wrong
denominator, so `test_scores_match_a_direct_formula_evaluation` recomputes Okapi BM25 the
slow, obvious way and compares.

---

## Dense retrieval: exact search, no ANN

~23k chunks × 384 dimensions is 35 MB as float32, and a full similarity pass is a single
matrix-vector product. FAISS or HNSW would add a dependency, a build step, an index-tuning
parameter and an approximation error — to accelerate an operation already measured in
milliseconds. The scale at which that trade flips belongs in README §7. Reaching for a
vector database at this size would be resume-driven development, and an interviewer who
knows the area will read it that way.

**The encoder is a Protocol, and torch is optional.** `sentence-transformers` lives in an
`embed` extra and is imported lazily. The test suite runs against a deterministic hashing
`FakeEmbedder`, so CI never downloads ~2 GB of wheels to verify that ranking works. CI
asserts torch is *absent* in the fast job, so nobody can quietly promote the dependency
and turn a 30-second job into a 10-minute one.

Project 20 shipped a 209-second suite and had to disclose it in its own README. Not
repeating that.

---

## Fusion: RRF and score fusion, both measured

BM25 and cosine are not on one scale, and the difference is not just magnitude — it is
semantic. BM25 has a true zero floor: a chunk sharing no query term is *absent*. Cosine
ranks the entire corpus every time; there is no absent, only less similar. Adding the raw
scores lets dense search's opinion about every irrelevant chunk outvote BM25's silence.

`rrf` sidesteps the scale problem by construction (rank only). `score` min-max normalises
each arm and keeps magnitude information, which is arguably better when one arm is very
confident. Which wins is empirical, so both are arms.

**Ties break by `chunk_id`.** Without an explicit tiebreak, two runs of the same evaluation
can report different recall@k from dict ordering alone — and a published number that moves
between runs is not a published number.

---

## CPU embedding is the real bottleneck, and the benchmark lied

Timing the encoder on short strings gave **160 texts/s**, which predicted ~2.4 minutes to
embed a 22,789-chunk index. That prediction is wrong by a wide margin: 13 minutes of wall
time (≈65 CPU-minutes across ~5 cores) had not finished the 14,660-chunk `fixed` index,
putting real throughput below **19 chunks/s** — at least 8× worse than the benchmark said.

The benchmark was wrong because the strings were wrong. It used ~50-character sentences
(~12 tokens); real chunks are ~1,000 characters (~250 tokens). Transformer cost at these
lengths is roughly linear in token count, so a ~20× longer input is roughly ~20× slower.
**Throughput per *text* is a meaningless unit for an encoder; throughput per *token* is
the one that transfers.**

### The three strategies became an accidental natural experiment

Building all three over the *same corpus* with the *same encoder* on the *same hardware* —
differing only in how the text was cut up — separates the two candidate units cleanly:

| Strategy | Chunks | Mean chars | Build | **chunks/s** | **tokens/s** |
|---|---:|---:|---:|---:|---:|
| `fixed` | 14,660 | 997 | 40.8 min | **6.0** | **1,494** |
| `heading` | 22,789 | 536 | 37.0 min | **10.3** | **1,375** |
| `heading_ctx` | 24,120 | 680 | 48.9 min | **8.2** | **1,397** |

Per-chunk throughput spans **1.7×**. Per-token throughput spans **8%** — and note the
ordering: `heading_ctx` sits between the other two on both mean chunk length and chunks/s,
exactly where a token-linear cost model puts it. That is about as direct a demonstration as
this project will produce that the encoder's cost is set by tokens, and that "texts per
second" is an artefact of whatever text length you happened to benchmark on.

It also retroactively explains the original error exactly: the toy benchmark used ~12-token
strings and reported 160 texts/s ≈ 1,900 tokens/s. The token figure was roughly right all
along. Only the unit was wrong.

Total: **127 minutes** for the full three-strategy grid, which is the cost the session-2
content-hash cache is meant to avoid paying repeatedly.

### The latency number moved three times, and the third one is right

The same `fixed` index measured **2.38 ms**, then **3.43 ms**, then **1.07 ms**. Nothing
about the index changed; only the machine's load did — the first two were taken while other
strategies were still embedding on five cores. Exact search is memory-bandwidth bound, so
it is far more load-sensitive than a CPU-bound operation would be.

The lesson is not subtle but it is easy to skip: **a latency figure without a stated machine
condition is not a measurement.** The README now reports idle-machine means over 500
queries and says so. The first two figures were published in this repository before being
corrected; they are recorded here rather than quietly overwritten.

Consequence: a full three-strategy rebuild is slow enough to make the chunking × arm grid
painful to iterate on. Fixes for session 2, in order of value:

1. **Cache embeddings by content hash.** `heading` and `heading_ctx` share a substantial
   amount of chunk text; today every strategy re-embeds from scratch.
2. Confirm no silent truncation — bge-small's window is 512 tokens and chunks top out
   around 350, so the character budget is comfortably inside it. Worth asserting rather
   than assuming, since truncation is silent.
3. Batch size and thread-count tuning, which is the smallest of the three wins.

Recording this mainly as a methodology point: the mistake was benchmarking on convenient
inputs instead of representative ones, and it produced an estimate that was wrong by more
than an order of magnitude.

## Session 1 boundary

Everything above runs with **no API key and no network** (after the one-time ingest).
Query rewriting, generation, the LLM judge and the answer metrics are session 2, and both
LLM-dependent steps go through an on-disk cache so `eval --replay` reproduces every
published number offline.

Retrieval *numbers* are not in this session either: they need ground truth, and ground
truth needs the LLM proposal pass plus the author's hand-verification. Session 1 ships the
machinery and the metric definitions with hand-computed tests; session 2 ships the table.

---

# Session 2 — reranker, rewriting, providers, ground truth

## The decision that shaped everything else: gold is a span, not a chunk

The obvious way to store ground truth is "question Q is answered by chunk 47". I got two
paragraphs into writing that before noticing it cannot work here. There are three chunking
strategies, they produce 14,660 / 22,789 / 24,120 chunks, and **no chunk id is shared
between them**. A chunk-level label belongs to exactly one strategy. Storing it that way
would mean either abandoning the cross-strategy comparison — the thing the project exists
to do — or labelling three times and then comparing numbers built on three different
labellings, which is worse because it looks like a comparison.

So a label is `(doc_id, start, end)` plus the verbatim quote at that span. Each chunker
already records the document span each chunk came from, so `gold_chunk_ids` derives a
strategy's gold set by overlap at scoring time. One labelling effort, three comparable
evaluations, and the labels survive a change to `--max-chars`.

The quote is not decoration. It is what lets a label be *re-verified* after a re-ingest: if
the pinned SHA moves and offsets shift, the quote either still locates or the label is
stale. Stale-and-loud beats wrong-and-quiet.

**Bug found in my own fallback rule.** When no chunk covers ≥50% of a span, the code falls
back to the best-overlapping chunk so a question never becomes unscoreable. The first
version checked "does this document already have a gold chunk?" — globally, across all of
the question's evidence. For a question with two spans *in the same document*, the first
span's match therefore suppressed the second span's fallback, silently halving the gold set
and depressing recall on exactly the multi-evidence questions the hybrid arm is supposed to
win. Fixed to decide per evidence item; a regression test asserts two spans yield two chunks.

## The generating model is not a reliable judge of its own output

An LLM writing questions from a passage and then being scored on whether retrieval finds
that passage is close to circular. Three defences, cheapest first.

**The quote must exist.** 43 of 237 proposals (18%) claimed a quote that is not in the
source document — usually a light paraphrase, occasionally an invention. Every one is
rejected rather than becoming a mislabelled gold span. This check costs nothing and removes
the single most common failure.

**The category is computed, not claimed.** The model was asked to write one "literal" and
one "conceptual" question per passage and to label which was which. Its label disagreed
with the corpus-statistics categorisation on **72 of 184 questions (39%)**. The
categorisation asks a checkable question — does the question share a token with its own
evidence that appears in ≤50 of ~23,000 chunks? — instead of asking a model to grade
itself. Spot-checking the disagreements, the computed label is the defensible one: a
question the model called "literal" that never mentions `EXPLAIN` is not a literal query,
whatever the model intended while writing it.

**A human verifies a sample.** Neither check above can tell whether a question is *sensible*
or whether its labelled passage is really the best evidence. That is the author's pass, and
until it happens the retrieval numbers are provisional and labelled as such.

Yield, for the record, because a pipeline that reports only its survivors is describing its
filter: 120 passages → 237 proposals → 174 accepted (73%), with 43 rejected for a
non-existent quote and 20 for being too short to be a real query. 0 context-dependent and 0
duplicates — those filters exist and fired zero times, which is worth saying rather than
quietly implying they did work.

## A "supports structured output" model that returns no JSON, and the one-line fix

`nemotron-3-super` is listed as honouring `response_format`, and session 1 verified that
against the live catalogue rather than trusting a blog post. It still failed to return JSON
on **30 of 120 passages**. The diagnosis was in the field I nearly did not record:
`finish_reason`. Every failure was `length`, not `stop` — the model spends a long reasoning
preamble ("We need to produce two questions: one literal, one conceptual...") and hit the
1,600-token ceiling before emitting a character of JSON.

The fix is a second pass over only the failures at 3× the budget. **All 30 recovered** —
100%. Two things follow. First, "supports structured output" is a statement about the API
contract, not about whether you will get JSON; a reasoning model needs headroom for the
reasoning *and* the answer. Second, retrying only the failures cost 30 extra calls instead
of re-running 120, because the successful responses were already cached under a key that
includes `max_tokens` and so were untouched by the change.

## Session 1's "worth asserting rather than assuming" turned out to be worth asserting

Session 1 argued from a chars-per-token estimate that no chunk exceeds bge-small's 512-token
window, and listed "confirm no silent truncation" as a session-2 task. It is not confirmed —
it is false:

| Strategy | chunks > 512 tokens | share | tokens discarded | chars/token (all) | chars/token (the offenders) |
|---|---:|---:|---:|---:|---:|
| `fixed` | 158 | 1.08% | 19,227 | 3.62 | **2.03** |
| `heading` | 86 | 0.38% | 10,484 | 3.58 | **1.78** |
| `heading_ctx` | 160 | 0.66% | 20,867 | 3.65 | **2.11** |

My first hypothesis — code fences tokenize badly — is also wrong, and measurably so: the
over-window chunks contain *less* code than average (3.8% vs 11.6% for `fixed`). Looking at
the actual worst offenders gives the real answer. The single worst chunk in the `heading`
index is 1,115 tokens from 1,119 characters — **1.00 characters per token** — and it is a
markdown table separator row: a line of pipes and hyphens.

**Markdown table rules are tokenizer poison.** They carry literally zero meaning and a
single row can consume the entire encoder window. Second place goes to identifier-dense
table bodies — DuckDB's encodings list, rows like `|770|` plus a backticked Java codepage
name — at 1.4 chars/token.

The fix belongs in normalisation: collapse table rules to a fixed short marker. It is
deferred, and the reason is a cost I would rather state than hide — changing chunk text
invalidates all three dense indexes *and* every cross-encoder score cached against those
chunks, which is a full re-run of the evaluation, for an effect bounded below 1% of chunks.
A parametrised `slow` test now asserts the measured share as an upper bound, so the number
cannot silently grow while nobody is looking.

The general lesson matches session 1's: **a budget in one unit, enforced against a limit in
another unit, needs a measurement at the boundary, not an estimate.** Characters are still
the right budget for the chunker — a token budget drags the tokenizer, and torch, into every
test — but the conversion factor has a 3.6× spread that an average hides.

## Reranking is a funnel stage, and it is not free

The cross-encoder reads `(query, passage)` jointly, so it cannot be precomputed and never
runs over the corpus — it rescores the pool the cheap arms produced. Measured on this
machine at ~500-character passages: **~8 pairs/s idle, ~3 pairs/s under load**, for
`ms-marco-MiniLM-L-6-v2` (22M parameters). That is why the larger `bge-reranker-base` (278M)
is not the default: reranking is the single most repeated operation in the grid, and an
order of magnitude more cost per pair would put a full evaluation out of reach on this
hardware. Choosing the higher-scoring model without pricing it is exactly the reasoning this
project exists to avoid.

The pool is deliberately separate from `k`. Reranking a top-10 can only reorder ten chunks;
the stage earns its cost by promoting a candidate that was at rank 34, and truncating to `k`
before reranking would throw away precisely those candidates.

Anecdotally — and it is an anecdote, the table is the evidence — the reranker *demoted* the
correct chunk on the README's worked example, from rank 1 under plain hybrid to rank 4.
Which is the point of running it as a measured arm rather than assuming it is an upgrade.

## Replay has to work from a clean clone or it is not replay

The project's rule is that every published number reproduces offline. That was already true
for retrieval and false for anything touching an LLM, because the cache lived in a
gitignored directory: "reproducible on the machine that produced the numbers" is not a
property anyone else can check.

The sharded one-file-per-entry layout is right for concurrent writes and wrong for git, so
each cache exports to a single sorted JSONL and expands on import. `eval --replay` then
turns a cache miss into an error rather than a live call. The embedding cache stays out —
24,000 float32 vectors belong in a rebuild, not in a repository.

Cache keys include everything that changes the output (provider, model, prompt, system
prompt, temperature, `max_tokens`, whether structured output was requested) and nothing that
does not. The API key is not in the key and is not stored; a test asserts that a secret
placed in the environment never appears anywhere under the cache directory.

## An operational note on the cost of a free tier

The primary model runs at a **median 65 s per call** (p90 126 s, max 162 s). That is fine
for a demo and structurally wrong for an evaluation loop: 184 questions × two rewrite modes
× three strategies, done inline, is measured in hours of waiting on one connection.

Two separations follow, and both are in the code rather than in a script I ran once. Network
calls fan out across workers; *screening and scoring stay strictly serial*, because
deduplication that depends on completion order would make the eval set depend on network
timing, and interleaved retrieval would poison the per-arm latency that is itself a reported
number. And the rewrite cache is **warmed before the grid runs**, so the latency the grid
reports for the rewrite stage is a cache-read latency and is labelled as such — the real
per-call latency lives in the cached responses.

## What a verification pass is actually for

The plan said: verify the question set, re-cut the tables on the verified subset, publish
those. What happened was a **0/60 correction rate** and a uniform 5–9 point *drop* in every
metric, which is the shape of result that quietly invites a bad decision — publish the
lower numbers as "the verified ones" and let a reader infer that verification corrected
something.

It corrected nothing. With zero edits and zero rejections the verified questions are the
same questions, with the same labels, scored by the same code; the only thing that changed
was *which 60 of the 184* the mean was taken over. Two consequences got built rather than
mentioned:

**A sampling band, because the numbers needed an error bar before they needed a re-cut.**
`production-rag band` resamples n questions without replacement from a saved results file.
For n = 60 the 95% interval on recall@5 is roughly ±0.10 — wider than the gap between most
of the arms in the tables. That is uncomfortable and it is the honest frame: the *ordering*
of the arms is what survives resampling, and the differences in the third decimal are
noise being read as signal. The bootstrap is without replacement because the question is
"what if I had verified a different 60 of these", not the textbook "what if the population
resembled my sample".

**A shuffle in `verify`, because a prefix is not a sample.** The first 60 were the first 60
in file order, and they scored at or below the lower edge of that band on three of four
metric/arm combinations. That is unlucky rather than sinister, but it is unlucky in a way
that a random 60 could not have been systematically, and it made "verified numbers are
lower" ambiguous between two explanations. `verify` now walks the pending set in
seeded-shuffled order, so any partial verification is an unbiased sample of the whole set.
The bias already baked into the first 60 cannot be undone and is disclosed instead.

The uncomfortable part stays in the README: 0/60 does not distinguish "the set is clean"
from "the reviewer accepted too readily", the accept key is the cheapest one in the loop,
and 124 questions are still unreviewed.

## Scoring answers without appointing an LLM judge

The obvious way to score generated answers is to ask a bigger model whether they are good.
It was rejected, on the project's own terms rather than on taste: every number here is
supposed to be re-derivable offline by someone who does not trust me, and an LLM judge
makes the headline figure depend on a model that cannot be pinned, cannot be cached
honestly across versions, and cannot be re-run by a reader. Replacing "is the answer
correct?" with "does a model I chose say it is?" is not a measurement, it is a deferral.

So the answer eval measures only things that either happened or did not:

- **Citation validity.** The model is handed N numbered passages; a `[n]` outside `1..N` is
  a fabricated source and is caught by arithmetic. Invalid citations are kept on the answer
  object as the measurement and scrubbed only from the display copy — deleting them from
  the object would delete the evidence.
- **Uncited answers**, counted separately from invalid citations, because the fix differs: a
  dangling marker is a prompting problem, a paragraph with no marker at all is the model
  ignoring the contract. (The very first end-to-end run produced exactly this: the local
  qwen answered a `read_parquet` question fluently, correctly, and with zero citations.)
- **Grounding**, meaning the answer cited at least one chunk containing a gold evidence
  span. That is *not* "the answer is correct" — a model can cite the right passage and
  summarise it wrongly — and the README states it as the weaker claim it is.
- **Refusal recall paired with false refusal.** Never one without the other: a system that
  refuses everything scores 1.000 on the first.

Retrieval is frozen for the comparison — the rankings are replayed out of
`eval/results/retrieval.json` rather than re-retrieved per provider — so every arm sees
byte-identical context and a difference between two rows cannot be a retrieval difference.
It also means the answer eval needs no encoder, no index and no torch.

## The refusal gate that does not work, kept anyway

The appealing design is to refuse before paying for a generation: low top score means
nothing relevant was found, so say so for free. It is implemented (`--min-top-score`) and it
ships **off**, because the data says it would fire at random.

The fusion score is min-max normalised per query — that is what makes two rankings addable
— and normalisation is precisely the step that discards the absolute magnitude such a gate
needs. On the session-2 grid the six unanswerable questions have a *higher* median fused top
score (2.80) than the 121 conceptual ones (2.76). The raw pre-fusion scores do retain
something (BM25 AUC 0.81 and dense 0.84 separating answerable from unanswerable), but with
n = 6 unanswerable that is an observation, not a threshold anyone should ship behind.

Keeping the flag rather than deleting the code is deliberate: on a corpus where the gate
does work, it is one argument away, and the README carries the measurement that says it
does not work *here* rather than silently omitting an idea a reviewer would ask about.

## Deviation from the plan: no Pydantic

`CLAUDE.md` lists "Pydantic answer contract" in the stack. The answer contract is a plain
dataclass plus the existing `extract_json` repair path instead, and the dependency was not
added. The reason is consistency with a decision already made and argued for the provider
layer: this project talks to models over stdlib `urllib` because a client library covers one
of the two providers and the second still needs hand-written code. The same logic applies a
level up — the models that need the most parsing help are exactly the ones that do not
honour `response_format`, so validation cannot replace the repair path, only sit on top of
it. One dataclass and one already-tested parser beat a dependency that would duplicate half
of both.
