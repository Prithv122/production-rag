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
embed a 22,789-chunk index. The actual full-corpus build ran **40+ minutes on the first
strategy alone**.

The benchmark was wrong because the strings were wrong. It used ~50-character sentences
(~12 tokens); real chunks are ~1,000 characters (~250 tokens). Transformer cost at these
lengths is roughly linear in token count, so a 20× longer input is ~20× slower — about
8 texts/s, which is exactly what the real build shows. **Throughput per *text* is a
meaningless unit for an encoder; throughput per *token* is the one that transfers.**

Consequence: rebuilding all three strategies is ~1.5 hours, which makes the
chunking × arm grid painful to iterate on. Fixes for session 2, in order of value:

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
