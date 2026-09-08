"""Command line entry point.

Subcommands mirror the pipeline stages so each can be run and inspected on its
own::

    production-rag ingest                 # clone at pinned SHAs, normalise
    production-rag index --strategy ...   # chunk, build BM25 + dense indexes
    production-rag search "..."           # query one arm, or the hybrid
    production-rag ask "..."              # retrieve, then answer with citations
    production-rag answer-eval            # score generation across the providers
    production-rag stats                  # corpus and index shape

`ingest` and `index` need the network and (for `index`) the `embed` extra;
`search` and `stats` run entirely off the built artefacts. `ask` additionally
needs a generation provider -- an OpenRouter key or a local Ollama -- and
`answer-eval` replays frozen rankings out of `eval/results/retrieval.json`, so
it needs neither an index nor an encoder.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .answers import DEFAULT_OUT as ANSWER_RESULTS_OUT
from .answers import DEFAULT_RESULTS as ANSWER_RESULTS_IN
from .answers import DEFAULT_SUBSET, FROZEN_ARM, FROZEN_STRATEGY
from .bm25 import BM25Index
from .bm25 import build_from_chunks as build_bm25
from .chunking import CHUNKERS, USES_BREADCRUMB, Chunk, chunk_corpus
from .dense import build_from_chunks as build_dense
from .fuse import FUSIONS
from .groundtruth import HAND_WRITTEN_PATH, QUESTIONS_PATH
from .ingest import ingest_source, read_jsonl, write_jsonl
from .pipeline import ARMS, Retriever, arms_for
from .providers import LOCAL_ARMS, MODEL_ARMS, build_provider
from .rerank import DEFAULT_POOL
from .sources import DUCKDB_VERSION_ARCHIVE, SOURCES

CORPUS_DIR = Path("corpus")
REPOS_DIR = CORPUS_DIR / "repos"
DOCS_PATH = CORPUS_DIR / "normalised" / "docs.jsonl"
INDEX_DIR = Path("indexes")
EMBED_CACHE = Path(".cache/embeddings")
RERANK_CACHE = Path(".cache/rerank")
LLM_CACHE = Path(".cache/llm")


# ---------------------------------------------------------------------------
# ingest
# ---------------------------------------------------------------------------
def cmd_ingest(args: argparse.Namespace) -> int:
    sources = list(SOURCES)
    if args.include_version_archive:
        sources.append(DUCKDB_VERSION_ARCHIVE)

    docs = []
    for source in sources:
        started = time.time()
        fetched = ingest_source(source, REPOS_DIR)
        docs.extend(fetched)
        chars = sum(len(d.text) for d in fetched)
        print(
            f"{source.name:16} {len(fetched):5} docs  {chars / 1e6:6.2f} M chars  "
            f"{source.licence:<11} {source.sha[:10]}  ({time.time() - started:.1f}s)"
        )

    written = write_jsonl(docs, args.out)
    print(f"\nwrote {written:,} documents to {args.out}")
    return 0


# ---------------------------------------------------------------------------
# index
# ---------------------------------------------------------------------------
def _chunk_path(root: Path, strategy: str) -> Path:
    return root / strategy / "chunks.jsonl"


def _write_chunks(chunks: list[Chunk], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for chunk in chunks:
            handle.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")


def load_chunks(root: Path, strategy: str) -> list[Chunk]:
    with _chunk_path(root, strategy).open(encoding="utf-8") as handle:
        return [Chunk.from_dict(json.loads(line)) for line in handle if line.strip()]


def cmd_index(args: argparse.Namespace) -> int:
    docs = read_jsonl(args.docs)
    strategies = args.strategy or list(CHUNKERS)

    for strategy in strategies:
        target = args.out / strategy
        started = time.time()
        chunks = chunk_corpus(docs, strategy, max_chars=args.max_chars)
        _write_chunks(chunks, _chunk_path(args.out, strategy))
        breadcrumb = USES_BREADCRUMB[strategy]

        bm25 = build_bm25(chunks, with_breadcrumb=breadcrumb)
        bm25.save(target / "bm25")
        lexical_done = time.time()

        note = ""
        if not args.no_dense:
            from .cache import CachedEmbedder
            from .dense import SentenceTransformerEmbedder

            embedder = SentenceTransformerEmbedder(args.model)
            cached = CachedEmbedder(embedder, args.embed_cache) if args.embed_cache else embedder
            dense = build_dense(chunks, cached, with_breadcrumb=breadcrumb)
            dense.save(target / "dense")
            if isinstance(cached, CachedEmbedder):
                note = f"  (cache {cached.hits:,} hit / {cached.misses:,} miss)"
        else:
            note = "  (dense skipped)"

        print(
            f"{strategy:12} {len(chunks):6,} chunks  "
            f"bm25 {lexical_done - started:5.1f}s  "
            f"total {time.time() - started:6.1f}s{note}"
        )
    return 0


# ---------------------------------------------------------------------------
# search
# ---------------------------------------------------------------------------
def cmd_search(args: argparse.Namespace) -> int:
    spec = ARMS[args.arm]
    rewriter = _provider(args) if spec.needs_llm else None
    retriever = Retriever.load(
        args.indexes,
        args.strategy,
        embedder=_embedder(args) if spec.semantic else None,
        with_dense=spec.semantic,
    )
    result = retriever.retrieve(
        args.query,
        arm=args.arm,
        k=args.k,
        pool=args.pool,
        reranker=_reranker(args) if spec.rerank else None,
        rewriter=rewriter,
    )
    if len(result.queries) > 1:
        print("queries: " + " | ".join(result.queries) + "\n")
    print(retriever.format(result))
    stages = "  ".join(f"{name} {seconds * 1000:.0f}ms" for name, seconds in result.stage_s.items())
    print(f"\n{result.latency_s * 1000:.0f}ms total   {stages}")
    return 0


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------
def cmd_stats(args: argparse.Namespace) -> int:
    docs = read_jsonl(args.docs)
    by_tool: dict[str, int] = {}
    for doc in docs:
        by_tool[doc.tool] = by_tool.get(doc.tool, 0) + 1

    print(f"documents      {len(docs):,}")
    print(f"characters     {sum(len(d.text) for d in docs):,}")
    for tool, count in sorted(by_tool.items()):
        print(f"  {tool:12} {count:5,}")

    for strategy in CHUNKERS:
        path = _chunk_path(args.indexes, strategy)
        if not path.exists():
            continue
        chunks = load_chunks(args.indexes, strategy)
        lengths = sorted(len(c.text) for c in chunks)
        mean = sum(lengths) / len(lengths)
        print(
            f"{strategy:14} {len(chunks):6,} chunks  "
            f"mean {mean:5.0f}  median {lengths[len(lengths) // 2]:5}  max {lengths[-1]:5}"
        )
    return 0


# ---------------------------------------------------------------------------
# shared components
# ---------------------------------------------------------------------------
def _embedder(args: argparse.Namespace):
    """The real encoder, wrapped in the content-hash cache."""
    from .cache import CachedEmbedder
    from .dense import DEFAULT_MODEL, SentenceTransformerEmbedder

    inner = SentenceTransformerEmbedder(getattr(args, "model", None) or DEFAULT_MODEL)
    cache = getattr(args, "embed_cache", None)
    return CachedEmbedder(inner, cache) if cache else inner


def _provider(args: argparse.Namespace):
    """The generation stack for this run, with the cache the flags point at."""
    return build_provider(args.model_arm, cache_dir=args.llm_cache, offline=args.replay)


def _reranker(args: argparse.Namespace):
    from .rerank import CachedReranker, CrossEncoderReranker

    return CachedReranker(
        CrossEncoderReranker(args.rerank_model),
        args.rerank_cache,
        offline=getattr(args, "replay", False),
    )


# ---------------------------------------------------------------------------
# propose -- LLM-generated question candidates
# ---------------------------------------------------------------------------
def cmd_propose(args: argparse.Namespace) -> int:
    from .groundtruth import (
        ProposalStats,
        assign_categories,
        assign_ids,
        document_frequencies,
        load_hand_written,
        propose_from_passage,
        sample_passages,
        save_questions,
        screen_proposals,
    )

    documents = {d.doc_id: d for d in read_jsonl(args.docs)}
    chunks = load_chunks(args.indexes, args.strategy)
    passages = sample_passages(chunks, n=args.passages, seed=args.seed)

    spec = MODEL_ARMS[args.model_arm]
    provider = _provider(args)
    stats = ProposalStats(passages=len(passages))
    seen: set[str] = set()
    accepted = []

    # Free-tier generation runs at 30-90 s per call, so the network step is
    # parallelised -- but the *screening* step is not. Screening mutates a
    # shared `seen` set, and deduplication that depends on completion order
    # would make the eval set depend on network timing. So calls fan out and
    # results are folded back in passage order.
    def _call(passage, max_tokens):
        document = documents.get(passage.doc_id)
        if document is None:  # pragma: no cover - only if docs and chunks disagree
            return passage, None, [], ""
        proposals, error = propose_from_passage(
            passage,
            document.text,
            provider,
            n=args.per_passage,
            structured=spec["structured"],
            max_tokens=max_tokens,
        )
        return passage, document, proposals, error

    def _pass(batch, max_tokens, label):
        failed = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for i, (passage, document, proposals, error) in enumerate(
                pool.map(lambda p: _call(p, max_tokens), batch), start=1
            ):
                if document is None:
                    continue
                if error:
                    failed.append(passage)
                    stats.errors.append(error)
                accepted.extend(
                    screen_proposals(
                        proposals, passage, document, model=spec["model"], seen=seen, stats=stats
                    )
                )
                if i % 10 == 0 or i == len(batch):
                    print(
                        f"  {label} {i:4}/{len(batch)} · {stats.accepted:4} accepted · "
                        f"{len(failed)} failed · {stats.rejected_no_quote} bad quote · "
                        f"{stats.rejected_duplicate} dup",
                        flush=True,
                    )
        return failed

    failed = _pass(passages, args.max_tokens, "pass 1")

    # A reasoning model can spend its entire token budget on a preamble and stop
    # with finish=length before emitting a character of JSON -- measured at
    # roughly a third of calls at 1600 tokens. Retrying only the failures with a
    # larger budget costs one extra call per failure instead of re-running the
    # whole set, and the successful responses stay served from cache.
    if failed and args.retry_multiplier > 1:
        budget = args.max_tokens * args.retry_multiplier
        print(f"\nretrying {len(failed)} failed passages at {budget} tokens")
        still_failed = _pass(failed, args.max_tokens * args.retry_multiplier, "pass 2")
        stats.rejected_unparseable = len(still_failed)
        stats.recovered_on_retry = len(failed) - len(still_failed)
    else:
        stats.rejected_unparseable = len(failed)

    # The hand-written cross-tool and unanswerable questions are merged in from
    # their own file. They are the part of the set no passage-sampling loop can
    # produce, so they are authored separately and never regenerated.
    hand = load_hand_written(args.hand, documents) if args.hand and args.hand.exists() else []
    bm25 = BM25Index.load(args.indexes / args.strategy / "bm25")
    df = document_frequencies(bm25)
    questions = assign_ids(assign_categories([*hand, *accepted], documents, df))
    save_questions(questions, args.out)

    print(f"\n{json.dumps(stats.as_dict(), indent=2)}")
    disagreed = sum(
        1 for q in questions if q.proposed_category and q.proposed_category != q.category
    )
    print(f"\nwrote {len(questions)} questions to {args.out}")
    print(f"category disagreement (model's label vs computed): {disagreed}/{len(questions)}")
    counts: dict[str, int] = {}
    for question in questions:
        counts[question.category] = counts.get(question.category, 0) + 1
    print(f"categories: {counts}")
    return 0


# ---------------------------------------------------------------------------
# verify -- the human pass
# ---------------------------------------------------------------------------
def cmd_verify(args: argparse.Namespace) -> int:
    """Walk the question set and record a human judgement on each.

    This is the step that makes the eval set ground truth rather than a model's
    opinion of itself. It is deliberately a terminal loop over the *unverified*
    questions only, so it can be done in several sittings, and it saves after
    every decision -- an interrupted session loses nothing.

    **The pending set is walked in shuffled order.** Verification happens in
    sittings, so at any moment the verified subset is a *prefix* of whatever
    order this loop uses -- and a prefix of file order is not a random sample.
    The first 60 questions verified in this project scored at the 2nd-14th
    percentile of the bootstrap distribution over 60-question subsets, which
    made "the verified numbers are lower" ambiguous between a real correction
    and an unlucky slice. A seeded shuffle makes any partial verification an
    unbiased sample of the set, so the two explanations stop being confounded.
    """
    import random
    from dataclasses import replace

    from .groundtruth import load_questions, save_questions

    documents = {d.doc_id: d for d in read_jsonl(args.docs)}
    questions = load_questions(args.path)
    pending = [
        (i, q)
        for i, q in enumerate(questions)
        if q.verified == "unverified" and (not args.category or q.category == args.category)
    ]
    if args.shuffle:
        random.Random(args.seed).shuffle(pending)
    if not pending:
        print("nothing left to verify")
        return 0

    print(
        f"{len(pending)} unverified of {len(questions)}. "
        "[a]ccept  [r]eject  [e]dit question  [s]kip  [q]uit\n"
    )
    done = 0
    for i, question in pending[: args.limit]:
        evidence = question.evidence[0] if question.evidence else None
        document = documents.get(evidence.doc_id) if evidence else None
        print("=" * 78)
        print(f"{question.qid}   [{question.category}]   {question.tools}")
        print(f"Q: {question.text}")
        if document and evidence:
            context_start = max(0, evidence.start - 200)
            context = document.text[context_start : evidence.end + 200]
            print(f"\n{document.url}")
            print(f"--- evidence ({evidence.end - evidence.start} chars) ---")
            print(context.replace(evidence.quote, f">>>{evidence.quote}<<<").strip()[:1400])
        else:
            print("\n(no evidence -- this question is labelled unanswerable)")

        try:
            answer = input("\n[a/r/e/s/q] > ").strip().lower()
        except EOFError:
            print("\nnot a terminal; run `verify` from an interactive shell")
            return 1

        if answer.startswith("q"):
            break
        if answer.startswith("s"):
            continue
        if answer.startswith("e"):
            edited = input("new question text > ").strip()
            if edited:
                questions[i] = replace(
                    questions[i], text=edited, verified="edited", verifier=args.verifier
                )
        elif answer.startswith("r"):
            reason = input("why? > ").strip()
            questions[i] = replace(
                questions[i], verified="rejected", verifier=args.verifier, notes=reason
            )
        else:
            questions[i] = replace(questions[i], verified="accepted", verifier=args.verifier)
        done += 1
        save_questions(questions, args.path)

    counts: dict[str, int] = {}
    for question in questions:
        counts[question.verified] = counts.get(question.verified, 0) + 1
    reviewed = sum(counts.get(k, 0) for k in ("accepted", "edited", "rejected"))
    corrected = counts.get("edited", 0) + counts.get("rejected", 0)
    print(f"\n{done} decided this session. Totals: {counts}")
    if reviewed:
        print(f"correction rate: {corrected}/{reviewed} = {corrected / reviewed:.1%}")
    return 0


# ---------------------------------------------------------------------------
# ask -- retrieve, then answer with citations
# ---------------------------------------------------------------------------
def cmd_ask(args: argparse.Namespace) -> int:
    from .generate import format_answer, generate

    spec = ARMS[args.arm]
    provider = _provider(args)
    retriever = Retriever.load(
        args.indexes,
        args.strategy,
        embedder=_embedder(args) if spec.semantic else None,
        with_dense=spec.semantic,
    )
    result = retriever.retrieve(
        args.question,
        arm=args.arm,
        k=args.k,
        pool=args.pool,
        reranker=_reranker(args) if spec.rerank else None,
        rewriter=provider if spec.needs_llm else None,
    )
    top_score = result.ranked[0][1] if result.ranked else None
    answer = generate(
        args.question,
        result.chunk_ids,
        retriever.chunks,
        provider,
        top_score=top_score,
        min_top_score=args.min_top_score,
        sentences=args.sentences,
    )
    print(format_answer(answer))
    if args.show_context:
        print("\nRetrieved\n" + retriever.format(result))
    return 0


# ---------------------------------------------------------------------------
# answer-eval -- the generation grid over the provider arms
# ---------------------------------------------------------------------------
def cmd_answer_eval(args: argparse.Namespace) -> int:
    from .answers import (
        DEFAULT_SEED,
        format_by_category,
        format_table,
        load_frozen_retrievals,
        run_grid,
        save,
        select_questions,
    )
    from .groundtruth import index_chunks_by_doc, load_questions

    questions = [q for q in load_questions(args.questions) if q.verified != "rejected"]
    if args.verified_only:
        questions = [q for q in questions if q.verified in ("accepted", "edited")]
    rankings = load_frozen_retrievals(
        args.results, arm=args.retrieval_arm, strategy=args.strategy, k=args.k
    )
    questions = [q for q in questions if q.qid in rankings or not q.is_answerable]
    if args.subset:
        questions = select_questions(questions, n=args.subset, seed=args.seed or DEFAULT_SEED)
    if not questions:
        print("no questions to answer")
        return 1

    chunks = {c.chunk_id: c for c in load_chunks(args.indexes, args.strategy)}
    by_doc = index_chunks_by_doc(chunks.values())
    # `fallback=False`, unlike everywhere else. Graceful degradation to a local
    # model is a feature of `ask` and a liability here: it turns "arm X failed"
    # into "arm X quietly answered as arm Y", and a comparison between models
    # that can silently substitute one for another is not a comparison. It cost
    # a 65-minute run to learn that -- every OpenRouter call in it fell through
    # to Ollama, and the table showed four identical rows. Fail loudly instead;
    # `fell_back` in the summary is the second line of defence.
    providers = {
        arm: build_provider(arm, cache_dir=args.llm_cache, offline=args.replay, fallback=False)
        for arm in (args.model_arm_list or list(LOCAL_ARMS))
    }

    total = len(questions) * len(providers)
    done = 0

    def progress(row) -> None:
        nonlocal done
        done += 1
        if done % 10 == 0 or done == total:
            print(f"  {done}/{total}", flush=True)

    print(f"{len(questions)} questions x {len(providers)} arms, frozen on {args.retrieval_arm}")
    rows, summaries, config = run_grid(
        questions,
        rankings,
        chunks,
        providers,
        by_doc=by_doc,
        workers=args.workers,
        k=args.k,
        progress=progress,
    )
    save(args.out, rows, summaries, config)
    print("\n" + format_table(summaries))
    print("\ngrounded, by category\n" + format_by_category(rows))
    print(f"\nwrote {args.out}")
    return 0


# ---------------------------------------------------------------------------
# band -- how much of a difference between two arms is the question set?
# ---------------------------------------------------------------------------
def cmd_band(args: argparse.Namespace) -> int:
    from .evaluate import subset_bands

    print(
        subset_bands(
            args.results,
            n=args.n,
            metric=args.metric,
            strategy=args.strategy,
            trials=args.trials,
            seed=args.seed,
        )
    )
    return 0


# ---------------------------------------------------------------------------
# cache -- bundle the LLM and rerank caches so replay works from a clean clone
# ---------------------------------------------------------------------------
LLM_BUNDLE = Path("eval/cache/llm.jsonl")
RERANK_BUNDLE = Path("eval/cache/rerank.jsonl")

CACHES = {
    "llm": (LLM_CACHE, LLM_BUNDLE),
    "rerank": (RERANK_CACHE, RERANK_BUNDLE),
}


def _audit_bundle(bundle: Path) -> int:
    """Report cache entries answered by a different model than was requested.

    This exists because the project shipped a table naming one model that had in
    fact been produced mostly by another. `FallbackProvider` degrades to a local
    model when the primary fails, which is the right behaviour for a demo and a
    silent falsification of provenance in an evaluation. The cache records both
    the requested model (in the key) and the answering model (in the value), so
    the discrepancy was always on disk -- nothing was looking at it.
    """
    if not bundle.exists():
        print(f"{bundle}: no bundle")
        return 0
    entries = [json.loads(line) for line in bundle.read_text(encoding="utf-8").splitlines() if line]
    mismatched: dict[tuple[str, str], int] = {}
    for entry in entries:
        requested = entry.get("key", {}).get("model", "")
        answered = entry.get("value", {}).get("model", "")
        if requested and answered and requested != answered:
            mismatched[(requested, answered)] = mismatched.get((requested, answered), 0) + 1

    total = sum(mismatched.values())
    print(f"{bundle}: {len(entries):,} entries, {total:,} answered by a different model")
    for (requested, answered), count in sorted(mismatched.items(), key=lambda kv: -kv[1]):
        print(f"    {count:5,}  requested {requested}  ->  answered by {answered}")
    return total


def cmd_cache(args: argparse.Namespace) -> int:
    """Export the committed caches, or expand them after a clone.

    The embedding cache is deliberately absent: it is 24,000 float32 vectors and
    belongs in a rebuild, not in git. The LLM and rerank caches are small,
    and without them `eval --replay` is only reproducible on the machine that
    produced the numbers -- which is not reproducible.
    """
    from .cache import JsonCache

    if args.action == "audit":
        return 0 if not sum(_audit_bundle(bundle) for _, bundle in CACHES.values()) else 1

    for name, (root, bundle) in CACHES.items():
        cache = JsonCache(root)
        if args.action == "export":
            if not root.exists():
                print(f"{name:8} no local cache at {root}")
                continue
            print(f"{name:8} exported {cache.export_jsonl(bundle):,} entries -> {bundle}")
        else:
            if not bundle.exists():
                print(f"{name:8} no bundle at {bundle}")
                continue
            print(f"{name:8} imported {cache.import_jsonl(bundle):,} new entries -> {root}")
    return 0


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def cmd_eval(args: argparse.Namespace) -> int:
    from .evaluate import run_grid, save_results, summarise
    from .groundtruth import load_questions

    questions = [q for q in load_questions(args.questions) if q.verified != "rejected"]
    if args.verified_only:
        questions = [q for q in questions if q.verified in ("accepted", "edited")]
    if not questions:
        print("no questions to evaluate")
        return 1

    arms = arms_for(args.arm, offline=args.offline)
    strategies = args.strategy or sorted(CHUNKERS)
    needs_rerank = any(ARMS[a].rerank for a in arms)
    needs_llm = any(ARMS[a].needs_llm for a in arms)

    rewriter = _provider(args) if needs_llm else None
    if rewriter is not None and not args.replay:
        _warm_rewrites(questions, rewriter, workers=args.workers)

    results = run_grid(
        questions,
        index_dir=args.indexes,
        strategies=strategies,
        arms=arms,
        embedder_factory=(lambda: _embedder(args)),
        reranker=_reranker(args) if needs_rerank else None,
        rewriter=rewriter,
        k=args.k,
        pool=args.pool,
    )
    save_results(results, args.out)
    print("\n" + summarise(results))
    print(f"\nwrote {args.out}")
    return 0


def _warm_rewrites(questions, provider, *, workers: int = 6) -> None:
    """Populate the rewrite cache concurrently before the grid runs.

    The grid itself is strictly sequential -- it has to be, because per-arm
    latency is one of the reported numbers and interleaving would poison it. But
    a free-tier rewrite call takes 30-90 seconds, and the same question is
    rewritten identically for every strategy and both rewrite modes, so doing
    them inline would spend hours re-deriving cache misses one at a time.

    Warming first separates the two concerns: the network cost is paid once, in
    parallel, and the measured run then reads from disk. The rewrite latency
    reported afterwards is therefore a *cache* latency and is labelled as such;
    the real per-call latency is in the cached responses themselves.
    """
    from .rewrite import rewrite_query

    texts = sorted({q.text for q in questions})
    print(f"warming the rewrite cache for {len(texts)} questions ({workers} workers)...")
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(lambda t: rewrite_query(t, provider), texts):
            done += 1
            if done % 20 == 0 or done == len(texts):
                print(f"  {done}/{len(texts)}", flush=True)
            if not result.parsed and result.error:
                print(f"  rewrite failed: {result.error[:120]}", flush=True)


# ---------------------------------------------------------------------------
# wiring
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="production-rag", description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest = subparsers.add_parser("ingest", help="fetch and normalise the corpus")
    ingest.add_argument("--out", type=Path, default=DOCS_PATH)
    ingest.add_argument(
        "--include-version-archive",
        action="store_true",
        help="also ingest DuckDB's superseded version trees (ablation only)",
    )
    ingest.set_defaults(func=cmd_ingest)

    index = subparsers.add_parser("index", help="chunk and build retrieval indexes")
    index.add_argument("--docs", type=Path, default=DOCS_PATH)
    index.add_argument("--out", type=Path, default=INDEX_DIR)
    index.add_argument("--strategy", action="append", choices=sorted(CHUNKERS))
    index.add_argument("--max-chars", type=int, default=1200)
    index.add_argument("--model", default=None)
    index.add_argument(
        "--embed-cache",
        type=Path,
        default=EMBED_CACHE,
        help="content-hash embedding cache; strategies share identical chunk text",
    )
    index.add_argument(
        "--no-dense",
        action="store_true",
        help="build only the lexical index (skips the embedding model)",
    )
    index.set_defaults(func=cmd_index)

    search = subparsers.add_parser("search", help="query the built indexes")
    search.add_argument("query")
    search.add_argument("--indexes", type=Path, default=INDEX_DIR)
    search.add_argument("--strategy", choices=sorted(CHUNKERS), default="heading")
    search.add_argument("--arm", choices=sorted(ARMS), default="hybrid")
    search.add_argument("--fusion", choices=sorted(FUSIONS), default="rrf")
    search.add_argument("-k", type=int, default=10)
    search.add_argument("--pool", type=int, default=DEFAULT_POOL)
    _add_component_args(search)
    search.set_defaults(func=cmd_search)

    ask = subparsers.add_parser("ask", help="retrieve, then answer with citations")
    ask.add_argument("question")
    ask.add_argument("--indexes", type=Path, default=INDEX_DIR)
    ask.add_argument("--strategy", choices=sorted(CHUNKERS), default="heading")
    ask.add_argument("--arm", choices=sorted(ARMS), default=FROZEN_ARM)
    ask.add_argument("-k", type=int, default=10)
    ask.add_argument("--pool", type=int, default=DEFAULT_POOL)
    ask.add_argument("--sentences", type=int, default=4)
    ask.add_argument(
        "--min-top-score",
        type=float,
        default=0.0,
        help="refuse before generating if the top retrieval score is below this. "
        "0 disables it, which is the default -- see generate.py for the measurement "
        "that says the fused score does not separate answerable from unanswerable here",
    )
    ask.add_argument("--show-context", action="store_true")
    _add_component_args(ask)
    ask.set_defaults(func=cmd_ask)

    answer_eval = subparsers.add_parser(
        "answer-eval", help="score generated answers across the provider arms"
    )
    answer_eval.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    answer_eval.add_argument("--indexes", type=Path, default=INDEX_DIR)
    answer_eval.add_argument("--results", type=Path, default=ANSWER_RESULTS_IN)
    answer_eval.add_argument("--out", type=Path, default=ANSWER_RESULTS_OUT)
    answer_eval.add_argument("--strategy", choices=sorted(CHUNKERS), default=FROZEN_STRATEGY)
    answer_eval.add_argument("--retrieval-arm", choices=sorted(ARMS), default=FROZEN_ARM)
    answer_eval.add_argument("-k", type=int, default=10)
    answer_eval.add_argument(
        "--subset",
        type=int,
        default=DEFAULT_SUBSET,
        help="stratified sample size; every unanswerable and cross_tool question is "
        "kept regardless. 0 runs the whole set",
    )
    answer_eval.add_argument("--seed", type=int, default=0)
    answer_eval.add_argument("--workers", type=int, default=4)
    answer_eval.add_argument("--verified-only", action="store_true")
    answer_eval.add_argument(
        "--model-arm-list",
        nargs="+",
        choices=sorted(MODEL_ARMS),
        help="which generation arms to run. Defaults to the local ones, which need no "
        "key and no quota; the hosted arms are selectable but see the README on what a "
        "free OpenRouter account can actually reach",
    )
    _add_component_args(answer_eval)
    answer_eval.set_defaults(func=cmd_answer_eval)

    stats = subparsers.add_parser("stats", help="corpus and index shape")
    stats.add_argument("--docs", type=Path, default=DOCS_PATH)
    stats.add_argument("--indexes", type=Path, default=INDEX_DIR)
    stats.set_defaults(func=cmd_stats)

    propose = subparsers.add_parser("propose", help="generate candidate eval questions")
    propose.add_argument("--docs", type=Path, default=DOCS_PATH)
    propose.add_argument("--indexes", type=Path, default=INDEX_DIR)
    propose.add_argument("--strategy", choices=sorted(CHUNKERS), default="heading")
    propose.add_argument("--out", type=Path, default=QUESTIONS_PATH)
    propose.add_argument("--passages", type=int, default=100)
    propose.add_argument("--per-passage", type=int, default=2)
    propose.add_argument("--seed", type=int, default=20240908)
    propose.add_argument(
        "--hand",
        type=Path,
        default=HAND_WRITTEN_PATH,
        help="human-authored cross-tool and unanswerable questions, merged in",
    )
    propose.add_argument(
        "--max-tokens",
        type=int,
        default=1600,
        help="generous: a reasoning model can spend a 700-token budget entirely on "
        "its preamble and finish with reason=length before emitting any JSON",
    )
    propose.add_argument(
        "--retry-multiplier",
        type=int,
        default=3,
        help="retry failed passages once at this multiple of --max-tokens (0 to disable)",
    )
    propose.add_argument(
        "--workers",
        type=int,
        default=6,
        help="parallel API calls; screening stays serial so the set is deterministic",
    )
    _add_component_args(propose)
    propose.set_defaults(func=cmd_propose)

    verify = subparsers.add_parser("verify", help="human-verify the question set")
    verify.add_argument("--path", type=Path, default=QUESTIONS_PATH)
    verify.add_argument("--docs", type=Path, default=DOCS_PATH)
    verify.add_argument("--limit", type=int, default=60)
    verify.add_argument("--category", default="")
    verify.add_argument("--verifier", default="author")
    verify.add_argument(
        "--no-shuffle",
        dest="shuffle",
        action="store_false",
        help="walk the pending questions in file order. Off by default: a partial "
        "verification in file order is a prefix, not a sample",
    )
    verify.add_argument("--seed", type=int, default=20260908)
    verify.set_defaults(func=cmd_verify)

    evaluate = subparsers.add_parser("eval", help="run the arm x strategy grid")
    evaluate.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    evaluate.add_argument("--indexes", type=Path, default=INDEX_DIR)
    evaluate.add_argument("--out", type=Path, default=Path("eval/results/retrieval.json"))
    evaluate.add_argument("--strategy", action="append", choices=sorted(CHUNKERS))
    evaluate.add_argument("--arm", action="append", choices=sorted(ARMS))
    evaluate.add_argument("-k", type=int, default=10)
    evaluate.add_argument("--pool", type=int, default=DEFAULT_POOL)
    evaluate.add_argument(
        "--offline",
        action="store_true",
        help="only the arms that need no API key (the default arm selection)",
    )
    evaluate.add_argument("--workers", type=int, default=6, help="parallelism for cache warming")
    evaluate.add_argument(
        "--verified-only",
        action="store_true",
        help="restrict to human-verified questions -- the number the README quotes",
    )
    _add_component_args(evaluate)
    evaluate.set_defaults(func=cmd_eval)

    band = subparsers.add_parser(
        "band", help="sampling band for a subset of the question set, from saved results"
    )
    band.add_argument("--results", type=Path, default=Path("eval/results/retrieval.json"))
    band.add_argument("-n", type=int, default=60, help="subset size to resample")
    band.add_argument("--metric", default="recall@5")
    band.add_argument("--strategy", choices=sorted(CHUNKERS), default="heading")
    band.add_argument("--trials", type=int, default=20_000)
    band.add_argument("--seed", type=int, default=20260908)
    band.set_defaults(func=cmd_band)

    cache = subparsers.add_parser("cache", help="bundle or restore the replay caches")
    cache.add_argument(
        "action",
        choices=["export", "import", "audit"],
        help="audit reports entries whose answering model differs from the requested one "
        "-- the silent-fallback check; exits non-zero if any are found",
    )
    cache.set_defaults(func=cmd_cache)

    return parser


def _add_component_args(parser: argparse.ArgumentParser) -> None:
    """Flags shared by every subcommand that builds a model-backed component."""
    from .rerank import DEFAULT_RERANK_MODEL

    parser.add_argument("--model", default=None, help="embedding model")
    parser.add_argument("--embed-cache", type=Path, default=EMBED_CACHE)
    parser.add_argument("--rerank-model", default=DEFAULT_RERANK_MODEL)
    parser.add_argument("--rerank-cache", type=Path, default=RERANK_CACHE)
    parser.add_argument("--llm-cache", type=Path, default=LLM_CACHE)
    parser.add_argument("--model-arm", choices=sorted(MODEL_ARMS), default="nemotron-super")
    parser.add_argument(
        "--replay",
        action="store_true",
        help="never call out; every LLM and rerank result must already be cached",
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if getattr(args, "model", None) is None and hasattr(args, "model"):
        from .dense import DEFAULT_MODEL

        args.model = DEFAULT_MODEL
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
