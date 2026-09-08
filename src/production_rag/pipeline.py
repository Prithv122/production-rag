"""Composing the retrieval arms.

Session 1 built the pieces -- BM25, dense, fusion -- and wired them together
inline in the `search` subcommand. That was fine for one query and wrong for an
evaluation: the harness has to run seven arms across three chunking strategies
over the same questions, and every one of those runs must be the *same code
path* the CLI and the demo use. An eval that measures a re-implementation of the
pipeline measures the re-implementation.

So the composition lives here, once, as data:

======================  =========================================================
``bm25``                Lexical only.
``dense``               Vector only.
``hybrid``              Both, fused by reciprocal rank.
``hybrid_score``        Both, fused by normalised score. Tests whether keeping
                        magnitude information beats discarding it for rank.
``hybrid_weighted``     RRF again, but the lexical arm counts double. Equal
``hybrid_score_weighted`` weighting gives a weak arm as many votes as a strong
                        one; these two ask what that costs.
``hybrid_rerank``       Hybrid, then a cross-encoder over the pool.
``rerank_rewrite``      ...plus LLM query expansion (original + variants).
``rerank_rewrite_only`` ...but retrieving with the rewrite *instead of* the
                        original. The arm that should expose the paraphrase
                        failure described in :mod:`production_rag.rewrite`.
======================  =========================================================

Two things are deliberate in the shape of :func:`retrieve`.

**The pool is separate from k.** Every arm gathers `pool` candidates and returns
`k`. Reranking a top-10 can only reorder ten chunks; the reranker earns its cost
by promoting something that was at rank 34. Reporting recall@10 off a pool of 10
would flatter the cheap arms and hide what reranking does.

**Every stage is injected.** The embedder, the reranker and the rewriter are
protocols passed in, never constructed here. That is what lets the fast test
suite exercise all seven arms with no torch and no network, and it is why the
eval can freeze retrieval and vary only generation later.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

from .bm25 import BM25Index
from .chunking import USES_BREADCRUMB, Chunk
from .dense import DenseIndex, Embedder
from .fuse import fuse
from .rerank import DEFAULT_POOL, Reranker
from .rerank import rerank as rerank_candidates
from .rewrite import RewriteResult, rewrite_query


@dataclass(frozen=True)
class ArmSpec:
    """One retrieval configuration."""

    lexical: bool = True
    semantic: bool = True
    fusion: str = "rrf"
    rerank: bool = False
    rewrite: str = ""
    """`""`, `"expand"` or `"replace"` -- see :mod:`production_rag.rewrite`."""

    weights: tuple[float, ...] = ()
    """Per-arm fusion weights, lexical first. Empty means equal weighting.

    Equal weighting is the textbook default and it is not obviously right: RRF
    gives a weak arm exactly as many votes as a strong one, so a dense arm that
    is 20 points worse still gets to dilute the lexical ranking. Whether that
    costs anything is measurable, so it is an arm rather than an assumption."""

    @property
    def needs_dense(self) -> bool:
        return self.semantic

    @property
    def needs_llm(self) -> bool:
        return bool(self.rewrite)


ARMS: dict[str, ArmSpec] = {
    "bm25": ArmSpec(lexical=True, semantic=False),
    "dense": ArmSpec(lexical=False, semantic=True),
    "hybrid": ArmSpec(fusion="rrf"),
    "hybrid_score": ArmSpec(fusion="score"),
    "hybrid_weighted": ArmSpec(fusion="rrf", weights=(2.0, 1.0)),
    "hybrid_score_weighted": ArmSpec(fusion="score", weights=(2.0, 1.0)),
    "hybrid_rerank": ArmSpec(fusion="rrf", rerank=True),
    "rerank_rewrite": ArmSpec(fusion="rrf", rerank=True, rewrite="expand"),
    "rerank_rewrite_only": ArmSpec(fusion="rrf", rerank=True, rewrite="replace"),
}

#: Arms that need neither an API key nor a network connection. Everything in
#: this list reproduces from a clean clone with the indexes rebuilt.
OFFLINE_ARMS = (
    "bm25",
    "dense",
    "hybrid",
    "hybrid_score",
    "hybrid_weighted",
    "hybrid_score_weighted",
    "hybrid_rerank",
)


@dataclass
class RetrievalResult:
    """A ranking plus enough provenance to explain how it was produced."""

    ranked: list[tuple[str, float]]
    arm: str
    queries: list[str] = field(default_factory=list)
    """Every query string actually issued. Length > 1 means expansion ran."""

    rewrite: RewriteResult | None = None
    latency_s: float = 0.0
    stage_s: dict[str, float] = field(default_factory=dict)
    pool_size: int = 0

    @property
    def chunk_ids(self) -> list[str]:
        return [chunk_id for chunk_id, _ in self.ranked]


class Retriever:
    """Holds the loaded indexes for one chunking strategy."""

    def __init__(
        self,
        chunks: dict[str, Chunk],
        bm25: BM25Index,
        dense: DenseIndex | None = None,
        *,
        embedder: Embedder | None = None,
        with_breadcrumb: bool = False,
        strategy: str = "",
    ) -> None:
        self.chunks = chunks
        self.bm25 = bm25
        self.dense = dense
        self.embedder = embedder
        self.with_breadcrumb = with_breadcrumb
        self.strategy = strategy

    @classmethod
    def load(
        cls,
        index_dir: Path,
        strategy: str,
        *,
        embedder: Embedder | None = None,
        with_dense: bool = True,
        chunks: dict[str, Chunk] | None = None,
    ) -> Retriever:
        from .cli import load_chunks  # local import: cli imports this module

        root = Path(index_dir) / strategy
        loaded = (
            chunks
            if chunks is not None
            else {c.chunk_id: c for c in load_chunks(Path(index_dir), strategy)}
        )
        dense = DenseIndex.load(root / "dense") if with_dense else None
        return cls(
            loaded,
            BM25Index.load(root / "bm25"),
            dense,
            embedder=embedder,
            with_breadcrumb=USES_BREADCRUMB.get(strategy, False),
            strategy=strategy,
        )

    # -- the pipeline -----------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        arm: str = "hybrid",
        k: int = 10,
        pool: int = DEFAULT_POOL,
        reranker: Reranker | None = None,
        rewriter=None,
        rewrite_n: int = 2,
    ) -> RetrievalResult:
        try:
            spec = ARMS[arm]
        except KeyError:
            raise ValueError(f"unknown arm {arm!r}; expected one of {sorted(ARMS)}") from None
        if spec.needs_dense and self.dense is None:
            raise ValueError(f"arm {arm!r} needs a dense index, which was not loaded")
        if spec.needs_llm and rewriter is None:
            raise ValueError(f"arm {arm!r} needs an LLM provider for query rewriting")
        if spec.rerank and reranker is None:
            raise ValueError(f"arm {arm!r} needs a reranker")

        started = time.perf_counter()
        stage: dict[str, float] = {}
        pool = max(pool, k)

        rewrite: RewriteResult | None = None
        queries = [query]
        if spec.rewrite:
            mark = time.perf_counter()
            rewrite = rewrite_query(query, rewriter, mode=spec.rewrite, n=rewrite_n)
            queries = list(rewrite.queries)
            stage["rewrite"] = time.perf_counter() - mark

        mark = time.perf_counter()
        rankings: list[list[tuple[str, float]]] = []
        for sub_query in queries:
            if spec.lexical:
                rankings.append(self.bm25.search(sub_query, k=pool))
            if spec.semantic:
                assert self.dense is not None and self.embedder is not None
                rankings.append(self.dense.search(sub_query, self.embedder, k=pool))
        stage["first_stage"] = time.perf_counter() - mark

        if len(rankings) == 1:
            candidates = list(rankings[0])
        else:
            # `k=pool` here, not `k`: fusion feeds the reranker, and truncating
            # to 10 before reranking would throw away exactly the candidates the
            # reranker exists to rescue.
            weights = list(spec.weights) * (len(rankings) // max(len(spec.weights), 1)) or None
            candidates = fuse(rankings, spec.fusion, k=pool, weights=weights)

        pool_size = len(candidates)
        if spec.rerank:
            mark = time.perf_counter()
            ranked = rerank_candidates(
                query,
                candidates,
                self.chunks,
                reranker,
                k=k,
                with_breadcrumb=self.with_breadcrumb,
            )
            stage["rerank"] = time.perf_counter() - mark
        else:
            ranked = candidates[:k]

        return RetrievalResult(
            ranked=ranked,
            arm=arm,
            queries=queries,
            rewrite=rewrite,
            latency_s=time.perf_counter() - started,
            stage_s=stage,
            pool_size=pool_size,
        )

    def chunk(self, chunk_id: str) -> Chunk:
        return self.chunks[chunk_id]

    def format(self, result: RetrievalResult, *, width: int = 78) -> str:
        lines = []
        for rank, (chunk_id, score) in enumerate(result.ranked, start=1):
            chunk = self.chunks[chunk_id]
            lines.append(f"{rank:2}. {score:8.4f}  {chunk.breadcrumb[:width]}")
            lines.append(f"     {chunk.url}")
        return "\n".join(lines) if lines else "no results"


def arms_for(names: Sequence[str] | None, *, offline: bool = False) -> list[str]:
    """Resolve an arm selection, defaulting to everything runnable."""
    if names:
        unknown = [n for n in names if n not in ARMS]
        if unknown:
            raise ValueError(f"unknown arm(s) {unknown}; expected from {sorted(ARMS)}")
        return list(names)
    return list(OFFLINE_ARMS) if offline else list(ARMS)
