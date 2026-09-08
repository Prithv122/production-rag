"""Cross-encoder reranking.

The retrieval arms score a query against a chunk *independently*: BM25 counts
term overlap, and the dense arm compares two vectors that were computed without
ever seeing each other. That independence is what makes them fast enough to
score 24,000 chunks per query, and it is also their ceiling -- neither can
notice that a passage mentions the right identifier in the wrong role, or that
it answers a superficially similar question about a different tool.

A cross-encoder gives up the independence: it feeds ``(query, passage)`` through
one transformer and reads a relevance score off the joint representation. That
is quadratically more expensive, which is why it never runs over the corpus --
it reranks the top *n* candidates the cheap arms already produced. The whole
design is a funnel: recall is the first stage's job, precision is the second's.

**Model choice, with the cost stated.** ``ms-marco-MiniLM-L-6-v2`` (22M
parameters) over ``bge-reranker-base`` (278M). On the CPU this project is
constrained to, the larger model is roughly an order of magnitude slower per
pair, and reranking 50 candidates for every question in the eval grid is the
single most repeated operation in the whole harness. The measured latency for
both is in the README; picking the big one because it scores higher on a
leaderboard, without pricing it, is exactly the reasoning this project exists to
avoid.

**Reranking can make things worse, and the eval has to be able to say so.** The
reranker only sees the candidate pool, so it cannot recover a gold chunk the
first stage missed -- its recall@pool is fixed by definition. What it can do is
demote a correct chunk it disagrees with. So it is an arm to be measured, not a
stage to be assumed, and the pool size is the parameter that trades first-stage
recall against second-stage precision.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .cache import JsonCache
from .chunking import Chunk

DEFAULT_RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

#: How many first-stage candidates the reranker sees. Above this the cost grows
#: linearly for a recall gain that has already flattened; below it the reranker
#: is being asked to fix a pool that does not contain the answer.
DEFAULT_POOL = 50


@runtime_checkable
class Reranker(Protocol):
    """Anything that scores (query, passage) pairs jointly."""

    @property
    def name(self) -> str: ...

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray: ...


class CrossEncoderReranker:
    """`sentence-transformers` CrossEncoder. Requires the `embed` extra.

    Imports torch lazily inside the constructor for the same reason
    :class:`~production_rag.dense.SentenceTransformerEmbedder` does: the fast
    test suite substitutes a fake through this protocol and must never download
    a model to check that reranking reorders things.
    """

    def __init__(self, model_name: str = DEFAULT_RERANK_MODEL, *, batch_size: int = 32) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise ModuleNotFoundError(
                "sentence-transformers is not installed. It lives in an optional "
                "group so CI does not pull torch: `uv sync --extra embed`."
            ) from exc

        self._model = CrossEncoder(model_name)
        self._name = model_name
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return self._name

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        if not passages:
            return np.zeros(0, dtype=np.float32)
        scores = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        return np.asarray(scores, dtype=np.float32).reshape(-1)


class CachedReranker:
    """Memoises pair scores so the eval grid replays without a model.

    One cache entry per ``(model, query)``, holding a mapping from passage hash
    to score, rather than one entry per pair. The eval reranks the same query
    against overlapping pools across arms and strategies, so per-pair files
    would mean tens of thousands of tiny writes -- slow on Windows, and a
    directory nobody can inspect. Per-query files stay readable and are still
    exact: a passage the entry has not seen is simply a miss.
    """

    def __init__(self, inner: Reranker, cache_dir: Path, *, offline: bool = False) -> None:
        self._inner = inner
        self._cache = JsonCache(Path(cache_dir))
        self.offline = offline
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self._inner.name

    @staticmethod
    def _digest(passage: str) -> str:
        return hashlib.sha256(passage.encode("utf-8")).hexdigest()[:32]

    def score(self, query: str, passages: Sequence[str]) -> np.ndarray:
        passages = list(passages)
        if not passages:
            return np.zeros(0, dtype=np.float32)

        key = {"reranker": self._inner.name, "query": query}
        known: dict[str, float] = self._cache.get(key) or {}
        digests = [self._digest(p) for p in passages]
        missing = {d: p for d, p in zip(digests, passages, strict=True) if d not in known}

        if missing:
            if self.offline:
                raise LookupError(
                    f"replay mode: {len(missing)} passage(s) for this query are not in the "
                    "rerank cache. Re-run without --replay to populate it."
                )
            fresh = self._inner.score(query, list(missing.values()))
            known.update({d: float(s) for d, s in zip(missing.keys(), fresh, strict=True)})
            self._cache.put(key, known)

        self.hits += len(passages) - len(missing)
        self.misses += len(missing)
        return np.asarray([known[d] for d in digests], dtype=np.float32)


def rerank(
    query: str,
    candidates: Sequence[tuple[str, float]],
    chunks: dict[str, Chunk],
    reranker: Reranker,
    *,
    k: int = 10,
    with_breadcrumb: bool = False,
) -> list[tuple[str, float]]:
    """Rescore first-stage `candidates` and return the top `k`.

    The returned scores are the cross-encoder's, not the first stage's -- they
    are on a different scale and mixing the two would produce a ranking that
    means nothing. The first-stage order survives only as the tie-break, via a
    stable sort over the input order.
    """
    if not candidates or k <= 0:
        return []
    ids = [chunk_id for chunk_id, _ in candidates]
    passages = [chunks[chunk_id].embed_text(with_breadcrumb=with_breadcrumb) for chunk_id in ids]
    scores = reranker.score(query, passages)
    if len(scores) != len(ids):  # pragma: no cover - defensive
        raise ValueError("reranker returned the wrong number of scores")
    order = np.argsort(-scores, kind="stable")
    return [(ids[i], float(scores[i])) for i in order[:k]]
