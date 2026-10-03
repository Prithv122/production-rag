"""A question-level semantic cache that sits in front of the exact LLM cache.

The exact cache (`CachedProvider`) only helps when a prompt repeats byte for
byte. A semantic cache also serves a *different* question that embeds close to a
cached one, which buys more hits and adds one new failure: a false hit, where a
question that reads almost the same needs a different answer (`read_parquet`
versus `write_parquet`, "supports" versus "does not support"). Whether that
trade is worth making is a measurement, not an opinion, so the cache ships
disabled and `semcache-sweep` (see `semcache_eval.py`) is the instrument that
decides the threshold.

Three constraints shape it. It stores whole answers keyed by the question's
embedding and never touches an LLM-call entry. It is bypassed entirely whenever
`is_replaying(provider)` is true: replay must reproduce recorded numbers, and a
semantic hit there would serve another question's answer. And every entry lives
in a bucket keyed by (embedder name, scope), so two scopes (tenants, corpora,
index builds) can share a store without ever seeing each other's answers.

In-memory only, by design: no persistence and no disk I/O. Not thread-safe.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from .dense import Embedder, l2_normalise
from .providers import is_replaying

# Stays None (cache disabled) until the owner commits a threshold measured by
# `semcache-sweep`. Nothing else may set a default.
DEFAULT_THRESHOLD: float | None = None

# Caching a failure serves it forever, which is the failure mode `is_replaying`'s
# docstring describes. Deliberate refusals (no_context, low_score, model) are
# real outcomes for that question and are cacheable.
TRANSIENT_REFUSALS = ("provider_error", "unparseable", "truncated")


@dataclass(frozen=True)
class Hit:
    question: str
    """The cached question that matched, not the one that was asked."""
    answer: dict
    similarity: float
    scope: str


@dataclass
class _Bucket:
    questions: list[str] = field(default_factory=list)
    matrix: np.ndarray | None = None
    answers: list[dict] = field(default_factory=list)


class SemanticStore:
    """Entries bucketed by (embedder name, scope); shareable between caches."""

    def __init__(self) -> None:
        self._buckets: dict[tuple[str, str], _Bucket] = {}

    def bucket(self, embedder_name: str, scope: str) -> _Bucket:
        return self._buckets.setdefault((embedder_name, scope), _Bucket())


def top1(matrix: np.ndarray, vector: np.ndarray) -> tuple[int, float]:
    """Best row of unit-vector `matrix` for unit `vector`; the earliest row wins ties.

    The one scoring routine, shared with the sweep so a measured classification
    and a production lookup cannot drift apart.
    """
    scores = matrix @ vector
    best = int(np.argmax(scores))  # argmax returns the first maximum
    return best, float(scores[best])


class SemanticCache:
    def __init__(
        self,
        embedder: Embedder,
        threshold: float | None,
        *,
        scope: str = "",
        store: SemanticStore | None = None,
    ) -> None:
        if threshold is None:
            raise ValueError(
                "the semantic cache is disabled until a measured threshold is set "
                "(see `semcache-sweep`)"
            )
        if isinstance(threshold, bool) or not isinstance(threshold, (int, float)):
            raise ValueError(f"threshold must be a float, got {threshold!r}")
        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must satisfy 0 < threshold <= 1, got {threshold}")
        self.embedder = embedder
        self.threshold = float(threshold)
        self.scope = scope
        self.store = store if store is not None else SemanticStore()
        self.hits = 0
        self.misses = 0
        self._bucket = self.store.bucket(embedder.name, scope)
        self._last: tuple[str, np.ndarray] | None = None

    def _encode(self, question: str) -> np.ndarray:
        # Both sides of every comparison are questions, so always the query side.
        if self._last is not None and self._last[0] == question:
            return self._last[1]
        vector = l2_normalise(np.asarray(self.embedder.encode_query(question), dtype=np.float32))
        self._last = (question, vector)
        return vector

    def lookup(self, question: str) -> Hit | None:
        bucket = self._bucket
        if bucket.matrix is not None:
            index, score = top1(bucket.matrix, self._encode(question))
            if score >= self.threshold:  # the boundary counts as a hit
                self.hits += 1
                return Hit(
                    bucket.questions[index], copy.deepcopy(bucket.answers[index]), score, self.scope
                )
        self.misses += 1
        return None

    def insert(self, question: str, answer: dict) -> bool:
        json.dumps(answer)  # raises TypeError on anything that could not be persisted later
        if answer.get("refusal_reason") in TRANSIENT_REFUSALS:
            return False
        bucket, stored = self._bucket, copy.deepcopy(answer)
        if question in bucket.questions:
            bucket.answers[bucket.questions.index(question)] = stored
            return True
        row = self._encode(question).reshape(1, -1)
        bucket.matrix = row if bucket.matrix is None else np.vstack([bucket.matrix, row])
        bucket.questions.append(question)
        bucket.answers.append(stored)
        return True

    def __len__(self) -> int:
        return len(self._bucket.questions)


def maybe_semantic_cache(
    embedder: Embedder,
    threshold: float | None = DEFAULT_THRESHOLD,
    *,
    scope: str = "",
    store: SemanticStore | None = None,
) -> SemanticCache | None:
    """The constructor callers use: None while the cache is disabled."""
    if threshold is None:
        return None
    return SemanticCache(embedder, threshold, scope=scope, store=store)


def answer_through(
    cache: SemanticCache | None,
    question: str,
    provider,
    compute: Callable[[], dict],
) -> tuple[dict, Hit | None]:
    """Lookup order: semantic cache first, then `compute` (the normal pipeline).

    The pipeline's provider is the exact cache, untouched. When the provider
    stack is replaying, the semantic cache is not consulted and not written.
    """
    if cache is None or is_replaying(provider):
        return compute(), None
    hit = cache.lookup(question)
    if hit is not None:
        return hit.answer, hit
    answer = compute()
    cache.insert(question, answer)
    return answer, None
