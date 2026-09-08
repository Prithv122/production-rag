"""Content-addressed on-disk caches.

Two things in this pipeline are expensive and deterministic given their input,
which is exactly the shape a cache wants:

**Embeddings.** Session 1 built all three chunking strategies from scratch and
paid 127 minutes of CPU for it. `heading` and `heading_ctx` chunk the same
corpus by the same rules and differ mostly in a prefixed breadcrumb, so a large
fraction of their chunk text is byte-identical -- and was embedded twice. A
cache keyed by the hash of the text being embedded removes that.

**LLM calls.** Query rewrites and generations cost money, take seconds, and are
not reproducible: the same prompt to the same model can return different text
tomorrow, or the model can be withdrawn entirely. Every published number in this
repository has to survive that, so both go through :class:`JsonCache` and
`eval --replay` refuses to make a network call at all. A number nobody else can
reproduce is not a result.

The cache key is a hash of *everything that affects the output* -- for an LLM
that means provider, model, prompt, system prompt and sampling parameters. Get
that wrong in the direction of too little and the cache silently serves a
different model's answer, which is far worse than a cache miss. **API keys are
never part of the key and never stored**; the key is a hash of the prompt, and
the value is the response.
"""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def stable_hash(payload: Any) -> str:
    """SHA-256 of a canonical JSON encoding.

    `sort_keys` is what makes this stable: without it two dicts that differ only
    in insertion order hash differently and the cache misses every time.
    """
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class CacheMiss(LookupError):
    """Raised by a cache in offline mode when the entry is not present."""


class JsonCache:
    """One JSON file per entry, named by the hash of its key.

    A directory of small files rather than one big file or a database: it is
    concurrency-safe enough for our purposes (writes are atomic renames), it
    diffs sanely, and a single corrupt entry costs one call rather than the
    whole cache.
    """

    def __init__(self, root: Path, *, offline: bool = False) -> None:
        self.root = Path(root)
        self.offline = offline
        self.hits = 0
        self.misses = 0
        self._lock = threading.Lock()

    def path_for(self, key: Any) -> Path:
        digest = stable_hash(key)
        # Shard by the first two hex characters; 256 directories keeps any one
        # of them small enough that Windows directory listing stays quick.
        return self.root / digest[:2] / f"{digest}.json"

    def get(self, key: Any) -> Any | None:
        path = self.path_for(key)
        if not path.exists():
            with self._lock:
                self.misses += 1
            return None
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            # A half-written entry from an interrupted run. Treat as a miss.
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return entry["value"]

    def put(self, key: Any, value: Any) -> None:
        path = self.path_for(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"key": key, "value": value}, ensure_ascii=False),
            encoding="utf-8",
        )
        tmp.replace(path)

    def require(self, key: Any) -> Any:
        """Fetch, or raise if absent -- the offline-replay path."""
        value = self.get(key)
        if value is None:
            raise CacheMiss(
                f"no cached entry for {stable_hash(key)[:12]}. Running with --replay "
                "means every call must already be cached; re-run without it to populate."
            )
        return value

    def __len__(self) -> int:
        return sum(1 for _ in self.root.rglob("*.json"))


class CachedEmbedder:
    """Wraps an :class:`~production_rag.dense.Embedder`, memoising by text hash.

    Implements the same protocol as what it wraps, so it drops in anywhere an
    embedder is accepted -- including `DenseIndex.build`.

    The key includes the model name. Two encoders producing 384-dimensional
    vectors for the same string is not a reason to share a cache entry, and
    silently serving bge vectors from a MiniLM run would corrupt an index in a
    way that produces plausible-looking but wrong numbers.
    """

    def __init__(self, inner, root: Path) -> None:
        self._inner = inner
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.hits = 0
        self.misses = 0

    @property
    def name(self) -> str:
        return self._inner.name

    @property
    def dim(self) -> int:
        return self._inner.dim

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self._inner.name}\x00{text}".encode()).hexdigest()

    def _path(self, digest: str) -> Path:
        return self.root / digest[:2] / f"{digest}.npy"

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        texts = list(texts)
        digests = [self._key(t) for t in texts]
        out: list[np.ndarray | None] = [None] * len(texts)

        # Deduplicate within the batch as well as against disk: a corpus with
        # repeated boilerplate sections embeds each distinct string once.
        pending: dict[str, list[int]] = {}
        for i, digest in enumerate(digests):
            path = self._path(digest)
            if path.exists():
                out[i] = np.load(path)
                self.hits += 1
            else:
                pending.setdefault(digest, []).append(i)

        if pending:
            order = list(pending)
            fresh = np.asarray(
                self._inner.encode_documents([texts[pending[d][0]] for d in order]),
                dtype=np.float32,
            )
            for digest, vector in zip(order, fresh, strict=True):
                path = self._path(digest)
                path.parent.mkdir(parents=True, exist_ok=True)
                tmp = path.with_suffix(".tmp.npy")
                np.save(tmp, vector)
                tmp.replace(path)
                for i in pending[digest]:
                    out[i] = vector
                    self.misses += 1

        return np.stack([v for v in out if v is not None]).astype(np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        # Queries are not cached here. There are a handful of them per run, they
        # are cheap, and caching them would mean a stale entry survives a change
        # to the query prefix -- a bug that is invisible in the numbers.
        return self._inner.encode_query(text)
