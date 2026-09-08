"""Dense retrieval by exact search over normalised embeddings.

**Exact search, not ANN, and that is a decision rather than an omission.** The
index holds ~23k chunks at 384 dimensions: 35 MB as float32, and a full
similarity pass is one `(n, d) @ (d,)` matrix-vector product. The measured
latency is in the README. FAISS or HNSW would add a dependency, a build step, an
index-tuning parameter and an approximation error, in exchange for making a
millisecond-scale operation faster. The scale at which that trade flips is
covered in README section 7; it is not this scale. Reaching for a vector database
here would be resume-driven development.

**The encoder is an interface.** `Embedder` is a protocol, and nothing in this
module imports torch at module scope -- `SentenceTransformerEmbedder` imports it
lazily inside the constructor. That keeps `sentence-transformers` in an optional
dependency group, so the test suite and CI run against a deterministic fake and
never download 2 GB of wheels to check that ranking works.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

DEFAULT_MODEL = "BAAI/bge-small-en-v1.5"

# bge asks for an instruction prefix on the *query* side only; the corpus side is
# embedded bare. Getting this backwards silently costs a few points of recall and
# produces no error, which is why it lives in a named constant with a test.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


@runtime_checkable
class Embedder(Protocol):
    """Anything that turns text into unit-norm vectors."""

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray: ...

    def encode_query(self, text: str) -> np.ndarray: ...


def l2_normalise(matrix: np.ndarray) -> np.ndarray:
    """Scale rows to unit norm so a dot product is a cosine similarity."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.maximum(norms, 1e-12)


class SentenceTransformerEmbedder:
    """`sentence-transformers` encoder. Requires the `embed` extra."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        query_prefix: str = BGE_QUERY_PREFIX,
        batch_size: int = 64,
    ) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ModuleNotFoundError as exc:  # pragma: no cover - env-dependent
            raise ModuleNotFoundError(
                "sentence-transformers is not installed. It lives in an optional "
                "group so CI does not pull torch: `uv sync --extra embed`."
            ) from exc

        self._model = SentenceTransformer(model_name)
        self._name = model_name
        self._query_prefix = query_prefix
        self._batch_size = batch_size

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        vectors = self._model.encode(
            list(texts),
            batch_size=self._batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        vectors = self._model.encode(
            [f"{self._query_prefix}{text}"],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return np.asarray(vectors, dtype=np.float32)[0]


class DenseIndex:
    """Exact cosine search over a dense matrix of chunk embeddings."""

    def __init__(
        self,
        chunk_ids: Sequence[str],
        vectors: np.ndarray,
        *,
        model_name: str = DEFAULT_MODEL,
    ) -> None:
        if len(chunk_ids) != len(vectors):
            raise ValueError("chunk_ids and vectors must be the same length")
        self.chunk_ids = list(chunk_ids)
        self.vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self.model_name = model_name

    @classmethod
    def build(
        cls,
        chunk_ids: Sequence[str],
        texts: Sequence[str],
        embedder: Embedder,
    ) -> DenseIndex:
        if len(chunk_ids) != len(texts):
            raise ValueError("chunk_ids and texts must be the same length")
        if not chunk_ids:
            raise ValueError("cannot build an index over zero chunks")
        vectors = l2_normalise(np.asarray(embedder.encode_documents(texts), dtype=np.float32))
        return cls(chunk_ids, vectors, model_name=embedder.name)

    def scores_for_vector(self, query_vector: np.ndarray) -> np.ndarray:
        """Cosine similarity of every chunk against an already-encoded query."""
        vector = l2_normalise(np.asarray(query_vector, dtype=np.float32).reshape(-1))
        return self.vectors @ vector

    def search_vector(self, query_vector: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        scores = self.scores_for_vector(query_vector)
        if k <= 0:
            return []
        k = min(k, len(scores))
        top = np.argpartition(-scores, k - 1)[:k]
        top = top[np.argsort(-scores[top], kind="stable")]
        return [(self.chunk_ids[i], float(scores[i])) for i in top]

    def search(self, query: str, embedder: Embedder, k: int = 10) -> list[tuple[str, float]]:
        return self.search_vector(embedder.encode_query(query), k)

    # -- persistence ------------------------------------------------------
    def save(self, path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)
        (path / "meta.json").write_text(
            json.dumps({"chunk_ids": self.chunk_ids, "model_name": self.model_name}),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> DenseIndex:
        vectors = np.load(path / "vectors.npy")
        meta = json.loads((path / "meta.json").read_text(encoding="utf-8"))
        return cls(meta["chunk_ids"], vectors, model_name=meta["model_name"])

    def __len__(self) -> int:
        return len(self.chunk_ids)


def build_from_chunks(
    chunks: Sequence, embedder: Embedder, *, with_breadcrumb: bool = False
) -> DenseIndex:
    return DenseIndex.build(
        [c.chunk_id for c in chunks],
        [c.embed_text(with_breadcrumb=with_breadcrumb) for c in chunks],
        embedder,
    )
