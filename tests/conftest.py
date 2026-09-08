"""Shared fixtures.

The important one is :class:`FakeEmbedder`. Every retrieval test that needs
vectors uses it rather than a real encoder, which is why the fast suite runs
without torch, without a model download, and in well under a second. The real
encoder is exercised by the `slow` marker only.

It is a hashing embedder: deterministic, dependency-free, and -- crucially --
it puts documents sharing vocabulary near each other, so ranking assertions
about *relative* order are meaningful rather than arbitrary.
"""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np
import pytest

from production_rag.bm25 import tokenize
from production_rag.chunking import Chunk
from production_rag.dense import l2_normalise


class FakeEmbedder:
    """Deterministic bag-of-hashed-tokens embedder. No model, no network."""

    def __init__(self, dim: int = 64) -> None:
        self._dim = dim

    @property
    def name(self) -> str:
        return f"fake-hashing-{self._dim}"

    @property
    def dim(self) -> int:
        return self._dim

    def _vector(self, text: str) -> np.ndarray:
        vector = np.zeros(self._dim, dtype=np.float32)
        for token in tokenize(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            index = int.from_bytes(digest[:4], "big") % self._dim
            sign = 1.0 if digest[4] % 2 else -1.0
            vector[index] += sign
        return vector

    def encode_documents(self, texts: Sequence[str]) -> np.ndarray:
        return l2_normalise(np.stack([self._vector(t) for t in texts]))

    def encode_query(self, text: str) -> np.ndarray:
        return l2_normalise(self._vector(text))


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


def make_chunk(chunk_id: str, text: str, *, tool: str = "dbt", title: str = "Guide") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=chunk_id.split("#")[0],
        tool=tool,
        title=title,
        url=f"https://example.test/{chunk_id}",
        heading_path=("Section",),
        text=text,
        start=0,
        end=len(text),
    )


@pytest.fixture
def sample_chunks() -> list[Chunk]:
    return [
        make_chunk("dbt/inc#0", "dbt incremental models append new rows to a table"),
        make_chunk("dagster/assets#0", "dagster assets declare dependencies", tool="dagster"),
        make_chunk("duckdb/parquet#0", "duckdb read_parquet reads parquet files", tool="duckdb"),
        make_chunk("dbt/tests#0", "dbt generic tests assert not null and unique"),
    ]


class FakeReranker:
    """Scores by token overlap, so reordering is predictable without torch.

    The point of a fake here is not to imitate a cross-encoder's judgement --
    nothing dependency-free can -- but to let every arm's *wiring* be tested:
    that the reranker sees the pool and not the top-k, that its scores replace
    the first stage's rather than being added to them, and that ties keep the
    incoming order.
    """

    def __init__(self, name: str = "fake-cross-encoder") -> None:
        self._name = name
        self.calls: list[tuple[str, int]] = []

    @property
    def name(self) -> str:
        return self._name

    def score(self, query: str, passages) -> np.ndarray:
        passages = list(passages)
        self.calls.append((query, len(passages)))
        wanted = set(tokenize(query))
        return np.asarray(
            [len(wanted & set(tokenize(passage))) / (len(wanted) or 1) for passage in passages],
            dtype=np.float32,
        )


class FakeProvider:
    """Returns scripted text. Optionally fails, to exercise the fallback path."""

    def __init__(
        self,
        responses: list[str] | None = None,
        *,
        name: str = "fake",
        model: str = "fake-model",
        fail: bool = False,
    ) -> None:
        self._responses = list(responses or [])
        self._name = name
        self._model = model
        self.fail = fail
        self.prompts: list[str] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def model(self) -> str:
        return self._model

    def complete(self, prompt: str, **kwargs):
        from production_rag.providers import LLMResponse, ProviderError

        self.prompts.append(prompt)
        if self.fail:
            raise ProviderError(f"{self._name} is down")
        text = self._responses.pop(0) if self._responses else "{}"
        return LLMResponse(text=text, model=self._model, provider=self._name)


@pytest.fixture
def reranker() -> FakeReranker:
    return FakeReranker()
