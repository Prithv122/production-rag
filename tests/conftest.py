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
