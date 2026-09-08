from __future__ import annotations

import numpy as np
import pytest

from conftest import FakeEmbedder
from production_rag.cache import CachedEmbedder, CacheMiss, JsonCache, stable_hash


def test_stable_hash_ignores_key_order():
    assert stable_hash({"a": 1, "b": 2}) == stable_hash({"b": 2, "a": 1})


def test_stable_hash_distinguishes_values():
    assert stable_hash({"model": "a"}) != stable_hash({"model": "b"})


def test_round_trip(tmp_path):
    cache = JsonCache(tmp_path)
    key = {"prompt": "hello", "model": "m"}
    assert cache.get(key) is None
    cache.put(key, {"text": "world"})
    assert cache.get(key) == {"text": "world"}
    assert cache.hits == 1 and cache.misses == 1


def test_require_raises_on_miss(tmp_path):
    with pytest.raises(CacheMiss):
        JsonCache(tmp_path).require({"prompt": "absent"})


def test_a_truncated_entry_is_a_miss_not_a_crash(tmp_path):
    """An interrupted write must cost one call, not poison the whole run."""
    cache = JsonCache(tmp_path)
    key = {"prompt": "half"}
    cache.put(key, {"text": "ok"})
    cache.path_for(key).write_text('{"key": {"prom', encoding="utf-8")
    assert cache.get(key) is None


def test_embedding_cache_serves_the_second_call(tmp_path):
    class Counting(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.encoded = 0

        def encode_documents(self, texts):
            self.encoded += len(list(texts))
            return super().encode_documents(texts)

    inner = Counting()
    cached = CachedEmbedder(inner, tmp_path)
    texts = ["alpha beta", "gamma delta"]
    first = cached.encode_documents(texts)
    second = CachedEmbedder(Counting(), tmp_path).encode_documents(texts)

    assert inner.encoded == 2
    np.testing.assert_allclose(first, second)


def test_embedding_cache_deduplicates_within_a_batch(tmp_path):
    """Repeated boilerplate should be encoded once, not once per occurrence."""

    class Counting(FakeEmbedder):
        def __init__(self):
            super().__init__()
            self.encoded = 0

        def encode_documents(self, texts):
            self.encoded += len(list(texts))
            return super().encode_documents(texts)

    inner = Counting()
    cached = CachedEmbedder(inner, tmp_path)
    vectors = cached.encode_documents(["same text", "other", "same text"])

    assert inner.encoded == 2
    assert vectors.shape == (3, inner.dim)
    np.testing.assert_allclose(vectors[0], vectors[2])


def test_embedding_cache_is_keyed_by_model(tmp_path):
    """Two encoders must never share an entry, however alike their output looks."""
    a = CachedEmbedder(FakeEmbedder(), tmp_path)
    b = CachedEmbedder(FakeEmbedder(dim=32), tmp_path)
    a.encode_documents(["shared"])
    b.encode_documents(["shared"])
    assert b.hits == 0
