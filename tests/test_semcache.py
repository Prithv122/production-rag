"""The semantic cache and its seam. Fakes only: no torch, no network, no disk outside tmp_path."""

from __future__ import annotations

import inspect
import subprocess
import sys

import numpy as np
import pytest

from conftest import FakeEmbedder, FakeProvider
from production_rag import semcache
from production_rag.cache import JsonCache
from production_rag.providers import CachedProvider, is_replaying
from production_rag.semcache import (
    DEFAULT_THRESHOLD,
    TRANSIENT_REFUSALS,
    SemanticCache,
    SemanticStore,
    answer_through,
    maybe_semantic_cache,
)


class DictEmbedder:
    """Hand-set vectors, so a similarity can be exactly what a test says it is."""

    def __init__(self, vectors: dict[str, list[float]], name: str = "dict") -> None:
        self._vectors = {k: np.asarray(v, dtype=np.float32) for k, v in vectors.items()}
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return len(next(iter(self._vectors.values())))

    def encode_query(self, text: str) -> np.ndarray:
        return self._vectors[text]

    def encode_documents(self, texts):
        raise AssertionError("the semantic cache must only encode queries")


ANSWER = {"text": "an answer", "refused": False, "refusal_reason": ""}


def test_empty_lookup_is_none():
    cache = SemanticCache(FakeEmbedder(), 0.9)
    assert cache.lookup("anything") is None and len(cache) == 0 and cache.misses == 1


def test_same_question_hits_at_about_one():
    cache = SemanticCache(FakeEmbedder(), 0.99)
    cache.insert("how do I read parquet in duckdb", ANSWER)
    hit = cache.lookup("how do I read parquet in duckdb")
    assert hit is not None and hit.similarity == pytest.approx(1.0, abs=1e-5)
    assert hit.question == "how do I read parquet in duckdb" and hit.answer == ANSWER
    assert cache.hits == 1


def test_boundary_is_a_hit_and_just_below_is_a_miss():
    emb = DictEmbedder(
        {
            "cached": [1, 0, 0, 0],
            "at": [0.5, 0.5, 0.5, 0.5],  # dot is exactly 0.5
            "below": [0.49, 0.5, 0.5, 0.5],
        }
    )
    cache = SemanticCache(emb, 0.5)
    cache.insert("cached", ANSWER)
    hit = cache.lookup("at")
    assert hit is not None and hit.similarity == 0.5
    assert cache.lookup("below") is None


def test_top1_wins_and_ties_go_to_the_earliest_insert():
    emb = DictEmbedder({"e1": [1, 0, 0], "e2": [0, 1, 0], "near-e1": [1, 0.1, 0], "mid": [1, 1, 0]})
    cache = SemanticCache(emb, 0.5)
    cache.insert("e2", {"text": "second"})
    cache.insert("e1", {"text": "first"})
    assert cache.lookup("near-e1").question == "e1"
    assert cache.lookup("mid").question == "e2"  # exact tie: e2 was inserted first


@pytest.mark.parametrize("bad", [None, 0, 0.0, -0.1, 1.5, True])
def test_invalid_threshold_raises(bad):
    with pytest.raises(ValueError):
        SemanticCache(FakeEmbedder(), bad)


def test_disabled_by_default():
    assert DEFAULT_THRESHOLD is None
    assert maybe_semantic_cache(FakeEmbedder()) is None
    assert isinstance(maybe_semantic_cache(FakeEmbedder(), 0.9), SemanticCache)
    with pytest.raises(ValueError, match="disabled"):
        SemanticCache(FakeEmbedder(), None)


def test_scope_isolation_on_a_shared_store():
    store, emb = SemanticStore(), FakeEmbedder()
    a = SemanticCache(emb, 0.9, scope="A", store=store)
    b = SemanticCache(emb, 0.9, scope="B", store=store)
    a.insert("same question", ANSWER)
    assert b.lookup("same question") is None
    assert a.lookup("same question") is not None
    assert len(a) == 1 and len(b) == 0


def test_different_embedders_do_not_share_entries():
    store = SemanticStore()
    one = SemanticCache(DictEmbedder({"q": [1, 0]}, name="one"), 0.9, store=store)
    two = SemanticCache(DictEmbedder({"q": [1, 0]}, name="two"), 0.9, store=store)
    one.insert("q", ANSWER)
    assert two.lookup("q") is None and one.lookup("q") is not None


@pytest.mark.parametrize("reason", TRANSIENT_REFUSALS)
def test_transient_refusals_are_not_inserted(reason):
    cache = SemanticCache(FakeEmbedder(), 0.9)
    assert cache.insert("q", {"refused": True, "refusal_reason": reason}) is False
    assert len(cache) == 0 and cache.lookup("q") is None


@pytest.mark.parametrize("reason", ["no_context", "low_score", "model", ""])
def test_deliberate_outcomes_are_cacheable(reason):
    cache = SemanticCache(FakeEmbedder(), 0.9)
    assert cache.insert("q", {"refused": bool(reason), "refusal_reason": reason}) is True
    assert cache.lookup("q") is not None


def test_answers_are_copied_in_and_out():
    cache = SemanticCache(FakeEmbedder(), 0.9)
    original = {"text": "x", "citations": [1]}
    cache.insert("q", original)
    original["citations"].append(2)
    served = cache.lookup("q").answer
    served["citations"].append(3)
    assert cache.lookup("q").answer == {"text": "x", "citations": [1]}


def test_duplicate_question_replaces_and_unserialisable_is_rejected():
    cache = SemanticCache(FakeEmbedder(), 0.9)
    cache.insert("q", {"text": "old"})
    cache.insert("q", {"text": "new"})
    assert len(cache) == 1 and cache.lookup("q").answer == {"text": "new"}
    with pytest.raises(TypeError):
        cache.insert("other", {"text": object()})


# -- the seam -----------------------------------------------------------------
def test_answer_through_without_a_cache_always_computes():
    calls = []
    compute = lambda: calls.append(1) or {"text": "a"}  # noqa: E731
    assert answer_through(None, "q", FakeProvider(), compute) == ({"text": "a"}, None)
    answer_through(None, "q", FakeProvider(), compute)
    assert len(calls) == 2


def test_answer_through_second_identical_call_is_a_hit():
    calls = []
    compute = lambda: calls.append(1) or {"text": "a"}  # noqa: E731
    cache = SemanticCache(FakeEmbedder(), 0.9)
    first, hit1 = answer_through(cache, "q", FakeProvider(), compute)
    second, hit2 = answer_through(cache, "q", FakeProvider(), compute)
    assert hit1 is None and hit2 is not None and first == second == {"text": "a"}
    assert len(calls) == 1


# -- replay must not move -----------------------------------------------------
def _files(root):
    return {p: p.read_bytes() for p in root.rglob("*") if p.is_file()}


def test_replay_bypasses_the_semantic_cache_entirely(tmp_path, monkeypatch):
    exact = JsonCache(tmp_path / "llm")
    CachedProvider(FakeProvider(['{"answer": "recorded"}']), exact).complete("prompt")
    provider = CachedProvider(FakeProvider(), JsonCache(tmp_path / "llm"), offline=True)
    sem = SemanticCache(FakeEmbedder(), 0.9)
    sem.insert("the question", {"text": "a different cached answer"})
    before = _files(tmp_path / "llm")
    assert len(before) == 1

    def compute():
        return {"text": provider.complete("prompt").text}

    def forbidden(*args, **kwargs):
        raise AssertionError("the semantic cache was touched during replay")

    monkeypatch.setattr(SemanticCache, "lookup", forbidden)
    monkeypatch.setattr(SemanticCache, "insert", forbidden)
    answer, hit = answer_through(sem, "the question", provider, compute)
    assert hit is None and answer == compute() == {"text": '{"answer": "recorded"}'}
    assert len(sem) == 1 and (sem.hits, sem.misses) == (0, 0)
    assert _files(tmp_path / "llm") == before


def test_is_replaying_is_the_only_replay_signal(tmp_path):
    inner = FakeProvider()
    assert is_replaying(CachedProvider(inner, JsonCache(tmp_path), offline=True)) is True
    assert is_replaying(CachedProvider(inner, JsonCache(tmp_path), offline=False)) is False
    source = inspect.getsource(semcache)
    assert "is_replaying" in source and "offline" not in source


def test_nothing_in_the_replay_path_imports_the_semantic_cache():
    code = (
        "import sys\n"
        "import production_rag.pipeline, production_rag.evaluate, production_rag.answers\n"
        "import production_rag.generate, production_rag.providers, production_rag.cache\n"
        "import production_rag.cli\n"
        "bad = [m for m in ('production_rag.semcache', 'production_rag.semcache_eval')"
        " if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert done.returncode == 0, done.stderr
