from __future__ import annotations

import pytest

from conftest import FakeReranker, make_chunk
from production_rag.rerank import CachedReranker, rerank


@pytest.fixture
def chunks():
    return {
        c.chunk_id: c
        for c in [
            make_chunk("a#0", "unrelated prose about billing and invoices"),
            make_chunk("b#0", "incremental models append new rows on every run"),
            make_chunk("c#0", "incremental"),
        ]
    }


def test_rerank_reorders_by_the_cross_encoder_not_the_first_stage(chunks, reranker):
    # First stage ranks the irrelevant chunk top; the reranker must fix that.
    candidates = [("a#0", 9.0), ("b#0", 0.1), ("c#0", 0.05)]
    ranked = rerank("incremental models append rows", candidates, chunks, reranker, k=3)
    assert ranked[0][0] == "b#0"


def test_returned_scores_are_the_reranker_scores(chunks, reranker):
    candidates = [("a#0", 9.0), ("b#0", 0.1)]
    ranked = rerank("incremental models", candidates, chunks, reranker, k=2)
    assert max(score for _, score in ranked) <= 1.0  # not the BM25 scale


def test_k_truncates_after_reranking(chunks, reranker):
    candidates = [("a#0", 1.0), ("b#0", 1.0), ("c#0", 1.0)]
    assert len(rerank("incremental", candidates, chunks, reranker, k=2)) == 2


def test_empty_pool_and_zero_k(chunks, reranker):
    assert rerank("q", [], chunks, reranker, k=5) == []
    assert rerank("q", [("a#0", 1.0)], chunks, reranker, k=0) == []


def test_ties_keep_the_first_stage_order(chunks):
    class Flat(FakeReranker):
        def score(self, query, passages):
            import numpy as np

            return np.zeros(len(list(passages)), dtype="float32")

    ranked = rerank("q", [("b#0", 1.0), ("a#0", 0.5)], chunks, Flat(), k=2)
    assert [chunk_id for chunk_id, _ in ranked] == ["b#0", "a#0"]


def test_cached_reranker_skips_the_model_on_a_repeat_query(tmp_path):
    inner = FakeReranker()
    cached = CachedReranker(inner, tmp_path)
    passages = ["incremental models", "billing"]

    first = cached.score("how do incremental models work", passages)
    second = CachedReranker(FakeReranker(), tmp_path).score(
        "how do incremental models work", passages
    )

    assert len(inner.calls) == 1
    assert list(first) == list(second)


def test_cached_reranker_only_scores_the_passages_it_has_not_seen(tmp_path):
    inner = FakeReranker()
    cached = CachedReranker(inner, tmp_path)
    cached.score("q", ["one", "two"])
    cached.score("q", ["two", "three"])
    assert inner.calls == [("q", 2), ("q", 1)]


def test_replay_refuses_to_score_an_uncached_passage(tmp_path):
    cached = CachedReranker(FakeReranker(), tmp_path, offline=True)
    with pytest.raises(LookupError, match="replay mode"):
        cached.score("q", ["never seen"])
