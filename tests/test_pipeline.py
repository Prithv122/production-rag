from __future__ import annotations

import json

import pytest

from conftest import FakeProvider, make_chunk
from production_rag.bm25 import build_from_chunks as build_bm25
from production_rag.dense import build_from_chunks as build_dense
from production_rag.pipeline import ARMS, OFFLINE_ARMS, Retriever, arms_for


@pytest.fixture
def retriever(embedder, sample_chunks):
    return Retriever(
        {c.chunk_id: c for c in sample_chunks},
        build_bm25(sample_chunks),
        build_dense(sample_chunks, embedder),
        embedder=embedder,
        strategy="heading",
    )


@pytest.mark.parametrize("arm", OFFLINE_ARMS)
def test_every_offline_arm_runs_without_a_key(arm, retriever, reranker):
    result = retriever.retrieve("incremental models", arm=arm, k=3, pool=4, reranker=reranker)
    assert result.arm == arm
    assert len(result.ranked) <= 3


def test_unknown_arm_is_rejected(retriever):
    with pytest.raises(ValueError, match="unknown arm"):
        retriever.retrieve("q", arm="magic")


def test_a_dense_arm_without_a_dense_index_fails_loudly(sample_chunks):
    lexical_only = Retriever({c.chunk_id: c for c in sample_chunks}, build_bm25(sample_chunks))
    with pytest.raises(ValueError, match="needs a dense index"):
        lexical_only.retrieve("q", arm="hybrid")


def test_a_rerank_arm_without_a_reranker_fails_loudly(retriever):
    with pytest.raises(ValueError, match="needs a reranker"):
        retriever.retrieve("q", arm="hybrid_rerank")


def test_a_rewrite_arm_without_a_provider_fails_loudly(retriever, reranker):
    with pytest.raises(ValueError, match="needs an LLM provider"):
        retriever.retrieve("q", arm="rerank_rewrite", reranker=reranker)


def test_the_reranker_sees_the_pool_not_the_top_k(retriever, reranker):
    """If it only saw k candidates it could never promote anything new, which
    is the entire reason the stage exists."""
    retriever.retrieve("dbt tests", arm="hybrid_rerank", k=1, pool=4, reranker=reranker)
    _, seen = reranker.calls[-1]
    assert seen > 1


def test_expansion_issues_one_retrieval_per_query(retriever, reranker):
    provider = FakeProvider([json.dumps({"queries": ["assets in dagster", "dag dependencies"]})])
    result = retriever.retrieve(
        "how do I declare dependencies",
        arm="rerank_rewrite",
        k=3,
        pool=4,
        reranker=reranker,
        rewriter=provider,
    )
    assert result.queries[0] == "how do I declare dependencies"
    assert len(result.queries) == 3
    assert result.rewrite is not None and result.rewrite.parsed


def test_replace_mode_drops_the_original_from_retrieval(retriever, reranker):
    provider = FakeProvider([json.dumps({"queries": ["parquet reading in duckdb"]})])
    result = retriever.retrieve(
        "how do I load columnar files",
        arm="rerank_rewrite_only",
        k=3,
        pool=4,
        reranker=reranker,
        rewriter=provider,
    )
    assert result.queries == ["parquet reading in duckdb"]


def test_the_cross_encoder_still_scores_against_the_users_words(retriever, reranker):
    """Expansion changes what is *retrieved*; it must not change what the
    reranker is judging relevance to, or the arm would be optimising for a
    query the user never asked."""
    provider = FakeProvider([json.dumps({"queries": ["totally different words"]})])
    original = "how do I declare dependencies"
    retriever.retrieve(
        original, arm="rerank_rewrite", k=2, pool=4, reranker=reranker, rewriter=provider
    )
    assert reranker.calls[-1][0] == original


def test_pool_is_never_smaller_than_k(retriever):
    """A caller asking for k=4 out of a pool of 1 has asked for something
    incoherent; the pool widens rather than the result being truncated to 1."""
    result = retriever.retrieve("dbt", arm="bm25", k=4, pool=1)
    assert result.pool_size > 1


def test_stage_timings_are_recorded(retriever, reranker):
    result = retriever.retrieve("dbt tests", arm="hybrid_rerank", k=2, pool=4, reranker=reranker)
    assert set(result.stage_s) == {"first_stage", "rerank"}


def test_format_renders_urls(retriever):
    result = retriever.retrieve("dbt tests", arm="bm25", k=1)
    assert "https://example.test/" in retriever.format(result)


def test_format_handles_no_results(retriever):
    result = retriever.retrieve("zzzz-nothing-matches", arm="bm25", k=5)
    assert retriever.format(result) == "no results"


def test_arms_for_defaults_to_offline_when_asked():
    assert arms_for(None, offline=True) == list(OFFLINE_ARMS)
    assert arms_for(None) == list(ARMS)
    assert arms_for(["bm25"]) == ["bm25"]
    with pytest.raises(ValueError):
        arms_for(["nope"])


def test_missing_chunk_text_is_not_silently_skipped(embedder):
    """A chunk in the index but absent from the chunk map is a build bug; the
    reranker must raise rather than quietly drop the candidate."""
    chunks = [make_chunk("x#0", "one"), make_chunk("y#0", "two")]
    retriever = Retriever(
        {"x#0": chunks[0]},  # y#0 deliberately missing
        build_bm25(chunks),
        build_dense(chunks, embedder),
        embedder=embedder,
    )
    with pytest.raises(KeyError):
        retriever.retrieve("two", arm="hybrid_rerank", k=2, pool=2, reranker=_always_zero())


def _always_zero():
    from conftest import FakeReranker

    return FakeReranker()
