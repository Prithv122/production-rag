"""Dense-index tests. Uses the hashing `FakeEmbedder` -- no torch, no download."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from conftest import FakeEmbedder
from production_rag.dense import (
    BGE_QUERY_PREFIX,
    DEFAULT_MODEL,
    DenseIndex,
    Embedder,
    build_from_chunks,
    l2_normalise,
)

TEXTS = [
    ("a", "dbt incremental models append new rows"),
    ("b", "dagster assets declare dependencies"),
    ("c", "duckdb read_parquet reads parquet files"),
]


@pytest.fixture
def index(embedder: FakeEmbedder) -> DenseIndex:
    return DenseIndex.build([i for i, _ in TEXTS], [t for _, t in TEXTS], embedder)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------
def test_l2_normalise_gives_unit_rows() -> None:
    matrix = np.array([[3.0, 4.0], [1.0, 0.0]], dtype=np.float32)
    np.testing.assert_allclose(np.linalg.norm(l2_normalise(matrix), axis=1), [1.0, 1.0])


def test_l2_normalise_survives_a_zero_row() -> None:
    """A chunk of pure punctuation embeds to zero; it must not produce NaN."""
    out = l2_normalise(np.zeros((1, 4), dtype=np.float32))
    assert np.isfinite(out).all()


def test_l2_normalise_handles_a_single_vector() -> None:
    out = l2_normalise(np.array([0.0, 3.0], dtype=np.float32))
    np.testing.assert_allclose(np.linalg.norm(out), 1.0)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------
def test_search_ranks_the_lexically_closest_first(
    index: DenseIndex, embedder: FakeEmbedder
) -> None:
    assert index.search("parquet files duckdb", embedder, k=1)[0][0] == "c"


def test_scores_are_cosine_similarities(index: DenseIndex, embedder: FakeEmbedder) -> None:
    scores = index.scores_for_vector(embedder.encode_query("dbt incremental models"))
    assert ((scores >= -1.0001) & (scores <= 1.0001)).all()


def test_an_identical_query_scores_near_one(embedder: FakeEmbedder) -> None:
    index = DenseIndex.build(["a"], ["dbt incremental models"], embedder)
    assert index.scores_for_vector(embedder.encode_query("dbt incremental models"))[0] == (
        pytest.approx(1.0, abs=1e-5)
    )


def test_search_respects_k(index: DenseIndex, embedder: FakeEmbedder) -> None:
    assert len(index.search("dbt", embedder, k=2)) == 2


def test_search_k_zero(index: DenseIndex, embedder: FakeEmbedder) -> None:
    assert index.search("dbt", embedder, k=0) == []


def test_search_k_above_corpus_size(index: DenseIndex, embedder: FakeEmbedder) -> None:
    assert len(index.search("dbt", embedder, k=50)) == len(TEXTS)


def test_search_is_sorted_descending(index: DenseIndex, embedder: FakeEmbedder) -> None:
    scores = [s for _, s in index.search("dbt incremental", embedder, k=3)]
    assert scores == sorted(scores, reverse=True)


def test_dense_returns_results_even_with_no_lexical_overlap(
    index: DenseIndex, embedder: FakeEmbedder
) -> None:
    """Unlike BM25, dense search always ranks everything -- there is no zero floor.

    This asymmetry is exactly why fusion cannot simply add the two score scales.
    """
    assert len(index.search("kubernetes helm chart", embedder, k=3)) == 3


# ---------------------------------------------------------------------------
# Build contract
# ---------------------------------------------------------------------------
def test_build_rejects_mismatched_lengths(embedder: FakeEmbedder) -> None:
    with pytest.raises(ValueError, match="same length"):
        DenseIndex.build(["a", "b"], ["only one"], embedder)


def test_build_rejects_an_empty_corpus(embedder: FakeEmbedder) -> None:
    with pytest.raises(ValueError, match="zero chunks"):
        DenseIndex.build([], [], embedder)


def test_constructor_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        DenseIndex(["a", "b"], np.zeros((1, 4), dtype=np.float32))


def test_build_records_the_model_name(index: DenseIndex, embedder: FakeEmbedder) -> None:
    assert index.model_name == embedder.name


def test_len(index: DenseIndex) -> None:
    assert len(index) == len(TEXTS)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_save_load_round_trip(index: DenseIndex, embedder: FakeEmbedder, tmp_path: Path) -> None:
    index.save(tmp_path / "dense")
    reloaded = DenseIndex.load(tmp_path / "dense")

    assert reloaded.chunk_ids == index.chunk_ids
    assert reloaded.model_name == index.model_name
    np.testing.assert_allclose(reloaded.vectors, index.vectors)
    assert reloaded.search("parquet", embedder, k=1) == index.search("parquet", embedder, k=1)


# ---------------------------------------------------------------------------
# Interface and configuration
# ---------------------------------------------------------------------------
def test_fake_embedder_satisfies_the_protocol(embedder: FakeEmbedder) -> None:
    """The point of the protocol: CI never needs the real encoder."""
    assert isinstance(embedder, Embedder)


def test_query_prefix_is_asymmetric() -> None:
    """bge prefixes queries only. Applying it to documents too costs recall silently."""
    assert BGE_QUERY_PREFIX.startswith("Represent this sentence")
    assert DEFAULT_MODEL == "BAAI/bge-small-en-v1.5"


def test_build_from_chunks_honours_the_breadcrumb_flag(
    sample_chunks: list, embedder: FakeEmbedder
) -> None:
    plain = build_from_chunks(sample_chunks, embedder, with_breadcrumb=False)
    crumbed = build_from_chunks(sample_chunks, embedder, with_breadcrumb=True)

    assert plain.chunk_ids == crumbed.chunk_ids
    assert not np.allclose(plain.vectors, crumbed.vectors)
