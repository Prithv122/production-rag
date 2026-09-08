"""BM25 tests.

The scoring assertions matter more than usual here: this is a hand-written
implementation, so nothing else is going to catch a wrong exponent or a flipped
IDF. `test_scores_match_a_direct_formula_evaluation` recomputes BM25 the slow,
obvious way and compares.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from production_rag.bm25 import K1, B, BM25Index, build_from_chunks, tokenize
from production_rag.chunking import Chunk

DOCS = [
    ("a", "dbt incremental models append new rows to an existing table"),
    ("b", "dagster assets declare dependencies between materializations"),
    ("c", "duckdb reads parquet files directly with read_parquet"),
    ("d", "incremental strategies in dbt include append merge and delete_insert"),
]


@pytest.fixture
def index() -> BM25Index:
    return BM25Index.build([i for i, _ in DOCS], [t for _, t in DOCS])


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
def test_tokenize_lowercases_and_drops_punctuation() -> None:
    assert tokenize("Hello, WORLD!") == ["hello", "world"]


def test_tokenize_keeps_identifiers_whole_and_in_pieces() -> None:
    """`on_schema_change` must match both the exact term and "schema change"."""
    assert tokenize("on_schema_change") == ["on_schema_change", "on", "schema", "change"]


def test_tokenize_splits_dotted_attribute_access() -> None:
    assert tokenize("dg.asset") == ["dg.asset", "dg", "asset"]


def test_tokenize_keeps_digits() -> None:
    assert tokenize("duckdb 1.2 release") == ["duckdb", "1.2", "1", "2", "release"]


def test_tokenize_empty() -> None:
    assert tokenize("   !!!  ") == []


def test_sql_keywords_are_not_stripped() -> None:
    """No stoplist: `order by` and `in` are real queries against this corpus."""
    assert tokenize("order by") == ["order", "by"]
    assert "in" in tokenize("values in a list")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def test_search_ranks_the_relevant_document_first(index: BM25Index) -> None:
    assert index.search("read_parquet", k=1)[0][0] == "c"


def test_search_finds_a_subtoken_match(index: BM25Index) -> None:
    """Query "parquet" must reach a document that only writes `read_parquet`."""
    assert "c" in [cid for cid, _ in index.search("parquet", k=4)]


def test_rare_terms_outrank_common_ones(index: BM25Index) -> None:
    """ "incremental" appears in two docs, "delete_insert" in one."""
    ranked = [cid for cid, _ in index.search("incremental delete_insert", k=4)]
    assert ranked[0] == "d"


def test_unknown_terms_score_zero(index: BM25Index) -> None:
    assert index.search("kubernetes helm chart", k=4) == []


def test_scores_are_non_negative(index: BM25Index) -> None:
    assert (index.scores("dbt incremental models") >= 0).all()


def test_scores_match_a_direct_formula_evaluation() -> None:
    """Recompute Okapi BM25 the naive way and compare, term by term."""
    ids = [i for i, _ in DOCS]
    texts = [t for _, t in DOCS]
    index = BM25Index.build(ids, texts)

    tokenised = [tokenize(t) for t in texts]
    lengths = [len(t) for t in tokenised]
    avgdl = sum(lengths) / len(lengths)
    n_docs = len(texts)

    query = "incremental dbt models"
    expected = []
    for doc_tokens, length in zip(tokenised, lengths, strict=True):
        total = 0.0
        for term in tokenize(query):
            freq = doc_tokens.count(term)
            if not freq:
                continue
            df = sum(1 for d in tokenised if term in d)
            idf = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))
            denom = freq + K1 * (1.0 - B + B * length / avgdl)
            total += idf * freq * (K1 + 1.0) / denom
        expected.append(total)

    np.testing.assert_allclose(index.scores(query), expected, rtol=1e-5)


def test_repeated_query_terms_do_not_double_count_uniquely() -> None:
    """BM25 sums per query occurrence, so a repeated term scores twice."""
    index = BM25Index.build(["a"], ["alpha beta"])
    once = index.scores("alpha")[0]
    twice = index.scores("alpha alpha")[0]
    assert twice == pytest.approx(2 * once)


# ---------------------------------------------------------------------------
# search() contract
# ---------------------------------------------------------------------------
def test_search_returns_at_most_k(index: BM25Index) -> None:
    assert len(index.search("dbt incremental dagster duckdb", k=2)) == 2


def test_search_k_larger_than_corpus_is_safe(index: BM25Index) -> None:
    assert len(index.search("incremental", k=99)) <= len(DOCS)


def test_search_k_zero(index: BM25Index) -> None:
    assert index.search("incremental", k=0) == []


def test_search_is_sorted_descending(index: BM25Index) -> None:
    scores = [s for _, s in index.search("dbt incremental append", k=4)]
    assert scores == sorted(scores, reverse=True)


def test_empty_query_returns_nothing(index: BM25Index) -> None:
    assert index.search("", k=5) == []


# ---------------------------------------------------------------------------
# Build contract
# ---------------------------------------------------------------------------
def test_build_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError, match="same length"):
        BM25Index.build(["a", "b"], ["only one"])


def test_build_rejects_an_empty_corpus() -> None:
    with pytest.raises(ValueError, match="zero chunks"):
        BM25Index.build([], [])


def test_len(index: BM25Index) -> None:
    assert len(index) == len(DOCS)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_save_load_round_trip(index: BM25Index, tmp_path: Path) -> None:
    index.save(tmp_path / "bm25")
    reloaded = BM25Index.load(tmp_path / "bm25")

    assert reloaded.chunk_ids == index.chunk_ids
    assert reloaded.vocabulary == index.vocabulary
    np.testing.assert_allclose(reloaded.scores("incremental dbt"), index.scores("incremental dbt"))


# ---------------------------------------------------------------------------
# Chunk integration
# ---------------------------------------------------------------------------
def _chunk(chunk_id: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="dbt/guide",
        tool="dbt",
        title="Guide",
        url="https://docs.getdbt.com/guide",
        heading_path=("Materializations",),
        text=text,
        start=0,
        end=len(text),
    )


def test_build_from_chunks_can_include_the_breadcrumb() -> None:
    chunks = [_chunk("dbt/guide#0", "Set this to true.")]

    without = build_from_chunks(chunks, with_breadcrumb=False)
    with_crumb = build_from_chunks(chunks, with_breadcrumb=True)

    assert without.search("materializations", k=1) == []
    assert with_crumb.search("materializations", k=1)[0][0] == "dbt/guide#0"
