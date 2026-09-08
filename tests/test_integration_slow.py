"""End-to-end checks against real artefacts.

Everything here is marked `slow` and deselected in CI's fast job: these need
either the network, the real encoder, or a built index on disk. They exist so
the paths the fast suite deliberately fakes are still exercised somewhere.

Run them locally with::

    uv sync --extra embed
    uv run production-rag ingest && uv run production-rag index
    uv run pytest -m slow
"""

from __future__ import annotations

from pathlib import Path

import pytest

from production_rag.bm25 import BM25Index
from production_rag.chunking import CHUNKERS
from production_rag.cli import load_chunks
from production_rag.dense import DenseIndex
from production_rag.fuse import fuse
from production_rag.ingest import read_jsonl
from production_rag.sources import SOURCES

pytestmark = pytest.mark.slow

DOCS = Path("corpus/normalised/docs.jsonl")
INDEXES = Path("indexes")

needs_corpus = pytest.mark.skipif(not DOCS.exists(), reason="run `production-rag ingest` first")


def _needs_index(strategy: str, arm: str = "bm25") -> pytest.MarkDecorator:
    return pytest.mark.skipif(
        not (INDEXES / strategy / arm).exists(),
        reason=f"run `production-rag index --strategy {strategy}` first",
    )


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------
@needs_corpus
def test_every_source_contributed_documents() -> None:
    tools = {doc.tool for doc in read_jsonl(DOCS)}
    assert tools == {source.name for source in SOURCES}


@needs_corpus
def test_no_duckdb_version_archive_leaked_into_the_corpus() -> None:
    """Guards the corpus decision against a careless re-ingest.

    If `docs/0.10` or `docs/lts` ever appears here, ground truth silently stops
    being well-defined -- six near-identical chunks all "answer" the question.
    """
    archived = [
        doc.path
        for doc in read_jsonl(DOCS)
        if doc.tool == "duckdb" and not doc.path.startswith("docs/current/")
    ]
    assert archived == []


@needs_corpus
def test_documents_carry_a_resolvable_url_and_title() -> None:
    for doc in read_jsonl(DOCS):
        assert doc.url.startswith("https://"), doc.doc_id
        assert doc.title.strip(), doc.doc_id


@needs_corpus
def test_normalisation_left_almost_no_markup() -> None:
    """One known residual out of 2,155 -- a literal `<>` inside an attribute.

    Asserted as a ceiling rather than zero so the number stays honest: if a
    future change makes it worse, this fails.
    """
    import re

    from production_rag.ingest import _FENCE

    def prose(text: str) -> str:
        return "".join(s for i, s in enumerate(_FENCE.split(text)) if i % 2 == 0)

    offenders = [
        doc.doc_id for doc in read_jsonl(DOCS) if re.search(r"</?[A-Z][\w.]*[ />]", prose(doc.text))
    ]
    assert len(offenders) <= 1, offenders


# ---------------------------------------------------------------------------
# Indexes
# ---------------------------------------------------------------------------
@_needs_index("heading")
def test_bm25_index_matches_its_chunk_file() -> None:
    chunks = load_chunks(INDEXES, "heading")
    index = BM25Index.load(INDEXES / "heading" / "bm25")
    assert index.chunk_ids == [c.chunk_id for c in chunks]


@_needs_index("heading")
@pytest.mark.parametrize(
    ("query", "expect_tool"),
    [
        ("on_schema_change incremental model", "dbt"),
        ("read_parquet glob multiple files", "duckdb"),
        ("asset materialization dependency", "dagster"),
    ],
)
def test_bm25_routes_a_tool_specific_query_to_that_tool(query: str, expect_tool: str) -> None:
    """A weak but real relevance check that needs no labelled ground truth."""
    chunks = {c.chunk_id: c for c in load_chunks(INDEXES, "heading")}
    index = BM25Index.load(INDEXES / "heading" / "bm25")

    top = index.search(query, k=5)

    assert top, query
    assert any(chunks[cid].tool == expect_tool for cid, _ in top), query


@_needs_index("heading", "dense")
def test_dense_index_is_unit_norm_and_aligned() -> None:
    import numpy as np

    chunks = load_chunks(INDEXES, "heading")
    index = DenseIndex.load(INDEXES / "heading" / "dense")

    assert index.chunk_ids == [c.chunk_id for c in chunks]
    norms = np.linalg.norm(index.vectors, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-4)


@_needs_index("heading", "dense")
def test_hybrid_surfaces_candidates_from_both_arms() -> None:
    """The point of running two arms: fusion must not collapse to one of them."""
    from production_rag.dense import SentenceTransformerEmbedder

    bm25 = BM25Index.load(INDEXES / "heading" / "bm25")
    dense = DenseIndex.load(INDEXES / "heading" / "dense")
    embedder = SentenceTransformerEmbedder(dense.model_name)

    query = "how do I avoid rebuilding the whole table on every run"
    lexical = bm25.search(query, k=50)
    semantic = dense.search(query, embedder, k=50)
    hybrid = fuse([lexical, semantic], "rrf", k=10)

    hybrid_ids = {cid for cid, _ in hybrid}
    assert hybrid_ids & {cid for cid, _ in lexical[:20]}
    assert hybrid_ids & {cid for cid, _ in semantic[:20]}


@pytest.mark.parametrize("strategy", sorted(CHUNKERS))
def test_built_chunk_files_respect_the_budget(strategy: str) -> None:
    """Budget is `max_chars + overlap_chars + 2`.

    The `+ 2` is the `\\n\\n` that joins the carried-over overlap tail to the
    body -- a real part of the emitted chunk, so it is stated here rather than
    hidden by rounding the bound up.
    """
    path = INDEXES / strategy / "chunks.jsonl"
    if not path.exists():
        pytest.skip(f"run `production-rag index --strategy {strategy}` first")
    assert max(len(c.text) for c in load_chunks(INDEXES, strategy)) <= 1200 + 200 + 2
