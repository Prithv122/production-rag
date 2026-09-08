"""Chunking tests. Pure string work -- no corpus, no network, no model."""

from __future__ import annotations

import itertools

import pytest

from production_rag.chunking import (
    CHUNKERS,
    USES_BREADCRUMB,
    Chunk,
    _overlap_tail,
    chunk_corpus,
    chunk_fixed,
    chunk_heading,
    chunk_heading_ctx,
    split_blocks,
)
from production_rag.ingest import Document

FENCE = "```"


def make_doc(text: str, doc_id: str = "dbt/guide") -> Document:
    return Document(
        doc_id=doc_id,
        tool="dbt",
        path="website/docs/guide.md",
        title="Guide",
        url="https://docs.getdbt.com/guide",
        text=text,
    )


# ---------------------------------------------------------------------------
# Block splitting
# ---------------------------------------------------------------------------
def test_split_blocks_separates_prose_and_code() -> None:
    doc = f"Intro paragraph.\n\n{FENCE}sql\nSELECT 1;\n{FENCE}\n\nOutro paragraph.\n"
    kinds = [b.kind for b in split_blocks(doc)]
    assert kinds == ["prose", "code", "prose"]


def test_split_blocks_tracks_heading_hierarchy() -> None:
    doc = "# Top\n\nA.\n\n## Middle\n\nB.\n\n### Deep\n\nC.\n"
    paths = [b.heading_path for b in split_blocks(doc)]
    assert paths == [("Top",), ("Top", "Middle"), ("Top", "Middle", "Deep")]


def test_split_blocks_pops_the_stack_on_a_sibling_heading() -> None:
    doc = "# Top\n\n## A\n\nfirst.\n\n## B\n\nsecond.\n"
    paths = [b.heading_path for b in split_blocks(doc)]
    assert paths == [("Top", "A"), ("Top", "B")]


def test_split_blocks_pops_multiple_levels() -> None:
    doc = "# Top\n\n## A\n\n### Deep\n\nx.\n\n## B\n\ny.\n"
    assert [b.heading_path for b in split_blocks(doc)] == [
        ("Top", "A", "Deep"),
        ("Top", "B"),
    ]


def test_split_blocks_offsets_point_into_the_source_text() -> None:
    text = "First para.\n\nSecond para.\n"
    blocks = split_blocks(text)
    for block in blocks:
        assert text[block.start : block.end].strip() == block.text


def test_a_heading_inside_a_code_fence_is_not_a_heading() -> None:
    """`# comment` in a shell example must not restructure the document."""
    doc = f"# Real\n\n{FENCE}bash\n# just a comment\ndbt run\n{FENCE}\n\nAfter.\n"
    blocks = split_blocks(doc)
    assert [b.heading_path for b in blocks] == [("Real",), ("Real",)]
    assert "# just a comment" in blocks[0].text


# ---------------------------------------------------------------------------
# Shared guarantees
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("strategy", sorted(CHUNKERS))
def test_every_strategy_keeps_code_fences_whole(strategy: str) -> None:
    code = "\n".join(f"line_{i} = {i}" for i in range(12))
    doc = make_doc(f"Intro.\n\n{FENCE}python\n{code}\n{FENCE}\n\nOutro.\n")

    chunks = CHUNKERS[strategy](doc, max_chars=4000, overlap_chars=100)
    holders = [c for c in chunks if "line_0" in c.text]

    assert len(holders) == 1
    assert "line_11" in holders[0].text


@pytest.mark.parametrize("strategy", sorted(CHUNKERS))
def test_every_strategy_produces_non_empty_chunks(strategy: str) -> None:
    doc = make_doc("# H\n\n" + "Sentence. " * 400)
    chunks = CHUNKERS[strategy](doc, max_chars=500, overlap_chars=50)
    assert chunks
    assert all(c.text.strip() for c in chunks)


@pytest.mark.parametrize("strategy", sorted(CHUNKERS))
def test_chunk_ids_are_unique_and_ordered(strategy: str) -> None:
    doc = make_doc("# H\n\n" + "Sentence. " * 300)
    chunks = CHUNKERS[strategy](doc, max_chars=400, overlap_chars=40)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))
    assert ids == [f"{doc.doc_id}#{i}" for i in range(len(chunks))]


@pytest.mark.parametrize("strategy", sorted(CHUNKERS))
def test_empty_document_yields_no_chunks(strategy: str) -> None:
    assert CHUNKERS[strategy](make_doc("")) == []


# ---------------------------------------------------------------------------
# Strategy-specific behaviour
# ---------------------------------------------------------------------------
def test_heading_starts_a_new_chunk_at_each_section() -> None:
    doc = make_doc("# Top\n\n## Alpha\n\nAlpha body.\n\n## Beta\n\nBeta body.\n")
    chunks = chunk_heading(doc, max_chars=4000)

    assert len(chunks) == 2
    assert chunks[0].heading_path == ("Top", "Alpha")
    assert chunks[1].heading_path == ("Top", "Beta")
    assert "Beta body." not in chunks[0].text


def test_fixed_ignores_headings_and_packs_across_sections() -> None:
    doc = make_doc("# Top\n\n## Alpha\n\nAlpha body.\n\n## Beta\n\nBeta body.\n")
    chunks = chunk_fixed(doc, max_chars=4000, overlap_chars=0)

    assert len(chunks) == 1
    assert "Alpha body." in chunks[0].text
    assert "Beta body." in chunks[0].text


def test_heading_has_no_overlap_between_chunks() -> None:
    doc = make_doc("# Top\n\n## A\n\n" + "alpha " * 200 + "\n\n## B\n\nbeta body.\n")
    chunks = chunk_heading(doc, max_chars=400)
    assert "alpha" not in chunks[-1].text


def test_heading_ctx_carries_an_overlap_tail_forward() -> None:
    words = " ".join(f"w{i}" for i in range(400))
    chunks = chunk_heading_ctx(make_doc(f"# Top\n\n{words}"), max_chars=300, overlap_chars=120)

    assert len(chunks) > 1
    for earlier, later in itertools.pairwise(chunks):
        assert later.text.startswith(_overlap_tail(earlier.text, 120))


def test_only_heading_ctx_prefixes_the_breadcrumb() -> None:
    assert USES_BREADCRUMB == {"fixed": False, "heading": False, "heading_ctx": True}


def test_breadcrumb_joins_title_and_headings() -> None:
    doc = make_doc("# Top\n\n## Section\n\nBody.\n")
    chunk = chunk_heading(doc)[0]
    assert chunk.breadcrumb == "Guide > Top > Section"
    assert chunk.embed_text(with_breadcrumb=True).startswith("Guide > Top > Section\n\n")
    assert chunk.embed_text(with_breadcrumb=False) == chunk.text


# ---------------------------------------------------------------------------
# Overlap tail
# ---------------------------------------------------------------------------
def test_overlap_tail_advances_to_a_word_boundary() -> None:
    """A raw slice would start the next chunk mid-word."""
    assert _overlap_tail("alpha beta gamma delta", 12) == "gamma delta"


def test_overlap_tail_of_zero_is_empty() -> None:
    assert _overlap_tail("anything at all", 0) == ""


def test_overlap_tail_shorter_than_the_text_is_returned_whole() -> None:
    assert _overlap_tail("short", 100) == "short"


def test_overlap_is_charged_against_the_budget() -> None:
    """The tail is prepended to the next chunk, so it must count towards it.

    Omitting it pushed 49% of `fixed` chunks over `max_chars` and past the
    encoder's context window.
    """
    doc = make_doc("# H\n\n" + "word " * 2000)
    chunks = chunk_fixed(doc, max_chars=400, overlap_chars=150)
    # The +2 is the blank-line separator joining the carried tail to the body.
    assert max(len(c.text) for c in chunks) <= 400 + 150 + 2


def test_a_single_unwrapped_paragraph_still_respects_the_budget() -> None:
    """One physical line holding a whole paragraph must still be split.

    Line-boundary splitting alone produced a single 9,999-character chunk here.
    """
    doc = make_doc("# H\n\n" + "word " * 2000)
    chunks = chunk_heading(doc, max_chars=400)
    assert len(chunks) > 20
    assert max(len(c.text) for c in chunks) <= 400


# ---------------------------------------------------------------------------
# Corpus-level helper
# ---------------------------------------------------------------------------
def test_chunk_corpus_spans_documents() -> None:
    docs = [make_doc("# A\n\nfirst.\n", "dbt/a"), make_doc("# B\n\nsecond.\n", "dbt/b")]
    chunks = chunk_corpus(docs, "heading")
    assert {c.doc_id for c in chunks} == {"dbt/a", "dbt/b"}


def test_chunk_corpus_rejects_an_unknown_strategy() -> None:
    with pytest.raises(ValueError, match="unknown chunking strategy"):
        chunk_corpus([], "semantic-magic")


def test_chunk_round_trips_through_a_dict() -> None:
    chunk = chunk_heading(make_doc("# Top\n\n## S\n\nBody.\n"))[0]
    assert Chunk.from_dict(chunk.to_dict()) == chunk
