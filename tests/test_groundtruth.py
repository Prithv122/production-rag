from __future__ import annotations

from conftest import make_chunk
from production_rag.bm25 import build_from_chunks as build_bm25
from production_rag.groundtruth import (
    Evidence,
    ProposalStats,
    Question,
    assign_categories,
    assign_ids,
    categorise,
    coverage_report,
    document_frequencies,
    gold_chunk_ids,
    index_chunks_by_doc,
    load_questions,
    locate_quote,
    rare_shared_terms,
    sample_passages,
    save_questions,
    screen_proposals,
)
from production_rag.ingest import Document

DOC_TEXT = (
    "Incremental models\n\n"
    "Set on_schema_change to append_new_columns when the source gains a column.\n\n"
    "Otherwise the run fails. Set on_schema_change to fail for strict pipelines.\n"
)


def make_doc(text: str = DOC_TEXT, doc_id: str = "dbt/incremental") -> Document:
    return Document(
        doc_id=doc_id,
        tool=doc_id.split("/")[0],
        path=f"{doc_id}.md",
        title="Incremental models",
        url=f"https://example.test/{doc_id}",
        text=text,
    )


def spanned(doc: Document, needle: str) -> Evidence:
    start = doc.text.index(needle)
    return Evidence(doc.doc_id, start, start + len(needle), needle)


# ---------------------------------------------------------------------------
# locate_quote
# ---------------------------------------------------------------------------
def test_exact_quote_is_located():
    doc = make_doc()
    quote = "Set on_schema_change to append_new_columns"
    assert locate_quote(doc.text, quote) == (
        doc.text.index(quote),
        doc.text.index(quote) + len(quote),
    )


def test_a_quote_reflowed_across_a_line_break_still_matches():
    """A model reproducing a quote without the source's line wrap is still
    quoting; anything looser would accept an invented quote too."""
    doc = make_doc("alpha beta\ngamma delta epsilon zeta eta theta")
    assert locate_quote(doc.text, "beta gamma delta epsilon zeta") is not None


def test_an_invented_quote_is_rejected():
    assert locate_quote(DOC_TEXT, "Set on_schema_change to merge_all_columns instead") is None


def test_a_quote_too_short_to_identify_a_span_is_rejected():
    assert locate_quote(DOC_TEXT, "fail") is None


def test_the_hint_picks_the_nearer_of_two_identical_quotes():
    repeated = "boilerplate warning sentence that repeats. " * 3
    text = "A" * 500 + repeated + "B" * 500 + repeated
    quote = "boilerplate warning sentence that repeats."
    near_end = locate_quote(text, quote, hint=(1200, 1300))
    near_start = locate_quote(text, quote, hint=(500, 520))
    assert near_end is not None and near_start is not None
    assert near_end[0] > near_start[0]


# ---------------------------------------------------------------------------
# spans -> gold chunks, the cross-strategy contract
# ---------------------------------------------------------------------------
def _question(doc: Document, needle: str, category: str = "exact_term") -> Question:
    return Question(
        qid="q1",
        text="how do I handle a new column?",
        category=category,
        evidence=(spanned(doc, needle),),
        tools=(doc.tool,),
    )


def test_a_chunk_covering_the_span_is_gold():
    doc = make_doc()
    chunk = make_chunk("dbt/incremental#0", DOC_TEXT)
    chunk = type(chunk)(
        **{
            **chunk.to_dict(),
            "heading_path": (),
            "doc_id": doc.doc_id,
            "start": 0,
            "end": len(DOC_TEXT),
        }
    )
    by_doc = index_chunks_by_doc([chunk])
    assert gold_chunk_ids(_question(doc, "Otherwise the run fails."), by_doc) == {chunk.chunk_id}


def test_the_same_label_maps_onto_two_different_chunkings():
    """The reason gold is a span and not a chunk id: one label, two strategies."""
    doc = make_doc()
    question = _question(doc, "Set on_schema_change to append_new_columns")

    coarse = [_chunk_at(doc, "coarse#0", 0, len(DOC_TEXT))]
    fine = [
        _chunk_at(doc, "fine#0", 0, 20),
        _chunk_at(doc, "fine#1", 20, 95),
        _chunk_at(doc, "fine#2", 95, len(DOC_TEXT)),
    ]
    assert gold_chunk_ids(question, index_chunks_by_doc(coarse)) == {"coarse#0"}
    assert gold_chunk_ids(question, index_chunks_by_doc(fine)) == {"fine#1"}


def test_a_chunk_that_merely_clips_the_span_is_not_gold():
    doc = make_doc()
    question = _question(doc, "Set on_schema_change to append_new_columns when the source gains")
    start = question.evidence[0].start
    clipping = _chunk_at(doc, "clip#0", 0, start + 5)
    covering = _chunk_at(doc, "cover#0", start, question.evidence[0].end)
    gold = gold_chunk_ids(question, index_chunks_by_doc([clipping, covering]))
    assert gold == {"cover#0"}


def test_a_question_is_never_left_unscoreable():
    """Falling back to the best-overlapping chunk keeps every strategy scored on
    the same questions -- otherwise their means are not comparable."""
    doc = make_doc()
    question = _question(doc, "Set on_schema_change to append_new_columns when the source gains")
    start, end = question.evidence[0].start, question.evidence[0].end
    halves = [
        _chunk_at(doc, "half#0", 0, start + (end - start) // 3),
        _chunk_at(doc, "half#1", start + (end - start) // 3 + 1, len(DOC_TEXT)),
    ]
    assert len(gold_chunk_ids(question, index_chunks_by_doc(halves))) >= 1


def test_coverage_report_counts_mapped_questions():
    doc = make_doc()
    question = _question(doc, "Otherwise the run fails.")
    by_doc = index_chunks_by_doc([_chunk_at(doc, "c#0", 0, len(DOC_TEXT))])
    report = coverage_report([question], by_doc)
    assert report["questions"] == 1.0 and report["mapped"] == 1.0


def test_coverage_notices_a_document_with_no_chunks_at_all():
    doc = make_doc()
    report = coverage_report([_question(doc, "Otherwise the run fails.")], {})
    assert report["mapped"] == 0.0


def _chunk_at(doc: Document, chunk_id: str, start: int, end: int):
    chunk = make_chunk(chunk_id, doc.text[start:end])
    return type(chunk)(
        **{
            **chunk.to_dict(),
            "heading_path": (),
            "doc_id": doc.doc_id,
            "tool": doc.tool,
            "start": start,
            "end": end,
        }
    )


# ---------------------------------------------------------------------------
# category, computed rather than claimed
# ---------------------------------------------------------------------------
def test_document_frequency_matches_a_hand_count():
    chunks = [
        make_chunk("a#0", "read_parquet is here"),
        make_chunk("b#0", "read_parquet again and again"),
        make_chunk("c#0", "nothing relevant"),
    ]
    df = document_frequencies(build_bm25(chunks))
    assert df["read_parquet"] == 2
    assert df["nothing"] == 1


def test_a_shared_rare_identifier_makes_it_an_exact_term_question():
    # The tokenizer emits the identifier *and* its parts; only the parts that
    # are themselves rare in the corpus count, which on the real corpus excludes
    # "schema" and "change" (they appear in thousands of chunks).
    df = {"on_schema_change": 4, "how": 900, "do": 900, "schema": 4000, "change": 3000}
    assert rare_shared_terms(
        "how do I use on_schema_change", "set on_schema_change to append", df
    ) == {"on_schema_change"}


def test_a_common_shared_word_does_not():
    df = {"how": 900, "column": 900}
    assert categorise("how do I add a column", "adding a column is easy", df) == "conceptual"


def test_a_rare_term_the_gold_passage_does_not_contain_does_not_count():
    """The question must share the term *with its own evidence* -- otherwise a
    rare word from an unrelated tool would mislabel the category."""
    df = {"read_parquet": 3}
    assert categorise("how does read_parquet work", "dbt models are SQL files", df) == "conceptual"


def test_multiple_tools_make_it_cross_tool():
    assert categorise("q", "text", {}, tools=("dbt", "dagster")) == "cross_tool"


def test_no_evidence_makes_it_unanswerable():
    assert categorise("q", "", {}, answerable=False) == "unanswerable"


# ---------------------------------------------------------------------------
# screening
# ---------------------------------------------------------------------------
def _screen(proposals, stats=None, seen=None):
    doc = make_doc()
    chunk = _chunk_at(doc, "dbt/incremental#0", 0, len(DOC_TEXT))
    return (
        screen_proposals(
            proposals,
            chunk,
            doc,
            model="test-model",
            seen=seen if seen is not None else set(),
            stats=stats or ProposalStats(),
        ),
        doc,
    )


def test_a_good_proposal_is_accepted_with_a_resolved_span():
    accepted, doc = _screen(
        [
            {
                "question": "What value of on_schema_change adds new columns?",
                "style": "literal",
                "quote": "Set on_schema_change to append_new_columns",
            }
        ]
    )
    assert len(accepted) == 1
    evidence = accepted[0].evidence[0]
    assert doc.text[evidence.start : evidence.end] == "Set on_schema_change to append_new_columns"
    assert accepted[0].proposed_category == "exact_term"


def test_a_hallucinated_quote_is_rejected_and_counted():
    stats = ProposalStats()
    accepted, _ = _screen(
        [
            {
                "question": "What is the default behaviour here for schema drift?",
                "quote": "Set on_schema_change to sync_all_columns_quietly",
            }
        ],
        stats,
    )
    assert accepted == [] and stats.rejected_no_quote == 1


def test_a_question_about_the_passage_is_rejected():
    stats = ProposalStats()
    accepted, _ = _screen(
        [
            {
                "question": "According to this passage, what does on_schema_change do?",
                "quote": "Set on_schema_change to append_new_columns",
            }
        ],
        stats,
    )
    assert accepted == [] and stats.rejected_context_dependent == 1


def test_a_too_short_question_is_rejected():
    stats = ProposalStats()
    _screen([{"question": "what?", "quote": "Set on_schema_change to append_new_columns"}], stats)
    assert stats.rejected_too_short == 1


def test_a_duplicate_question_is_rejected_across_passages():
    stats = ProposalStats()
    seen: set[str] = set()
    proposal = [
        {
            "question": "What value of on_schema_change adds new columns?",
            "quote": "Set on_schema_change to append_new_columns",
        }
    ]
    _screen(proposal, stats, seen)
    _screen(proposal, stats, seen)
    assert stats.accepted == 1 and stats.rejected_duplicate == 1


def test_every_rejection_is_counted_so_the_yield_can_be_reported():
    stats = ProposalStats()
    _screen(
        [
            {"question": "short", "quote": "x"},
            {
                "question": "What value of on_schema_change adds new columns here?",
                "quote": "invented text that is definitely not present",
            },
        ],
        stats,
    )
    assert stats.returned == 2
    assert stats.rejected_too_short + stats.rejected_no_quote == 2


# ---------------------------------------------------------------------------
# assembly and persistence
# ---------------------------------------------------------------------------
def test_categories_and_ids_are_assigned_from_the_corpus():
    doc = make_doc()
    chunks = [make_chunk("x#0", DOC_TEXT)]
    df = document_frequencies(build_bm25(chunks))
    questions = assign_ids(
        assign_categories(
            [
                Question(
                    "",
                    "how do I set on_schema_change?",
                    "",
                    (spanned(doc, "Set on_schema_change to append_new_columns"),),
                    ("dbt",),
                ),
                Question(
                    "",
                    "what happens when a new column appears upstream?",
                    "",
                    (spanned(doc, "Otherwise the run fails."),),
                    ("dbt",),
                ),
            ],
            {doc.doc_id: doc},
            df,
        )
    )
    by_text = {q.text: q for q in questions}
    assert by_text["how do I set on_schema_change?"].category == "exact_term"
    assert by_text["what happens when a new column appears upstream?"].category == "conceptual"
    assert all(q.qid.startswith("q00") for q in questions)


def test_questions_round_trip_through_jsonl(tmp_path):
    doc = make_doc()
    original = [
        Question(
            "q1",
            "text",
            "exact_term",
            (spanned(doc, "Otherwise the run fails."),),
            ("dbt",),
            source="hand",
            verified="accepted",
        )
    ]
    path = tmp_path / "questions.jsonl"
    save_questions(original, path)
    assert load_questions(path) == original


def test_passage_sampling_is_stratified_and_deterministic():
    chunks = [make_chunk(f"dbt/{i}#0", "x" * 500, tool="dbt") for i in range(50)]
    chunks += [make_chunk(f"duckdb/{i}#0", "y" * 500, tool="duckdb") for i in range(5)]
    picked = sample_passages(chunks, n=6)
    assert sample_passages(chunks, n=6) == picked
    assert {c.tool for c in picked} == {"dbt", "duckdb"}


def test_passages_outside_the_length_band_are_not_sampled():
    chunks = [make_chunk("a#0", "tiny"), make_chunk("b#0", "z" * 500)]
    assert [c.chunk_id for c in sample_passages(chunks, n=5)] == ["b#0"]


# ---------------------------------------------------------------------------
# hand-written questions
# ---------------------------------------------------------------------------
def _write_hand(tmp_path, rows):
    import json as _json

    path = tmp_path / "handwritten.jsonl"
    path.write_text(
        "\n".join(_json.dumps(row) for row in rows) + "\n", encoding="utf-8", newline="\n"
    )
    return path


def test_hand_written_quotes_are_resolved_to_spans(tmp_path):
    from production_rag.groundtruth import load_hand_written

    doc = make_doc()
    path = _write_hand(
        tmp_path,
        [
            {
                "text": "how do I cope with a new upstream column?",
                "tools": ["dbt"],
                "evidence": [
                    {"doc_id": doc.doc_id, "quote": "Set on_schema_change to append_new_columns"}
                ],
            }
        ],
    )
    [question] = load_hand_written(path, {doc.doc_id: doc})
    evidence = question.evidence[0]
    assert doc.text[evidence.start : evidence.end] == "Set on_schema_change to append_new_columns"
    assert question.source == "hand"


def test_an_unanswerable_hand_written_question_carries_no_evidence(tmp_path):
    from production_rag.groundtruth import load_hand_written

    path = _write_hand(tmp_path, [{"text": "how do I tune Kafka retention?", "evidence": []}])
    [question] = load_hand_written(path, {})
    assert question.evidence == () and question.is_answerable is False


def test_a_stale_hand_written_quote_raises_rather_than_vanishing(tmp_path):
    """Dropping it silently would shrink the cross-tool bucket without anyone
    noticing, and that bucket is small enough that one question moves its mean."""
    import pytest as _pytest

    from production_rag.groundtruth import load_hand_written

    doc = make_doc()
    path = _write_hand(
        tmp_path,
        [
            {
                "text": "q",
                "evidence": [{"doc_id": doc.doc_id, "quote": "text that was removed upstream"}],
            }
        ],
    )
    with _pytest.raises(ValueError, match="quote not found"):
        load_hand_written(path, {doc.doc_id: doc})


def test_an_unknown_doc_id_raises(tmp_path):
    import pytest as _pytest

    from production_rag.groundtruth import load_hand_written

    path = _write_hand(
        tmp_path, [{"text": "q", "evidence": [{"doc_id": "gone", "quote": "x" * 30}]}]
    )
    with _pytest.raises(ValueError, match="unknown doc_id"):
        load_hand_written(path, {})
