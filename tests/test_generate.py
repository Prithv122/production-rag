"""Generation, citation validation and the refusal path.

No network and no model: :class:`FakeProvider` returns scripted JSON, which is
the right level for these tests. What is under test is not whether an LLM
writes good prose -- nothing offline can assert that -- but whether the code
around it correctly refuses to trust the model: that markers are resolved
against the passages actually shown, that a fabricated citation is recorded
rather than rendered, and that a response the parser cannot read becomes a
refusal instead of unvalidated text in the answer slot.
"""

from __future__ import annotations

import json

import pytest

from conftest import FakeProvider, make_chunk
from production_rag.generate import (
    DEFAULT_CONTEXT_CHARS,
    MAX_PASSAGE_CHARS,
    REFUSAL_TEXT,
    Answer,
    build_passages,
    build_prompt,
    format_answer,
    generate,
    parse_citations,
)


def payload(answer: str, *, sufficient: bool = True) -> str:
    return json.dumps({"sufficient": sufficient, "answer": answer})


@pytest.fixture
def chunks(sample_chunks):
    return {c.chunk_id: c for c in sample_chunks}


@pytest.fixture
def ranked(sample_chunks):
    return [c.chunk_id for c in sample_chunks]


# ---------------------------------------------------------------------------
# context construction
# ---------------------------------------------------------------------------
def test_passages_are_numbered_in_retrieval_order(chunks, ranked):
    passages, dropped = build_passages(ranked, chunks)
    assert dropped == 0
    assert [p.number for p in passages] == [1, 2, 3, 4]
    assert [p.chunk_id for p in passages] == ranked


def test_unknown_chunk_ids_are_dropped_not_renumbered(chunks, ranked):
    passages, dropped = build_passages(["nope#0", *ranked], chunks)
    assert dropped == 1
    # The missing id must not leave a gap: the model is told 1..N exist.
    assert [p.number for p in passages] == [1, 2, 3, 4]


def test_context_budget_drops_whole_passages(chunks):
    big = {f"doc#{i}": make_chunk(f"doc#{i}", "x" * 900) for i in range(30)}
    passages, dropped = build_passages(list(big), big, max_chars=2000)
    assert dropped > 0
    assert sum(len(p.text) for p in passages) <= 2000 + 900
    assert all(p.text.endswith("x") for p in passages)


def test_a_single_huge_passage_is_truncated_and_marked():
    huge = {"doc#0": make_chunk("doc#0", "y" * (MAX_PASSAGE_CHARS + 500))}
    passages, _ = build_passages(["doc#0"], huge)
    assert passages[0].truncated
    assert len(passages[0].text) == MAX_PASSAGE_CHARS
    assert "truncated" in passages[0].render()


def test_the_first_passage_is_kept_even_when_it_blows_the_budget():
    huge = {"doc#0": make_chunk("doc#0", "y" * 5_000)}
    passages, dropped = build_passages(["doc#0"], huge, max_chars=100)
    assert len(passages) == 1 and dropped == 0


def test_prompt_carries_every_passage_and_the_refusal_sentence(chunks, ranked):
    passages, _ = build_passages(ranked, chunks)
    prompt = build_prompt("how do incremental models work?", passages)
    for passage in passages:
        assert passage.url in prompt
    assert REFUSAL_TEXT in prompt
    assert "how do incremental models work?" in prompt


# ---------------------------------------------------------------------------
# citation parsing
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("a [1] b", [1]),
        ("a [1][2] b", [1, 2]),
        ("a [1, 2] b", [1, 2]),
        ("a [1;3] b", [1, 3]),
        ("no markers here", []),
        ("an array index like x[0] is still read as a marker", [0]),
    ],
)
def test_marker_forms(chunks, ranked, text, expected):
    passages, _ = build_passages(ranked, chunks)
    assert [c.marker for c in parse_citations(text, passages)] == expected


def test_a_marker_outside_the_passage_range_is_invalid(chunks, ranked):
    passages, _ = build_passages(ranked, chunks)
    citations = parse_citations("grounded [1], fabricated [9]", passages)
    assert [c.valid for c in citations] == [True, False]
    assert citations[0].chunk_id == ranked[0]
    assert citations[1].chunk_id == ""


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------
def test_a_cited_answer_resolves_to_the_chunk_it_was_shown(chunks, ranked):
    provider = FakeProvider([payload("dbt appends rows [1].")])
    answer = generate("q", ranked, chunks, provider)
    assert not answer.refused
    assert answer.cited_chunk_ids == [ranked[0]]
    assert answer.citations[0].url.endswith(ranked[0])
    assert not answer.uncited


def test_duplicate_citations_collapse_but_stay_in_the_raw_list(chunks, ranked):
    provider = FakeProvider([payload("one [1]. two [1]. three [2].")])
    answer = generate("q", ranked, chunks, provider)
    assert answer.cited_chunk_ids == [ranked[0], ranked[1]]
    assert len(answer.citations) == 3


def test_an_answer_with_no_citations_is_flagged_ungrounded(chunks, ranked):
    provider = FakeProvider([payload("I simply know this.")])
    answer = generate("q", ranked, chunks, provider)
    assert answer.uncited and not answer.refused
    assert "ungrounded" in format_answer(answer)


def test_a_fabricated_citation_is_recorded_and_scrubbed_from_the_display(chunks, ranked):
    provider = FakeProvider([payload("real [2] and invented [12].")])
    answer = generate("q", ranked, chunks, provider)
    assert [c.marker for c in answer.invalid_citations] == [12]
    assert "[12]" in answer.text  # the evidence is preserved
    assert "[12]" not in answer.clean_text  # ...but never shown
    assert "[2]" in answer.clean_text


def test_scrubbing_a_marker_does_not_leave_a_space_before_the_full_stop(chunks, ranked):
    provider = FakeProvider([payload("a claim [9].")])
    answer = generate("q", ranked, chunks, provider)
    assert answer.clean_text == "a claim."


def test_a_refusal_is_not_counted_as_ungrounded(chunks, ranked):
    provider = FakeProvider([payload(REFUSAL_TEXT, sufficient=False)])
    answer = generate("q", ranked, chunks, provider)
    assert answer.refused and answer.refusal_reason == "model"
    assert not answer.uncited


def test_refusing_with_an_empty_answer_still_produces_the_sentence(chunks, ranked):
    provider = FakeProvider([payload("", sufficient=False)])
    answer = generate("q", ranked, chunks, provider)
    assert answer.text == REFUSAL_TEXT


def test_no_retrieved_context_refuses_without_calling_the_model(chunks):
    provider = FakeProvider([payload("should never be used")])
    answer = generate("q", [], chunks, provider)
    assert answer.refused and answer.refusal_reason == "no_context"
    assert provider.prompts == []


def test_the_score_gate_is_off_unless_a_threshold_is_given(chunks, ranked):
    provider = FakeProvider([payload("answered anyway [1].")])
    answer = generate("q", ranked, chunks, provider, top_score=0.0001)
    assert not answer.refused


def test_the_score_gate_refuses_before_paying_for_a_generation(chunks, ranked):
    provider = FakeProvider([payload("should never be used")])
    answer = generate("q", ranked, chunks, provider, top_score=0.4, min_top_score=1.0)
    assert answer.refused and answer.refusal_reason == "low_score"
    assert provider.prompts == []
    assert answer.passages, "the gate still records what was retrieved"


def test_an_unparseable_response_refuses_rather_than_showing_raw_text(chunks, ranked):
    provider = FakeProvider(["I refuse to emit JSON, here is prose instead."])
    answer = generate("q", ranked, chunks, provider)
    assert answer.refused and answer.refusal_reason == "unparseable"
    assert answer.text == REFUSAL_TEXT
    assert "prose instead" in answer.raw  # kept for the failure count
    assert answer.error


def test_json_missing_the_answer_key_is_unparseable(chunks, ranked):
    provider = FakeProvider(['{"sufficient": true}'])
    answer = generate("q", ranked, chunks, provider)
    assert answer.refusal_reason == "unparseable"


def test_json_wrapped_in_a_fence_is_still_read(chunks, ranked):
    provider = FakeProvider(['```json\n{"sufficient": true, "answer": "fenced [1]."}\n```'])
    answer = generate("q", ranked, chunks, provider)
    assert not answer.refused and answer.cited_chunk_ids == [ranked[0]]


def test_a_provider_failure_refuses_and_keeps_the_reason(chunks, ranked):
    answer = generate("q", ranked, chunks, FakeProvider(fail=True))
    assert answer.refused and answer.refusal_reason == "provider_error"
    assert "is down" in answer.error


def test_as_dict_round_trips_through_json(chunks, ranked):
    provider = FakeProvider([payload("grounded [1].")])
    answer = generate("q", ranked, chunks, provider)
    assert json.loads(json.dumps(answer.as_dict()))["context_ids"] == ranked


def test_format_lists_only_the_cited_sources(chunks, ranked):
    provider = FakeProvider([payload("only the third one [3].")])
    rendered = format_answer(generate("q", ranked, chunks, provider))
    assert ranked[2] in rendered
    assert ranked[1] not in rendered


def test_default_context_budget_is_not_accidentally_tiny():
    assert DEFAULT_CONTEXT_CHARS > MAX_PASSAGE_CHARS
    assert isinstance(Answer("q", "t", False).context_ids, list)


# ---------------------------------------------------------------------------
# markdown rendering (used by the demo, tested here so the demo has no logic)
# ---------------------------------------------------------------------------
def test_link_citations_points_each_marker_at_its_passage(chunks, ranked):
    from production_rag.generate import link_citations

    provider = FakeProvider([payload("first [1] and second [2].")])
    answer = generate("q", ranked, chunks, provider)
    rendered = link_citations(answer)
    assert f"[[1]](https://example.test/{ranked[0]})" in rendered
    assert f"[[2]](https://example.test/{ranked[1]})" in rendered


def test_link_citations_drops_a_fabricated_marker_and_tidies_the_gap(chunks, ranked):
    from production_rag.generate import link_citations

    provider = FakeProvider([payload("a claim [11].")])
    answer = generate("q", ranked, chunks, provider)
    assert link_citations(answer) == "a claim."


def test_link_citations_expands_a_grouped_marker(chunks, ranked):
    from production_rag.generate import link_citations

    provider = FakeProvider([payload("both [1, 2] agree.")])
    rendered = link_citations(generate("q", ranked, chunks, provider))
    assert rendered.count("]](") == 2


def test_markdown_sources_lists_the_cited_and_flags_the_fabricated(chunks, ranked):
    from production_rag.generate import markdown_sources

    provider = FakeProvider([payload("real [3], invented [42].")])
    rendered = markdown_sources(generate("q", ranked, chunks, provider))
    assert ranked[2] in rendered
    assert ranked[0] not in rendered
    assert "dropped 1 citation" in rendered


def test_markdown_sources_is_empty_for_an_uncited_answer(chunks, ranked):
    from production_rag.generate import markdown_sources

    provider = FakeProvider([payload("no markers at all.")])
    assert markdown_sources(generate("q", ranked, chunks, provider)) == ""
