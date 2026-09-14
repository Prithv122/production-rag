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
from production_rag.cache import JsonCache
from production_rag.generate import (
    ANSWER_SYSTEM,
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
from production_rag.providers import CachedProvider, LLMResponse


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


# ---------------------------------------------------------------------------
# the budget retry
#
# What these cover is one defect wearing three costumes. `max_tokens` on a
# reasoning model is not an answer budget -- it is shared with the thinking
# trace, and the trace goes first. Measured against the deployed image,
# nemotron-3-super spent 504-857 completion tokens reasoning, so a 700-token
# budget was gone before the JSON started and came back as `finish_reason=
# 'length'` carrying either a truncated object, the reasoning trace echoed into
# `content`, or a bare `{}`. Earlier revisions of this file asserted that
# raising max_tokens "does not help" and that the remedy was to drop the format
# constraint; a controlled pair of live calls -- same question, same constraint,
# 700 tokens gives `{}` and 2048 gives a cited answer -- says otherwise.
# ---------------------------------------------------------------------------
def _truncated(text: str = "") -> LLMResponse:
    return LLMResponse(text=text, model="fake-model", provider="fake", finish_reason="length")


def test_a_truncated_response_is_retried_with_a_wider_budget(chunks, ranked):
    provider = FakeProvider(
        [
            _truncated('{"sufficient": true, "answer": "cut off mid-'),
            payload("recovered on the retry [1]."),
        ]
    )
    answer = generate("q", ranked, chunks, provider, max_tokens=700, retry_multiplier=3)
    assert not answer.refused
    assert answer.cited_chunk_ids == [ranked[0]]
    assert [k["max_tokens"] for k in provider.kwargs] == [700, 2100]
    # Truncation is a budget failure, not a shape failure, so the constraint is
    # held fixed. Changing both would confound the two remedies.
    assert [k["json_object"] for k in provider.kwargs] == [True, True]


def test_a_vacuous_reply_that_stopped_normally_drops_the_constraint_instead(chunks, ranked):
    # Measured on nemotron-3-super at 2048 tokens, finish_reason='stop': under
    # `response_format` the same prompt returns whitespace or an object with no
    # `answer` key, and unconstrained it returns a fully cited answer. Widening
    # the budget here would be treating a shape problem as a budget problem --
    # which is what the live service did for a whole revision.
    provider = FakeProvider(["{}", payload("recovered [1].")])
    answer = generate("q", ranked, chunks, provider, max_tokens=700, retry_multiplier=3)
    assert not answer.refused and len(provider.prompts) == 2
    assert [k["json_object"] for k in provider.kwargs] == [True, False]
    assert [k["max_tokens"] for k in provider.kwargs] == [700, 700]


def test_a_payload_with_an_empty_answer_string_also_retries(chunks, ranked):
    provider = FakeProvider(['{"sufficient": true, "answer": "   "}', payload("real [2].")])
    answer = generate("q", ranked, chunks, provider)
    assert not answer.refused and len(provider.prompts) == 2


def test_the_retry_happens_once_and_a_second_empty_reply_refuses(chunks, ranked):
    provider = FakeProvider(["{}", "{}"])
    answer = generate("q", ranked, chunks, provider)
    assert answer.refused and answer.refusal_reason == "unparseable"
    assert len(provider.prompts) == 2, "exactly one retry, not a loop"


def test_still_truncated_after_the_retry_refuses_as_truncated_not_unparseable(chunks, ranked):
    # The distinction is the point: "unparseable" sends an operator to look at
    # the prompt, when the budget is what needs changing. Mislabelling this is
    # what made the live defect take three sessions to identify.
    provider = FakeProvider([_truncated("{partial"), _truncated("{still partial")])
    answer = generate("q", ranked, chunks, provider)
    assert answer.refused and answer.refusal_reason == "truncated"
    assert len(provider.prompts) == 2


def _warm(cache_dir, chunks, ranked, text):
    """Put one response in a cache under the key `generate` will look up."""
    cache = JsonCache(cache_dir)
    prompt = build_prompt("q", build_passages(ranked, chunks)[0], sentences=4)
    CachedProvider(FakeProvider([text]), cache).complete(
        prompt, system=ANSWER_SYSTEM, max_tokens=700, json_object=True
    )
    return cache


def test_replay_never_retries(tmp_path, chunks, ranked):
    # Replay must reproduce what was recorded, and a retry is a live call: its
    # key would miss and surface as `provider_error` where the recorded run had
    # a refusal. Three of the 180 committed answer entries reach this branch.
    cache = _warm(tmp_path, chunks, ranked, "{}")
    replaying = CachedProvider(FakeProvider(fail=True), cache, offline=True)
    answer = generate("q", ranked, chunks, replaying, max_tokens=700)
    assert answer.refused and answer.refusal_reason == "unparseable"


def test_a_cached_but_unusable_reply_is_still_retried_when_live(tmp_path, chunks, ranked):
    # The opposite of the above, and the reason the gate is replay rather than
    # `cached`: observed on the deployed service, where asking the same question
    # twice served the first failure back out of the cache and refused forever
    # after. The retry's own result is cached under its own key, so a later
    # repeat still costs one call rather than two.
    cache = _warm(tmp_path, chunks, ranked, "{}")
    live = CachedProvider(FakeProvider([payload("recovered [1].")]), cache)
    answer = generate("q", ranked, chunks, live, max_tokens=700)
    assert not answer.refused and answer.cited_chunk_ids == [ranked[0]]


def test_unparseable_prose_is_not_retried(chunks, ranked):
    # Malformed output that ran to completion is not the retryable case.
    provider = FakeProvider(["I will not emit JSON.", payload("unused [1].")])
    answer = generate("q", ranked, chunks, provider)
    assert answer.refused and answer.refusal_reason == "unparseable"
    assert len(provider.prompts) == 1


def test_a_genuine_refusal_is_not_mistaken_for_an_empty_payload(chunks, ranked):
    provider = FakeProvider([payload(REFUSAL_TEXT, sufficient=False)])
    answer = generate("q", ranked, chunks, provider)
    assert answer.refusal_reason == "model" and len(provider.prompts) == 1


def test_truncation_is_retried_even_without_the_format_constraint(chunks, ranked):
    # `finish_reason` is a fact about the call, not about the format asked for.
    provider = FakeProvider([_truncated("half an ans"), payload("recovered [1].")])
    answer = generate("q", ranked, chunks, provider, json_object=False)
    assert not answer.refused and len(provider.prompts) == 2
    assert [k["json_object"] for k in provider.kwargs] == [False, False]


def test_an_empty_payload_is_not_chased_when_json_was_never_requested(chunks, ranked):
    provider = FakeProvider(["{}", payload("unused [1].")])
    answer = generate("q", ranked, chunks, provider, json_object=False)
    assert answer.refused and len(provider.prompts) == 1
