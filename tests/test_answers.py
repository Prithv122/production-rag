"""Answer scoring: the metrics, the subset selection and the frozen replay.

The arithmetic in :mod:`production_rag.answers` is the part of the generation
half that ends up in the README, so it is tested against hand-built rows where
the right answer is obvious by inspection. The point of note is the pairing:
`refusal_recall` and `false_refusal` are computed over disjoint denominators,
and a test here asserts that a model which refuses everything scores 1.000 on
the first and 1.000 on the second -- which is the reason the table never prints
one without the other.
"""

from __future__ import annotations

import json

import pytest

from conftest import FakeProvider, make_chunk
from production_rag.answers import (
    AnswerRow,
    format_by_category,
    format_table,
    load_frozen_retrievals,
    run_grid,
    save,
    score_answer,
    select_questions,
    summarise,
)
from production_rag.generate import generate
from production_rag.groundtruth import Evidence, Question, index_chunks_by_doc


def row(**overrides) -> AnswerRow:
    base = dict(
        qid="q1",
        category="conceptual",
        arm="a",
        model="m",
        refused=False,
        refusal_reason="",
        should_refuse=False,
        n_citations=1,
        n_valid_citations=1,
        cited_gold=True,
        uncited=False,
        latency_s=1.0,
        prompt_tokens=10,
        completion_tokens=20,
    )
    base.update(overrides)
    return AnswerRow(**base)


def question(qid: str, category: str, *, answerable: bool = True) -> Question:
    evidence = (Evidence(doc_id="d1", start=0, end=10, quote="dbt models"),) if answerable else ()
    return Question(qid=qid, text=f"text {qid}", category=category, evidence=evidence)


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def test_refusal_recall_and_false_refusal_use_disjoint_denominators():
    rows = [
        row(qid="u1", should_refuse=True, refused=True, refusal_reason="model"),
        row(qid="u2", should_refuse=True, refused=False),
        row(qid="a1", should_refuse=False, refused=True, refusal_reason="model"),
        row(qid="a2", should_refuse=False, refused=False),
    ]
    s = summarise(rows, arm="a", model="m")
    assert s.n_unanswerable == 2 and s.n_answerable == 2
    assert s.refusal_recall == pytest.approx(0.5)
    assert s.false_refusal == pytest.approx(0.5)


def test_refusing_everything_scores_perfectly_on_recall_and_terribly_next_to_it():
    rows = [
        row(qid="u1", should_refuse=True, refused=True),
        row(qid="a1", refused=True),
        row(qid="a2", refused=True),
    ]
    s = summarise(rows, arm="a", model="m")
    assert s.refusal_recall == 1.0
    assert s.false_refusal == 1.0
    # Nothing was answered, so grounding has no denominator and must not be 1.0.
    assert s.grounded == 0.0


def test_grounded_is_measured_over_answered_questions_only():
    rows = [
        row(qid="a1", cited_gold=True),
        row(qid="a2", cited_gold=False),
        row(qid="a3", refused=True, cited_gold=False),  # excluded, not counted against
    ]
    assert summarise(rows, arm="a", model="m").grounded == pytest.approx(0.5)


def test_citation_validity_is_over_markers_not_answers():
    rows = [row(n_citations=4, n_valid_citations=2), row(n_citations=0, n_valid_citations=0)]
    assert summarise(rows, arm="a", model="m").citation_validity == pytest.approx(0.5)


def test_unparseable_and_provider_errors_are_counted_apart():
    rows = [
        row(refused=True, refusal_reason="unparseable"),
        row(refused=True, refusal_reason="provider_error"),
        row(refused=True, refusal_reason="model", should_refuse=True),
        row(),
    ]
    s = summarise(rows, arm="a", model="m")
    assert s.parse_failure == pytest.approx(0.25)
    assert s.provider_error == pytest.approx(0.25)


def test_summarise_ignores_rows_from_other_arms():
    rows = [row(arm="a", cited_gold=True), row(arm="b", cited_gold=False)]
    assert summarise(rows, arm="a", model="m").n == 1


def test_empty_denominators_do_not_divide_by_zero():
    s = summarise([row(should_refuse=False)], arm="a", model="m")
    assert s.refusal_recall == 0.0 and s.n_unanswerable == 0


# ---------------------------------------------------------------------------
# scoring one answer
# ---------------------------------------------------------------------------
def test_score_answer_marks_gold_when_a_cited_chunk_carries_the_evidence(sample_chunks):
    chunks = {c.chunk_id: c for c in sample_chunks}
    provider = FakeProvider([json.dumps({"sufficient": True, "answer": "grounded [1]."})])
    ranked = [c.chunk_id for c in sample_chunks]
    answer = generate("q", ranked, chunks, provider)
    scored = score_answer(question("q1", "conceptual"), answer, {ranked[0]}, arm="a")
    assert scored.cited_gold and scored.n_valid_citations == 1

    missed = score_answer(question("q1", "conceptual"), answer, {ranked[3]}, arm="a")
    assert not missed.cited_gold


def test_an_unanswerable_question_is_marked_should_refuse(sample_chunks):
    chunks = {c.chunk_id: c for c in sample_chunks}
    provider = FakeProvider([json.dumps({"sufficient": False, "answer": "no."})])
    answer = generate("q", list(chunks), chunks, provider)
    scored = score_answer(question("u1", "unanswerable", answerable=False), answer, set(), arm="a")
    assert scored.should_refuse and scored.refused


# ---------------------------------------------------------------------------
# subset selection
# ---------------------------------------------------------------------------
def test_every_rare_category_question_survives_the_subset():
    questions = (
        [question(f"c{i}", "conceptual") for i in range(100)]
        + [question(f"e{i}", "exact_term") for i in range(40)]
        + [question(f"x{i}", "cross_tool") for i in range(4)]
        + [question(f"u{i}", "unanswerable", answerable=False) for i in range(6)]
    )
    picked = select_questions(questions, n=40)
    categories = [q.category for q in picked]
    assert categories.count("cross_tool") == 4
    assert categories.count("unanswerable") == 6
    assert len(picked) <= 41  # rounding on the proportional split


def test_the_subset_is_deterministic_from_the_seed():
    questions = [question(f"c{i}", "conceptual") for i in range(80)]
    a = [q.qid for q in select_questions(questions, n=20, seed=7)]
    b = [q.qid for q in select_questions(questions, n=20, seed=7)]
    c = [q.qid for q in select_questions(questions, n=20, seed=8)]
    assert a == b and a != c


def test_asking_for_more_than_exists_returns_everything():
    questions = [question(f"c{i}", "conceptual") for i in range(5)]
    assert len(select_questions(questions, n=100)) == 5


# ---------------------------------------------------------------------------
# frozen retrieval replay
# ---------------------------------------------------------------------------
def test_frozen_rankings_are_read_back_for_one_arm_and_strategy(tmp_path):
    path = tmp_path / "retrieval.json"
    path.write_text(
        json.dumps(
            {
                "rows": [
                    {"qid": "q1", "arm": "bm25", "strategy": "heading", "ranked": ["a", "b", "c"]},
                    {"qid": "q1", "arm": "dense", "strategy": "heading", "ranked": ["z"]},
                    {"qid": "q1", "arm": "bm25", "strategy": "fixed", "ranked": ["y"]},
                ]
            }
        ),
        encoding="utf-8",
    )
    assert load_frozen_retrievals(path, arm="bm25", strategy="heading", k=2) == {"q1": ["a", "b"]}


def test_a_missing_arm_is_an_error_not_an_empty_grid(tmp_path):
    path = tmp_path / "retrieval.json"
    path.write_text(json.dumps({"rows": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="no rows"):
        load_frozen_retrievals(path, arm="bm25", strategy="heading")


# ---------------------------------------------------------------------------
# the grid end to end, with fakes
# ---------------------------------------------------------------------------
def test_run_grid_feeds_every_arm_identical_context(tmp_path):
    chunks = {
        "d1#0": make_chunk("d1#0", "dbt models are select statements"),
        "d1#1": make_chunk("d1#1", "unrelated prose"),
    }
    by_doc = index_chunks_by_doc(chunks.values())
    questions = [
        Question(
            qid="q1",
            text="what is a model?",
            category="conceptual",
            evidence=(
                Evidence(doc_id="d1", start=0, end=31, quote="dbt models are select statements"),
            ),
        ),
        question("u1", "unanswerable", answerable=False),
    ]
    rankings = {"q1": ["d1#0", "d1#1"], "u1": ["d1#1"]}
    answered = json.dumps({"sufficient": True, "answer": "a model is a select statement [1]."})
    refused = json.dumps({"sufficient": False, "answer": "no."})
    providers = {
        "arm_a": FakeProvider([answered, refused], name="a", model="model-a"),
        "arm_b": FakeProvider([answered, refused], name="b", model="model-b"),
    }

    rows, summaries, config = run_grid(
        questions, rankings, chunks, providers, by_doc=by_doc, workers=1
    )
    assert config["n_questions"] == 2 and config["arms"] == ["arm_a", "arm_b"]
    assert len(rows) == 4 and len(summaries) == 2
    assert all(s.refusal_recall == 1.0 and s.false_refusal == 0.0 for s in summaries)
    assert all(s.grounded == 1.0 for s in summaries)
    # Same frozen context both times: the prompts differ only by provider, so
    # a difference between the two rows cannot be a retrieval difference.
    assert providers["arm_a"].prompts == providers["arm_b"].prompts

    out = tmp_path / "answers.json"
    save(out, rows, summaries, config)
    assert json.loads(out.read_text(encoding="utf-8"))["config"]["n_questions"] == 2
    assert "refusal recall" in format_table(summaries)
    assert "conceptual" in format_by_category(rows)


def test_format_by_category_shows_a_dash_when_a_category_is_empty():
    rendered = format_by_category([row(category="conceptual")])
    assert "--" in rendered
