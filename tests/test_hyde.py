"""The HyDE arm on fakes: no model, no torch, no network."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeProvider
from production_rag import hyde
from production_rag.bm25 import build_from_chunks as build_bm25
from production_rag.dense import build_from_chunks as build_dense
from production_rag.pipeline import Retriever

NOTES = Path(__file__).resolve().parents[1] / "NOTES.md"


@pytest.fixture
def retriever(embedder, sample_chunks):
    return Retriever(
        {c.chunk_id: c for c in sample_chunks},
        build_bm25(sample_chunks),
        build_dense(sample_chunks, embedder),
        embedder=embedder,
        strategy="heading",
    )


def test_prompt_is_the_registered_text_byte_for_byte() -> None:
    notes = NOTES.read_text(encoding="utf-8").replace("\r\n", "\n")
    section = notes.split("**Draft prompt (frozen", 1)[1].split("**Arms.**", 1)[0]
    block = [line[4:] for line in section.split("\n") if line.startswith("    ") or line == ""]
    frozen = "\n".join(block).strip("\n")
    assert frozen == hyde.PROMPT


def test_prompt_fill_is_one_pass_so_braces_in_a_question_survive() -> None:
    text = hyde.render_prompt("What does {{ ref('x') }} do?")
    assert text.endswith("Question: What does {{ ref('x') }} do?")


def test_empty_reply_and_provider_error_are_failed_drafts_without_retry() -> None:
    provider = FakeProvider(["  ", "A passage about read_parquet."])
    drafts = hyde.write_drafts(provider, [("q1", "a"), ("q2", "b")], log=lambda *_: None)
    assert drafts["q1"].failed and not drafts["q2"].failed
    assert drafts["q2"].text == "A passage about read_parquet."
    assert len(provider.prompts) == 2  # no retry
    assert provider.kwargs[0] == {"temperature": 0.0, "max_tokens": hyde.MAX_TOKENS}
    down = hyde.write_drafts(FakeProvider(fail=True), [("q3", "c")], log=lambda *_: None)
    assert down["q3"].failed and down["q3"].text == ""


@pytest.mark.parametrize(
    ("mean", "lo", "hi", "expected"),
    [
        (0.03, 0.001, 0.05, "gain"),
        (0.01, 0.001, 0.02, "no clear difference"),
        (-0.02, -0.04, -0.001, "worse"),
        (0.0, -0.01, 0.01, "no clear difference"),
    ],
)
def test_decision_rule(mean, lo, hi, expected) -> None:
    assert hyde.decide(mean, lo, hi) == expected


def test_baselines_match_the_retriever_and_a_failed_draft_falls_back(
    retriever, embedder, monkeypatch
) -> None:
    monkeypatch.setattr(hyde, "POOL", 4)
    monkeypatch.setattr(hyde, "K", 3)
    scored = [
        hyde.Scored("q1", "conceptual", ("duckdb/parquet#0",)),
        hyde.Scored("q2", "exact_term", ("dbt/inc#0",)),
    ]
    texts = {"q1": "how do I read parquet files", "q2": "incremental models"}
    drafts = {
        "q1": hyde.Draft("q1", "duckdb read_parquet reads parquet files", False, 1.0, 9, False),
        "q2": hyde.Draft("q2", "", True, 0.0, 0, False),
    }
    res = hyde.evaluate(retriever, embedder, drafts, scored, texts)
    for qid, text in texts.items():
        dense = retriever.retrieve(text, arm="dense", k=3, pool=4).chunk_ids
        hybrid = retriever.retrieve(text, arm="hybrid_score_weighted", k=3, pool=4).chunk_ids
        assert res["ranked"]["dense"][qid] == dense
        assert res["ranked"][hyde.BASELINE_HYBRID][qid] == hybrid
    # q2's draft failed, so its HyDE arms are the plain query.
    assert res["ranked"]["dense_hyde"]["q2"] == res["ranked"]["dense"]["q2"]
    # A draft that copies the gold passage puts it first.
    assert res["ranked"]["dense_hyde"]["q1"][0] == "duckdb/parquet#0"
    assert len(res["encode_s"]) == 1


def test_average_vector_is_the_normalised_mean(embedder) -> None:
    q = embedder.encode_query("parquet files")
    d = embedder.encode_documents(["duckdb read_parquet"])[0]
    avg = (q + d) / np.linalg.norm(q + d)
    assert np.isclose(np.linalg.norm(avg), 1.0)
    assert np.dot(avg, q) > 0 and np.dot(avg, d) > 0


def test_compare_reports_every_arm_against_its_own_baseline() -> None:
    scored = [hyde.Scored(f"q{i}", "conceptual", ("x",)) for i in range(6)]
    per = {
        arm: {s.qid: {"ndcg@10": 0.5, "recall@5": 0.5} for s in scored}
        for arm, _, _ in hyde.HYDE_ARMS
    }
    for s in scored:
        per["hsw_hyde"][s.qid] = {"ndcg@10": 0.6, "recall@5": 0.5}
    out = hyde.compare(per, scored)
    assert set(out) == set(hyde.BASELINE_OF)
    assert out["hsw_hyde"]["decision"] == "gain"
    assert out["dense_avg"]["decision"] == "no clear difference"
    assert re.match(
        r"\| `hsw_hyde` \| `hybrid_score_weighted`",
        hyde.format_report({"comparisons": out}).split("\n")[4],
    )


def test_replay_reads_the_committed_drafts_and_refuses_a_gap(tmp_path) -> None:
    drafts = {"q1": hyde.Draft("q1", "text", False, 1.0, 5, True)}
    path = tmp_path / "drafts.json"
    hyde.save_drafts(drafts, path, model="m")
    assert hyde.load_drafts(path) == drafts
    with pytest.raises(ValueError, match="no draft"):
        hyde.run(
            indexes=tmp_path,
            retrieval_json=tmp_path / "r.json",
            drafts_path=path,
            out=tmp_path / "o.json",
            provider=FakeProvider(),
            commit="x",
            replay=True,
        )


def test_local_bootstrap_and_percentile_match_the_agentic_versions() -> None:
    from production_rag import agentic

    rng = np.random.default_rng(7)
    diffs = {"a": list(rng.normal(size=40)), "b": list(rng.normal(size=40))}
    assert hyde.paired_bootstrap(diffs, trials=500, seed=3) == agentic.paired_bootstrap(
        diffs, trials=500, seed=3
    )
    values = list(rng.random(25))
    assert hyde.percentile(values, 0.95) == agentic.percentile(values, 0.95)
