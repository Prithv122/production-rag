from __future__ import annotations

import json

import pytest

from conftest import make_chunk
from production_rag.bm25 import build_from_chunks as build_bm25
from production_rag.dense import build_from_chunks as build_dense
from production_rag.evaluate import evaluate_arm, markdown_table, run_grid, save_results, summarise
from production_rag.groundtruth import Evidence, Question, index_chunks_by_doc
from production_rag.pipeline import Retriever


def _chunk(chunk_id: str, text: str, doc_id: str, start: int, tool: str = "dbt"):
    base = make_chunk(chunk_id, text, tool=tool)
    return type(base)(
        **{
            **base.to_dict(),
            "heading_path": (),
            "doc_id": doc_id,
            "tool": tool,
            "start": start,
            "end": start + len(text),
        }
    )


@pytest.fixture
def corpus():
    return [
        _chunk(
            "dbt/inc#0", "incremental models append new rows to an existing table", "dbt/inc", 0
        ),
        _chunk("dbt/inc#1", "full refresh rebuilds the table from scratch", "dbt/inc", 60),
        _chunk("dag/assets#0", "assets declare upstream dependencies", "dag/assets", 0, "dagster"),
        _chunk("duck/pq#0", "read_parquet scans parquet files lazily", "duck/pq", 0, "duckdb"),
    ]


@pytest.fixture
def questions():
    return [
        Question(
            "q1",
            "how do I append rows instead of rebuilding",
            "conceptual",
            (Evidence("dbt/inc", 0, 55, "incremental models append new rows"),),
            ("dbt",),
        ),
        Question(
            "q2",
            "read_parquet lazy scanning",
            "exact_term",
            (Evidence("duck/pq", 0, 39, "read_parquet scans parquet files lazily"),),
            ("duckdb",),
        ),
        Question("q3", "how do I configure Snowflake warehouses", "unanswerable", (), ()),
    ]


@pytest.fixture
def retriever(corpus, embedder):
    return Retriever(
        {c.chunk_id: c for c in corpus},
        build_bm25(corpus),
        build_dense(corpus, embedder),
        embedder=embedder,
        strategy="heading",
    )


def test_metrics_are_computed_only_for_questions_with_gold(retriever, corpus, questions):
    result, rows = evaluate_arm(
        retriever, questions, index_chunks_by_doc(corpus), arm="bm25", k=3, pool=4
    )
    assert result.n_scored == 2
    assert result.n_unanswerable == 1
    assert {row.qid for row in rows if not row.gold} == {"q3"}


def test_an_unanswerable_question_never_enters_the_mean(retriever, corpus, questions):
    """Scoring it 0 would punish, and scoring it 1 would reward, a system for
    something recall is not defined over."""
    result, _ = evaluate_arm(
        retriever, questions, index_chunks_by_doc(corpus), arm="bm25", k=3, pool=4
    )
    per_category = result.by_category
    assert "unanswerable" not in per_category
    assert set(per_category) == {"conceptual", "exact_term"}


def test_a_perfect_arm_scores_one(retriever, corpus, questions):
    result, _ = evaluate_arm(
        retriever, questions, index_chunks_by_doc(corpus), arm="hybrid", k=3, pool=4
    )
    assert result.overall["recall@5"] > 0.0
    assert 0.0 <= result.overall["ndcg@10"] <= 1.0


def test_rewrite_failures_are_counted_not_hidden(retriever, corpus, questions, reranker):
    from conftest import FakeProvider

    result, _ = evaluate_arm(
        retriever,
        questions,
        index_chunks_by_doc(corpus),
        arm="rerank_rewrite",
        k=3,
        pool=4,
        reranker=reranker,
        rewriter=FakeProvider(["not json at all"] * 5),
    )
    assert result.rewrite_failures == len(questions)


def _write_index(tmp_path, corpus, embedder, strategy="heading"):
    root = tmp_path / strategy
    root.mkdir(parents=True)
    with (root / "chunks.jsonl").open("w", encoding="utf-8") as handle:
        for chunk in corpus:
            handle.write(json.dumps(chunk.to_dict()) + "\n")
    build_bm25(corpus).save(root / "bm25")
    build_dense(corpus, embedder).save(root / "dense")


def test_grid_runs_every_arm_on_every_strategy(tmp_path, corpus, questions, embedder, reranker):
    _write_index(tmp_path, corpus, embedder, "heading")
    _write_index(tmp_path, corpus, embedder, "fixed")

    results = run_grid(
        questions,
        index_dir=tmp_path,
        strategies=["heading", "fixed"],
        arms=["bm25", "hybrid", "hybrid_rerank"],
        embedder_factory=lambda: embedder,
        reranker=reranker,
        k=3,
        pool=4,
        verbose=False,
    )
    assert len(results["arms"]) == 6
    assert len(results["rows"]) == 6 * len(questions)
    assert set(results["coverage"]) == {"heading", "fixed"}
    assert results["config"]["categories"]["unanswerable"] == 1


def test_coverage_is_reported_per_strategy(tmp_path, corpus, questions, embedder):
    _write_index(tmp_path, corpus, embedder)
    results = run_grid(
        questions,
        index_dir=tmp_path,
        strategies=["heading"],
        arms=["bm25"],
        embedder_factory=lambda: embedder,
        k=3,
        pool=4,
        verbose=False,
    )
    # Two of three questions are answerable and both map onto a chunk.
    assert results["coverage"]["heading"]["mapped"] == 1.0
    assert results["coverage"]["heading"]["questions"] == 2.0


def test_tables_have_a_row_per_arm_and_a_column_per_strategy(tmp_path, corpus, questions, embedder):
    _write_index(tmp_path, corpus, embedder, "heading")
    _write_index(tmp_path, corpus, embedder, "fixed")
    results = run_grid(
        questions,
        index_dir=tmp_path,
        strategies=["heading", "fixed"],
        arms=["bm25", "hybrid"],
        embedder_factory=lambda: embedder,
        k=3,
        pool=4,
        verbose=False,
    )
    table = markdown_table(results, metric="recall@5")
    lines = table.splitlines()
    assert lines[0] == "| Arm | heading | fixed |"
    assert len(lines) == 4  # header, rule, two arms

    by_category = markdown_table(results, metric="recall@5", by_category=True)
    assert "exact_term" in by_category and "conceptual" in by_category
    assert "unanswerable" not in by_category


def test_results_are_saved_as_readable_json(tmp_path, corpus, questions, embedder):
    _write_index(tmp_path, corpus, embedder)
    results = run_grid(
        questions,
        index_dir=tmp_path,
        strategies=["heading"],
        arms=["bm25"],
        embedder_factory=lambda: embedder,
        k=3,
        pool=4,
        verbose=False,
    )
    path = save_results(results, tmp_path / "out" / "retrieval.json")
    assert json.loads(path.read_text(encoding="utf-8"))["config"]["n_questions"] == 3
    assert "recall@5" in summarise(results)
