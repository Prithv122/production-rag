"""pgvector behind the ann-bench interface.

The first group needs neither psycopg nor a database, so it runs in the fast job.
The second needs a pgvector-enabled Postgres and skips unless `PGVECTOR_DSN` names one
(`docker compose -f pgvector.compose.yml up -d --wait`, then
`PGVECTOR_DSN=postgresql://postgres@127.0.0.1:5433/postgres`). Scores from Postgres and
numpy sum floats in a different order, so "same results" is checked as in test_ann.py:
scores agree position by position within a tolerance.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import pytest

from production_rag.ann import (
    EXACT_SHARE_LIMIT,
    PGVECTOR_BUDGET_MS,
    PGVECTOR_E2E_P95_MS,
    PGVECTOR_RECALL_FLOOR,
    PgVectorIndex,
    ann_recall,
    pgvector_verdict,
    synthetic_queries,
    synthetic_vectors,
    vector_literal,
)
from production_rag.cli import main
from production_rag.dense import DenseIndex

DIM = 32
DSN = os.environ.get("PGVECTOR_DSN")


def run_cli(tmp_path, monkeypatch, *argv):
    monkeypatch.chdir(tmp_path)
    return main(["ann-bench", "--synthetic", "--queries", "20", "--repeats", "1", *argv])


# -- no database needed ---------------------------------------------------------
def test_vector_literal_round_trips_float32_exactly():
    rng = np.random.default_rng(0)
    vector = rng.standard_normal(64).astype(np.float32)
    text = vector_literal(vector)
    assert text.startswith("[") and text.endswith("]")
    back = np.array(json.loads(text), dtype=np.float32)
    assert np.array_equal(back, vector)


def row(index, params, recall, p95, n=22789):
    return {
        "n": n,
        "index": index,
        "params": params,
        "recall_at_10_vs_exact": recall,
        "p95_ms": p95,
    }


REGISTERED = dict(real=True, k=10, n_queries=184, default_grids=True)


def both_kinds(*extra):
    """One passive row of each kind, so the 'both kinds measured' condition is met."""
    base = [
        row("pgvector-hnsw", {"ef_search": 16}, 0.0, 1e9),
        row("pgvector-ivfflat", {"nprobe": 1}, 0.0, 1e9),
    ]
    return [*base, *extra]


def test_verdict_budget_is_the_pinned_16_45_ms():
    out = pgvector_verdict(both_kinds(), 22789, **REGISTERED)
    assert out["budget_p95_ms"] == PGVECTOR_BUDGET_MS == pytest.approx(16.45)
    assert pytest.approx(EXACT_SHARE_LIMIT * PGVECTOR_E2E_P95_MS) == PGVECTOR_BUDGET_MS
    assert "164.5" in out["budget_source"]


def test_verdict_needs_recall_and_latency_in_the_same_config():
    fast_but_lossy = row("pgvector-hnsw", {"ef_search": 16}, PGVECTOR_RECALL_FLOOR - 0.01, 1.0)
    accurate_but_slow = row("pgvector-hnsw", {"ef_search": 256}, 0.999, PGVECTOR_BUDGET_MS + 0.01)
    out = pgvector_verdict(both_kinds(fast_but_lossy, accurate_but_slow), 22789, **REGISTERED)
    assert out["qualifies"] is False and out["passing_configs"] == []
    assert out["statement"].startswith("pgvector did not meet")


def test_verdict_boundaries_are_inclusive():
    on_the_line = row("pgvector-ivfflat", {"nprobe": 16}, PGVECTOR_RECALL_FLOOR, PGVECTOR_BUDGET_MS)
    out = pgvector_verdict(both_kinds(on_the_line), 22789, **REGISTERED)
    assert out["qualifies"] is True


def test_verdict_passes_on_one_config_and_uses_the_narrow_wording():
    good = row("pgvector-ivfflat", {"nprobe": 16}, PGVECTOR_RECALL_FLOOR, 5.0)
    out = pgvector_verdict(both_kinds(good), 22789, **REGISTERED)
    assert out["passing_configs"] == [{"index": "pgvector-ivfflat", "params": {"nprobe": 16}}]
    assert "retrieval backend" in out["statement"]
    assert "fits" not in out["statement"]  # the run does not test the whole application
    assert out["statement_withheld"] == []


@pytest.mark.parametrize(
    ("override", "reason"),
    [
        ({"real": False}, "synthetic"),
        ({"k": 5}, "k=5"),
        ({"n_queries": 177}, "177 queries"),
        ({"default_grids": False}, "grids"),
    ],
)
def test_the_statement_is_withheld_when_the_run_departs_from_the_protocol(override, reason):
    good = row("pgvector-hnsw", {"ef_search": 64}, 1.0, 1.0)
    out = pgvector_verdict(both_kinds(good), 22789, **{**REGISTERED, **override})
    assert out["statement"] is None
    assert any(reason in why for why in out["statement_withheld"])
    assert out["qualifies"] is True  # the numbers are still reported, only the claim is held back


def test_the_statement_is_withheld_unless_both_kinds_were_measured():
    good = row("pgvector-hnsw", {"ef_search": 64}, 1.0, 1.0)
    out = pgvector_verdict([good], 22789, **REGISTERED)
    assert out["statement"] is None
    assert any("both" in why for why in out["statement_withheld"])


def test_a_changed_budget_cannot_carry_the_statement():
    good = row("pgvector-hnsw", {"ef_search": 64}, 1.0, 50.0)
    out = pgvector_verdict(both_kinds(good), 22789, budget_ms=100.0, **REGISTERED)
    assert out["statement"] is None and out["qualifies"] is True


def test_verdict_only_counts_the_corpus_size_asked_about():
    other_size = row("pgvector-hnsw", {"ef_search": 64}, 1.0, 1.0, n=100_000)
    assert pgvector_verdict([other_size], 22789, **REGISTERED)["qualifies"] is False


def test_missing_psycopg_names_the_extra(monkeypatch, tmp_path, capsys):
    monkeypatch.setitem(sys.modules, "psycopg", None)
    vectors = synthetic_vectors(50, DIM)
    with pytest.raises(ModuleNotFoundError, match=r"uv sync --extra pgvector"):
        PgVectorIndex([f"c{i}" for i in range(50)], vectors, "hnsw")
    argv = ("--index", "pgvector-hnsw", "--n", "100", "--dim", "16", "--out", "x.json")
    assert run_cli(tmp_path, monkeypatch, *argv) == 1
    assert "psycopg is not installed" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


def test_unknown_kind_and_bad_table_prefix_are_rejected_before_any_connection():
    ones = np.ones((1, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="unknown pgvector kind"):
        PgVectorIndex(["a"], ones, "flat")
    with pytest.raises(ValueError, match="table_prefix"):
        PgVectorIndex(["a"], ones, "hnsw", table_prefix="x; DROP TABLE y")


def seed_result(tmp_path, name, text="{}"):
    target = tmp_path / "eval" / "results" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    return target


def test_an_existing_pgvector_result_file_is_not_overwritten(tmp_path, monkeypatch, capsys):
    target = seed_result(tmp_path, "ann_pgvector.json")
    argv = ("--index", "pgvector-hnsw", "--n", "100", "--dim", "16")
    assert run_cli(tmp_path, monkeypatch, *argv) == 1
    assert "already exists" in capsys.readouterr().err
    assert target.read_text(encoding="utf-8") == "{}"


def test_a_pgvector_run_never_writes_ann_json_even_with_overwrite(tmp_path, monkeypatch, capsys):
    target = seed_result(tmp_path, "ann.json", '{"faiss": "record"}')
    argv = ("--index", "pgvector-hnsw", "--n", "100", "--dim", "16", "--overwrite")
    assert run_cli(tmp_path, monkeypatch, *argv, "--out", str(target)) == 1
    assert "never writes ann.json" in capsys.readouterr().err
    assert target.read_text(encoding="utf-8") == '{"faiss": "record"}'


def test_any_run_aimed_at_the_pgvector_result_file_is_protected(tmp_path, monkeypatch, capsys):
    target = seed_result(tmp_path, "ann_pgvector.json", '{"finished": true}')
    argv = ("--index", "exact", "--n", "100", "--dim", "16", "--out", str(target))
    assert run_cli(tmp_path, monkeypatch, *argv) == 1
    assert "already exists" in capsys.readouterr().err
    assert target.read_text(encoding="utf-8") == '{"finished": true}'


def test_a_result_that_appeared_during_the_run_is_not_replaced(tmp_path, capsys):
    from production_rag.cli import _write_result

    target = tmp_path / "ann_pgvector.json"
    assert _write_result(target, "first", exclusive=True) == target
    spare = _write_result(target, "second", exclusive=True)
    assert spare != target and spare.read_text(encoding="utf-8") == "second"
    assert target.read_text(encoding="utf-8") == "first"
    assert "appeared during the run" in capsys.readouterr().err
    assert _write_result(target, "third", exclusive=False) == target
    assert target.read_text(encoding="utf-8") == "third"


def test_pgvector_runs_never_touch_the_faiss_result_file(tmp_path, monkeypatch, capsys):
    pytest.importorskip("psycopg")
    monkeypatch.setattr("production_rag.ann.PGVECTOR_CONNECT_TIMEOUT_S", 2)
    # Nothing listens on port 1, so the run fails after the output path was chosen and
    # before anything is written: no result file of either name may appear.
    dsn = "postgresql://x@127.0.0.1:1/y"
    argv = ("--index", "pgvector-hnsw", "--n", "100", "--dim", "16", "--dsn", dsn)
    assert run_cli(tmp_path, monkeypatch, *argv) == 1
    assert "cannot reach Postgres" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


# -- with a database ----------------------------------------------------------------
needs_db = pytest.mark.skipif(not DSN, reason="PGVECTOR_DSN is not set")


@pytest.fixture(scope="module")
def data():
    vectors = synthetic_vectors(3000, DIM)
    ids = [f"c{i}" for i in range(len(vectors))]
    return ids, vectors, synthetic_queries(60, DIM), DenseIndex(ids, vectors)


@pytest.fixture(scope="module")
def hnsw(data):
    pytest.importorskip("psycopg")
    index = PgVectorIndex(data[0], data[1], "hnsw", dsn=DSN, table_prefix="ann_test")
    yield index
    index.close()


@pytest.fixture(scope="module")
def ivfflat(data):
    pytest.importorskip("psycopg")
    index = PgVectorIndex(data[0], data[1], "ivfflat", dsn=DSN, table_prefix="ann_test")
    yield index
    index.close()


def recall(index, exact, queries, k=10):
    found = [[c for c, _ in index.search_vector(q, k)] for q in queries]
    return ann_recall(found, [[c for c, _ in exact.search_vector(q, k)] for q in queries], k)


@needs_db
def test_hnsw_recall_improves_with_ef_search_and_reaches_near_exact(hnsw, data):
    *_, queries, exact = data
    hnsw.set_search_params(ef_search=10)
    low = recall(hnsw, exact, queries)
    hnsw.set_search_params(ef_search=400)
    high = recall(hnsw, exact, queries)
    assert high >= low and high >= 0.97


@needs_db
def test_ivfflat_probing_every_list_is_exact(ivfflat, data):
    *_, queries, exact = data
    ivfflat.set_search_params(nprobe=10_000)  # clipped to the number of lists
    assert ivfflat.params["nprobe"] == ivfflat.nlist
    for q in queries:
        got, want = ivfflat.search_vector(q, 10), exact.search_vector(q, 10)
        assert all(abs(a[1] - b[1]) <= 1e-5 for a, b in zip(got, want, strict=True))
    assert recall(ivfflat, exact, queries) >= 0.99


@needs_db
def test_ivfflat_recall_never_falls_as_probes_grow(ivfflat, data):
    *_, queries, exact = data
    seen = []
    for probes in (1, 4, 16, 10_000):
        ivfflat.set_search_params(nprobe=probes)
        seen.append(recall(ivfflat, exact, queries))
    assert seen == sorted(seen) and seen[0] < seen[-1]


@needs_db
def test_scores_are_cosine_similarities_in_descending_order(hnsw, data):
    *_, queries, exact = data
    hnsw.set_search_params(ef_search=400)
    got = hnsw.search_vector(queries[0], 10)
    scores = [s for _, s in got]
    assert scores == sorted(scores, reverse=True)
    assert abs(scores[0] - exact.search_vector(queries[0], 10)[0][1]) <= 1e-5


@needs_db
def test_the_measured_plan_really_uses_the_index(hnsw, ivfflat):
    assert hnsw.plan_uses_index() and ivfflat.plan_uses_index()


@needs_db
def test_a_search_param_for_the_other_kind_is_rejected(hnsw, ivfflat):
    with pytest.raises(ValueError, match="only applies to hnsw"):
        ivfflat.set_search_params(ef_search=64)
    with pytest.raises(ValueError, match="only applies to ivfflat"):
        hnsw.set_search_params(nprobe=4)


@needs_db
def test_k_zero_is_empty_and_k_is_honoured(hnsw, data):
    *_, queries, _ = data
    assert hnsw.search_vector(queries[0], 0) == []
    assert len(hnsw.search_vector(queries[0], 5)) == 5
    assert len(hnsw) == 3000


@needs_db
def test_sizes_and_server_info_are_reported(hnsw):
    assert hnsw.index_bytes() > 0 and hnsw.table_bytes() > 0
    info = hnsw.server_info()
    assert info["settings"]["enable_seqscan"] == "off"
    assert info["settings"]["max_parallel_maintenance_workers"] == "0"
    assert info["pgvector"] and info["postgres"]


@needs_db
def test_select1_timings_have_the_requested_count(hnsw):
    timings = hnsw.select1_timings(25)
    assert len(timings) == 25 and all(t > 0 for t in timings)


@needs_db
def test_the_bench_writes_pgvector_rows_with_the_round_trip_and_a_verdict(tmp_path, monkeypatch):
    argv = (
        "--index",
        "pgvector-hnsw",
        "--index",
        "pgvector-ivfflat",
        "--n",
        "1500",
        "--dim",
        "32",
        "--e2e-p95-ms",
        "100",
        "--dsn",
        DSN,
    )
    assert run_cli(tmp_path, monkeypatch, *argv) == 0
    written = tmp_path / "eval" / "results"
    out = json.loads((written / "ann_pgvector.json").read_text(encoding="utf-8"))
    assert {r["index"] for r in out["rows"]} == {"exact", "pgvector-hnsw", "pgvector-ivfflat"}
    pg_rows = [r for r in out["rows"] if r["index"] != "exact"]
    assert all(r["select1_p95_ms"] > 0 and r["table_bytes"] > 0 for r in pg_rows)
    assert all(r["plan_uses_index"] is True for r in pg_rows)
    assert out["config"]["pgvector"]["settings"]["max_parallel_maintenance_workers"] == "0"
    (verdict,) = out["pgvector_verdicts"]
    assert verdict["budget_p95_ms"] == pytest.approx(16.45)  # pinned, not the 100 ms given
    assert verdict["statement"] is None  # synthetic data never carries the claim
    assert any("synthetic" in why for why in verdict["statement_withheld"])
    assert [p.name for p in written.iterdir()] == ["ann_pgvector.json"]


@needs_db
def test_real_mode_scores_pgvector_end_to_end_and_withholds_the_off_protocol_verdict(
    embedder, sample_chunks
):
    pytest.importorskip("psycopg")
    from production_rag.ann import run_bench
    from production_rag.bm25 import build_from_chunks as build_bm25
    from production_rag.dense import build_from_chunks as build_dense
    from production_rag.groundtruth import Evidence, Question, index_chunks_by_doc
    from production_rag.pipeline import Retriever

    dense = build_dense(sample_chunks, embedder)
    retriever = Retriever(
        {c.chunk_id: c for c in sample_chunks},
        build_bm25(sample_chunks),
        dense,
        embedder=embedder,
        strategy="fake",
    )
    text = sample_chunks[0].text
    question = Question(
        "q1", "dbt incremental models", "conceptual", (Evidence("dbt/inc", 0, len(text), text),)
    )
    out = run_bench(
        kinds=("pgvector-hnsw",),
        repeats=1,
        dsn=DSN,
        retriever=retriever,
        embedder=embedder,
        questions=[question],
        by_doc=index_chunks_by_doc(sample_chunks),
        e2e_p95_ms=1000.0,
        e2e_source="flag",
    )
    assert out["config"]["mode"] == "real"
    assert all(r["e2e"]["n_scored"] == 1 for r in out["rows"])
    assert retriever.dense is dense and retriever.embedder is embedder
    (verdict,) = out["pgvector_verdicts"]
    # One question and one pgvector kind: the run departs from the registered protocol.
    assert verdict["statement"] is None and verdict["budget_p95_ms"] == pytest.approx(16.45)
    assert any("184" in why for why in verdict["statement_withheld"])


@needs_db
def test_the_plan_still_uses_the_index_at_every_registered_setting(ivfflat, hnsw):
    for probes in (1, 4, 16, 64, 10_000):
        ivfflat.set_search_params(nprobe=probes)  # raises if the plan stops using the index
        assert ivfflat.plan_uses_index()
    for ef in (16, 32, 64, 128, 256):
        hnsw.set_search_params(ef_search=ef)
        assert hnsw.plan_uses_index()


@needs_db
def test_the_planner_is_stopped_from_answering_exactly_around_the_index(hnsw):
    settings = hnsw.server_info()["settings"]
    assert settings["enable_seqscan"] == "off" and settings["enable_sort"] == "off"
    assert hnsw.server_info()["prepared_statements"] is False


@needs_db
def test_a_second_run_on_the_same_tables_is_refused_while_the_first_holds_them(data):
    pytest.importorskip("psycopg")
    ids, vectors = data[0][:100], data[1][:100]
    first = PgVectorIndex(ids, vectors, "hnsw", dsn=DSN, table_prefix="ann_test_lock")
    try:
        with pytest.raises(RuntimeError, match="lock"):
            PgVectorIndex(ids, vectors, "hnsw", dsn=DSN, table_prefix="ann_test_lock")
        assert len(first.search_vector(data[2][0], 5)) == 5  # the holder is untouched
    finally:
        first.close()
    PgVectorIndex(ids, vectors, "hnsw", dsn=DSN, table_prefix="ann_test_lock").close()


@needs_db
def test_a_failed_build_does_not_leak_its_connection(data, monkeypatch):
    pytest.importorskip("psycopg")
    ids, vectors = data[0][:100], data[1][:100]
    closed = []
    real_close = PgVectorIndex.close
    monkeypatch.setattr(PgVectorIndex, "close", lambda self: (closed.append(1), real_close(self)))
    monkeypatch.setattr(PgVectorIndex, "plan_uses_index", lambda self: False)
    with pytest.raises(RuntimeError, match="planner did not use"):
        PgVectorIndex(ids, vectors, "hnsw", dsn=DSN, table_prefix="ann_test_plan")
    assert closed == [1]
