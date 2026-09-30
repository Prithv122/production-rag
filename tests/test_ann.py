"""The ann-bench harness, on synthetic vectors only.

FAISS and numpy sum floats in a different order, and the hashing FakeEmbedder
produces exact score ties, so "same results" never means identical id order:
scores must agree position by position within 1e-5, and the ids strictly above
the k-th score must be the same set.
"""

from __future__ import annotations

import json
import sys

import pytest

from production_rag.ann import (
    EXACT_SHARE_LIMIT,
    FaissIndex,
    VectorIndex,
    ann_recall,
    exact_share,
    percentile,
    synthetic_queries,
    synthetic_vectors,
    time_searches,
)
from production_rag.bm25 import build_from_chunks as build_bm25
from production_rag.cli import main
from production_rag.dense import DenseIndex
from production_rag.dense import build_from_chunks as build_dense
from production_rag.pipeline import Retriever

DIM = 32


@pytest.fixture
def faiss():
    return pytest.importorskip("faiss")


@pytest.fixture(scope="module")
def data():
    vectors = synthetic_vectors(3000, DIM)
    ids = [f"c{i}" for i in range(len(vectors))]
    return ids, vectors, synthetic_queries(100, DIM), DenseIndex(ids, vectors)


def same_topk(got, want, tol=1e-5):
    assert len(got) == len(want)
    assert all(abs(a[1] - b[1]) <= tol for a, b in zip(got, want, strict=True))
    kth = want[-1][1]
    assert {c for c, s in got if s > kth + tol} == {c for c, s in want if s > kth + tol}


def recall(index, exact, queries, k=10):
    found = [[c for c, _ in index.search_vector(q, k)] for q in queries]
    return ann_recall(found, [[c for c, _ in exact.search_vector(q, k)] for q in queries], k)


def run_cli(tmp_path, monkeypatch, *argv):
    monkeypatch.chdir(tmp_path)
    return main(["ann-bench", "--synthetic", "--queries", "20", "--repeats", "1", *argv])


# -- without faiss ------------------------------------------------------------
def test_ann_recall_arithmetic():
    assert ann_recall([["a", "b"]], [["b", "a"]], k=2) == 1.0
    assert ann_recall([["a", "b"]], [["c", "d"]], k=2) == 0.0
    assert ann_recall([["a", "b"], ["a", "x"]], [["a", "b"], ["a", "y"]], k=2) == 0.75


def test_synthetic_generators_are_seeded_and_unit_norm():
    a, b = synthetic_vectors(50, DIM, seed=1), synthetic_vectors(50, DIM, seed=1)
    assert (a == b).all() and not (a == synthetic_vectors(50, DIM, seed=2)).all()
    assert (synthetic_queries(9, DIM, seed=1) == synthetic_queries(9, DIM, seed=1)).all()
    for m in (a, synthetic_queries(9, DIM)):
        assert m.dtype == "float32"
        assert abs((m**2).sum(axis=1) - 1).max() < 1e-5


def test_queries_are_not_rows_of_the_base_set(data):
    _, vectors, queries, _ = data
    assert queries.shape[1] == vectors.shape[1]
    assert not ((queries[:, None, :] == vectors[None, :, :]).all(axis=2)).any()


def test_exact_share_rule():
    assert exact_share(1.0, None) == (None, None)
    assert EXACT_SHARE_LIMIT == 0.10
    assert exact_share(1.0, 10.0) == (0.1, True)  # the boundary counts as enough
    share, enough = exact_share(1.1, 10.0)
    assert share > 0.1 and enough is False


def test_dense_index_satisfies_the_protocol(data):
    assert isinstance(data[3], VectorIndex)


def test_percentile_matches_evaluate_arms_definition():
    values = [5.0, 1.0, 9.0, 3.0, 7.0, 2.0, 8.0, 4.0, 6.0, 0.0]
    assert percentile(values, 0.95) == sorted(values)[int(0.95 * (len(values) - 1))] == 8.0
    assert percentile(values, 0.50) == 4.0


def test_time_searches_pools_repeats(data):
    assert len(time_searches(data[3], data[2][:7], 10, repeats=3)) == 21


def test_cli_exact_only_writes_one_file(tmp_path, monkeypatch):
    assert (
        run_cli(
            tmp_path,
            monkeypatch,
            "--n",
            "300",
            "--dim",
            "16",
            "--index",
            "exact",
            "--out",
            "ann.json",
        )
        == 0
    )
    assert [p.name for p in tmp_path.iterdir()] == ["ann.json"]
    out = json.loads((tmp_path / "ann.json").read_text())
    assert set(out) == {"config", "rows", "verdicts"}
    assert [r["index"] for r in out["rows"]] == ["exact"]
    assert out["rows"][0]["recall_at_10_vs_exact"] == 1.0
    assert out["verdicts"][0]["exact_enough"] is None and out["config"]["e2e_p95_ms"] is None


def test_e2e_from_reads_the_denominator_back(tmp_path, monkeypatch):
    common = ("--n", "300", "--dim", "16", "--index", "exact")
    run_cli(tmp_path, monkeypatch, *common, "--e2e-p95-ms", "500", "--out", "first.json")
    first = json.loads((tmp_path / "first.json").read_text())
    assert first["config"]["e2e_source"] == "flag"
    run_cli(tmp_path, monkeypatch, *common, "--e2e-from", "first.json", "--out", "second.json")
    second = json.loads((tmp_path / "second.json").read_text())
    assert second["config"]["e2e_p95_ms"] == 500.0
    assert second["config"]["e2e_source"] == "file:first.json"
    assert second["verdicts"][0]["exact_enough"] is True


def test_missing_faiss_names_the_extra(monkeypatch, tmp_path, capsys, data):
    monkeypatch.setitem(sys.modules, "faiss", None)
    with pytest.raises(ModuleNotFoundError, match=r"uv sync --extra ann"):
        FaissIndex(data[0], data[1], "hnsw")
    assert run_cli(tmp_path, monkeypatch, "--index", "hnsw", "--out", "x.json") == 1
    assert "faiss is not installed. It lives in the optional `ann` extra" in capsys.readouterr().err
    assert list(tmp_path.iterdir()) == []


# -- with faiss ---------------------------------------------------------------
def test_flat_reproduces_the_exact_index(faiss, data):
    ids, vectors, queries, exact = data
    flat = FaissIndex(ids, vectors, "flat")
    assert recall(flat, exact, queries) >= 0.999
    for q in queries:
        same_topk(flat.search_vector(q, 10), exact.search_vector(q, 10))


def test_ivf_probing_every_list_is_exact(faiss, data):
    ids, vectors, queries, exact = data
    ivf = FaissIndex(ids, vectors, "ivf")
    ivf.set_search_params(nprobe=ivf.nlist)
    for q in queries:
        same_topk(ivf.search_vector(q, 10), exact.search_vector(q, 10))


def test_ivf_recall_never_falls_as_nprobe_grows(faiss, data):
    ids, vectors, queries, exact = data
    ivf = FaissIndex(ids, vectors, "ivf")
    seen = []
    for nprobe in (1, 4, 16, ivf.nlist):
        ivf.set_search_params(nprobe=nprobe)
        seen.append(recall(ivf, exact, queries))
    assert seen == sorted(seen)  # probed lists are nested, so this is guaranteed


def test_hnsw_recall_improves_with_ef_search(faiss, data):
    ids, vectors, queries, exact = data
    hnsw = FaissIndex(ids, vectors, "hnsw")
    hnsw.set_search_params(ef_search=16)
    low = recall(hnsw, exact, queries)
    hnsw.set_search_params(ef_search=256)
    high = recall(hnsw, exact, queries)
    assert high >= low and high >= 0.9


def test_ivfpq_rejects_a_dim_m_does_not_divide(faiss):
    vectors = synthetic_vectors(300, 64)
    with pytest.raises(ValueError, match="divisible"):
        FaissIndex([str(i) for i in range(300)], vectors, "ivfpq", m=48)


def test_missing_ids_are_dropped_and_search_params_need_no_rebuild(faiss, data):
    ids, vectors, queries, _ = data
    ivf = FaissIndex(ids, vectors, "ivf", nprobe=1)
    built, native = ivf.build_s, ivf._index
    narrow = ivf.search_vector(queries[0], 200)
    assert 0 < len(narrow) < 200 and len({c for c, _ in narrow}) == len(narrow)
    ivf.set_search_params(nprobe=ivf.nlist)
    assert len(ivf.search_vector(queries[0], 200)) == 200
    assert ivf.build_s == built and ivf._index is native


def test_faiss_index_satisfies_the_protocol(faiss, data):
    assert isinstance(FaissIndex(data[0], data[1], "flat"), VectorIndex)


@pytest.mark.parametrize("arm", ["dense", "hybrid"])
def test_retriever_gives_the_same_results_on_a_flat_faiss_index(
    faiss, embedder, sample_chunks, arm
):
    dense = build_dense(sample_chunks, embedder)
    retriever = Retriever(
        {c.chunk_id: c for c in sample_chunks}, build_bm25(sample_chunks), dense, embedder=embedder
    )
    want = retriever.retrieve("dbt incremental models", arm=arm).ranked
    retriever.dense = FaissIndex(dense.chunk_ids, dense.vectors, "flat")
    same_topk(retriever.retrieve("dbt incremental models", arm=arm).ranked, want)


def test_cli_with_every_kind(faiss, tmp_path, monkeypatch):
    assert run_cli(tmp_path, monkeypatch, "--n", "1500", "--dim", "64", "--out", "all.json") == 0
    out = json.loads((tmp_path / "all.json").read_text())
    assert {r["index"] for r in out["rows"]} == {"exact", "flat", "hnsw", "ivf", "ivfpq"}
    assert all(r["build_s"] is not None for r in out["rows"] if r["index"] != "exact")
    assert all(v["exact_enough"] is None for v in out["verdicts"])


def test_each_sweep_row_is_measured_at_its_own_setting(faiss):
    from production_rag.ann import run_bench

    out = run_bench(
        sizes=(3000,), dim=DIM, n_queries=50, kinds=("ivf",), repeats=1, nprobe_grid=(1, 10_000)
    )
    narrow, wide = [r for r in out["rows"] if r["index"] == "ivf"]
    assert narrow["params"]["nprobe"] == 1 and wide["params"]["nprobe"] == wide["params"]["nlist"]
    assert narrow["recall_at_10_vs_exact"] < wide["recall_at_10_vs_exact"]


def test_real_mode_scores_each_config_end_to_end_and_restores_the_retriever(
    faiss, embedder, sample_chunks
):
    from production_rag.ann import run_bench
    from production_rag.groundtruth import Evidence, Question, index_chunks_by_doc

    dense = build_dense(sample_chunks, embedder)
    retriever = Retriever(
        {c.chunk_id: c for c in sample_chunks},
        build_bm25(sample_chunks),
        dense,
        embedder=embedder,
        strategy="fake",
    )
    text = sample_chunks[0].text
    questions = [
        Question(
            "q1", "dbt incremental models", "conceptual", (Evidence("dbt/inc", 0, len(text), text),)
        )
    ]
    by_doc = index_chunks_by_doc(sample_chunks)
    common = dict(kinds=("flat", "hnsw", "ivf"), repeats=1, retriever=retriever, embedder=embedder)
    out = run_bench(questions=questions, by_doc=by_doc, **common)
    assert out["config"]["mode"] == "real"
    assert out["config"]["e2e_source"] == "measured:hybrid_score_weighted"
    assert out["verdicts"][0]["exact_enough"] is not None
    assert all(r["e2e"]["n_scored"] == 1 for r in out["rows"])
    assert retriever.dense is dense and retriever.embedder is embedder
    given = run_bench(
        questions=questions, by_doc=by_doc, e2e_p95_ms=5.0, e2e_source="flag", **common
    )
    assert given["config"]["e2e_source"] == "flag" and given["verdicts"][0]["e2e_p95_ms"] == 5.0
