"""The semcache-sweep harness. Synthetic inputs and hand-set vectors; the committed files are
only read, as data checks, and no test asserts a measured number."""

from __future__ import annotations

import hashlib
import inspect
import json
import re
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest

from production_rag import cli
from production_rag import semcache_eval as se
from production_rag.dense import DEFAULT_MODEL
from production_rag.groundtruth import Evidence, Question, load_questions
from production_rag.pipeline import Retriever
from production_rag.rewrite import REWRITE_SYSTEM, REWRITE_TEMPLATE
from production_rag.semcache import SemanticCache
from test_semcache import DictEmbedder

ROOT = Path(__file__).resolve().parents[1]


def question(qid, text, docs=("d",)):
    return Question(qid, text, "c", tuple(Evidence(d, 0, 1, "q") for d in docs))


def rewrite_entry(text, payload, model="writer"):
    key = {"system": REWRITE_SYSTEM, "prompt": REWRITE_TEMPLATE.format(n=se.REWRITE_N, query=text)}
    return {"key": key, "value": {"text": payload, "model": model}}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# -- pinned constants and wiring --------------------------------------------------
def test_pinned_constants():
    assert (se.FALSE_HIT_BAR, se.MIN_HIT_RATE) == (0.02, 0.10)
    assert se.RULE == (
        "recommend the lowest threshold with false_hit_rate <= 0.02 and wrong_entry_rate <= 0.02 "
        "and hit_rate >= 0.10; otherwise recommend nothing"
    )
    assert (se.RULE_DATE, se.SEED, se.SPOT_CHECK_N) == ("2026-09-30", 20261004, 30)
    assert se.NEAR_MISS_SHA256 == "43a5189c68b27d5d3d0de741b17db12f5c1efc3574dbf95f0d266fa03b54fbe7"
    assert len(se.THRESHOLDS) == 30
    assert tuple(round(0.70 + 0.01 * i, 2) for i in range(30)) == se.THRESHOLDS
    assert se.THRESHOLDS[0] == 0.70 and se.THRESHOLDS[-1] == 0.99


def test_constants_match_the_code_they_mirror():
    default = inspect.signature(Retriever.retrieve).parameters["rewrite_n"].default
    assert default == se.REWRITE_N
    assert se.REWRITE_BUNDLE == cli.LLM_BUNDLE
    assert se.EMBEDDER_NAME == DEFAULT_MODEL


# -- true pairs ------------------------------------------------------------------
def test_load_true_pairs_on_a_synthetic_bundle(tmp_path):
    qs = [question(f"q{i}", f"question {i}") for i in range(5)]
    other = {"key": {"system": "something else", "prompt": "question 0"}, "value": {"text": "x"}}
    bundle = write_jsonl(
        tmp_path / "b.jsonl",
        [
            other,
            rewrite_entry("question 0", '{"queries": ["alpha zero", "beta zero"]}', "m1"),
            rewrite_entry("question 1", "not json at all"),
            rewrite_entry("question 2", '{"queries": ["  QUESTION   2 ", "gamma two"]}', "m2"),
            rewrite_entry("question 3", '{"queries": ["delta three"]}', "m2"),
        ],
    )
    pairs, stats = se.load_true_pairs(bundle, qs)
    assert [(p.qid, p.variant, p.answered_by) for p in pairs] == [
        ("q0", "alpha zero", "m1"),
        ("q0", "beta zero", "m1"),
        ("q2", "gamma two", "m2"),
        ("q3", "delta three", "m2"),
    ]
    assert stats["matched"] == 4 and stats["unmatched"] == 1  # question 4 has no entry
    assert stats["parse_failures"] == 1 and stats["identical_dropped"] == 1
    assert stats["answered_by"] == {"m1": 2, "m2": 2}


def test_committed_bundle_matches_every_question_and_is_left_untouched():
    bundle = ROOT / "eval/cache/llm.jsonl"
    before = sha(bundle)
    qs = load_questions(ROOT / "eval/questions.jsonl")
    pairs, stats = se.load_true_pairs(bundle, qs)
    assert len(qs) == 184 and stats["matched"] == 184 and stats["unmatched"] == 0
    assert stats["pairs"] == len(pairs) == sum(stats["answered_by"].values()) > 0
    assert sha(bundle) == before


# -- near misses -------------------------------------------------------------------
def test_committed_near_miss_file_is_the_frozen_one():
    path = ROOT / "eval/near_miss.jsonl"
    qs = load_questions(ROOT / "eval/questions.jsonl")
    rows = se.load_near_misses(path, qs)
    assert len(rows) == 40 and sha(path) == se.NEAR_MISS_SHA256
    texts = {q.text for q in qs}
    assert all(r["anchor"] in texts for r in rows)
    assert not {se._norm(r["near_miss"]) for r in rows} & {se._norm(t) for t in texts}
    assert len({r["id"] for r in rows}) == len({r["anchor"] for r in rows}) == 40


GOOD = {"id": "nm-1", "anchor": "read parquet", "near_miss": "write parquet", "why": "verb"}


@pytest.mark.parametrize(
    ("bad", "fragment"),
    [
        ({**GOOD, "id": "nm-0"}, "duplicate id"),
        ({**GOOD, "id": "nm-2", "why": "  "}, "'why'"),
        ({k: v for k, v in GOOD.items() if k != "id"}, "'id'"),
        ({**GOOD, "id": "nm-2", "near_miss": " READ   parquet"}, "same question"),
        ({**GOOD, "id": "nm-2", "anchor": "not in corpus"}, "verbatim corpus"),
        ({**GOOD, "id": "nm-2", "near_miss": "READ csv"}, "corpus question"),
        ({**GOOD, "id": "nm-2", "anchor": "read csv", "near_miss": "x"}, "duplicate anchor"),
    ],
)
def test_near_miss_errors_name_the_line(tmp_path, bad, fragment):
    qs = [question("q1", "read parquet"), question("q2", "read csv")]
    path = tmp_path / "nm.jsonl"
    first = {**GOOD, "id": "nm-0", "anchor": "read csv", "near_miss": "write csv"}
    path.write_text(json.dumps(first) + "\n" + json.dumps(bad) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"{path}:2")) as err:
        se.load_near_misses(path, qs)
    assert fragment in str(err.value)


def test_near_miss_bad_json_and_missing_file(tmp_path):
    path = tmp_path / "nm.jsonl"
    path.write_text(json.dumps(GOOD) + "\n{oops\n", encoding="utf-8")
    with pytest.raises(ValueError, match=re.escape(f"{path}:2: bad JSON")):
        se.load_near_misses(path, [question("q1", "read parquet")])
    with pytest.raises(FileNotFoundError, match="frozen file is missing"):
        se.load_near_misses(tmp_path / "absent.jsonl", [])


# -- paraphrase checks and the spot-check sample ---------------------------------------
def pairs_of(n):
    return [se.TruePair(f"q{i}", f"question {i}", f"variant {i}", "m") for i in range(n)]


def test_paraphrase_checks_exclude_false_ignore_null_and_reject_strangers(tmp_path):
    pairs = pairs_of(4)

    def row(i, verdict):
        return {"qid": f"q{i}", "question": f"question {i}", "variant": f"variant {i}"} | {
            "same_intent": verdict,
            "note": "",
        }

    path = write_jsonl(tmp_path / "c.jsonl", [row(0, True), row(1, False), row(2, None)])
    checks = se.load_paraphrase_checks(path, pairs)
    assert checks["excluded"] == {("question 1", "variant 1")}
    assert checks["undecided"] == 1 and len(checks["keys"]) == 3
    stranger = row(9, True)
    path = write_jsonl(tmp_path / "d.jsonl", [row(0, True), stranger])
    with pytest.raises(ValueError, match=re.escape(f"{path}:2: ") + ".*not a true pair"):
        se.load_paraphrase_checks(path, pairs)


def test_spot_check_sample_is_seeded_and_never_overwrites(tmp_path):
    pairs = pairs_of(50)
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    first = se.write_spot_check_sample(pairs, 7, a)
    assert first == se.write_spot_check_sample(pairs, 7, b)
    assert a.read_text() == b.read_text()
    assert first != se.write_spot_check_sample(pairs, 7, tmp_path / "c.jsonl", seed=1)
    rows = [json.loads(line) for line in a.read_text().splitlines()]
    assert len(rows) == 7 and all(r["same_intent"] is None and r["note"] == "" for r in rows)
    before = a.read_bytes()
    with pytest.raises(FileExistsError):
        se.write_spot_check_sample(pairs, 7, a)
    assert a.read_bytes() == before


def test_wilson_upper():
    assert se.wilson_upper(0, 40) == pytest.approx(0.0876, abs=1e-3)
    assert se.wilson_upper(0, 0) is None
    assert se.wilson_upper(40, 40) == pytest.approx(1.0)


# -- the sweep, on hand-set vectors ------------------------------------------------------
class World:
    """Cache questions Q0.. on unit axes; items score exactly `sim` against one of them."""

    def __init__(self, n_cache, pairs, near, docs=None):
        self.texts = [f"Q{i}" for i in range(n_cache)]
        vectors = {t: [float(i == j) for j in range(n_cache + 1)] for i, t in enumerate(self.texts)}

        def at(target, sim):  # dot with axis `target` is sim; the rest is on a spare axis
            v = [0.0] * (n_cache + 1)
            v[target], v[n_cache] = sim, (1 - sim * sim) ** 0.5
            return v

        self.pairs = []
        for k, (orig, target, sim) in enumerate(pairs):
            vectors[f"var{k}"] = at(target, sim)
            self.pairs.append(se.TruePair(f"q{orig}", f"Q{orig}", f"var{k}", "m"))
        self.near = []
        for k, (anchor, target, sim) in enumerate(near):
            vectors[f"nm{k}"] = at(target, sim)
            self.near.append(
                {"id": f"nm-{k:03d}", "anchor": f"Q{anchor}", "near_miss": f"nm{k}", "why": "w"}
            )
        docs = docs or {}
        self.questions = [
            question(f"q{i}", t, docs.get(i, (f"doc{i}",))) for i, t in enumerate(self.texts)
        ]
        self.embedder = DictEmbedder(vectors, name="world")


BASE_PAIRS = [(0, 0, 0.95), (0, 0, 0.85), (1, 1, 0.75), (0, 2, 0.91), (2, 2, 0.60)]
BASE_NEAR = [(0, 0, 0.925), (1, 2, 0.78), (2, 2, 0.50), (1, 1, 0.72)]


def run(world, thresholds, **kwargs):
    return se.sweep(
        world.embedder,
        world.pairs,
        world.near,
        questions=world.questions,
        thresholds=thresholds,
        **kwargs,
    )


def test_sweep_counts_are_exact_and_monotone():
    world = World(3, BASE_PAIRS, BASE_NEAR, docs={0: ("shared", "a"), 2: ("shared",)})
    rows = {r["threshold"]: r for r in run(world, (0.70, 0.80, 0.90, 0.93))["rows"]}
    got = {t: (r["correct_hits"], r["wrong_entry_hits"], r["false_hits"]) for t, r in rows.items()}
    assert got == {0.70: (3, 1, 3), 0.80: (2, 1, 1), 0.90: (1, 1, 1), 0.93: (1, 0, 0)}
    assert rows[0.70]["hit_rate"] == 3 / 5 and rows[0.70]["wrong_entry_rate"] == 1 / 5
    assert rows[0.70]["false_hit_rate"] == 3 / 4 and rows[0.70]["n_true"] == 5
    assert rows[0.90]["wrong_entry_shared_evidence"] == 1  # Q0 and Q2 share a doc
    assert rows[0.93]["wrong_entry_shared_evidence"] == 0
    grid = run(world, se.THRESHOLDS)["rows"]
    for a, b in pairwise(grid):
        assert a["hit_rate"] >= b["hit_rate"] and a["false_hit_rate"] >= b["false_hit_rate"]


def test_sweep_items_name_the_top1_for_each_near_miss():
    result = run(World(3, BASE_PAIRS, BASE_NEAR), (0.7,))
    items = result["items"]
    assert [i["id"] for i in items] == ["nm-000", "nm-001", "nm-002", "nm-003"]
    assert [i["top1_qid"] for i in items] == ["q0", "q2", "q2", "q1"]
    assert [i["top1_is_anchor"] for i in items] == [True, False, True, True]
    assert set(result) == {"config", "rows", "verdicts", "items"}


@pytest.mark.parametrize("t", [0.70, 0.80, 0.93])
def test_classification_agrees_with_real_lookups(t):
    world = World(3, BASE_PAIRS, BASE_NEAR)
    cache = SemanticCache(world.embedder, t)
    for text in world.texts:
        cache.insert(text, {"text": text})
    correct = wrong = 0
    for p in world.pairs:
        hit = cache.lookup(p.variant)
        correct += bool(hit) and hit.question == p.question
        wrong += bool(hit) and hit.question != p.question
    false = sum(cache.lookup(r["near_miss"]) is not None for r in world.near)
    row = run(world, (t,))["rows"][0]
    assert (row["correct_hits"], row["wrong_entry_hits"], row["false_hits"]) == (
        correct,
        wrong,
        false,
    )


def test_recommend_picks_the_lowest_qualifying_threshold_or_says_why():
    def row(t, fh, we, hr):
        return {"threshold": t, "false_hit_rate": fh, "wrong_entry_rate": we, "hit_rate": hr}

    rows = [row(0.8, 0.05, 0.0, 0.5), row(0.85, 0.0, 0.0, 0.5), row(0.9, 0.0, 0.0, 0.3)]
    assert se.recommend(rows)[0] == 0.85
    assert se.recommend([row(0.8, 0.02, 0.02, 0.10)])[0] == 0.8  # every bar is inclusive
    for bad in (row(0.8, 0.025, 0, 1), row(0.8, 0, 0.025, 1), row(0.8, 0, 0, 0.09)):
        threshold, reason = se.recommend([bad])
        assert threshold is None and "no threshold" in reason
    assert se.recommend([row(0.8, None, None, None)])[0] is None


@pytest.fixture
def eligible(tmp_path, monkeypatch):
    """Build a run that satisfies every condition for a verdict that counts, on synthetic files."""

    def build(pairs=BASE_PAIRS, near=BASE_NEAR, n_cache=3):
        world = World(n_cache, pairs, near)
        files = {name: write_jsonl(tmp_path / f"{name}.jsonl", [{}]) for name in ("q", "b")}
        nm = write_jsonl(tmp_path / "nm.jsonl", world.near)
        checks = tmp_path / "checks.jsonl"
        se.write_spot_check_sample(world.pairs, 3, checks)
        rows = [json.loads(x) | {"same_intent": True} for x in checks.read_text().splitlines()]
        write_jsonl(checks, rows)
        monkeypatch.setattr(se, "QUESTIONS_PATH", files["q"])
        monkeypatch.setattr(se, "REWRITE_BUNDLE", files["b"])
        monkeypatch.setattr(se, "NEAR_MISS_PATH", nm)
        monkeypatch.setattr(se, "NEAR_MISS_SHA256", sha(nm))
        monkeypatch.setattr(se, "EMBEDDER_NAME", world.embedder.name)
        monkeypatch.setattr(se, "SPOT_CHECK_N", 3)
        world.inputs = {
            "paths": {"questions": files["q"], "bundle": files["b"], "near_miss": nm},
            "checks": se.load_paraphrase_checks(checks, world.pairs),
        }
        world.checks_path = checks
        return world

    return build


def test_eligible_verdict_picks_the_lowest_qualifying_threshold(eligible):
    world = eligible()
    verdict = run(world, se.THRESHOLDS, inputs=world.inputs)["verdicts"][0]
    assert verdict["eligible"] is True and verdict["recommended_threshold"] == 0.93
    assert verdict["hit_rate"] == 1 / 5 and verdict["false_hit_rate"] == 0.0


def test_a_threshold_that_passes_near_misses_but_fails_wrong_entry_is_skipped(eligible):
    world = eligible(pairs=[(0, 0, 0.98), (0, 0, 0.85), (1, 1, 0.75), (0, 2, 0.965), (2, 2, 0.6)])
    result = run(world, se.THRESHOLDS, inputs=world.inputs)
    rows = {r["threshold"]: r for r in result["rows"]}
    assert rows[0.93]["false_hit_rate"] == 0.0 and rows[0.93]["wrong_entry_rate"] == 1 / 5
    assert result["verdicts"][0]["recommended_threshold"] == 0.97


def test_hit_rate_counts_correct_hits_only(eligible):
    pairs = [(0, 0, 0.95)] * 5 + [(0, 1, 0.95)] + [(0, 0, 0.6)] * 54
    world = eligible(pairs=pairs, near=[(0, 0, 0.5)])
    result = run(world, se.THRESHOLDS, inputs=world.inputs)
    row = result["rows"][0]
    assert row["n_true"] == 60 and row["hit_rate"] == 5 / 60
    assert (row["correct_hits"] + row["wrong_entry_hits"]) / 60 >= se.MIN_HIT_RATE
    verdict = result["verdicts"][0]
    assert verdict["eligible"] is True and verdict["recommended_threshold"] is None
    assert "hit_rate fails" in verdict["reason"]


def test_shared_evidence_is_a_diagnostic_and_never_moves_the_verdict(eligible):
    world = eligible()
    base = run(world, se.THRESHOLDS, inputs=world.inputs)
    world.questions = [question(q.qid, q.text, ("one doc",)) for q in world.questions]
    other = run(world, se.THRESHOLDS, inputs=world.inputs)
    assert base["rows"][0]["wrong_entry_shared_evidence"] == 0
    assert other["rows"][0]["wrong_entry_shared_evidence"] == 1
    assert other["verdicts"] == base["verdicts"]


def test_every_override_and_smoke_make_the_run_a_diagnostic(eligible, tmp_path, monkeypatch):
    world = eligible()
    assert run(world, se.THRESHOLDS, inputs=world.inputs)["verdicts"][0]["eligible"] is True

    def diagnostic(**overrides):
        kwargs = {"inputs": world.inputs} | overrides
        result = run(world, kwargs.pop("thresholds", se.THRESHOLDS), **kwargs)
        verdict = result["verdicts"][0]
        assert verdict["eligible"] is False and verdict["recommended_threshold"] is None
        assert "diagnostic" in verdict["reason"]
        return verdict["reason"]

    assert "hashing embedder is not a measurement" in diagnostic(smoke=True)
    assert "grid" in diagnostic(thresholds=(0.9, 0.95))
    monkeypatch.setattr(se, "EMBEDDER_NAME", "some-other-embedder")
    assert "embedder" in diagnostic()
    monkeypatch.setattr(se, "EMBEDDER_NAME", world.embedder.name)
    other_nm = write_jsonl(tmp_path / "other.jsonl", world.near[:2])
    paths = world.inputs["paths"] | {"near_miss": other_nm}
    assert "frozen" in diagnostic(inputs=world.inputs | {"paths": paths})
    paths = world.inputs["paths"] | {"questions": tmp_path / "elsewhere.jsonl"}
    assert "default path" in diagnostic(inputs=world.inputs | {"paths": paths})
    assert "paraphrase checks" in diagnostic(inputs=world.inputs | {"checks": None})
    partial = se.load_paraphrase_checks(world.checks_path, world.pairs)
    partial["keys"] = set(list(partial["keys"])[:2])
    assert "paraphrase checks" in diagnostic(inputs=world.inputs | {"checks": partial})
    undecided = {**world.inputs["checks"], "undecided": 1}
    assert "paraphrase checks" in diagnostic(inputs=world.inputs | {"checks": undecided})


def test_spot_check_exclusions_shrink_n_true():
    world = World(3, BASE_PAIRS, BASE_NEAR)
    dropped = {(world.pairs[0].question, world.pairs[0].variant)}
    checks = {"keys": dropped, "excluded": dropped, "undecided": 0}
    result = run(world, (0.7,), inputs={"checks": checks})
    assert result["rows"][0]["n_true"] == 4 and result["config"]["spot_check"]["excluded"] == 1
    assert result["config"]["n_true_before_exclusion"] == 5


def test_hash_embedder_is_deterministic_and_unit_norm():
    emb = se.HashEmbedder(64)
    assert emb.name == "hash-smoke-64" and emb.dim == 64
    a = emb.encode_query("read parquet files")
    assert np.array_equal(a, emb.encode_query("read parquet files"))
    assert np.linalg.norm(a) == pytest.approx(1.0, abs=1e-5)


# -- the command ---------------------------------------------------------------------------
@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    qs = [question(f"q{i}", f"how do I configure option {i} in the tool?") for i in range(4)]
    (tmp_path / "questions.jsonl").write_text(
        "".join(json.dumps(q.as_dict()) + "\n" for q in qs), encoding="utf-8"
    )
    entries = [
        rewrite_entry(q.text, json.dumps({"queries": [f"option {i} setup", f"set up {i}"]}))
        for i, q in enumerate(qs)
    ]
    write_jsonl(tmp_path / "bundle.jsonl", entries)
    near = [
        {"id": "nm-1", "anchor": qs[0].text, "near_miss": "how do I remove option 0?", "why": "w"},
        {"id": "nm-2", "anchor": qs[1].text, "near_miss": "how do I remove option 1?", "why": "w"},
    ]
    write_jsonl(tmp_path / "nm.jsonl", near)
    return ["--questions", "questions.jsonl", "--bundle", "bundle.jsonl"]


def test_cli_smoke_run_writes_one_file(synthetic, tmp_path):
    before = {p.name for p in tmp_path.iterdir()}
    argv = ["semcache-sweep", *synthetic, "--near-miss", "nm.jsonl", "--fake-embedder"]
    assert cli.main([*argv, "--out", "s.json"]) == 0
    result = json.loads((tmp_path / "s.json").read_text())
    assert set(result) == {"config", "rows", "verdicts", "items"} and len(result["rows"]) == 30
    assert result["verdicts"][0]["recommended_threshold"] is None
    assert result["verdicts"][0]["eligible"] is False and result["config"]["mode"] == "smoke"
    assert {p.name for p in tmp_path.iterdir()} == before | {"s.json"}


def test_cli_missing_near_miss_file_fails_cleanly(synthetic, capsys):
    argv = ["semcache-sweep", *synthetic, "--near-miss", "absent.jsonl", "--fake-embedder"]
    assert cli.main(argv) == 1
    err = capsys.readouterr().err
    assert err.startswith("semcache-sweep: ") and "frozen file is missing" in err
    assert "Traceback" not in err


def test_cli_sample_paraphrases_writes_a_template_and_never_overwrites(synthetic, tmp_path, capsys):
    argv = ["semcache-sweep", *synthetic, "--sample-paraphrases", "3"]
    argv += ["--paraphrase-checks", "checks.jsonl"]
    assert cli.main(argv) == 0
    path = tmp_path / "checks.jsonl"
    rows = [json.loads(x) for x in path.read_text().splitlines()]
    assert len(rows) == 3 and all(r["same_intent"] is None for r in rows)
    before = path.read_bytes()
    assert cli.main(argv) == 1
    assert "semcache-sweep:" in capsys.readouterr().err and path.read_bytes() == before


def test_cli_threshold_flag_replaces_the_grid(synthetic, tmp_path):
    argv = ["semcache-sweep", *synthetic, "--near-miss", "nm.jsonl", "--fake-embedder"]
    assert cli.main([*argv, "--threshold", "0.8", "--threshold", "0.9", "--out", "t.json"]) == 0
    result = json.loads((tmp_path / "t.json").read_text())
    assert [r["threshold"] for r in result["rows"]] == [0.8, 0.9]
