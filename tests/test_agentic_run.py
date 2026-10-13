"""The command that runs the agentic arm -- still with fakes only.

What matters here is what happens before a model is touched and what a dry run is: the command
must stop before any language-model call on every mismatch, the dry run must make none, a failed
gate-1 check must stop the live run too, the result file must exist only when the run completed,
and an interrupted run must resume from the cache without asking the model the same thing twice.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import pytest

from production_rag import agentic as ag
from production_rag import cli
from production_rag.cache import JsonCache
from production_rag.cli import main
from production_rag.providers import CachedProvider, OllamaProvider
from test_agentic import MODEL, World


class Stub:
    """Wires the command to a fake world and records what it touched."""

    def __init__(self, monkeypatch, tmp_path, *, dirty=False, ollama=None, world=None, swap=None):
        self.world = world or World()
        self.inner = self.world.provider()
        self.inner.host = "http://ollama.test"
        self.provider = CachedProvider(self.inner, JsonCache(tmp_path / "llm-cache"))
        self.loaded = 0
        retrieve = self.world.retrieve

        def retrieve_fake(query, arm, k, pool):
            assert (arm, k, pool) == (ag.ARM, ag.POOL, ag.POOL)
            ranked, seconds = retrieve(query)
            if swap and query == swap:
                ranked[0], ranked[1] = ranked[1], ranked[0]
            return SimpleNamespace(ranked=ranked, latency_s=seconds)

        retriever = SimpleNamespace(retrieve=retrieve_fake)

        def load(*args, **kwargs):
            self.loaded += 1
            return retriever

        monkeypatch.setattr(cli, "_git_state", lambda: ("abc1234def", dirty))
        monkeypatch.setattr(cli, "_ollama_ready", lambda host, model: (ollama, "sha256:test"))
        monkeypatch.setattr(
            cli, "load_chunks", lambda root, strategy: list(self.world.chunks.values())
        )
        monkeypatch.setattr(ag, "load_inputs", lambda chunks, by_doc, **kw: self.world.inputs)
        monkeypatch.setattr(cli, "_embedder", lambda args: None)
        monkeypatch.setattr(cli, "Retriever", SimpleNamespace(load=load))
        monkeypatch.setattr(cli, "_agentic_provider", lambda args: self.provider)

    @property
    def untouched(self) -> bool:
        return not self.inner.prompts

    @property
    def judge_prompts(self) -> list[str]:
        return self.inner.judge_prompts()


def command(tmp_path, *extra, out="agentic.json") -> int:
    return main(
        ["agentic-run", "--out", str(tmp_path / out), "--llm-cache", str(tmp_path / "c"), *extra]
    )


# ---------------------------------------------------------------------------
# dry run
# ---------------------------------------------------------------------------
def test_the_dry_run_checks_gate_one_prints_the_budget_and_makes_no_model_call(
    tmp_path, monkeypatch, capsys
):
    stub = Stub(monkeypatch, tmp_path)
    assert command(tmp_path, "--dry-run") == 0
    out = capsys.readouterr().out
    assert stub.untouched
    assert "gate 1: ok, 16 of 16 step-1 rankings equal the stored top 10" in out
    assert "questions run through the loop: 16" in out
    assert "questions scored on retrieval (have a gold chunk): 14" in out
    assert "'unanswerable': 0" not in out  # empty categories are not listed
    assert "planned judge calls: 16 to 48" in out
    assert "planned answer generations: 0 to 16" in out  # 8 subset questions
    assert "dry run: no language-model call was made" in out
    assert not (tmp_path / "agentic.json").exists()


def test_the_dry_run_needs_neither_a_clean_tree_nor_a_listed_model(tmp_path, monkeypatch, capsys):
    stub = Stub(monkeypatch, tmp_path, dirty=True, ollama="model is not available")
    assert command(tmp_path, "--dry-run") == 0
    out = capsys.readouterr().out
    assert "(dirty)" in out and "ollama: model is not available" in out and stub.untouched


def test_a_failed_gate_one_stops_the_dry_run_and_the_live_run_before_any_call(
    tmp_path, monkeypatch, capsys
):
    stub = Stub(monkeypatch, tmp_path, swap="question number 3?")
    assert command(tmp_path, "--dry-run") == 1
    assert command(tmp_path) == 1
    captured = capsys.readouterr()
    assert "gate 1: FAILED" in captured.out and "gate 1 failed" in captured.err
    assert "q3" in captured.err and stub.untouched
    assert not (tmp_path / "agentic.json").exists()


# ---------------------------------------------------------------------------
# fail closed
# ---------------------------------------------------------------------------
def test_an_existing_result_stops_the_command_before_anything_else(tmp_path, monkeypatch, capsys):
    (tmp_path / "agentic.json").write_text("{}", encoding="utf-8")
    stub = Stub(monkeypatch, tmp_path)
    assert command(tmp_path) == 1
    assert "already exists" in capsys.readouterr().err
    assert stub.untouched and stub.loaded == 0
    assert (tmp_path / "agentic.json").read_text("utf-8") == "{}"


def test_a_dirty_tree_stops_the_live_run_unless_allowed_and_the_allowance_is_recorded(
    tmp_path, monkeypatch, capsys
):
    stub = Stub(monkeypatch, tmp_path, dirty=True)
    assert command(tmp_path) == 1
    assert "not clean" in capsys.readouterr().err and stub.untouched and stub.loaded == 0
    assert command(tmp_path, "--allow-dirty") == 0
    saved = json.loads((tmp_path / "agentic.json").read_text("utf-8"))
    assert saved["config"]["git_dirty"] is True


def test_a_local_model_that_is_not_listed_stops_the_live_run_before_any_call(
    tmp_path, monkeypatch, capsys
):
    stub = Stub(monkeypatch, tmp_path, ollama="qwen is not available at http://ollama.test")
    assert command(tmp_path) == 1
    assert "not available" in capsys.readouterr().err
    assert stub.untouched and stub.loaded == 0
    assert not (tmp_path / "agentic.json").exists()


def test_missing_indexes_stop_the_command(tmp_path, monkeypatch, capsys):
    stub = Stub(monkeypatch, tmp_path)

    def missing(root, strategy):
        raise FileNotFoundError(f"{root}: no chunks")

    monkeypatch.setattr(cli, "load_chunks", missing)
    assert command(tmp_path) == 1
    assert "no chunks" in capsys.readouterr().err and stub.untouched


def test_inputs_that_do_not_validate_stop_the_command(tmp_path, monkeypatch, capsys):
    stub = Stub(monkeypatch, tmp_path)

    def bad(chunks, by_doc, **kw):
        raise ValueError("subset is 53 answerable + 7 unanswerable")

    monkeypatch.setattr(ag, "load_inputs", bad)
    assert command(tmp_path) == 1
    assert "53 answerable" in capsys.readouterr().err and stub.untouched


def test_the_shared_llm_cache_is_refused_because_it_may_hold_the_published_answers(
    tmp_path, monkeypatch, capsys
):
    stub = Stub(monkeypatch, tmp_path)
    assert (
        main(["agentic-run", "--out", str(tmp_path / "a.json"), "--llm-cache", ".cache/llm"]) == 1
    )
    assert "published answers" in capsys.readouterr().err and stub.untouched


def test_the_command_takes_no_model_choice():
    options = {
        o
        for action in cli.build_parser()
        ._subparsers._group_actions[0]
        .choices["agentic-run"]
        ._actions
        for o in action.option_strings
    }
    assert "--model-arm" not in options
    flags = {"--out", "--dry-run", "--workers", "--phase", "--allow-dirty"}
    assert flags | {"--accept-provider-errors", "--replay", "--bundle"} <= options


def test_the_default_cache_is_the_runs_own_and_the_default_workers_are_four():
    args = cli.build_parser().parse_args(["agentic-run"])
    assert args.llm_cache == cli.AGENTIC_LLM_CACHE != cli.LLM_CACHE
    assert args.workers == 4 and args.out == ag.OUT_PATH and args.phase == "all"


# ---------------------------------------------------------------------------
# the live run
# ---------------------------------------------------------------------------
def test_the_command_runs_end_to_end_and_writes_one_complete_result_once(
    tmp_path, monkeypatch, capsys
):
    stub = Stub(monkeypatch, tmp_path)
    assert command(tmp_path) == 0
    saved = json.loads((tmp_path / "agentic.json").read_text("utf-8"))
    config = saved["config"]
    assert config["git_commit"] == "abc1234def" and config["git_dirty"] is False
    assert config["ollama_digest"] == "sha256:test" and config["workers"] == 4
    assert config["protocol_commit"] == "4523723" and config["model"] == MODEL
    assert config["judge_calls"] == 12 * 2 + 2 + 2 * 3 and config["judge_calls_from_cache"] == 0
    assert config["answer_generations"] == 4 * 2 + 2 and config["answer_retries"] == 0
    assert config["llm_calls"] == len(stub.inner.prompts) and config["llm_calls_from_cache"] == 0
    result = saved["result"]
    assert result["verdict"] == ag.GAIN and result["complete"]
    assert [g["passed"] for g in result["gates"]] == [True] * 4
    assert len(saved["questions"]) == 16 and len(saved["answers"]) == 8
    assert not (tmp_path / "agentic.json.partial").exists()
    assert "Gain: delta nDCG@10" in capsys.readouterr().out
    # A second run is refused outright: the run is made once.
    stub = Stub(monkeypatch, tmp_path)
    assert command(tmp_path) == 1 and stub.untouched


def test_a_run_makes_exactly_the_planned_calls_and_no_retrieval_beyond_the_requeries(
    tmp_path, monkeypatch
):
    stub = Stub(monkeypatch, tmp_path)
    assert command(tmp_path) == 0
    # 16 questions: 12 + 2 + 2 judge calls... = 12*2 + 2*1 + 2*3
    assert len(stub.judge_prompts) == 32
    assert len(stub.inner.answer_prompts()) == 10
    asked = stub.world.retrieved
    assert asked.count("question number 0?") == 2  # the warm-up is the first question only
    assert asked.count("question number 1?") == 1


def test_the_loop_phase_writes_nothing_and_a_rerun_resumes_from_the_cache(
    tmp_path, monkeypatch, capsys
):
    stub = Stub(monkeypatch, tmp_path)
    assert command(tmp_path, "--phase", "loop") == 0
    out = capsys.readouterr().out
    assert "loop phase only: nothing written" in out and "planned answer generations: 10" in out
    assert not (tmp_path / "agentic.json").exists()
    after_loop = list(stub.judge_prompts)
    assert len(after_loop) == 32 and stub.inner.answer_prompts() == []

    assert command(tmp_path) == 0
    # The judge was not asked a single question again; only the answers were generated.
    assert stub.judge_prompts == after_loop
    assert len(stub.inner.answer_prompts()) == 10
    saved = json.loads((tmp_path / "agentic.json").read_text("utf-8"))
    assert saved["config"]["judge_calls_from_cache"] == 32
    assert saved["config"]["llm_calls_from_cache"] == 32


def test_an_interrupted_answer_phase_resumes_without_repeating_any_call(tmp_path, monkeypatch):
    stub = Stub(monkeypatch, tmp_path)
    original = stub.inner._answer
    calls = {"n": 0}

    def flaky(prompt):
        calls["n"] += 1
        if calls["n"] == 5:
            raise KeyboardInterrupt
        return original(prompt)

    stub.inner._answer = flaky
    with pytest.raises(KeyboardInterrupt):
        command(tmp_path, "--workers", "1")
    assert not (tmp_path / "agentic.json").exists()
    done = len(stub.inner.prompts)
    stub.inner._answer = original
    assert command(tmp_path, "--workers", "1") == 0
    # Nothing that had already been answered went to the model a second time (the interrupted
    # call is the one request that was never cached).
    repeated = [p for p in set(stub.inner.prompts) if stub.inner.prompts.count(p) > 1]
    assert len(repeated) == 1 and done == 32 + 5
    assert (tmp_path / "agentic.json").exists()


def down(prompt):
    raise ag.ProviderError("ollama is down")


def test_failed_answer_calls_exit_non_zero_with_no_result_and_a_rerun_retries_only_those(
    tmp_path, monkeypatch, capsys
):
    stub = Stub(monkeypatch, tmp_path)
    original = stub.inner._answer
    stub.inner._answer = down
    assert command(tmp_path) == 1
    err = capsys.readouterr().err
    assert "provider errors: 10" in err
    assert "rerun the same command to resume from the cache" in err
    assert not (tmp_path / "agentic.json").exists()
    judged = list(stub.judge_prompts)

    stub.inner._answer = original  # the model is back
    assert command(tmp_path) == 0
    assert stub.judge_prompts == judged  # the judge was not asked again
    saved = json.loads((tmp_path / "agentic.json").read_text("utf-8"))
    assert saved["result"]["verdict"] == ag.GAIN
    assert saved["config"]["provider_errors"] == {"judge": 0, "answers": 0}
    assert saved["config"]["accept_provider_errors"] is False


def test_failed_judge_calls_stop_the_run_before_the_answer_phase(tmp_path, monkeypatch, capsys):
    stub = Stub(monkeypatch, tmp_path)
    original = stub.inner._judge

    def flaky(prompt):
        if "question number 3?" in prompt:
            raise ag.ProviderError("ollama is down")
        return original(prompt)

    stub.inner._judge = flaky
    assert command(tmp_path) == 1
    assert "provider errors: 1" in capsys.readouterr().err
    assert stub.inner.answer_prompts() == [] and not (tmp_path / "agentic.json").exists()
    stub.inner._judge = original
    assert command(tmp_path) == 0  # resumed; only the failed question was asked again


def test_accepting_provider_errors_finalises_with_the_registered_handling_and_records_it(
    tmp_path, monkeypatch
):
    stub = Stub(monkeypatch, tmp_path)
    stub.inner._answer = down
    assert command(tmp_path, "--accept-provider-errors") == 0
    saved = json.loads((tmp_path / "agentic.json").read_text("utf-8"))
    assert saved["config"]["accept_provider_errors"] is True
    assert saved["config"]["provider_errors"] == {"judge": 0, "answers": 10}
    assert saved["result"]["verdict"] is None and "gate 2" in saved["result"]["verdict_text"]


def test_a_failed_judge_call_accepted_fails_open_and_counts_in_gate_three(tmp_path, monkeypatch):
    stub = Stub(monkeypatch, tmp_path)
    stub.inner._judge = down
    assert command(tmp_path, "--accept-provider-errors") == 0
    saved = json.loads((tmp_path / "agentic.json").read_text("utf-8"))
    assert saved["result"]["gates"][2]["by_kind"]["provider_error"] == 16
    assert saved["result"]["verdict"] is None


def test_the_live_provider_is_the_registered_local_model_with_no_fallback(tmp_path):
    provider = cli._agentic_provider(argparse.Namespace(llm_cache=tmp_path / "c"))
    assert type(provider) is CachedProvider and type(provider.inner) is OllamaProvider
    assert provider.inner.num_ctx == 8192 == ag.NUM_CTX
    assert provider.model == MODEL and provider.offline is False
    assert provider.cache.root == tmp_path / "c" and not (tmp_path / "c").exists()


def test_the_model_digest_is_pinned_to_the_cache_at_the_first_live_run(
    tmp_path, monkeypatch, capsys
):
    stub = Stub(monkeypatch, tmp_path)
    assert command(tmp_path, "--phase", "loop") == 0
    marker = tmp_path / "c" / "model_digest.txt"
    assert marker.read_text("utf-8").strip() == "sha256:test"
    calls = len(stub.inner.prompts)
    monkeypatch.setattr(cli, "_ollama_ready", lambda host, model: (None, "sha256:other"))
    assert command(tmp_path, "--phase", "loop") == 1
    assert "sha256:test" in capsys.readouterr().err and len(stub.inner.prompts) == calls
    monkeypatch.setattr(cli, "_ollama_ready", lambda host, model: (None, "sha256:test"))
    assert command(tmp_path, "--phase", "loop") == 0  # the same weights resume


def test_a_model_with_no_digest_cannot_be_pinned(tmp_path):
    with pytest.raises(ValueError, match="no digest"):
        cli._check_model_digest(tmp_path, "")


def test_the_marker_is_not_read_as_a_cache_entry_when_the_cache_is_exported(tmp_path):
    cache = JsonCache(tmp_path / "c")
    cache.put({"k": 1}, {"v": 1})
    cli._check_model_digest(tmp_path / "c", "sha256:x")
    assert cache.export_jsonl(tmp_path / "bundle.jsonl") == 1


def test_a_baseline_that_does_not_match_the_tables_metrics_stops_the_run_before_any_call(
    tmp_path, monkeypatch, capsys
):
    from dataclasses import replace

    world = World()
    specs = world.inputs.specs
    bad = [
        replace(specs[0], stored_metrics={**specs[0].stored_metrics, "ndcg@10": 0.9}),
        *specs[1:],
    ]
    world.inputs = ag.Inputs(
        bad, world.inputs.subset, world.inputs.published, world.chunks, world.inputs.by_doc
    )
    stub = Stub(monkeypatch, tmp_path, world=world)
    assert command(tmp_path, "--dry-run") == 1
    assert command(tmp_path) == 1
    assert "baseline metrics" in capsys.readouterr().err and stub.untouched and stub.loaded == 0


def test_the_result_records_the_inputs_and_every_argument(tmp_path, monkeypatch):
    Stub(monkeypatch, tmp_path)
    assert command(tmp_path, "--workers", "2") == 0
    config = json.loads((tmp_path / "agentic.json").read_text("utf-8"))["config"]
    assert set(config["inputs_sha256"]) == {"questions", "results", "answers"}
    assert all(len(v) == 64 for v in config["inputs_sha256"].values())
    assert config["args"]["workers"] == 2 and config["args"]["phase"] == "all"
    assert config["args"]["llm_cache"] == str(tmp_path / "c") and "func" not in config["args"]
    assert config["replay"] is False and config["bundle"] is None


def test_an_answer_retry_is_counted_once_per_row_that_needed_it(tmp_path, monkeypatch):
    stub = Stub(monkeypatch, tmp_path)
    original, seen = stub.inner._answer, set()

    def empty_first(prompt):
        if prompt not in seen:
            seen.add(prompt)
            return "{}"  # valid JSON, no answer: `generate` retries once without JSON mode
        return original(prompt)

    stub.inner._answer = empty_first
    assert command(tmp_path) == 0
    config = json.loads((tmp_path / "agentic.json").read_text("utf-8"))["config"]
    assert config["answer_generations"] == 10 and config["answer_retries"] == 10


def test_the_agentic_cache_exports_to_a_bundle_the_run_can_replay_with_no_model(
    tmp_path, monkeypatch, capsys
):
    assert cli.CACHES["agentic"] == (cli.AGENTIC_LLM_CACHE, cli.AGENTIC_BUNDLE)
    assert str(cli.AGENTIC_BUNDLE).replace("\\", "/") == "eval/cache/agentic_llm.jsonl"
    stub = Stub(monkeypatch, tmp_path)
    assert command(tmp_path) == 0
    first = json.loads((tmp_path / "agentic.json").read_text("utf-8"))
    bundle = tmp_path / "agentic_llm.jsonl"
    assert JsonCache(tmp_path / "llm-cache").export_jsonl(bundle) == len(stub.inner.prompts)

    def never(*args, **kwargs):
        raise AssertionError("a replay must not look for a model")

    monkeypatch.setattr(cli, "_ollama_ready", never)
    monkeypatch.setattr(cli, "_git_state", lambda: ("abc1234def", True))  # dirty: replay allows it
    asked = len(stub.inner.prompts)
    out = ["--out", str(tmp_path / "replayed.json"), "--bundle", str(bundle)]
    assert main(["agentic-run", "--replay", *out]) == 0
    assert len(stub.inner.prompts) == asked  # the fake model was never reached
    replayed = json.loads((tmp_path / "replayed.json").read_text("utf-8"))
    assert replayed["config"]["replay"] is True
    assert replayed["config"]["judge_calls_from_cache"] == replayed["config"]["judge_calls"] == 32
    assert replayed["result"]["verdict"] == first["result"]["verdict"] == ag.GAIN
    assert replayed["result"]["retrieval"]["overall"] == first["result"]["retrieval"]["overall"]
    assert replayed["result"]["answers"]["guardrail"] == first["result"]["answers"]["guardrail"]


def test_a_replay_that_lacks_a_request_stops_and_writes_nothing(tmp_path, monkeypatch, capsys):
    Stub(monkeypatch, tmp_path)
    empty = tmp_path / "empty.jsonl"
    empty.write_text("", encoding="utf-8")
    out = ["--out", str(tmp_path / "replayed.json"), "--bundle", str(empty)]
    assert main(["agentic-run", "--replay", *out]) == 1
    assert "replay misses" in capsys.readouterr().err
    assert not (tmp_path / "replayed.json").exists()
    assert (
        main(
            [
                "agentic-run",
                "--replay",
                "--out",
                str(tmp_path / "r.json"),
                "--bundle",
                str(tmp_path / "gone"),
            ]
        )
        == 1
    )


def test_a_failed_judge_run_is_counted_against_gate_three(tmp_path, monkeypatch):
    stub = Stub(monkeypatch, tmp_path)
    stub.inner._judge = lambda prompt: "not json"
    assert command(tmp_path) == 0
    saved = json.loads((tmp_path / "agentic.json").read_text("utf-8"))
    gate3 = saved["result"]["gates"][2]
    assert not gate3["passed"] and gate3["by_kind"]["no_json_object"] == 16
    assert saved["result"]["verdict"] is None


def test_the_wrong_model_leaves_no_verdict(tmp_path, monkeypatch):
    stub = Stub(monkeypatch, tmp_path, world=World())
    stub.provider = CachedProvider(
        stub.world.provider(model="not-the-registered-model"), JsonCache(tmp_path / "other")
    )
    stub.provider.inner.host = "http://ollama.test"
    monkeypatch.setattr(cli, "_agentic_provider", lambda args: stub.provider)
    assert command(tmp_path) == 1  # refused before any call: the model is not the registered one
    assert not (tmp_path / "agentic.json").exists() and stub.provider.inner.prompts == []
