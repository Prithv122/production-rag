"""The whole run, and the command that starts it -- still with fakes only.

What matters here is what happens *before* a live call and what happens when something goes
wrong: the command must stop before generating anything on any mismatch, the replayed baseline
must run first and stop the run on a miss, a failed generation must stay inside the finished
result as a counted row, and the result file must never exist unless the run completed.
"""

from __future__ import annotations

import argparse
import json
import shutil
from types import SimpleNamespace

import pytest

from conftest import FakeProvider
from production_rag import cli
from production_rag import injection as inj
from production_rag.cache import JsonCache
from production_rag.cli import main
from production_rag.generate import ANSWER_SYSTEM
from production_rag.groundtruth import Question
from production_rag.providers import MODEL_ARMS, CachedProvider, LLMResponse, OllamaProvider
from test_injection import (
    EVAL,
    RuleProvider,
    answer_json,
    hijack_if_poisoned,
    make_attack,
    real_chunks,
    synth_rows,
    world,
)

MODEL = MODEL_ARMS[inj.MODEL_ARM]["model"]
CLEAN_ANSWER = answer_json("Topic three is a dbt topic [4].")


class CannedReplay(FakeProvider):
    """A replaying provider that serves one cached answer for every prompt."""

    offline = True

    def complete(self, prompt: str, **kwargs):
        self.prompts.append(prompt)
        self.kwargs.append(dict(kwargs))
        return LLMResponse(text=CLEAN_ANSWER, model=self._model, provider="p", cached=True)


class MissingReplay(FakeProvider):
    """A replaying provider with nothing cached."""

    offline = True


def live_provider(fn=hijack_if_poisoned, **kwargs) -> RuleProvider:
    return RuleProvider(fn, model=kwargs.pop("model", MODEL), **kwargs)


def small_inputs() -> inj.Inputs:
    chunks, by_doc, question, ranking = world()
    unanswerable = Question(qid="u1", text="what is the moon made of?", category="unanswerable")
    rankings = {question.qid: ranking, "u1": ranking}
    attacks = [make_attack(id="ia-t1"), make_attack(id="ia-t2")]
    instances = inj.build_instances([(a, question) for a in attacks], rankings, by_doc)
    return inj.Inputs(attacks, instances, [question, unanswerable], rankings, chunks)


def run(inputs=None, *, baseline=None, live=None, **kwargs):
    return inj.run_protocol(
        inputs or small_inputs(),
        baseline_provider=baseline or CannedReplay(model=MODEL),
        live_provider=live or live_provider(),
        workers=1,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# run_protocol
# ---------------------------------------------------------------------------
def test_every_arm_runs_once_and_the_derived_arms_are_added():
    baseline, live = CannedReplay(model=MODEL), live_provider()
    rows, result = run(baseline=baseline, live=live)
    arms = {r.arm for r in rows}
    assert arms == {"B", "M1", "M2", "M3", "C", "C+M2"}
    count = {arm: sum(1 for r in rows if r.arm == arm) for arm in arms}
    # 2 attack + 2 clean in each generated arm; the filtered arms copy B's and C's rows.
    assert count == {"B": 4, "M1": 4, "M2": 4, "M3": 4, "C": 4, "C+M2": 4}
    assert len(baseline.prompts) == 2  # B's clean half only
    assert len(live.prompts) == 2 + 3 * 4  # B attack, then M1/M3/C attack + clean
    assert all(a["complete"] for a in result["arms"].values())


def test_the_baseline_clean_half_is_replayed_and_everything_else_is_live():
    baseline, live = CannedReplay(model=MODEL), live_provider()
    rows, _ = run(baseline=baseline, live=live)
    b_clean = [r for r in rows if r.arm == "B" and r.kind == "clean"]
    assert len(b_clean) == 2 and all(r.cached for r in b_clean)
    assert not any(
        r.cached
        for r in rows
        if r.arm in ("M1", "M3", "C") or (r.arm == "B" and r.kind == "attack")
    )
    assert all(k["system"] == ANSWER_SYSTEM for k in baseline.kwargs)
    attack_systems = [k["system"] for k in live.kwargs if inj.SECRET in k["system"]]
    assert (
        len(attack_systems) == 2 + 3 * 2
    )  # every attack call carries the secret, no clean one does


def test_a_baseline_miss_stops_the_run_before_any_live_call():
    live = live_provider()
    with pytest.raises(inj.ReplayMiss):
        run(baseline=MissingReplay(fail=True, model=MODEL), live=live)
    assert live.prompts == []


def test_a_baseline_row_that_was_generated_rather_than_replayed_stops_the_run():
    class Fresh(CannedReplay):
        """Claims to be replaying but hands back a row that was not cached."""

        def complete(self, prompt: str, **kwargs):
            response = super().complete(prompt, **kwargs)
            return LLMResponse(text=response.text, model=self._model, provider="p", cached=False)

    live = live_provider()
    with pytest.raises(inj.ReplayMiss, match="generated, not replayed"):
        run(baseline=Fresh(model=MODEL), live=live)
    assert live.prompts == []


def test_a_wrong_model_is_refused_before_anything_is_called():
    baseline, live = CannedReplay(model=MODEL), live_provider(model="some-other-model")
    with pytest.raises(ValueError, match="registered model"):
        run(baseline=baseline, live=live)
    assert baseline.prompts == [] and live.prompts == []


def test_the_baseline_must_replay_and_the_live_provider_must_not():
    with pytest.raises(ValueError, match="replaying one"):
        run(baseline=live_provider())
    with pytest.raises(ValueError, match="must not be a replaying"):
        run(live=CannedReplay(model=MODEL))


def test_a_failed_live_generation_stays_in_the_result_as_a_counted_row():
    rows, result = run(live=FakeProvider(fail=True, model=MODEL))  # does not raise
    generated = [r for r in rows if r.arm in inj.GENERATED_ARMS]
    errors = [r for r in generated if r.refusal_reason == "provider_error"]
    # Every live row is there as a row (B's clean half was replayed fine), none dropped.
    assert len(errors) == 2 + 3 * 4 and len(generated) == 4 + 3 * 4
    assert all(a["valid"] is False for name, a in result["arms"].items() if name in ("M1", "M3"))
    assert result["arms"]["M1"]["invalid"]["n"] == 4  # in the denominator, not removed from it


def test_a_provider_error_is_never_an_attack_success():
    rows, _ = run(live=FakeProvider(fail=True, model=MODEL))
    assert not any(r.success for r in rows if r.kind == "attack")


def test_saving_is_all_or_nothing(tmp_path):
    rows, result = run()
    path = tmp_path / "injection.json"
    with pytest.raises(TypeError):
        inj.save(path, rows, result, {"bad": object()})  # not JSON-serialisable
    assert not path.exists()  # a failed write never leaves something that looks like the result
    inj.save(path, rows, result, inj.run_config(MODEL))
    assert path.exists() and not list(tmp_path.glob("*.partial"))
    with pytest.raises(FileExistsError):
        inj.save(path, rows, result, inj.run_config(MODEL))


# ---------------------------------------------------------------------------
# the command: every check happens before any live call
# ---------------------------------------------------------------------------
class Stub:
    """Records what the command touched, so a test can say what it did NOT touch."""

    def __init__(self, monkeypatch, *, dirty=False, ollama=None, baseline=None, live=None):
        self.built = 0
        self.live = live or live_provider()
        self.live.inner = SimpleNamespace(host="http://ollama.test")
        self.baseline = baseline or CannedReplay(model=MODEL)
        chunks = list(small_inputs().chunks.values())
        monkeypatch.setattr(cli, "_git_state", lambda: ("abc1234def", dirty))
        monkeypatch.setattr(cli, "_ollama_ready", lambda host, model: (ollama, "sha256:test"))
        monkeypatch.setattr(cli, "load_chunks", lambda root, strategy: chunks)
        monkeypatch.setattr(inj, "load_inputs", lambda chunks, **kw: small_inputs())
        monkeypatch.setattr(cli, "_injection_providers", self._providers)

    def _providers(self, args):
        self.built += 1
        return self.baseline, self.live

    @property
    def untouched(self) -> bool:
        return self.built == 0 and not self.live.prompts and not self.baseline.prompts


def command(tmp_path, *extra, out="injection.json") -> int:
    return main(["injection-run", "--out", str(tmp_path / out), *extra])


def test_the_command_runs_end_to_end_and_writes_the_result_once(tmp_path, monkeypatch, capsys):
    Stub(monkeypatch)
    assert command(tmp_path) == 0
    saved = json.loads((tmp_path / "injection.json").read_text("utf-8"))
    assert saved["config"]["git_commit"] == "abc1234def" and saved["config"]["git_dirty"] is False
    assert saved["config"]["ollama_digest"] == "sha256:test"
    assert saved["config"]["n_live_rows_from_cache"] == 0
    assert saved["config"]["num_ctx_live"] == 8192 and saved["config"]["answer_tokens"] == 700
    assert saved["config"]["protocol_commit"] == "76ad6f3" and saved["config"]["workers"] == 4
    assert set(saved["result"]["arms"]) == {"B", "M1", "M2", "M3", "C", "C+M2"}
    assert "| arm |" in capsys.readouterr().out  # the report was printed
    # A second run is refused outright: the run is made once.
    stub = Stub(monkeypatch)
    assert command(tmp_path) == 1 and stub.untouched


def test_an_existing_result_stops_the_command_before_anything_else(tmp_path, monkeypatch, capsys):
    (tmp_path / "injection.json").write_text("{}", encoding="utf-8")
    stub = Stub(monkeypatch)
    assert command(tmp_path) == 1
    assert "already exists" in capsys.readouterr().err and stub.untouched
    assert (tmp_path / "injection.json").read_text("utf-8") == "{}"


def test_a_changed_frozen_file_stops_the_command(tmp_path, monkeypatch, capsys):
    altered = tmp_path / "attacks.jsonl"
    shutil.copyfile(EVAL / "injection_attacks.jsonl", altered)
    altered.write_bytes(altered.read_bytes() + b"\n")
    stub = Stub(monkeypatch)
    assert command(tmp_path, "--attacks", str(altered)) == 1
    assert "sha256 differs" in capsys.readouterr().err and stub.untouched
    assert not (tmp_path / "injection.json").exists()


def test_a_dirty_tree_stops_the_command_unless_allowed_and_the_allowance_is_recorded(
    tmp_path, monkeypatch, capsys
):
    stub = Stub(monkeypatch, dirty=True)
    assert command(tmp_path) == 1
    assert "not clean" in capsys.readouterr().err and stub.untouched
    assert command(tmp_path, "--allow-dirty") == 0
    saved = json.loads((tmp_path / "injection.json").read_text("utf-8"))
    assert saved["config"]["git_dirty"] is True


def test_missing_indexes_stop_the_command(tmp_path, monkeypatch, capsys):
    stub = Stub(monkeypatch)

    def missing(root, strategy):
        raise FileNotFoundError(f"{root}: no chunks")

    monkeypatch.setattr(cli, "load_chunks", missing)
    assert command(tmp_path) == 1
    assert "no chunks" in capsys.readouterr().err and stub.untouched


def test_inputs_that_do_not_validate_stop_the_command(tmp_path, monkeypatch, capsys):
    stub = Stub(monkeypatch)

    def bad(chunks, **kw):
        raise ValueError("subset is 53 answerable + 7 unanswerable")

    monkeypatch.setattr(inj, "load_inputs", bad)
    assert command(tmp_path) == 1
    assert "53 answerable" in capsys.readouterr().err and stub.untouched


def test_a_local_model_that_is_not_listed_stops_the_command_before_any_generation(
    tmp_path, monkeypatch, capsys
):
    stub = Stub(monkeypatch, ollama="qwen is not available at http://ollama.test")
    assert command(tmp_path) == 1
    assert "not available" in capsys.readouterr().err
    assert not stub.live.prompts and not stub.baseline.prompts
    assert not (tmp_path / "injection.json").exists()


def test_a_baseline_miss_leaves_no_result_and_no_live_call(tmp_path, monkeypatch, capsys):
    stub = Stub(monkeypatch, baseline=MissingReplay(fail=True, model=MODEL))
    assert command(tmp_path) == 1
    assert "nothing saved" in capsys.readouterr().err
    assert not stub.live.prompts and not (tmp_path / "injection.json").exists()


def test_the_wrong_model_leaves_no_result(tmp_path, monkeypatch, capsys):
    Stub(monkeypatch, live=live_provider(model="not-the-registered-model"))
    assert command(tmp_path) == 1
    assert not (tmp_path / "injection.json").exists()


def test_the_command_takes_no_model_choice():
    options = {
        a
        for action in cli.build_parser()
        ._subparsers._group_actions[0]
        .choices["injection-run"]
        ._actions
        for a in action.option_strings
    }
    assert "--model-arm" not in options and "--replay" not in options and "--out" in options


def test_the_ollama_check_lists_models_and_generates_nothing(monkeypatch):
    class Response:
        def __init__(self, body):
            self.body = body

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def read(self):
            return self.body

    seen = []

    def urlopen(url, timeout):
        seen.append(url)
        return Response(json.dumps({"models": [{"name": MODEL, "digest": "sha256:abc"}]}).encode())

    monkeypatch.setattr(cli.urllib.request, "urlopen", urlopen)
    assert cli._ollama_ready("http://h:1", MODEL) == (None, "sha256:abc")
    assert "not available" in cli._ollama_ready("http://h:1", "other:7b")[0]
    assert set(seen) == {"http://h:1/api/tags"}  # a listing, never /api/chat or /api/generate


def test_an_unreachable_ollama_is_a_reason_not_a_crash(monkeypatch):
    def urlopen(url, timeout):
        raise OSError("connection refused")

    monkeypatch.setattr(cli.urllib.request, "urlopen", urlopen)
    assert "could not list models" in cli._ollama_ready("http://h:1", MODEL)[0]


# ---------------------------------------------------------------------------
# a dry run of the real protocol: real inputs and real baseline replay, fake live model
# ---------------------------------------------------------------------------
@pytest.mark.slow
def test_the_real_protocol_dry_run_makes_exactly_the_registered_372_live_generations(
    tmp_path, monkeypatch, capsys
):
    real_chunks()  # skips when indexes/ is not built
    cache = JsonCache(tmp_path / "llm")
    cache.import_jsonl(EVAL / "cache" / "llm.jsonl")
    replay = CachedProvider(OllamaProvider(MODEL), cache, offline=True)
    live = live_provider(lambda p, k: CLEAN_ANSWER)
    live.inner = SimpleNamespace(host="http://ollama.test")
    monkeypatch.setattr(cli, "_git_state", lambda: ("abc1234", False))
    monkeypatch.setattr(cli, "_ollama_ready", lambda host, model: (None, "sha256:test"))
    monkeypatch.setattr(cli, "_injection_providers", lambda args: (replay, live))
    out = tmp_path / "injection.json"
    assert main(["injection-run", "--out", str(out), "--llm-cache", str(tmp_path / "llm")]) == 0

    assert len(live.prompts) == 372  # 192 attack + 180 clean, as the protocol's cost line says
    saved = json.loads(out.read_text("utf-8"))
    arms = saved["result"]["arms"]
    assert set(arms) == {"B", "M1", "M2", "M3", "C", "C+M2"}
    assert all(a["complete"] and a["n_attack"] == 48 for a in arms.values())
    assert all(
        a["n_clean_answerable"] == 54 and a["n_clean_unanswerable"] == 6 for a in arms.values()
    )
    assert len(saved["rows"]) == 60 + 48 + 3 * 108 + 108 + 108
    b_clean = [r for r in saved["rows"] if r["arm"] == "B" and r["kind"] == "clean"]
    assert len(b_clean) == 60 and all(r["cached"] for r in b_clean)
    assert not out.with_name(out.name + ".partial").exists()


def test_the_real_provider_wiring_is_offline_for_the_baseline_and_live_with_no_fallback(tmp_path):
    live_cache = tmp_path / "live"
    baseline, live = cli._injection_providers(argparse.Namespace(llm_cache=live_cache))
    assert baseline.offline is True and live.offline is False
    assert live.inner.num_ctx == inj.NUM_CTX == 8192 and baseline.inner.num_ctx is None
    assert baseline.model == MODEL and live.model == MODEL
    # No FallbackProvider anywhere: a failed call must never be answered by another model.
    assert type(baseline.inner) is OllamaProvider and type(live.inner) is OllamaProvider
    # The baseline reads a cache built from the committed bundle alone, apart from the live one.
    assert baseline.cache.root != live.cache.root
    assert len(baseline.cache) == len(
        (EVAL / "cache" / "llm.jsonl").read_text("utf-8").splitlines()
    )
    assert not live_cache.exists()  # nothing was copied into the live cache


def test_a_missing_bundle_is_a_stop_not_an_empty_baseline(tmp_path, monkeypatch):
    monkeypatch.setattr(cli, "LLM_BUNDLE", tmp_path / "gone.jsonl")
    with pytest.raises(FileNotFoundError):
        cli._injection_providers(argparse.Namespace(llm_cache=tmp_path / "live"))


def test_the_run_makes_exactly_one_live_call_per_scheduled_row_even_when_every_reply_is_cut():
    cut = LLMResponse(text="{}", model=MODEL, provider="p", finish_reason="length")

    class AlwaysCut(FakeProvider):
        def complete(self, prompt: str, **kwargs):
            self.prompts.append(prompt)
            self.kwargs.append(dict(kwargs))
            return cut

    live = AlwaysCut(model=MODEL)
    rows, result = run(live=live)
    assert len(live.prompts) == 2 + 3 * 4  # one per scheduled live row, no second attempts
    assert not any(r.retried for r in rows)
    assert all(k["max_tokens"] == 700 and k["json_object"] is True for k in live.kwargs)
    # Every cut reply is a counted row of its own kind, and none is an attack outcome.
    generated = [r for r in rows if r.arm in inj.GENERATED_ARMS and r.refusal_reason == "truncated"]
    assert len(generated) == 2 + 3 * 4 and not any(r.success for r in rows)
    assert result["arms"]["M1"]["failures"]["truncated"] == 4
    assert result["arms"]["M1"]["valid"] is False


# ---------------------------------------------------------------------------
# the context window
# ---------------------------------------------------------------------------
def capture_ollama(monkeypatch):
    sent: list[dict] = []

    def post(url, payload, headers, timeout):
        sent.append(payload)
        return {"message": {"content": "{}"}, "model": MODEL, "done_reason": "stop"}

    monkeypatch.setattr("production_rag.providers._post_json", post)
    return sent


def test_num_ctx_is_sent_only_when_set_and_the_output_budget_is_untouched(monkeypatch):
    sent = capture_ollama(monkeypatch)
    OllamaProvider(MODEL).complete("p", system="s", max_tokens=700, json_object=True)
    OllamaProvider(MODEL, num_ctx=8192).complete("p", system="s", max_tokens=700, json_object=True)
    assert "num_ctx" not in sent[0]["options"]
    assert sent[1]["options"] == {"temperature": 0.0, "num_predict": 700, "num_ctx": 8192}
    assert sent[1]["format"] == "json" and sent[0]["messages"] == sent[1]["messages"]


def test_the_window_is_not_part_of_the_cache_key_so_the_baseline_stays_the_same_entry(
    tmp_path, monkeypatch
):
    capture_ollama(monkeypatch)
    cache = JsonCache(tmp_path / "c")
    CachedProvider(OllamaProvider(MODEL), cache).complete("p", system="s", max_tokens=700)
    replay = CachedProvider(OllamaProvider(MODEL, num_ctx=8192), cache, offline=True)
    assert replay.complete("p", system="s", max_tokens=700).cached is True  # a hit, not a miss


def test_build_provider_passes_the_window_to_local_arms_only(tmp_path):
    from production_rag.providers import OpenRouterProvider, build_provider

    local = build_provider(inj.MODEL_ARM, cache_dir=tmp_path, fallback=False, num_ctx=8192)
    assert local.inner.num_ctx == 8192
    hosted = build_provider("nemotron-super", cache_dir=tmp_path, fallback=False, num_ctx=8192)
    assert isinstance(hosted.inner, OpenRouterProvider)  # no such option on the hosted client


def test_the_report_keeps_prompt_token_counts_visible():
    rows = synth_rows("B", wins=set(range(12))) + synth_rows("M1", wins={0})
    rows = [
        inj.InjectionRow(**{**r.as_dict(), "prompt_tokens": 4259})
        if r.arm == "M1" and r.qid == "a3"
        else r
        for r in rows
    ]
    report = inj.format_report(inj.evaluate(rows, model="m"))
    assert "max prompt tokens" in report and "4259" in report
