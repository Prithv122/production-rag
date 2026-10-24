"""Tracing: what the spans look like, and that tracing never changes a request.

Every test here runs in the fast suite with no torch and no network. Spans go to
OpenTelemetry's in-memory exporter; the one test that points at an OTLP endpoint
points it at a port nothing listens on, on purpose.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import pytest

from conftest import FakeProvider, make_chunk
from production_rag import tracing
from production_rag.bm25 import build_from_chunks as build_bm25
from production_rag.dense import build_from_chunks as build_dense
from production_rag.generate import generate
from production_rag.pipeline import OFFLINE_ARMS, Retriever
from production_rag.pricing import cost_usd, price_for
from production_rag.providers import (
    CachedProvider,
    FallbackProvider,
    LLMResponse,
    ProviderError,
)

pytest.importorskip("opentelemetry.sdk.trace")
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)
from opentelemetry.trace import StatusCode

ANSWER = json.dumps({"sufficient": True, "answer": "Use incremental models [1]."})
QUESTION = "how do I declare incremental dependencies"
REWRITES = json.dumps({"queries": ["assets in dagster", "dag dependencies"]})


@pytest.fixture(autouse=True)
def clean_tracing(monkeypatch):
    """Each test starts with tracing unresolved and the environment empty."""
    for name in (
        tracing.ENV_ENABLE,
        tracing.ENV_CONTENT,
        tracing.ENV_EXPORT_TIMEOUT,
        tracing.ENV_FLUSH_MS,
        "OTEL_BSP_SCHEDULE_DELAY",
        *tracing.ENV_ENDPOINTS,
    ):
        monkeypatch.delenv(name, raising=False)
    tracing.reset()
    yield
    tracing.reset()


@pytest.fixture
def spans():
    """Turn tracing on with an in-memory exporter; yields a callable returning finished spans."""
    exporter = InMemorySpanExporter()
    assert tracing.setup_tracing(exporter, simple=True)
    return exporter.get_finished_spans


@pytest.fixture
def retriever(embedder, sample_chunks):
    return Retriever(
        {c.chunk_id: c for c in sample_chunks},
        build_bm25(sample_chunks),
        build_dense(sample_chunks, embedder),
        embedder=embedder,
        strategy="heading",
    )


def by_name(finished):
    out: dict[str, list] = {}
    for span in finished():
        out.setdefault(span.name, []).append(span)
    return out


def full_pipeline(retriever, reranker, *, provider=None, answerer=None, arm="rerank_rewrite"):
    """Rewrite -> retrieve -> rerank -> generate inside one trace; returns both results."""
    provider = provider or FakeProvider([REWRITES])
    answerer = answerer or FakeProvider([ANSWER])
    with tracing.query(QUESTION, arm=arm, k=3, strategy="heading"):
        result = retriever.retrieve(
            QUESTION, arm=arm, k=3, pool=4, reranker=reranker, rewriter=provider
        )
        answer = generate(QUESTION, result.chunk_ids, retriever.chunks, answerer)
    return result, answer


def comparable(result, answer):
    """Everything about a result except wall-clock time."""
    return (result.ranked, result.queries, result.pool_size, result.arm, answer.as_dict())


# ---------------------------------------------------------------------------
# off by default
# ---------------------------------------------------------------------------
def test_tracing_is_off_by_default_and_every_helper_is_the_shared_noop():
    assert tracing.enabled() is False
    null = tracing.stage("x", a=1)
    assert null is tracing.query("q", arm="a") is tracing.llm_span("g", provider=FakeProvider())
    with null as handle:
        handle.set(a=1)
        handle.record(object())
    assert tracing._HANDLE is None and tracing._TRACER is None


def test_the_pipeline_path_never_imports_opentelemetry_when_disabled():
    """Run a retrieval in a clean interpreter with no tracing variables set."""
    script = (
        "import sys\n"
        "from conftest import FakeEmbedder, FakeProvider, make_chunk\n"
        "from production_rag.bm25 import build_from_chunks as b\n"
        "from production_rag.dense import build_from_chunks as d\n"
        "from production_rag.generate import generate\n"
        "from production_rag.pipeline import Retriever\n"
        "chunks = [make_chunk('dbt/a#0', 'dbt incremental models'),\n"
        "          make_chunk('dbt/b#0', 'tests')]\n"
        "emb = FakeEmbedder()\n"
        "r = Retriever({c.chunk_id: c for c in chunks}, b(chunks), d(chunks, emb), embedder=emb)\n"
        "res = r.retrieve('incremental', arm='hybrid_score_weighted', k=2)\n"
        "generate('q', res.chunk_ids, r.chunks, FakeProvider(['{\"answer\": \"a [1]\"}']))\n"
        "loaded = sorted(m for m in sys.modules if m.split('.')[0] == 'opentelemetry')\n"
        "assert not loaded, loaded\n"
        "print('clean')\n"
    )
    root = Path(__file__).resolve().parent.parent
    env = {
        k: v
        for k, v in os.environ.items()
        if not k.startswith("OTEL_") and not k.startswith("PRODUCTION_RAG_TRACE")
    }
    env["PYTHONPATH"] = os.pathsep.join([str(root / "src"), str(root / "tests")])
    done = subprocess.run(
        [sys.executable, "-c", script], env=env, capture_output=True, text=True, check=False
    )
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "clean"


def test_enabling_never_installs_a_global_tracer_provider(spans):
    from opentelemetry import trace
    from opentelemetry.sdk.trace import TracerProvider

    with tracing.stage("probe"):
        pass
    assert not isinstance(trace.get_tracer_provider(), TracerProvider)


def test_the_environment_switch_turns_tracing_on_and_off(monkeypatch):
    exporter = InMemorySpanExporter()
    monkeypatch.setattr(tracing, "otlp_exporter", lambda endpoint=None: exporter)
    monkeypatch.setenv(tracing.ENV_ENABLE, "1")
    with tracing.stage("probe"):
        pass
    assert tracing.enabled()
    tracing.reset()

    # An endpoint alone enables it; an explicit 0 overrides the endpoint.
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector.invalid:4318")
    monkeypatch.setenv(tracing.ENV_ENABLE, "0")
    assert tracing.enabled() is False
    tracing.reset()
    monkeypatch.delenv(tracing.ENV_ENABLE)
    assert tracing.enabled() is True


def test_enabling_without_the_extra_warns_once_and_stays_off(monkeypatch, caplog):
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.resources", None)
    monkeypatch.setenv(tracing.ENV_ENABLE, "1")
    with caplog.at_level(logging.WARNING, logger="production_rag.tracing"):
        assert tracing.enabled() is False
        tracing.setup_tracing()
        tracing.setup_tracing()
    warnings = [r for r in caplog.records if "not installed" in r.getMessage()]
    assert len(warnings) == 1


# ---------------------------------------------------------------------------
# span tree
# ---------------------------------------------------------------------------
def test_a_full_query_is_one_trace_with_a_span_per_stage(spans, retriever, reranker):
    full_pipeline(retriever, reranker)
    named = by_name(spans)

    assert len(named["rag.query"]) == 1, "the inner retrieve joins the outer root"
    assert len(named["rewrite"]) == 1
    assert len(named["retrieve.bm25"]) == 3  # original + two variants
    assert len(named["retrieve.dense"]) == 3
    assert len(named["fuse"]) == 1
    assert len(named["rerank"]) == 1
    assert len(named["generate"]) == 1

    root = named["rag.query"][0]
    everything = [s for group in named.values() for s in group]
    assert {s.context.trace_id for s in everything} == {root.context.trace_id}
    for span in everything:
        if span is not root:
            assert span.parent is not None and span.parent.span_id == root.context.span_id

    attrs = root.attributes
    assert attrs["production_rag.arm"] == "rerank_rewrite"
    assert attrs["production_rag.k"] == 3
    assert attrs["production_rag.strategy"] == "heading"
    assert attrs["production_rag.n_queries"] == 3


def test_retrieval_alone_opens_its_own_root(spans, retriever):
    retriever.retrieve("incremental models", arm="hybrid", k=2, pool=4)
    named = by_name(spans)
    assert set(named) == {"rag.query", "retrieve.bm25", "retrieve.dense", "fuse"}


def test_a_single_list_arm_has_no_fuse_span(spans, retriever):
    retriever.retrieve("incremental models", arm="bm25", k=2, pool=4)
    assert "fuse" not in by_name(spans)


def test_the_dense_span_covers_the_query_embedding(spans, retriever, embedder, monkeypatch):
    from opentelemetry import trace

    seen: list[str] = []
    real = embedder.encode_query

    def encode_query(text):
        seen.append(trace.get_current_span().name)
        return real(text)

    monkeypatch.setattr(embedder, "encode_query", encode_query)
    retriever.retrieve("incremental models", arm="dense", k=2, pool=4)
    assert seen == ["retrieve.dense"]


# ---------------------------------------------------------------------------
# LLM spans: GenAI + Langfuse attributes and cost
# ---------------------------------------------------------------------------
def response(model: str, *, provider: str = "fake", **kwargs) -> LLMResponse:
    return LLMResponse(
        text=ANSWER,
        model=model,
        provider=provider,
        prompt_tokens=kwargs.pop("prompt_tokens", 1000),
        completion_tokens=kwargs.pop("completion_tokens", 200),
        **kwargs,
    )


def generate_span(spans, retriever, *, model: str, **kwargs):
    answerer = FakeProvider([response(model, **kwargs)], model=model)
    result = retriever.retrieve("incremental models", arm="bm25", k=2, pool=4)
    generate("incremental models", result.chunk_ids, retriever.chunks, answerer)
    return by_name(spans)["generate"][0]


def test_an_llm_span_carries_genai_and_langfuse_attributes(spans, retriever):
    span = generate_span(spans, retriever, model="google/gemini-2.5-flash-lite")
    a = span.attributes
    assert a["gen_ai.system"] == "fake"
    assert a["gen_ai.request.model"] == "google/gemini-2.5-flash-lite"
    assert a["gen_ai.response.model"] == "google/gemini-2.5-flash-lite"
    assert a["gen_ai.usage.input_tokens"] == 1000
    assert a["gen_ai.usage.output_tokens"] == 200
    assert a["langfuse.observation.type"] == "generation"
    assert a["langfuse.observation.model.name"] == "google/gemini-2.5-flash-lite"
    assert json.loads(a["langfuse.observation.usage_details"]) == {
        "input": 1000,
        "output": 200,
        "total": 1200,
    }
    cost = json.loads(a["langfuse.observation.cost_details"])
    assert cost["input"] == pytest.approx(1000 * 0.10 / 1e6)
    assert cost["output"] == pytest.approx(200 * 0.40 / 1e6)
    assert cost["total"] == pytest.approx(cost["input"] + cost["output"])
    assert a["production_rag.cost_known"] is True
    assert a["production_rag.cost_usd"] == pytest.approx(cost["total"])


def test_a_free_model_costs_zero_and_says_it_is_known(spans, retriever):
    span = generate_span(spans, retriever, model="vendor/some-model:free")
    assert span.attributes["production_rag.cost_known"] is True
    assert span.attributes["production_rag.cost_usd"] == 0.0


def test_an_unknown_model_has_its_cost_omitted_not_zeroed(spans, retriever):
    span = generate_span(spans, retriever, model="mystery-model")
    assert span.attributes["production_rag.cost_known"] is False
    assert "langfuse.observation.cost_details" not in span.attributes
    assert "production_rag.cost_usd" not in span.attributes
    # Usage is still reported; only the price is unknown.
    assert span.attributes["gen_ai.usage.input_tokens"] == 1000


def test_pricing_table():
    assert price_for("llama3.2:3b") == (0.0, 0.0)
    assert price_for("anything", "ollama") == (0.0, 0.0)
    assert price_for("a/b:free") == (0.0, 0.0)
    assert price_for("mystery-model") is None
    assert cost_usd("mystery-model", 10, 10) is None
    cost = cost_usd("google/gemini-2.5-flash-lite", 1_000_000, 1_000_000)
    assert cost is not None and cost.total == pytest.approx(0.50)


def test_the_cached_flag_and_the_recorded_latency_are_exported(spans, retriever, tmp_path):
    from production_rag.cache import JsonCache

    inner = FakeProvider([response("m:free", latency_s=12.5)], model="m:free")
    provider = CachedProvider(inner, JsonCache(tmp_path))
    ids = retriever.retrieve("incremental models", arm="bm25", k=2, pool=4).chunk_ids
    generate("q", ids, retriever.chunks, provider)
    generate("q", ids, retriever.chunks, provider)

    first, second = by_name(spans)["generate"]
    assert first.attributes["production_rag.cached"] is False
    assert second.attributes["production_rag.cached"] is True
    assert second.attributes["production_rag.recorded_latency_s"] == 12.5
    assert len(inner.prompts) == 1


def test_a_cached_response_costs_nothing_now_and_keeps_the_original_cost(
    spans, retriever, tmp_path
):
    from production_rag.cache import JsonCache

    model = "google/gemini-2.5-flash-lite"
    inner = FakeProvider([response(model)], model=model)
    provider = CachedProvider(inner, JsonCache(tmp_path))
    ids = retriever.retrieve("incremental models", arm="bm25", k=2, pool=4).chunk_ids
    generate("q", ids, retriever.chunks, provider)
    generate("q", ids, retriever.chunks, provider)

    live, cached = (s.attributes for s in by_name(spans)["generate"])
    original = 1000 * 0.10 / 1e6 + 200 * 0.40 / 1e6
    assert live["production_rag.cost_usd"] == pytest.approx(original)
    assert "production_rag.recorded_cost_usd" not in live

    assert cached["production_rag.cached"] is True
    assert cached["production_rag.cost_usd"] == 0.0
    assert json.loads(cached["langfuse.observation.cost_details"]) == {
        "input": 0.0,
        "output": 0.0,
        "total": 0.0,
    }
    assert cached["production_rag.recorded_cost_usd"] == pytest.approx(original)
    # Usage is reported as recorded, so token charts stay meaningful on a cached answer.
    assert cached["gen_ai.usage.input_tokens"] == 1000


def test_both_names_of_the_genai_provider_attribute_are_emitted(spans, retriever):
    span = generate_span(spans, retriever, model="vendor/some-model:free")
    assert span.attributes["gen_ai.system"] == "fake"
    assert span.attributes["gen_ai.provider.name"] == "fake"


# ---------------------------------------------------------------------------
# content capture
# ---------------------------------------------------------------------------
def dump(finished) -> str:
    """Every attribute and event of every span, as one string."""
    parts = []
    for span in finished():
        parts.append(json.dumps(dict(span.attributes), default=str))
        parts.extend(json.dumps(dict(e.attributes), default=str) for e in span.events)
    return "\n".join(parts)


def test_content_capture_is_off_by_default(spans, retriever, reranker):
    full_pipeline(retriever, reranker)
    text = dump(spans)
    assert QUESTION not in text
    assert "Use incremental models" not in text  # the model's reply
    assert "dbt incremental models append" not in text  # a retrieved passage in the prompt
    root = by_name(spans)["rag.query"][0]
    assert "production_rag.question" not in root.attributes
    assert "input.value" not in root.attributes
    assert "langfuse.observation.input" not in by_name(spans)["generate"][0].attributes


def test_content_capture_is_opt_in(spans, retriever, reranker, monkeypatch):
    monkeypatch.setenv(tracing.ENV_CONTENT, "1")
    full_pipeline(retriever, reranker)
    named = by_name(spans)
    assert named["rag.query"][0].attributes["production_rag.question"] == QUESTION
    generated = named["generate"][0].attributes
    assert "dbt incremental models append" in json.loads(generated["langfuse.observation.input"])
    assert "Use incremental models" in json.loads(generated["langfuse.observation.output"])


# ---------------------------------------------------------------------------
# tracing does not change a result
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("arm", OFFLINE_ARMS)
def test_retrieval_is_identical_with_tracing_on_and_off(arm, retriever, reranker):
    off = retriever.retrieve("incremental models", arm=arm, k=3, pool=4, reranker=reranker)
    exporter = InMemorySpanExporter()
    tracing.setup_tracing(exporter, simple=True)
    on = retriever.retrieve("incremental models", arm=arm, k=3, pool=4, reranker=reranker)
    assert (on.ranked, on.queries, on.pool_size) == (off.ranked, off.queries, off.pool_size)
    assert exporter.get_finished_spans()


def test_the_full_pipeline_is_identical_with_tracing_on_and_off(retriever, reranker):
    off = comparable(*full_pipeline(retriever, reranker))
    exporter = InMemorySpanExporter()
    tracing.setup_tracing(exporter, simple=True)
    on = comparable(*full_pipeline(retriever, reranker))
    assert on == off


# ---------------------------------------------------------------------------
# errors
# ---------------------------------------------------------------------------
def test_a_failing_stage_gets_error_status_and_the_exception_is_reraised(spans, retriever):
    with pytest.raises(ValueError, match="unknown arm"):
        retriever.retrieve("secret question text", arm="magic")
    root = by_name(spans)["rag.query"][0]
    assert root.status.status_code == StatusCode.ERROR
    event = root.events[0]
    assert event.name == "exception"
    assert event.attributes["exception.type"] == "ValueError"
    # Not an operational error type, so its message is withheld without content capture.
    assert "exception.message" not in event.attributes
    assert "exception.stacktrace" not in event.attributes
    assert root.attributes["error.type"] == "ValueError"


def test_an_exception_message_is_recorded_with_content_capture(spans, retriever, monkeypatch):
    monkeypatch.setenv(tracing.ENV_CONTENT, "1")
    with pytest.raises(ValueError):
        retriever.retrieve("q", arm="magic")
    root = by_name(spans)["rag.query"][0]
    assert "unknown arm" in root.events[0].attributes["exception.message"]


def test_a_provider_error_marks_the_generate_span(spans, retriever):
    ids = retriever.retrieve("incremental models", arm="bm25", k=2, pool=4).chunk_ids
    answer = generate("q", ids, retriever.chunks, FakeProvider(fail=True))
    assert answer.refusal_reason == "provider_error"  # behaviour unchanged

    span = by_name(spans)["generate"][0]
    assert span.status.status_code == StatusCode.ERROR
    event = span.events[0]
    assert event.attributes["exception.type"] == "ProviderError"
    # "fake is down" is not a recognised prefix, so no message leaves the process.
    assert "exception.message" not in event.attributes


class Raises:
    """A provider whose call fails with a chosen exception."""

    name = "raiser"
    model = "raiser-1"

    def __init__(self, exc: Exception) -> None:
        self.exc = exc

    def complete(self, prompt, **kwargs):
        raise self.exc


LEAK = "SECRET-QUESTION-TEXT"


@pytest.mark.parametrize(
    ("message", "exported"),
    [
        (f"HTTP 403 from https://api.example/v1/chat?q={LEAK}: " + LEAK * 3, "HTTP 403"),
        (f"could not reach https://host/path?key={LEAK}: refused", "could not reach"),
        (f"openrouter error: {LEAK}", "openrouter error"),
        (f"unexpected response shape: {LEAK}", "unexpected response shape"),
        (f"unexpected ollama response: {LEAK}", "unexpected ollama response"),
        (f"all providers failed:\n  a(m): HTTP 500 from u: {LEAK}", "all providers failed"),
        (f"replay mode: this prompt is not in the cache {LEAK}", "replay mode"),
        (f"some new error shape that quotes {LEAK}", None),
    ],
)
def test_provider_error_bodies_never_leave_the_process(spans, retriever, message, exported):
    ids = retriever.retrieve("incremental models", arm="bm25", k=2, pool=4).chunk_ids
    generate("q", ids, retriever.chunks, Raises(ProviderError(message)))

    span = by_name(spans)["generate"][0]
    assert LEAK not in dump(spans)
    assert LEAK not in (span.status.description or "")
    attrs = span.events[0].attributes
    assert attrs["exception.type"] == "ProviderError"
    assert attrs.get("exception.message") == exported


def test_other_exception_types_export_no_message_even_if_they_look_operational(spans, retriever):
    ids = retriever.retrieve("incremental models", arm="bm25", k=2, pool=4).chunk_ids
    with pytest.raises(OSError):
        generate("q", ids, retriever.chunks, Raises(OSError(f"HTTP 403 {LEAK}")))
    assert LEAK not in dump(spans)
    assert "exception.message" not in by_name(spans)["generate"][0].events[0].attributes


def test_the_full_error_text_is_exported_only_with_content_capture(spans, retriever, monkeypatch):
    monkeypatch.setenv(tracing.ENV_CONTENT, "1")
    ids = retriever.retrieve("incremental models", arm="bm25", k=2, pool=4).chunk_ids
    generate("q", ids, retriever.chunks, Raises(ProviderError(f"HTTP 403 from u: {LEAK}")))
    message = by_name(spans)["generate"][0].events[0].attributes["exception.message"]
    assert LEAK in message


@pytest.mark.parametrize(
    ("reply", "kind"),
    [
        ("not json at all", "unparseable"),
        ("", "empty_reply"),
        ("{}", "empty_payload"),
        (LLMResponse(text="{", model="m", provider="p", finish_reason="length"), "truncated"),
    ],
)
def test_unusable_replies_are_classified_on_the_llm_span(spans, retriever, reply, kind):
    ids = retriever.retrieve("incremental models", arm="bm25", k=2, pool=4).chunk_ids
    generate("q", ids, retriever.chunks, FakeProvider([reply, reply]))
    first = by_name(spans)["generate"][0]
    assert first.attributes["production_rag.failure_kind"] == kind
    assert first.attributes["langfuse.observation.level"] == "WARNING"


def test_a_fallback_answer_is_flagged_on_the_span(spans, retriever):
    primary = FakeProvider(fail=True, name="primary", model="big")
    backup = FakeProvider([ANSWER], name="backup", model="small")
    ids = retriever.retrieve("incremental models", arm="bm25", k=2, pool=4).chunk_ids
    answer = generate("q", ids, retriever.chunks, FallbackProvider([primary, backup]))
    assert not answer.refused

    span = by_name(spans)["generate"][0]
    assert span.attributes["production_rag.fallback_used"] is True
    assert span.attributes["production_rag.failure_kind"] == "fallback"
    assert span.attributes["gen_ai.system"] == "backup"
    assert span.attributes["production_rag.model_substituted"] is True


def test_an_unparseable_rewrite_is_marked_and_still_degrades_to_the_original(
    spans, retriever, reranker
):
    result = retriever.retrieve(
        "incremental models",
        arm="rerank_rewrite",
        k=3,
        pool=4,
        reranker=reranker,
        rewriter=FakeProvider(["not json"]),
    )
    assert result.queries == ["incremental models"]
    rewrite = by_name(spans)["rewrite"][0]
    assert rewrite.attributes["production_rag.failure_kind"] == "unparseable"


# ---------------------------------------------------------------------------
# failure isolation: tracing must never break or alter a request
# ---------------------------------------------------------------------------
def without_clock(answer) -> dict:
    """An answer's fields except the wall-clock time a failed call records."""
    return {k: v for k, v in answer.as_dict().items() if k != "latency_s"}


def baseline(retriever, reranker):
    return comparable(*full_pipeline(retriever, reranker))


ISOLATION_METHODS = (
    "start_span",
    "set_attributes",
    "set_attribute",
    "add_event",
    "set_status",
    "end",
)


def break_tracing(monkeypatch, method: str) -> None:
    """Make one OpenTelemetry call raise, wherever the pipeline makes it."""
    from opentelemetry.sdk.trace import Span

    def boom(*args, **kwargs):
        raise RuntimeError(f"{method} is broken")

    target = tracing._TRACER if method == "start_span" else Span
    monkeypatch.setattr(target, method, boom)


@pytest.mark.parametrize("method", ISOLATION_METHODS)
def test_a_failing_tracing_call_leaves_a_successful_request_unchanged(
    method, retriever, reranker, monkeypatch, caplog
):
    expected = baseline(retriever, reranker)
    tracing.setup_tracing(InMemorySpanExporter(), simple=True)
    break_tracing(monkeypatch, method)
    with caplog.at_level(logging.WARNING, logger="production_rag.tracing"):
        assert comparable(*full_pipeline(retriever, reranker)) == expected
        assert comparable(*full_pipeline(retriever, reranker)) == expected
    # Logged once per call site, not once per request.
    messages = [r.getMessage() for r in caplog.records if "tracing failed" in r.getMessage()]
    assert len(messages) == len(set(messages))


@pytest.mark.parametrize("method", ISOLATION_METHODS)
def test_a_failing_tracing_call_leaves_a_failing_request_failing_the_same_way(
    method, retriever, monkeypatch
):
    """set_status, add_event and the error attribute only run when a stage raised."""
    ids = retriever.retrieve("incremental models", arm="bm25", k=2, pool=4).chunk_ids
    off_answer = generate("q", ids, retriever.chunks, FakeProvider(fail=True))
    with pytest.raises(ValueError, match="unknown arm"):
        retriever.retrieve("q", arm="magic")

    tracing.setup_tracing(InMemorySpanExporter(), simple=True)
    break_tracing(monkeypatch, method)
    on_answer = generate("q", ids, retriever.chunks, FakeProvider(fail=True))
    assert without_clock(on_answer) == without_clock(off_answer)
    with pytest.raises(ValueError, match="unknown arm"):
        retriever.retrieve("q", arm="magic")


def test_a_reply_object_that_fails_to_inspect_is_swallowed_by_the_classifier(spans):
    class Hostile:
        text = "{}"

        @property
        def finish_reason(self):
            raise RuntimeError("cannot read finish_reason")

    with tracing.llm_span("generate", provider=FakeProvider()) as span:
        span.check_answer(Hostile())


def test_a_failing_span_does_not_swallow_the_pipelines_own_exception(retriever, monkeypatch):
    from opentelemetry.sdk.trace import Span

    tracing.setup_tracing(InMemorySpanExporter(), simple=True)
    monkeypatch.setattr(Span, "end", lambda *a, **k: 1 / 0)
    with pytest.raises(ValueError, match="unknown arm"):
        retriever.retrieve("q", arm="magic")


def test_setup_with_a_failing_exporter_build_stays_off_and_does_not_raise(monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("cannot build exporter")

    monkeypatch.setattr(tracing, "otlp_exporter", boom)
    monkeypatch.setenv(tracing.ENV_ENABLE, "1")
    assert tracing.enabled() is False


def test_an_unreachable_collector_neither_breaks_nor_slows_requests(
    retriever, reranker, monkeypatch
):
    """Port 9 (discard) has no listener. Export fails on the worker thread, not here."""
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    monkeypatch.setenv(tracing.ENV_EXPORT_TIMEOUT, "1")
    expected = baseline(retriever, reranker)

    assert tracing.setup_tracing(tracing.otlp_exporter("http://127.0.0.1:9"))
    started = time.perf_counter()
    for _ in range(20):
        assert comparable(*full_pipeline(retriever, reranker)) == expected
    served = time.perf_counter() - started
    # 20 requests with a dead collector: if export were on the request path this would
    # take at least the export timeout per request.
    assert served < 3.0, served

    started = time.perf_counter()
    tracing.shutdown_tracing()  # flushes; bounded by the export timeout, never raises
    assert time.perf_counter() - started < 30.0


def test_the_export_timeout_is_bounded_by_default():
    assert tracing.export_timeout_s() == tracing.DEFAULT_EXPORT_TIMEOUT_S == 5.0


def test_the_tracing_conftest_helpers_are_importable():
    # Guards the subprocess test above, which imports conftest by name.
    assert make_chunk("a#0", "x").chunk_id == "a#0"


def test_a_provider_that_raises_is_not_hidden_by_the_span(spans, retriever):
    class Boom:
        name = "boom"
        model = "boom-1"

        def complete(self, prompt, **kwargs):
            raise KeyError("a bug, not a provider error")

    ids = retriever.retrieve("incremental models", arm="bm25", k=2, pool=4).chunk_ids
    with pytest.raises(KeyError):
        generate("q", ids, retriever.chunks, Boom())
    span = by_name(spans)["generate"][0]
    assert span.status.status_code == StatusCode.ERROR
    assert ProviderError.__name__ not in span.events[0].attributes["exception.type"]


class _SlowExporter(InMemorySpanExporter):
    """Takes `delay` seconds per export, like a far-away collector."""

    def __init__(self, delay: float) -> None:
        super().__init__()
        self.delay = delay

    def export(self, spans):
        import time

        time.sleep(self.delay)
        return super().export(spans)


def test_the_request_flush_is_a_noop_when_tracing_is_off():
    assert tracing.flush_after_request() is None


def test_the_request_flush_exports_a_batched_trace_before_returning(monkeypatch):
    exporter = InMemorySpanExporter()
    assert tracing.setup_tracing(exporter, simple=False)  # batched, as on the server
    with tracing.query(QUESTION, arm="hybrid"), tracing.stage("retrieve.bm25"):
        pass
    assert exporter.get_finished_spans() == ()  # still queued on the worker thread
    assert tracing.flush_after_request() == "flushed"
    assert {s.name for s in exporter.get_finished_spans()} == {"rag.query", "retrieve.bm25"}


def test_the_request_flush_is_bounded_by_its_budget(monkeypatch):
    import time

    monkeypatch.setenv(tracing.ENV_FLUSH_MS, "100")
    exporter = _SlowExporter(delay=1.5)
    assert tracing.setup_tracing(exporter, simple=False)
    with tracing.query(QUESTION):
        pass
    started = time.perf_counter()
    done = tracing.flush_after_request()
    waited = time.perf_counter() - started
    assert done == "timeout" and waited < 1.0  # returned at the budget, not after the export
    assert tracing._FLUSH_THREAD is not None and tracing._FLUSH_THREAD.is_alive()  # it did start


def test_the_budget_is_clamped(monkeypatch):
    monkeypatch.setenv(tracing.ENV_FLUSH_MS, "60000")
    assert tracing.flush_budget_ms() == tracing.MAX_FLUSH_MS
    monkeypatch.setenv(tracing.ENV_FLUSH_MS, "-5")
    assert tracing.flush_budget_ms() == 0


def test_a_zero_budget_or_a_failing_flush_never_reaches_the_request(monkeypatch, caplog):
    assert tracing.setup_tracing(InMemorySpanExporter(), simple=False)
    monkeypatch.setenv(tracing.ENV_FLUSH_MS, "0")
    assert tracing.flush_after_request() is None
    monkeypatch.setenv(tracing.ENV_FLUSH_MS, "not-a-number")
    assert tracing.flush_budget_ms() == tracing.DEFAULT_FLUSH_MS

    def boom(_timeout_ms):
        raise RuntimeError("collector exploded")

    monkeypatch.setattr(tracing._HANDLE, "flush", boom)
    with caplog.at_level("WARNING", logger="production_rag.tracing"):
        assert tracing.flush_after_request() == "error"
    assert "flush_after_request" in caplog.text  # swallowed inside the thread, and logged


def test_a_flush_still_running_is_not_doubled_up(monkeypatch):
    monkeypatch.setenv(tracing.ENV_FLUSH_MS, "50")
    assert tracing.setup_tracing(_SlowExporter(delay=0.6), simple=False)
    with tracing.query(QUESTION):
        pass
    assert tracing.flush_after_request() == "timeout"  # budget passed, export still running
    first = tracing._FLUSH_THREAD
    assert first is not None and first.is_alive()
    with tracing.query(QUESTION):
        pass
    assert tracing.flush_after_request() == "skipped"
    assert tracing._FLUSH_THREAD is first  # no second thread was started
    first.join(5)


def test_concurrent_requests_start_one_flush_thread(monkeypatch):
    import threading

    monkeypatch.setenv(tracing.ENV_FLUSH_MS, "50")
    assert tracing.setup_tracing(_SlowExporter(delay=0.6), simple=False)
    with tracing.query(QUESTION):
        pass
    gate = threading.Barrier(4)
    results: list[str | None] = []

    def request():
        gate.wait()
        results.append(tracing.flush_after_request())

    callers = [threading.Thread(target=request) for _ in range(4)]
    for t in callers:
        t.start()
    for t in callers:
        t.join(5)
    assert sorted(results) == ["skipped", "skipped", "skipped", "timeout"]
    tracing._FLUSH_THREAD.join(5)
