"""Per-query traces of the pipeline, over vendor-neutral OpenTelemetry.

The question this answers is the one a dashboard of averages cannot: for *this*
query, where did the time go, and what did the model call cost? A query is one
trace, ``rag.query``, with a child span per stage::

    rag.query
    |-- rewrite            (LLM; only on the rewrite arms)
    |-- retrieve.bm25      (one per issued query)
    |-- retrieve.dense     (includes the query embedding)
    |-- fuse               (only when more than one ranking is fused)
    |-- rerank
    `-- generate           (LLM)

**Why OpenTelemetry and not a vendor SDK.** Langfuse and Arize Phoenix both ingest
OTLP/HTTP, so one exporter feeds either, and the application never imports a
vendor library. Where the live service sends its traces is a hosting decision
that does not belong in the code; it is three environment variables.

**Off means off.** Tracing is enabled only when an OTLP endpoint is configured
(``OTEL_EXPORTER_OTLP_ENDPOINT``) or ``PRODUCTION_RAG_TRACING=1`` is set, *and* the
``tracing`` extra is installed. Otherwise :func:`stage`, :func:`query` and
:func:`llm_span` return one shared do-nothing object, ``opentelemetry`` is never
imported, and no TracerProvider exists. (The extra being absent while tracing is
requested logs one warning and stays off; a typo in an environment variable must
not take the demo down.) This module never installs a *global* provider either:
it keeps its own, so it cannot hijack another library's telemetry.

**Tracing must never change a request.** Every call into OpenTelemetry in this
module is wrapped: an exception inside tracing is swallowed and logged once per
call site, and the pipeline's own exceptions pass through a span untouched. Export
runs on a background thread with a bounded timeout, so an unreachable collector
costs dropped spans and a log line, not request latency.

**Errors.** A stage that raises gets status ERROR and an ``exception`` event
carrying the exception type. With content capture off, a message is exported only
for provider and cache errors, and then only a fixed safe prefix (``HTTP 403``,
``could not reach``, ``replay mode``): provider errors are built from response bodies
and URLs, which can echo the prompt, so the body is never exported. No stack trace
is recorded: locals and frames can hold the question. The exception is then
re-raised unchanged. Failures the code already classifies (a truncated or
unparseable reply, a fallback model answering) are attributes on the LLM span,
``production_rag.failure_kind``.

**Privacy default.** The public demo takes arbitrary user input, so question text
and prompt/response bodies are *not* exported unless
``PRODUCTION_RAG_TRACE_CONTENT=1``. Timings, token counts, models, arm and k are
always exported.

**The LLM spans** carry the OpenTelemetry GenAI semantic-convention attributes
(``gen_ai.system`` and its newer name ``gen_ai.provider.name``,
``gen_ai.request.model``, ``gen_ai.response.model``,
``gen_ai.usage.input_tokens``, ``gen_ai.usage.output_tokens``) and the attributes
Langfuse maps to a *generation* with usage and cost
(``langfuse.observation.type``, ``.usage_details``, ``.cost_details``).
The cost is tokens x the price in :mod:`production_rag.pricing`; an unpriced model
has its cost omitted and ``production_rag.cost_known`` set to false.

A cached response keeps the tokens and latency of the call that was recorded, but
costs nothing now: ``production_rag.cached`` is true, the exported cost is 0, and
``production_rag.recorded_cost_usd`` holds what the original call cost (when its price
is known). ``production_rag.recorded_latency_s`` is the original latency, which is
*not* the span's duration (a cache read). Summing ``cost_usd`` over a window therefore
gives spend, and summing ``recorded_cost_usd`` gives what the same answers would cost
uncached.
"""

from __future__ import annotations

import atexit
import contextvars
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("production_rag.tracing")

SERVICE_NAME = "production-rag"
ENV_ENABLE = "PRODUCTION_RAG_TRACING"
ENV_CONTENT = "PRODUCTION_RAG_TRACE_CONTENT"
ENV_EXPORT_TIMEOUT = "PRODUCTION_RAG_TRACE_EXPORT_TIMEOUT_S"
ENV_FLUSH_MS = "PRODUCTION_RAG_TRACE_FLUSH_MS"
ENV_ENDPOINTS = ("OTEL_EXPORTER_OTLP_ENDPOINT", "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")

#: Upper bound on one export call, retries included, in seconds, applied as the OTLP exporter's own
#: timeout. That is the real bound: the SDK's batch processor accepts an export timeout
#: but does not read it (its source says "Not used"), so it is not passed. The exporter's
#: default is 10 s; a dead collector would hold the worker thread (and the flush at
#: exit) that long per export. Spans are best-effort, so the bound is short. The
#: standard OTEL_EXPORTER_OTLP_TIMEOUT variables still win when the operator sets them.
DEFAULT_EXPORT_TIMEOUT_S = 5.0

#: Most a served request waits, in milliseconds, for its spans to leave before it returns.
#: On Cloud Run with request-based billing the CPU is throttled once a response is sent,
#: so a batch left on the worker thread can sit until the next request, or be lost when
#: the instance stops. The flush is bounded and best-effort: past the budget the request
#: returns anyway and the spans stay queued. ``0`` turns it off.
DEFAULT_FLUSH_MS = 1000
#: Ceiling on the budget, so a mistyped value cannot hold every request for a minute.
MAX_FLUSH_MS = 2000

_PREFIX = "production_rag."
_FALSE = {"0", "false", "no", "off"}
_TRUE = {"1", "true", "yes", "on"}
#: Exceptions whose message may be reduced to a fixed prefix. Provider errors embed the
#: response body and URL (``HTTP 403 from <url>: <body>``), and a body can echo the
#: prompt, so only these prefixes ever leave the process with content capture off.
_SAFE_MESSAGE_TYPES = {"ProviderError", "CacheMiss", "LookupError"}
_SAFE_PREFIXES = (
    "could not reach",
    "replay mode",
    "openrouter error",
    "unexpected response shape",
    "unexpected ollama response",
    "non-JSON response",
    "all providers failed",
    "OPENROUTER_API_KEY is not set",
    "no cached entry",
)
_HTTP_STATUS = re.compile(r"^HTTP (\d{3})\b")

# Module state. `_TRACER is None` is the disabled fast path.
_TRACER: Any = None
_API: Any = None
_HANDLE: TracerHandle | None = None
_RESOLVED = False
_LOCK = threading.Lock()
_WARNED: set[str] = set()
_ATEXIT = False
_ROOT: contextvars.ContextVar[Any] = contextvars.ContextVar("production_rag_root", default=None)


def _swallow(where: str) -> None:
    """Log a tracing failure once per call site and carry on. Never raises."""
    try:
        if where not in _WARNED:
            _WARNED.add(where)
            logger.warning("tracing failed in %s; ignoring (logged once)", where, exc_info=True)
    except Exception:  # pragma: no cover - logging itself must not break a request
        pass


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        logger.warning(message)


# ---------------------------------------------------------------------------
# the do-nothing path
# ---------------------------------------------------------------------------
class _Null:
    """Shared stand-in for a span when tracing is off. Allocates nothing."""

    __slots__ = ()

    def __enter__(self) -> _Null:
        return self

    def __exit__(self, *exc: object) -> bool:
        return False

    def set(self, **attrs: Any) -> None:
        return None

    def record(self, response: Any, **attrs: Any) -> None:
        return None

    def content(self, *, input: Any = None, output: Any = None) -> None:
        return None

    def failure(self, kind: str, **attrs: Any) -> None:
        return None

    def check_answer(self, response: Any) -> None:
        return None


_NULL = _Null()


class _PassThrough:
    """Re-entering the root span: reuse the active one instead of nesting."""

    __slots__ = ("_active",)

    def __init__(self, active: Any) -> None:
        self._active = active

    def __enter__(self) -> Any:
        return self._active

    def __exit__(self, *exc: object) -> bool:
        return False


# ---------------------------------------------------------------------------
# configuration
# ---------------------------------------------------------------------------
def content_enabled() -> bool:
    """Whether question text and prompt/response bodies may be exported."""
    return os.environ.get(ENV_CONTENT, "").strip().lower() in _TRUE


def _wants_tracing() -> bool:
    flag = os.environ.get(ENV_ENABLE, "").strip().lower()
    if flag in _FALSE:
        return False
    if flag in _TRUE:
        return True
    return any(os.environ.get(name) for name in ENV_ENDPOINTS)


def export_timeout_s() -> float:
    try:
        return float(os.environ.get(ENV_EXPORT_TIMEOUT, DEFAULT_EXPORT_TIMEOUT_S))
    except ValueError:
        return DEFAULT_EXPORT_TIMEOUT_S


@dataclass
class TracerHandle:
    """One TracerProvider plus its tracer. Built without touching global state."""

    provider: Any
    tracer: Any
    exporter: Any
    api: Any = field(default=None, repr=False)

    def flush(self, timeout_ms: int = 10_000) -> bool:
        return bool(self.provider.force_flush(timeout_ms))

    def shutdown(self) -> None:
        self.provider.shutdown()


def otlp_exporter(endpoint: str | None = None) -> Any:
    """An OTLP/HTTP span exporter. `endpoint` is a base URL, as in the standard env var.

    With no argument the exporter reads ``OTEL_EXPORTER_OTLP_*`` itself, headers
    included, which is how credentials reach it without ever being a parameter here.
    The per-attempt timeout is bounded (see ``DEFAULT_EXPORT_TIMEOUT_S``) unless the
    standard timeout variables are set.
    """
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    kwargs: dict[str, Any] = {}
    if not (
        os.environ.get("OTEL_EXPORTER_OTLP_TIMEOUT")
        or os.environ.get("OTEL_EXPORTER_OTLP_TRACES_TIMEOUT")
    ):
        kwargs["timeout"] = export_timeout_s()
    if endpoint:
        kwargs["endpoint"] = endpoint.rstrip("/") + "/v1/traces"
    return OTLPSpanExporter(**kwargs)


def build_tracer(
    exporter: Any = None, *, simple: bool = False, service_name: str = SERVICE_NAME
) -> TracerHandle | None:
    """Build a tracer, or return ``None`` (with one warning) if the extra is missing.

    `simple=True` exports each span as it ends, which is what a test wants;
    otherwise spans are batched on a worker thread, which is what a server wants.
    """
    try:
        from opentelemetry import context, trace
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SimpleSpanProcessor
        from opentelemetry.trace import Status, StatusCode
    except ImportError:
        _warn_once(
            "no-sdk",
            "tracing was requested but opentelemetry is not installed; staying off. "
            "Install it with `uv sync --extra tracing`.",
        )
        return None
    if exporter is None:
        try:
            exporter = otlp_exporter()
        except ImportError:
            _warn_once(
                "no-otlp",
                "tracing was requested but the OTLP exporter is not installed; staying off. "
                "Install it with `uv sync --extra tracing`.",
            )
            return None
    provider = TracerProvider(resource=Resource.create({"service.name": service_name}))
    provider.add_span_processor(
        SimpleSpanProcessor(exporter) if simple else BatchSpanProcessor(exporter)
    )
    api = (context, trace, Status, StatusCode)
    return TracerHandle(provider, provider.get_tracer("production_rag"), exporter, api)


def activate(handle: TracerHandle | None) -> None:
    """Make `handle` the tracer in use; ``None`` turns tracing off."""
    global _TRACER, _HANDLE, _RESOLVED, _API
    _HANDLE = handle
    _TRACER = handle.tracer if handle is not None else None
    _API = handle.api if handle is not None else None
    _RESOLVED = True


def setup_tracing(exporter: Any = None, *, simple: bool = False) -> bool:
    """Build and activate tracing. Returns whether it is now on. Never raises."""
    global _ATEXIT
    try:
        handle = build_tracer(exporter, simple=simple)
    except Exception:
        _swallow("setup_tracing")
        handle = None
    if handle is None:
        activate(None)
        return False
    activate(handle)
    if not _ATEXIT:
        atexit.register(shutdown_tracing)
        _ATEXIT = True
    return True


def flush_budget_ms() -> int:
    """The per-request flush budget, clamped to ``[0, MAX_FLUSH_MS]``."""
    try:
        return min(MAX_FLUSH_MS, max(0, int(os.environ.get(ENV_FLUSH_MS, DEFAULT_FLUSH_MS))))
    except ValueError:
        return DEFAULT_FLUSH_MS


_FLUSH_THREAD: threading.Thread | None = None
_FLUSH_LOCK = threading.Lock()


def flush_after_request() -> str | None:
    """Push the finished spans out before a served request returns. Never raises.

    ``None`` when tracing is off or the budget is 0 (nothing was attempted). Otherwise
    one of:

    - ``"flushed"``: the export ran to completion within the budget. That is not proof
      the collector *accepted* it: the SDK ignores the exporter's result, so a rejected
      batch (a wrong key, a 429) shows up only as the OTLP exporter's own ERROR log.
    - ``"timeout"``: the budget ran out. The flush carries on in the background and the
      spans stay queued.
    - ``"skipped"``: a flush from an earlier request is still running, so this one
      returns at once; its spans leave on that flush, the next one, or the worker.
    - ``"error"``: the flush raised (logged once).

    The budget is enforced here, not by the SDK: ``force_flush`` in opentelemetry-sdk
    1.45 ignores its timeout and exports synchronously, bounded only by the exporter's
    timeout per export call (retries included). So the flush runs on a helper thread
    and the request waits for it at most the budget.
    """
    global _FLUSH_THREAD
    handle = _HANDLE
    budget = flush_budget_ms()
    if handle is None or budget == 0:
        return None
    try:
        outcome: list[str] = []

        def run() -> None:
            try:
                handle.flush(budget)
                outcome.append("flushed")
            except Exception:
                _swallow("flush_after_request")
                outcome.append("error")

        with _FLUSH_LOCK:
            if _FLUSH_THREAD is not None and _FLUSH_THREAD.is_alive():
                return "skipped"
            worker = threading.Thread(target=run, name="production-rag-flush", daemon=True)
            _FLUSH_THREAD = worker
            worker.start()
        worker.join(budget / 1000)
        return outcome[0] if outcome and not worker.is_alive() else "timeout"
    except Exception:
        _swallow("flush_after_request")
        return "error"


def shutdown_tracing() -> None:
    """Flush and stop the active tracer, then leave tracing off. Never raises."""
    handle = _HANDLE
    activate(None)
    if handle is not None:
        try:
            handle.shutdown()
        except Exception:
            _swallow("shutdown_tracing")


def reset() -> None:
    """Shut down and forget the environment check, so the next span re-reads it. For tests."""
    global _RESOLVED, _FLUSH_THREAD
    shutdown_tracing()
    with _FLUSH_LOCK:
        _FLUSH_THREAD = None
    _RESOLVED = False
    _WARNED.clear()


def _autostart() -> None:
    global _RESOLVED
    with _LOCK:
        if _RESOLVED:
            return
        _RESOLVED = True
        try:
            wanted = _wants_tracing()
        except Exception:
            _swallow("autostart")
            return
        if wanted:
            setup_tracing()


def enabled() -> bool:
    if _TRACER is None and not _RESOLVED:
        _autostart()
    return _TRACER is not None


# ---------------------------------------------------------------------------
# spans
# ---------------------------------------------------------------------------
def _clean(attrs: dict[str, Any]) -> dict[str, Any]:
    """Drop ``None``, prefix bare keys, and keep only types OpenTelemetry accepts."""
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if value is None:
            continue
        if "." not in key:
            key = _PREFIX + key
        if isinstance(value, str | bool | int | float):
            out[key] = value
        elif isinstance(value, list | tuple) and all(isinstance(v, str) for v in value):
            out[key] = list(value)
        else:
            out[key] = str(value)
    return out


def _message_for(exc: BaseException) -> str:
    """The exception text that may be exported: all of it with content capture, else a prefix."""
    if content_enabled():
        return str(exc)[:300]
    if type(exc).__name__ not in _SAFE_MESSAGE_TYPES:
        return ""
    text = str(exc)
    match = _HTTP_STATUS.match(text)
    if match:
        return f"HTTP {match.group(1)}"
    for prefix in _SAFE_PREFIXES:
        if text.startswith(prefix):
            return prefix
    return ""


class _Live:
    """A real span, entered as a context manager. Every OpenTelemetry call is guarded."""

    __slots__ = ("_attrs", "_name", "_root", "_root_token", "_token", "_tracer", "api", "span")

    def __init__(self, name: str, attrs: dict[str, Any], *, root: bool = False) -> None:
        self._name = name
        self._attrs = attrs
        self._root = root
        self._tracer = _TRACER
        self.api = _API
        self._token: Any = None
        self._root_token: Any = None
        self.span: Any = None

    def __enter__(self) -> _Live:
        try:
            context, trace, _status, _code = self.api
            span = self._tracer.start_span(self._name, attributes=_clean(self._attrs))
            self._token = context.attach(trace.set_span_in_context(span))
            self.span = span
            if self._root:
                self._root_token = _ROOT.set(self)
        except Exception:
            self.span = None
            _swallow(f"start:{self._name}")
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        if self.span is None:
            return False
        # Three independent guards: a failure recording the error must not leave the
        # span open or the context attached.
        try:
            if self._root_token is not None:
                _ROOT.reset(self._root_token)
            if exc is not None:
                self._record_error(exc)
        except Exception:
            _swallow(f"error:{self._name}")
        try:
            self.api[0].detach(self._token)
        except Exception:
            _swallow(f"detach:{self._name}")
        try:
            self.span.end()
        except Exception:
            _swallow(f"end:{self._name}")
        return False  # never suppress the pipeline's own exception

    def _record_error(self, exc: BaseException) -> None:
        _context, _trace, status, code = self.api
        kind = type(exc).__name__
        message = _message_for(exc)
        self.span.set_status(status(code.ERROR, f"{kind}: {message}" if message else kind))
        event = {"exception.type": kind}
        if message:
            event["exception.message"] = message
        self.span.add_event("exception", event)
        self.span.set_attribute("error.type", kind)

    def set(self, **attrs: Any) -> None:
        if self.span is None:
            return
        try:
            self.span.set_attributes(_clean(attrs))
        except Exception:
            _swallow(f"set:{self._name}")

    def content(self, *, input: Any = None, output: Any = None) -> None:
        """Attach a body, but only when content capture was asked for."""
        if self.span is None:
            return
        try:
            if not content_enabled():
                return
            body: dict[str, Any] = {}
            if input is not None:
                text = json.dumps(input, ensure_ascii=False)
                body["langfuse.observation.input"] = text
                body["input.value"] = text
            if output is not None:
                text = json.dumps(output, ensure_ascii=False)
                body["langfuse.observation.output"] = text
                body["output.value"] = text
            self.span.set_attributes(body)
        except Exception:
            _swallow(f"content:{self._name}")

    def failure(self, kind: str, **attrs: Any) -> None:
        """Mark a failure the code already classified; the call itself did not raise."""
        self.set(**{"failure_kind": kind, "langfuse.observation.level": "WARNING"}, **attrs)

    def check_answer(self, response: Any) -> None:
        """Classify a reply the answer path will treat as unusable (runs only when tracing)."""
        if self.span is None:
            return
        try:
            if str(response.finish_reason) == "length":
                return  # a truncated reply is already marked by `record`
            from .providers import extract_json

            text = response.text
            try:
                payload = extract_json(text)
            except ValueError:
                self.failure("empty_reply" if not (text or "").strip() else "unparseable")
                return
            if isinstance(payload, dict) and not str(payload.get("answer", "")).strip():
                self.failure("empty_payload")
        except Exception:
            _swallow(f"check:{self._name}")

    def record(self, response: Any, **attrs: Any) -> None:
        """Fill an LLM span from the :class:`~production_rag.providers.LLMResponse`."""
        if self.span is None:
            return
        try:
            self._record(response, attrs)
        except Exception:
            _swallow(f"record:{self._name}")

    def _record(self, response: Any, extra: dict[str, Any]) -> None:
        from .pricing import cost_usd

        prompt_tokens = int(response.prompt_tokens)
        completion_tokens = int(response.completion_tokens)
        opened = self.span.attributes or {}
        requested = opened.get("gen_ai.request.model")
        requested_system = str(opened.get("gen_ai.system") or "")
        # `FallbackProvider` is named "primary+secondary"; the response names whichever
        # actually answered, so a mismatch with the first part is a fallback.
        fallback = "+" in requested_system and response.provider != requested_system.split("+")[0]
        details: dict[str, Any] = {
            "gen_ai.system": response.provider,
            "gen_ai.provider.name": response.provider,
            "gen_ai.response.model": response.model,
            "gen_ai.usage.input_tokens": prompt_tokens,
            "gen_ai.usage.output_tokens": completion_tokens,
            "gen_ai.response.finish_reasons": [response.finish_reason]
            if response.finish_reason
            else None,
            "langfuse.observation.model.name": response.model,
            "langfuse.observation.usage_details": json.dumps(
                {
                    "input": prompt_tokens,
                    "output": completion_tokens,
                    "total": prompt_tokens + completion_tokens,
                }
            ),
            "production_rag.cached": bool(response.cached),
            "production_rag.recorded_latency_s": float(response.latency_s),
            "production_rag.model_substituted": bool(requested and requested != response.model),
            "production_rag.fallback_used": fallback,
        }
        cost = cost_usd(
            response.model, prompt_tokens, completion_tokens, provider=response.provider
        )
        if cost is None:
            _warn_once(
                f"price:{response.model}",
                f"no price for model {response.model!r}; its cost is omitted from traces",
            )
            details["production_rag.cost_known"] = False
        else:
            details["production_rag.cost_known"] = True
        if response.cached:
            # A cache hit spends nothing; the original call's cost is kept apart.
            zero = {"input": 0.0, "output": 0.0, "total": 0.0}
            details["production_rag.cost_usd"] = 0.0
            details["langfuse.observation.cost_details"] = json.dumps(zero)
            if cost is not None:
                details["production_rag.recorded_cost_usd"] = cost.total
        elif cost is not None:
            details["production_rag.cost_usd"] = cost.total
            details["langfuse.observation.cost_details"] = json.dumps(cost.as_dict())
        if fallback:
            details["production_rag.failure_kind"] = "fallback"
            details["langfuse.observation.level"] = "WARNING"
        elif str(response.finish_reason) == "length":
            details["production_rag.failure_kind"] = "truncated"
            details["langfuse.observation.level"] = "WARNING"
        details.update(extra)
        self.span.set_attributes(_clean(details))
        self.content(output=response.text)


def _off() -> bool:
    return _TRACER is None and (_RESOLVED or not enabled())


def stage(name: str, /, **attrs: Any) -> Any:
    """A child span for one pipeline stage. Use as ``with stage("fuse", fusion="rrf") as s:``."""
    if _off():
        return _NULL
    try:
        return _Live(name, attrs)
    except Exception:
        _swallow(f"stage:{name}")
        return _NULL


def query(question: str, /, **attrs: Any) -> Any:
    """The root span for one question. Re-entrant: an inner call joins the outer trace."""
    if _off():
        return _NULL
    try:
        active = _ROOT.get()
        if active is not None:
            return _PassThrough(active)
        if content_enabled():
            body = json.dumps(question, ensure_ascii=False)
            attrs = {
                **attrs,
                "production_rag.question": question,
                "langfuse.trace.input": body,
                "input.value": body,
            }
        return _Live("rag.query", attrs, root=True)
    except Exception:
        _swallow("query")
        return _NULL


class _WithPrompt:
    """An LLM span that also records the prompt on entry (content capture on)."""

    __slots__ = ("_prompt", "_span")

    def __init__(self, span: _Live, prompt: str) -> None:
        self._span = span
        self._prompt = prompt

    def __enter__(self) -> _Live:
        self._span.__enter__()
        self._span.content(input=self._prompt)
        return self._span

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> bool:
        return self._span.__exit__(exc_type, exc, tb)


def llm_span(
    name: str,
    /,
    *,
    provider: Any,
    temperature: float | None = None,
    max_tokens: int | None = None,
    json_object: bool | None = None,
    prompt: str | None = None,
    **attrs: Any,
) -> Any:
    """A span for one model call. Call ``.record(response)`` on it once the call returns."""
    if _off():
        return _NULL
    try:
        model = getattr(provider, "model", "")
        opening = {
            "gen_ai.operation.name": "chat",
            "gen_ai.system": getattr(provider, "name", ""),
            "gen_ai.provider.name": getattr(provider, "name", ""),
            "gen_ai.request.model": model,
            "gen_ai.request.temperature": temperature,
            "gen_ai.request.max_tokens": max_tokens,
            "langfuse.observation.type": "generation",
            "langfuse.observation.model.name": model,
            "production_rag.json_object": json_object,
            **attrs,
        }
        span = _Live(name, opening)
        if prompt is not None and content_enabled():
            return _WithPrompt(span, prompt)
        return span
    except Exception:
        _swallow(f"llm_span:{name}")
        return _NULL
