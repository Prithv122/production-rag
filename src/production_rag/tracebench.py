"""Measuring tracing: what it adds to a query, and where a query's time and tokens go.

Two offline commands, neither of which makes a live LLM call.

``trace-bench``
    The registered overhead measurement (NOTES.md, "Tracing: registered overhead
    budget"). The same retrieval is timed with tracing off, with tracing on and an
    in-memory exporter, and with tracing on and OTLP export to a real collector. The
    three conditions run back to back for every (question, repeat), in rotated order,
    so a drift in machine load lands on all of them.

``trace-replay``
    The full pipeline (rewrite, hybrid, rerank, generate) on a few questions, with
    every LLM and rerank result read from the committed caches, traced. It writes a
    per-stage breakdown with token counts and cost. A cache miss is an error: a replay
    that quietly calls out, or quietly degrades, would be reporting a different run.

The percentile and budget arithmetic is plain functions so the fast suite can test it.
"""

from __future__ import annotations

import json
import platform
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from . import tracing

BENCH_ARM = "hybrid_score_weighted"
BENCH_STRATEGY = "heading"
DEFAULT_REPEATS = 5
BUDGET_FLOOR_MS = 1.0
BUDGET_FRACTION = 0.05
CONDITIONS = ("off", "in_memory", "otlp")


# ---------------------------------------------------------------------------
# arithmetic (unit-tested)
# ---------------------------------------------------------------------------
def percentile(values: Sequence[float], q: float) -> float:
    """`numpy.percentile` with linear interpolation, q in 0..100."""
    import numpy as np

    return float(np.percentile(np.asarray(values, dtype=float), q))


def summarise_ms(seconds: Sequence[float]) -> dict[str, float]:
    ms = [s * 1000.0 for s in seconds]
    return {
        "n": len(ms),
        "p50_ms": percentile(ms, 50),
        "p95_ms": percentile(ms, 95),
        "mean_ms": sum(ms) / len(ms),
    }


def budget_limit_ms(off_p50_ms: float) -> float:
    """The registered limit: max(1 ms, 5% of the tracing-off p50)."""
    return max(BUDGET_FLOOR_MS, BUDGET_FRACTION * off_p50_ms)


def overhead(off: Sequence[float], on: Sequence[float]) -> dict[str, float]:
    """Added p50/p95 (difference of percentiles) and the paired per-sample difference.

    `off` and `on` are aligned: sample i of each is the same (question, repeat).
    """
    if len(off) != len(on):
        raise ValueError("off and on must be the same length")
    off_s, on_s = summarise_ms(off), summarise_ms(on)
    deltas = [(b - a) * 1000.0 for a, b in zip(off, on, strict=True)]
    return {
        "added_p50_ms": on_s["p50_ms"] - off_s["p50_ms"],
        "added_p95_ms": on_s["p95_ms"] - off_s["p95_ms"],
        "paired_median_ms": percentile(deltas, 50),
        "paired_mean_ms": sum(deltas) / len(deltas),
        "paired_p95_ms": percentile(deltas, 95),
    }


def verdict(off_p50_ms: float, added_p50_ms: float) -> dict[str, Any]:
    limit = budget_limit_ms(off_p50_ms)
    return {
        "rule": "added p50 <= max(1 ms, 5% of the tracing-off p50)",
        "off_p50_ms": off_p50_ms,
        "limit_ms": limit,
        "added_p50_ms": added_p50_ms,
        "passed": added_p50_ms <= limit,
    }


def rotated(conditions: Sequence[str], shift: int) -> list[str]:
    """`conditions` rotated left by `shift`, so no condition is always first."""
    n = len(conditions)
    return [conditions[(i + shift) % n] for i in range(n)]


# ---------------------------------------------------------------------------
# exporters used by the commands
# ---------------------------------------------------------------------------
class CountingExporter:
    """Wraps an exporter, counting the spans it was asked to send and the ones that failed."""

    def __init__(self, inner: Any) -> None:
        self.inner = inner
        self.exported = 0
        self.failed = 0

    def export(self, spans: Sequence[Any]) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        result = self.inner.export(spans)
        self.exported += len(spans)
        if result != SpanExportResult.SUCCESS:
            self.failed += len(spans)
        return result

    def shutdown(self) -> None:
        self.inner.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return bool(self.inner.force_flush(timeout_millis))


class TeeExporter:
    """Send every batch to several exporters."""

    def __init__(self, *exporters: Any) -> None:
        self.exporters = exporters

    def export(self, spans: Sequence[Any]) -> Any:
        from opentelemetry.sdk.trace.export import SpanExportResult

        results = [e.export(spans) for e in self.exporters]
        ok = all(r == SpanExportResult.SUCCESS for r in results)
        return SpanExportResult.SUCCESS if ok else SpanExportResult.FAILURE

    def shutdown(self) -> None:
        for e in self.exporters:
            e.shutdown()

    def force_flush(self, timeout_millis: int = 30_000) -> bool:
        return all(bool(e.force_flush(timeout_millis)) for e in self.exporters)


# ---------------------------------------------------------------------------
# trace-bench
# ---------------------------------------------------------------------------
def run_conditions(
    work: Callable[[str], None],
    n_items: int,
    handles: dict[str, Any],
    *,
    repeats: int,
    warmup: bool = True,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, list[float]]:
    """Time `work(item_index)` under every condition, interleaved and rotated.

    `handles` maps a condition name to a tracer handle, or ``None`` for tracing off.
    The returned lists are aligned across conditions: element i of each is the same
    (repeat, item). The swap between conditions is outside the timed region.
    """
    names = list(handles)

    def one(name: str, item: Any) -> float:
        tracing.activate(handles[name])
        started = clock()
        work(item)
        return clock() - started

    if warmup:
        for item in range(n_items):
            for name in names:
                one(name, item)

    samples: dict[str, list[float]] = {name: [] for name in names}
    pending: dict[str, float] = {}
    for repeat in range(repeats):
        for item in range(n_items):
            pending.clear()
            for name in rotated(names, item + repeat):
                pending[name] = one(name, item)
            for name in names:
                samples[name].append(pending[name])
    tracing.activate(None)
    return samples


def cmd_trace_bench(args: Any) -> int:
    from .cli import _embedder
    from .groundtruth import load_questions

    questions = [q for q in load_questions(args.questions) if q.verified != "rejected"]
    if args.limit:
        questions = questions[: args.limit]
    if not questions:
        print("no questions to time")
        return 1
    if args.arm == "hybrid_rerank" or args.arm.startswith("rerank"):
        print("trace-bench times the offline retrieval path; pick an arm without rerank/rewrite")
        return 1

    try:
        import opentelemetry.sdk.trace  # noqa: F401
    except ImportError:
        print("trace-bench needs OpenTelemetry: `uv sync --extra tracing`")
        return 1

    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    from .pipeline import Retriever

    tracing.activate(None)
    retriever = Retriever.load(
        args.indexes, args.strategy, embedder=_embedder(args), with_dense=True
    )
    texts = [q.text for q in questions]

    def work(index: int) -> None:
        retriever.retrieve(texts[index], arm=args.arm, k=args.k, pool=args.pool)

    memory_store = InMemorySpanExporter()
    memory_counter = CountingExporter(memory_store)
    handles: dict[str, Any] = {
        "off": None,
        "in_memory": tracing.build_tracer(memory_counter),
    }
    otlp_counter: CountingExporter | None = None
    if args.otlp_endpoint:
        otlp_counter = CountingExporter(tracing.otlp_exporter(args.otlp_endpoint))
        handles["otlp"] = tracing.build_tracer(otlp_counter)
    if handles["in_memory"] is None or ("otlp" in handles and handles["otlp"] is None):
        print("could not build a tracer; is the `tracing` extra installed?")
        return 1

    # How many spans one query makes, so dropped spans can be counted afterwards.
    probe_store = InMemorySpanExporter()
    probe = tracing.build_tracer(probe_store, simple=True)
    tracing.activate(probe)
    work(0)
    spans_per_query = len(probe_store.get_finished_spans())
    tracing.activate(None)

    print(
        f"{len(texts)} questions x {args.repeats} repeats x {len(handles)} conditions "
        f"({', '.join(handles)}), arm={args.arm} strategy={args.strategy} k={args.k}",
        flush=True,
    )
    if not args.otlp_endpoint:
        print("no --otlp-endpoint: the otlp condition is skipped and the verdict is not evaluated")

    started = time.perf_counter()
    samples = run_conditions(work, len(texts), handles, repeats=args.repeats)
    elapsed = time.perf_counter() - started

    for handle in handles.values():
        if handle is not None:
            handle.flush(30_000)

    result: dict[str, Any] = {
        "protocol": "NOTES.md, 'Tracing: registered overhead budget, fixed before measuring'",
        "arm": args.arm,
        "strategy": args.strategy,
        "k": args.k,
        "pool": args.pool,
        "n_questions": len(texts),
        "repeats": args.repeats,
        "samples_per_condition": len(samples["off"]),
        "warmup_passes": 1,
        "interleaved": "all conditions per (question, repeat), order rotated",
        "conditions": {name: summarise_ms(values) for name, values in samples.items()},
        "added": {},
        "spans_per_query": spans_per_query,
        "export": {},
        "elapsed_s": elapsed,
        "environment": _environment(),
    }
    expected = spans_per_query * len(texts) * (args.repeats + 1)
    result["export"]["in_memory"] = {
        "spans_expected": expected,
        "spans_exported": memory_counter.exported,
        "failed": memory_counter.failed,
    }
    for name in ("in_memory", "otlp"):
        if name in samples:
            result["added"][name] = overhead(samples["off"], samples[name])
    if otlp_counter is not None:
        result["export"]["otlp"] = {
            "endpoint": args.otlp_endpoint,
            "spans_expected": expected,
            "spans_exported": otlp_counter.exported,
            "failed": otlp_counter.failed,
        }

    off_p50 = result["conditions"]["off"]["p50_ms"]
    if "otlp" in samples:
        result["verdict"] = verdict(off_p50, result["added"]["otlp"]["added_p50_ms"])
    else:
        result["verdict"] = {"evaluated": False, "reason": "no --otlp-endpoint was given"}
    result["in_memory_vs_budget"] = {
        "informational": True,
        "limit_ms": budget_limit_ms(off_p50),
        "added_p50_ms": result["added"]["in_memory"]["added_p50_ms"],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(format_bench(result))
    print(f"\nwrote {args.out}")
    return 0


def format_bench(result: dict[str, Any]) -> str:
    lines = [f"{'condition':<10} {'n':>6} {'p50 ms':>9} {'p95 ms':>9} {'mean ms':>9}"]
    for name, row in result["conditions"].items():
        lines.append(
            f"{name:<10} {row['n']:>6} {row['p50_ms']:>9.3f} {row['p95_ms']:>9.3f} "
            f"{row['mean_ms']:>9.3f}"
        )
    lines.append("")
    lines.append(
        f"{'added':<10} {'p50 ms':>9} {'p95 ms':>9} {'paired median':>14} {'paired p95':>11}"
    )
    for name, row in result["added"].items():
        lines.append(
            f"{name:<10} {row['added_p50_ms']:>9.3f} {row['added_p95_ms']:>9.3f} "
            f"{row['paired_median_ms']:>14.3f} {row['paired_p95_ms']:>11.3f}"
        )
    for name, row in result["export"].items():
        lines.append(
            f"spans {name}: expected {row['spans_expected']}, exported {row['spans_exported']}, "
            f"failed {row['failed']}"
        )
    v = result["verdict"]
    if v.get("evaluated") is False:
        lines.append(f"verdict: not evaluated ({v['reason']})")
    else:
        lines.append(
            f"verdict (otlp): added p50 {v['added_p50_ms']:.3f} ms vs limit "
            f"{v['limit_ms']:.3f} ms -> {'PASS' if v['passed'] else 'FAIL'}"
        )
    return "\n".join(lines)


def _environment() -> dict[str, Any]:
    from .cli import _git_state

    commit, dirty = _git_state()
    import os

    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "commit": commit,
        "dirty": dirty,
        "note": "CPU load is not recorded here; the quiet-machine gate is checked separately",
    }


# ---------------------------------------------------------------------------
# trace-replay
# ---------------------------------------------------------------------------
class _NameOnlyReranker:
    """Stands in for the cross-encoder in a replay: it has a name and can never score."""

    def __init__(self, name: str) -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    def score(self, query: str, passages: Sequence[str]) -> Any:
        raise LookupError("replay: the reranker must be served from the cache, never run")


class ReplayMiss(RuntimeError):
    """A call that should have been served from a committed cache was not."""


def stage_breakdown(spans: Sequence[Any]) -> dict[str, dict[str, float]]:
    """Total seconds and span count per span name, excluding the root."""
    out: dict[str, dict[str, float]] = {}
    for span in spans:
        if span.name == "rag.query":
            continue
        row = out.setdefault(span.name, {"n": 0, "total_s": 0.0})
        row["n"] += 1
        row["total_s"] += (span.end_time - span.start_time) / 1e9
    return out


def llm_rows(spans: Sequence[Any]) -> list[dict[str, Any]]:
    rows = []
    for span in spans:
        a = span.attributes or {}
        if a.get("langfuse.observation.type") != "generation":
            continue
        rows.append(
            {
                "span": span.name,
                "provider": a.get("gen_ai.system"),
                "model": a.get("gen_ai.response.model"),
                "cached": a.get("production_rag.cached"),
                "prompt_tokens": a.get("gen_ai.usage.input_tokens", 0),
                "completion_tokens": a.get("gen_ai.usage.output_tokens", 0),
                "cost_usd": a.get("production_rag.cost_usd"),
                "cost_known": a.get("production_rag.cost_known"),
                "replayed_read_s": (span.end_time - span.start_time) / 1e9,
                "recorded_latency_s": a.get("production_rag.recorded_latency_s"),
                "failure_kind": a.get("production_rag.failure_kind"),
            }
        )
    return rows


def cmd_trace_replay(args: Any) -> int:
    from .answers import DEFAULT_SEED, FROZEN_ARM, load_frozen_retrievals, select_questions
    from .cache import JsonCache
    from .cli import _embedder, load_chunks
    from .generate import DEFAULT_ANSWER_TOKENS, generate
    from .groundtruth import load_questions
    from .pipeline import ARMS, Retriever
    from .providers import answer_budget, build_provider
    from .rerank import CachedReranker

    for label, path in (("llm", args.llm_cache), ("rerank", args.rerank_cache)):
        if not Path(path).exists() or not len(JsonCache(Path(path))):
            print(
                f"no {label} cache at {path}. Expand the committed bundles first: "
                "`uv run production-rag cache import`"
            )
            return 1
    try:
        import opentelemetry.sdk.trace  # noqa: F401
    except ImportError:
        print("trace-replay needs OpenTelemetry: `uv sync --extra tracing`")
        return 1

    spec = ARMS[args.arm]
    questions = [q for q in load_questions(args.questions) if q.verified != "rejected"]
    subset = select_questions(questions, n=args.subset, seed=DEFAULT_SEED)[: args.n]
    rankings = (
        load_frozen_retrievals(args.results, arm=args.frozen_arm, strategy=args.strategy, k=args.k)
        if args.context == "frozen"
        else {}
    )

    # Always offline: a replay that may call out is not a replay.
    rewriter = (
        build_provider(args.model_arm, cache_dir=args.llm_cache, offline=True)
        if spec.needs_llm
        else None
    )
    answerer = build_provider(
        args.answer_arm, cache_dir=args.llm_cache, offline=True, fallback=False
    )
    reranker = None
    if spec.rerank and not args.no_rerank:
        reranker = CachedReranker(
            _NameOnlyReranker(args.rerank_model), args.rerank_cache, offline=True
        )
    arm = args.arm
    if spec.rerank and args.no_rerank:
        # There is no arm that rewrites without reranking, so skipping the reranker also
        # drops the rewrite. The result file records both the requested and the used arm.
        arm = FROZEN_ARM
        print(f"--no-rerank: no rerank stage, and no rewrite either; arm {args.arm} -> {arm}")

    embedder = _embedder(args)
    retriever = Retriever.load(args.indexes, args.strategy, embedder=embedder, with_dense=True)
    chunks = {c.chunk_id: c for c in load_chunks(args.indexes, args.strategy)}

    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    store = InMemorySpanExporter()
    exporters: list[Any] = [store]
    if args.otlp_endpoint:
        exporters.append(tracing.otlp_exporter(args.otlp_endpoint))
    handle = tracing.build_tracer(TeeExporter(*exporters), simple=True)
    if handle is None:
        print("could not build a tracer; is the `tracing` extra installed?")
        return 1
    tracing.activate(handle)

    rows = []
    try:
        for question in subset:
            store.clear()
            started = time.perf_counter()
            with tracing.query(
                question.text, arm=arm, k=args.k, strategy=args.strategy, replay=True
            ) as root:
                result = retriever.retrieve(
                    question.text,
                    arm=arm,
                    k=args.k,
                    pool=args.pool,
                    reranker=reranker if ARMS[arm].rerank else None,
                    rewriter=rewriter if ARMS[arm].needs_llm else None,
                )
                if result.rewrite is not None and "replay mode" in (result.rewrite.error or ""):
                    raise ReplayMiss(f"rewrite for {question.qid}: {result.rewrite.error}")
                ids = (
                    rankings.get(question.qid, [])[: args.k]
                    if args.context == "frozen"
                    else result.chunk_ids
                )
                answer = generate(
                    question.text,
                    ids,
                    chunks,
                    answerer,
                    max_tokens=answer_budget(args.answer_arm, DEFAULT_ANSWER_TOKENS),
                )
                if answer.refusal_reason == "provider_error" and "replay mode" in answer.error:
                    raise ReplayMiss(f"generation for {question.qid}: {answer.error}")
                root.set(refused=answer.refused, qid=question.qid)
            wall = time.perf_counter() - started
            spans = list(store.get_finished_spans())
            llm = llm_rows(spans)
            known = [r["cost_usd"] for r in llm if r["cost_usd"] is not None]
            rows.append(
                {
                    "qid": question.qid,
                    "category": question.category,
                    "wall_s": wall,
                    "n_spans": len(spans),
                    "stages": stage_breakdown(spans),
                    "llm": llm,
                    "prompt_tokens": sum(r["prompt_tokens"] for r in llm),
                    "completion_tokens": sum(r["completion_tokens"] for r in llm),
                    "cost_usd": sum(known),
                    "cost_unknown_calls": sum(1 for r in llm if r["cost_usd"] is None),
                    "refused": answer.refused,
                    "refusal_reason": answer.refusal_reason,
                    "n_citations": len(answer.citations),
                }
            )
            print(f"  {question.qid}: {wall:.2f}s, {len(spans)} spans", flush=True)
    except (ReplayMiss, LookupError) as exc:
        print(f"replay miss, nothing written: {exc}")
        return 2
    finally:
        handle.flush(30_000)
        tracing.activate(None)

    out = {
        "protocol": "trace-replay: descriptive stage breakdown, not a budget",
        "arm": arm,
        "requested_arm": args.arm,
        "strategy": args.strategy,
        "k": args.k,
        "pool": args.pool,
        "n_questions": len(rows),
        "rewrite_model_arm": args.model_arm if spec.needs_llm else None,
        "answer_model_arm": args.answer_arm,
        "generation_context": args.context,
        "generation_context_note": (
            "frozen: the answer is generated from the hybrid_score_weighted top-k the answer "
            "table was recorded on, because no committed answer entry exists for the reranked "
            "context. Rewrite and rerank run on their own recorded entries."
            if args.context == "frozen"
            else "pipeline: the answer is generated from the context this run retrieved; a "
            "context without a committed entry is a replay miss and an error."
        ),
        "rerank": "replayed from the committed rerank cache"
        if reranker is not None
        else "absent (--no-rerank)",
        "timing_note": (
            "LLM and rerank durations here are cache reads, not model latency. The latency "
            "recorded when each call was first made is in llm[].recorded_latency_s."
        ),
        "environment": _environment(),
        "rows": rows,
        "totals": {
            "prompt_tokens": sum(r["prompt_tokens"] for r in rows),
            "completion_tokens": sum(r["completion_tokens"] for r in rows),
            "cost_usd": sum(r["cost_usd"] for r in rows),
            "cost_unknown_calls": sum(r["cost_unknown_calls"] for r in rows),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(format_replay(out))
    print(f"\nwrote {args.out}")
    return 0


def format_replay(out: dict[str, Any]) -> str:
    names: list[str] = []
    for row in out["rows"]:
        for name in row["stages"]:
            if name not in names:
                names.append(name)
    header = (
        f"{'qid':<14}"
        + "".join(f"{n[:12]:>14}" for n in names)
        + f"{'tok in/out':>14}{'cost $':>9}"
    )
    lines = [header]
    for row in out["rows"]:
        cells = "".join(
            f"{row['stages'].get(n, {}).get('total_s', 0.0) * 1000:>12.1f}ms" for n in names
        )
        lines.append(
            f"{row['qid']:<14}{cells}"
            f"{row['prompt_tokens']:>8}/{row['completion_tokens']:<5}{row['cost_usd']:>9.4f}"
        )
    t = out["totals"]
    lines.append(
        f"total tokens in/out {t['prompt_tokens']}/{t['completion_tokens']}, "
        f"cost ${t['cost_usd']:.4f} (unpriced calls: {t['cost_unknown_calls']})"
    )
    lines.append("durations of LLM and rerank stages are cache reads (see timing_note)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# argparse wiring (kept here so cli.py only needs one call)
# ---------------------------------------------------------------------------
def register(subparsers: Any, add_component_args: Callable[[Any], None], index_dir: Path) -> None:
    from .answers import DEFAULT_RESULTS, DEFAULT_SUBSET, FROZEN_ARM
    from .chunking import CHUNKERS
    from .groundtruth import QUESTIONS_PATH
    from .pipeline import ARMS, DEFAULT_POOL
    from .providers import MODEL_ARMS

    bench = subparsers.add_parser(
        "trace-bench",
        help="time tracing off / in-memory / OTLP over the offline retrieval path",
    )
    bench.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    bench.add_argument("--indexes", type=Path, default=index_dir)
    bench.add_argument("--strategy", choices=sorted(CHUNKERS), default=BENCH_STRATEGY)
    bench.add_argument("--arm", choices=sorted(ARMS), default=BENCH_ARM)
    bench.add_argument("-k", type=int, default=10)
    bench.add_argument("--pool", type=int, default=DEFAULT_POOL)
    bench.add_argument("--repeats", type=int, default=DEFAULT_REPEATS)
    bench.add_argument("--limit", type=int, default=0, help="first N questions only (smoke run)")
    bench.add_argument(
        "--otlp-endpoint",
        default=None,
        help="base OTLP/HTTP URL, e.g. http://127.0.0.1:3000/api/public/otel for Langfuse. "
        "Headers come from OTEL_EXPORTER_OTLP_HEADERS. Without it the otlp condition is skipped",
    )
    bench.add_argument("--out", type=Path, default=Path("eval/results/tracing_overhead.json"))
    add_component_args(bench)
    bench.set_defaults(func=cmd_trace_bench)

    replay = subparsers.add_parser(
        "trace-replay",
        help="trace the full pipeline on cached LLM and rerank results; per-stage breakdown",
    )
    replay.add_argument("--questions", type=Path, default=QUESTIONS_PATH)
    replay.add_argument("--indexes", type=Path, default=index_dir)
    replay.add_argument("--results", type=Path, default=DEFAULT_RESULTS)
    replay.add_argument("--strategy", choices=sorted(CHUNKERS), default=BENCH_STRATEGY)
    replay.add_argument("--arm", choices=sorted(ARMS), default="rerank_rewrite")
    replay.add_argument("--frozen-arm", choices=sorted(ARMS), default=FROZEN_ARM)
    replay.add_argument("-k", type=int, default=10)
    replay.add_argument("--pool", type=int, default=DEFAULT_POOL)
    replay.add_argument("-n", type=int, default=10, help="first N of the answer subset")
    replay.add_argument("--subset", type=int, default=DEFAULT_SUBSET)
    replay.add_argument(
        "--answer-arm",
        choices=sorted(MODEL_ARMS),
        default="ollama-qwen-coder",
        help="the generation arm whose recorded answers are replayed",
    )
    replay.add_argument(
        "--context",
        choices=["frozen", "pipeline"],
        default="frozen",
        help="frozen: generate from the answer table's recorded top-k (replayable). "
        "pipeline: generate from this run's reranked context; a miss is an error",
    )
    replay.add_argument("--no-rerank", action="store_true", help="skip the rerank stage")
    replay.add_argument("--otlp-endpoint", default=None, help="also export the spans over OTLP")
    replay.add_argument("--out", type=Path, default=Path("eval/results/tracing_stages.json"))
    add_component_args(replay)
    replay.set_defaults(func=cmd_trace_replay)
