"""The arithmetic and plumbing behind trace-bench and trace-replay.

The commands themselves need the real indexes and encoder (they are exercised by the smoke
runs recorded in NOTES.md); what can be pinned down without them is pinned down here.
"""

from __future__ import annotations

import pytest

from production_rag import tracebench, tracing
from production_rag.cli import build_parser

pytest.importorskip("opentelemetry.sdk.trace")
from opentelemetry.sdk.trace.export.in_memory_span_exporter import (
    InMemorySpanExporter,
)


@pytest.fixture(autouse=True)
def tracing_off():
    tracing.reset()
    yield
    tracing.reset()


def test_percentile_is_linear_interpolation():
    assert tracebench.percentile([1, 2, 3, 4], 50) == 2.5
    assert tracebench.percentile([1, 2, 3, 4, 5], 95) == pytest.approx(4.8)


def test_budget_is_the_larger_of_one_millisecond_and_five_percent():
    assert tracebench.budget_limit_ms(10.0) == 1.0
    assert tracebench.budget_limit_ms(40.0) == 2.0
    assert tracebench.budget_limit_ms(20.0) == 1.0  # 5% is exactly 1 ms


def test_verdict_passes_at_the_limit_and_fails_above_it():
    assert tracebench.verdict(40.0, 2.0)["passed"] is True
    assert tracebench.verdict(40.0, 2.0001)["passed"] is False
    assert tracebench.verdict(40.0, -3.0)["passed"] is True


def test_overhead_reports_added_percentiles_and_the_paired_difference():
    off = [0.010, 0.020, 0.030, 0.040]
    on = [0.011, 0.021, 0.031, 0.041]
    result = tracebench.overhead(off, on)
    assert result["added_p50_ms"] == pytest.approx(1.0)
    assert result["paired_median_ms"] == pytest.approx(1.0)
    assert result["paired_mean_ms"] == pytest.approx(1.0)
    with pytest.raises(ValueError):
        tracebench.overhead([1.0], [1.0, 2.0])


def test_rotation_puts_every_condition_first_equally_often():
    names = ["off", "in_memory", "otlp"]
    firsts = [tracebench.rotated(names, shift)[0] for shift in range(6)]
    assert firsts == ["off", "in_memory", "otlp", "off", "in_memory", "otlp"]
    assert sorted(tracebench.rotated(names, 2)) == sorted(names)


def test_run_conditions_interleaves_aligns_and_leaves_tracing_off():
    order: list[tuple[int, str]] = []
    exporter = InMemorySpanExporter()
    handles = {
        "off": None,
        "on": tracing.build_tracer(exporter, simple=True),
    }

    def work(item: int) -> None:
        order.append((item, "on" if tracing.enabled() else "off"))
        with tracing.stage("w"):
            pass

    samples = tracebench.run_conditions(work, 3, handles, repeats=2, warmup=False)
    assert {name: len(v) for name, v in samples.items()} == {"off": 6, "on": 6}
    # Both conditions ran for each item before the next item started.
    for start in range(0, len(order), 2):
        pair = order[start : start + 2]
        assert pair[0][0] == pair[1][0] and {p[1] for p in pair} == {"on", "off"}
    assert len(exporter.get_finished_spans()) == 6  # only the "on" runs made spans
    assert tracing.enabled() is False


def test_counting_exporter_counts_what_it_forwards():
    store = InMemorySpanExporter()
    counter = tracebench.CountingExporter(store)
    handle = tracing.build_tracer(counter, simple=True)
    tracing.activate(handle)
    for _ in range(3):
        with tracing.stage("x"):
            pass
    assert (counter.exported, counter.failed) == (3, 0)


def test_stage_breakdown_sums_durations_per_name_and_skips_the_root():
    exporter = InMemorySpanExporter()
    tracing.setup_tracing(exporter, simple=True)
    with tracing.query("q", arm="a"):
        with tracing.stage("fuse"):
            pass
        with tracing.stage("fuse"):
            pass
    breakdown = tracebench.stage_breakdown(exporter.get_finished_spans())
    assert set(breakdown) == {"fuse"}
    assert breakdown["fuse"]["n"] == 2


def test_both_commands_are_registered_with_the_documented_defaults():
    parser = build_parser()
    bench = parser.parse_args(["trace-bench"])
    assert (bench.arm, bench.strategy, bench.repeats, bench.otlp_endpoint) == (
        "hybrid_score_weighted",
        "heading",
        5,
        None,
    )
    replay = parser.parse_args(["trace-replay"])
    assert (replay.arm, replay.context, replay.answer_arm) == (
        "rerank_rewrite",
        "frozen",
        "ollama-qwen-coder",
    )
