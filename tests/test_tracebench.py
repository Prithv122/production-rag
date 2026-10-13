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


# ---------------------------------------------------------------------------
# a verdict needs a working export
# ---------------------------------------------------------------------------
def export_row(expected=100, exported=100, failed=0):
    return {"spans_expected": expected, "spans_exported": exported, "failed": failed}


def test_a_verdict_over_a_working_export_is_pass_or_fail():
    assert tracebench.verdict(40.0, 1.0, export_row())["result"] == "pass"
    assert tracebench.verdict(40.0, 9.0, export_row())["result"] == "fail"


@pytest.mark.parametrize(
    "row", [export_row(failed=3), export_row(exported=90), export_row(exported=110)]
)
def test_a_verdict_over_a_broken_export_is_invalid_not_a_pass(row):
    got = tracebench.verdict(40.0, 0.1, row)  # a tiny overhead that would otherwise pass
    assert got["result"] == "invalid: export failed"
    assert got["valid"] is False and got["passed"] is None


def test_a_failing_exporter_is_counted_and_makes_the_run_invalid():
    from opentelemetry.sdk.trace.export import SpanExportResult

    class Failing:
        def export(self, spans):
            return SpanExportResult.FAILURE

        def shutdown(self):
            pass

        def force_flush(self, timeout_millis=0):
            return True

    counter = tracebench.CountingExporter(Failing())
    tracing.activate(tracing.build_tracer(counter, simple=True))
    for _ in range(4):
        with tracing.stage("x"):
            pass
    row = {"spans_expected": 4, "spans_exported": counter.exported, "failed": counter.failed}
    assert counter.failed == 4
    assert tracebench.export_valid(row) is False
    assert tracebench.verdict(40.0, 0.0, row)["result"] == "invalid: export failed"


def test_export_worker_cpu_is_accumulated_per_exporter():
    class Burner(InMemorySpanExporter):
        def export(self, spans):
            sum(i * i for i in range(300_000))
            return super().export(spans)

    counter = tracebench.CountingExporter(Burner())
    tracing.activate(tracing.build_tracer(counter, simple=True))
    with tracing.stage("x"):
        pass
    assert counter.worker_cpu_s > 0.0
    quiet_counter = tracebench.CountingExporter(InMemorySpanExporter())
    assert quiet_counter.worker_cpu_s == 0.0


# ---------------------------------------------------------------------------
# the quiet-machine gate
# ---------------------------------------------------------------------------
from production_rag import quiet  # noqa: E402


def test_utilisation_is_busy_share_between_two_readings():
    assert quiet.utilisation((0, 0), (75, 100)) == pytest.approx(25.0)
    assert quiet.utilisation((10, 10), (10, 10)) == 0.0  # no elapsed time


def test_sampling_takes_one_reading_per_interval():
    ticks = iter(range(0, 1000, 10))

    def reader():
        t = next(ticks)
        return t // 2, t  # always 50% busy

    sleeps: list[float] = []
    readings = quiet.sample_cpu(reader, seconds=5, interval=1.0, sleep=sleeps.append)
    assert readings == [50.0] * 5 and sleeps == [1.0] * 5


@pytest.mark.parametrize(
    ("readings", "processes", "passed"),
    [
        ([5.0] * 30, [], True),
        ([14.9] * 30, [], True),
        ([15.0] * 30, [], False),  # mean must be below 15
        ([5.0] * 29 + [40.0], [], False),  # max must be below 40
        ([5.0] * 29 + [39.9], [], True),
        ([5.0] * 30, ["llama-server.exe"], False),
        ([], [], False),
    ],
)
def test_gate_thresholds_are_the_registered_ones(readings, processes, passed):
    assert quiet.evaluate_gate(readings, processes)["passed"] is passed


def test_llama_server_and_ollama_runners_are_flagged_but_an_idle_ollama_app_is_not():
    listing = [
        ("llama-server.exe", ""),
        ("ollama.exe", "ollama.exe runner --model x"),
        ("ollama app.exe", ""),
        ("ollama.exe", "ollama.exe serve"),
        ("python.exe", "python llama-server-notes.py"),
    ]
    assert quiet.flag_processes(listing) == ["llama-server.exe", "ollama.exe (runner)"]


def test_the_gate_fails_closed_when_it_cannot_be_evaluated():
    def broken():
        raise NotImplementedError("no CPU reader")

    record, proceed = tracebench.check_gate(False, broken)
    assert proceed is False and "NotImplementedError" in record["error"]
    assert record["skipped"] is False


def test_a_failed_gate_stops_the_run_and_a_pass_lets_it_go():
    quiet_record = quiet.evaluate_gate([2.0] * 30, [])
    busy_record = quiet.evaluate_gate([60.0] * 30, [])
    assert tracebench.check_gate(False, lambda: dict(quiet_record))[1] is True
    assert tracebench.check_gate(False, lambda: dict(busy_record))[1] is False


def test_the_override_is_allowed_and_recorded():
    record, proceed = tracebench.check_gate(True, lambda: pytest.fail("must not sample"))
    assert proceed is True and record["skipped"] is True and record["passed"] is None


def test_the_cpu_log_collects_readings_on_its_own_interval():
    import time

    counter = iter(range(0, 10_000, 100))

    def reader():
        t = next(counter)
        return t // 4, t  # 75% busy

    with quiet.CpuLog(reader, interval=0.01) as log:
        time.sleep(0.15)
    summary = log.summary()
    assert summary["samples"] and all(s == 75.0 for s in summary["samples"])
    assert summary["mean_pct"] == 75.0 and summary["max_pct"] == 75.0


def test_trace_bench_has_an_explicit_gate_override():
    args = build_parser().parse_args(["trace-bench"])
    assert args.skip_quiet_gate is False
    assert build_parser().parse_args(["trace-bench", "--skip-quiet-gate"]).skip_quiet_gate is True
