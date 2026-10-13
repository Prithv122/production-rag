"""The quiet-machine gate, and a CPU log for the duration of a run.

A latency benchmark on a busy machine measures the machine. The gate is the one the
pgvector replication used (NOTES.md, "pgvector: quiet-machine replication"): sample total
CPU once a second for 30 seconds; the run may start only if the **mean is below 15%** and
the **maximum is below 40%**, and no ``llama-server`` process exists. That gate was run by
hand with the ``\\Processor(_Total)\\% Processor Time`` counter; this module reads the same
quantity from the operating system (``GetSystemTimes`` on Windows, ``/proc/stat`` on Linux)
so the gate can be part of the command and its readings can be saved with the result.

One addition to the process rule, recorded in NOTES.md before the first run: an Ollama
model runner counts too, because on current Ollama versions the runner is what executes the
model and an idle-but-loaded one is a process the old rule did not name. Failing closed on
it costs a ``ollama stop`` and nothing else.

Everything that touches the operating system is injected, so the arithmetic is tested
without a busy or an idle machine.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

GATE_SECONDS = 30
GATE_INTERVAL_S = 1.0
GATE_MEAN_MAX = 15.0
GATE_MAX_MAX = 40.0
LOG_INTERVAL_S = 2.0

#: Cumulative ``(idle, total)`` CPU time in any consistent unit.
CpuReader = Callable[[], tuple[int, int]]


def system_cpu_reader() -> CpuReader:
    """A reader of cumulative system-wide CPU time for this platform."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class FileTime(ctypes.Structure):
            _fields_ = [("low", wintypes.DWORD), ("high", wintypes.DWORD)]

        def value(ft: FileTime) -> int:
            return (ft.high << 32) | ft.low

        def read() -> tuple[int, int]:
            idle, kernel, user = FileTime(), FileTime(), FileTime()
            if not ctypes.windll.kernel32.GetSystemTimes(
                ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)
            ):
                raise OSError("GetSystemTimes failed")
            # Kernel time already includes idle time.
            return value(idle), value(kernel) + value(user)

        return read

    if sys.platform.startswith("linux"):

        def read() -> tuple[int, int]:
            with open("/proc/stat", encoding="ascii") as handle:
                fields = [int(x) for x in handle.readline().split()[1:]]
            return fields[3] + fields[4], sum(fields[:8])  # idle + iowait; user..steal

        return read

    raise NotImplementedError(f"no CPU reader for platform {sys.platform!r}")


def utilisation(before: tuple[int, int], after: tuple[int, int]) -> float:
    """Percent busy between two cumulative readings."""
    idle = after[0] - before[0]
    total = after[1] - before[1]
    if total <= 0:
        return 0.0
    return max(0.0, min(100.0, 100.0 * (1.0 - idle / total)))


def sample_cpu(
    reader: CpuReader,
    *,
    seconds: float = GATE_SECONDS,
    interval: float = GATE_INTERVAL_S,
    sleep: Callable[[float], None] = time.sleep,
) -> list[float]:
    """`seconds / interval` readings of percent busy, one per interval."""
    readings = []
    previous = reader()
    for _ in range(round(seconds / interval)):
        sleep(interval)
        current = reader()
        readings.append(utilisation(previous, current))
        previous = current
    return readings


def busy_processes() -> list[str]:
    """Names of processes that make a machine unfit for timing: llama-server, Ollama runners."""
    if sys.platform == "win32":
        command = [
            "powershell",
            "-NoProfile",
            "-NonInteractive",
            "-Command",
            "Get-CimInstance Win32_Process | Select-Object Name,CommandLine | ConvertTo-Json",
        ]
        out = subprocess.run(command, capture_output=True, text=True, timeout=60, check=False)
        rows = json.loads(out.stdout or "[]")
        rows = [rows] if isinstance(rows, dict) else rows
        listing = [(r.get("Name") or "", r.get("CommandLine") or "") for r in rows]
    else:
        out = subprocess.run(
            ["ps", "-eo", "comm=,args="], capture_output=True, text=True, timeout=30, check=False
        )
        listing = []
        for line in out.stdout.splitlines():
            parts = line.strip().split(None, 1)
            if parts:
                listing.append((parts[0], parts[1] if len(parts) > 1 else ""))
    return flag_processes(listing)


def flag_processes(listing: list[tuple[str, str]]) -> list[str]:
    """Pick the offending entries out of ``(name, command line)`` pairs."""
    flagged = []
    for name, command in listing:
        lowered = name.lower()
        if lowered.startswith("llama-server"):
            flagged.append(name)
        elif lowered.startswith("ollama") and "runner" in command.lower():
            flagged.append(f"{name} (runner)")
    return sorted(set(flagged))


def evaluate_gate(readings: list[float], processes: list[str]) -> dict[str, Any]:
    """Apply the registered thresholds to a set of readings."""
    mean = sum(readings) / len(readings) if readings else float("nan")
    peak = max(readings) if readings else float("nan")
    passed = bool(readings) and mean < GATE_MEAN_MAX and peak < GATE_MAX_MAX and not processes
    return {
        "rule": f"mean < {GATE_MEAN_MAX:g}% and max < {GATE_MAX_MAX:g}% over "
        f"{len(readings)} one-second samples, and no llama-server or Ollama runner process",
        "samples": [round(r, 1) for r in readings],
        "mean_pct": round(mean, 1),
        "max_pct": round(peak, 1),
        "busy_processes": processes,
        "passed": passed,
    }


def run_gate(
    reader: CpuReader | None = None,
    *,
    seconds: float = GATE_SECONDS,
    interval: float = GATE_INTERVAL_S,
    lister: Callable[[], list[str]] = busy_processes,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Sample, look for offending processes, and decide."""
    reader = reader or system_cpu_reader()
    readings = sample_cpu(reader, seconds=seconds, interval=interval, sleep=sleep)
    return evaluate_gate(readings, lister())


class CpuLog:
    """Logs system CPU every `interval` seconds on a background thread, for the run's record.

    A description of the conditions, not a gate. The thread sleeps between readings, so it
    runs equally under every condition being compared.
    """

    def __init__(self, reader: CpuReader, interval: float = LOG_INTERVAL_S) -> None:
        self._reader = reader
        self._interval = interval
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="cpu-log", daemon=True)
        self.readings: list[float] = []
        self.started_at = 0.0

    def _run(self) -> None:
        previous = self._reader()
        while not self._stop.wait(self._interval):
            current = self._reader()
            self.readings.append(round(utilisation(previous, current), 1))
            previous = current

    def __enter__(self) -> CpuLog:
        self.started_at = time.time()
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        self._thread.join(timeout=5)

    def summary(self) -> dict[str, Any]:
        r = self.readings
        return {
            "interval_s": self._interval,
            "samples": r,
            "mean_pct": round(sum(r) / len(r), 1) if r else None,
            "max_pct": max(r) if r else None,
        }
