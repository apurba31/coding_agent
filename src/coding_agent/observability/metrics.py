"""Thread-safe structured timing, counter, and distribution metrics."""

from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from threading import RLock
from time import perf_counter


@dataclass(frozen=True)
class TimingSummary:
    """Aggregate statistics for one named duration measurement."""

    count: int
    total_ms: float
    average_ms: float
    max_ms: float


@dataclass(frozen=True)
class MetricsSnapshot:
    """Immutable view of accumulated counters, values, and durations."""

    timings: dict[str, TimingSummary]
    counters: dict[str, int]
    values: dict[str, tuple[float, ...]]


class MetricsCollector:
    """Collect lightweight named measurements without choosing an exporter."""

    def __init__(self) -> None:
        self._lock = RLock()
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._counters: dict[str, int] = defaultdict(int)
        self._values: dict[str, list[float]] = defaultdict(list)

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        """Record elapsed milliseconds for a named operation, including failures."""
        started = perf_counter()
        try:
            yield
        finally:
            self.record_duration(name, (perf_counter() - started) * 1000)

    def record_duration(self, name: str, duration_ms: float) -> None:
        """Record one duration sample in milliseconds."""
        if duration_ms < 0:
            raise ValueError("duration_ms cannot be negative")
        with self._lock:
            self._timings[name].append(float(duration_ms))

    def increment(self, name: str, amount: int = 1) -> None:
        """Increment a named counter."""
        with self._lock:
            self._counters[name] += amount

    def observe(self, name: str, value: float) -> None:
        """Record a numeric distribution sample, such as batch size or token count."""
        with self._lock:
            self._values[name].append(float(value))

    def snapshot(self) -> MetricsSnapshot:
        """Return aggregate timings and immutable copies of current observations."""
        with self._lock:
            timing_summaries = {
                name: TimingSummary(
                    count=len(samples),
                    total_ms=sum(samples),
                    average_ms=sum(samples) / len(samples),
                    max_ms=max(samples),
                )
                for name, samples in self._timings.items()
                if samples
            }
            return MetricsSnapshot(
                timings=timing_summaries,
                counters=dict(self._counters),
                values={name: tuple(samples) for name, samples in self._values.items()},
            )

    def clear(self) -> None:
        """Clear all accumulated metrics, primarily for tests and scoped runs."""
        with self._lock:
            self._timings.clear()
            self._counters.clear()
            self._values.clear()


_DEFAULT_COLLECTOR = MetricsCollector()


def get_metrics_collector() -> MetricsCollector:
    """Return the process-wide collector used by default across subsystems."""
    return _DEFAULT_COLLECTOR


def set_default_metrics_collector(collector: MetricsCollector) -> None:
    """Replace the process-wide collector, useful for app-scoped/test runs."""
    global _DEFAULT_COLLECTOR
    _DEFAULT_COLLECTOR = collector
