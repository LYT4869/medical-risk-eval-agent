from __future__ import annotations

import json
import re
import secrets
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class TraceState:
    request_id: str
    trace_id: str
    span_id: str
    parent_span_id: str

    @classmethod
    def from_headers(cls, request_id: str, traceparent: str) -> "TraceState":
        trace_id, parent = secrets.token_hex(16), ""
        matched = re.fullmatch(
            r"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})", traceparent)
        if matched and matched.group(1) != "0" * 32 and matched.group(2) != "0" * 16:
            trace_id, parent = matched.group(1), matched.group(2)
        safe_request = request_id if re.fullmatch(
            r"req_[0-9a-f]{32}", request_id) else "req_" + secrets.token_hex(16)
        return cls(safe_request, trace_id,
                   secrets.token_hex(8), parent)


@dataclass
class _Histogram:
    buckets: list[int]
    count: int = 0
    total: float = 0.0


def trace_event(trace: TraceState, started: float, outcome: str, **fields: object) -> None:
    event: dict[str, object] = {
        "event": "trace_span", "service": "treesem-knowledge",
        "request_id": trace.request_id, "trace_id": trace.trace_id,
        "span_id": trace.span_id, "parent_span_id": trace.parent_span_id,
        "operation": "knowledge.search_medical_knowledge",
        "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "outcome": outcome,
    }
    event.update(fields)
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


class Metrics:
    _bounds = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
               1.0, 2.5, 5.0, 10.0, 30.0)

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
        self._durations: dict[tuple[str, tuple[tuple[str, str], ...]], _Histogram] = {}

    def increment(self, name: str, **labels: str) -> None:
        key = name, tuple(sorted(labels.items()))
        with self._lock:
            self._counters[key] = self._counters.get(key, 0) + 1

    def observe(self, name: str, value: float, **labels: str) -> None:
        key = name, tuple(sorted(labels.items()))
        with self._lock:
            histogram = self._durations.setdefault(
                key, _Histogram([0] * len(self._bounds)))
            histogram.count += 1
            histogram.total += value
            for index, bound in enumerate(self._bounds):
                if value <= bound:
                    histogram.buckets[index] += 1

    @staticmethod
    def _labels(labels: tuple[tuple[str, str], ...]) -> str:
        return "" if not labels else "{" + ",".join(
            f'{key}="{value}"' for key, value in labels) + "}"

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"{name}{self._labels(labels)} {value}")
            for (name, labels), histogram in sorted(self._durations.items()):
                for bound, count in zip(self._bounds, histogram.buckets):
                    suffix = self._labels(labels + (("le", str(bound)),))
                    lines.append(f"{name}_bucket{suffix} {count}")
                suffix = self._labels(labels + (("le", "+Inf"),))
                lines.append(f"{name}_bucket{suffix} {histogram.count}")
                lines.append(f"{name}_sum{self._labels(labels)} {histogram.total}")
                lines.append(f"{name}_count{self._labels(labels)} {histogram.count}")
        return "\n".join(lines) + "\n"


metrics = Metrics()
