from __future__ import annotations

import json
import os
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
    parent_span_id: str = ""

    @classmethod
    def from_headers(cls, request_id: str | None, traceparent: str | None) -> "TraceState":
        trace_id = secrets.token_hex(16)
        parent = ""
        if traceparent:
            matched = re.fullmatch(
                r"00-([0-9a-f]{32})-([0-9a-f]{16})-([0-9a-f]{2})",
                traceparent)
            if matched and matched.group(1) != "0" * 32 and matched.group(2) != "0" * 16:
                trace_id, parent = matched.group(1), matched.group(2)
        safe_request_id = (request_id if request_id and re.fullmatch(
            r"req_[0-9a-f]{32}", request_id) else "req_" + secrets.token_hex(16))
        return cls(safe_request_id, trace_id,
                   secrets.token_hex(8), parent)

    def child(self) -> "TraceState":
        return TraceState(self.request_id, self.trace_id, secrets.token_hex(8), self.span_id)

    @property
    def traceparent(self) -> str:
        return f"00-{self.trace_id}-{self.span_id}-01"


@dataclass
class _Histogram:
    buckets: list[int]
    count: int = 0
    total: float = 0.0


def trace_event(trace: TraceState, operation: str, started: float,
                outcome: str, **fields: object) -> None:
    if os.getenv("TREESEM_TRACE_STDOUT", "true").lower() == "false":
        return
    event: dict[str, object] = {
        "event": "trace_span", "service": "treesem-agent",
        "request_id": trace.request_id, "trace_id": trace.trace_id,
        "span_id": trace.span_id, "parent_span_id": trace.parent_span_id,
        "operation": operation, "duration_ms": round((time.monotonic() - started) * 1000, 3),
        "outcome": outcome,
    }
    event.update(fields)
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)


class Metrics:
    _bounds = (0.001, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5,
               1.0, 2.5, 5.0, 10.0, 30.0)
    _structured_labels = {
        "treesem_agent_intent_router_requests_total": {
            "result": frozenset({
                "success", "intent_router_unavailable",
                "invalid_intent_frame"}),
        },
        "treesem_agent_intent_router_repairs_total": {
            "result": frozenset({"success", "failed"}),
        },
        "treesem_agent_intent_router_duration_seconds": {
            "result": frozenset({
                "success", "intent_router_unavailable",
                "invalid_intent_frame"}),
        },
        "treesem_agent_intent_dispatch_total": {
            "dispatch": frozenset({
                "workflow", "composite_workflow", "clarification",
                "open_agent", "invalid"}),
        },
        "treesem_agent_workflow_executions_total": {
            "result": frozenset({"success", "failure"}),
            "recipe_class": frozenset({"single", "composite"}),
        },
        "treesem_agent_llm_calls_per_run": {},
        "treesem_agent_llm_tokens_per_run": {
            "kind": frozenset({"prompt", "completion", "total"}),
        },
        "treesem_agent_llm_duration_per_run_seconds": {},
    }

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._counters: dict[tuple[str, tuple[tuple[str, str], ...]], int] = {}
        self._histograms: dict[tuple[str, tuple[tuple[str, str], ...]], _Histogram] = {}

    @classmethod
    def _key(cls, name: str, labels: dict[str, str]) -> tuple[str, tuple[tuple[str, str], ...]]:
        contract = cls._structured_labels.get(name)
        if contract is not None:
            if set(labels) != set(contract):
                raise ValueError(
                    f"{name} labels do not match the fixed contract")
            for key, value in labels.items():
                if value not in contract[key]:
                    raise ValueError(
                        f"{name} has an invalid {key} label value")
        return name, tuple(sorted(labels.items()))

    def increment(self, name: str, **labels: str) -> None:
        self.add(name, 1, **labels)

    def add(self, name: str, amount: int, **labels: str) -> None:
        if amount < 0:
            raise ValueError("counter amount must be non-negative")
        with self._lock:
            key = self._key(name, labels)
            self._counters[key] = self._counters.get(key, 0) + amount

    def observe(self, name: str, value: float, **labels: str) -> None:
        with self._lock:
            histogram = self._histograms.setdefault(
                self._key(name, labels), _Histogram([0] * len(self._bounds)))
            histogram.count += 1
            histogram.total += value
            for index, bound in enumerate(self._bounds):
                if value <= bound:
                    histogram.buckets[index] += 1

    @staticmethod
    def _labels(labels: tuple[tuple[str, str], ...], extra: tuple[str, str] | None = None) -> str:
        values = list(labels)
        if extra:
            values.append(extra)
        if not values:
            return ""
        return "{" + ",".join(f'{key}="{value}"' for key, value in values) + "}"

    def render(self) -> str:
        lines: list[str] = []
        with self._lock:
            for (name, labels), value in sorted(self._counters.items()):
                lines.append(f"{name}{self._labels(labels)} {value}")
            for (name, labels), histogram in sorted(self._histograms.items()):
                for bound, count in zip(self._bounds, histogram.buckets):
                    lines.append(f'{name}_bucket{self._labels(labels, ("le", str(bound)))} {count}')
                lines.append(f'{name}_bucket{self._labels(labels, ("le", "+Inf"))} {histogram.count}')
                lines.append(f"{name}_sum{self._labels(labels)} {histogram.total}")
                lines.append(f"{name}_count{self._labels(labels)} {histogram.count}")
        return "\n".join(lines) + "\n"


metrics = Metrics()
