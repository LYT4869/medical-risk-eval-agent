from __future__ import annotations

import unittest

from agent.observability import Metrics, TraceState


class ObservabilityTest(unittest.TestCase):
    def test_traceparent_continues_trace_with_new_span(self):
        trace = TraceState.from_headers(
            "req_" + "a" * 32,
            "00-" + "b" * 32 + "-" + "c" * 16 + "-01")
        self.assertEqual(trace.trace_id, "b" * 32)
        self.assertEqual(trace.parent_span_id, "c" * 16)
        self.assertNotEqual(trace.span_id, trace.parent_span_id)
        self.assertEqual(trace.child().parent_span_id, trace.span_id)

    def test_histogram_memory_is_bounded_by_bucket_count(self):
        registry = Metrics()
        for index in range(10_000):
            registry.observe("treesem_test_duration_seconds", index / 1000,
                             operation="fixed")
        histogram = next(iter(registry._histograms.values()))
        self.assertEqual(histogram.count, 10_000)
        self.assertEqual(len(histogram.buckets), len(registry._bounds))
        rendered = registry.render()
        self.assertIn("treesem_test_duration_seconds_count", rendered)
        self.assertNotIn("request_id", rendered)

    def test_malformed_trace_and_request_ids_are_replaced(self):
        trace = TraceState.from_headers("req_attacker", "00-bad-bad-01")
        self.assertRegex(trace.request_id, r"^req_[0-9a-f]{32}$")
        self.assertRegex(trace.trace_id, r"^[0-9a-f]{32}$")
        self.assertEqual(trace.parent_span_id, "")


if __name__ == "__main__":
    unittest.main()
