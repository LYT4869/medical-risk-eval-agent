"""M7-A trace contract tests; no database or native build required."""
from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from v2_m7_trace import ReplayTrace, STEPS


class M7ReplayTraceTests(unittest.TestCase):
    def test_full_trace_is_strictly_bounded_and_json_serializable(self):
        trace = ReplayTrace()
        for scenario, steps in STEPS.items():
            for step in steps:
                trace.record(scenario, step, refs=["E-002", "I-04"],
                             metrics={"report_revision": 1})
        data = trace.finish()
        self.assertEqual(data["classification"], "SIM")
        self.assertFalse(data["model_quality_evaluated"])
        self.assertEqual(len(data["events"]), sum(map(len, STEPS.values())))
        with tempfile.TemporaryDirectory() as temp:
            filename = Path(temp) / "nested" / "replay.json"
            trace.write(filename)
            reread = json.loads(filename.read_text(encoding="utf-8"))
            self.assertEqual(reread, data)
            self.assertNotIn("token", filename.read_text(encoding="utf-8").lower())
            self.assertNotIn("run_id", filename.read_text(encoding="utf-8"))

    def test_rejects_out_of_order_incomplete_and_unknown_phases(self):
        trace = ReplayTrace()
        with self.assertRaises(ValueError):
            trace.record("late_after_delivery", "draft_delivered")
        with self.assertRaises(ValueError):
            trace.record("unlisted_scenario", "contract_confirmed")
        trace.record("late_after_delivery", "contract_confirmed")
        with self.assertRaises(ValueError):
            trace.finish()

    def test_does_not_accept_patient_text_or_credentials(self):
        trace = ReplayTrace()
        for refs in (["P001"], ["JWT-secret"], ["run_f424"],
                     ["E-002", "patient narrative"], "E-002"):
            with self.subTest(refs=refs):
                with self.assertRaises(ValueError):
                    trace.record("late_after_delivery", "contract_confirmed",
                                 refs=refs)
        for metrics in ({"jwt": 1}, {"steps": "secret"}, {"steps": True},
                        {"steps": -1}, {"steps": 100001}):
            with self.subTest(metrics=metrics):
                with self.assertRaises(ValueError):
                    trace.record("late_after_delivery", "contract_confirmed",
                                 metrics=metrics)
        self.assertEqual(trace.events, [])


if __name__ == "__main__":
    unittest.main()
