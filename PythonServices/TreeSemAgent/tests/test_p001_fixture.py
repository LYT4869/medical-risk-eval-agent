"""Synthetic P001 source and discovery invariants (M0, no live services)."""

import copy
import unittest

from evaluation.p001_contract import FixtureError, load, validate
from evaluation.p001_trace import view_at


class P001ContractTests(unittest.TestCase):
    def setUp(self):
        self.data = load()

    def test_available_does_not_mean_retrieved(self):
        t0 = view_at(self.data, "T0")
        self.assertIn("E-003", t0["available"])
        self.assertNotIn("E-003", t0["retrieved"])
        self.assertNotIn("E-005", t0["available"])
        self.assertEqual(t0["current_by_logical_id"]["LIS-P001-HB-20261010"], "E-004")

    def test_separate_sources_stay_separate(self):
        t1 = view_at(self.data, "T1")
        self.assertEqual(t1["current_by_logical_id"]["EMR-P001-DEL-20220508"], "E-002")
        self.assertEqual(t1["current_by_logical_id"]["REF-P001-20261012"], "E-003")

    def test_correction_keeps_history_but_changes_current_hb(self):
        t2 = view_at(self.data, "T2")
        self.assertIn("E-004", t2["retrieved"])
        self.assertIn("EV-001", t2["applied_events"])
        self.assertEqual(t2["current_by_logical_id"]["LIS-P001-HB-20261010"], "E-005")
        self.assertEqual(self.data["expected"]["stale_statements"], ["S-04"])

    def test_future_result_rejected(self):
        bad = copy.deepcopy(self.data)
        bad["stages"][0]["retrieved"].append("E-005")
        bad["stages"][2]["retrieved"].remove("E-005")
        with self.assertRaises(FixtureError):
            validate(bad)

    def test_referral_is_not_a_lis_style_revision(self):
        bad = copy.deepcopy(self.data)
        bad["records"][2]["logical_id"] = "EMR-P001-DEL-20220508"
        with self.assertRaises(FixtureError):
            validate(bad)

    def test_correction_requires_trusted_source(self):
        bad = copy.deepcopy(self.data)
        bad["correction"]["trusted_source"] = "LLM"
        with self.assertRaises(FixtureError):
            validate(bad)

    def test_search_result_is_an_audit_not_patient_record(self):
        bad = copy.deepcopy(self.data)
        bad["audit"]["kind"] = "patient_record"
        with self.assertRaises(FixtureError):
            validate(bad)

    def test_unknown_stage_rejected(self):
        with self.assertRaises(FixtureError):
            view_at(self.data, "T-999")


if __name__ == "__main__":
    unittest.main()
