"""M4 deterministic policy; never trust merely newer contradictory sources."""
import copy
import unittest

from v2_m1.run_store import WorkflowConflict
from v2_m4.policy import (
    validate_event_fixture, verify_overlay, verify_report_admission,
)


def final_overlay():
    return {
        "classification": "SIM",
        "source_versions": [
            {"evidence_id": "E-004", "logical_id": "LIS-P001-HB-20261010",
             "version": 1, "hb_g_l": 110, "validity": "stale"},
            {"evidence_id": "E-005", "logical_id": "LIS-P001-HB-20261010",
             "version": 2, "hb_g_l": 130, "validity": "current"},
        ],
        "statements": [
            {"statement_id": "S-04", "evidence_id": "E-004",
             "hb_g_l": 110, "validity": "stale"},
            {"statement_id": "S-05", "evidence_id": "E-005",
             "hb_g_l": 130, "validity": "current"},
        ],
        "events": ["M4-HISTORY", "EV-001"],
        "followup_issues": [{
            "issue_id": "I-05", "status": "needs_review",
            "old_evidence_id": "E-004", "new_evidence_id": "E-005",
        }],
        "model_capability": "not_verified_for_prenatal",
    }


class M4PolicyTests(unittest.TestCase):
    def test_official_source_version_change_has_consistent_logical_record(self):
        ev = validate_event_fixture()
        self.assertEqual((ev["old"], ev["new"], ev["old_value"], ev["new_value"]),
                         ("E-004", "E-005", 110, 130))
        self.assertEqual(ev["affected_statement_ids"], ["S-04"])

    def test_stale_overlay_propagates_only_to_dependent_s04(self):
        result = verify_overlay(final_overlay())
        self.assertEqual(result["stale_statement_ids"], ["S-04"])
        self.assertEqual(result["current_statement"], "S-05")
        self.assertEqual(result["followup_issue"], "I-05")
        self.assertEqual(result["independent_conflict"], ["E-002", "E-003"])

    def test_modified_hb_or_record_version_is_rejected(self):
        for change in (
            ("hb_g_l", 135, 1),
            ("version", 8, 1),
            ("logical_id", "UNRELATED-LIS", 1),
        ):
            wrong = final_overlay()
            wrong["source_versions"][change[2]][change[0]] = change[1]
            with self.subTest(change=change), self.assertRaises(WorkflowConflict):
                verify_overlay(wrong)

    def test_missing_s04_stale_propagation_is_rejected(self):
        wrong = final_overlay()
        wrong["statements"][0]["validity"] = "current"
        with self.assertRaises(WorkflowConflict):
            verify_overlay(wrong)

    def test_independent_conflict_is_not_correction(self):
        wrong = final_overlay()
        wrong["source_versions"][1]["logical_id"] = "REF-P001-20261012"
        with self.assertRaises(WorkflowConflict):
            verify_overlay(wrong)

    def test_unresolved_i05_is_not_silently_closed(self):
        wrong = final_overlay()
        wrong["followup_issues"][0]["status"] = "done"
        with self.assertRaises(WorkflowConflict):
            verify_overlay(wrong)

    def test_unverified_model_must_remain_blocked(self):
        wrong = final_overlay()
        wrong["model_capability"] = "validated_prenatal_model"
        with self.assertRaises(WorkflowConflict):
            verify_overlay(wrong)

    def test_current_source_validation_checks_negative_and_positive(self):
        self.assertTrue(verify_report_admission({
            "scope": "source_version_only_not_semantic_grounding",
            "evidence_ids": ["E-005"], "admissible": True,
            "model_capability": "blocked_unverified_prenatal",
        }, ["E-005"], True)["admissible"])
        self.assertFalse(verify_report_admission({
            "scope": "source_version_only_not_semantic_grounding",
            "evidence_ids": ["E-004"], "admissible": False,
            "stale_refs": ["E-004"],
            "model_capability": "blocked_unverified_prenatal",
        }, ["E-004"], False)["admissible"])

    def test_fake_old_admission_cannot_be_accepted(self):
        with self.assertRaises(WorkflowConflict):
            verify_report_admission({
                "scope": "source_version_only_not_semantic_grounding",
                "evidence_ids": ["E-004"], "admissible": True,
                "model_capability": "blocked_unverified_prenatal",
            }, ["E-004"], False)

    def test_semantic_grounding_is_not_claimed(self):
        with self.assertRaises(WorkflowConflict):
            verify_report_admission({
                "scope": "fully_verified_clinical_grounding",
                "evidence_ids": ["E-005"], "admissible": True,
                "model_capability": "blocked_unverified_prenatal",
            }, ["E-005"], True)


if __name__ == "__main__":
    unittest.main()
