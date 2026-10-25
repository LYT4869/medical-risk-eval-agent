"""Deterministic policy for SIM-only official-source correction and source admission.

Different *independent* records may disagree without superseding one another.
Only a proven, same-logical-record, higher-version official correction may
mark a prior source/derived statement stale. No diagnosis or model inference.
"""
from __future__ import annotations

from evaluation.p001_contract import load
from v2_m1.run_store import WorkflowConflict

def validate_event_fixture() -> dict:
    scenario = load()
    event = scenario["correction"]
    by_id = {x["id"]: x for x in scenario["records"]}
    old, new = by_id["E-004"], by_id["E-005"]
    original, referral = by_id["E-002"], by_id["E-003"]
    if not (
        scenario["classification"] == "SIM" and
        event["id"] == "EV-001" and event["trusted_source"] == "LIS" and
        event["kind"] == "official_correction" and
        event["logical_id"] == old["logical_id"] == new["logical_id"] and
        event["old"] == old["id"] and event["new"] == new["id"] and
        old["version"] == 1 and new["version"] == 2 and
        old["hb_g_l"] == 110 and new["hb_g_l"] == 130 and
        old["event_on"] == new["event_on"] and
        original["logical_id"] != referral["logical_id"] and
        original["blood_loss_ml"] == 250 and
        referral["blood_loss_ml"] == 800 and
        referral["approximate"] and
        scenario["model_capability"] == "not_verified_for_prenatal"
    ):
        raise WorkflowConflict("untrusted or inconsistent synthetic LIS event")
    return {
        "event_id": event["id"], "logical_id": event["logical_id"],
        "old": old["id"], "new": new["id"],
        "old_value": 110, "new_value": 130,
        "affected_statement_ids": ["S-04"], "issue_id": "I-05",
        "classification": "SIM",
    }


def verify_overlay(overlay: dict) -> dict:
    expected = validate_event_fixture()
    if overlay.get("classification") != "SIM":
        raise WorkflowConflict("correction overlay lacks SIM identity")
    sources = {v["evidence_id"]: v for v in overlay.get("source_versions", [])}
    statements = {v["statement_id"]: v for v in overlay.get("statements", [])}
    issues = {v["issue_id"]: v for v in overlay.get("followup_issues", [])}
    if (
        set(sources) != {"E-004", "E-005"} or
        sources["E-004"]["logical_id"] != expected["logical_id"] or
        sources["E-005"]["logical_id"] != expected["logical_id"] or
        (sources["E-004"]["version"], sources["E-005"]["version"]) != (1, 2) or
        (sources["E-004"]["hb_g_l"], sources["E-005"]["hb_g_l"]) != (110, 130) or
        (sources["E-004"]["validity"], sources["E-005"]["validity"]) !=
        ("stale", "current") or
        set(statements) != {"S-04"} or
        statements["S-04"]["evidence_id"] != "E-004" or
        statements["S-04"]["hb_g_l"] != 110 or
        statements["S-04"]["validity"] != "stale" or
        "EV-001" not in overlay["events"] or
        set(issues) != {"I-05"} or
        issues["I-05"]["status"] != "needs_review" or
        overlay.get("model_capability") != "not_verified_for_prenatal"
    ):
        raise WorkflowConflict("unexpected correction/impact lineage")
    return {
        "event_id": "EV-001", "old_source": "E-004",
        "current_source": "E-005", "stale_statement_ids": ["S-04"],
        "followup_issue": "I-05",
        "model_admission": "blocked_unverified_prenatal",
        "independent_conflict": ["E-002", "E-003"],
    }


def verify_report_admission(result: dict, refs: list[str], accepted: bool):
    if (result.get("scope") != "source_version_only_not_semantic_grounding"
            or result.get("evidence_ids") != refs or
            result.get("admissible") != accepted or
            result.get("model_capability") != "blocked_unverified_prenatal"):
        raise WorkflowConflict("incorrect report source version admission")
    if not accepted and "E-004" not in result.get("stale_refs", []):
        raise WorkflowConflict("stale Hb source was not rejected")
    return result
