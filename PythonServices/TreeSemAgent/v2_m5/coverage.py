"""Whole-Run Coverage, provenance snapshot and bounded P001 admission."""
from __future__ import annotations

from v2_m1.run_store import WorkflowConflict
from .catalog import FACTS, SECTION_PLAN


class P001ReportContext:
    def __init__(self, run_id: str, contract_revision: int, gateway, doctor):
        self.run_id = run_id
        self.contract_revision = contract_revision
        self.gateway = gateway
        self.doctor = doctor
        self.business = None
        self.evidence = {}
        self.issues = {}
        self.sources = {}
        self.statements = {}
        self.overlay = None
        self.coverage = None

    def load(self) -> "P001ReportContext":
        status = self.gateway.status(self.run_id)
        if (status.get("run_id") != self.run_id or
                status.get("classification") != "SIM" or
                status.get("fixture_id") != "P001" or
                status.get("contract_revision") != self.contract_revision or
                status.get("status") not in ("draft_ready", "contract_confirmed")):
            raise WorkflowConflict("Report requires the exact authorized confirmed Run")
        self.business = status
        for ref in ("E-001", "E-002", "E-003"):
            from v2_m3.policy import M3FixtureTools
            fixture = M3FixtureTools().records[ref]
            key = f"{fixture['logical_id']}:v{fixture['version']}"
            row = self.gateway.get(self.run_id, "evidence", key)
            if (row.get("run_id") != self.run_id or
                    row.get("evidence_id") != ref or
                    row.get("record") != fixture):
                raise WorkflowConflict("original Evidence identity or snapshot changed")
            self.evidence[ref] = row["record"]
        for ref in ("I-01", "I-02", "I-03", "I-04"):
            record = self.gateway.get(self.run_id, "issue", f"{ref}:r1")
            if (record.get("run_id") != self.run_id or
                    record.get("issue_id") != ref or
                    record.get("no_diagnosis") is not True):
                raise WorkflowConflict("IssueResult provenance is not authorized")
            self.issues[ref] = record

        overlay = self.doctor.m4_state(self.run_id)
        if overlay.get("run_id") != self.run_id or overlay.get("classification") != "SIM":
            raise WorkflowConflict("M4 overlay does not belong to Report Run")
        # A later same-source official correction invalidates the fixed
        # M5 P001 report inputs. Block rather than silently presenting old Hb.
        if "EV-002" in overlay.get("events", []):
            raise WorkflowConflict(
                "new LIS source revision: regenerate coverage after bounded investigation"
            )
        from v2_m4.policy import verify_overlay
        verify_overlay(overlay)
        self.overlay = overlay
        self.sources = {row["evidence_id"]: row for row in overlay["source_versions"]}
        self.statements = {row["statement_id"]: row for row in overlay["statements"]}
        # M4's current-source gate is only a reference freshness precheck.
        current = self.doctor.validate_m4_current(
            self.run_id, ["E-001", "E-002", "E-003", "E-005"]
        )
        if (current.get("admissible") is not True or
                current.get("stale_refs") != [] or
                current.get("missing_refs") != [] or
                current.get("scope") != "source_version_only_not_semantic_grounding"):
            raise WorkflowConflict("P001 report includes missing/stale current sources")
        if self.doctor.validate_m4_current(
                self.run_id, ["E-004"]).get("stale_refs") != ["E-004"]:
            raise WorkflowConflict("old Hb was not formally invalidated")
        self.coverage = calculate_coverage(self)
        return self


def calculate_coverage(ctx: P001ReportContext):
    states = {x: ctx.issues[x]["status"] for x in ctx.issues}
    if states != {"I-01": "closed_with_limitations",
                  "I-02": "closed_unresolved",
                  "I-03": "blocked", "I-04": "closed_unresolved"}:
        raise WorkflowConflict("Issue coverage diverged from synthetic investigation")
    if (ctx.sources["E-004"]["validity"] != "stale" or
            ctx.sources["E-005"]["validity"] != "current" or
            ctx.sources["E-005"]["hb_g_l"] != 130 or
            ctx.statements["S-05"]["validity"] != "current"):
        raise WorkflowConflict("Hb source/statement is not current")
    if len(ctx.overlay["followup_issues"]) != 1 or (
        ctx.overlay["followup_issues"][0]["issue_id"],
        ctx.overlay["followup_issues"][0]["status"]
    ) != ("I-05", "needs_review"):
        raise WorkflowConflict("source-correction followup was lost")
    if (ctx.overlay["model_capability"] != "not_verified_for_prenatal" or
            ctx.business["classification"] != "SIM"):
        raise WorkflowConflict("unverified model eligibility")
    return {
        "status": "restricted",
        "method": "fixed_p001_objectives_not_semantic_contract_interpretation",
        "contract_revision": ctx.contract_revision,
        "issue_states": {**states, "I-05": "needs_review"},
        "objectives": [
            {"key": "prior_history", "disposition": "limited", "issues": ["I-01"]},
            {"key": "source_comparison", "disposition": "unresolved",
             "issues": ["I-02", "I-04"]},
            {"key": "model_reference", "disposition": "blocked", "issues": ["I-03"]},
            {"key": "lab_correction", "disposition": "needs_review", "issues": ["I-05"]},
            {"key": "literature", "disposition": "not_searched", "issues": []},
        ],
        "reason_codes": [
            "DIFFERENT_SOURCE_ESTIMATES_UNRESOLVED",
            "PRENATAL_MODEL_NOT_VERIFIED",
            "LAB_REVISION_NEEDS_REVIEW",
            "NO_APPROVED_LITERATURE_RETRIEVAL",
        ],
        "source_snapshot": ["E-001:v1", "E-002:v1", "E-003:v1",
                            "E-004:v1:historical_stale", "E-005:v2"],
        "no_clinical_diagnosis": True, "not_signed": True,
    }
