"""Small, program-auditable V2 M3 Agenda, tool and close contracts.

No clinical semantics are accepted directly from an LLM. Only identified source
records may support deterministic synthetic IssueResult statements.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import date

from v2_m1.run_store import WorkflowConflict
from v2_m2.fixture_tools import SyntheticP001Tools

INITIAL_ISSUES = (
    ("I-01", "prior_history", ("E-001",)),
    ("I-02", "source_review", ("E-002", "E-003")),
    ("I-03", "model_applicability", ()),
)
MAX_ISSUES = 5
MAX_STEPS = 16
MAX_TOOL_CALLS = 5
MAX_NO_PROGRESS = 2
AS_OF = "2026-10-12"  # M3 historical synthetic T1; later LIS correction is M4.


def initial_agenda() -> list[dict]:
    return [
        {"issue_id": issue_id, "topic": topic, "status": "todo",
         "evidence_ids": [], "result_key": None}
        for issue_id, topic, _ in INITIAL_ISSUES
    ]


def active_issue(agenda: list[dict]) -> dict | None:
    return next((issue for issue in agenda if issue["status"] == "todo"), None)


class M3FixtureTools(SyntheticP001Tools):
    ALLOWED = frozenset(("get_patient_timeline", "read_patient_record"))
    SOURCES = frozenset(("E-001", "E-002", "E-003"))

    def call(self, tool: str, *, actor_id: str, patient_id: str,
             as_of: str, record_id: str | None = None):
        if (actor_id, patient_id) != ("DOC-001", "P001"):
            raise WorkflowConflict("synthetic tool identity not authorized")
        if tool not in self.ALLOWED or as_of != AS_OF:
            raise WorkflowConflict("tool or time snapshot outside M3 policy")
        if tool == "get_patient_timeline":
            if record_id is not None:
                raise WorkflowConflict("timeline takes no record ID")
            return ["E-001", "E-002", "E-003"]
        if record_id not in self.SOURCES:
            raise WorkflowConflict("M3 tool cannot access unapproved source")
        record = self.records[record_id]
        if date.fromisoformat(record["available_on"]) > date.fromisoformat(as_of):
            raise WorkflowConflict("source did not exist at selected T1 snapshot")
        return deepcopy(record)


def allowed_sources(issue: dict) -> set[str]:
    topics = {topic: set(src) for _, topic, src in INITIAL_ISSUES}
    topics["source_discrepancy"] = {"E-002", "E-003"}
    return topics[issue["topic"]]


def checked_decision(decision: object, issue: dict, state: dict) -> dict:
    """Strict decision: tools, IDs and topic enums only, never free-form facts."""
    if not isinstance(decision, dict) or not isinstance(decision.get("action"), str):
        raise WorkflowConflict("invalid single-Agent proposal")
    action = decision["action"]
    expected = allowed_sources(issue)
    if action == "tool":
        if set(decision) - {"action", "tool", "record_id"}:
            raise WorkflowConflict("unrecognized tool proposal fields")
        name = decision.get("tool")
        if name == "get_patient_timeline":
            if state["timeline_loaded"] or decision.get("record_id") is not None:
                raise WorkflowConflict("timeline already loaded or unexpected argument")
            if state["tool_calls"] >= MAX_TOOL_CALLS:
                raise WorkflowConflict("tool budget exhausted")
            return {"action": "tool", "tool": name}
        if name != "read_patient_record":
            raise WorkflowConflict("tool is not authorized by catalog")
        ref = decision.get("record_id")
        if (ref not in expected or ref in issue["evidence_ids"] or
                state["tool_calls"] >= MAX_TOOL_CALLS or not state["timeline_loaded"]):
            raise WorkflowConflict("record unrelated, duplicate, or budget exhausted")
        return {"action": "tool", "tool": name, "record_id": ref}
    if action != "close":
        raise WorkflowConflict("unsupported Agent action")
    if set(decision) - {"action", "evidence_ids", "followup"}:
        raise WorkflowConflict("unexpected close proposal data")
    refs = decision.get("evidence_ids")
    if (not isinstance(refs, list) or len(refs) != len(set(refs))
            or not all(isinstance(ref, str) for ref in refs)):
        raise WorkflowConflict("invalid close evidence IDs")
    if set(refs) != expected:
        raise WorkflowConflict("Issue Close missing required source context")
    if not set(refs).issubset(set(state["seen_evidence"])):
        raise WorkflowConflict("cannot close Issue with unseen Evidence")
    followup = decision.get("followup")
    if issue["issue_id"] == "I-02":
        if followup != "source_discrepancy":
            raise WorkflowConflict("explicit discrepancy followup proposal required")
        if any(x["topic"] == followup for x in state["agenda"]):
            raise WorkflowConflict("duplicate discrepancy investigation")
        if len(state["agenda"]) >= MAX_ISSUES:
            raise WorkflowConflict("Agenda capacity reached")
    elif followup is not None:
        raise WorkflowConflict("followup not relevant to this Issue")
    return {"action": "close", "evidence_ids": sorted(refs), "followup": followup}


def synthetic_result(issue: dict, refs: list[str], run_id: str) -> dict:
    """Controlled, source-aware text templates; *not* LLM medical reasoning."""
    issue_id = issue["issue_id"]
    assertions = {
        "I-01": [{"text": "存在患者自述类记录；不能据此确认 PPH 诊断",
                  "evidence_ids": ["E-001"]}],
        "I-02": [
            {"text": "2022 年原始记录记载估计失血量 250 mL", "evidence_ids": ["E-002"]},
            {"text": "转诊摘要记载约 800 mL", "evidence_ids": ["E-003"]},
            {"text": "不同来源之间存在数值差异，原因尚未核实",
             "evidence_ids": ["E-002", "E-003"]},
        ],
        "I-03": [{"text": "当前模型未验证适合产前任务，不执行预测",
                  "evidence_ids": []}],
        "I-04": [{"text": "原始记录 250 mL 与转诊摘要约 800 mL 并存，尚不能裁定真实数值",
                  "evidence_ids": ["E-002", "E-003"]}],
    }
    if issue_id not in assertions:
        raise WorkflowConflict("unexpected Issue ID")
    return {
        "classification": "SIM", "run_id": run_id, "issue_id": issue_id,
        "result_revision": 1,
        "status": ("blocked" if issue_id == "I-03"
                   else "closed_unresolved" if issue_id in ("I-02", "I-04")
                   else "closed_with_limitations"),
        "no_diagnosis": True,
        "statements": assertions[issue_id],
        "evidence_ids": refs,
        "synthetic_template_only": True,
    }
