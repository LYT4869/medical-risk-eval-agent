"""Bounded synthetic M2 Graph; no LLM and no clinical prediction."""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from v2_m1.run_store import WorkflowConflict
from .artifact_store import SyntheticArtifactStore
from .fixture_tools import SyntheticP001Tools


class M2State(TypedDict, total=False):
    run_id: str
    contract_revision: int
    phase: str
    source_ids: list[str]
    issue_key: str
    report_key: str


SECTIONS = (
    ("summary", "既往分娩相关记录存在数值差异，本草稿仅陈述资料状态。"),
    ("timeline", "2022 年的分娩事件存在两份不同来源记录。"),
    ("evidence", "原始记录记载估计失血量 250 mL；转诊摘要记载约 800 mL。"),
    ("risk", "现有来源不足以确认既往 PPH 诊断或真实出血量。"),
    ("limitations", "尚未核实两份来源差异的原因；未计算产前模型风险概率。"),
    ("followup", "可由医生核对原始完整病历及转诊记录的测量口径。"),
)


def build_m2_graph(artifacts: SyntheticArtifactStore, tools: SyntheticP001Tools,
                   check_authority, checkpointer):
    def authorized(state: M2State):
        check_authority(state["run_id"], state["contract_revision"])

    def prepare(state: M2State) -> dict:
        authorized(state)
        refs = tools.call(
            "get_patient_timeline", actor_id="DOC-001",
            patient_id="P001", as_of="2026-10-18",
        )
        if refs != ["E-002", "E-003"]:
            raise WorkflowConflict("unexpected issue source list")
        return {"phase": "prepared", "source_ids": refs}

    def persist_evidence(state: M2State) -> dict:
        authorized(state)
        for ref, phase in zip(state["source_ids"],
                              ("source_original", "source_referral")):
            record = tools.call(
                "read_patient_record", actor_id="DOC-001",
                patient_id="P001", as_of="2026-10-18", record_id=ref,
            )
            if ref != record["id"]:
                raise WorkflowConflict("source identity mismatch")
            artifacts.save(
                state["run_id"], phase, f"{record['logical_id']}:v{record['version']}",
                {"evidence_id": ref, "record": record, "classification": "SIM"},
            )
        return {"phase": "evidence_persisted"}

    def close_issue(state: M2State) -> dict:
        authorized(state)
        original = tools.records["E-002"]
        referral = tools.records["E-003"]
        for record in (original, referral):
            artifacts.get(state["run_id"], "evidence",
                          f"{record['logical_id']}:v{record['version']}")
        if (original["blood_loss_ml"], referral["blood_loss_ml"]) != (250, 800):
            raise WorkflowConflict("unexpected P001 fixture values")
        result = {
            "issue_id": "I-04",
            "result_revision": 1,
            "status": "closed_unresolved",
            "statements": [
                {"text": "原始记录记载估计失血量 250 mL", "evidence_ids": ["E-002"]},
                {"text": "转诊摘要记载约 800 mL", "evidence_ids": ["E-003"]},
                {"text": "来源数值不同，原因尚未核实",
                 "evidence_ids": ["E-002", "E-003"]},
            ],
            "no_diagnosis": True,
        }
        artifacts.save(state["run_id"], "issue", "I-04:r1", result)
        return {"phase": "issue_closed", "issue_key": "I-04:r1"}

    def coverage_and_draft(state: M2State) -> dict:
        authorized(state)
        result = artifacts.get(state["run_id"], "issue", state["issue_key"])
        if (result["status"] != "closed_unresolved"
                or set(state["source_ids"]) != {"E-002", "E-003"}):
            raise WorkflowConflict("cannot build draft from unsupported Issue")
        report_id = "report_" + state["run_id"]
        draft = {
            "report_id": report_id,
            "draft_revision": 1,
            "run_id": state["run_id"],
            "contract_revision": state["contract_revision"],
            "status": "restricted_draft",
            "classification": "SIM",
            "template_only": True,
            "not_signed": True,
            "model_score": None,
            "coverage": {"I-04": "unresolved"},
            "source_snapshot": ["E-002:v1", "E-003:v1"],
            "sections": [
                {"section_id": name, "paragraph_id": f"P-{index:02}",
                 "text": body, "candidate_evidence_ids":
                    (["E-002", "E-003"] if name in
                     ("summary", "timeline", "evidence", "risk", "limitations") else [])}
                for index, (name, body) in enumerate(SECTIONS, start=1)
            ],
            # Source-ID and forbidden-string sanity checks only. M5 builds
            # actual sentence-by-sentence Assertion Grounding.
            "validation": "template_source_ids_only_not_semantic_grounding",
        }
        artifacts.save(state["run_id"], "draft", f"{report_id}:v1", draft)
        return {"phase": "draft_ready", "report_key": f"{report_id}:v1"}

    graph = StateGraph(M2State)
    graph.add_node("prepare", prepare)
    graph.add_node("persist_evidence", persist_evidence)
    graph.add_node("close_issue", close_issue)
    graph.add_node("coverage_and_draft", coverage_and_draft)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "persist_evidence")
    graph.add_edge("persist_evidence", "close_issue")
    graph.add_edge("close_issue", "coverage_and_draft")
    graph.add_edge("coverage_and_draft", END)
    return graph.compile(checkpointer=checkpointer)
