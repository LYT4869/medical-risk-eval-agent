"""Single issue-driven Investigation Agent over durable LangGraph checkpoints.

The model only proposes bounded actions; tools/IssueResult/Agenda deltas are
validated and effects journalled before the graph continues.
"""
from __future__ import annotations

from copy import deepcopy
from typing import TypedDict, Any

from langgraph.graph import START, END, StateGraph

from v2_m1.run_store import WorkflowConflict
from v2_m2.workflow import SECTIONS
from .agent import FakeInvestigationAgent
from .policy import (
    AS_OF, MAX_ISSUES, MAX_NO_PROGRESS, MAX_STEPS, MAX_TOOL_CALLS,
    M3FixtureTools, active_issue, allowed_sources, checked_decision,
    initial_agenda, synthetic_result,
)


class M3State(TypedDict, total=False):
    run_id: str
    contract_revision: int
    phase: str
    agenda: list[dict]
    seen_evidence: list[str]
    source_keys: dict[str, str]
    timeline_loaded: bool
    steps: int
    tool_calls: int
    no_progress: int
    pending: dict
    report_key: str
    trace: list[str]


def working_context(state: M3State, issue: dict, repository, tools,
                    contract_goal: str) -> dict:
    summaries = []
    for ref in issue["evidence_ids"][:3]:
        key = state["source_keys"].get(ref)
        if not key:
            continue
        item = repository.get(state["run_id"], "evidence", key)
        record = item["record"]
        # Explicit bounded, sanitized projections; not whole EMR in prompt.
        summaries.append({
            "evidence_id": ref, "source_type": record.get("kind"),
            "version": record.get("version"),
            "blood_loss_ml": record.get("blood_loss_ml"),
            "approximate": record.get("approximate", False),
        })
    return {
        "contract_goal": contract_goal[:360],
        "skill": {"id": "perinatal-investigation", "version": "SIM-v1",
                  "constraints": ["preserve_source_discrepancy",
                                  "never_infer_prior_pph_diagnosis",
                                  "no_unverified_prenatal_model"]},
        "risk_profile": {"id": "PPH-SIM", "version": 1},
        "issue": deepcopy(issue),
        "evidence_summaries": summaries,
        "timeline_loaded": state["timeline_loaded"],
        "available_tools": ["get_patient_timeline", "read_patient_record"],
        "remaining_tool_calls": MAX_TOOL_CALLS - state["tool_calls"],
    }


def build_m3_graph(repository, tools: M3FixtureTools, authorize,
                   checkpointer, contract_goal: str,
                   agent=None, *, max_steps: int = MAX_STEPS,
                   max_tool_calls: int = MAX_TOOL_CALLS):
    """authorize(run_id, contract_revision) revalidates the signed Run."""
    if max_steps < 1 or max_steps > MAX_STEPS:
        raise ValueError("invalid M3 step budget")
    if max_tool_calls < 1 or max_tool_calls > MAX_TOOL_CALLS:
        raise ValueError("invalid M3 tool budget")
    agent = agent or FakeInvestigationAgent()

    def required(state: M3State):
        authorize(state["run_id"], state["contract_revision"])

    def seed(state: M3State) -> dict:
        required(state)
        if state.get("agenda"):
            raise WorkflowConflict("Agenda initialization cannot run twice")
        return {
            "phase": "investigating", "agenda": initial_agenda(),
            "seen_evidence": [], "source_keys": {}, "timeline_loaded": False,
            "steps": 0, "tool_calls": 0, "no_progress": 0,
            "trace": ["agenda:init:I-01,I-02,I-03"],
        }

    def decide(state: M3State) -> dict:
        required(state)
        issue = active_issue(state["agenda"])
        if issue is None:
            return {"phase": "complete", "pending": {}}
        if state["steps"] >= max_steps:
            return {"phase": "blocked", "pending": {}, "trace": [
                *state["trace"], "budget:max_steps",
            ]}
        proposal = agent.decide(
            working_context(state, issue, repository, tools, contract_goal)
        )
        try:
            checked = checked_decision(proposal, issue, state)
            if (checked["action"] == "tool" and
                    state["tool_calls"] >= max_tool_calls):
                raise WorkflowConflict("M3 configured tool budget exhausted")
        except (WorkflowConflict, KeyError, TypeError, ValueError):
            failures = state["no_progress"] + 1
            return {
                "phase": "blocked" if failures >= MAX_NO_PROGRESS else "retry",
                "no_progress": failures, "steps": state["steps"] + 1,
                "pending": {},
                "trace": [*state["trace"], "guard:invalid_proposal"],
            }
        return {
            "phase": "execute", "pending": checked, "no_progress": 0,
            "trace": [*state["trace"], f"agent:{issue['issue_id']}:{checked['action']}"],
        }

    def execute(state: M3State) -> dict:
        required(state)
        selected = active_issue(state["agenda"])
        if selected is None or state["phase"] != "execute":
            raise WorkflowConflict("no valid pending Issue")
        proposal = checked_decision(state["pending"], selected, state)
        agenda = deepcopy(state["agenda"])
        issue = active_issue(agenda)
        seen = list(state["seen_evidence"])
        keys = dict(state["source_keys"])
        timeline_loaded = state["timeline_loaded"]
        calls = state["tool_calls"]
        events = list(state["trace"])
        if proposal["action"] == "tool":
            tool = proposal["tool"]
            if tool == "get_patient_timeline":
                refs = tools.call(tool, actor_id="DOC-001", patient_id="P001",
                                  as_of=AS_OF)
                if refs != ["E-001", "E-002", "E-003"]:
                    raise WorkflowConflict("unexpected synthetic timeline")
                timeline_loaded = True
                events.append("tool:timeline")
            else:
                ref = proposal["record_id"]
                data = tools.call(tool, actor_id="DOC-001",
                                  patient_id="P001", as_of=AS_OF,
                                  record_id=ref)
                if data["id"] != ref:
                    raise WorkflowConflict("source ID mismatch")
                key = f"{data['logical_id']}:v{data['version']}"
                repository.save(
                    state["run_id"], "evidence", key,
                    {"classification": "SIM", "run_id": state["run_id"],
                     "evidence_id": ref, "record": data},
                )
                if ref not in seen:
                    seen.append(ref)
                if ref not in issue["evidence_ids"]:
                    issue["evidence_ids"].append(ref)
                keys[ref] = key
                events.append(f"tool:{selected['issue_id']}:{ref}")
            calls += 1
        else:
            refs = proposal["evidence_ids"]
            if selected["issue_id"] in ("I-02", "I-04"):
                original = repository.get(
                    state["run_id"], "evidence", keys["E-002"])
                referral = repository.get(
                    state["run_id"], "evidence", keys["E-003"])
                if (original["record"].get("blood_loss_ml"),
                        referral["record"].get("blood_loss_ml"),
                        referral["record"].get("approximate")) != (250, 800, True):
                    raise WorkflowConflict("fixture discrepancy was not proven")
            result = synthetic_result(selected, refs, state["run_id"])
            artifact_key = selected["issue_id"] + ":r1"
            repository.save(state["run_id"], "issue", artifact_key, result)
            issue["result_key"] = artifact_key
            issue["status"] = (
                "blocked" if result["status"] == "blocked" else "done"
            )
            events.append(f"issue:close:{selected['issue_id']}")
            if proposal["followup"] == "source_discrepancy":
                if len(agenda) >= MAX_ISSUES or any(
                    item["issue_id"] == "I-04" for item in agenda
                ):
                    raise WorkflowConflict("followup duplicate or budget exceeded")
                agenda.append({
                    "issue_id": "I-04", "topic": "source_discrepancy",
                    "status": "todo", "evidence_ids": ["E-002", "E-003"],
                    "result_key": None,
                })
                events.append("agenda:add:I-04")
        return {
            "phase": "investigating", "agenda": agenda,
            "seen_evidence": seen, "source_keys": keys,
            "timeline_loaded": timeline_loaded,
            "tool_calls": calls, "steps": state["steps"] + 1,
            "pending": {}, "trace": events,
        }

    def finalize(state: M3State) -> dict:
        required(state)
        statuses = {item["issue_id"]: item["status"] for item in state["agenda"]}
        if statuses != {
                "I-01": "done", "I-02": "done", "I-03": "blocked", "I-04": "done"}:
            raise WorkflowConflict("required Issue closure/limitation missing")
        if set(state["seen_evidence"]) != {"E-001", "E-002", "E-003"}:
            raise WorkflowConflict("missing required synthetic Evidence")
        for item in state["agenda"]:
            result = repository.get(
                state["run_id"], "issue", item["result_key"])
            if (result["issue_id"] != item["issue_id"] or
                    not result["no_diagnosis"]):
                raise WorkflowConflict("missing or changed immutable IssueResult")
        report_id = "report_" + state["run_id"]
        report = {
            "classification": "SIM", "run_id": state["run_id"],
            "report_id": report_id, "draft_revision": 1,
            "contract_revision": state["contract_revision"],
            "status": "restricted_draft", "template_only": True,
            "not_signed": True, "model_score": None,
            "investigation_mode": "m3_single_agent",
            "source_as_of": AS_OF, "source_validity": "historical_T1_not_current",
            "source_snapshot": ["E-001:v1", "E-002:v1", "E-003:v1"],
            "coverage": {item["issue_id"]: item["status"]
                         for item in state["agenda"]},
            "issue_result_refs": [item["result_key"] for item in state["agenda"]],
            "sections": [
                {"section_id": name, "paragraph_id": f"P-{idx:02}",
                 "text": body,
                 "candidate_evidence_ids": (
                    ["E-002", "E-003"] if name in
                    ("summary", "timeline", "evidence", "risk", "limitations") else []
                 )}
                for idx, (name, body) in enumerate(SECTIONS, 1)
            ],
            "validation": "synthetic_template_source_ids_only_not_grounding",
        }
        key = report_id + ":v1"
        repository.save(state["run_id"], "draft", key, report)
        return {
            "phase": "draft_ready", "report_key": key,
            "trace": [*state["trace"], "draft:restricted:v1"],
        }

    graph = StateGraph(M3State)
    graph.add_node("seed", seed)
    graph.add_node("decide", decide)
    graph.add_node("execute", execute)
    graph.add_node("finalize", finalize)
    graph.add_edge(START, "seed")
    graph.add_edge("seed", "decide")
    def next_after_decide(state: M3State):
        if state["phase"] == "execute":
            return "execute"
        if state["phase"] == "complete":
            return "finalize"
        if state["phase"] == "blocked":
            return END
        return "decide"
    graph.add_conditional_edges("decide", next_after_decide)
    graph.add_edge("execute", "decide")
    graph.add_edge("finalize", END)
    return graph.compile(checkpointer=checkpointer)
