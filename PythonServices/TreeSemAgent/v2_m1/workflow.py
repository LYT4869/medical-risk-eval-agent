"""Two genuine LangGraph HITL interrupts, no LLM or clinical tooling yet."""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .run_store import DemoRunStore, WorkflowConflict


class RunState(TypedDict):
    run_id: str
    contract_revision: int
    business_revision: int
    contract_operation_id: str
    review_operation_id: str
    phase: str


def build_graph(store: DemoRunStore, actor_id: str, checkpointer):
    def contract_gate(state: RunState) -> dict:
        response = interrupt({"kind": "confirm_contract", "run_id": state["run_id"],
                              "contract_revision": state["contract_revision"],
                              "synthetic_only": True})
        if not isinstance(response, dict) or response.get("expected_revision") != state["business_revision"]:
            raise WorkflowConflict("invalid/stale contract response")
        result = store.apply(run_id=state["run_id"], actor_id=actor_id,
                             phase="contract", decision=response.get("decision", ""),
                             op_id=state["contract_operation_id"],
                             expected_revision=state["business_revision"],
                             contract_revision=state["contract_revision"])
        return {"phase": result["status"], "business_revision": result["revision"]}

    def after_contract(state: RunState) -> str:
        return "review_gate" if state["phase"] == "awaiting_review" else END

    def review_gate(state: RunState) -> dict:
        response = interrupt({"kind": "demo_review_only", "run_id": state["run_id"],
                              "no_report_exists": True, "synthetic_only": True})
        if not isinstance(response, dict) or response.get("expected_revision") != state["business_revision"]:
            raise WorkflowConflict("invalid/stale review response")
        result = store.apply(run_id=state["run_id"], actor_id=actor_id,
                             phase="review", decision=response.get("decision", ""),
                             op_id=state["review_operation_id"],
                             expected_revision=state["business_revision"],
                             contract_revision=state["contract_revision"])
        return {"phase": result["status"], "business_revision": result["revision"]}

    g = StateGraph(RunState)
    g.add_node("contract_gate", contract_gate)
    g.add_node("review_gate", review_gate)
    g.add_edge(START, "contract_gate")
    g.add_conditional_edges("contract_gate", after_contract)
    g.add_edge("review_gate", END)
    return g.compile(checkpointer=checkpointer)
