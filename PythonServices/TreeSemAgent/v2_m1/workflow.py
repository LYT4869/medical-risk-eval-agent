"""Synthetic contract HITL: revise/clarify loops to the same gate; no report gate."""
from __future__ import annotations

from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from .run_store import DemoRunStore, WorkflowConflict, operation_id


class RunState(TypedDict):
    run_id: str
    contract_revision: int
    business_revision: int
    contract_operation_id: str
    phase: str


def build_graph(store: DemoRunStore, actor_id: str, checkpointer):
    def contract_gate(state: RunState) -> dict:
        response = interrupt({
            "kind": "confirm_contract",
            "run_id": state["run_id"],
            "contract_revision": state["contract_revision"],
            "expected_revision": state["business_revision"],
            "synthetic_only": True,
        })
        if (not isinstance(response, dict)
                or response.get("expected_revision") != state["business_revision"]
                or response.get("contract_revision") != state["contract_revision"]):
            raise WorkflowConflict("invalid or stale contract response")
        result = store.apply(
            run_id=state["run_id"], actor_id=actor_id,
            phase="contract", decision=response.get("decision", ""),
            op_id=state["contract_operation_id"],
            expected_revision=state["business_revision"],
            contract_revision=state["contract_revision"],
            summary=response.get("summary"), overview=response.get("overview"),
            answer=response.get("answer"),
        )
        return {
            "phase": result["status"],
            "business_revision": result["revision"],
            "contract_revision": result["contract_revision"],
            "contract_operation_id": operation_id(
                state["run_id"], "contract", result["revision"]
            ),
        }

    def after_contract(state: RunState) -> str:
        return "contract_gate" if state["phase"] == "awaiting_confirmation" else END

    graph = StateGraph(RunState)
    graph.add_node("contract_gate", contract_gate)
    graph.add_edge(START, "contract_gate")
    graph.add_conditional_edges("contract_gate", after_contract)
    return graph.compile(checkpointer=checkpointer)
