"""M5: bounded Coverage → one Writer → Assertion Review → MySQL publish.

Only business refs, never report prose or tokens, enter the LangGraph
Checkpoint. The C++ server independently checks P001 claim text and source
versions atomically with publication. No mandatory doctor review interrupt.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import StateGraph, START, END

from v2_m1.run_store import WorkflowConflict
from v2_m2.gateway_adapter import (
    GatewayArtifactRepository, GatewayDoctorClient, checked_id,
)
from .catalog import FACTS, SECTION_PLAN, paragraph_text, plan
from .coverage import P001ReportContext
from .grounding import review_document
from .writer import FakeParagraphWriter


class M5State(TypedDict, total=False):
    run_id: str
    contract_revision: int
    phase: str
    report_id: str
    report_revision: int
    writer_kind: str


def operation_id(*parts: str) -> str:
    joined = "|".join(parts)
    return "op_" + hashlib.sha256(joined.encode("utf-8")).hexdigest()[:32]


def verified_document(ctx: P001ReportContext, writer=None,
                      *, draft_revision: int = 1) -> dict:
    writer = writer or FakeParagraphWriter()
    bounded = {
        "classification": "SIM",
        "confirmed_goal": ctx.business["summary"][:360],
        "contract_revision": ctx.contract_revision,
        "section_plan": plan(),
        "available_claims": [
            {"fact_id": fid, "statement": info["full"],
             "candidate_evidence_refs": info["evidence_ids"]}
            for fid, info in FACTS.items()
        ],
        "source_snapshot": ctx.coverage["source_snapshot"],
        "constraints": {
            "no_diagnosis": True, "model_score": None,
            "only_historical_E004": True,
        },
    }
    proposals = writer.write(bounded)
    sections, grounding = review_document(proposals, ctx)
    writer_kind = ("fake_catalog" if isinstance(writer, FakeParagraphWriter)
                   else "local_model_checked_catalog")
    return {
        "classification": "SIM", "run_id": ctx.run_id,
        "report_id": "m5_" + ctx.run_id,
        "draft_revision": draft_revision,
        "contract_revision": ctx.contract_revision,
        "status": "restricted_draft", "not_signed": True,
        "clinical_unverified": True,
        "synthetic_catalog_only": True, "model_score": None,
        "writer_kind": writer_kind,
        "source_snapshot": ctx.coverage["source_snapshot"],
        "coverage": deepcopy(ctx.coverage), "sections": sections,
        "grounding": grounding,
    }


def build_m5_graph(repository: GatewayArtifactRepository,
                   doctor: GatewayDoctorClient, checkpointer, *, writer=None):
    writer = writer or FakeParagraphWriter()

    def coverage(state: M5State):
        P001ReportContext(
            state["run_id"], state["contract_revision"], repository, doctor
        ).load()
        return {"phase": "coverage_checked"}

    def compose_review_publish(state: M5State):
        ctx = P001ReportContext(
            state["run_id"], state["contract_revision"], repository, doctor
        ).load()
        document = verified_document(ctx, writer)
        op = operation_id(state["run_id"], "m5", "publish",
                          str(state["contract_revision"]))
        result = repository.publish_m5(state["run_id"], op, document)
        if (
            result.get("report_id") != document["report_id"] or
            result.get("revision") != 1 or
            result.get("status") != "restricted_draft"
        ):
            raise WorkflowConflict("M5 publish acknowledgement does not match document")
        return {
            "phase": "delivered",
            "report_id": document["report_id"],
            "report_revision": 1,
            "writer_kind": document["writer_kind"],
        }

    graph = StateGraph(M5State)
    graph.add_node("coverage", coverage)
    graph.add_node("compose_review_publish", compose_review_publish)
    graph.add_edge(START, "coverage")
    graph.add_edge("coverage", "compose_review_publish")
    graph.add_edge("compose_review_publish", END)
    return graph.compile(checkpointer=checkpointer)


class M5GatewayWorkflow:
    def __init__(self, directory: str | Path, repository: GatewayArtifactRepository,
                 doctor: GatewayDoctorClient, *, writer=None):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.checkpoints = str(root / "m5_graph_checkpoints.sqlite3")
        self.repository = repository
        self.doctor = doctor
        self.writer = writer or FakeParagraphWriter()

    @staticmethod
    def config(run_id):
        return {"configurable": {"thread_id": f"treesem-v2:m5:{checked_id(run_id)}"}}

    def execute(self, run_id: str, contract_revision: int) -> dict:
        if not isinstance(contract_revision, int) or contract_revision < 1:
            raise WorkflowConflict("M5 requires selected confirmed Contract revision")
        # Read current auth before checking a potentially completed checkpoint.
        self.repository.status(run_id)
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph = build_m5_graph(
                self.repository, self.doctor, saver, writer=self.writer
            )
            config = self.config(run_id)
            last = graph.get_state(config)
            if not last.values:
                graph.invoke(
                    {"run_id": run_id, "contract_revision": contract_revision},
                    config=config, durability="sync",
                )
            elif last.next:
                graph.invoke(None, config=config, durability="sync")
            current = graph.get_state(config)
        state = current.values
        if (current.next or state.get("phase") != "delivered" or
                state.get("contract_revision") != contract_revision):
            raise WorkflowConflict("M5 did not deliver a matching report revision")
        result = self.doctor.m5_revision(
            run_id, state["report_id"], state["report_revision"]
        )
        if result["document"]["contract_revision"] != contract_revision:
            raise WorkflowConflict("delivered report Contract version drift")
        if result.get("source_freshness") != "current":
            raise WorkflowConflict("historical Draft exists but current citations are stale")
        return result


class M5DoctorSession:
    """No automatic report identity matching; caller MUST select a report."""

    def __init__(self, doctor: GatewayDoctorClient):
        self.doctor = doctor

    def explain(self, run_id: str, report_id: str, *,
                base_revision: int, paragraph_id: str, conversation_id: str):
        return self.doctor.m5_message(
            run_id, report_id, conversation_id=conversation_id,
            base_revision=base_revision, paragraph_id=paragraph_id,
            intent="explain",
        )

    def shorten(self, run_id: str, report_id: str, *,
                base_revision: int, paragraph_id: str, conversation_id: str,
                request_key: str):
        checked_id(request_key)
        # Renew short-lived capability for each doctor-initiated edit. Never
        # place an access JWT, capability or free-form chat in a checkpoint.
        capability = self.doctor.issue_capability(run_id)
        gateway = GatewayArtifactRepository(self.doctor.transport, capability)
        snapshot = self.doctor.m5_revision(run_id, report_id, base_revision)
        if snapshot["current_revision"] != base_revision:
            raise WorkflowConflict("doctor edit is based on stale report revision")
        previous = snapshot["document"]
        if (previous.get("report_id") != report_id or
                previous.get("run_id") != run_id or
                previous.get("classification") != "SIM"):
            raise WorkflowConflict("explicit report identity mismatch")
        ctx = P001ReportContext(
            run_id, previous["contract_revision"], gateway, self.doctor
        ).load()
        if previous.get("coverage") != ctx.coverage:
            raise WorkflowConflict("Report coverage is not current")
        proposals = []
        matches = 0
        for section in previous["sections"]:
            for p in section["paragraphs"]:
                proposal = {
                    "section_id": section["section_id"],
                    "paragraph_id": p["paragraph_id"],
                    "fact_ids": list(p["fact_ids"]),
                    "candidate_evidence_refs": list(p["candidate_evidence_refs"]),
                    "text": p["text"],
                }
                if p["paragraph_id"] == paragraph_id:
                    matches += 1
                    if p["render_variant"] != "full":
                        raise WorkflowConflict("paragraph already shortened")
                    proposal["text"] = paragraph_text(tuple(p["fact_ids"]), "short")
                proposals.append(proposal)
        if matches != 1:
            raise WorkflowConflict("paragraph not found in the selected report")
        sections, grounding = review_document(proposals, ctx)
        revision = deepcopy(previous)
        revision.update({
            "draft_revision": base_revision + 1,
            "writer_kind": "sim_deterministic_patch",
            "sections": sections,
            "grounding": grounding,
        })
        op = operation_id(run_id, report_id, conversation_id,
                          str(base_revision), paragraph_id, request_key)
        return self.doctor.m5_message(
            run_id, report_id, conversation_id=conversation_id,
            base_revision=base_revision, paragraph_id=paragraph_id,
            intent="shorten", operation_id=op, document=revision,
        )
