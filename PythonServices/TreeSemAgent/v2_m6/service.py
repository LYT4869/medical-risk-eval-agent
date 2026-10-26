"""M6 durable candidate recovery and fresh-source publishing.

Graph checkpoint saves ONLY Run/ref/phase. The full untrusted candidate is
staged in MySQL, read before any Writer invocation on restart, and rechecked
by the original M5 C++ publish validator after a fresh source check.
This deliberately does NOT claim concurrent multi-worker orchestration.
"""
from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.sqlite import SqliteSaver

from v2_m1.run_store import WorkflowConflict
from v2_m2.gateway_adapter import (
    GatewayArtifactRepository, GatewayDoctorClient, GatewayError, checked_id,
)
from v2_m5.coverage import P001ReportContext
from v2_m5.service import operation_id, verified_document
from v2_m5.writer import FakeParagraphWriter


class RecoveryState(TypedDict, total=False):
    run_id: str
    contract_revision: int
    phase: str
    proposal_key: str
    report_id: str
    report_revision: int


def proposal_key(run_id: str, contract_revision: int) -> str:
    return operation_id(run_id, "m6", "initial_candidate", str(contract_revision))


@contextmanager
def checkpoint_backend(sqlite_path: str, postgres_dsn: str | None = None):
    if postgres_dsn is None:
        with SqliteSaver.from_conn_string(sqlite_path) as saver:
            yield saver
    else:
        # Official LangGraph PostgresSaver; no custom MySQL checkpoint format.
        # Needs explicit approval before an actual hospital deployment.
        if not postgres_dsn.startswith(("postgresql://", "postgres://")):
            raise WorkflowConflict("M6 checkpoint requires PostgreSQL DSN")
        from langgraph.checkpoint.postgres import PostgresSaver
        with PostgresSaver.from_conn_string(postgres_dsn) as saver:
            saver.setup()
            yield saver


def prepare_proposal(repo, ctx, writer, key):
    # Never run an LLM again if the authoritative Run already has a proposal.
    try:
        cached = repo.get_m6_proposal(ctx.run_id, key)
    except GatewayError as error:
        if error.status != 404:
            raise
    else:
        _assert_staged(cached, ctx.run_id, key, ctx.contract_revision)
        return cached

    document = verified_document(ctx, writer)
    try:
        staged = repo.stage_m6_proposal(ctx.run_id, key, document)
    except GatewayError as error:
        if error.status != 409:
            raise
        # Another worker may have staged a different, still-untrusted output.
        # Discard our generated text and use the one immutable SQL winner.
        staged = repo.get_m6_proposal(ctx.run_id, key)
    _assert_staged(staged, ctx.run_id, key, ctx.contract_revision)
    return staged


def _assert_staged(result, run_id, key, contract_revision):
    if (result.get("run_id") != run_id or
            result.get("proposal_key") != key or
            result.get("contract_revision") != contract_revision or
            result.get("status") != "staged_untrusted" or
            result.get("classification") != "SIM" or
            result.get("document", {}).get("report_id") != "m5_" + run_id or
            result.get("document", {}).get("contract_revision") != contract_revision):
        raise WorkflowConflict("staged candidate belongs to another Run/Contract")


def build_graph(repo, doctor, writer, saver):
    def prepare(state: RecoveryState) -> dict:
        ctx = P001ReportContext(
            state["run_id"], state["contract_revision"], repo, doctor
        ).load()
        key = proposal_key(state["run_id"], state["contract_revision"])
        staged = prepare_proposal(repo, ctx, writer, key)
        return {"phase": "proposal_staged", "proposal_key": staged["proposal_key"]}

    def publish(state: RecoveryState) -> dict:
        if state["phase"] != "proposal_staged":
            raise WorkflowConflict("cannot publish before staging Writer candidate")
        key = proposal_key(state["run_id"], state["contract_revision"])
        if key != state["proposal_key"]:
            raise WorkflowConflict("checkpoint contains inconsistent proposal identity")
        operation = operation_id(
            state["run_id"], "m5", "publish", str(state["contract_revision"])
        )
        result = repo.publish_m6_proposal(state["run_id"], key, operation)
        if (result.get("run_id") != state["run_id"] or
                result.get("report_id") != "m5_" + state["run_id"] or
                result.get("revision") != 1 or
                result.get("status") != "restricted_draft"):
            raise WorkflowConflict("M6 publish result mismatches staged report")
        return {"phase": "delivered", "report_id": result["report_id"],
                "report_revision": 1}

    graph = StateGraph(RecoveryState)
    graph.add_node("prepare", prepare)
    graph.add_node("publish", publish)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "publish")
    graph.add_edge("publish", END)
    return graph.compile(checkpointer=saver)


class M6ReliableGatewayWorkflow:
    def __init__(self, directory: str | Path,
                 repository: GatewayArtifactRepository, doctor: GatewayDoctorClient,
                 *, writer=None, postgres_dsn: str | None = None):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.checkpoints = str(root / "m6_checkpoints.sqlite3")
        self.repo = repository
        self.doctor = doctor
        self.writer = writer or FakeParagraphWriter()
        self.postgres_dsn = postgres_dsn

    @staticmethod
    def config(run_id: str):
        return {"configurable": {"thread_id": f"treesem-v2:m6:{checked_id(run_id)}"}}

    def execute(self, run_id: str, contract_revision: int):
        if not isinstance(contract_revision, int) or contract_revision < 1:
            raise WorkflowConflict("confirmed Contract revision required")
        self.repo.status(run_id)  # fresh authorization even on completed checkpoint
        with checkpoint_backend(self.checkpoints, self.postgres_dsn) as saver:
            graph = build_graph(self.repo, self.doctor, self.writer, saver)
            config = self.config(run_id)
            old = graph.get_state(config)
            if not old.values:
                graph.invoke({
                    "run_id": run_id, "contract_revision": contract_revision,
                    "phase": "started",
                },config=config,durability="sync")
            elif old.next:
                graph.invoke(None,config=config,durability="sync")
            done = graph.get_state(config)
        if (done.next or done.values.get("phase") != "delivered" or
                done.values.get("contract_revision") != contract_revision):
            raise WorkflowConflict("recovered M6 Graph has not completed")
        report = self.doctor.m5_revision(
            run_id,done.values["report_id"],done.values["report_revision"])
        if report.get("source_freshness") != "current":
            raise WorkflowConflict(
                "report can be read as history, but no longer delivered as current")
        if report.get("document",{}).get("contract_revision") != contract_revision:
            raise WorkflowConflict("report revision/Contract binding changed")
        return report
