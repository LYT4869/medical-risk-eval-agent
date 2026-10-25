"""Isolated local service demonstrating durable Graph / independent business state."""
from __future__ import annotations

import uuid
from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from .run_store import DemoRunStore, WorkflowConflict, operation_id
from .workflow import build_graph


class DemoWorkflowService:
    def __init__(self, directory: str | Path):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.store = DemoRunStore(root / "demo_business.sqlite3")
        self.store.setup()
        self.checkpoints = str(root / "graph_checkpoints.sqlite3")

    @staticmethod
    def config(run_id: str) -> dict:
        return {"configurable": {"thread_id": f"treesem-v2:{run_id}"}}

    def start(self, actor_id: str = "DOC-001", subject_id: str = "P001") -> dict:
        run_id = "run_" + uuid.uuid4().hex
        record = self.store.create(run_id, actor_id, subject_id)
        state = {"run_id": run_id, "contract_revision": record["contract_revision"],
                 "business_revision": 0, "phase": record["status"],
                 "contract_operation_id": operation_id(run_id, "contract", 0),
                 "review_operation_id": operation_id(run_id, "review", 1)}
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph = build_graph(self.store, actor_id, saver)
            graph.invoke(state, config=self.config(run_id), durability="sync")
            snap = graph.get_state(self.config(run_id))
        return {"run_id": run_id, "business": record["status"], "next": list(snap.next)}

    def status(self, run_id: str, actor_id: str = "DOC-001") -> dict:
        record = self.store.get(run_id, actor_id)
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph = build_graph(self.store, actor_id, saver)
            snap = graph.get_state(self.config(run_id))
        return {"run_id": run_id, "business": record["status"],
                "revision": record["revision"], "next": list(snap.next),
                "operations": self.store.operation_count(run_id)}

    def resume(self, run_id: str, phase: str, decision: str,
               expected_revision: int, actor_id: str = "DOC-001") -> dict:
        self.store.get(run_id, actor_id)  # Re-check synthetic identity on every resume.
        node = {"contract": "contract_gate", "review": "review_gate"}.get(phase)
        if node is None:
            raise WorkflowConflict("unknown resume phase")
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph = build_graph(self.store, actor_id, saver)
            config = self.config(run_id)
            snap = graph.get_state(config)
            if snap.next != (node,):
                raise WorkflowConflict("graph is not waiting at the requested gate")
            graph.invoke(Command(resume={"decision": decision,
                                          "expected_revision": expected_revision}),
                         config=config, durability="sync")
            latest = graph.get_state(config)
        record = self.store.get(run_id, actor_id)
        return {"run_id": run_id, "business": record["status"],
                "revision": record["revision"], "next": list(latest.next),
                "operations": self.store.operation_count(run_id)}
