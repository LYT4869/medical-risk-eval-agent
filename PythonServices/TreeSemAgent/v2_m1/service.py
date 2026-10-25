"""Durable synthetic Contract confirmation, no real investigation or report."""
from __future__ import annotations

import os
import uuid
from pathlib import Path

os.environ.setdefault("LANGGRAPH_STRICT_MSGPACK", "true")

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

    def start(self, actor_id: str = "DOC-001", subject_id: str = "P001",
              *, ask_clarification: bool = False) -> dict:
        run_id = "run_" + uuid.uuid4().hex
        record = self.store.create(
            run_id, actor_id, subject_id, ask_clarification=ask_clarification
        )
        state = {
            "run_id": run_id,
            "contract_revision": record["contract_revision"],
            "business_revision": record["revision"],
            "phase": record["status"],
            "contract_operation_id": operation_id(run_id, "contract", 0),
        }
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph = build_graph(self.store, actor_id, saver)
            graph.invoke(state, config=self.config(run_id), durability="sync")
            snapshot = graph.get_state(self.config(run_id))
        return {
            "run_id": run_id, "business": record["status"],
            "revision": record["revision"], "next": list(snapshot.next),
            "contract": self.store.contract(run_id, actor_id),
        }

    def status(self, run_id: str, actor_id: str = "DOC-001") -> dict:
        record = self.store.get(run_id, actor_id)
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph = build_graph(self.store, actor_id, saver)
            snapshot = graph.get_state(self.config(run_id))
        return {
            "run_id": run_id, "business": record["status"],
            "revision": record["revision"], "next": list(snapshot.next),
            "operations": self.store.operation_count(run_id),
            "contract": self.store.contract(run_id, actor_id),
        }

    def resume(self, run_id: str, decision: str, expected_revision: int,
               contract_revision: int, *, actor_id: str = "DOC-001",
               summary: str | None = None, overview: str | None = None,
               answer: str | None = None) -> dict:
        self.store.get(run_id, actor_id)  # Synthetic identity checked per call.
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph = build_graph(self.store, actor_id, saver)
            config = self.config(run_id)
            snapshot = graph.get_state(config)
            if snapshot.next != ("contract_gate",):
                raise WorkflowConflict("graph is not awaiting doctor confirmation")
            if (snapshot.values["business_revision"] != expected_revision
                    or snapshot.values["contract_revision"] != contract_revision):
                raise WorkflowConflict("response targets a different contract snapshot")
            graph.invoke(
                Command(resume={
                    "decision": decision,
                    "expected_revision": expected_revision,
                    "contract_revision": contract_revision,
                    "summary": summary,
                    "overview": overview,
                    "answer": answer,
                }),
                config=config,
                durability="sync",
            )
            latest = graph.get_state(config)
        record = self.store.get(run_id, actor_id)
        return {
            "run_id": run_id, "business": record["status"],
            "revision": record["revision"], "next": list(latest.next),
            "operations": self.store.operation_count(run_id),
            "contract": self.store.contract(run_id, actor_id),
        }
