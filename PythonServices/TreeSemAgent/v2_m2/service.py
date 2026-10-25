"""M2 synthetic P001 integration using M1 confirmed Contract and new graph."""
from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from v2_m1.run_store import WorkflowConflict
from v2_m1.service import DemoWorkflowService
from .artifact_store import SyntheticArtifactStore
from .fixture_tools import SyntheticP001Tools
from .workflow import build_m2_graph


class M2SyntheticService:
    def __init__(self, directory: str | Path):
        self.root = Path(directory)
        self.m1 = DemoWorkflowService(directory)
        self.artifacts = SyntheticArtifactStore(self.root / "m2_artifacts.sqlite3")
        self.tools = SyntheticP001Tools()
        self.checkpoints = str(self.root / "m2_graph_checkpoints.sqlite3")

    def authorized(self, run_id: str, contract_revision: int,
                   actor_id: str = "DOC-001") -> None:
        run = self.m1.store.get(run_id, actor_id)
        if (run["subject_id"] != "P001" or run["status"] != "contract_confirmed"
                or run["contract_revision"] != contract_revision):
            raise WorkflowConflict("confirmed P001 Contract required")

    @staticmethod
    def config(run_id: str) -> dict:
        return {"configurable": {"thread_id": f"treesem-v2:m2:{run_id}"}}

    def execute(self, run_id: str, actor_id: str = "DOC-001") -> dict:
        run = self.m1.store.get(run_id, actor_id)
        self.authorized(run_id, run["contract_revision"], actor_id)
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph = build_m2_graph(
                self.artifacts, self.tools,
                lambda rid, revision: self.authorized(rid, revision, actor_id),
                saver,
            )
            config = self.config(run_id)
            snapshot = graph.get_state(config)
            if not snapshot.values:
                graph.invoke(
                    {"run_id": run_id, "contract_revision": run["contract_revision"],
                     "phase": "confirmed"},
                    config=config, durability="sync",
                )
            elif snapshot.next:
                graph.invoke(None, config=config, durability="sync")
            latest = graph.get_state(config)
            if latest.next or latest.values.get("phase") != "draft_ready":
                raise WorkflowConflict("M2 Graph has not completed")
        draft = self.artifacts.get(run_id, "draft", latest.values["report_key"])
        return {
            "run_id": run_id, "business": "draft_ready",
            "next": list(latest.next), "report": draft,
            "artifacts": self.artifacts.count(run_id),
            "operations": self.artifacts.operation_count(run_id),
        }
