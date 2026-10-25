"""M3 runs either locally against M1 demo or via authoritative C++ MySQL Run."""
from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver

from v2_m1.run_store import WorkflowConflict
from v2_m1.service import DemoWorkflowService
from v2_m2.gateway_adapter import GatewayArtifactRepository

from .agent import FakeInvestigationAgent
from .policy import M3FixtureTools
from .repository import LocalM3Repository, GatewayM3Repository
from .workflow import build_m3_graph


class _BaseWorkflow:
    def __init__(self, directory: str | Path, artifact_store, *,
                 agent=None, contract_goal: str):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.artifacts = artifact_store
        self.agent = agent or FakeInvestigationAgent()
        self.tools = M3FixtureTools()
        self.contract_goal = contract_goal
        self.checkpoints = str(root / "m3_graph_checkpoints.sqlite3")

    @staticmethod
    def config(run_id: str) -> dict:
        return {"configurable": {"thread_id": f"treesem-v2:m3:{run_id}"}}

    def _authorized(self, run_id: str, contract_revision: int) -> None:
        raise NotImplementedError

    def execute(self, run_id: str, contract_revision: int, *,
                max_steps: int = 16, max_tool_calls: int = 5) -> dict:
        self._authorized(run_id, contract_revision)
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph = build_m3_graph(
                self.artifacts, self.tools, self._authorized, saver,
                contract_goal=self.contract_goal, agent=self.agent,
                max_steps=max_steps, max_tool_calls=max_tool_calls,
            )
            config = self.config(run_id)
            previous = graph.get_state(config)
            if not previous.values:
                graph.invoke(
                    {"run_id": run_id, "contract_revision": contract_revision},
                    config=config, durability="sync",
                )
            elif previous.next:
                graph.invoke(None, config=config, durability="sync")
            current = graph.get_state(config)
        values = current.values
        result = {
            "run_id": run_id, "phase": values.get("phase"),
            "agenda": values.get("agenda", []),
            "trace": values.get("trace", []),
            "steps": values.get("steps", 0),
            "tool_calls": values.get("tool_calls", 0),
            "next": list(current.next),
        }
        if current.next:
            raise WorkflowConflict("M3 execution did not terminate")
        if values.get("phase") == "draft_ready":
            result["report"] = self.artifacts.get(
                run_id, "draft", values["report_key"])
        return result


class M3LocalWorkflow(_BaseWorkflow):
    def __init__(self, directory: str | Path, *, agent=None):
        self.m1 = DemoWorkflowService(directory)
        super().__init__(
            directory,
            LocalM3Repository(Path(directory) / "m3_business.sqlite3"),
            agent=agent,
            contract_goal="SIM P001：核查既往资料、证据冲突、模型资格",
        )

    def _authorized(self, run_id: str, contract_revision: int) -> None:
        run = self.m1.store.get(run_id, "DOC-001")
        if (run["status"] != "contract_confirmed" or
                run["subject_id"] != "P001" or
                run["contract_revision"] != contract_revision):
            raise WorkflowConflict("M3 requires confirmed synthetic Contract")


class M3GatewayWorkflow(_BaseWorkflow):
    def __init__(self, directory: str | Path, gateway: GatewayArtifactRepository,
                 *, agent=None):
        self.gateway = gateway
        super().__init__(
            directory, GatewayM3Repository(gateway), agent=agent,
            contract_goal="SIM P001：核查既往资料、证据冲突、模型资格",
        )

    def execute(self, run_id: str, contract_revision: int, **limits) -> dict:
        # Doctor-confirmed goal is fetched only from the authoritative C++
        # Run and projected into this invocation's bounded working context.
        # Do not persist the goal text in a Graph Checkpoint.
        status = self.gateway.status(run_id)
        if not isinstance(status.get("summary"), str):
            raise WorkflowConflict("authoritative Contract summary missing")
        self.contract_goal = status["summary"][:360]
        return super().execute(run_id, contract_revision, **limits)

    def _authorized(self, run_id: str, contract_revision: int) -> None:
        status = self.gateway.status(run_id)
        if (status.get("run_id") != run_id or
                status.get("classification") != "SIM" or
                status.get("fixture_id") != "P001" or
                status.get("contract_revision") != contract_revision or
                status.get("status") not in ("contract_confirmed", "draft_ready")):
            raise WorkflowConflict("gateway Run lacks a current confirmed Contract")
