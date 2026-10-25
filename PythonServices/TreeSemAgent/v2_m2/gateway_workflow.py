"""One authoritative C++/MySQL SIM Run for M1 HITL and M2 Artifact writes.

This is the first confirmed-Contract path: initial confirmation only.
M1-B's full revise/clarify loop remains in the isolated SQLite demo until
the native versioned Contract edit API is implemented.
"""
from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from v2_m1.run_store import WorkflowConflict, operation_id
from v2_m1.workflow import build_graph

from .fixture_tools import SyntheticP001Tools
from .gateway_adapter import GatewayDoctorClient, GatewayArtifactRepository
from .workflow import build_m2_graph


class _GatewayContractStore:
    def __init__(self, doctor: GatewayDoctorClient):
        self.doctor = doctor

    def apply(self, *, run_id: str, actor_id: str, phase: str, decision: str,
              op_id: str, expected_revision: int, contract_revision: int,
              summary=None, overview=None, answer=None):
        if (phase != "contract" or decision != "confirm"
                or contract_revision != 1 or expected_revision != 0
                or any(x is not None for x in (summary, overview, answer))
                or op_id != operation_id(run_id, "contract", expected_revision)):
            raise WorkflowConflict("M2-B gateway currently supports initial confirmation only")
        approved = self.doctor.confirm(run_id, expected_revision)
        if approved.get("status") != "contract_confirmed" or approved.get("revision") != 1:
            raise WorkflowConflict("gateway Contract was not confirmed")
        return {"status": "contract_confirmed", "revision": 1,
                "contract_revision": 1}


class GatewayContractWorkflow:
    """Reuses the *actual M1* interrupt Graph; C++ Run is the business truth."""

    def __init__(self, directory: str | Path, doctor: GatewayDoctorClient):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.doctor = doctor
        self.checkpoints = str(self.path / "gateway_m1_graph.sqlite3")

    @staticmethod
    def config(run_id: str) -> dict:
        return {"configurable": {"thread_id": "treesem-v2:gateway-contract:" + run_id}}

    def start(self, idempotency_key: str, summary: str, overview: str) -> dict:
        created = self.doctor.create(idempotency_key, summary, overview)
        run_id = created["run_id"]
        remote = self.doctor.status(run_id)
        if remote.get("classification") != "SIM" or remote.get("fixture_id") != "P001":
            raise WorkflowConflict("unrecognized synthetic Contract fixture")
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph = build_graph(_GatewayContractStore(self.doctor), "doctor", saver)
            snapshot = graph.get_state(self.config(run_id))
            if not snapshot.values:
                if remote["status"] != "awaiting_confirmation":
                    raise WorkflowConflict("remote Run already confirmed but no checkpoint exists")
                state = {
                    "run_id": run_id, "contract_revision": remote["contract_revision"],
                    "business_revision": remote["revision"],
                    "contract_operation_id": operation_id(run_id, "contract", 0),
                    "phase": remote["status"],
                }
                graph.invoke(state, config=self.config(run_id), durability="sync")
                snapshot = graph.get_state(self.config(run_id))
        return {"run_id": run_id, "business": remote["status"],
                "next": list(snapshot.next),
                "revision": remote["revision"],
                "contract_revision": remote["contract_revision"],
                "summary": remote["summary"], "overview": remote["overview"]}

    def resume_confirm(self, run_id: str, expected_revision: int,
                       contract_revision: int) -> dict:
        remote = self.doctor.status(run_id)
        if (remote.get("fixture_id") != "P001"
                or contract_revision != remote["contract_revision"]
                or expected_revision != 0
                or remote["revision"] not in (0, 1)
                or remote["status"] not in ("awaiting_confirmation", "contract_confirmed")):
            raise WorkflowConflict("unexpected Contract state")
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph = build_graph(_GatewayContractStore(self.doctor), "doctor", saver)
            snapshot = graph.get_state(self.config(run_id))
            if snapshot.next != ("contract_gate",):
                raise WorkflowConflict("the M1 Graph is not awaiting confirmation")
            if (snapshot.values["business_revision"] != expected_revision
                    or snapshot.values["contract_revision"] != contract_revision):
                raise WorkflowConflict("stale M1 confirmation snapshot")
            graph.invoke(
                Command(resume={"decision": "confirm",
                                "expected_revision": expected_revision,
                                "contract_revision": contract_revision}),
                config=self.config(run_id), durability="sync",
            )
            latest = graph.get_state(self.config(run_id))
        status = self.doctor.status(run_id)
        if latest.next or latest.values.get("phase") != "contract_confirmed":
            raise WorkflowConflict("M1 Graph did not exit confirmed")
        return {"run_id": run_id, "business": status["status"],
                "next": [], "revision": status["revision"],
                "contract_revision": status["contract_revision"]}


class GatewayM2Workflow:
    """Remote ArtifactRepository variant; no separate M1 business SQLite."""

    def __init__(self, directory: str | Path, repository: GatewayArtifactRepository):
        root = Path(directory)
        root.mkdir(parents=True, exist_ok=True)
        self.checkpoints = str(root / "gateway_m2_graph.sqlite3")
        self.artifacts = repository
        self.tools = SyntheticP001Tools()

    @staticmethod
    def config(run_id: str) -> dict:
        return {"configurable": {"thread_id": "treesem-v2:gateway-m2:" + run_id}}

    def _authorized(self, run_id: str, contract_revision: int) -> None:
        status = self.artifacts.status(run_id)
        if (status.get("run_id") != run_id
                or status.get("classification") != "SIM"
                or status.get("fixture_id") != "P001"
                or status.get("contract_revision") != contract_revision
                or status.get("status") not in ("contract_confirmed", "draft_ready")):
            raise WorkflowConflict("gateway refused confirmed P001 Contract")

    def execute(self, run_id: str) -> dict:
        snapshot = self.artifacts.status(run_id)
        revision = snapshot.get("contract_revision")
        self._authorized(run_id, revision)
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph = build_m2_graph(
                self.artifacts, self.tools, self._authorized, saver
            )
            config = self.config(run_id)
            saved = graph.get_state(config)
            if not saved.values:
                graph.invoke(
                    {"run_id": run_id, "contract_revision": revision,
                     "phase": "confirmed"},
                    config=config, durability="sync",
                )
            elif saved.next:
                graph.invoke(None, config=config, durability="sync")
            finished = graph.get_state(config)
            if finished.next or finished.values.get("phase") != "draft_ready":
                raise WorkflowConflict("M2 Gateway Graph did not finish")
            artifact_key = finished.values["report_key"]
        report = self.artifacts.get(run_id, "draft", artifact_key)
        return {"run_id": run_id, "business": "draft_ready",
                "next": [], "report": report}
