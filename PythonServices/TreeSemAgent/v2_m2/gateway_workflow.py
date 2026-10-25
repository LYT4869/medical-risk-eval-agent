"""One authoritative C++/MySQL SIM Run for M1 HITL and M2 Artifact writes.

M1's real Contract interrupt now supports doctor revise/clarify/reconfirm
through one C++/MySQL authoritative Run; SQLite contains graph state only.
"""
from __future__ import annotations

from pathlib import Path

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.types import Command

from v2_m1.run_store import WorkflowConflict, operation_id
from v2_m1.transitions import validate_decision
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
        if (phase != "contract" or expected_revision < 0 or
                contract_revision < 1 or
                op_id != operation_id(run_id, "contract", expected_revision)):
            raise WorkflowConflict("invalid M1 Gateway operation identity")
        summary, overview, answer = validate_decision(
            decision, summary, overview, answer
        )
        if decision == "confirm":
            result = self.doctor.confirm(
                run_id, expected_revision, contract_revision
            )
        else:
            result = self.doctor.change_contract(
                run_id, decision, expected_revision, contract_revision,
                summary=summary, overview=overview, answer=answer,
            )
        expected_status = ("contract_confirmed" if decision == "confirm"
                           else "cancelled" if decision == "cancel"
                           else "awaiting_confirmation")
        next_contract = contract_revision + (
            1 if decision in ("revise", "clarify") else 0
        )
        if (result.get("status") != expected_status or
                result.get("revision") != expected_revision + 1 or
                result.get("contract_revision") != next_contract):
            raise WorkflowConflict("unexpected authoritative Contract transition")
        return {"status": expected_status, "revision": expected_revision + 1,
                "contract_revision": next_contract}


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

    def start(self, idempotency_key: str, summary: str, overview: str,
              *, clarification_question: str | None = None) -> dict:
        created = self.doctor.create(
            idempotency_key, summary, overview,
            clarification_question=clarification_question,
        )
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
                "summary": remote["summary"], "overview": remote["overview"],
                "clarification_question": remote["clarification_question"],
                "clarification_answer": remote["clarification_answer"]}

    def resume(self, run_id: str, decision: str, expected_revision: int,
               contract_revision: int, *, summary: str | None = None,
               overview: str | None = None, answer: str | None = None) -> dict:
        # Verify input *before* Command(resume): failed interrupt execution can
        # otherwise replay a prior invalid doctor decision at the next resume.
        summary, overview, answer = validate_decision(
            decision, summary, overview, answer
        )
        remote = self.doctor.status(run_id)
        if remote.get("fixture_id") != "P001":
            raise WorkflowConflict("unexpected synthetic Contract identity")
        if not (
            (remote["revision"] == expected_revision and
             remote["contract_revision"] == contract_revision and
             remote["status"] == "awaiting_confirmation")
            or
            # SQL COMMIT succeeded, but Graph checkpoint has not advanced.
            (remote["revision"] == expected_revision + 1 and
             remote["contract_revision"] ==
                contract_revision + (1 if decision in ("revise", "clarify") else 0)
             and remote["status"] in
                ("awaiting_confirmation", "contract_confirmed", "cancelled", "draft_ready"))
        ):
            raise WorkflowConflict("stale or unrelated Contract revision")
        if remote["revision"] == expected_revision:
            if decision == "confirm" and remote["clarification_question"]:
                raise WorkflowConflict("answer pending clarification before confirm")
            if decision == "clarify" and not remote["clarification_question"]:
                raise WorkflowConflict("no pending clarification")
            if decision in ("revise", "clarify") and contract_revision >= 10:
                raise WorkflowConflict("Contract revision limit reached")
        with SqliteSaver.from_conn_string(self.checkpoints) as saver:
            graph = build_graph(_GatewayContractStore(self.doctor), "doctor", saver)
            config = self.config(run_id)
            snapshot = graph.get_state(config)
            if snapshot.next != ("contract_gate",):
                raise WorkflowConflict("M1 Gateway Graph is not awaiting confirmation")
            if (snapshot.values["business_revision"] != expected_revision or
                    snapshot.values["contract_revision"] != contract_revision):
                raise WorkflowConflict("stale checkpoint Contract revision")
            graph.invoke(Command(resume={
                "decision": decision, "expected_revision": expected_revision,
                "contract_revision": contract_revision,
                "summary": summary, "overview": overview, "answer": answer,
            }), config=config, durability="sync")
            latest = graph.get_state(config)
        updated = self.doctor.status(run_id)
        should_wait = decision in ("revise", "clarify")
        if (list(latest.next) != (["contract_gate"] if should_wait else []) or
                latest.values.get("phase") != updated["status"]):
            raise WorkflowConflict("M1 Gateway transition/checkpoint diverged")
        return {
            "run_id": run_id, "business": updated["status"],
            "next": list(latest.next), "revision": updated["revision"],
            "contract_revision": updated["contract_revision"],
            "summary": updated["summary"], "overview": updated["overview"],
            "clarification_question": updated["clarification_question"],
            "clarification_answer": updated["clarification_answer"],
        }

    def resume_confirm(self, run_id: str, expected_revision: int,
                       contract_revision: int) -> dict:
        return self.resume(run_id, "confirm", expected_revision, contract_revision)


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
