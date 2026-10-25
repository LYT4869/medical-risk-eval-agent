"""Atomic Contract revision, Run CAS, and operation journal in one demo DB."""
from __future__ import annotations

import hashlib
import json

from .run_store import DemoRunStore, WorkflowConflict, operation_id

MAX_PREVIEW_CHARS = 1200


def _text(value: str | None, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise WorkflowConflict(f"{field} must be nonempty text")
    result = value.strip()
    if len(result) > MAX_PREVIEW_CHARS:
        raise WorkflowConflict(f"{field} exceeds synthetic demo limit")
    return result



def validate_decision(decision: str, summary: str | None = None,
                      overview: str | None = None,
                      answer: str | None = None) -> tuple[str | None, str | None, str | None]:
    """Reject invalid input *before* LangGraph Command(resume), too.

    An exception inside a resumed interrupt node can leave that invalid
    resume value in the pending task; the service must preflight first.
    """
    if decision not in ("confirm", "cancel", "revise", "clarify"):
        raise WorkflowConflict("unsupported decision")
    if decision in ("confirm", "cancel") and any(
            item is not None for item in (summary, overview, answer)):
        raise WorkflowConflict("unexpected edit payload")
    if decision == "revise":
        summary = _text(summary, "summary")
        overview = _text(overview, "overview")
        if answer is not None:
            raise WorkflowConflict("clarification answer belongs to clarify")
    if decision == "clarify":
        answer = _text(answer, "answer")
        if summary is not None or overview is not None:
            raise WorkflowConflict("provide clarification separately from revised draft")
    return summary, overview, answer


def apply_transition(self: DemoRunStore, *, run_id: str, actor_id: str, phase: str,
                     decision: str, op_id: str, expected_revision: int,
                     contract_revision: int, summary: str | None = None,
                     overview: str | None = None,
                     answer: str | None = None) -> dict:
    if phase != "contract" or decision not in ("confirm", "cancel", "revise", "clarify"):
        raise WorkflowConflict("unsupported gate/decision")
    if (not isinstance(expected_revision, int) or isinstance(expected_revision, bool)
            or expected_revision < 0):
        raise WorkflowConflict("invalid business revision")
    if (not isinstance(contract_revision, int) or isinstance(contract_revision, bool)
            or contract_revision < 1):
        raise WorkflowConflict("invalid contract revision")
    if op_id != operation_id(run_id, "contract", expected_revision):
        raise WorkflowConflict("operation ID not tied to the expected run/revision")
    summary, overview, answer = validate_decision(
        decision, summary, overview, answer
    )

    request = [run_id, actor_id, phase, decision, expected_revision,
               contract_revision, summary, overview, answer]
    fingerprint = hashlib.sha256(json.dumps(
        request, ensure_ascii=False, separators=(",", ":")
    ).encode()).hexdigest()

    with self._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        run = conn.execute(
            "SELECT * FROM v2_demo_runs WHERE run_id=?", (run_id,)
        ).fetchone()
        if run is None or run["actor_id"] != actor_id or actor_id != "DOC-001":
            raise WorkflowConflict("unauthorized synthetic actor")

        # MUST precede the current-contract check: a crash after COMMIT but
        # before a graph checkpoint leaves the graph at the *old* revision.
        previous = conn.execute(
            "SELECT * FROM v2_demo_operations WHERE run_id=? AND phase=? "
            "AND operation_id=?",
            (run_id, phase, op_id),
        ).fetchone()
        if previous is not None:
            if previous["request_sha256"] != fingerprint:
                raise WorkflowConflict("operation ID reused with different payload")
            return {
                "status": previous["result_status"],
                "revision": previous["result_revision"],
                "contract_revision": previous["result_contract_revision"],
            }

        if run["contract_revision"] != contract_revision:
            raise WorkflowConflict("stale contract revision")
        if run["revision"] != expected_revision or run["status"] != "awaiting_confirmation":
            raise WorkflowConflict("stale business revision or illegal transition")
        current = conn.execute(
            "SELECT * FROM v2_demo_contracts WHERE run_id=? "
            "AND contract_revision=?",
            (run_id, contract_revision),
        ).fetchone()
        if current is None:
            raise WorkflowConflict("missing contract snapshot")
        if decision == "confirm" and current["clarification_question"]:
            raise WorkflowConflict("unanswered clarification blocks confirmation")
        if decision == "clarify" and not current["clarification_question"]:
            raise WorkflowConflict("no clarification question is pending")

        next_contract_revision = contract_revision
        if decision in ("revise", "clarify"):
            next_contract_revision += 1
            conn.execute(
                "INSERT INTO v2_demo_contracts VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id, next_contract_revision,
                    summary if decision == "revise" else current["summary"],
                    overview if decision == "revise" else current["investigation_overview"],
                    current["clarification_question"] if decision == "revise" else None,
                    current["clarification_answer"] if decision == "revise" else answer,
                ),
            )
        status = {
            "confirm": "contract_confirmed",
            "cancel": "cancelled",
            "revise": "awaiting_confirmation",
            "clarify": "awaiting_confirmation",
        }[decision]
        updated = conn.execute(
            "UPDATE v2_demo_runs SET status=?, revision=revision+1, "
            "contract_revision=? WHERE run_id=? AND status=? AND revision=? "
            "AND contract_revision=?",
            (status, next_contract_revision, run_id, "awaiting_confirmation",
             expected_revision, contract_revision),
        ).rowcount
        if updated != 1:
            raise WorkflowConflict("concurrent business transition")

        result_revision = expected_revision + 1
        conn.execute(
            "INSERT INTO v2_demo_operations VALUES (?, ?, ?, ?, ?, ?, ?)",
            (run_id, phase, op_id, fingerprint, status, result_revision,
             next_contract_revision),
        )
    return {"status": status, "revision": result_revision,
            "contract_revision": next_contract_revision}
