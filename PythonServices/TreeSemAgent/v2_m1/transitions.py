"""Atomic, idempotent demo status transitions independent of graph checkpoints."""
from __future__ import annotations

import hashlib
import json

from .run_store import DemoRunStore, WorkflowConflict, operation_id


_TRANSITIONS = {
    ("contract", "confirm"): ("awaiting_confirmation", "awaiting_review"),
    ("contract", "cancel"): ("awaiting_confirmation", "cancelled"),
    ("contract", "revise"): ("awaiting_confirmation", "needs_revision"),
    ("review", "acknowledge"): ("awaiting_review", "completed_demo"),
    ("review", "cancel"): ("awaiting_review", "cancelled"),
}


def apply_transition(self: DemoRunStore, *, run_id: str, actor_id: str, phase: str,
                     decision: str, op_id: str, expected_revision: int,
                     contract_revision: int) -> dict:
    if op_id != operation_id(run_id, phase, expected_revision):
        raise WorkflowConflict("operation ID not tied to the expected run and revision")
    if (phase, decision) not in _TRANSITIONS:
        raise WorkflowConflict("invalid phase or decision")
    payload = json.dumps([run_id, actor_id, phase, decision, expected_revision,
                          contract_revision], separators=(",", ":"))
    fingerprint = hashlib.sha256(payload.encode()).hexdigest()
    old_status, next_status = _TRANSITIONS[phase, decision]
    with self._connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT * FROM v2_demo_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None or row["actor_id"] != actor_id or actor_id != "DOC-001":
            raise WorkflowConflict("unauthorized synthetic actor")
        if row["contract_revision"] != contract_revision:
            raise WorkflowConflict("stale contract revision")
        prior = conn.execute(
            "SELECT * FROM v2_demo_operations WHERE run_id=? AND phase=? AND operation_id=?",
            (run_id, phase, op_id)).fetchone()
        if prior:
            if prior["request_sha256"] != fingerprint:
                raise WorkflowConflict("operation id reused with a different payload")
            return {"status": prior["result_status"], "revision": prior["result_revision"]}
        if row["status"] != old_status or row["revision"] != expected_revision:
            raise WorkflowConflict("stale run revision or illegal transition")
        updated = conn.execute(
            "UPDATE v2_demo_runs SET status=?, revision=revision+1 "
            "WHERE run_id=? AND status=? AND revision=?",
            (next_status, run_id, old_status, expected_revision)).rowcount
        if updated != 1:
            raise WorkflowConflict("concurrent run transition")
        result_revision = expected_revision + 1
        conn.execute(
            "INSERT INTO v2_demo_operations VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, phase, op_id, fingerprint, next_status, result_revision))
    return {"status": next_status, "revision": result_revision}
