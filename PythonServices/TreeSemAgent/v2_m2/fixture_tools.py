"""Fixed P001-only read adapter with explicit tool policy; no clinical API."""
from __future__ import annotations

from datetime import date

from evaluation.p001_contract import load
from v2_m1.run_store import WorkflowConflict


class SyntheticP001Tools:
    ALLOWED = frozenset(("get_patient_timeline", "read_patient_record"))

    def __init__(self):
        self.data = load()
        self.records = {r["id"]: r for r in self.data["records"]}

    def call(self, tool: str, *, actor_id: str, patient_id: str,
             as_of: str, record_id: str | None = None):
        if (actor_id, patient_id) != ("DOC-001", "P001"):
            raise WorkflowConflict("synthetic scope rejected")
        if tool not in self.ALLOWED:
            raise WorkflowConflict("tool not authorized for M2 fixture")
        cutoff = date.fromisoformat(as_of)
        if tool == "get_patient_timeline":
            # Deliberately bounded: one PPH conflict Issue for the Level-0 slice.
            return ["E-002", "E-003"]
        if record_id not in ("E-002", "E-003"):
            raise WorkflowConflict("record outside M2 issue allowlist")
        record = self.records[record_id]
        if date.fromisoformat(record["available_on"]) > cutoff:
            raise WorkflowConflict("future source must not be visible")
        return dict(record)
