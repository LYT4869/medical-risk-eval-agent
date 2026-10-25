"""Local demonstration business store. NOT the production C++/MySQL API."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


class WorkflowConflict(ValueError):
    """Stale revision, wrong actor, or conflicting idempotency payload."""


def operation_id(run_id: str, phase: str, expected_revision: int) -> str:
    """One stable logical operation ID, minted before the side-effecting node."""
    body = f"{run_id}|{phase}|{expected_revision}".encode()
    return "op_" + hashlib.sha256(body).hexdigest()[:32]


class DemoRunStore:
    def __init__(self, path: str | Path):
        self.path = str(path)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        return conn

    def setup(self) -> None:
        with self._connect() as conn:
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS v2_demo_runs (
              run_id TEXT PRIMARY KEY, actor_id TEXT NOT NULL,
              subject_id TEXT NOT NULL, contract_revision INTEGER NOT NULL,
              revision INTEGER NOT NULL DEFAULT 0, status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS v2_demo_operations (
              run_id TEXT NOT NULL, phase TEXT NOT NULL, operation_id TEXT NOT NULL,
              request_sha256 TEXT NOT NULL, result_status TEXT NOT NULL,
              result_revision INTEGER NOT NULL,
              PRIMARY KEY (run_id, phase, operation_id),
              FOREIGN KEY (run_id) REFERENCES v2_demo_runs(run_id)
            );
            """)

    def create(self, run_id: str, actor_id: str, subject_id: str) -> dict:
        # P001 and DOC-001 are synthetic fixture identities, not real credentials.
        if (actor_id, subject_id) != ("DOC-001", "P001"):
            raise WorkflowConflict("synthetic demo identity is not allowed")
        with self._connect() as conn:
            conn.execute("INSERT INTO v2_demo_runs VALUES (?, ?, ?, 1, 0, ?)",
                         (run_id, actor_id, subject_id, "awaiting_confirmation"))
        return self.get(run_id, actor_id)

    def get(self, run_id: str, actor_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM v2_demo_runs WHERE run_id=?", (run_id,)).fetchone()
        if row is None or row["actor_id"] != actor_id or actor_id != "DOC-001":
            raise WorkflowConflict("unknown run or unauthorized demo actor")
        return dict(row)

    def operation_count(self, run_id: str) -> int:
        with self._connect() as conn:
            return conn.execute("SELECT COUNT(*) FROM v2_demo_operations WHERE run_id=?",
                                (run_id,)).fetchone()[0]

    def apply(self, *, run_id: str, actor_id: str, phase: str,
              decision: str, op_id: str, expected_revision: int,
              contract_revision: int) -> dict:
        from .transitions import apply_transition
        return apply_transition(self, run_id=run_id, actor_id=actor_id,
                                phase=phase, decision=decision, op_id=op_id,
                                expected_revision=expected_revision,
                                contract_revision=contract_revision)
