"""Synthetic-only M1 business store; NOT the authorized C++/MySQL API."""
from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


class WorkflowConflict(ValueError):
    """Invalid synthetic identity, stale revision, or conflicting operation."""


def operation_id(run_id: str, phase: str, expected_revision: int) -> str:
    """Stable before an external effect; retries at a revision use the same ID."""
    raw = f"{run_id}|{phase}|{expected_revision}".encode()
    return "op_" + hashlib.sha256(raw).hexdigest()[:32]


INITIAL_SUMMARY = (
    "核查 P001 既往分娩和近期检验的 PPH 相关资料，"
    "形成标注来源、不确定性和模型适用性限制的辅助草稿。"
)
INITIAL_OVERVIEW = (
    "拟先核对授权患者的既往记录与近期检查；出现独立来源差异时"
    "再增补核查问题，按模型资格决定是否调用模型。"
    "具体 Issue 和 Tool 顺序由确认后的 Agent 动态决定。"
)
OPTIONAL_QUESTION = "资料不足时，是否允许清楚标注局限的受限草稿？"


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
            existing = {row["name"] for row in
                        conn.execute("PRAGMA table_info(v2_demo_operations)")}
            if existing and "result_contract_revision" not in existing:
                raise WorkflowConflict("old M1 demo schema; use a fresh --data-dir")
            conn.executescript("""
            CREATE TABLE IF NOT EXISTS v2_demo_runs (
              run_id TEXT PRIMARY KEY,
              actor_id TEXT NOT NULL,
              subject_id TEXT NOT NULL,
              contract_revision INTEGER NOT NULL,
              revision INTEGER NOT NULL DEFAULT 0,
              status TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS v2_demo_contracts (
              run_id TEXT NOT NULL,
              contract_revision INTEGER NOT NULL,
              summary TEXT NOT NULL,
              investigation_overview TEXT NOT NULL,
              clarification_question TEXT,
              clarification_answer TEXT,
              PRIMARY KEY (run_id, contract_revision),
              FOREIGN KEY (run_id) REFERENCES v2_demo_runs(run_id)
            );
            CREATE TABLE IF NOT EXISTS v2_demo_operations (
              run_id TEXT NOT NULL,
              phase TEXT NOT NULL,
              operation_id TEXT NOT NULL,
              request_sha256 TEXT NOT NULL,
              result_status TEXT NOT NULL,
              result_revision INTEGER NOT NULL,
              result_contract_revision INTEGER NOT NULL,
              PRIMARY KEY (run_id, phase, operation_id),
              FOREIGN KEY (run_id) REFERENCES v2_demo_runs(run_id)
            );
            """)

    def create(self, run_id: str, actor_id: str, subject_id: str,
               *, ask_clarification: bool = False) -> dict:
        # Synthetic allowlist is not authentication.
        if (actor_id, subject_id) != ("DOC-001", "P001"):
            raise WorkflowConflict("synthetic demo identity is not allowed")
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO v2_demo_runs VALUES (?, ?, ?, 1, 0, ?)",
                (run_id, actor_id, subject_id, "awaiting_confirmation"),
            )
            conn.execute(
                "INSERT INTO v2_demo_contracts VALUES (?, 1, ?, ?, ?, NULL)",
                (run_id, INITIAL_SUMMARY, INITIAL_OVERVIEW,
                 OPTIONAL_QUESTION if ask_clarification else None),
            )
        return self.get(run_id, actor_id)

    def get(self, run_id: str, actor_id: str) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM v2_demo_runs WHERE run_id=?", (run_id,)
            ).fetchone()
        if row is None or row["actor_id"] != actor_id or actor_id != "DOC-001":
            raise WorkflowConflict("unknown run or unauthorized synthetic actor")
        return dict(row)

    def contract(self, run_id: str, actor_id: str) -> dict:
        record = self.get(run_id, actor_id)
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM v2_demo_contracts WHERE run_id=? "
                "AND contract_revision=?",
                (run_id, record["contract_revision"]),
            ).fetchone()
        if row is None:
            raise WorkflowConflict("missing versioned contract")
        return {
            "revision": row["contract_revision"],
            "summary": row["summary"],
            "investigation_overview": row["investigation_overview"],
            "clarification_question": row["clarification_question"],
            "clarification_answer": row["clarification_answer"],
            "synthetic_only": True,
        }

    def operation_count(self, run_id: str) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM v2_demo_operations WHERE run_id=?", (run_id,)
            ).fetchone()[0]

    def contract_count(self, run_id: str) -> int:
        with self._connect() as conn:
            return conn.execute(
                "SELECT COUNT(*) FROM v2_demo_contracts WHERE run_id=?", (run_id,)
            ).fetchone()[0]

    def apply(self, *, run_id: str, actor_id: str, phase: str,
              decision: str, op_id: str, expected_revision: int,
              contract_revision: int, summary: str | None = None,
              overview: str | None = None, answer: str | None = None) -> dict:
        from .transitions import apply_transition
        return apply_transition(
            self, run_id=run_id, actor_id=actor_id, phase=phase,
            decision=decision, op_id=op_id, expected_revision=expected_revision,
            contract_revision=contract_revision,
            summary=summary, overview=overview, answer=answer,
        )
