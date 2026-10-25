"""SQLite fake ArtifactRepository; MySQL/C++ trusted backend is separate M2 work."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from v2_m1.run_store import WorkflowConflict


def canonical(value) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def op_id(run_id: str, phase: str) -> str:
    return "op_" + hashlib.sha256(f"{run_id}|m2|{phase}".encode()).hexdigest()[:32]


class SyntheticArtifactStore:
    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._connect() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS m2_artifacts (
              run_id TEXT NOT NULL, kind TEXT NOT NULL, artifact_key TEXT NOT NULL,
              payload TEXT NOT NULL, payload_hash TEXT NOT NULL,
              PRIMARY KEY (run_id, kind, artifact_key)
            );
            CREATE TABLE IF NOT EXISTS m2_operations (
              run_id TEXT NOT NULL, phase TEXT NOT NULL,
              operation_id TEXT NOT NULL, request_hash TEXT NOT NULL,
              result TEXT NOT NULL, PRIMARY KEY(run_id, phase),
              UNIQUE(run_id, operation_id)
            );
            """)

    def _connect(self):
        conn = sqlite3.connect(self.path, timeout=5)
        conn.row_factory = sqlite3.Row
        return conn

    def save(self, run_id: str, phase: str, artifact_key: str,
             value: dict, *, request: dict | None = None) -> dict:
        if phase not in ("source_original", "source_referral", "issue", "draft"):
            raise WorkflowConflict("unsupported artifact phase")
        kind = "evidence" if phase.startswith("source_") else phase
        request = value if request is None else request
        digest = hashlib.sha256(canonical(request).encode()).hexdigest()
        serialized = canonical(value)
        payload_hash = hashlib.sha256(serialized.encode()).hexdigest()
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            prior = db.execute(
                "SELECT * FROM m2_operations WHERE run_id=? AND phase=?",
                (run_id, phase),
            ).fetchone()
            if prior is not None:
                if (prior["operation_id"] != op_id(run_id, phase)
                        or prior["request_hash"] != digest):
                    raise WorkflowConflict("artifact replay payload conflict")
                return json.loads(prior["result"])
            collision = db.execute(
                "SELECT payload_hash FROM m2_artifacts WHERE run_id=? AND kind=? "
                "AND artifact_key=?", (run_id, kind, artifact_key),
            ).fetchone()
            if collision:
                if collision["payload_hash"] != payload_hash:
                    raise WorkflowConflict("artifact key reused with changed content")
            else:
                db.execute(
                    "INSERT INTO m2_artifacts VALUES (?, ?, ?, ?, ?)",
                    (run_id, kind, artifact_key, serialized, payload_hash),
                )
            result = {"kind": kind, "key": artifact_key, "run_id": run_id}
            db.execute(
                "INSERT INTO m2_operations VALUES (?, ?, ?, ?, ?)",
                (run_id, phase, op_id(run_id, phase), digest, canonical(result)),
            )
        return result

    def get(self, run_id: str, kind: str, artifact_key: str) -> dict:
        with self._connect() as db:
            found = db.execute(
                "SELECT payload FROM m2_artifacts WHERE run_id=? AND kind=? "
                "AND artifact_key=?", (run_id, kind, artifact_key),
            ).fetchone()
        if found is None:
            raise WorkflowConflict("artifact missing or wrong Run")
        return json.loads(found["payload"])

    def count(self, run_id: str, kind: str | None = None) -> int:
        with self._connect() as db:
            if kind is not None:
                return db.execute("SELECT COUNT(*) FROM m2_artifacts WHERE run_id=? "
                                  "AND kind=?", (run_id, kind)).fetchone()[0]
            return db.execute("SELECT COUNT(*) FROM m2_artifacts WHERE run_id=?",
                              (run_id,)).fetchone()[0]

    def operation_count(self, run_id: str) -> int:
        with self._connect() as db:
            return db.execute(
                "SELECT COUNT(*) FROM m2_operations WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
