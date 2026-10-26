"""M3 ArtifactRepository adapters: local fake or *scoped C++/MySQL* gateway."""
from __future__ import annotations

import hashlib
import json
import sqlite3
from pathlib import Path

from v2_m1.run_store import WorkflowConflict
from v2_m2.artifact_store import canonical
from v2_m2.gateway_adapter import GatewayArtifactRepository, checked_id


def operation_id(run_id: str, kind: str, artifact_key: str) -> str:
    raw = f"{run_id}|m3|{kind}|{artifact_key}".encode()
    return "op_" + hashlib.sha256(raw).hexdigest()[:32]


class LocalM3Repository:
    """Independent deterministic SIM fake. Not a clinical business database."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        with self._db() as db:
            db.executescript("""
            CREATE TABLE IF NOT EXISTS m3_artifacts(
                run_id TEXT,kind TEXT,artifact_key TEXT,payload TEXT,payload_hash TEXT,
                PRIMARY KEY(run_id,kind,artifact_key));
            CREATE TABLE IF NOT EXISTS m3_operations(
                run_id TEXT,operation_id TEXT,request_hash TEXT,result_json TEXT,
                PRIMARY KEY(run_id,operation_id));
            """)

    def _db(self):
        db = sqlite3.connect(self.path, timeout=5)
        db.row_factory = sqlite3.Row
        return db

    def save(self, run_id: str, kind: str, key: str, payload: dict):
        if (kind not in ("evidence", "issue", "draft", "agenda_seed") or
                not isinstance(payload, dict) or
                payload.get("classification") != "SIM" or
                payload.get("run_id") != run_id):
            raise WorkflowConflict("invalid M3 SIM artifact")
        op = operation_id(run_id, kind, key)
        encoded = canonical(payload)
        digest = hashlib.sha256(encoded.encode()).hexdigest()
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute(
                "SELECT * FROM m3_operations WHERE run_id=? AND operation_id=?",
                (run_id, op)).fetchone()
            if old:
                if old["request_hash"] != digest:
                    raise WorkflowConflict("M3 operation reused with different payload")
                return json.loads(old["result_json"])
            prior = db.execute(
                "SELECT payload_hash FROM m3_artifacts "
                "WHERE run_id=? AND kind=? AND artifact_key=?",
                (run_id, kind, key),
            ).fetchone()
            if prior and prior["payload_hash"] != digest:
                raise WorkflowConflict("immutable M3 artifact changed")
            if not prior:
                db.execute("INSERT INTO m3_artifacts VALUES(?,?,?,?,?)",
                           (run_id, kind, key, encoded, digest))
            result = {"run_id": run_id, "kind": kind, "artifact_key": key}
            db.execute("INSERT INTO m3_operations VALUES(?,?,?,?)",
                       (run_id, op, digest, canonical(result)))
        return result

    def get(self, run_id: str, kind: str, key: str) -> dict:
        with self._db() as db:
            result = db.execute(
                "SELECT payload FROM m3_artifacts "
                "WHERE run_id=? AND kind=? AND artifact_key=?",
                (run_id, kind, key)).fetchone()
        if result is None:
            raise WorkflowConflict("M3 artifact missing")
        return json.loads(result["payload"])

    def count(self, run_id: str) -> int:
        with self._db() as db:
            return db.execute(
                "SELECT COUNT(*) FROM m3_artifacts WHERE run_id=?",
                (run_id,)).fetchone()[0]


class GatewayM3Repository:
    """Delegates to existing C++ JWT/Run scoped MySQL API, not a doctor JWT."""

    def __init__(self, gateway: GatewayArtifactRepository):
        self.gateway = gateway

    def save(self, run_id: str, kind: str, key: str, payload: dict):
        if (kind not in ("evidence", "issue", "draft") or
                payload.get("classification") != "SIM" or
                payload.get("run_id") != run_id):
            raise WorkflowConflict("M3 would write untrusted or cross-Run artifact")
        return self.gateway.transport.request(
            "POST", self.gateway._prefix(run_id) + "/artifacts",
            self.gateway.capability,
            {"operation_id": operation_id(run_id, kind, key),
             "kind": kind, "artifact_key": checked_id(key), "payload": payload},
            run_id=run_id,
        )

    def get(self, run_id: str, kind: str, key: str) -> dict:
        return self.gateway.get(run_id, kind, key)
