"""M9 Run-capability adapter for isolated unverified dynamic artifacts.

Never treats model-suggested M8 IssueResult as legacy M3 verified Issue.
"""
from __future__ import annotations
from v2_m1.run_store import WorkflowConflict
from v2_m2.gateway_adapter import GatewayArtifactRepository, checked_id
from v2_m3.repository import operation_id

KINDS = {"agenda_seed": "m8_seed", "evidence": "m8_evidence",
         "issue": "m8_issue"}

class GatewayM8Repository:
    def __init__(self, gateway: GatewayArtifactRepository):
        self.gateway = gateway

    def save(self, run_id: str, kind: str, key: str, payload: dict):
        if kind not in KINDS:
            raise WorkflowConflict("M8 cannot write legacy verified artifacts")
        if (not isinstance(payload, dict) or
                payload.get("classification") != "SIM" or
                payload.get("run_id") != run_id):
            raise WorkflowConflict("M8 synthetic Run identity required")
        return self.gateway.transport.request(
            "POST", self.gateway._prefix(run_id) + "/artifacts",
            self.gateway.capability,
            {"operation_id": operation_id(run_id, KINDS[kind], key),
             "kind": KINDS[kind],
             "artifact_key": checked_id(key), "payload": payload},
            run_id=run_id)

    def get(self, run_id: str, kind: str, key: str):
        if kind not in KINDS:
            raise WorkflowConflict("M8 cannot read legacy verified artifacts")
        value = self.gateway.transport.request(
            "GET", self.gateway._prefix(run_id) +
            f"/artifacts/{KINDS[kind]}/{checked_id(key)}",
            self.gateway.capability, run_id=run_id)
        if (not isinstance(value, dict) or
                value.get("classification") != "SIM" or
                value.get("run_id") != run_id):
            raise WorkflowConflict("M8 artifact belongs to another Run")
        return value
