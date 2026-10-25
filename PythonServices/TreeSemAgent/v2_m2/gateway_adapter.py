"""Scoped V2 SIM HTTP adapter. MySQL-backed C++ Run is authoritative.

Credentials are never included in LangGraph State or persisted in checkpoints.
This adapter intentionally has no ability to create or confirm medical facts.
"""
from __future__ import annotations

import json
import re
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from v2_m1.run_store import WorkflowConflict
from .artifact_store import op_id


_ID = re.compile(r"^[A-Za-z0-9_.:-]{1,96}$")


def checked_id(raw: str) -> str:
    if not isinstance(raw, str) or not _ID.fullmatch(raw):
        raise WorkflowConflict("invalid V2 SIM gateway identifier")
    return raw


class GatewayError(WorkflowConflict):
    def __init__(self, status: int):
        super().__init__(f"V2 gateway rejected request (HTTP {status})")
        self.status = status


class GatewayTransport:
    """HTTP only on loopback; HTTPS required for any other target."""

    def __init__(self, base_url: str, timeout: float = 5.0):
        parts = urlsplit(base_url)
        if (parts.scheme not in ("https", "http") or not parts.hostname
                or parts.username or parts.password or parts.query or parts.fragment
                or (parts.scheme == "http"
                    and parts.hostname not in ("127.0.0.1", "::1", "localhost"))):
            raise ValueError("V2 gateway requires HTTPS or local-loopback HTTP")
        if timeout <= 0 or timeout > 30:
            raise ValueError("invalid V2 gateway timeout")
        self.base = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, method: str, path: str, token: str, payload=None,
                *, run_id: str | None = None):
        if not path.startswith("/") or not token or "\r" in token or "\n" in token:
            raise WorkflowConflict("invalid V2 gateway request")
        body = None if payload is None else json.dumps(
            payload, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        headers = {"Authorization": f"Bearer {token}", "Accept": "application/json"}
        if body is not None:
            headers["Content-Type"] = "application/json"
        if run_id is not None:
            headers["X-TreeSem-Session-Id"] = "v2:" + checked_id(run_id)
        request = Request(
            self.base + path, data=body, headers=headers, method=method,
        )
        try:
            with urlopen(request, timeout=self.timeout) as response:
                data = response.read(262144)
                if len(data) == 262144:
                    raise WorkflowConflict("gateway response exceeds configured limit")
                decoded = json.loads(data)
                if not isinstance(decoded, dict):
                    raise WorkflowConflict("unexpected gateway response")
                return decoded
        except HTTPError as error:
            # Do not embed response body: it may contain diagnostic/clinical data.
            error.read(4096)
            raise GatewayError(error.code) from None
        except (URLError, TimeoutError, OSError) as error:
            raise WorkflowConflict("V2 gateway temporarily unavailable") from error
        except (ValueError, json.JSONDecodeError) as error:
            raise WorkflowConflict("V2 gateway returned malformed JSON") from error


class GatewayDoctorClient:
    """Only the doctor UI/server orchestration uses an access JWT."""

    def __init__(self, transport: GatewayTransport, doctor_jwt: str, patient_id: str):
        self.transport = transport
        self.doctor_jwt = doctor_jwt
        self.patient_id = checked_id(patient_id)

    def _prefix(self) -> str:
        return f"/api/v2/doctor/patients/{self.patient_id}/sim-runs"

    def create(self, idempotency_key: str, summary: str, overview: str,
               *, clarification_question: str | None = None):
        body = {
            "idempotency_key": checked_id(idempotency_key),
            "fixture_id": "P001",
            "summary": summary, "overview": overview,
        }
        if clarification_question is not None:
            if not isinstance(clarification_question, str) or not clarification_question.strip():
                raise WorkflowConflict("clarification must be nonempty")
            body["clarification_question"] = clarification_question
        return self.transport.request(
            "POST", self._prefix(), self.doctor_jwt, body
        )

    def status(self, run_id: str):
        return self.transport.request(
            "GET", self._prefix() + f"/{checked_id(run_id)}", self.doctor_jwt,
        )

    def confirm(self, run_id: str, expected_revision: int,
                contract_revision: int = 1):
        return self.transport.request(
            "POST", self._prefix() + f"/{checked_id(run_id)}/confirm",
            self.doctor_jwt,
            {"expected_revision": expected_revision,
             "contract_revision": contract_revision},
        )

    def change_contract(
        self, run_id: str, decision: str, expected_revision: int,
        contract_revision: int, *, summary: str | None = None,
        overview: str | None = None, answer: str | None = None,
    ):
        if decision not in ("revise", "clarify", "cancel"):
            raise WorkflowConflict("unsupported native Contract edit")
        body = {"decision": decision, "expected_revision": expected_revision,
                "contract_revision": contract_revision}
        if summary is not None:
            body["summary"] = summary
        if overview is not None:
            body["overview"] = overview
        if answer is not None:
            body["answer"] = answer
        return self.transport.request(
            "POST", self._prefix() + f"/{checked_id(run_id)}/contract",
            self.doctor_jwt, body,
        )

    def issue_capability(self, run_id: str) -> str:
        result = self.transport.request(
            "POST", self._prefix() + f"/{checked_id(run_id)}/capability",
            self.doctor_jwt, {},
        )
        if result.get("run_id") != run_id or result.get("classification") != "SIM":
            raise WorkflowConflict("invalid gateway capability response")
        token = result.get("capability_token")
        if not isinstance(token, str) or not token:
            raise WorkflowConflict("gateway capability missing")
        return token

    def _m4(self, run_id: str, method: str, endpoint: str, payload=None):
        return self.transport.request(
            method, self._prefix() + f"/{checked_id(run_id)}/m4/{endpoint}",
            self.doctor_jwt, payload,
        )

    def stage_m4_history(self, run_id: str):
        # Test-only: doctor explicitly asks the server to stage fixed old LIS v1.
        return self._m4(run_id, "POST", "history", {})

    def apply_m4_simulated_correction(self, run_id: str):
        # A bundled server fixture EV-001; never accept arbitrary patient values.
        return self._m4(run_id, "POST", "correction", {})

    def m4_state(self, run_id: str):
        return self._m4(run_id, "GET", "state")

    def validate_m4_current(self, run_id: str, evidence_ids: list[str]):
        if not isinstance(evidence_ids, list) or not all(
            isinstance(ref, str) for ref in evidence_ids
        ):
            raise WorkflowConflict("invalid synthetic source references")
        return self._m4(run_id, "POST", "validate-current",
                        {"evidence_ids": evidence_ids})

    def read_report(self, run_id: str, artifact_key: str):
        return self.transport.request(
            "GET", self._prefix() + f"/{checked_id(run_id)}"
                   f"/artifacts/draft/{checked_id(artifact_key)}",
            self.doctor_jwt,
        )


class GatewayArtifactRepository:
    """Only internal Run/Tool capability, never a doctor access token."""

    def __init__(self, transport: GatewayTransport, capability: str):
        if not capability:
            raise ValueError("a scoped capability is required")
        self.transport = transport
        self.capability = capability

    @staticmethod
    def _prefix(run_id: str) -> str:
        return "/internal/v2/sim-runs/" + checked_id(run_id)

    def status(self, run_id: str) -> dict:
        return self.transport.request("GET", self._prefix(run_id),
                                      self.capability, run_id=run_id)

    def save(self, run_id: str, phase: str, artifact_key: str,
             value: dict, *, request: dict | None = None):
        # The M2 Graph uses only fixed SIM phases; no generic tool write.
        if phase not in ("source_original", "source_referral", "issue", "draft"):
            raise WorkflowConflict("unsupported Artifact phase")
        if not isinstance(value, dict) or value.get("classification") != "SIM":
            raise WorkflowConflict("synthetic payload label required")
        kind = "evidence" if phase.startswith("source_") else phase
        # HTTP business journal independently canonicalizes/validates payload.
        return self.transport.request(
            "POST", self._prefix(run_id) + "/artifacts", self.capability,
            {"operation_id": op_id(run_id, phase),
             "kind": kind, "artifact_key": checked_id(artifact_key),
             "payload": value},
            run_id=run_id,
        )

    def get(self, run_id: str, kind: str, artifact_key: str) -> dict:
        if kind not in ("evidence", "issue", "draft"):
            raise WorkflowConflict("invalid artifact kind")
        return self.transport.request(
            "GET", self._prefix(run_id) +
            f"/artifacts/{kind}/{checked_id(artifact_key)}",
            self.capability, run_id=run_id,
        )
