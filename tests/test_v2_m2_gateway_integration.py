#!/usr/bin/env python3
"""Actual C++ HTTP + MySQL + M1 interrupt + M2 LangGraph integration.

Test fixtures are SQL-created fake identities; all artifact data is SIM P001.
Never run against a real patient database. No production credentials are needed.
"""
from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import time
import uuid

from v2_m1.run_store import WorkflowConflict
from v2_m2.gateway_adapter import (
    GatewayArtifactRepository, GatewayDoctorClient, GatewayError,
    GatewayTransport,
)
from v2_m2.gateway_workflow import GatewayContractWorkflow, GatewayM2Workflow


ACCESS_SECRET = "m2-test-access-secret-for-JWT-at-least-32-bytes"
CAPABILITY_SECRET = "m2-test-capability-secret-at-least-32-bytes"

DOC = "m2_doc"
OTHER_DOC = "m2_doc_b"
PAT = "m2_pat"
OTHER_PAT = "m2_pat_b"
ADMIN = "m2_admin"
ASSIGNMENT = "m2_asn"
OTHER_ASSIGNMENT = "m2_asn_b"


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def jwt_token(subject: str, role: str, *, secret: str = ACCESS_SECRET,
              audience: str = "treesem-public", expires_in: int = 600) -> str:
    def enc(value):
        return base64.urlsafe_b64encode(json.dumps(
            value, separators=(",", ":"), sort_keys=True
        ).encode()).rstrip(b"=").decode()
    issued = int(time.time())
    header = enc({"alg": "HS256", "typ": "JWT"})
    payload = enc({
        "iss": "treesem-backend", "aud": audience, "sub": subject,
        "role": role, "ver": 0, "iat": issued, "exp": issued + expires_in,
    })
    original = header + "." + payload
    signature = base64.urlsafe_b64encode(
        hmac.new(secret.encode(), original.encode(), hashlib.sha256).digest()
    ).rstrip(b"=").decode()
    return original + "." + signature


def mysql(sql: str) -> str:
    cmd = [
        "mysql", "--protocol=TCP", "--host=127.0.0.1",
        "--port=" + os.environ.get("TREESEM_DB_PORT", "3306"),
        "--user=" + os.environ.get("TREESEM_DB_USER", "treesem_app"),
        "--database=" + os.environ.get("TREESEM_DB_NAME", "treesem_test"),
        "--batch", "--skip-column-names", "--execute", sql,
    ]
    proc = subprocess.run(
        cmd, text=True, capture_output=True,
        env={**os.environ, "MYSQL_PWD": os.environ["TREESEM_DB_PASSWORD"]},
        check=True,
    )
    return proc.stdout.strip()


def seed_fake_users():
    users = [
        (ADMIN, "admin"), (DOC, "doctor"), (OTHER_DOC, "doctor"),
        (PAT, "patient"), (OTHER_PAT, "patient"),
    ]
    for user, role in users:
        mysql(
            "INSERT INTO treesem_users "
            "(user_id,email_normalized,password_phc,display_name,role,"
            "status,created_at,updated_at) VALUES "
            f"('{user}','{user}@sim.invalid','invalid-login-hash',"
            f"'{user}','{role}','active',UTC_TIMESTAMP(6),UTC_TIMESTAMP(6))"
        )
    for ident, doctor, patient in (
        (ASSIGNMENT, DOC, PAT), (OTHER_ASSIGNMENT, OTHER_DOC, OTHER_PAT)
    ):
        mysql(
            "INSERT INTO treesem_doctor_patient_assignments "
            "(assignment_id,doctor_user_id,patient_user_id,status,"
            "created_by_admin_id,created_at) VALUES "
            f"('{ident}','{doctor}','{patient}','active','{ADMIN}',UTC_TIMESTAMP(6))"
        )


def wait_healthy(port: int, proc: subprocess.Popen):
    import urllib.request
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            raise AssertionError("C++ V2 server exited before ready")
        try:
            with urllib.request.urlopen(
                f"http://127.0.0.1:{port}/health", timeout=0.3
            ) as response:
                if response.status == 200:
                    return
        except OSError:
            pass
        time.sleep(0.07)
    raise AssertionError("C++ V2 server did not start")


def expect_gateway_error(call, *statuses):
    try:
        call()
    except GatewayError as e:
        assert e.status in statuses, (e.status, statuses)
    else:
        raise AssertionError("expected gateway to reject unsafe request")


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_v2_m2_gateway_integration.py /path/to/treesem_server")
    server = str(Path(sys.argv[1]).resolve())
    seed_fake_users()
    port = free_port()
    env = {
        **os.environ,
        "TREESEM_STORAGE_BACKEND": "mysql",
        "TREESEM_MODEL_BACKEND": "remote",
        "TREESEM_AGENT_ENABLED": "false",
        "TREESEM_AUTH_MODE": "required",
        "TREESEM_V2_SIM_ARTIFACT_API": "1",
        "TREESEM_ACCESS_JWT_SECRET": ACCESS_SECRET,
        "TREESEM_CAPABILITY_JWT_SECRET": CAPABILITY_SECRET,
        "TREESEM_AGENT_SERVICE_SECRET": "m2-test-agent-service-secret-at-least-32-bytes",
        "TREESEM_KNOWLEDGE_JWT_SECRET": "m2-test-knowledge-secret-at-least-32-bytes",
        "TREESEM_CAPABILITY_TOKEN_TTL_SECONDS": "120",
        "TREESEM_DB_POOL_SIZE": "3",
        "TREESEM_DB_HOST": "127.0.0.1",
        "TREESEM_DB_PORT": os.environ.get("TREESEM_DB_PORT", "3306"),
        "TREESEM_DB_NAME": os.environ.get("TREESEM_DB_NAME", "treesem_test"),
        "TREESEM_DB_USER": os.environ.get("TREESEM_DB_USER", "treesem_app"),
        "TREESEM_DB_PASSWORD": os.environ["TREESEM_DB_PASSWORD"],
    }
    with tempfile.TemporaryDirectory() as tmp:
        with open(Path(tmp) / "backend.log", "w") as log:
            def start():
                p = subprocess.Popen([server, str(port)], env=env,
                                     stdout=log, stderr=subprocess.STDOUT)
                wait_healthy(port, p)
                return p
            process = start()
            try:
                transport = GatewayTransport(f"http://127.0.0.1:{port}")
                doctor = GatewayDoctorClient(
                    transport, jwt_token(DOC, "doctor"), PAT
                )
                other = GatewayDoctorClient(
                    transport, jwt_token(OTHER_DOC, "doctor"), OTHER_PAT
                )

                # Same server-allocated Run ID is passed into actual M1 interrupt
                # and then to M2; the M1 SQLite file is only a Graph checkpoint.
                m1 = GatewayContractWorkflow(tmp, doctor)
                created = m1.start(
                    "idem_" + uuid.uuid4().hex,
                    "核查 P001 合成分娩来源差异",
                    "确认目标后检查已有合成原始记录及转诊摘要",
                )
                run_id = created["run_id"]
                assert (created["business"], created["next"]) == (
                    "awaiting_confirmation", ["contract_gate"],
                ), created
                assert doctor.status(run_id)["revision"] == 0
                expect_gateway_error(lambda: doctor.issue_capability(run_id), 409)
                expect_gateway_error(lambda: other.status(run_id), 404)

                # A doctor Access JWT cannot be presented as an Agent capability.
                expect_gateway_error(
                    lambda: transport.request(
                        "GET", f"/internal/v2/sim-runs/{run_id}",
                        doctor.doctor_jwt, run_id=run_id
                    ), 403
                )
                # No public direct Artifact writer exists.
                expect_gateway_error(
                    lambda: transport.request(
                        "POST", doctor._prefix() +
                        f"/{run_id}/artifacts", doctor.doctor_jwt,
                        {"kind": "evidence"}
                    ), 404
                )

                approved = m1.resume_confirm(run_id, 0, 1)
                assert (approved["business"], approved["next"]) == (
                    "contract_confirmed", [],
                ), approved
                assert doctor.status(run_id)["contract_revision"] == 1

                token = doctor.issue_capability(run_id)
                artifacts = GatewayArtifactRepository(transport, token)
                assert artifacts.status(run_id)["run_id"] == run_id
                m2 = GatewayM2Workflow(tmp, artifacts)
                result = m2.execute(run_id)
                draft = result["report"]
                assert result["next"] == [] and draft["template_only"]
                assert draft["status"] == "restricted_draft"
                assert draft["source_snapshot"] == ["E-002:v1", "E-003:v1"]
                assert len(draft["sections"]) == 6
                assert draft["not_signed"] and draft["model_score"] is None
                assert "已确诊" not in json.dumps(draft, ensure_ascii=False)
                assert doctor.read_report(run_id, draft["report_id"] + ":v1") == draft

                # A new Python process/Graph instance recovers without duplicate writes.
                result2 = GatewayM2Workflow(tmp, artifacts).execute(run_id)
                assert result2["report"] == draft
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_artifacts "
                    f"WHERE run_id='{run_id}'"
                ) == "4"
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_operations "
                    f"WHERE run_id='{run_id}'"
                ) == "4"

                # Signed capability is scoped to only this Run, regardless of
                # matching actor or patient on the second Run.
                second = doctor.create(
                    "idem_" + uuid.uuid4().hex, "另一个 P001 合成报告",
                    "只用合成记录作单独调查",
                )
                second_id = second["run_id"]
                assert second_id != run_id
                expect_gateway_error(lambda: artifacts.status(second_id), 403)
                expect_gateway_error(
                    lambda: doctor.read_report(second_id,
                        draft["report_id"] + ":v1"), 404
                )
                # Token with a bad signature / wrong audience is rejected.
                forged = jwt_token(DOC, "doctor", secret=ACCESS_SECRET)
                expect_gateway_error(
                    lambda: transport.request(
                        "GET", f"/internal/v2/sim-runs/{run_id}",
                        forged, run_id=run_id,
                    ), 403
                )

                # Restart native server: MySQL stays authoritative; neither
                # report nor doctor/Run ownership is process memory.
                process.terminate()
                process.wait(timeout=6)
                process = start()
                assert doctor.read_report(run_id, draft["report_id"] + ":v1") == draft

                # Live assignment revocation invalidates *existing* signed
                # capability and public doctor read after the C++ restart.
                mysql(
                    "UPDATE treesem_doctor_patient_assignments SET "
                    "status='revoked',revoked_at=UTC_TIMESTAMP(6) "
                    f"WHERE assignment_id='{ASSIGNMENT}'"
                )
                expect_gateway_error(lambda: artifacts.status(run_id), 404, 403)
                expect_gateway_error(lambda: doctor.status(run_id), 404, 403)
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_artifacts "
                    f"WHERE run_id='{run_id}'"
                ) == "4"
                print("M2-B real C++ HTTP→MySQL + M1 interrupt + M2 Graph "
                      "scoped-capability integration passed")
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=6)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=3)


if __name__ == "__main__":
    main()
