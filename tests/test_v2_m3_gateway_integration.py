#!/usr/bin/env python3
"""M3: M1 real HITL → C++ scoped MySQL Run → dynamic Agent → Draft.

Intentionally separate ephemeral CI database from the M2 integration suite.
Only synthetic P001 fixture and fake users; never patient healthcare data.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import uuid

from test_v2_m2_gateway_integration import (
    ACCESS_SECRET, CAPABILITY_SECRET, DOC, OTHER_DOC, PAT, OTHER_PAT,
    free_port, jwt_token, mysql, seed_fake_users, wait_healthy,
    expect_gateway_error,
)
from v2_m2.gateway_adapter import GatewayArtifactRepository, GatewayDoctorClient, GatewayTransport
from v2_m2.gateway_workflow import GatewayContractWorkflow
from v2_m3.policy import M3FixtureTools
from v2_m3.service import M3GatewayWorkflow


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_v2_m3_gateway_integration.py executable")
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
        "TREESEM_AGENT_SERVICE_SECRET": "m3-test-agent-service-secret-at-least-32-bytes",
        "TREESEM_KNOWLEDGE_JWT_SECRET": "m3-test-knowledge-secret-at-least-32-bytes",
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
            proc = subprocess.Popen(
                [str(Path(sys.argv[1]).resolve()), str(port)],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            try:
                wait_healthy(port, proc)
                transport = GatewayTransport(f"http://127.0.0.1:{port}")
                doctor = GatewayDoctorClient(transport, jwt_token(DOC, "doctor"), PAT)
                other = GatewayDoctorClient(
                    transport, jwt_token(OTHER_DOC, "doctor"), OTHER_PAT
                )
                hitl = GatewayContractWorkflow(tmp, doctor)
                started = hitl.start(
                    "m3_" + uuid.uuid4().hex,
                    "P001 合成既往病史、来源差异及模型资格调查",
                    "医生先确认目标；Agent 可依据来源调查动态补充 Issue",
                )
                run = started["run_id"]
                assert started["next"] == ["contract_gate"], started
                expect_gateway_error(lambda: doctor.issue_capability(run), 409)
                assert hitl.resume_confirm(run, 0, 1)["business"] == "contract_confirmed"
                scoped = doctor.issue_capability(run)
                repository = GatewayArtifactRepository(transport, scoped)
                # Actual LangGraph execution uses the same Run ID and real MySQL.
                workflow = M3GatewayWorkflow(tmp, repository)
                out = workflow.execute(run, 1)
                assert (out["phase"], out["next"]) == ("draft_ready", []), out
                assert [issue["issue_id"] for issue in out["agenda"]] == [
                    "I-01", "I-02", "I-03", "I-04",
                ]
                assert out["agenda"][2]["status"] == "blocked"
                assert "agenda:add:I-04" in out["trace"]
                draft = out["report"]
                assert draft["template_only"] and draft["not_signed"]
                assert draft["source_as_of"] == "2026-10-12"
                assert draft["source_validity"] == "historical_T1_not_current"
                assert draft["model_score"] is None
                assert len(draft["sections"]) == 6
                assert doctor.read_report(run, draft["report_id"] + ":v1") == draft
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_artifacts "
                    f"WHERE run_id='{run}'"
                ) == "8", "3 Evidence + 4 IssueResult + 1 Draft"
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_operations "
                    f"WHERE run_id='{run}'"
                ) == "8"
                old_i01 = repository.get(
                    run, "issue", "I-01:r1"
                )
                assert old_i01["result_revision"] == 1
                conflict = repository.get(run, "issue", "I-04:r1")
                assert conflict["status"] == "closed_unresolved"
                assert set(conflict["evidence_ids"]) == {"E-002", "E-003"}

                # New agent worker with old Checkpointer: no duplicate effects.
                restored = M3GatewayWorkflow(tmp, repository).execute(run, 1)
                assert restored["report"] == draft
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_operations "
                    f"WHERE run_id='{run}'"
                ) == "8"

                # This simulates the **MySQL COMMIT -> no Graph Checkpoint**
                # window on a different Run: persisted E-001 must replay.
                started2 = hitl.start(
                    "m3_" + uuid.uuid4().hex, "不同报告相同合成患者",
                    "需经过独立医生确认并使用不同 Run capability",
                )
                run2 = started2["run_id"]
                hitl.resume_confirm(run2, 0, 1)
                cap2 = doctor.issue_capability(run2)
                repo2 = GatewayArtifactRepository(transport, cap2)
                from v2_m3.repository import GatewayM3Repository
                store2 = GatewayM3Repository(repo2)
                record = M3FixtureTools().records["E-001"]
                key = f"{record['logical_id']}:v{record['version']}"
                staged = {"classification": "SIM", "run_id": run2,
                          "evidence_id": "E-001", "record": record}
                store2.save(run2, "evidence", key, staged)
                second = M3GatewayWorkflow(tmp, repo2).execute(run2, 1)
                assert second["phase"] == "draft_ready"
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_artifacts "
                    f"WHERE run_id='{run2}'"
                ) == "8"
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_operations "
                    f"WHERE run_id='{run2}'"
                ) == "8"
                expect_gateway_error(lambda: repository.status(run2), 403)
                expect_gateway_error(lambda: repo2.status(run), 403)
                expect_gateway_error(lambda: other.status(run), 404)
                # Old closed IssueResult cannot be rewritten.
                expect_gateway_error(
                    lambda: store2.save(run2, "issue", "I-01:r1",
                                        {"classification": "SIM", "run_id": run2,
                                         "issue_id": "I-01", "no_diagnosis": True,
                                         "status": "confirmed_pph"}), 409
                )
                print("M3 actual C++/MySQL + scoped single-Agent dynamic Agenda "
                      "integration passed; I-04 appended and recovery idempotent")
            finally:
                if proc.poll() is None:
                    proc.terminate()
                    try:
                        proc.wait(timeout=6)
                    except subprocess.TimeoutExpired:
                        proc.kill()
                        proc.wait(timeout=3)


if __name__ == "__main__":
    main()
