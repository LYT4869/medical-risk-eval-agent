#!/usr/bin/env python3
"""M4 P001 correction: actual JWT C++ gateway + MySQL 8 + LangGraph.

Only the server's built-in *synthetic* EV-001 may mark an LIS record corrected.
Patient records used here are synthetic; no real clinical system is queried.
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
from v2_m1.run_store import WorkflowConflict
from v2_m2.gateway_adapter import GatewayArtifactRepository, GatewayDoctorClient, GatewayTransport
from v2_m2.gateway_workflow import GatewayContractWorkflow
from v2_m3.service import M3GatewayWorkflow
from v2_m4.service import M4GatewayWorkflow


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_v2_m4_gateway_integration.py C++_server")
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
        "TREESEM_AGENT_SERVICE_SECRET": "m4-test-agent-service-secret-at-least-32-bytes",
        "TREESEM_KNOWLEDGE_JWT_SECRET": "m4-test-knowledge-secret-at-least-32-bytes",
        "TREESEM_CAPABILITY_TOKEN_TTL_SECONDS": "120",
        "TREESEM_DB_POOL_SIZE": "3",
        "TREESEM_DB_HOST": "127.0.0.1",
        "TREESEM_DB_PORT": os.environ.get("TREESEM_DB_PORT", "3306"),
        "TREESEM_DB_NAME": os.environ.get("TREESEM_DB_NAME", "treesem_test"),
        "TREESEM_DB_USER": os.environ.get("TREESEM_DB_USER", "treesem_app"),
        "TREESEM_DB_PASSWORD": os.environ["TREESEM_DB_PASSWORD"],
    }
    with tempfile.TemporaryDirectory() as tmp:
        with open(Path(tmp) / "server.log", "w") as log:
            server = subprocess.Popen(
                [str(Path(sys.argv[1]).resolve()), str(port)],
                stdout=log, stderr=subprocess.STDOUT, env=env,
            )
            try:
                wait_healthy(port, server)
                transport = GatewayTransport(f"http://127.0.0.1:{port}")
                doctor = GatewayDoctorClient(
                    transport, jwt_token(DOC, "doctor"), PAT)
                other = GatewayDoctorClient(
                    transport, jwt_token(OTHER_DOC, "doctor"), OTHER_PAT)
                m1 = GatewayContractWorkflow(tmp, doctor)
                started = m1.start(
                    "m4_" + uuid.uuid4().hex,
                    "P001 SIM 正式 LIS 来源版本更正追踪",
                    "先调查既往分娩异源差异，再判断 LIS 同源版本更新",
                )
                run = started["run_id"]
                assert started["next"] == ["contract_gate"]
                expect_gateway_error(lambda: doctor.stage_m4_history(run), 409)
                assert m1.resume_confirm(run, 0, 1)["business"] == "contract_confirmed"

                # Without an official event old source does not become stale
                # just because a separate referral source disagrees.
                expect_gateway_error(
                    lambda: doctor.apply_m4_simulated_correction(run), 409)
                cap = doctor.issue_capability(run)
                M3GatewayWorkflow(
                    tmp, GatewayArtifactRepository(transport, cap)
                ).execute(run, 1)
                before = doctor.read_report(run, "report_" + run + ":v1")
                assert before["source_as_of"] == "2026-10-12"
                assert before["source_snapshot"] == [
                    "E-001:v1", "E-002:v1", "E-003:v1"
                ]
                assert doctor.validate_m4_current(
                    run, ["E-002","E-003"])["admissible"]
                assert not doctor.validate_m4_current(
                    run, ["E-004"])["admissible"]

                # The Agent's scoped artifact capability cannot impersonate
                # a doctor to trigger the official source replay.
                expect_gateway_error(
                    lambda: transport.request(
                        "POST", doctor._prefix() + f"/{run}/m4/correction",
                        cap, {}, run_id=run,
                    ), 401, 403
                )
                expect_gateway_error(
                    lambda: transport.request(
                        "POST", doctor._prefix() + f"/{run}/m4/correction",
                        doctor.doctor_jwt, {"hb_g_l": 900},
                    ), 400, 422
                )
                expect_gateway_error(
                    lambda: other.m4_state(run), 404, 403
                )

                history = doctor.stage_m4_history(run)
                assert history == doctor.stage_m4_history(run)
                assert history["statement_id"] == "S-04"
                old = doctor.m4_state(run)
                assert old["source_versions"][0]["hb_g_l"] == 110
                assert old["source_versions"][0]["validity"] == "current"
                assert doctor.validate_m4_current(run, ["E-004"])["admissible"]
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_m4_statement_snapshots "
                    f"WHERE run_id='{run}'"
                ) == "1"

                # Simulate official LIS correction COMMIT *before* M4 Graph
                # Checkpointer advances. Retried nodes must be idempotent.
                event = doctor.apply_m4_simulated_correction(run)
                assert event["event_id"] == "EV-001"
                assert event == doctor.apply_m4_simulated_correction(run)
                workflow = M4GatewayWorkflow(tmp, doctor)
                outcome = workflow.execute(
                    run, explicit_simulated_lis_event=True
                )
                assert outcome["phase"] == "impact_recorded"
                assert outcome["next"] == []
                assert outcome["current_source"] == "E-005"
                assert outcome["stale_statement_ids"] == ["S-04"]
                assert outcome["new_issue"] == "I-05"

                overlay = outcome["projection"]
                versions = {x["evidence_id"]: x for x in overlay["source_versions"]}
                statements = {x["statement_id"]: x for x in overlay["statements"]}
                assert versions["E-004"]["hb_g_l"] == 110
                assert versions["E-004"]["validity"] == "stale"
                assert versions["E-005"]["hb_g_l"] == 130
                assert versions["E-005"]["validity"] == "current"
                assert statements["S-04"]["hb_g_l"] == 110
                assert statements["S-04"]["validity"] == "stale"
                assert overlay["followup_issues"] == [{
                    "issue_id": "I-05", "status": "needs_review",
                    "old_evidence_id": "E-004", "new_evidence_id": "E-005",
                }]

                bad_admission = doctor.validate_m4_current(run, ["E-004"])
                assert not bad_admission["admissible"]
                assert bad_admission["stale_refs"] == ["E-004"]
                assert bad_admission["affected_statement_ids"] == ["S-04"]
                assert doctor.validate_m4_current(run, ["E-005"])["admissible"]
                assert doctor.validate_m4_current(
                    run, ["E-002", "E-003", "E-005"])["admissible"]
                assert doctor.validate_m4_current(
                    run, ["E-002", "E-003"])["admissible"]
                assert doctor.validate_m4_current(
                    run, ["E-004", "E-005"])["admissible"] is False
                assert bad_admission["model_capability"] == "blocked_unverified_prenatal"
                expect_gateway_error(
                    lambda: doctor.validate_m4_current(run, ["E-999"]), 400, 422
                )
                expect_gateway_error(
                    lambda: doctor.validate_m4_current(run, ["E-005","E-005"]), 400, 422
                )
                # Existing M3 report only cites E001..E003 and MUST NOT be
                # falsely marked as stale; it remains historical T1.
                assert doctor.read_report(run, "report_" + run + ":v1") == before

                # Entire overlay is unchanged across new service instance/retry.
                second = M4GatewayWorkflow(tmp, doctor).execute(
                    run, explicit_simulated_lis_event=True
                )
                assert second["projection"] == overlay
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_m4_correction_events "
                    f"WHERE run_id='{run}'"
                ) == "2"
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_m4_source_versions "
                    f"WHERE run_id='{run}'"
                ) == "2"
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_m4_followups "
                    f"WHERE run_id='{run}'"
                ) == "1"
                # No mutation of existing M3's source conflict or issue snapshots.
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_artifacts "
                    f"WHERE run_id='{run}'"
                ) == "8"

                # Different Run cannot see correction and must not infer that
                # LIS correction is patient-global without a trusted matching
                # record identity or a new scoped source query.
                second_run = m1.start(
                    "m4_" + uuid.uuid4().hex, "P001 另一独立调查 Run",
                    "医生先确认后查询自身来源"
                )["run_id"]
                m1.resume_confirm(second_run, 0, 1)
                assert doctor.m4_state(second_run)["events"] == []
                assert not doctor.validate_m4_current(
                    second_run, ["E-005"])["admissible"]
                assert doctor.stage_m4_history(second_run)["source_id"] == "E-004"
                assert doctor.m4_state(second_run)["source_versions"][0]["validity"] == "current"
                expect_gateway_error(lambda: other.m4_state(run), 404, 403)

                # A revoked doctor assignment prevents fresh reads/events.
                mysql(
                    "UPDATE treesem_doctor_patient_assignments "
                    "SET status='revoked',revoked_at=UTC_TIMESTAMP(6) "
                    "WHERE assignment_id='m2_asn'"
                )
                expect_gateway_error(lambda: doctor.m4_state(run), 403, 404)
                expect_gateway_error(
                    lambda: doctor.apply_m4_simulated_correction(second_run), 403, 404
                )
                print("M4 C++/MySQL source versions + S-04 staleness + I-05 "
                      "replay/current-source gate integration passed")
            finally:
                if server.poll() is None:
                    server.terminate()
                    try:
                        server.wait(timeout=6)
                    except subprocess.TimeoutExpired:
                        server.kill()
                        server.wait(timeout=3)


if __name__ == "__main__":
    main()
