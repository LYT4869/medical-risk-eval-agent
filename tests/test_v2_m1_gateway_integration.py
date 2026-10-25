#!/usr/bin/env python3
"""Actual MySQL Task Contract vN, M1 interrupt loop, M3 compatibility and replay."""
from __future__ import annotations

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


def main():
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_v2_m1_gateway_integration.py C++_server")
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
        "TREESEM_AGENT_SERVICE_SECRET": "m1-reconfirm-agent-service-secret-at-least-32-bytes",
        "TREESEM_KNOWLEDGE_JWT_SECRET": "m1-reconfirm-knowledge-secret-at-least-32-bytes",
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
                graph = GatewayContractWorkflow(tmp, doctor)
                started = graph.start(
                    "m1_" + uuid.uuid4().hex,
                    "核查 P001 的历史来源差异",
                    "先核对已有记录，再决定是否形成受限草稿",
                    clarification_question="是否允许标注证据不足的受限草稿？",
                )
                run = started["run_id"]
                assert started["next"] == ["contract_gate"]
                assert started["clarification_question"]
                assert doctor.status(run)["contract_revision"] == 1
                try:
                    graph.resume_confirm(run, 0, 1)
                    raise AssertionError("unanswered clarification confirmed")
                except WorkflowConflict:
                    pass
                # Invalid Command was NOT sent to LangGraph: next valid answer works.
                answered = graph.resume(
                    run, "clarify", 0, 1,
                    answer="可以，但必须明确来源差异与局限",
                )
                assert (answered["business"], answered["next"],
                        answered["contract_revision"], answered["revision"]) == (
                            "awaiting_confirmation", ["contract_gate"], 2, 1
                        ), answered
                assert answered["clarification_answer"] == "可以，但必须明确来源差异与局限"
                assert not answered["clarification_question"]
                expect_gateway_error(lambda: doctor.confirm(run, 0, 1), 409)

                revised = graph.resume(
                    run, "revise", 1, 2,
                    summary="医生修订后仅核查 P001 异源资料冲突",
                    overview="先查原始记录与转诊摘要，再动态补充 Issue",
                )
                assert (revised["business"], revised["next"],
                        revised["contract_revision"], revised["revision"]) == (
                            "awaiting_confirmation", ["contract_gate"], 3, 2
                        ), revised
                assert "仅核查" in revised["summary"]
                assert revised["clarification_answer"]
                # Duplicate request after commit must *not* create Contract v4.
                replay = doctor.change_contract(
                    run, "revise", 1, 2,
                    summary="医生修订后仅核查 P001 异源资料冲突",
                    overview="先查原始记录与转诊摘要，再动态补充 Issue",
                )
                assert replay["contract_revision"] == 3 and replay["revision"] == 2
                expect_gateway_error(lambda: doctor.change_contract(
                    run, "revise", 1, 2, summary="偷偷更换医学目标",
                    overview="先查原始记录与转诊摘要，再动态补充 Issue"
                ), 409)
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_contract_versions "
                    f"WHERE run_id='{run}'"
                ) == "3"
                with __import__("contextlib").suppress(Exception):
                    pass
                try:
                    graph.resume_confirm(run, 0, 1)
                    raise AssertionError("old contract revision should fail")
                except WorkflowConflict:
                    pass
                finished = graph.resume_confirm(run, 2, 3)
                assert (finished["business"], finished["next"],
                        finished["contract_revision"], finished["revision"]) == (
                            "contract_confirmed", [], 3, 3
                        ), finished
                assert doctor.status(run)["contract_revision"] == 3
                # Idempotent SQL replay survives a later M3 Draft and does not
                # overwrite a previous immutable Contract revision.
                token = doctor.issue_capability(run)
                investigation = M3GatewayWorkflow(
                    tmp, GatewayArtifactRepository(transport, token)
                ).execute(run, 3)
                assert investigation["phase"] == "draft_ready"
                assert investigation["report"]["contract_revision"] == 3
                assert "agenda:add:I-04" in investigation["trace"]
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_artifacts "
                    f"WHERE run_id='{run}'"
                ) == "8"
                assert mysql(
                    "SELECT COUNT(*) FROM treesem_v2_operations "
                    f"WHERE run_id='{run}'"
                ) == "11", "3 Contract actions + 8 Agent artifacts"
                assert doctor.confirm(run, 2, 3)["status"] == "contract_confirmed"
                expect_gateway_error(lambda: doctor.change_contract(
                    run, "revise", 3, 3, summary="改掉已确认任务",
                    overview="不应该允许在调查后覆盖"
                ), 409)
                expect_gateway_error(lambda: other.status(run), 404)

                # Explicit doctor cancellation: terminal without any Tool access.
                run2 = graph.start(
                    "m1_" + uuid.uuid4().hex, "另一个待取消任务", "预览调查概要",
                )["run_id"]
                canceled = graph.resume(run2, "cancel", 0, 1)
                assert (canceled["business"], canceled["next"]) == ("cancelled", [])
                expect_gateway_error(lambda: doctor.issue_capability(run2), 409)
                assert mysql("SELECT COUNT(*) FROM treesem_v2_contract_versions "
                             f"WHERE run_id='{run2}'") == "1"

                # Commit to MySQL *before* the corresponding LangGraph
                # checkpoint advances: resume must replay the same edit/result.
                run3 = graph.start(
                    "m1_" + uuid.uuid4().hex, "原始模拟调查", "模拟调查概要",
                )["run_id"]
                p = dict(summary="业务已提交的新目标", overview="再次确认调查方向")
                doctor.change_contract(run3, "revise", 0, 1, **p)
                restored = GatewayContractWorkflow(tmp, doctor).resume(
                    run3, "revise", 0, 1, **p
                )
                assert restored["revision"] == 1 and restored["contract_revision"] == 2
                assert restored["next"] == ["contract_gate"]
                assert mysql("SELECT COUNT(*) FROM treesem_v2_contract_versions "
                             f"WHERE run_id='{run3}'") == "2"
                doctor.confirm(run3, 1, 2)
                done = GatewayContractWorkflow(tmp, doctor).resume_confirm(run3, 1, 2)
                assert done["business"] == "contract_confirmed" and done["next"] == []
                assert mysql("SELECT COUNT(*) FROM treesem_v2_operations "
                             f"WHERE run_id='{run3}'") == "2"

                print("M1 Gateway real MySQL Contract versions + clarification + "
                      "reconfirmation + M3 reuse and crash replay passed")
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
