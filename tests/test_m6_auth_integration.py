#!/usr/bin/env python3
"""Process-level M6 authentication and refresh-token rotation checks."""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from typing import Any


def free_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def request(
    port: int,
    method: str,
    path: str,
    body: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> tuple[int, list[tuple[str, str]], dict[str, Any]]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=8)
    payload = None if body is None else json.dumps(body)
    effective_headers = {"Connection": "close", **(headers or {})}
    if payload is not None:
        effective_headers["Content-Type"] = "application/json"
    connection.request(method, path, payload, effective_headers)
    response = connection.getresponse()
    raw = response.read()
    result = json.loads(raw) if raw else {}
    status, response_headers = response.status, response.getheaders()
    connection.close()
    return status, response_headers, result


def wait_until_healthy(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise AssertionError(
                f"backend exited early ({process.returncode})\n{stdout}\n{stderr}"
            )
        try:
            if request(port, "GET", "/health")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(0.04)
    raise AssertionError("backend did not become healthy")


def response_cookie(headers: list[tuple[str, str]], name: str) -> str:
    prefix = name + "="
    for key, value in headers:
        if key.lower() == "set-cookie" and value.startswith(prefix):
            return value.split(";", 1)[0]
    raise AssertionError(f"response did not set {name}")


def stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=5)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_m6_auth_integration.py /path/to/treesem_server")
    executable = Path(sys.argv[1]).resolve()
    port = free_port()
    environment = os.environ.copy()
    storage_backend = environment.get("TREESEM_AUTH_TEST_STORAGE", "memory")
    environment.update(
        {
            "TREESEM_STORAGE_BACKEND": storage_backend,
            "TREESEM_MODEL_BACKEND": "remote",
            "TREESEM_AGENT_ENABLED": "false",
            "TREESEM_AUTH_MODE": "required",
            "TREESEM_ACCESS_JWT_SECRET": "access-secret-for-process-test-at-least-32-bytes",
            "TREESEM_CAPABILITY_JWT_SECRET": "capability-secret-for-process-test-at-least-32-bytes",
            "TREESEM_AGENT_SERVICE_SECRET": "agent-secret-for-process-test-at-least-32-bytes",
            "TREESEM_KNOWLEDGE_JWT_SECRET": "knowledge-secret-for-process-test-at-least-32-bytes",
        }
    )
    process = subprocess.Popen(
        [str(executable), str(port)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        wait_until_healthy(port, process)
        status, _, body = request(port, "GET", "/api/v1/chat/history")
        assert (status, body.get("error")) == (401, "authentication_required")

        email = f"process-{time.time_ns()}@example.com"
        status, headers, registered = request(
            port,
            "POST",
            "/api/v1/auth/register",
            {
                "email": email,
                "password": "correct horse battery staple",
                "display_name": "Process Patient",
            },
        )
        assert status == 200, registered
        refresh_cookie = response_cookie(headers, "treeSemRefresh")
        response_cookie(headers, "treeSemSession")
        access_token = registered["access_token"]

        status, _, me = request(
            port,
            "GET",
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer " + access_token},
        )
        assert status == 200 and me["role"] == "patient", me

        status, cors_headers, _ = request(
            port,
            "GET",
            "/api/v1/auth/me",
            headers={
                "Authorization": "Bearer " + access_token,
                "Origin": "http://127.0.0.1:3000",
            },
        )
        assert status == 200
        assert dict(cors_headers)["Access-Control-Allow-Origin"] == \
            "http://127.0.0.1:3000"
        status, preflight_headers, _ = request(
            port,
            "OPTIONS",
            "/api/v1/chat",
            headers={
                "Origin": "http://127.0.0.1:3000",
                "Access-Control-Request-Method": "POST",
            },
        )
        assert status == 204
        assert dict(preflight_headers)["Access-Control-Allow-Origin"] == \
            "http://127.0.0.1:3000"

        status, _, denied = request(
            port,
            "POST",
            "/api/v1/auth/refresh",
            headers={"Cookie": refresh_cookie, "Origin": "https://attacker.invalid"},
        )
        assert (status, denied.get("error")) == (403, "forbidden")

        status, rotated_headers, rotated = request(
            port,
            "POST",
            "/api/v1/auth/refresh",
            headers={"Cookie": refresh_cookie},
        )
        assert status == 200, rotated
        replacement_cookie = response_cookie(rotated_headers, "treeSemRefresh")
        assert replacement_cookie != refresh_cookie

        status, _, reused = request(
            port,
            "POST",
            "/api/v1/auth/refresh",
            headers={"Cookie": refresh_cookie},
        )
        assert (status, reused.get("error")) == (401, "invalid_refresh_token")

        # Reuse detection revokes the entire token family, including the latest token.
        status, _, revoked = request(
            port,
            "POST",
            "/api/v1/auth/refresh",
            headers={"Cookie": replacement_cookie},
        )
        assert (status, revoked.get("error")) == (401, "invalid_refresh_token")

        if storage_backend == "mysql":
            # Authentication and audit records are the database truth source,
            # so login must continue to work after the gateway restarts.
            stop(process)
            process = subprocess.Popen(
                [str(executable), str(port)],
                env=environment,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            wait_until_healthy(port, process)
            status, _, logged_in = request(
                port,
                "POST",
                "/api/v1/auth/login",
                {"email": email, "password": "correct horse battery staple"},
            )
            assert status == 200 and logged_in["user"]["role"] == "patient"

        print("M6 authentication integration checks passed")
        return 0
    finally:
        stop(process)


if __name__ == "__main__":
    raise SystemExit(main())
