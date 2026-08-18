#!/usr/bin/env python3
"""Process-level C++ chat workflow against a deterministic fake Agent service."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from typing import Any


class AgentHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    requests: list[dict[str, Any]] = []
    service_secret = "agent-service-secret-for-m5-process-test"

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        body = json.loads(self.rfile.read(length))
        if self.path != "/v1/agent/runs" or \
                self.headers.get("X-TreeSem-Agent-Token") != self.service_secret:
            self.send_error(401)
            return
        type(self).requests.append(body)
        payload = json.dumps(
            {
                "answer": "The deterministic agent completed the request.",
                "step_count": 1,
                "tools_used": [],
                "grounding_prediction_ids": [],
            },
            separators=(",", ":"),
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args: object) -> None:
        return


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
    effective = {"Connection": "close", **(headers or {})}
    if payload is not None:
        effective["Content-Type"] = "application/json"
    connection.request(method, path, payload, effective)
    response = connection.getresponse()
    raw = response.read()
    result = json.loads(raw) if raw else {}
    status, response_headers = response.status, response.getheaders()
    connection.close()
    return status, response_headers, result


def wait_healthy(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise AssertionError(f"backend exited early\n{stdout}\n{stderr}")
        try:
            if request(port, "GET", "/health")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(0.04)
    raise AssertionError("backend did not become healthy")


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_m5_agent_integration.py /path/to/treesem_server")
    executable = Path(sys.argv[1]).resolve()
    fake = ThreadingHTTPServer(("127.0.0.1", 0), AgentHandler)
    fake_thread = threading.Thread(target=fake.serve_forever, daemon=True)
    fake_thread.start()
    port = free_port()
    environment = os.environ.copy()
    storage_backend = environment.get("TREESEM_AGENT_TEST_STORAGE", "memory")
    auth_mode = environment.get("TREESEM_AGENT_TEST_AUTH", "development")
    environment.update(
        {
            "TREESEM_STORAGE_BACKEND": storage_backend,
            "TREESEM_MODEL_BACKEND": "remote",
            "TREESEM_AUTH_MODE": auth_mode,
            "TREESEM_AGENT_ENABLED": "true",
            "TREESEM_AGENT_URL": f"http://127.0.0.1:{fake.server_address[1]}/v1/agent/runs",
            "TREESEM_AGENT_SERVICE_SECRET": AgentHandler.service_secret,
        }
    )
    if auth_mode == "required":
        environment.update({
            "TREESEM_ACCESS_JWT_SECRET":
                "access-secret-for-m5-process-test-at-least-32-bytes",
            "TREESEM_CAPABILITY_JWT_SECRET":
                "capability-secret-for-m5-process-test-at-least-32-bytes",
        })
    process = subprocess.Popen(
        [str(executable), str(port)], env=environment,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        wait_healthy(port, process)
        authenticated_headers: dict[str, str] = {}
        if auth_mode == "required":
            status, auth_headers, registered = request(
                port, "POST", "/api/v1/auth/register",
                {
                    "email": f"agent-{time.time_ns()}@example.com",
                    "password": "correct horse battery staple",
                    "display_name": "Agent Patient",
                },
            )
            assert status == 200, registered
            session_cookie = next(
                value.split(";", 1)[0] for name, value in auth_headers
                if name.lower() == "set-cookie" and value.startswith("treeSemSession=")
            )
            authenticated_headers = {
                "Authorization": "Bearer " + registered["access_token"],
                "Cookie": session_cookie,
            }
        key = "agent-" + str(time.time_ns())
        status, headers, created = request(
            port, "POST", "/api/v1/chat", {"message": "How does treeSem work?"},
            {"Idempotency-Key": key, **authenticated_headers},
        )
        assert status == 200, created
        cookie = authenticated_headers.get("Cookie") or next(
            value.split(";", 1)[0] for name, value in headers
            if name.lower() == "set-cookie")
        assert len(AgentHandler.requests) == 1
        assert AgentHandler.requests[0]["recent_messages"] == []
        if auth_mode == "required":
            assert isinstance(AgentHandler.requests[0]["capability_token"], str)
        else:
            assert AgentHandler.requests[0]["capability_token"] is None

        status, _, replay = request(
            port, "POST", "/api/v1/chat", {"message": "How does treeSem work?"},
            {"Idempotency-Key": key, "Cookie": cookie,
             **{key: value for key, value in authenticated_headers.items()
                if key != "Cookie"}},
        )
        assert status == 200 and replay["run_id"] == created["run_id"], replay
        assert len(AgentHandler.requests) == 1

        status, _, conflict = request(
            port, "POST", "/api/v1/chat", {"message": "A different request"},
            {"Idempotency-Key": key, "Cookie": cookie,
             **{key: value for key, value in authenticated_headers.items()
                if key != "Cookie"}},
        )
        assert (status, conflict.get("error")) == (409, "idempotency_conflict")

        status, _, history = request(
            port, "GET", "/api/v1/chat/history?limit=20",
            headers={"Cookie": cookie,
                     **{key: value for key, value in authenticated_headers.items()
                        if key != "Cookie"}},
        )
        assert status == 200 and [item["role"] for item in history["items"]] == \
            ["assistant", "user"]

        status, _, second = request(
            port, "POST", "/api/v1/chat", {"message": "Continue"},
            {"Idempotency-Key": key + "-next", "Cookie": cookie,
             **{key: value for key, value in authenticated_headers.items()
                if key != "Cookie"}},
        )
        assert status == 200, second
        assert len(AgentHandler.requests) == 2
        assert len(AgentHandler.requests[1]["recent_messages"]) == 2
        print("M5 Agent integration checks passed")
        return 0
    finally:
        process.terminate()
        try:
            process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=5)
        fake.shutdown()
        fake.server_close()


if __name__ == "__main__":
    raise SystemExit(main())
