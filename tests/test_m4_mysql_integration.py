#!/usr/bin/env python3
"""Real MySQL persistence smoke test.

The caller owns database startup and migration. This test deliberately uses a
separate treesem_test schema and proves that a prediction survives a C++ server
restart. It is opt-in because it requires an external MySQL 8 process.
"""

import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(port, method, path, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    payload = None if body is None else json.dumps(body)
    request_headers = dict(headers or {})
    if payload is not None:
        request_headers["Content-Type"] = "application/json"
    connection.request(method, path, body=payload, headers=request_headers)
    response = connection.getresponse()
    data = response.read().decode("utf-8")
    result = (response.status, dict(response.getheaders()), json.loads(data))
    connection.close()
    return result


def wait_ready(port, process):
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("treeSem server exited during startup")
        try:
            status, _, _ = request(port, "GET", "/ready")
            if status == 200:
                return
        except (ConnectionError, OSError, json.JSONDecodeError):
            pass
        time.sleep(0.05)
    raise RuntimeError("treeSem server did not become ready")


def start(server, bundle, port):
    environment = os.environ.copy()
    environment.update({
        "TREESEM_MODEL_BACKEND": "onnx",
        "TREESEM_SERVING_BUNDLE_DIR": bundle,
        "TREESEM_STORAGE_BACKEND": "mysql",
        "TREESEM_AUTH_MODE": "development",
        "TREESEM_DB_NAME": environment.get("TREESEM_TEST_DB_NAME", "treesem_test"),
    })
    if not environment.get("TREESEM_DB_PASSWORD"):
        raise RuntimeError("TREESEM_DB_PASSWORD is required")
    process = subprocess.Popen(
        [server, str(port)], env=environment,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
    try:
        wait_ready(port, process)
    except Exception:
        error = process.stderr.read() if process.poll() is not None else ""
        stop(process)
        raise RuntimeError(error or "startup failed")
    return process


def stop(process):
    if process.poll() is None:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)


def main():
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_m4_mysql_integration.py SERVER BUNDLE")
    server, bundle = map(os.path.abspath, sys.argv[1:])
    port = free_port()
    process = start(server, bundle, port)
    try:
        status, headers, created = request(
            port, "POST", "/api/v1/predictions", {"sample_index": 0})
        assert status == 200, created
        session_id = created["session_id"]
        prediction_id = created["prediction_id"]
        cookie = headers["Set-Cookie"].split(";", 1)[0]
    finally:
        stop(process)

    process = start(server, bundle, port)
    try:
        status, _, ready = request(port, "GET", "/ready")
        assert status == 200, ready
        status, _, loaded = request(
            port, "GET", f"/api/v1/predictions/{prediction_id}",
            headers={"Cookie": cookie})
        assert status == 200, loaded
        assert loaded["prediction_id"] == prediction_id
        assert loaded["session_id"] == session_id
        internal = {"X-TreeSem-Session-Id": session_id}
        status, _, second = request(
            port, "POST", "/internal/v1/predictions",
            {"sample_index": 1}, headers=internal)
        assert status == 200, second
        second_id = second["prediction_id"]
        status, _, history = request(
            port, "GET", "/api/v1/sessions/current/history",
            headers={"Cookie": cookie})
        assert status == 200, history
        assert any(item["prediction_id"] == prediction_id for item in history["items"])
        assert any(item["prediction_id"] == second_id for item in history["items"])

        feedback_headers = dict(internal)
        feedback_headers["Idempotency-Key"] = "mysql-feedback-key-01"
        feedback_body = {
            "reviewer_reference": "demo-doctor-mysql",
            "assessment": "uncertain",
            "comment": "Persistent review note",
        }
        feedback_path = f"/internal/v1/predictions/{prediction_id}/feedback"
        status, _, saved = request(
            port, "POST", feedback_path, feedback_body, feedback_headers)
        assert status == 201, saved
        status, _, replay = request(
            port, "POST", feedback_path, feedback_body, feedback_headers)
        assert status == 200 and replay["feedback_id"] == saved["feedback_id"], replay
        conflicting = dict(feedback_body)
        conflicting["comment"] = "A conflicting retry"
        status, _, conflict = request(
            port, "POST", feedback_path, conflicting, feedback_headers)
        assert status == 409 and conflict["error"] == "idempotency_conflict", conflict

        status, other_headers, other = request(
            port, "POST", "/api/v1/predictions", {"sample_index": 2})
        assert status == 200, other
        other_cookie = other_headers["Set-Cookie"].split(";", 1)[0]
        status, _, hidden = request(
            port, "GET", f"/api/v1/predictions/{prediction_id}",
            headers={"Cookie": other_cookie})
        assert status == 404 and hidden["error"] == "resource_not_found", hidden
    finally:
        stop(process)


if __name__ == "__main__":
    main()
