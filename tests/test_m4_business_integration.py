#!/usr/bin/env python3
import http.client
import json
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def request(port: int, method: str, path: str, body=None, headers=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    request_headers = {"Connection": "close"}
    if headers:
        request_headers.update(headers)
    payload = None
    if body is not None:
        payload = json.dumps(body, separators=(",", ":"))
        request_headers["Content-Type"] = "application/json"
    connection.request(method, path, body=payload, headers=request_headers)
    response = connection.getresponse()
    response_body = response.read().decode("utf-8")
    result = response.status, dict(response.getheaders()), response_body
    connection.close()
    return result


def wait_ready(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("treeSem server exited before becoming healthy")
        try:
            status, _, _ = request(port, "GET", "/health")
            if status == 200:
                return
        except OSError:
            pass
        time.sleep(0.05)
    raise RuntimeError("treeSem server did not become healthy")


def main() -> int:
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_m4_business_integration.py SERVER BUNDLE")
    server = Path(sys.argv[1]).resolve()
    bundle = Path(sys.argv[2]).resolve()
    port = free_port()
    env = os.environ.copy()
    env.update({
        "TREESEM_MODEL_BACKEND": "onnx",
        "TREESEM_SERVING_BUNDLE_DIR": str(bundle),
        "TREESEM_STORAGE_BACKEND": "memory",
        "TREESEM_INFERENCE_WORKERS": "2",
        "TREESEM_INFERENCE_QUEUE_CAPACITY": "8",
        "TREESEM_DATABASE_WORKERS": "2",
        "TREESEM_DATABASE_QUEUE_CAPACITY": "8",
    })
    process = subprocess.Popen(
        [str(server), str(port)], env=env, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True)
    try:
        wait_ready(port, process)
        status, _, body = request(port, "GET", "/ready")
        assert status == 200, body

        status, _, body = request(
            port, "POST", "/internal/v1/predictions", {"sample_index": 0})
        assert status == 400 and json.loads(body)["error"] == "invalid_session", body

        status, headers, body = request(
            port, "POST", "/api/v1/predictions", {"sample_index": 0})
        assert status == 200, body
        first = json.loads(body)
        assert first["prediction_id"].startswith("pred_")
        assert first["session_id"].startswith("ses_")
        cookie = headers["Set-Cookie"].split(";", 1)[0]
        assert cookie.startswith("treeSemSession=ses_")

        public_headers = {"Cookie": cookie}
        status, _, body = request(
            port, "GET", f"/api/v1/predictions/{first['prediction_id']}",
            headers=public_headers)
        assert status == 200, body
        assert json.loads(body)["prediction_id"] == first["prediction_id"]

        status, _, body = request(
            port, "GET",
            f"/api/v1/predictions/{first['prediction_id']}/explanation",
            headers=public_headers)
        assert status == 200, body
        assert "decision_path" in json.loads(body)

        internal_headers = {"X-TreeSem-Session-Id": first["session_id"]}
        status, _, body = request(
            port, "POST", "/internal/v1/predictions",
            {"sample_index": 1}, internal_headers)
        assert status == 200, body
        second = json.loads(body)

        status, _, body = request(
            port, "GET", "/api/v1/sessions/current/history?limit=1",
            headers=public_headers)
        assert status == 200, body
        history = json.loads(body)
        assert len(history["items"]) == 1
        assert history["next_cursor"] is not None

        status, _, body = request(
            port, "POST", "/api/v1/comparisons",
            {"prediction_id_a": first["prediction_id"],
             "prediction_id_b": second["prediction_id"]}, public_headers)
        assert status == 200, body
        assert "positive_probability_delta" in json.loads(body)

        feedback_headers = dict(internal_headers)
        feedback_headers["Idempotency-Key"] = "m4-e2e-feedback-0001"
        feedback_body = {
            "reviewer_reference": "demo-doctor-001",
            "assessment": "uncertain",
            "comment": "Requires a second clinical review",
        }
        feedback_path = (
            f"/internal/v1/predictions/{first['prediction_id']}/feedback")
        status, _, body = request(
            port, "POST", feedback_path, feedback_body, feedback_headers)
        assert status == 201, body
        feedback_id = json.loads(body)["feedback_id"]
        status, _, body = request(
            port, "POST", feedback_path, feedback_body, feedback_headers)
        assert status == 200, body
        assert json.loads(body)["feedback_id"] == feedback_id
        status, _, body = request(
            port, "GET", feedback_path, headers=internal_headers)
        assert status == 200 and len(json.loads(body)["items"]) == 1, body

        status, other_headers, body = request(
            port, "POST", "/api/v1/predictions", {"sample_index": 2})
        assert status == 200, body
        other_cookie = other_headers["Set-Cookie"].split(";", 1)[0]
        status, _, body = request(
            port, "GET", f"/api/v1/predictions/{first['prediction_id']}",
            headers={"Cookie": other_cookie})
        assert status == 404, body

        unknown_cookie = "treeSemSession=ses_ffffffffffffffffffffffffffffffff"
        status, replacement_headers, body = request(
            port, "GET", f"/api/v1/predictions/{first['prediction_id']}",
            headers={"Cookie": unknown_cookie})
        assert status == 404, body
        assert replacement_headers["Set-Cookie"].split(";", 1)[0] != unknown_cookie

        status, _, body = request(
            port, "GET", f"/api/v1/predictions/{first['prediction_id']}",
            headers={"Cookie": "treeSemSession=not-an-id"})
        assert status == 400 and json.loads(body)["error"] == "invalid_session", body
    finally:
        if process.poll() is None:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        if process.returncode not in (0, -signal.SIGTERM):
            output = process.stdout.read() if process.stdout else ""
            raise RuntimeError(f"treeSem server failed: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
