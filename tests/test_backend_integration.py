#!/usr/bin/env python3
"""Exercise the C++ async boundary against a controllable model adapter."""

from __future__ import annotations

import http.client
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


VALID_RESPONSE = {
    "model": "treeSem",
    "dataset": "pph",
    "input_source": "pph_test_split",
    "sample_index": 0,
    "prediction": {
        "label": 0,
        "positive_probability": 0.1,
        "confidence": 0.9,
        "cluster_id": 1,
        "tree_probability": 0.08,
        "tree_leaf_id": 8,
    },
    "important_features": [
        {
            "index": 48,
            "name": "Intrapartum_Bleeding",
            "standardized_value": -0.12,
            "tree_importance": 0.94,
        }
    ],
    "decision_path": [
        {
            "node_id": 0,
            "feature_index": 48,
            "feature_name": "Intrapartum_Bleeding",
            "operator": "<=",
            "threshold_standardized": 1.74,
            "value_standardized": -0.12,
        },
        {"leaf_id": 8},
    ],
}


class Gate:
    def __init__(self) -> None:
        self.entered = threading.Event()
        self.release = threading.Event()

    def reset(self) -> None:
        self.entered.clear()
        self.release.clear()


GATE = Gate()


class AdapterHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_POST(self) -> None:  # noqa: N802 - stdlib callback name
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        GATE.entered.set()
        if not GATE.release.wait(timeout=5):
            self.send_error(500)
            return

        body = json.dumps(VALID_RESPONSE, separators=(",", ":")).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def log_message(self, _format: str, *_args: object) -> None:
        return


def free_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def request(port: int, method: str, path: str, body: str = "") -> tuple[int, str]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=4)
    headers = {"Connection": "close"}
    if body:
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    payload = response.read().decode()
    status = response.status
    connection.close()
    return status, payload


def wait_until_healthy(port: int, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise AssertionError(
                f"backend exited early ({process.returncode})\n{stdout}\n{stderr}"
            )
        try:
            status, _ = request(port, "GET", "/health")
            if status == 200:
                return
        except OSError:
            pass
        time.sleep(0.03)
    raise AssertionError("backend did not become healthy")


def threaded_prediction(port: int, output: list[tuple[int, str]]) -> threading.Thread:
    thread = threading.Thread(
        target=lambda: output.append(
            request(port, "POST", "/api/v1/predictions", '{"sample_index":0}')
        ),
        daemon=True,
    )
    thread.start()
    return thread


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("usage: test_backend_integration.py /path/to/treesem_server")

    executable = Path(sys.argv[1]).resolve()
    adapter = ThreadingHTTPServer(("127.0.0.1", 0), AdapterHandler)
    adapter_port = int(adapter.server_address[1])
    adapter_thread = threading.Thread(target=adapter.serve_forever, daemon=True)
    adapter_thread.start()

    backend_port = free_port()
    environment = os.environ.copy()
    environment.update(
        {
            "TREESEM_MODEL_ADAPTER_URL": (
                f"http://127.0.0.1:{adapter_port}/v1/predict"
            ),
            "TREESEM_MODEL_CONNECT_TIMEOUT_MS": "300",
            "TREESEM_MODEL_TIMEOUT_MS": "3000",
            "TREESEM_INFERENCE_WORKERS": "1",
            "TREESEM_INFERENCE_QUEUE_CAPACITY": "1",
            "TREESEM_MODEL_BACKEND": "remote",
            "TREESEM_STORAGE_BACKEND": "memory",
            "TREESEM_AUTH_MODE": "development",
        }
    )
    process = subprocess.Popen(
        [str(executable), str(backend_port)],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        wait_until_healthy(backend_port, process)

        # A blocked model call occupies the only worker, while the EventLoop
        # continues serving health checks.
        GATE.reset()
        first_results: list[tuple[int, str]] = []
        first = threaded_prediction(backend_port, first_results)
        assert GATE.entered.wait(timeout=2), "model adapter was not called"
        started = time.monotonic()
        health_status, _ = request(backend_port, "GET", "/health")
        elapsed = time.monotonic() - started
        assert health_status == 200
        assert elapsed < 0.5, f"health endpoint blocked for {elapsed:.3f}s"
        GATE.release.set()
        first.join(timeout=3)
        assert not first.is_alive()
        assert first_results[0][0] == 200

        # With one running request and one queued request, the next request is
        # rejected immediately and deterministically with 503.
        GATE.reset()
        running_results: list[tuple[int, str]] = []
        queued_results: list[tuple[int, str]] = []
        running = threaded_prediction(backend_port, running_results)
        assert GATE.entered.wait(timeout=2)
        queued = threaded_prediction(backend_port, queued_results)
        time.sleep(0.15)
        overloaded_status, overloaded_body = request(
            backend_port,
            "POST",
            "/api/v1/predictions",
            '{"sample_index":0}',
        )
        assert overloaded_status == 503
        assert json.loads(overloaded_body)["error"] == "prediction_overloaded"
        GATE.release.set()
        running.join(timeout=3)
        queued.join(timeout=3)
        assert running_results[0][0] == 200
        assert queued_results[0][0] == 200

        # Closing the client before inference completes must not invalidate a
        # connection object captured by the worker or crash the process.
        GATE.reset()
        disconnected = socket.create_connection(("127.0.0.1", backend_port), timeout=2)
        disconnected.sendall(
            b"POST /api/v1/predictions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 18\r\n"
            b"Connection: close\r\n\r\n"
            b'{"sample_index":0}'
        )
        assert GATE.entered.wait(timeout=2)
        disconnected.close()
        GATE.release.set()
        time.sleep(0.15)
        assert process.poll() is None
        assert request(backend_port, "GET", "/health")[0] == 200

        # SIGTERM stops the EventLoop first, then drains work already accepted
        # by the scheduler before process teardown.
        GATE.reset()
        draining_client = socket.create_connection(
            ("127.0.0.1", backend_port), timeout=2
        )
        draining_client.sendall(
            b"POST /api/v1/predictions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\n"
            b"Content-Type: application/json\r\n"
            b"Content-Length: 18\r\n"
            b"Connection: close\r\n\r\n"
            b'{"sample_index":0}'
        )
        assert GATE.entered.wait(timeout=2)
        draining_client.close()
        process.terminate()
        time.sleep(0.1)
        assert process.poll() is None, "backend did not drain accepted work"
        GATE.release.set()
    finally:
        GATE.release.set()
        adapter.shutdown()
        adapter.server_close()
        process.terminate()
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate(timeout=3)

    assert process.returncode == 0, f"backend shutdown returned {process.returncode}"
    combined_logs = stdout + stderr
    assert "important_features" not in combined_logs
    assert "preprocessed_features" not in combined_logs
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
