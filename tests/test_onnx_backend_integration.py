#!/usr/bin/env python3
"""Process-level ONNX, shadow, startup-validation and concurrency checks."""

from __future__ import annotations

import concurrent.futures
import hashlib
import http.client
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from treesem_adapter.model import ServingBundlePredictor


def free_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def request(port: int, method: str, path: str, payload: dict | None = None) -> tuple[int, dict]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = "" if payload is None else json.dumps(payload, separators=(",", ":"))
    headers = {"Connection": "close"}
    if payload is not None:
        headers["Content-Type"] = "application/json"
    connection.request(method, path, body=body, headers=headers)
    response = connection.getresponse()
    parsed = json.loads(response.read())
    status = response.status
    connection.close()
    return status, parsed


def wait_healthy(port: int, process: subprocess.Popen[str]) -> dict:
    deadline = time.monotonic() + 8
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stdout, stderr = process.communicate(timeout=1)
            raise AssertionError(f"process exited early\n{stdout}\n{stderr}")
        try:
            status, body = request(port, "GET", "/health")
            if status == 200:
                return body
        except OSError:
            pass
        time.sleep(0.03)
    raise AssertionError("process did not become healthy")


def stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate(timeout=3)


def launch_backend(
    executable: Path,
    bundle: Path,
    backend: str,
    port: int,
    adapter_port: int,
    capture_logs: bool = False,
) -> subprocess.Popen[str]:
    environment = os.environ.copy()
    environment.update(
        {
            "TREESEM_MODEL_BACKEND": backend,
            "TREESEM_STORAGE_BACKEND": "memory",
            "TREESEM_SERVING_BUNDLE_DIR": str(bundle),
            "TREESEM_MODEL_ADAPTER_URL": f"http://127.0.0.1:{adapter_port}/v1/predict",
            "TREESEM_MODEL_CONNECT_TIMEOUT_MS": "200",
            "TREESEM_MODEL_TIMEOUT_MS": "3000",
            "TREESEM_INFERENCE_WORKERS": "4",
            "TREESEM_INFERENCE_QUEUE_CAPACITY": "64",
        }
    )
    return subprocess.Popen(
        [str(executable), str(port)],
        env=environment,
        stdout=subprocess.PIPE if capture_logs else subprocess.DEVNULL,
        stderr=subprocess.PIPE if capture_logs else subprocess.DEVNULL,
        text=True,
    )


def exercise_primary(executable: Path, bundle: Path) -> None:
    backend_port = free_port()
    dead_adapter_port = free_port()
    process = launch_backend(
        executable, bundle, "onnx_fallback", backend_port, dead_adapter_port
    )
    try:
        health = wait_healthy(backend_port, process)
        assert health["configured_backend"] == "onnx_fallback"
        assert health["primary_backend"] == "onnx"
        assert health["fallback_enabled"] is True
        assert health["model_version"] == bundle.name
        expected: dict[int, tuple] = {}
        for index in range(16):
            status, result = request(
                backend_port, "POST", "/api/v1/predictions", {"sample_index": index}
            )
            assert status == 200
            assert result["serving_backend"] == "onnx"
            prediction = result["prediction"]
            expected[index] = (
                prediction["label"],
                prediction["cluster_id"],
                prediction["tree_leaf_id"],
                prediction["positive_probability"],
            )

        def predict(index: int) -> tuple:
            status, result = request(
                backend_port, "POST", "/api/v1/predictions", {"sample_index": index}
            )
            assert status == 200
            prediction = result["prediction"]
            return (
                prediction["label"],
                prediction["cluster_id"],
                prediction["tree_leaf_id"],
                prediction["positive_probability"],
            )

        for _ in range(20):
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
                futures = {pool.submit(predict, index): index for index in range(16)}
                for future, index in ((future, futures[future]) for future in futures):
                    assert future.result() == expected[index]
        assert request(backend_port, "GET", "/health")[0] == 200

        disconnected = socket.create_connection(("127.0.0.1", backend_port), timeout=2)
        disconnected.sendall(
            b"POST /api/v1/predictions HTTP/1.1\r\n"
            b"Host: 127.0.0.1\r\nContent-Type: application/json\r\n"
            b"Content-Length: 18\r\nConnection: close\r\n\r\n"
            b'{"sample_index":0}'
        )
        disconnected.close()
        time.sleep(0.1)
        assert process.poll() is None
    finally:
        stop(process)


def exercise_corrupt_startup(executable: Path, bundle: Path) -> None:
    with tempfile.TemporaryDirectory(prefix="treesem-corrupt-cpp-") as temporary:
        for case in ("manifest", "tree_checksum", "onnx_checksum", "tree_topology"):
            copied = Path(temporary) / case / bundle.name
            shutil.copytree(bundle, copied)
            if case == "manifest":
                (copied / "manifest.json").write_text("{", encoding="utf-8")
            elif case == "tree_checksum":
                with (copied / "tree.json").open("ab") as output:
                    output.write(b"\n")
            elif case == "onnx_checksum":
                with (copied / "model.onnx").open("ab") as output:
                    output.write(b"\x00")
            else:
                tree_path = copied / "tree.json"
                tree = json.loads(tree_path.read_text(encoding="utf-8"))
                tree["nodes"][0]["left"] = 0
                tree_path.write_text(
                    json.dumps(tree, sort_keys=True, separators=(",", ":")) + "\n",
                    encoding="utf-8",
                )
                manifest_path = copied / "manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest["files"]["tree"]["size"] = tree_path.stat().st_size
                manifest["files"]["tree"]["sha256"] = hashlib.sha256(
                    tree_path.read_bytes()
                ).hexdigest()
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True, separators=(",", ":"))
                    + "\n",
                    encoding="utf-8",
                )
            process = launch_backend(
                executable,
                copied,
                "onnx",
                free_port(),
                free_port(),
                capture_logs=True,
            )
            try:
                process.wait(timeout=5)
                assert process.returncode != 0, case
            finally:
                stop(process)


def exercise_shadow(executable: Path, bundle: Path) -> None:
    adapter_port = free_port()
    environment = os.environ.copy()
    adapter = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "treesem_adapter.server",
            "--bundle",
            str(bundle),
            "--port",
            str(adapter_port),
        ],
        env=environment,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    backend = None
    try:
        wait_healthy(adapter_port, adapter)
        backend_port = free_port()
        backend = launch_backend(
            executable, bundle, "shadow", backend_port, adapter_port
        )
        health = wait_healthy(backend_port, backend)
        assert health["configured_backend"] == "shadow"
        assert health["fallback_enabled"] is False
        status, result = request(
            backend_port, "POST", "/api/v1/predictions", {"sample_index": 0}
        )
        assert status == 200
        assert result["serving_backend"] == "onnx"
    finally:
        if backend is not None:
            stop(backend)
        stop(adapter)


def exercise_shadow_mismatch_log(executable: Path, bundle: Path) -> None:
    predictor = ServingBundlePredictor(bundle)

    class MismatchHandler(BaseHTTPRequestHandler):
        def do_POST(self) -> None:  # noqa: N802
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            result = predictor.predict(payload)
            result["prediction"]["positive_probability"] += 0.05
            encoded = json.dumps(result, separators=(",", ":")).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

        def log_message(self, _format: str, *_args: object) -> None:
            return

    adapter = ThreadingHTTPServer(("127.0.0.1", 0), MismatchHandler)
    adapter_thread = threading.Thread(target=adapter.serve_forever, daemon=True)
    adapter_thread.start()
    backend = launch_backend(
        executable,
        bundle,
        "shadow",
        free_port(),
        int(adapter.server_address[1]),
        capture_logs=True,
    )
    backend_port = int(backend.args[-1])
    try:
        wait_healthy(backend_port, backend)
        distinctive = [123.456] + [0.0] * 48
        status, _ = request(
            backend_port,
            "POST",
            "/api/v1/predictions",
            {"preprocessed_features": distinctive},
        )
        assert status == 200
    finally:
        adapter.shutdown()
        adapter.server_close()
        backend.terminate()
        stdout, stderr = backend.communicate(timeout=5)
    logs = (stdout or "") + (stderr or "")
    assert "treeSem shadow mismatch" in logs
    assert "max_probability_delta=" in logs
    assert "123.456" not in logs
    assert "preprocessed_features" not in logs


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("usage: test_onnx_backend_integration.py SERVER BUNDLE")
    executable = Path(sys.argv[1]).resolve()
    bundle = Path(sys.argv[2]).resolve()
    exercise_primary(executable, bundle)
    exercise_corrupt_startup(executable, bundle)
    exercise_shadow(executable, bundle)
    exercise_shadow_mismatch_log(executable, bundle)
    print("treeSem ONNX process integration passed")


if __name__ == "__main__":
    main()
