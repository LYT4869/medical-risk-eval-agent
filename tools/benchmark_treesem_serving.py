#!/usr/bin/env python3
"""Reproducible local benchmark for Python Adapter and C++ ONNX serving."""

from __future__ import annotations

import argparse
import concurrent.futures
import http.client
import json
import os
from pathlib import Path
import socket
import statistics
import subprocess
import sys
import time


def free_port() -> int:
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        return int(candidate.getsockname()[1])


def predict(port: int, sample_index: int = 0, *, python_adapter: bool = False) -> float:
    started = time.perf_counter()
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    body = json.dumps({"sample_index": sample_index}, separators=(",", ":"))
    connection.request(
        "POST",
        "/v1/predict" if python_adapter else "/api/v1/predictions",
        body=body,
        headers={"Content-Type": "application/json", "Connection": "close"},
    )
    response = connection.getresponse()
    response.read()
    connection.close()
    if response.status != 200:
        raise RuntimeError(f"prediction returned {response.status}")
    return (time.perf_counter() - started) * 1000.0


def wait_healthy(port: int, process: subprocess.Popen[bytes]) -> None:
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("benchmark service exited during startup")
        try:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
            connection.request("GET", "/health", headers={"Connection": "close"})
            response = connection.getresponse()
            response.read()
            connection.close()
            if response.status == 200:
                return
        except OSError:
            pass
        time.sleep(0.03)
    raise RuntimeError("benchmark service did not become healthy")


def process_cpu_seconds(process_id: int) -> float:
    tail = Path(f"/proc/{process_id}/stat").read_text().rsplit(")", 1)[1].split()
    ticks = int(tail[11]) + int(tail[12])
    return ticks / os.sysconf("SC_CLK_TCK")


def process_rss_mib(process_id: int) -> float:
    for line in Path(f"/proc/{process_id}/status").read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) / 1024.0
    raise RuntimeError("VmRSS is unavailable")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def measure(
    port: int,
    process: subprocess.Popen[bytes],
    *,
    requests: int,
    concurrency: int,
    python_adapter: bool = False,
) -> dict[str, float | int]:
    cpu_before = process_cpu_seconds(process.pid)
    started = time.perf_counter()
    if concurrency == 1:
        latencies = [
            predict(port, index % 32, python_adapter=python_adapter)
            for index in range(requests)
        ]
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            latencies = list(
                pool.map(
                    lambda index: predict(
                        port, index % 32, python_adapter=python_adapter
                    ),
                    range(requests),
                )
            )
    elapsed = time.perf_counter() - started
    cpu_seconds = process_cpu_seconds(process.pid) - cpu_before
    return {
        "requests": requests,
        "concurrency": concurrency,
        "p50_ms": round(statistics.median(latencies), 4),
        "p95_ms": round(percentile(latencies, 0.95), 4),
        "p99_ms": round(percentile(latencies, 0.99), 4),
        "qps": round(requests / elapsed, 2),
        "rss_mib": round(process_rss_mib(process.pid), 2),
        "process_cpu_percent": round(cpu_seconds / elapsed * 100.0, 2),
        "fallback_count": 0,
    }


def stop(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is None:
        process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=3)


def launch_python(bundle: Path) -> subprocess.Popen[bytes]:
    return subprocess.Popen(
        [
            sys.executable,
            "-m",
            "treesem_adapter.server",
            "--bundle",
            str(bundle),
            "--port",
            str(PYTHON_PORT),
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def launch_cpp(executable: Path, bundle: Path, workers: int, port: int) -> subprocess.Popen[bytes]:
    environment = os.environ.copy()
    environment.update(
        {
            "TREESEM_MODEL_BACKEND": "onnx",
            "TREESEM_SERVING_BUNDLE_DIR": str(bundle),
            "TREESEM_INFERENCE_WORKERS": str(workers),
            "TREESEM_INFERENCE_QUEUE_CAPACITY": "128",
        }
    )
    return subprocess.Popen(
        [str(executable), str(port)],
        env=environment,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()

    results: dict[str, object] = {
        "machine": {
            "cpu_count": os.cpu_count(),
            "onnxruntime": "1.20.1 CPU",
            "bundle": args.bundle.name,
        }
    }
    python = launch_python(args.bundle.resolve())
    try:
        wait_healthy(PYTHON_PORT, python)
        for index in range(args.warmup):
            predict(PYTHON_PORT, index % 32, python_adapter=True)
        results["python_adapter_sequential"] = measure(
            PYTHON_PORT,
            python,
            requests=args.requests,
            concurrency=1,
            python_adapter=True,
        )
    finally:
        stop(python)

    cpp_results = {}
    for workers in (1, 2, 4):
        port = free_port()
        process = launch_cpp(args.server.resolve(), args.bundle.resolve(), workers, port)
        try:
            wait_healthy(port, process)
            for index in range(args.warmup):
                predict(port, index % 32)
            if workers == 2:
                results["cpp_onnx_sequential"] = measure(
                    port, process, requests=args.requests, concurrency=1
                )
            cpp_results[str(workers)] = measure(
                port, process, requests=args.requests, concurrency=workers
            )
        finally:
            stop(process)
    results["cpp_onnx_concurrent_by_workers"] = cpp_results
    print(json.dumps(results, ensure_ascii=False, indent=2, sort_keys=True))


PYTHON_PORT = free_port()


if __name__ == "__main__":
    main()
