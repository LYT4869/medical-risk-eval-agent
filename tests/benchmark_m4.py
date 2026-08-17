#!/usr/bin/env python3
"""Small reproducible HTTP latency harness for the M3/M4 comparison.

Start the server in the desired storage mode, then run this script. It reports
client-observed p50/p95/p99 and QPS; CPU/RSS should be captured externally with
pidstat/time so the measurement process does not depend on non-portable modules.
"""

import argparse
import concurrent.futures
import http.client
import json
import math
import time
from urllib.parse import urlsplit


def percentile(values, quantile):
    ordered = sorted(values)
    index = max(0, math.ceil(quantile * len(ordered)) - 1)
    return ordered[index]


def call(target, method, path, body, cookie):
    connection = http.client.HTTPConnection(target.hostname, target.port, timeout=10)
    headers = {}
    payload = None
    if body is not None:
        payload = json.dumps(body, separators=(",", ":"))
        headers["Content-Type"] = "application/json"
    if cookie:
        headers["Cookie"] = cookie
    started = time.perf_counter()
    connection.request(method, path, body=payload, headers=headers)
    response = connection.getresponse()
    response_body = response.read()
    elapsed_ms = (time.perf_counter() - started) * 1000
    set_cookie = response.getheader("Set-Cookie")
    connection.close()
    if response.status != 200:
        raise RuntimeError(f"{method} {path} returned {response.status}")
    parsed = json.loads(response_body.decode("utf-8"))
    return elapsed_ms, set_cookie.split(";", 1)[0] if set_cookie else cookie, parsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:18080")
    parser.add_argument(
        "--endpoint",
        choices=("prediction", "prediction_get", "history", "ready"),
        default="prediction")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--requests", type=int, default=1000)
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    target = urlsplit(args.base_url)
    if args.concurrency <= 0 or args.requests <= 0 or args.warmup < 0:
        raise SystemExit("concurrency/requests must be positive and warmup non-negative")

    def prepare():
        cookie = None
        if args.endpoint in ("prediction_get", "history"):
            _, cookie, created = call(
                target, "POST", "/api/v1/predictions", {"sample_index": 0}, cookie)
            method = "GET"
            path = (f"/api/v1/predictions/{created['prediction_id']}"
                    if args.endpoint == "prediction_get"
                    else "/api/v1/sessions/current/history?limit=20")
            return method, path, None, cookie
        if args.endpoint == "prediction":
            return "POST", "/api/v1/predictions", {"sample_index": 0}, cookie
        return "GET", "/ready", None, cookie

    method, path, body, cookie = prepare()
    for _ in range(args.warmup):
        _, cookie, _ = call(target, method, path, body, cookie)
    def run_batch(count):
        local_method, local_path, local_body, local_cookie = prepare()
        values = []
        for _ in range(count):
            latency, local_cookie, _ = call(
                target, local_method, local_path, local_body, local_cookie)
            values.append(latency)
        return values

    started = time.perf_counter()
    if args.concurrency == 1:
        latencies = run_batch(args.requests)
    else:
        base, remainder = divmod(args.requests, args.concurrency)
        counts = [base + (1 if index < remainder else 0)
                  for index in range(args.concurrency)]
        with concurrent.futures.ThreadPoolExecutor(
                max_workers=args.concurrency) as executor:
            latencies = [latency
                         for batch in executor.map(run_batch, counts)
                         for latency in batch]
    duration = time.perf_counter() - started
    print(json.dumps({
        "endpoint": args.endpoint,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "p50_ms": round(percentile(latencies, 0.50), 3),
        "p95_ms": round(percentile(latencies, 0.95), 3),
        "p99_ms": round(percentile(latencies, 0.99), 3),
        "qps": round(args.requests / duration, 2),
    }, indent=2))


if __name__ == "__main__":
    main()
