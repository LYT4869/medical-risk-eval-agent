#!/usr/bin/env python3
from __future__ import annotations

import argparse
import http.cookiejar
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_env() -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1); values[key] = value
    return values


class Client:
    def __init__(self, base: str):
        self.base, self.token = base.rstrip("/"), None
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def request(self, path: str, method: str = "GET", body=None,
                idempotency: bool = False, expected=(200, 201)):
        headers = {"Content-Type": "application/json"}
        if self.token: headers["Authorization"] = f"Bearer {self.token}"
        if idempotency: headers["Idempotency-Key"] = uuid.uuid4().hex
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.base + path, data=data, headers=headers, method=method)
        try:
            response = self.opener.open(request, timeout=40)
            payload = json.loads(response.read() or b"{}")
            if response.status not in expected: raise RuntimeError(f"unexpected HTTP {response.status}")
            return payload, dict(response.headers)
        except urllib.error.HTTPError as error:
            payload = json.loads(error.read() or b"{}")
            raise RuntimeError(f"HTTP {error.code}: {payload.get('error', payload)}") from error

    def login(self, email: str, password: str):
        result, _ = self.request("/api/v1/auth/login", "POST",
                                 {"email": email, "password": password})
        self.token = result["access_token"]
        return result["user"]


def compose(*arguments: str, check: bool = True):
    return subprocess.run(["docker", "compose", *arguments], cwd=ROOT,
                          check=check, text=True, capture_output=True)


def wait(base: str) -> None:
    for _ in range(90):
        try:
            urllib.request.urlopen(base + "/health", timeout=2)
            return
        except OSError:
            time.sleep(1)
    raise RuntimeError("treeSem demo did not become healthy")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:3000")
    parser.add_argument("--faults", action="store_true")
    args = parser.parse_args()
    env = load_env()
    wait(args.base_url)
    bootstrap = compose("exec", "-T", "-e",
        f'TREESEM_BOOTSTRAP_ADMIN_EMAIL={env["TREESEM_DEMO_ADMIN_EMAIL"]}', "-e",
        f'TREESEM_BOOTSTRAP_ADMIN_PASSWORD={env["TREESEM_DEMO_ADMIN_PASSWORD"]}',
        "backend", "treesem-admin", "bootstrap", check=False)
    if bootstrap.returncode and "already exists" not in bootstrap.stderr:
        raise RuntimeError(bootstrap.stderr.strip())

    admin, patient, doctor, intruder = (Client(args.base_url) for _ in range(4))
    admin_user = admin.login(env["TREESEM_DEMO_ADMIN_EMAIL"], env["TREESEM_DEMO_ADMIN_PASSWORD"])
    try:
        patient_result, _ = patient.request("/api/v1/auth/register", "POST", {
            "email": env["TREESEM_DEMO_PATIENT_EMAIL"],
            "password": env["TREESEM_DEMO_PATIENT_PASSWORD"], "display_name": "Demo Patient"})
        patient.token, patient_user = patient_result["access_token"], patient_result["user"]
    except RuntimeError:
        patient_user = patient.login(env["TREESEM_DEMO_PATIENT_EMAIL"], env["TREESEM_DEMO_PATIENT_PASSWORD"])
    try:
        doctor_user, _ = admin.request("/api/v1/admin/doctors", "POST", {
            "email": env["TREESEM_DEMO_DOCTOR_EMAIL"],
            "password": env["TREESEM_DEMO_DOCTOR_PASSWORD"], "display_name": "Demo Doctor"})
    except RuntimeError:
        doctor_user = doctor.login(env["TREESEM_DEMO_DOCTOR_EMAIL"], env["TREESEM_DEMO_DOCTOR_PASSWORD"])
    if not doctor.token:
        doctor.login(env["TREESEM_DEMO_DOCTOR_EMAIL"], env["TREESEM_DEMO_DOCTOR_PASSWORD"])
    try:
        admin.request("/api/v1/admin/doctor-patient-assignments", "POST", {
            "doctor_user_id": doctor_user["user_id"],
            "patient_user_id": patient_user["user_id"]})
    except RuntimeError:
        pass

    predictions = []
    for index in (0, 1):
        result, headers = patient.request("/api/v1/predictions", "POST", {"sample_index": index})
        predictions.append(result["prediction_id"])
        print("prediction", result["prediction_id"], result["prediction"]["positive_probability"],
              headers.get("X-Trace-Id", ""))
    for message in ("解释一下刚才的预测", "比较最近两次预测", "介绍产后出血的权威资料"):
        result, _ = patient.request("/api/v1/chat", "POST", {"message": message}, idempotency=True)
        print("agent", message, "tools=", [item["name"] for item in result["tools_used"]],
              "citations=", result.get("grounding_source_ids", []))
    history, _ = doctor.request(
        f'/api/v1/doctor/patients/{patient_user["user_id"]}/history')
    print("doctor history items", len(history["items"]))
    doctor.request(f"/api/v1/predictions/{predictions[-1]}/feedback", "POST",
                   {"assessment": "agree"}, idempotency=True)
    try:
        intruder_result, _ = intruder.request("/api/v1/auth/register", "POST", {
            "email": f"intruder-{uuid.uuid4().hex[:8]}@treesem.local",
            "password": env["TREESEM_DEMO_PATIENT_PASSWORD"], "display_name": "Intruder"})
        intruder.token = intruder_result["access_token"]
        intruder.request(f"/api/v1/predictions/{predictions[-1]}")
        raise RuntimeError("horizontal authorization check unexpectedly succeeded")
    except RuntimeError as error:
        if "HTTP 404" not in str(error): raise
        print("horizontal authorization: denied with 404")
    if args.faults:
        compose("stop", "knowledge", "model-adapter")
        try:
            result, _ = patient.request("/api/v1/predictions", "POST", {"sample_index": 2})
            print("ONNX survived optional service outage", result["prediction_id"])
        finally:
            compose("start", "model-adapter", "knowledge")
    print("admin", admin_user["user_id"], "patient", patient_user["user_id"],
          "doctor", doctor_user["user_id"])
    print("treeSem demo flow completed")


if __name__ == "__main__":
    main()
