#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def run(name: str, command: list[str], required: bool = True,
        env: dict[str, str] | None = None) -> bool:
    print(f"\n== {name} ==")
    result = subprocess.run(command, cwd=ROOT, env=env)
    if result.returncode and required:
        raise SystemExit(f"{name} failed with exit code {result.returncode}")
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--with-compose", action="store_true")
    parser.add_argument("--with-load", action="store_true")
    args = parser.parse_args()
    run("C++ and registered integration tests",
        ["ctest", "--test-dir", "build", "--output-on-failure"])
    evaluation_env = os.environ.copy()
    evaluation_env["PYTHONPATH"] = str(ROOT / "PythonServices/TreeSemAgent")
    run("Agent deterministic evaluation", [
        sys.executable, "PythonServices/TreeSemAgent/evaluation/run_evaluation.py",
        "--mode", "deterministic", "--output", "build/reports/m9-agent-evaluation.json"],
        env=evaluation_env)
    run("Git whitespace validation", ["git", "diff", "--check"])
    if args.with_compose:
        if shutil.which("docker") is None: raise SystemExit("docker is unavailable")
        run("Artifact preflight", [sys.executable, "scripts/prepare_demo.py"])
        run("Compose configuration", ["docker", "compose", "config", "--quiet"])
        run("Compose build and start", ["docker", "compose", "up", "-d", "--build"])
        run("Offline demo", ["python3", "scripts/demo_flow.py"])
    if args.with_load:
        if shutil.which("k6") is None: raise SystemExit("k6 is unavailable")
        run("k6 prediction load", ["k6", "run", "load/k6/prediction.js"])


if __name__ == "__main__": main()
