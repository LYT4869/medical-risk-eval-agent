#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import secrets
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def checksum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_bundle(directory: Path) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file(): return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("model") != "treeSem" or not manifest.get("onnx", {}).get("present"):
        return False
    for metadata in manifest.get("files", {}).values():
        path = directory / metadata["name"]
        if not path.is_file() or path.stat().st_size != metadata["size"] or checksum(path) != metadata["sha256"]:
            return False
    return True


def valid_index(directory: Path) -> bool:
    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file(): return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if set(manifest.get("thresholds", {})) != {
            "reranker_min_score", "all_scope_reranker_min_score", "rrf_min_score"}:
        return False
    for name, metadata in manifest.get("files", {}).items():
        path = directory / name
        if not path.is_file() or checksum(path) != metadata["sha256"]:
            return False
    return True


def newest_valid(root: Path, validator) -> Path:
    candidates = [item for item in root.iterdir() if item.is_dir() and validator(item)] if root.is_dir() else []
    if not candidates:
        raise SystemExit(f"no valid artifact found under {root}")
    return max(candidates, key=lambda item: item.stat().st_mtime).resolve()


def main() -> None:
    env_path = ROOT / ".env"
    bundle = newest_valid(ROOT / "artifacts" / "treesem" / "pph", valid_bundle)
    index = newest_valid(ROOT / "artifacts" / "knowledge", valid_index)
    cache = (Path.home() / ".cache" / "huggingface").resolve()
    if not cache.is_dir():
        raise SystemExit("Hugging Face cache is missing; build the knowledge index offline first")
    if env_path.exists():
        existing = env_path.read_text(encoding="utf-8")
        runtime_values = {
            "TREESEM_RUNTIME_UID": str(os.getuid()),
            "TREESEM_RUNTIME_GID": str(os.getgid()),
        }
        missing = [f"{key}={value}\n" for key, value in runtime_values.items()
                   if f"{key}=" not in existing]
        if missing:
            with env_path.open("a", encoding="utf-8") as target:
                target.writelines(missing)
            env_path.chmod(0o600)
            print("added runtime UID/GID to existing .env")
        print(f"existing {env_path} preserved")
        print(f"validated bundle: {bundle}")
        print(f"validated index:  {index}")
        return
    secret = lambda: secrets.token_hex(32)
    password = lambda: secrets.token_urlsafe(18)
    values = {
        "TREESEM_DB_PASSWORD": secret(), "TREESEM_DB_ROOT_PASSWORD": secret(),
        "TREESEM_ACCESS_JWT_SECRET": secret(),
        "TREESEM_CAPABILITY_JWT_SECRET": secret(),
        "TREESEM_AGENT_SERVICE_SECRET": secret(),
        "TREESEM_KNOWLEDGE_JWT_SECRET": secret(),
        "TREESEM_METRICS_BEARER_TOKEN": "",
        "TREESEM_RUNTIME_UID": str(os.getuid()),
        "TREESEM_RUNTIME_GID": str(os.getgid()),
        "TREESEM_SERVING_BUNDLE_DIR": str(bundle),
        "TREESEM_KNOWLEDGE_INDEX_DIR": str(index),
        "TREESEM_HF_CACHE_DIR": str(cache),
        "TREESEM_AGENT_LLM_MODE": "scripted_demo",
        "TREESEM_AGENT_LLM_BASE_URL": "", "TREESEM_AGENT_LLM_MODEL": "",
        "TREESEM_AGENT_LLM_API_KEY": "",
        "TREESEM_DEMO_ADMIN_EMAIL": "admin@treesem.local",
        "TREESEM_DEMO_ADMIN_PASSWORD": password(),
        "TREESEM_DEMO_PATIENT_EMAIL": "patient@treesem.local",
        "TREESEM_DEMO_PATIENT_PASSWORD": password(),
        "TREESEM_DEMO_DOCTOR_EMAIL": "doctor@treesem.local",
        "TREESEM_DEMO_DOCTOR_PASSWORD": password(),
    }
    env_path.write_text("".join(f"{key}={value}\n" for key, value in values.items()),
                        encoding="utf-8")
    env_path.chmod(0o600)
    print(f"created {env_path} with mode 0600")
    print("demo credentials are stored only in the ignored .env file")
    print(f"validated bundle: {bundle}")
    print(f"validated index:  {index}")


if __name__ == "__main__":
    main()
