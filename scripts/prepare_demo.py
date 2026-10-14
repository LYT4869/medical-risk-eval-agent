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


def read_environment_file(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.is_file():
        return values
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        name, value = stripped.split("=", 1)
        values[name.strip()] = value.strip()
    return values


def _routing_integer(configured: dict[str, str], name: str, default: int,
                     *, minimum: int = 1, maximum: int | None = None) -> int:
    try:
        value = int(configured.get(name, str(default)))
    except ValueError as exc:
        raise SystemExit(f"{name} must be an integer") from exc
    if value < minimum or (maximum is not None and value > maximum):
        raise SystemExit(f"invalid {name}")
    return value


def prepare_routing(existing: dict[str, str]) -> str:
    configured = dict(existing)
    configured.update(os.environ)
    mode = configured.get("TREESEM_AGENT_ROUTING_MODE", "legacy_rule")
    if mode not in {"legacy_rule", "structured_shadow", "structured_llm"}:
        raise SystemExit("invalid TREESEM_AGENT_ROUTING_MODE")
    llm_mode = configured.get("TREESEM_AGENT_LLM_MODE", "scripted_demo")
    if llm_mode not in {"real", "scripted_demo"}:
        raise SystemExit("invalid TREESEM_AGENT_LLM_MODE")
    request_ms = _routing_integer(
        configured, "TREESEM_AGENT_ROUTER_REQUEST_TIMEOUT_MS", 3000)
    deadline_ms = _routing_integer(
        configured, "TREESEM_AGENT_ROUTER_TOTAL_DEADLINE_MS", 7000)
    attempts = _routing_integer(
        configured, "TREESEM_AGENT_ROUTER_MAX_ATTEMPTS", 2, maximum=2)
    backoff_ms = _routing_integer(
        configured, "TREESEM_AGENT_ROUTER_RETRY_BACKOFF_MS", 100,
        minimum=0)
    _routing_integer(
        configured, "TREESEM_AGENT_ROUTER_CONTEXT_MESSAGES", 4,
        minimum=0, maximum=4)
    _routing_integer(
        configured, "TREESEM_AGENT_ROUTER_CONTEXT_MAX_CHARS", 6000)
    required_ms = request_ms * attempts + (backoff_ms if attempts > 1 else 0)
    if deadline_ms < required_ms:
        raise SystemExit("structured Router total deadline cannot cover attempts")
    if mode != "legacy_rule" and llm_mode == "real":
        if (not configured.get("TREESEM_AGENT_LLM_BASE_URL", "").strip() or
                not configured.get("TREESEM_AGENT_LLM_MODEL", "").strip()):
            raise SystemExit(
                "TREESEM_AGENT_LLM_BASE_URL and TREESEM_AGENT_LLM_MODEL "
                "are required for structured routing")
    return mode


def main() -> None:
    env_path = ROOT / ".env"
    existing_values = read_environment_file(env_path)
    routing_mode = prepare_routing(existing_values)
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
        print(f"routing mode:     {routing_mode}")
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
        "TREESEM_AGENT_ROUTING_MODE": "legacy_rule",
        "TREESEM_AGENT_ROUTER_REQUEST_TIMEOUT_MS": "3000",
        "TREESEM_AGENT_ROUTER_TOTAL_DEADLINE_MS": "7000",
        "TREESEM_AGENT_ROUTER_MAX_ATTEMPTS": "2",
        "TREESEM_AGENT_ROUTER_RETRY_BACKOFF_MS": "100",
        "TREESEM_AGENT_ROUTER_CONTEXT_MESSAGES": "4",
        "TREESEM_AGENT_ROUTER_CONTEXT_MAX_CHARS": "6000",
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
    print(f"routing mode:     {routing_mode}")


if __name__ == "__main__":
    main()
