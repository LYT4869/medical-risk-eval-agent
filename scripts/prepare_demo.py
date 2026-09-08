#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import os
import secrets
import sys
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


def valid_routing_artifact(directory: Path, backend: str) -> bool:
    try:
        agent_root = ROOT / "PythonServices" / "TreeSemAgent"
        sys.path.insert(0, str(agent_root))
        from agent.routing_artifact import load_routing_artifact
        from agent.task_registry import SUPPORTED_DOMAIN_TOOLS, TaskRegistry

        tasks = agent_root / "config" / "tasks.yaml"
        thresholds = agent_root / "config" / "routing_thresholds.json"
        registry = TaskRegistry.load(tasks, SUPPORTED_DOMAIN_TOOLS)
        load_routing_artifact(
            directory,
            task_registry_path=tasks,
            thresholds_path=thresholds,
            registry=registry,
            expected_backend=backend,
            expected_model_id="intfloat/multilingual-e5-small",
            expected_revision=(
                "614241f622f53c4eeff9890bdc4f31cfecc418b3"),
        )
        return True
    except (OSError, RuntimeError, ValueError):
        return False


def prepare_routing(existing: dict[str, str]) -> tuple[str, Path]:
    configured = dict(existing)
    configured.update(os.environ)
    mode = configured.get("TREESEM_AGENT_ROUTING_MODE", "rule")
    backend = configured.get(
        "TREESEM_AGENT_ROUTING_EMBEDDING_BACKEND", "onnx_fp32")
    if mode not in {"rule", "hybrid_optional", "hybrid_required"}:
        raise SystemExit("invalid TREESEM_AGENT_ROUTING_MODE")
    if backend not in {"onnx_fp32", "onnx_int8"}:
        raise SystemExit("invalid TREESEM_AGENT_ROUTING_EMBEDDING_BACKEND")

    disabled = (ROOT / "artifacts" / "agent-routing" / "disabled").resolve()
    if mode == "rule":
        disabled.mkdir(parents=True, exist_ok=True)
        return mode, disabled

    if configured.get("TREESEM_INSTALL_SEMANTIC_ROUTING", "false") != "true":
        raise SystemExit(
            "hybrid routing requires TREESEM_INSTALL_SEMANTIC_ROUTING=true")
    raw_directory = configured.get(
        "TREESEM_AGENT_ROUTING_ARTIFACT_DIR", "").strip()
    if not raw_directory:
        raise SystemExit(
            "hybrid routing requires TREESEM_AGENT_ROUTING_ARTIFACT_DIR")
    directory = Path(raw_directory).expanduser()
    if not directory.is_absolute():
        directory = ROOT / directory
    directory = directory.resolve()
    if not valid_routing_artifact(directory, backend):
        raise SystemExit(
            "configured Agent routing artifact failed checksum or contract validation")
    return mode, directory


def main() -> None:
    env_path = ROOT / ".env"
    existing_values = read_environment_file(env_path)
    routing_mode, routing_artifact = prepare_routing(existing_values)
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
        print(f"routing artifact: {routing_artifact}")
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
        "TREESEM_INSTALL_SEMANTIC_ROUTING": "false",
        "TREESEM_AGENT_ROUTING_MODE": "rule",
        "TREESEM_AGENT_ROUTING_EMBEDDING_BACKEND": "onnx_fp32",
        "TREESEM_AGENT_ROUTING_ARTIFACT_DIR": str(routing_artifact),
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
    print(f"routing artifact: {routing_artifact}")


if __name__ == "__main__":
    main()
