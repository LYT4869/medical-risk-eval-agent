#!/usr/bin/env bash
set -euo pipefail

readonly MODEL_ID="intfloat/multilingual-e5-small"
readonly MODEL_REVISION="614241f622f53c4eeff9890bdc4f31cfecc418b3"
readonly CACHE_DIR="${TREESEM_HF_CACHE_DIR:-${XDG_CACHE_HOME:-$HOME/.cache}/huggingface}"
readonly MANIFEST_PATH="${TREESEM_ROUTING_MODEL_MANIFEST:-artifacts/evaluation/routing/routing-model-files.sha256}"

TREESEM_ROUTING_MODEL_ID="$MODEL_ID" \
TREESEM_ROUTING_MODEL_REVISION="$MODEL_REVISION" \
TREESEM_ROUTING_CACHE_DIR="$CACHE_DIR" \
TREESEM_ROUTING_MODEL_MANIFEST="$MANIFEST_PATH" \
python3 - <<'PY'
from __future__ import annotations

import hashlib
import os
from pathlib import Path

from huggingface_hub import snapshot_download


model_id = os.environ["TREESEM_ROUTING_MODEL_ID"]
revision = os.environ["TREESEM_ROUTING_MODEL_REVISION"]
cache_dir = Path(os.environ["TREESEM_ROUTING_CACHE_DIR"]).expanduser()
manifest_path = Path(os.environ["TREESEM_ROUTING_MODEL_MANIFEST"])

snapshot_download(
    repo_id=model_id,
    revision=revision,
    cache_dir=cache_dir,
)
snapshot = Path(snapshot_download(
    repo_id=model_id,
    revision=revision,
    cache_dir=cache_dir,
    local_files_only=True,
))

entries = []
for path in sorted(item for item in snapshot.rglob("*") if item.is_file()):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    entries.append(f"{digest.hexdigest()}  {path.relative_to(snapshot)}")

if not entries:
    raise SystemExit("routing model snapshot contains no files")
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text("\n".join(entries) + "\n", encoding="utf-8")
print(f"routing model ready: {model_id}@{revision}")
print(f"snapshot: {snapshot}")
print(f"checksum manifest: {manifest_path.resolve()}")
PY
