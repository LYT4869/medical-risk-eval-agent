#!/usr/bin/env python3
"""Maintain an explicit, restart-based model promotion and rollback record."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from treesem_adapter.bundle import ServingBundle, sha256_file


def write_atomic(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".release-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(value, output, indent=2, sort_keys=True)
            output.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("promote", "rollback", "show"))
    parser.add_argument("--bundle", type=Path)
    parser.add_argument("--registry", type=Path,
                        default=Path("artifacts/treesem/release-state.json"))
    args = parser.parse_args()
    state = json.loads(args.registry.read_text()) if args.registry.exists() else {
        "current": None, "previous": None, "history": []}
    now = datetime.now(timezone.utc).isoformat()
    if args.action == "promote":
        if args.bundle is None:
            raise SystemExit("--bundle is required for promote")
        bundle = ServingBundle(args.bundle)
        entry = {"bundle": str(args.bundle.resolve()),
                 "manifest_sha256": sha256_file(args.bundle / "manifest.json"),
                 "model_version": bundle.model_version, "recorded_at": now}
        state["previous"], state["current"] = state.get("current"), entry
        state["history"].append({"action": "promote", **entry})
        write_atomic(args.registry, state)
    elif args.action == "rollback":
        if not state.get("previous"):
            raise SystemExit("no previous Bundle is available")
        ServingBundle(Path(state["previous"]["bundle"]))
        state["current"], state["previous"] = state["previous"], state["current"]
        state["history"].append({"action": "rollback", "recorded_at": now,
                                 "model_version": state["current"]["model_version"]})
        write_atomic(args.registry, state)
    print(json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True))
    if state.get("current"):
        print(f'Configure TREESEM_SERVING_BUNDLE_DIR={state["current"]["bundle"]} and restart the backend.')


if __name__ == "__main__":
    main()
