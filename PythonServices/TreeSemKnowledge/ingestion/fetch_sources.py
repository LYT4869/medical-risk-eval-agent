from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import urllib.request
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch approved external knowledge snapshots and verify SHA-256")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--source-root", required=True, type=Path)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    root = args.source_root.resolve()
    sources = json.loads(args.manifest.read_text(encoding="utf-8"))["sources"]
    for source in sources:
        relative = Path(source["local_path"])
        if not relative.parts or relative.parts[0] != "raw":
            continue
        destination = (root / relative).resolve()
        if root not in destination.parents:
            raise SystemExit("source path escapes trusted root")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if args.verify_only:
            if not destination.is_file():
                raise SystemExit(f"source snapshot is missing: {source['source_id']}")
            payload = destination.read_bytes()
        else:
            request = urllib.request.Request(
                source["original_url"],
                headers={"User-Agent": "treeSem-knowledge-ingestion/1"})
            with urllib.request.urlopen(request, timeout=60) as response:
                payload = response.read(20 * 1024 * 1024 + 1)
        if len(payload) > 20 * 1024 * 1024:
            raise SystemExit(f"source is too large: {source['source_id']}")
        digest = hashlib.sha256(payload).hexdigest()
        if digest != source["sha256"]:
            raise SystemExit(
                f"source changed and requires human review: {source['source_id']} ({digest})")
        if not args.verify_only:
            with tempfile.NamedTemporaryFile(
                    dir=destination.parent, delete=False) as temporary:
                temporary.write(payload)
                temporary_path = Path(temporary.name)
            temporary_path.replace(destination)
        print(f"verified {source['source_id']} {digest}")


if __name__ == "__main__":
    main()
