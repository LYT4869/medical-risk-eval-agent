#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def events(lines, trace_id: str):
    for order, line in enumerate(lines):
        begin = line.find("{")
        if begin < 0: continue
        try: event = json.loads(line[begin:])
        except json.JSONDecodeError: continue
        if event.get("trace_id") == trace_id:
            event["_order"] = order
            yield event


def read_paths(paths: list[Path]) -> list[str]:
    files = []
    for path in paths:
        if path.is_dir():
            files.extend(item for item in sorted(path.rglob("*")) if item.is_file())
        elif path.is_file():
            files.append(path)
    return [line for path in files for line in path.read_text(
        encoding="utf-8", errors="replace").splitlines()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Render one safe treeSem cross-service trace")
    parser.add_argument("trace_id")
    parser.add_argument("files", nargs="*", type=Path)
    args = parser.parse_args()
    if len(args.trace_id) != 32 or any(ch not in "0123456789abcdef" for ch in args.trace_id):
        parser.error("trace_id must contain 32 lowercase hexadecimal characters")
    lines = sys.stdin.readlines() if not args.files else read_paths(args.files)
    found = list(events(lines, args.trace_id))
    if not found:
        print("trace not found", file=sys.stderr); return 1
    spans = {item.get("span_id"): item for item in found}
    children: dict[str, list[dict]] = {}
    for item in found:
        children.setdefault(str(item.get("parent_span_id", "")), []).append(item)
    for values in children.values(): values.sort(key=lambda item: item["_order"])

    def render(item: dict, depth: int) -> None:
        print("  " * depth + f'{item.get("service","?")} :: {item.get("operation","?")} '
              f'[{item.get("outcome","?")}] {item.get("duration_ms",0):.3f} ms '
              f'span={item.get("span_id","")}')
        for child in children.get(str(item.get("span_id", "")), []): render(child, depth + 1)

    roots = [item for item in found if not item.get("parent_span_id") or
             item.get("parent_span_id") not in spans]
    for root in sorted(roots, key=lambda item: item["_order"]): render(root, 0)
    return 0


if __name__ == "__main__": raise SystemExit(main())
