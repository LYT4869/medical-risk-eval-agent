"""Read-only staged view of P001: source availability is not retrieval."""

from __future__ import annotations

import json
from typing import Any

from .p001_contract import FixtureError, day, load, validate


def view_at(data: dict[str, Any], stage_id: str) -> dict[str, Any]:
    validate(data)
    records = {r["id"]: r for r in data["records"]}
    retrieved: set[str] = set()
    events: set[str] = set()
    selected = None
    for stage in data["stages"]:
        retrieved.update(stage["retrieved"])
        events.update(stage["events"])
        if stage["id"] == stage_id:
            selected = stage
            break
    if selected is None:
        raise FixtureError(f"unknown stage: {stage_id}")
    current: dict[str, str] = {}
    for key in sorted(retrieved):
        r = records[key]
        old_id = current.get(r["logical_id"])
        if old_id is None or r["version"] > records[old_id]["version"]:
            current[r["logical_id"]] = key
    accessible = sorted(r["id"] for r in data["records"]
                        if day(r["available_on"]) <= day(selected["on"]) and
                        (r["id"] != "E-005" or "EV-001" in events))
    return {"stage": stage_id, "available": accessible,
            "retrieved": sorted(retrieved), "current_by_logical_id": current,
            "applied_events": sorted(events)}


if __name__ == "__main__":
    fixture = load()
    for item in fixture["stages"]:
        print(json.dumps(view_at(fixture, item["id"]), ensure_ascii=False))
