"""M0 contracts for synthetic P001. Not a clinical validator or Evidence Store."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

from .p001_sequence import validate_sequence

FIXTURE_PATH = Path(__file__).parent / "fixtures" / "pph_p001.json"


class FixtureError(ValueError):
    pass


def day(raw: str) -> date:
    try:
        return date.fromisoformat(raw)
    except (TypeError, ValueError) as exc:
        raise FixtureError(f"invalid fixture date: {raw!r}") from exc


def require(ok: bool, detail: str) -> None:
    if not ok:
        raise FixtureError(detail)


def validate(data: dict[str, Any]) -> None:
    require(data["classification"] == "SIM" and data["schema_version"] == 1,
            "fixture must be synthetic schema v1")
    require(data["patient_id"] == "P001" and data["run_id"] == "RUN-001",
            "unexpected scenario identity")
    records = data["records"]
    found = {r["id"]: r for r in records}
    require(len(found) == len(records) == 5 and set(found) ==
            {f"E-{i:03}" for i in range(1, 6)}, "duplicate/missing evidence")
    for r in records:
        require(r["logical_id"] and r["version"] > 0, "invalid source identity/version")
        available = day(r["available_on"])
        day(r["event_on"])
        if r["authored_on"] is not None:
            require(day(r["authored_on"]) <= available, "future-authored source")
    a, b, old, new = (found[k] for k in ("E-002", "E-003", "E-004", "E-005"))
    require(a["logical_id"] != b["logical_id"] and
            a["event_on"] == b["event_on"] == "2022-05-08" and
            (a["blood_loss_ml"], b["blood_loss_ml"]) == (250, 800) and
            b["approximate"] is True, "delivery/referral are independent estimates")
    require(old["logical_id"] == new["logical_id"] and
            (old["version"], new["version"]) == (1, 2) and
            old["event_on"] == new["event_on"] == "2026-10-10" and
            (old["hb_g_l"], new["hb_g_l"]) == (110, 130),
            "LIS versions must share the sample and logical record")
    validate_sequence(data, found, day, require)


def load(path: Path = FIXTURE_PATH) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    validate(result)
    return result
