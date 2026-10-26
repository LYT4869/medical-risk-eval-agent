"""M7-A: safe, fixed-schema P001 SIM integration replay manifest.

The integration suite emits this trace ONLY after checking actual Gateway/SQL
results. No patient prose, account IDs, operation keys, JWT or opaque Run IDs
are accepted into the manifest. This is not an LLM inference trace.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import re
import tempfile

STEPS = {
    "late_after_delivery": (
        "contract_confirmed", "investigation_complete", "source_corrected",
        "proposal_staged", "draft_delivered", "explain_readonly",
        "draft_revised", "draft_revised", "late_source_corrected",
        "historical_stale", "edit_rejected",
    ),
    "correction_before_publish": (
        "contract_confirmed", "investigation_complete", "source_corrected",
        "proposal_staged", "late_source_corrected", "publish_rejected",
    ),
    "report_commit_before_checkpoint": (
        "contract_confirmed", "investigation_complete", "source_corrected",
        "proposal_staged", "draft_delivered", "checkpoint_recovered",
    ),
    "fresh_generation_replay": (
        "contract_confirmed", "investigation_complete", "source_corrected",
        "draft_delivered", "checkpoint_recovered", "access_revoked",
    ),
}
REF_PATTERN = re.compile(r"(?:E-[0-9]{3}|[SIP]-[0-9]{2}|EV-[0-9]{3})\Z")
METRICS = frozenset({
    "contract_revision", "steps", "tool_calls", "proposal_rows",
    "report_rows", "report_revision", "assertions", "sections", "writer_calls",
})


class ReplayTrace:
    """Deterministic structural proof of replay checkpoints, not clinical data."""

    def __init__(self):
        self.events: list[dict] = []
        self.positions = {scenario: 0 for scenario in STEPS}

    def record(self, scenario: str, step: str, *,
               refs=(), metrics=None) -> None:
        if scenario not in STEPS:
            raise ValueError("unknown replay scenario")
        idx = self.positions[scenario]
        if idx >= len(STEPS[scenario]) or STEPS[scenario][idx] != step:
            raise ValueError("replay step missing or out of order")
        if not isinstance(refs, (list, tuple)) or len(refs) > 20 or any(
            not isinstance(x, str) or not REF_PATTERN.fullmatch(x) for x in refs
        ):
            raise ValueError("non-allowlisted reference in replay trace")
        values = {} if metrics is None else metrics
        if not isinstance(values, dict) or any(
            key not in METRICS or type(value) is not int or
            value < 0 or value > 100000
            for key, value in values.items()
        ):
            raise ValueError("invalid or unapproved replay metric")
        self.events.append({
            "seq": len(self.events) + 1,
            "scenario": scenario,
            "step": step,
            "refs": list(refs),
            "metrics": dict(sorted(values.items())),
            "result": "assertions_passed",
        })
        self.positions[scenario] = idx + 1

    def finish(self) -> dict:
        incomplete = [name for name, expected in STEPS.items()
                      if self.positions[name] != len(expected)]
        if incomplete:
            raise ValueError("incomplete replay scenarios: " + ",".join(incomplete))
        return {
            "schema": "treesem.v2.m7a.replay.v1",
            "classification": "SIM",
            "fixture": "P001",
            "evaluation_type": "engineering_invariants",
            "agent": "fake_golden_trace",
            "writer": "fake_catalog",
            "model_quality_evaluated": False,
            "scenarios": list(STEPS),
            "events": self.events,
        }

    def write(self, destination: str | Path) -> None:
        payload = self.finish()
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        filename = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=path.parent,
                prefix=".m7a-", suffix=".tmp", delete=False
            ) as f:
                filename = f.name
                json.dump(payload, f, ensure_ascii=False, indent=2)
                f.write("\n")
            os.replace(filename, path)
        finally:
            if filename is not None and os.path.exists(filename):
                os.unlink(filename)
