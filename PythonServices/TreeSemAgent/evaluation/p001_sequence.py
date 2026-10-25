"""Checks trusted correction events and staged retrieval chronology."""


def validate_sequence(data, found, day, require):
    old = found["E-004"]
    new = found["E-005"]
    ev = data["correction"]
    require(ev["id"] == "EV-001" and ev["kind"] == "official_correction" and
            ev["trusted_source"] == "LIS" and ev["logical_id"] == old["logical_id"] and
            (ev["old"], ev["new"]) == ("E-004", "E-005") and
            day(ev["effective_on"]) >= day(new["available_on"]),
            "untrusted or inconsistent correction")
    require(data["audit"]["id"] == "Q-001" and
            data["audit"]["kind"] == "bounded_negative_search", "search is audit, not patient evidence")
    require(data["model_capability"] == "not_verified_for_prenatal",
            "unverified prenatal model cannot produce a risk score")
    stages = data["stages"]
    require([(s["id"], s["on"]) for s in stages] ==
            [("T0", "2026-10-12"), ("T1", "2026-10-12"),
             ("T2", "2026-10-13"), ("T3", "2026-10-18")], "unexpected stage order")
    retrieved: set[str] = set()
    applied: set[str] = set()
    for stage in stages:
        when = day(stage["on"])
        for event in stage["events"]:
            require(event == "EV-001" and event not in applied and
                    when >= day(ev["effective_on"]), "early/duplicate version event")
            applied.add(event)
        for key in stage["retrieved"]:
            require(key in found and key not in retrieved, "unknown/duplicate evidence")
            require(day(found[key]["available_on"]) <= when, "future evidence retrieval")
            require(key != "E-005" or "EV-001" in applied,
                    "corrected value before trusted event")
            retrieved.add(key)
    require(retrieved == set(found) and applied == {"EV-001"}, "missing source or event")
    require("E-003" not in stages[0]["retrieved"] and stages[1]["retrieved"] == ["E-003"],
            "referral must be found during I-02, not at run start")
    expected = data["expected"]
    require(expected["stale_statements"] == ["S-04"] and
            expected["conflicting_sources"] == ["E-002", "E-003"] and
            expected["report_admission"] == "restricted_draft", "incorrect expected outcome")
