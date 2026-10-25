# V2 M0 synthetic P001 fixture

Status: **SIM / implemented fixture checks**, not a running medical Agent or clinical evaluation.

Source: Notion synthetic P001 data package and its T00-T09 trace. This fixture distinguishes source record time (`authored_on`), clinical event time (`event_on`), authorization-visible availability (`available_on`), and what the Run actually retrieved at each stage. Date-only comparisons avoid inventing hospital clock times.

Run from repository root:

```bash
cd PythonServices/TreeSemAgent
python -m unittest discover -s tests -p 'test_p001_fixture.py' -v
python -m evaluation.p001_trace
```

At T0 (2026-10-12), E-003 referral is **already available**, but the Agent discovers it at T1 while investigating I-02. E-002 estimates 250 mL and E-003 records approximately 800 mL. They are separate sources about one event, **not** a version update or a proven actual value. On T2 (2026-10-13), trusted LIS event EV-001 revises one logical Hb record from E-004/v1 (110 g/L) to E-005/v2 (130 g/L). E-004 stays in historical references while S-04 is expected to become stale. Q-001 is a bounded negative-search audit, not proof that a diagnosis never existed. The existing 49-feature PPH model is not validated for prenatal inference.

The validator checks fixture consistency and derives a read-only source view. It does **not** implement LangGraph, MySQL persistence, source-event subscriptions, per-assertion Grounding, doctor approval, or Tool authorization. Expected Issue IDs and forbidden claims are labels for later tests; they are not evidence that V2 abilities already work.
