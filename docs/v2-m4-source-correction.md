# TreeSem V2 M4：来源版本更正、影响追踪与当前来源准入（P001 SIM）

> **Implemented prototype, not clinical source ingestion**. M4 uses the existing C++/MySQL Run and physician authorization, a server-bundled **fixed synthetic LIS correction EV-001**, a separate small LangGraph recovery process and an immutable statement/source snapshot with a mutable validity overlay. No live HIS/EMR/LIS feed, real Hb lab processing, doctor-facing report regeneration or valid medical prediction exists. All rows remain `classification=SIM`.

## Why M4 is not a second Investigation Agent

M3 already produced a historical T1 (2026-10-12) investigation, with:
- E-002 (original delivery record: **250 mL**), E-003 (independent referral summary: **about 800 mL**) and I-04 `closed_unresolved`. They are **different logical sources** and may conflict; neither overwrites the other.
- A six-section, unsigned, `template_only` restricted Draft v1 that cites E-001/002/003. It **does not cite E-004**.
- The model-suitability I-03 remains `blocked` because the existing model has not been verified for prenatal use.

M4 extends that same Run with an **independent historical Hb statement scenario**, rather than pretending M3 had already used this laboratory value:

1. **M4-HISTORY**, explicitly triggered by the authenticated doctor in the SIM test, stores an **immutable** historical source snapshot E-004, logical record `LIS-P001-HB-20261010` version 1 / Hb **110 g/L**, and a dependent synthetic statement S-04 with that value. At this stage `E-004 = current`, `S-04 = current` **in the test's pre-correction snapshot**.
2. **EV-001**, a fixed server-bundled synthetic `LIS` source correction effective 2026-10-13, confirms **the same logical LIS record** and higher version 2, then inserts E-005 / Hb **130 g/L** as a *new immutable snapshot*.
3. In one MySQL transaction, mark **E-004 stale** and **S-04 stale**, persist a distinct immutable **S-05 with Hb 130 g/L citing E-005**, mark E-005/S-05 **current** in the validity overlay, and append **I-05 needs_review** with old/new references. Preserve E-004/S-04 original 110 values; do **not** auto-rewrite them to claim 130 or auto-close I-05.
4. The current-source admission gate checks the doctor-authorized Run and explicit source IDs. `[E-004]` is rejected as stale; `[E-005]` is accepted as current; `[E-002,E-003,E-005]` remains acceptable **for reference freshness only**, with E-002/E-003 separate.
5. The previously delivered **historical** M3 Draft v1 remains unchanged, since it does *not* cite E-004. We do **not** falsely declare that existing report invalid or claim it already describes the current 2026-10-18 Hb. Report Writer / Assertion Grounding / new Draft revision belong to **M5**.

The M4 graph comprises `stage → apply → project → END`, with official LangGraph SqliteSaver checkpoints. Graph State contains only run_id, event/ref identifiers and final phase. Every node revalidates the physician-confirmed Run via authenticated C++ Gateway. The stage/apply SQL effects are **idempotent**: rerunning the same event after DB COMMIT but before Graph Checkpoint returns its saved result rather than applying a second revision. A variant request with altered medical content is not allowed by the API at all. The event is **never a free-form Agent tool action**.

## Storage schema and security

New **additive** `009_v2_m4_source_corrections.sql`, without changing older migrations:

| Table | Stores |
|---|---|
| `treesem_v2_m4_source_versions` | Immutable E-004 v1 and E-005 v2 source fields and logical record identity |
| `treesem_v2_m4_statement_snapshots` | Immutable S-04 → E-004 (110 g/L) and new S-05 → E-005 (130 g/L) snapshots |
| `treesem_v2_m4_validity` | Current/stale overlay and causing event ID by `run_id + kind + ref`; **only this overlay changes** |
| `treesem_v2_m4_correction_events` | Immutable per-Run M4-HISTORY/EV-001 keys, SHA256 event fingerprint, recorded result |
| `treesem_v2_m4_followups` | I-05 `needs_review` caused by EV-001, with old/new source refs |

C++ stores all stage/apply data and validity transitions in a single InnoDB transaction guarded by the existing **Run row lock**. It checks confirmed status, ownership, SIM classification and explicit P001 fixture, then validates the precise prior source/statement identity before accepting EV-001. The update does **not** change Task Contract revision or independently rewrite M3's Evidence/IssueResult/Report Artifacts. No cross-database distributed transaction is asserted.

All APIs are under the **existing opt-in `TREESEM_V2_SIM_ARTIFACT_API=1`**, require C++ JWT + MySQL and a fresh doctor assignment check, and are **doctor-only**, not callable with scoped Agent capabilities:

```http
POST /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m4/history
{}
POST /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m4/correction
{}
GET /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m4/state
POST /api/v2/doctor/patients/:patient_id/sim-runs/:run_id/m4/validate-current
{"evidence_ids":["E-002","E-003","E-005"]}
```

The first two endpoints deliberately accept **no factual request fields**: they replay **one** server-defined P001 test event. A caller cannot claim `trusted_source=LIS`, submit an arbitrary Hb value or mark an unrelated referral as superseded. This is **not a production trustworthy LIS event adapter**: a real system would require authenticated LIS origin/signature, patient/sample ID mapping, timestamp and source-version checks, event delivery/cross-Run invalidation and clinical data policy.

The last endpoint returns only **source-version admissibility**, `stale_refs`, `missing_refs` and affected S-04 IDs. A positive answer does **not** establish sentence-level semantic support; M5 must still check provenance, numerical consistency, coverage and assertion-level correctness before calling any report “grounded.” Model eligibility is always `blocked_unverified_prenatal` for this fixture.

## Reproduce and verify

CI `v2_m4_mysql_integration` starts ephemeral MySQL 8, applies 001–009 migrations, compiles and runs the actual authenticated C++ server, and exercises:
- Rejection before DoctorConfirm, wrong patient or inaccessible Run, and rejection when trying to apply correction without original source.
- Actual M1 Contract confirm → M3 dynamic Agenda → 8 immutable Evidence/Issue/Draft Artifacts → M4 historical E-004/S-04 stage and EV-001 replay through C++/MySQL.
- An **event COMMIT before M4 Graph checkpoint**; subsequent Graph execution replays safely and produces one source v2, one S-04 stale and S-05 current overlay, one I-05.
- Current-reference admission rejects E-004 and unknown/duplicate refs, accepts E-005, and keeps E-002/E-003 as independent still-current sources.
- Doctor JWT required, scoped Agent JWT cannot invoke official-correction endpoint, arbitrary body rejected, original M3 historical Draft untouched, cross-Run correction independence, doctor assignment revocation refusal.
- M4 policy unit tests and existing M0–M3/V1 regression suites.

Run the M4 integration job using the same isolated **synthetic test database** recipe as `tests/test_v2_m3_gateway_integration.py`, with migration 009 applied. For a standalone run after M1+M3, `M4GatewayWorkflow(tmp, doctor).execute(run_id, explicit_simulated_lis_event=True)` triggers the single SIM fixture event, then returns `stale_statement_ids=['S-04']` and `new_issue='I-05'`. The flag is mandatory: no silent “official correction” just because a model suggested one.

## Deferred and acceptance boundary

- **Implemented**: exact P001 same-record LIS v1→v2 lineage, immutable historical source/statement data, transactional stale overlay + I-05 + event ledger, authorized current-reference check, deterministic restart behavior and real C++/MySQL tests when green.
- **Not implemented**: production LIS event origin verification and ingest, cross-Run/patient-wide notifications, arbitrary Claim dependency graph, clinical/LLM semantic verification, automatic localized Report v2, risk prediction eligibility validation, M5 report patching. Existing M3 historical drafts do not automatically become a current medical record.
- **Not in scope**: resolving 250 vs 800 mL by voting or making a diagnosis. Those are independent source observations and require clinical investigation, not source-version supersession.

M4 is a synthetic **data-consistency and audit** milestone. Continue to M5's report-generation and Grounding only after M4's specific code/CI is verified. Changes remain in a stacked Draft PR; never merge `main` without approval.
